"""Bring a machine from "just downloaded GoldLive" to "can run SESSION_001".

    BOOTSTRAP -> CAPABILITY PROBE -> DEPENDENCY CHECK
              -> MODEL/VOICE PROVISIONING -> SELFTEST -> READINESS

Two rules shape everything here.

**Never destroy a working artifact until its replacement is verified.** Every
download goes to a `.part` file, is checked, and only then replaces the target.
The previous code wrote `dest.write_bytes(resp.read())` straight to the final
path, so a connection dropped at 90% left a truncated voice where a working one
had been.

**Never silently install system-level components.** Ollama installs a
background service and VB-CABLE installs a kernel-mode audio driver. Both are
detected and explained with a link; neither is installed on the user's behalf,
whatever the convenience.
"""

from __future__ import annotations

import json
import logging
import shutil
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path

from shared.capability import HardwareProfile, probe
from shared.paths import config_path, data_root, resource_path
from shared.provisioning import Artifact, Provisioning, load, save, sha256_file
from shared.version import app_version

log = logging.getLogger(__name__)

HF_BASE = "https://huggingface.co/rhasspy/piper-voices/resolve/main"
OLLAMA_DEFAULT = "http://127.0.0.1:11434"
DOWNLOAD_TIMEOUT_S = 60.0
#: Truncated reads are a normal fact of real networks -- hotel wifi, tethering,
#: a proxy that closes early. Now that a short read is refused rather than
#: silently blessed, a retry is what turns that from a hard failure into a
#: recoverable one.
DOWNLOAD_ATTEMPTS = 3


class ProvisionError(RuntimeError):
    """Provisioning cannot continue. Carries an action the user can take."""

    def __init__(self, stage: str, reason: str, action: str = "") -> None:
        super().__init__(reason)
        self.stage = stage
        self.reason = reason
        self.action = action


# -- model catalogue and selection ----------------------------------------


@dataclass
class ModelChoice:
    model_id: str
    tier: str
    bytes: int
    ram_gb: float
    reason: str


def load_catalogue() -> dict:
    import yaml

    path = config_path("models.yaml")
    if not path.exists():
        path = resource_path("configs", "models.yaml")
    return yaml.safe_load(path.read_text(encoding="utf-8")) or {}


def select_model(
    profile: HardwareProfile,
    catalogue: dict | None = None,
    installed: list[str] | None = None,
) -> ModelChoice:
    """Pick the largest model this machine can actually run, CPU-first.

    "Fits in RAM" is necessary and nowhere near sufficient: a 7B on CPU fits
    comfortably in 16 GB and generates a 40-word utterance in tens of seconds,
    which is not a production choice. The size test here is a *filter*; the
    measured-latency check in `benchmark_choice` is the gate.

    Sizes are compared against the catalogue's `ram_gb` working set rather than
    the file size, because the KV cache grows with the context actually
    allocated and weights-only arithmetic is optimistic.
    """
    catalogue = catalogue or load_catalogue()
    defaults = catalogue.get("defaults", {})
    tiers = catalogue.get("tiers", [])
    if not tiers:
        raise ProvisionError("model", "the model catalogue is empty",
                             "reinstall, or check configs/models.yaml")

    fraction = float(defaults.get("ram_fraction", 0.6))
    headroom = float(defaults.get("disk_headroom", 1.3))

    # Available RAM is the honest number at provisioning time, but a machine
    # that is merely busy right now should not be permanently downgraded, so
    # allow the larger of "available now" and "half of total".
    usable_gb = max(profile.ram_available_gb, profile.ram_total_gb * 0.5) * fraction
    free_disk = profile.disk_data_free_gb

    affordable = []
    for spec in tiers:
        needs_disk = spec["bytes"] / 1e9 * headroom
        if spec["ram_gb"] <= usable_gb and needs_disk <= free_disk:
            affordable.append(spec)

    if not affordable:
        smallest = min(tiers, key=lambda s: s["ram_gb"])
        raise ProvisionError(
            "model",
            f"this machine cannot run even the smallest model: "
            f"{smallest['id']} needs about {smallest['ram_gb']} GB of working memory "
            f"and {smallest['bytes'] / 1e9 * headroom:.1f} GB of disk; "
            f"{usable_gb:.1f} GB and {free_disk:.1f} GB are available",
            "free up memory or disk space, or use a machine with more RAM",
        )

    # Prefer something already on the machine. Downloading a second model when
    # an affordable one is already installed is exactly the wasted work this
    # system exists to avoid -- and it is the common case on a re-provision or
    # after a version bump.
    already = [s for s in affordable if s["id"] in set(installed or [])]
    if already:
        best = already[-1]
        return ModelChoice(
            model_id=best["id"], tier=best["tier"], bytes=best["bytes"],
            ram_gb=best["ram_gb"],
            reason=(f"already installed; CPU inference, {usable_gb:.1f} GB usable "
                    f"memory"),
        )

    best = affordable[-1]
    return ModelChoice(
        model_id=best["id"], tier=best["tier"], bytes=best["bytes"],
        ram_gb=best["ram_gb"],
        reason=(f"CPU inference; {usable_gb:.1f} GB usable memory, "
                f"{free_disk:.1f} GB free disk"),
    )


# -- Ollama ----------------------------------------------------------------


def _ollama_base(profile: HardwareProfile) -> str:
    url = profile.model_server_url or OLLAMA_DEFAULT
    return url.rsplit("/v1", 1)[0]


def installed_models(base: str) -> list[dict]:
    try:
        with urllib.request.urlopen(f"{base}/api/tags", timeout=5) as resp:
            return (json.loads(resp.read()) or {}).get("models") or []
    except (urllib.error.URLError, TimeoutError, OSError, json.JSONDecodeError):
        return []


def server_running(base: str) -> bool:
    try:
        with urllib.request.urlopen(f"{base}/api/tags", timeout=3):
            return True
    except Exception:
        return False


def pull_model(base: str, model_id: str, on_progress=None) -> dict:
    """Stream Ollama's pull, reporting progress.

    Ollama owns the partial download and resumes it, so an interrupted pull is
    its problem to recover -- which is exactly why using it beats reimplementing
    a resumable multi-gigabyte download here.
    """
    body = json.dumps({"model": model_id, "stream": True}).encode()
    req = urllib.request.Request(
        f"{base}/api/pull", data=body, headers={"Content-Type": "application/json"}
    )
    last: dict = {}
    with urllib.request.urlopen(req, timeout=DOWNLOAD_TIMEOUT_S) as resp:
        for raw in resp:
            line = raw.decode("utf-8", "replace").strip()
            if not line:
                continue
            try:
                last = json.loads(line)
            except json.JSONDecodeError:
                continue
            if last.get("error"):
                raise ProvisionError("model", f"pull failed: {last['error']}",
                                     f"try manually: ollama pull {model_id}")
            if on_progress:
                on_progress(last)
    return last


def ensure_model(
    state: Provisioning, profile: HardwareProfile, model_id: str | None = None,
    on_progress=None,
) -> Artifact:
    base = _ollama_base(profile)
    if not server_running(base):
        raise ProvisionError(
            "model", f"no model server is running at {base}",
            "install Ollama from https://ollama.com, then run:  ollama serve",
        )

    if model_id is None:
        have = [m.get("model") or m.get("name") for m in installed_models(base)]
        choice = select_model(profile, installed=have)
        model_id = choice.model_id
        state.selection = {
            "model_id": choice.model_id, "tier": choice.tier,
            "device": "cpu", "reason": choice.reason,
            # Deliberately not a promise. Measured below, and the number is
            # approximate because the benchmark estimates tokens from
            # characters.
            "measured_tokens_per_s": None,
            "max_recommended_sessions": 1,
        }

    present = {m.get("model") or m.get("name"): m for m in installed_models(base)}
    entry = present.get(model_id)
    if entry is None:
        log.info("pulling %s (this is the large download)", model_id)
        pull_model(base, model_id, on_progress=on_progress)
        entry = {
            m.get("model") or m.get("name"): m for m in installed_models(base)
        }.get(model_id)
        if entry is None:
            raise ProvisionError(
                "model", f"{model_id} still not present after the pull",
                f"try manually: ollama pull {model_id}",
            )

    artifact = Artifact(
        kind="model", artifact_id=f"model:{model_id}", path=None,
        bytes=int(entry.get("size") or 0), digest=entry.get("digest"),
        source=f"ollama://{model_id}",
        licence=(entry.get("details") or {}).get("family"),
    )
    state.record_artifact(artifact)
    return artifact


# -- voices ----------------------------------------------------------------


def _download(url: str, dest: Path, attempts: int = DOWNLOAD_ATTEMPTS) -> int:
    """Download with bounded retries; see _download_once for the integrity rules."""
    last: ProvisionError | None = None
    for attempt in range(1, attempts + 1):
        try:
            return _download_once(url, dest)
        except ProvisionError as exc:
            last = exc
            if attempt < attempts:
                log.warning("download attempt %d/%d failed (%s); retrying",
                            attempt, attempts, exc.reason)
                time.sleep(2.0 * attempt)
    assert last is not None
    raise last


def _download_once(url: str, dest: Path) -> int:
    """Download to `dest`, via a .part file, never touching a good `dest`.

    Returns the byte count. The temporary file lives beside the target so the
    final os.replace stays on one volume and is therefore atomic.
    """
    dest.parent.mkdir(parents=True, exist_ok=True)
    part = dest.with_suffix(dest.suffix + ".part")
    try:
        with urllib.request.urlopen(url, timeout=DOWNLOAD_TIMEOUT_S) as resp,                 part.open("wb") as fh:
            declared = resp.headers.get("Content-Length")
            expected = int(declared) if declared and declared.isdigit() else None
            shutil.copyfileobj(resp, fh, length=1 << 20)

        size = part.stat().st_size
        if size == 0:
            raise ProvisionError("voice", f"{url} returned an empty file",
                                 "check the network and try again")

        # A connection dropped mid-stream does not always raise: urllib can
        # return a short read and copyfileobj stops happily at EOF. Without
        # this check a truncated file is written, hashed, and recorded as a
        # verified artifact -- so the corruption becomes the new "correct"
        # state and no later verification can ever detect it. Observed here:
        # a 63.5 MB voice silently arriving as 42.6 MB and being blessed.
        if expected is not None and size != expected:
            raise ProvisionError(
                "voice",
                f"download of {url} was truncated: got {size} bytes, "
                f"server declared {expected}",
                "check the network connection and run provisioning again",
            )

        part.replace(dest)
        return size
    except ProvisionError:
        part.unlink(missing_ok=True)
        raise
    except (urllib.error.URLError, TimeoutError, OSError) as exc:
        part.unlink(missing_ok=True)
        raise ProvisionError("voice", f"could not download {url}: {exc}",
                             "check the network and try again") from exc


def voice_catalogue() -> dict:
    from scripts.get_voices import CATALOGUE

    return CATALOGUE


def personas_voices() -> dict[str, str]:
    from scripts.get_voices import DEFAULT_PROFILE

    return DEFAULT_PROFILE


def ensure_voice(
    voice_id: str, voices_dir: Path, state: Provisioning, force: bool = False
) -> Artifact:
    """Install one voice if it is missing or fails verification.

    A voice that is already recorded and still matches its checksum is skipped
    entirely -- re-downloading 60 MB on every launch is exactly the behaviour
    this system exists to remove.
    """
    key = f"voice:{voice_id}"
    model = voices_dir / f"{voice_id}.onnx"
    config = voices_dir / f"{voice_id}.onnx.json"

    known = state.artifacts.get(key)
    if not force and known is not None and config.is_file():
        ok, detail = known.verify()
        if ok:
            log.debug("voice %s already provisioned (%s)", voice_id, detail)
            return known
        log.warning("voice %s failed verification (%s); re-downloading", voice_id, detail)

    catalogue = voice_catalogue()
    spec = catalogue.get(voice_id)
    if spec is None:
        raise ProvisionError(
            "voice", f"{voice_id} is not in the voice catalogue",
            f"known voices: {', '.join(sorted(catalogue))}",
        )

    base = f"{HF_BASE}/{spec['path']}/{voice_id}"
    size = _download(f"{base}.onnx", model)
    _download(f"{base}.onnx.json", config)

    artifact = Artifact(
        kind="voice", artifact_id=key, path=str(model), bytes=size,
        sha256=sha256_file(model), source=f"{base}.onnx",
        licence=spec.get("licence"),
    )
    state.record_artifact(artifact)
    log.info("installed voice %s (%.1f MB)", voice_id, size / 1e6)
    return artifact


def required_voices(session_id: str | None = None) -> list[str]:
    """Voices the configured sessions actually need.

    Only the personas in sessions.yaml, not the whole catalogue: downloading
    voices nobody is configured to use is 60 MB per mistake.
    """
    import yaml

    try:
        cfg = yaml.safe_load(config_path("sessions.yaml").read_text(encoding="utf-8"))
        sessions = (cfg or {}).get("sessions") or []
    except Exception:
        return sorted(set(personas_voices().values()))

    if session_id:
        sessions = [s for s in sessions if s.get("session_id") == session_id]

    # Resolve through the persona configs the SESSION reads, not through the
    # installer's DEFAULT_PROFILE. The two disagreed on a machine whose config
    # predated the real voice ids, so provisioning downloaded one voice while
    # the session asked for another and died on the first utterance.
    catalogue = voice_catalogue()
    wanted: set[str] = set()
    try:
        from intelligence.personas import load_personas
        from shared.paths import config_dir

        personas = load_personas(config_dir("personas"))
    except Exception:
        personas = {}

    fallback = personas_voices()
    for spec in sessions:
        persona_id = spec.get("persona_id")
        if not persona_id:
            continue
        persona = personas.get(persona_id)
        voice = getattr(persona, "voice_id", None) if persona else None
        if voice in catalogue:
            wanted.add(voice)
        elif persona_id in fallback:
            # Stale or placeholder config: readiness reports this properly, but
            # provisioning should still fetch something usable rather than
            # nothing at all.
            log.warning(
                "persona %r names voice %r, which is not a real voice; "
                "provisioning %r instead. Run `setup --force` to refresh config.",
                persona_id, voice, fallback[persona_id],
            )
            wanted.add(fallback[persona_id])
    return sorted(wanted)


# -- the orchestrator ------------------------------------------------------


def provision(
    session_id: str = "SESSION_001",
    force: bool = False,
    include_model: bool = True,
    echo=print,
) -> Provisioning:
    """Run provisioning to completion, or raise ProvisionError with an action."""
    from runtime.bootstrap import seed_config

    state = load()
    state.begin()
    save(state)

    try:
        echo("\n  Gold Live -- provisioning this PC\n")

        seed_config()

        echo("  Checking this PC ...")
        profile = probe()
        state.hardware = profile.to_json()
        state.profile_hash = profile.profile_hash()
        state.goldlive_version = app_version()
        from shared.contracts import SCHEMA_VERSION

        state.contracts_schema_version = SCHEMA_VERSION
        save(state)
        echo("\n" + "\n".join(f"    {line}" for line in profile.describe().splitlines()))

        if not profile.network_online and not profile.market_feed_reachable:
            raise ProvisionError(
                "network", "this PC has no internet connection",
                "connect to a network and run provisioning again",
            )

        # A per-install salt so viewer handles are not hashed into the same
        # space on every copy of the application.
        if not state.author_salt:
            import secrets

            state.author_salt = secrets.token_hex(16)
            save(state)

        echo("\n  Checking what is installed ...")
        from runtime.selftest import missing_imports

        missing = missing_imports()
        if missing:
            raise ProvisionError(
                "dependencies",
                f"this build is missing required packages: {', '.join(missing)}",
                "download Gold Live again; this cannot be repaired here",
            )
        echo("    bundled packages ... ok")

        voices_dir = data_root() / "voices"
        wanted = required_voices(session_id)
        echo(f"\n  Voices needed: {', '.join(wanted) or 'none'}")
        for voice_id in wanted:
            existing = state.artifacts.get(f"voice:{voice_id}")
            # Full checksum, not the cheap size check. `provision` is an
            # explicit repair action, and the whole point of recording a hash
            # is to catch corruption that leaves the size unchanged -- a
            # partially rewritten file passes looks_present() every time.
            # Hashing 60 MB costs a fraction of a second; the per-launch fast
            # path in needs_provisioning() still uses the cheap check.
            intact, detail = existing.verify() if existing else (False, "not recorded")
            if not force and intact:
                echo(f"    {voice_id} ... already installed")
                continue
            if existing and not intact:
                echo(f"    {voice_id} ... FAILED verification ({detail}); repairing")
            else:
                echo(f"    {voice_id} ... downloading (about 60 MB)")
            ensure_voice(voice_id, voices_dir, state, force=force)
            save(state)

        if include_model:
            echo("\n  Language model ...")
            profile_base = _ollama_base(profile)
            if not server_running(profile_base):
                raise ProvisionError(
                    "model", "Ollama is not running",
                    "install it from https://ollama.com, then run:  ollama serve\n"
                    "        (Gold Live does not install system services for you)",
                )
            have = [m.get("model") or m.get("name")
                    for m in installed_models(profile_base)]
            choice = select_model(profile, installed=have)
            echo(f"    chosen: {choice.model_id} ({choice.tier}) -- {choice.reason}")

            seen = {"pct": -1}

            def progress(update: dict) -> None:
                total, done = update.get("total"), update.get("completed")
                if total and done:
                    pct = int(done * 100 / total)
                    if pct >= seen["pct"] + 10:
                        seen["pct"] = pct
                        echo(f"      {pct}% of {total / 1e9:.1f} GB")

            ensure_model(state, profile, choice.model_id, on_progress=progress)
            save(state)
            echo(f"    {choice.model_id} ... ready")

        state.succeed()
        save(state)
        echo("\n  Provisioning complete.\n")
        return state

    except ProvisionError as exc:
        state.fail(exc.stage, exc.reason)
        save(state)
        raise
    except Exception as exc:
        state.fail("unknown", str(exc) or type(exc).__name__)
        save(state)
        raise


def needs_provisioning(state: Provisioning | None = None) -> tuple[bool, str]:
    """The cheap decision made on every launch.

    Metadata only -- no hashing, no network, no synthesis. Hashing a 2 GB model
    on every launch would cost ten seconds to learn something that has not
    changed since the previous launch. Liveness is readiness's job, and that is
    never cached; this only answers "is anything missing or stale".
    """
    state = state if state is not None else load()

    if not state.is_provisioned():
        return True, f"not provisioned (state: {state.state.value})"
    if state.version_changed():
        return True, (f"application updated: {state.goldlive_version} -> {app_version()}")
    gone = state.missing_artifacts()
    if gone:
        return True, f"missing artifact(s): {', '.join(a.artifact_id for a in gone)}"
    return False, "already provisioned"


def main() -> int:
    import argparse
    import logging as _logging

    ap = argparse.ArgumentParser(description="Provision this PC for Gold Live")
    ap.add_argument("--session", default="SESSION_001")
    ap.add_argument("--force", action="store_true", help="re-download even if valid")
    ap.add_argument("--skip-model", action="store_true")
    args = ap.parse_args()

    _logging.basicConfig(level=_logging.INFO, format="%(message)s")

    try:
        provision(session_id=args.session, force=args.force,
                  include_model=not args.skip_model)
    except ProvisionError as exc:
        print(f"\n  PROVISIONING FAILED at stage: {exc.stage}")
        print(f"  {exc.reason}")
        if exc.action:
            print(f"\n  What to do:\n    {exc.action}")
        print()
        return 1
    return 0
