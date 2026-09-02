"""Download Piper voices, and check what each one is actually licensed under.

Piper's repository is MIT, but that covers the code and packaging -- not the
speech corpus each voice was trained on. Voices inherit their corpus terms, and
those differ sharply: some are public domain, some CC-BY, some explicitly
non-commercial. VOICES.md does not say which is which.

Two things make this easy to get wrong, and both were got wrong here first:

  - A MODEL_CARD's Dataset licence is often "See URL", which looks harmless and
    quietly defers the real answer to another page.
  - Most English Piper voices are FINETUNED FROM `lessac`, trained on Blizzard
    Challenge 2013. That lineage appears only in the Training section, so
    reading the Dataset section alone misses it completely.

So this parses the card rather than trusting a hand-written table -- the table
was wrong for two of the first three voices picked.

    python -m scripts.get_voices --audit    # what every voice is licensed under
    python -m scripts.get_voices            # install the default profile
    python -m scripts.get_voices --list
"""

from __future__ import annotations

import argparse
import sys
import tempfile
import urllib.error
import urllib.request
from pathlib import Path

from shared.paths import data_root

HF_BASE = "https://huggingface.co/rhasspy/piper-voices/resolve/main"

# Candidates only. Deliberately no licence claims here.
CATALOGUE: dict[str, dict] = {
    "en_US-ryan-high": {"path": "en/en_US/ryan/high", "who": "American male, neutral"},
    "en_GB-alan-medium": {"path": "en/en_GB/alan/medium", "who": "British male, measured"},
    "en_GB-alba-medium": {"path": "en/en_GB/alba/medium", "who": "British female, Scottish"},
    "en_GB-northern_english_male-medium": {
        "path": "en/en_GB/northern_english_male/medium",
        "who": "British male, northern",
    },
    "en_GB-jenny_dioco-medium": {
        "path": "en/en_GB/jenny_dioco/medium",
        "who": "British female, slight Irish lilt",
    },
    "en_US-hfc_female-medium": {
        "path": "en/en_US/hfc_female/medium", "who": "American female, warm",
    },
    "en_US-hfc_male-medium": {"path": "en/en_US/hfc_male/medium", "who": "American male"},
    "en_US-amy-medium": {"path": "en/en_US/amy/medium", "who": "American female, neutral"},
    "en_US-lessac-high": {
        "path": "en/en_US/lessac/high", "who": "American female, high quality",
    },
    "en_US-ljspeech-high": {
        "path": "en/en_US/ljspeech/high", "who": "American female, audiobook style",
    },
    "en_US-bryce-medium": {"path": "en/en_US/bryce/medium", "who": "American male"},
    "en_US-john-medium": {"path": "en/en_US/john/medium", "who": "American male"},
    "en_US-norman-medium": {"path": "en/en_US/norman/medium", "who": "American male"},
    "en_US-kathleen-low": {"path": "en/en_US/kathleen/low", "who": "American female"},
    "en_US-libritts_r-medium": {
        "path": "en/en_US/libritts_r/medium", "who": "American, multi-speaker",
    },
}

# Which voice each shipped persona uses. Set from the audit, not from taste.
# Chosen from the audit, not from taste: every one of these reports a public
# domain corpus, so the build stays redistributable. The three are different
# speakers so sessions do not sound like each other -- seven identical voices
# across seven accounts is its own problem.
DEFAULT_PROFILE = {
    "scalper": "en_US-john-medium",
    "educator": "en_US-ljspeech-high",
    "macro": "en_US-norman-medium",
}

FLAGS = {
    "public domain": " CLEAN  ",
    "permissive": "  ok    ",
    "check": " CHECK  ",
    "unstated": "unstated",
    "restricted": "BLOCKED ",
    "unreachable": "  ??    ",
}


def fetch(url: str, dest: Path) -> bool:
    dest.parent.mkdir(parents=True, exist_ok=True)
    try:
        with urllib.request.urlopen(url, timeout=120) as resp:
            dest.write_bytes(resp.read())
        return True
    except (urllib.error.URLError, TimeoutError, OSError) as exc:
        print(f"    failed: {exc}")
        return False


def parse_model_card(text: str) -> dict:
    """Extract dataset URL, licence and training lineage from a MODEL_CARD."""
    dataset = licence = training = ""
    section = None

    for raw in text.splitlines():
        line = raw.strip()
        low = line.lower()
        if low.startswith("## dataset"):
            section = "dataset"
            continue
        if low.startswith("## training"):
            section = "training"
            continue
        if section == "dataset":
            if low.startswith("* url:"):
                dataset = line.split(":", 1)[1].strip()
            elif low.startswith("* license:"):
                licence = line.split(":", 1)[1].strip()
        elif section == "training" and line:
            training = f"{training} {line}".strip()

    low_lic = licence.lower()
    notes: list[str] = []
    verdict = "permissive"

    if "public domain" in low_lic:
        verdict = "public domain"
    elif any(m in low_lic for m in ("-nc", "nc-", "non-commercial", "noncommercial")):
        verdict = "restricted"
        notes.append(f"non-commercial licence: {licence}")
    elif "-nd" in low_lic:
        verdict = "restricted"
        notes.append(f"no-derivatives licence: {licence}")
    elif not licence or "see url" in low_lic:
        verdict = "unstated"
        notes.append("licence deferred to the dataset URL; read it before shipping")

    if "lessac" in training.lower():
        notes.append("finetuned from lessac -> Blizzard Challenge 2013 lineage")
        if verdict in ("permissive", "public domain"):
            verdict = "check"

    return {
        "dataset": dataset,
        "licence": licence or "(not stated)",
        "training": training,
        "verdict": verdict,
        "notes": notes,
    }


def audit(names: list[str]) -> dict[str, dict]:
    results: dict[str, dict] = {}
    tmp = Path(tempfile.mkdtemp())
    print(f"\n  Auditing {len(names)} voices against their MODEL_CARDs\n")

    for name in names:
        card = tmp / f"{name}.MODEL_CARD"
        if not fetch(f"{HF_BASE}/{CATALOGUE[name]['path']}/MODEL_CARD", card):
            results[name] = {"verdict": "unreachable", "notes": [], "licence": "?"}
            continue

        parsed = parse_model_card(card.read_text(encoding="utf-8", errors="replace"))
        results[name] = parsed
        flag = FLAGS.get(parsed["verdict"], "  ??    ")
        print(f"  [{flag}] {name:<36} {parsed['licence'][:42]}")
        for note in parsed["notes"]:
            print(f"             - {note}")

    return results


def download(name: str, voices_dir: Path) -> bool:
    meta = CATALOGUE.get(name)
    if meta is None:
        print(f"  unknown voice {name!r}")
        return False

    print(f"\n  {name}  ({meta['who']})")
    base = f"{HF_BASE}/{meta['path']}/{name}"
    ok = True
    for suffix in (".onnx", ".onnx.json"):
        dest = voices_dir / f"{name}{suffix}"
        if dest.exists():
            print(f"    have {dest.name}")
            continue
        print(f"    downloading {dest.name} ...")
        ok = fetch(f"{base}{suffix}", dest) and ok

    card = voices_dir / f"{name}.MODEL_CARD"
    if not card.exists():
        fetch(f"{HF_BASE}/{meta['path']}/MODEL_CARD", card)
    if card.exists():
        parsed = parse_model_card(card.read_text(encoding="utf-8", errors="replace"))
        if parsed["verdict"] in ("restricted", "check", "unstated"):
            print(f"    LICENCE {parsed['verdict'].upper()}: {parsed['licence']}")
            for note in parsed["notes"]:
                print(f"      {note}")
    return ok


def write_summary(voices_dir: Path) -> None:
    rows = []
    for path in sorted(voices_dir.glob("*.onnx")):
        name = path.stem
        card = voices_dir / f"{name}.MODEL_CARD"
        parsed = (
            parse_model_card(card.read_text(encoding="utf-8", errors="replace"))
            if card.exists()
            else {"licence": "?", "verdict": "unknown", "notes": []}
        )
        rows.append(
            f"| `{name}` | {parsed['verdict']} | {parsed['licence']} "
            f"| {'; '.join(parsed['notes']) or '-'} |"
        )

    (voices_dir / "VOICE_LICENCES.md").write_text(
        "\n".join([
            "# Voice licences",
            "",
            "Generated from each voice's MODEL_CARD. Piper's repository is MIT,",
            "but that covers the code -- a voice inherits the terms of the corpus",
            "it was trained on, and several English voices are finetuned from",
            "`lessac` (Blizzard Challenge 2013), which the Dataset section alone",
            "does not reveal.",
            "",
            "| Voice | Verdict | Licence | Notes |",
            "|---|---|---|---|",
            *rows,
            "",
            "`restricted` must not ship in a build handed to other people.",
            "`check` and `unstated` need the dataset URL read before it does.",
            "",
        ]),
        encoding="utf-8",
    )


def main() -> None:
    ap = argparse.ArgumentParser(description="Download and audit Piper voices")
    ap.add_argument("--voice", action="append")
    ap.add_argument("--dir", default=None)
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--audit", action="store_true",
                    help="fetch every MODEL_CARD and report actual licences")
    args = ap.parse_args()

    if args.audit:
        results = audit(sorted(CATALOGUE))
        clean = [n for n, r in results.items() if r["verdict"] == "public domain"]
        blocked = [n for n, r in results.items() if r["verdict"] == "restricted"]
        checks = [n for n, r in results.items() if r["verdict"] in ("check", "unstated")]

        print(f"\n  SAFE to redistribute : {', '.join(clean) or 'none found'}")
        print(f"  DO NOT redistribute  : {', '.join(blocked) or 'none'}")
        print(f"  needs the corpus read: {len(checks)} voice(s)")
        print()
        raise SystemExit(0)

    if args.list:
        print("\n  Candidate voices (run --audit for licences):\n")
        for name, meta in CATALOGUE.items():
            print(f"    {name:<36} {meta['who']}")
        print("\n  Personas currently map to:")
        for persona, voice in DEFAULT_PROFILE.items():
            print(f"    {persona:<12} -> {voice}")
        print()
        raise SystemExit(0)

    voices_dir = Path(args.dir) if args.dir else data_root() / "voices"
    wanted = args.voice or sorted(set(DEFAULT_PROFILE.values()))

    print(f"\n  Installing into {voices_dir}")
    results = {name: download(name, voices_dir) for name in wanted}
    write_summary(voices_dir)

    failed = [n for n, ok in results.items() if not ok]
    print(f"\n  installed {len(results) - len(failed)}/{len(results)}")
    if failed:
        print(f"  failed: {', '.join(failed)}")
    print(f"  licences: {voices_dir / 'VOICE_LICENCES.md'}")
    print("\n  Run with:  GoldLive.exe run --session SESSION_001 --tts piper\n")
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
