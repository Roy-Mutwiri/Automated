"""Generate a listening bundle: the same scenarios rendered three ways.

Measurements tell you whether the code does what it was told. They cannot tell
you whether the host sounds like a person. This builds the artefact that can:
twelve deterministic scenarios, each rendered under three conditions, stitched
into one continuous wav per scenario so they can actually be listened to
back-to-back rather than as a pile of chunks.

    python -m scripts.listening_bundle --out bundle
    # then open bundle/index.html

## The three conditions

    C_legacy       what the audio path did before any of this work: one
                   synthesis call per sentence, no prefetch, no trimming, no
                   delivery plan. Each sentence lands its own full terminal
                   pitch fall and the gap between them is whatever synthesis
                   happened to cost.

    B_timing       the timing work only. Sentences grouped, next chunk
                   prefetched during playback, edge padding trimmed, gaps
                   fixed at the neutral default. Delivery is neutral for every
                   scenario -- no reactions, no rate change, no pause changes.

    A_full         the current system. Timing, plus delivery driven by host
                   state: rate, gaps, pauses and reaction units.

C -> B isolates whether the timing work helped.
B -> A isolates whether the emotional layer helped.

If A and B are indistinguishable to a listener, the emotional layer is
complexity without benefit and should be cut back to whatever part of it does
survive the comparison.

## What this deliberately does not do

It does not score naturalness. There is no local model that can, and inventing
a number would be worse than having none -- it would get quoted. The bundle
exists so a person can listen and decide.
"""

from __future__ import annotations

import argparse
import json
import statistics
import time
import zlib
import wave
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path

from intelligence.delivery import DeliveryPlanner
from intelligence.host_state import HostState
from platform_.audio.router import group_for_prosody, plan_units
from platform_.audio.timing import trim_edge_silence
from platform_.tts.base import Prosody
from shared.contracts import MarketConfidence, MarketEvent, MarketEventKind
from shared.delivery import PIPER_CAPABILITIES, DeliveryPlan

T0 = datetime(2026, 9, 7, 12, 0, tzinfo=timezone.utc)

#: The gap the legacy path produced, measured on this machine: trailing
#: padding + the next chunk's synthesis + leading padding. Reproduced as a
#: constant so the legacy condition sounds like what it actually sounded like
#: rather than like the current path with the improvements switched off.
LEGACY_GAP_MS = 562


@dataclass
class Step:
    """One thing that happens, in order."""

    kind: str  # "event" | "stale" | "live" | "wait" | "speak"
    at_s: float = 0.0
    event_kind: MarketEventKind | None = None
    severity: int = 3
    text: tuple[str, ...] = ()


@dataclass
class Scenario:
    key: str
    title: str
    why: str
    steps: list[Step] = field(default_factory=list)


NEUTRAL_LINES = (
    "Price is holding just under the level we marked out earlier.",
    "Nothing has changed about the structure since the session open.",
    "I am watching to see whether buyers defend it again.",
)
EVENT_LINES = (
    "The level just gave way.",
    "That is the third test and it did not hold this time.",
    "If we accept below it, the scenario I am watching flips.",
)
EXPLAIN_LINES = (
    "Worth explaining what that means.",
    "A break of structure is the market telling you who lost control.",
    "It does not tell you what happens next, only what changed.",
)


def ev(kind: MarketEventKind, severity: int, at_s: float) -> Step:
    return Step("event", at_s, event_kind=kind, severity=severity)


def say(at_s: float, lines: tuple[str, ...] = NEUTRAL_LINES) -> Step:
    return Step("speak", at_s, text=lines)


SCENARIOS: list[Scenario] = [
    Scenario(
        "01_neutral", "Neutral commentary",
        "The common case. Most utterances follow no event at all and must "
        "sound ordinary -- a permanently animated host is as obviously "
        "synthetic as a flat one.",
        [say(0.0)],
    ),
    Scenario(
        "02_moderate", "Moderate event",
        "Severity 3. Should change how the host sounds without producing a "
        "spoken reaction.",
        [ev(MarketEventKind.LIQUIDITY_SWEEP, 3, 0.0), say(0.5, EVENT_LINES)],
    ),
    Scenario(
        "03_severe", "Sudden severe event",
        "Severity 5, unheralded. The case the whole expressive layer exists "
        "for: reaction, beat, then faster delivery.",
        [ev(MarketEventKind.VOL_EXPANSION, 5, 0.0), say(0.5, EVENT_LINES)],
    ),
    Scenario(
        "04_escalation", "Escalation of an existing event",
        "Same kind, higher severity. Novelty gating should let this through.",
        [
            ev(MarketEventKind.BOS, 3, 0.0), say(0.5, NEUTRAL_LINES),
            ev(MarketEventKind.BOS, 5, 6.0), say(6.5, EVENT_LINES),
        ],
    ),
    Scenario(
        "05_repeated", "Repeated events of the same kind",
        "The fake-human trap. Four identical breaks must produce one reaction, "
        "not four.",
        [
            ev(MarketEventKind.BOS, 4, 0.0), say(0.5, EVENT_LINES),
            ev(MarketEventKind.BOS, 4, 5.0), say(5.5, NEUTRAL_LINES),
            ev(MarketEventKind.BOS, 4, 10.0), say(10.5, EXPLAIN_LINES),
            ev(MarketEventKind.BOS, 4, 15.0), say(15.5, NEUTRAL_LINES),
        ],
    ),
    Scenario(
        "06_kind_change", "Change of event kind",
        "A different kind of event is genuinely new information and should "
        "read as such.",
        [
            ev(MarketEventKind.BOS, 4, 0.0), say(0.5, EVENT_LINES),
            ev(MarketEventKind.LIQUIDITY_SWEEP, 4, 8.0), say(8.5, EVENT_LINES),
        ],
    ),
    Scenario(
        "07_stale", "Uncertain / stale feed",
        "The host cannot verify a price and should sound like it: slower, with "
        "a beat before speaking.",
        [Step("stale", 0.0), say(0.5, EXPLAIN_LINES)],
    ),
    Scenario(
        "08_event_while_speaking", "Event arrives while the host is speaking",
        "State moves mid-utterance. The delivery of the CURRENT utterance was "
        "already decided, so the reaction belongs to the next one.",
        [say(0.0), ev(MarketEventKind.VOL_EXPANSION, 5, 1.0), say(4.0, EVENT_LINES)],
    ),
    Scenario(
        "09_event_after_reaction", "Event immediately after a reaction",
        "Tests the cooldown. A second reaction three seconds after the first "
        "would sound frantic.",
        [
            ev(MarketEventKind.BOS, 5, 0.0), say(0.5, EVENT_LINES),
            ev(MarketEventKind.CHOCH, 5, 3.0), say(3.5, EVENT_LINES),
        ],
    ),
    Scenario(
        "10_rapid_succession", "Several events in rapid succession",
        "A busy tape. Arousal should climb and stay up while reactions stay "
        "rare.",
        [
            ev(MarketEventKind.BOS, 4, 0.0),
            ev(MarketEventKind.LIQUIDITY_SWEEP, 5, 1.0),
            ev(MarketEventKind.VOL_EXPANSION, 5, 2.0),
            ev(MarketEventKind.CHOCH, 4, 3.0),
            say(3.5, EVENT_LINES),
        ],
    ),
    Scenario(
        "11_return_to_neutral", "Normal speech after an emotional event",
        "The decay test. Sixty seconds after a severity-5 event the host "
        "should sound ordinary again, not still keyed up.",
        [
            ev(MarketEventKind.VOL_EXPANSION, 5, 0.0), say(0.5, EVENT_LINES),
            Step("wait", 60.0), say(60.5, NEUTRAL_LINES),
        ],
    ),
    Scenario(
        "12_long_pause", "Long pause before an important line",
        "The largest deliberate silence the system can produce, so it can be "
        "judged against a dropout.",
        [Step("stale", 0.0), ev(MarketEventKind.NEWS, 5, 0.1), say(0.5, EVENT_LINES)],
    ),
]


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------


@dataclass
class Chunk:
    text: str
    path: Path
    synth_ms: float
    audio_ms: float
    gap_after_ms: int


def _render(voice, text: str, path: Path, prosody: Prosody | None, trim: bool) -> Chunk:
    from piper import SynthesisConfig

    config = None
    if prosody is not None:
        config = SynthesisConfig(
            length_scale=prosody.length_scale, noise_w_scale=prosody.noise_w_scale
        )
    started = time.perf_counter()
    with wave.open(str(path), "wb") as w:
        voice.synthesize_wav(text, w, syn_config=config)
    synth_ms = (time.perf_counter() - started) * 1000
    if trim:
        trim_edge_silence(path)
    with wave.open(str(path), "rb") as w:
        audio_ms = w.getnframes() / w.getframerate() * 1000
    return Chunk(text, path, round(synth_ms, 1), round(audio_ms, 1), 0)


def render_condition(
    voice, condition: str, segments: tuple[str, ...], plan: DeliveryPlan,
    work: Path, utterance_seed: int,
) -> list[Chunk]:
    """Produce the chunks and the silence between them for one condition."""
    from uuid import UUID

    uid = UUID(int=utterance_seed)
    work.mkdir(parents=True, exist_ok=True)

    if condition == "C_legacy":
        # One call per sentence, no plan, no trim, and the gap the sequential
        # path actually produced.
        chunks = []
        for i, sentence in enumerate(segments):
            c = _render(voice, sentence, work / f"{i:02d}.wav", None, trim=False)
            c.gap_after_ms = LEGACY_GAP_MS
            chunks.append(c)
        return chunks

    if condition == "B_timing":
        # Timing work, neutral delivery.
        neutral = DeliveryPlan.neutral().renderable_with(PIPER_CAPABILITIES)
        prosody = Prosody.for_utterance(uid, neutral.pace)
        chunks = []
        for i, text in enumerate(group_for_prosody(list(segments))):
            c = _render(voice, text, work / f"{i:02d}.wav", prosody, trim=True)
            c.gap_after_ms = neutral.gap_ms
            chunks.append(c)
        return chunks

    # A_full
    rendered = plan.renderable_with(PIPER_CAPABILITIES)
    prosody = Prosody.for_utterance(uid, rendered.pace)
    chunks = []
    for i, unit in enumerate(plan_units(rendered, list(segments))):
        c = _render(voice, unit.text, work / f"{i:02d}.wav", prosody, trim=True)
        c.gap_after_ms = unit.gap_after_ms
        chunks.append(c)
    return chunks


def stitch(chunks: list[Chunk], out: Path, lead_silence_ms: int = 0) -> float:
    """Join chunks into one continuous wav with the planned silence between.

    Stitching rather than shipping loose files is the point: the gaps ARE the
    thing under evaluation, and a listener clicking through separate files
    would never hear them.
    """
    frames = bytearray()
    rate = width = channels = None
    for i, chunk in enumerate(chunks):
        with wave.open(str(chunk.path), "rb") as w:
            channels, width, rate = w.getnchannels(), w.getsampwidth(), w.getframerate()
            if i == 0 and lead_silence_ms:
                frames.extend(b"\x00" * int(lead_silence_ms / 1000 * rate) * width)
            frames.extend(w.readframes(w.getnframes()))
        if i < len(chunks) - 1 and chunk.gap_after_ms:
            frames.extend(b"\x00" * int(chunk.gap_after_ms / 1000 * rate) * width)
    if rate is None:
        return 0.0
    out.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(out), "wb") as w:
        w.setnchannels(channels)
        w.setsampwidth(width)
        w.setframerate(rate)
        w.writeframes(bytes(frames))
    return len(frames) / width / rate * 1000


# ---------------------------------------------------------------------------
# Playing a scenario forward
# ---------------------------------------------------------------------------


def play_scenario(scenario: Scenario) -> list[dict]:
    """Run the scenario's timeline and collect what each utterance should be.

    Deterministic: same steps, same timestamps, same plans, every run.
    """
    state, planner = HostState(), DeliveryPlanner()
    startled_by: MarketEventKind | None = None
    utterances: list[dict] = []

    for step in scenario.steps:
        now = T0 + timedelta(seconds=step.at_s)
        if step.kind == "event":
            event = MarketEvent(
                kind=step.event_kind, timeframe="5m",
                occurred_at=now, severity=step.severity,
            )
            if state.observe_event(event, now):
                startled_by = event.kind
        elif step.kind == "stale":
            state.observe_market(MarketConfidence.STALE, now)
        elif step.kind == "live":
            state.observe_market(MarketConfidence.LIVE, now)
        elif step.kind == "wait":
            state.decay_to(now)
        elif step.kind == "speak":
            state.decay_to(now)
            plan = planner.plan(state, now, startling_event=startled_by)
            startled_by = None
            utterances.append(
                {
                    "at_s": step.at_s,
                    "segments": step.text,
                    "plan": plan,
                    "state": state.snapshot(),
                }
            )
    return utterances


CONDITIONS = ("C_legacy", "B_timing", "A_full")


def build(out_dir: Path, voice_path: str) -> dict:
    from piper import PiperVoice

    voice = PiperVoice.load(voice_path, use_cuda=False)
    report: dict = {"voice": voice_path, "scenarios": []}

    for scenario in SCENARIOS:
        utterances = play_scenario(scenario)
        entry = {
            "key": scenario.key,
            "title": scenario.title,
            "why": scenario.why,
            "utterances": [],
        }
        for u_index, utterance in enumerate(utterances):
            plan: DeliveryPlan = utterance["plan"]
            rendered = plan.renderable_with(PIPER_CAPABILITIES)
            row = {
                "index": u_index,
                "at_s": utterance["at_s"],
                "state": utterance["state"],
                "plan": {
                    "pace": round(rendered.pace, 4),
                    "gap_ms": rendered.gap_ms,
                    "pause_before_ms": rendered.pause_before_ms,
                    "pause_after_ms": rendered.pause_after_ms,
                    "reaction": rendered.reaction,
                },
                "conditions": {},
            }
            for condition in CONDITIONS:
                work = out_dir / scenario.key / f"{condition}_u{u_index}"
                chunks = render_condition(
                    voice, condition, utterance["segments"], plan, work,
                    # crc32, not hash(): str hashing is salted per process,
                    # so hash() here would make the bundle non-reproducible.
                    utterance_seed=zlib.crc32(
                        f"{scenario.key}:{u_index}".encode()
                    ),
                )
                stitched = out_dir / scenario.key / f"{condition}_u{u_index}.wav"
                lead = rendered.pause_before_ms if condition == "A_full" else 0
                total = stitch(chunks, stitched, lead_silence_ms=lead)
                words = sum(len(c.text.split()) for c in chunks)
                row["conditions"][condition] = {
                    "wav": str(stitched.relative_to(out_dir)).replace("\\", "/"),
                    "chunks": len(chunks),
                    "total_ms": round(total, 1),
                    "speech_ms": round(sum(c.audio_ms for c in chunks), 1),
                    "silence_ms": round(
                        total - sum(c.audio_ms for c in chunks), 1
                    ),
                    "gaps_ms": [c.gap_after_ms for c in chunks[:-1]],
                    "first_synth_ms": chunks[0].synth_ms if chunks else 0.0,
                    "wpm": round(
                        words / (sum(c.audio_ms for c in chunks) / 60000), 1
                    )
                    if chunks
                    else 0.0,
                }
            entry["utterances"].append(row)
        report["scenarios"].append(entry)
    return report


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------

INDEX_CSS = """
body{font:14px/1.5 system-ui,sans-serif;margin:0;padding:2rem;max-width:1100px;
background:#faf9f7;color:#1a1a1a}
h1{font-size:1.5rem;margin:0 0 .3rem} h2{font-size:1.05rem;margin:2rem 0 .2rem}
.why{color:#555;margin:0 0 .8rem;max-width:70ch}
table{border-collapse:collapse;width:100%;margin:.4rem 0 1rem;font-size:13px}
th,td{text-align:left;padding:.35rem .6rem;border-bottom:1px solid #e5e2dd;
vertical-align:middle}
th{font-weight:600;color:#444;background:#f2f0ec}
audio{height:32px;width:230px}
code{background:#f2f0ec;padding:.1rem .3rem;border-radius:3px;font-size:12px}
.n{font-variant-numeric:tabular-nums;color:#555}
.lead{background:#fff;border:1px solid #e5e2dd;border-radius:6px;padding:1rem;
margin-bottom:1.5rem}
"""


def write_index(report: dict, out_dir: Path) -> None:
    rows = []
    for scenario in report["scenarios"]:
        rows.append(f"<h2>{scenario['title']}</h2>")
        rows.append(f"<p class='why'>{scenario['why']}</p>")
        for u in scenario["utterances"]:
            p = u["plan"]
            st = u["state"]
            rows.append(
                f"<p class='n'>utterance {u['index']} at t+{u['at_s']:.1f}s &middot; "
                f"arousal {st['arousal']} surprise {st['surprise']} "
                f"uncertainty {st['uncertainty']} &middot; "
                f"pace <code>{p['pace']}</code> gap <code>{p['gap_ms']}ms</code> "
                f"before <code>{p['pause_before_ms']}ms</code> "
                f"reaction <code>{p['reaction'] or '-'}</code></p>"
            )
            rows.append(
                "<table><tr><th>condition</th><th>listen</th><th>chunks</th>"
                "<th>total</th><th>silence</th><th>gaps</th><th>wpm</th></tr>"
            )
            for condition in CONDITIONS:
                c = u["conditions"][condition]
                rows.append(
                    f"<tr><td><b>{condition}</b></td>"
                    f"<td><audio controls preload='none' src='{c['wav']}'></audio></td>"
                    f"<td class='n'>{c['chunks']}</td>"
                    f"<td class='n'>{c['total_ms']:.0f}ms</td>"
                    f"<td class='n'>{c['silence_ms']:.0f}ms</td>"
                    f"<td class='n'>{c['gaps_ms']}</td>"
                    f"<td class='n'>{c['wpm']:.0f}</td></tr>"
                )
            rows.append("</table>")

    html = f"""<!doctype html><meta charset="utf-8">
<title>GoldLive listening bundle</title><style>{INDEX_CSS}</style>
<h1>GoldLive listening bundle</h1>
<div class="lead">
<p><b>C_legacy</b> &mdash; the audio path before any of this work: one synthesis
call per sentence, no prefetch, no trimming, no delivery plan.<br>
<b>B_timing</b> &mdash; the timing work only: grouped sentences, prefetch,
trimmed padding, fixed neutral gaps. No expression at all.<br>
<b>A_full</b> &mdash; the current system: timing plus delivery driven by host
state.</p>
<p><b>C &rarr; B</b> answers whether the timing work helped.
<b>B &rarr; A</b> answers whether the emotional layer helped.
If A and B sound the same to you, the emotional layer is not earning its
complexity and should be cut back to the part that does.</p>
<p class="why">Voice: <code>{report['voice']}</code>. Everything here is
deterministic &mdash; regenerating gives byte-identical audio.</p>
</div>
{''.join(rows)}
"""
    (out_dir / "index.html").write_text(html, encoding="utf-8")


def print_summary(report: dict) -> None:
    print(f"\nvoice: {report['voice']}")
    print(f"{'scenario':<26} {'cond':<10} {'chunks':>6} {'total':>9} "
          f"{'silence':>9} {'wpm':>6}  reaction")
    print("-" * 84)
    agg: dict[str, list[float]] = {c: [] for c in CONDITIONS}
    reactions = 0
    utterances = 0
    for scenario in report["scenarios"]:
        for u in scenario["utterances"]:
            utterances += 1
            if u["plan"]["reaction"]:
                reactions += 1
            for condition in CONDITIONS:
                c = u["conditions"][condition]
                agg[condition].append(c["silence_ms"])
                label = f"{scenario['key']}#{u['index']}"
                print(
                    f"{label:<26} {condition:<10} {c['chunks']:>6} "
                    f"{c['total_ms']:>8.0f}ms {c['silence_ms']:>8.0f}ms "
                    f"{c['wpm']:>6.0f}  "
                    f"{u['plan']['reaction'] if condition == 'A_full' else ''}"
                )
    print("-" * 84)
    for condition in CONDITIONS:
        values = agg[condition]
        print(f"  {condition:<10} mean silence per utterance "
              f"{statistics.mean(values):7.0f}ms")
    print(f"\n  reactions fired: {reactions} of {utterances} utterances "
          f"({reactions / utterances * 100:.0f}%)")


def main() -> None:
    ap = argparse.ArgumentParser(description="Build an A/B/C listening bundle")
    ap.add_argument("--out", type=Path, default=Path("bundle"))
    ap.add_argument("--voice", default="voices/en_US-john-medium.onnx")
    args = ap.parse_args()

    report = build(args.out, args.voice)
    (args.out / "manifest.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    write_index(report, args.out)
    print_summary(report)
    print(f"\n  wrote {args.out / 'index.html'}")


if __name__ == "__main__":
    main()
