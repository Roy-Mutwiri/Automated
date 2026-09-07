"""Blind A/B/C/D bake-off: does a different local voice actually sound better?

One variable at a time. The behavioural system, the text, the events, the
event timing, the HostState, the DeliveryPlan, the reaction decisions, the
utterance ids, the prosody seeds, the stitching and the output format are all
held identical across arms. Only the voice model changes -- except in the one
arm that deliberately disables expression, which exists to tell voice quality
apart from behaviour.

    python -m scripts.voice_bakeoff --out out/bakeoff
    # open out/bakeoff/index.html, listen, then read answer_key.txt

## The arms

    john + full      the current shipping voice and the current behaviour
    norman + full    a DIFFERENT audiobook voice, same behaviour. The control
                     that matters: if this differs from john as much as bryce
                     does, then any voice change moves the needle and corpus
                     type is not the variable.
    bryce + full     the hypothesis. The only public-domain voice in the
                     catalogue that is not audiobook-derived -- its model card
                     says "This is my voice, I only recorded about 750
                     samples", against LibriVox recordings for the others.
    bryce + timing   expression disabled, timing kept. Isolates whether any
                     perceived improvement is the voice or the behaviour.

## Levels

Different Piper voices render at noticeably different levels, and louder is
reliably mistaken for better. Each arm is scaled by ONE gain factor computed
across all of its files so that its loudest peak sits at -3 dBFS.

That is peak normalisation, not loudness normalisation: it removes the gross
level difference between voice models while preserving every dynamic
difference within and between the scenarios of one arm. Deliberately not LUFS
matching, which would flatten exactly the expressive loudness differences a
later experiment wants to measure.

## What this does not do

It does not score anything. There is no trustworthy local perceptual metric,
and a number invented here would be quoted later as though it meant something.
The objective measurements are reported separately from the listening test,
and the listening test is a person with headphones.
"""

from __future__ import annotations

import argparse
import json
import random
import statistics
import wave
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from scripts.listening_bundle import (
    SCENARIOS,
    Chunk,
    play_scenario,
    render_condition,
    stitch,
)
from shared.delivery import PIPER_CAPABILITIES, DeliveryPlan

#: The scenarios that carry the most information per minute of listening:
#: a surprise, reaction frequency under repetition, emotional decay back to
#: baseline, and whether the voice can simply sustain ordinary commentary.
FOCUS = ("01_neutral", "03_severe", "05_repeated", "11_return_to_neutral")

#: Peak target. -3 dBFS leaves headroom for the broadcast chain rather than
#: pinning everything to full scale.
PEAK_TARGET = 0.708

#: Fixed so the blind labelling is reproducible but not inferable from the
#: order the arms are declared in.
SHUFFLE_SEED = 20260907


@dataclass(frozen=True)
class Arm:
    key: str
    voice: str
    mode: str  # "A_full" | "B_timing"
    note: str


ARMS = (
    Arm("john_full", "en_US-john-medium", "A_full", "current shipping voice, current behaviour"),
    Arm("norman_full", "en_US-norman-medium", "A_full", "different audiobook voice, same behaviour"),
    Arm("bryce_full", "en_US-bryce-medium", "A_full", "non-audiobook voice, same behaviour"),
    Arm("bryce_timing", "en_US-bryce-medium", "B_timing", "non-audiobook voice, expression OFF"),
)


def read_wav(path: Path) -> tuple[int, np.ndarray]:
    with wave.open(str(path), "rb") as w:
        sr, n = w.getframerate(), w.getnframes()
        raw = w.readframes(n)
    return sr, np.frombuffer(raw, dtype="<i2").astype(np.float32) / 32768.0


def write_wav(path: Path, sr: int, samples: np.ndarray) -> None:
    clipped = np.clip(samples, -1.0, 1.0)
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sr)
        w.writeframes((clipped * 32767).astype("<i2").tobytes())


def f0_stats(a: np.ndarray, sr: int) -> tuple[float, float]:
    """Mean and spread of the pitch track. Crude but consistent across arms."""
    hop, win = int(0.01 * sr), int(0.04 * sr)
    lo, hi = int(sr / 350), int(sr / 60)
    out: list[float] = []
    for i in range(0, max(0, len(a) - win), hop):
        frame = a[i : i + win]
        frame = frame - frame.mean()
        if np.sqrt((frame**2).mean()) < 0.01:
            continue
        corr = np.correlate(frame, frame, "full")[win - 1 :]
        if corr[0] <= 0:
            continue
        corr = corr / corr[0]
        window = corr[lo:hi]
        if not len(window):
            continue
        k = int(np.argmax(window)) + lo
        if corr[k] < 0.3:
            continue
        out.append(sr / k)
    if len(out) < 5:
        return 0.0, 0.0
    return float(np.mean(out)), float(np.std(out))


def build_arm(arm: Arm, voices_dir: Path, work_root: Path) -> dict:
    """Render every focus scenario for one arm. Nothing here varies by arm
    except the voice model and the mode."""
    from piper import PiperVoice

    voice = PiperVoice.load(str(voices_dir / f"{arm.voice}.onnx"), use_cuda=False)
    import zlib

    files: list[dict] = []
    for scenario in SCENARIOS:
        if scenario.key not in FOCUS:
            continue
        # Identical to the main bundle: same events, same clock, same state.
        for u_index, utterance in enumerate(play_scenario(scenario)):
            plan: DeliveryPlan = utterance["plan"]
            work = work_root / arm.key / f"{scenario.key}_u{u_index}"
            chunks: list[Chunk] = render_condition(
                voice,
                arm.mode,
                utterance["segments"],
                plan,
                work,
                # The SAME seed as the main bundle, so prosody jitter is
                # identical across arms and cannot be a confound.
                utterance_seed=zlib.crc32(f"{scenario.key}:{u_index}".encode()),
            )
            # The plan that was ACTUALLY rendered. render_condition ignores
            # the passed plan in timing-only mode and uses a neutral one, so
            # reporting the full plan here labelled the expression-off arm
            # with reactions it never spoke.
            effective = plan if arm.mode == "A_full" else DeliveryPlan.neutral()
            rendered = effective.renderable_with(PIPER_CAPABILITIES)
            lead = rendered.pause_before_ms
            out = work_root / arm.key / f"{scenario.key}_u{u_index}.wav"
            total = stitch(chunks, out, lead_silence_ms=lead)
            words = sum(len(c.text.split()) for c in chunks)
            speech_ms = sum(c.audio_ms for c in chunks)
            files.append(
                {
                    "scenario": scenario.key,
                    "title": scenario.title,
                    "utterance": u_index,
                    "path": out,
                    "total_ms": round(total, 1),
                    "speech_ms": round(speech_ms, 1),
                    "silence_ms": round(total - speech_ms, 1),
                    "chunks": len(chunks),
                    "gaps_ms": [c.gap_after_ms for c in chunks[:-1]],
                    "reaction": rendered.reaction,
                    "pace": round(rendered.pace, 4),
                    "first_synth_ms": chunks[0].synth_ms if chunks else 0.0,
                    "rtf": round(
                        sum(c.synth_ms for c in chunks) / speech_ms, 4
                    )
                    if speech_ms
                    else 0.0,
                    "wpm": round(words / (speech_ms / 60000), 1) if speech_ms else 0.0,
                }
            )
    return {"arm": arm, "files": files}


def level_match(built: dict, out_dir: Path) -> float:
    """One gain per arm, so voices are level-matched but dynamics survive."""
    peak = 0.0
    loaded: list[tuple[dict, int, np.ndarray]] = []
    for entry in built["files"]:
        sr, a = read_wav(entry["path"])
        loaded.append((entry, sr, a))
        if a.size:
            peak = max(peak, float(np.abs(a).max()))
    gain = PEAK_TARGET / peak if peak > 0 else 1.0

    for entry, sr, a in loaded:
        scaled = a * gain
        dest = out_dir / built["arm"].key / entry["path"].name
        dest.parent.mkdir(parents=True, exist_ok=True)
        write_wav(dest, sr, scaled)
        entry["wav"] = f"{built['arm'].key}/{dest.name}"
        entry["rms_dbfs"] = round(
            20 * np.log10(max(1e-9, float(np.sqrt((scaled**2).mean())))), 1
        )
        entry["peak_dbfs"] = round(
            20 * np.log10(max(1e-9, float(np.abs(scaled).max()))), 1
        )
        mean_f0, sd_f0 = f0_stats(scaled, sr)
        entry["f0_mean_hz"] = round(mean_f0, 1)
        entry["f0_sd_hz"] = round(sd_f0, 1)
        del entry["path"]
    return gain


CSS = """
body{font:14px/1.55 system-ui,sans-serif;margin:0;padding:2rem;max-width:1000px;
background:#faf9f7;color:#1a1a1a}
h1{font-size:1.5rem;margin:0 0 .2rem}h2{font-size:1.05rem;margin:2rem 0 .3rem}
.lead{background:#fff;border:1px solid #e5e2dd;border-radius:6px;padding:1rem 1.2rem;
margin-bottom:1.4rem}
table{border-collapse:collapse;width:100%;margin:.4rem 0 1rem;font-size:13px}
th,td{text-align:left;padding:.4rem .6rem;border-bottom:1px solid #e5e2dd}
th{background:#f2f0ec;font-weight:600}
audio{height:32px;width:250px}
ol{padding-left:1.2rem}li{margin:.3rem 0}
.q{background:#fff;border:1px solid #e5e2dd;border-radius:6px;padding:1rem 1.2rem}
code{background:#f2f0ec;padding:.1rem .3rem;border-radius:3px}
.muted{color:#666}
"""


def write_index(report: dict, out_dir: Path) -> None:
    labels = report["labels"]  # blind label -> arm key
    blind = sorted(labels)
    rows = []
    by_scenario: dict[str, dict] = {}
    for arm_key, entries in report["files_by_arm"].items():
        for e in entries:
            by_scenario.setdefault(f"{e['scenario']}#{e['utterance']}", {})[arm_key] = e

    for scen_key in sorted(by_scenario):
        first = next(iter(by_scenario[scen_key].values()))
        rows.append(f"<h2>{first['title']} <span class='muted'>({scen_key})</span></h2>")
        rows.append("<table><tr><th>sample</th><th>listen</th></tr>")
        for label in blind:
            arm_key = labels[label]
            e = by_scenario[scen_key].get(arm_key)
            if not e:
                continue
            rows.append(
                f"<tr><td><b>{label}</b></td>"
                f"<td><audio controls preload='none' src='{e['wav']}'></audio></td></tr>"
            )
        rows.append("</table>")

    html = f"""<!doctype html><meta charset="utf-8">
<title>Voice bake-off (blind)</title><style>{CSS}</style>
<h1>Voice bake-off &mdash; blind listening</h1>
<div class="lead">
<p>Four samples per scenario, labelled <b>{', '.join(blind)}</b>. The mapping
from label to voice is in <code>answer_key.txt</code> &mdash; do not open it
until you have written your answers down.</p>
<p>Everything except the voice model is identical across samples: same text,
same events, same timing, same host state, same delivery plan, same reaction
decisions, same prosody seeds, same stitching. One of the four has the
expressive behaviour switched off; the rest differ only by voice.</p>
<p>Each sample is peak-matched to &minus;3&nbsp;dBFS per voice, so loudness
does not decide the winner. Use headphones. Listen to the whole file.</p>
</div>
<div class="q">
<p><b>Answer these before opening the key:</b></p>
<ol>
<li>Which sounds most natural?</li>
<li>Which sounds most like a live host?</li>
<li>Which sounds most like an audiobook / TTS narrator?</li>
<li>Which handles the reactions most naturally? (see <code>03_severe</code>)</li>
<li>Which handles return-to-neutral best? (see <code>11_return_to_neutral</code>)</li>
<li>Which would you actually use for a live stream?</li>
</ol>
<p class="muted">A seventh, and the one that decides what happens next: with
the best of these, does the <em>behaviour</em> already feel like a live host,
or does the voice still give it away?</p>
</div>
{''.join(rows)}
"""
    (out_dir / "index.html").write_text(html, encoding="utf-8")


def print_measurements(report: dict) -> None:
    print(f"\n{'arm':<14} {'scenario':<24} {'total':>8} {'silence':>8} {'wpm':>6} "
          f"{'f0':>7} {'rms':>7} {'rtf':>6}  reaction")
    print("-" * 96)
    for arm_key, entries in report["files_by_arm"].items():
        for e in entries:
            print(
                f"{arm_key:<14} {e['scenario'] + '#' + str(e['utterance']):<24} "
                f"{e['total_ms']:>7.0f}ms {e['silence_ms']:>7.0f}ms {e['wpm']:>6.0f} "
                f"{e['f0_mean_hz']:>6.0f}Hz {e['rms_dbfs']:>6.1f}dB {e['rtf']:>6.3f}  "
                f"{e['reaction'] or ''}"
            )
    print("-" * 96)
    print(f"{'arm':<14} {'mean wpm':>9} {'mean f0':>9} {'f0 sd':>8} {'mean rtf':>9} "
          f"{'gain':>7}")
    for arm_key, entries in report["files_by_arm"].items():
        print(
            f"{arm_key:<14} "
            f"{statistics.mean(e['wpm'] for e in entries):>9.1f} "
            f"{statistics.mean(e['f0_mean_hz'] for e in entries):>8.0f}Hz "
            f"{statistics.mean(e['f0_sd_hz'] for e in entries):>7.0f}Hz "
            f"{statistics.mean(e['rtf'] for e in entries):>9.3f} "
            f"{report['gains'][arm_key]:>7.2f}"
        )


def main() -> None:
    ap = argparse.ArgumentParser(description="Blind voice bake-off")
    ap.add_argument("--out", type=Path, default=Path("out/bakeoff"))
    ap.add_argument("--voices", type=Path, default=Path("voices"))
    args = ap.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    work = args.out / "_work"

    files_by_arm: dict[str, list[dict]] = {}
    gains: dict[str, float] = {}
    for arm in ARMS:
        model = args.voices / f"{arm.voice}.onnx"
        if not model.exists():
            print(f"  skipping {arm.key}: {model} not installed")
            continue
        print(f"  rendering {arm.key} ({arm.voice}) ...")
        built = build_arm(arm, args.voices, work)
        gains[arm.key] = round(level_match(built, args.out), 3)
        files_by_arm[arm.key] = built["files"]

    rng = random.Random(SHUFFLE_SEED)
    keys = list(files_by_arm)
    shuffled = keys[:]
    rng.shuffle(shuffled)
    labels = {f"Sample {chr(65 + i)}": key for i, key in enumerate(shuffled)}

    report = {"files_by_arm": files_by_arm, "gains": gains, "labels": labels}
    write_index(report, args.out)

    notes = {a.key: f"{a.voice}  [{a.mode}]  {a.note}" for a in ARMS}
    key_lines = ["ANSWER KEY -- do not read before listening", ""]
    for label in sorted(labels):
        key_lines.append(f"  {label}  =  {notes[labels[label]]}")
    key_lines += [
        "",
        "Per-arm peak gain applied (level matching, not loudness matching):",
    ]
    for arm_key, gain in gains.items():
        key_lines.append(f"  {arm_key:<14} x{gain}")
    (args.out / "answer_key.txt").write_text("\n".join(key_lines), encoding="utf-8")

    (args.out / "measurements.json").write_text(
        json.dumps(
            {"gains": gains, "files_by_arm": files_by_arm}, indent=2, default=str
        ),
        encoding="utf-8",
    )
    print_measurements(report)
    print(f"\n  blind page : {args.out / 'index.html'}")
    print(f"  answer key : {args.out / 'answer_key.txt'}  (read it afterwards)")


if __name__ == "__main__":
    main()
