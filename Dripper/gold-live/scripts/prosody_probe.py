"""Measure how the host actually sounds, so "more natural" can be checked.

Runs the real voice, not a mock, and reports the numbers that move when
delivery changes. Run it before and after any change to the audio path:

    python -m scripts.prosody_probe
    python -m scripts.prosody_probe --json out.json

What it measures, and why each one is here:

  inter-chunk gap      dead air inside a single utterance. The router used to
                       synthesise and play strictly in sequence, so the gap
                       was trailing silence + synthesis + leading silence.
                       This is the number that made the host sound assembled
                       rather than spoken.
  real-time factor     synthesis wall time / audio duration. Decides whether
                       prefetching the next chunk is affordable at all.
  edge silence         near-silence Piper pads onto every render. Invisible in
                       one file, additive across chunks.
  f0 contour           start and end pitch, and the terminal fall. A full fall
                       at every boundary is what makes one thought sound like
                       several separate announcements.
  speaking rate        words per minute actually rendered.
  duration variance    spread across repeated renders of identical text. Piper
                       is stochastic, so this says how much of an observed
                       difference is just noise.
"""

from __future__ import annotations

import argparse
import json
import statistics
import time
import wave
from dataclasses import asdict, dataclass
from pathlib import Path
from tempfile import mkdtemp

import numpy as np

#: One lead sentence and a two-sentence tail: the shape group_for_prosody
#: produces, so the probe measures what the router actually renders.
LEAD = "Okay, look at what just happened."
TAIL = (
    "Volatility just expanded and ranges are widening. "
    "If that holds, the scenario I am watching is continuation."
)

#: Fraction of peak amplitude below which a sample counts as silence. Piper
#: pads with near-silence rather than true zeros, so an absolute floor misses
#: it entirely.
SILENCE_FRACTION = 0.02


@dataclass
class Render:
    label: str
    synth_ms: float
    audio_ms: float
    lead_silence_ms: float
    trail_silence_ms: float
    f0_start_hz: float
    f0_end_hz: float
    terminal_fall_hz: float
    words_per_minute: float

    @property
    def rtf(self) -> float:
        return self.synth_ms / self.audio_ms if self.audio_ms else 0.0


def read_wav(path: Path) -> tuple[int, np.ndarray]:
    with wave.open(str(path), "rb") as w:
        sr, n = w.getframerate(), w.getnframes()
        raw = w.readframes(n)
    return sr, np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0


def edge_silence_ms(a: np.ndarray, sr: int) -> tuple[float, float]:
    env = np.abs(a)
    if not len(env) or env.max() == 0:
        return 0.0, 0.0
    voiced = np.nonzero(env > env.max() * SILENCE_FRACTION)[0]
    if not len(voiced):
        return 0.0, 0.0
    return voiced[0] / sr * 1000, (len(a) - voiced[-1]) / sr * 1000


def f0_track(a: np.ndarray, sr: int, fmin: int = 60, fmax: int = 350) -> np.ndarray:
    """Autocorrelation pitch track.

    Crude, and deliberately so: it is not trying to be a pitch detector, only
    to be consistent enough that the same measurement before and after a change
    can be compared.
    """
    hop, win = int(0.01 * sr), int(0.04 * sr)
    lo, hi = int(sr / fmax), int(sr / fmin)
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
    return np.array(out)


def measure(voice, text: str, label: str, out_dir: Path, syn_config=None) -> Render:
    path = out_dir / f"{label}.wav"
    started = time.perf_counter()
    with wave.open(str(path), "wb") as w:
        voice.synthesize_wav(text, w, syn_config=syn_config)
    synth_ms = (time.perf_counter() - started) * 1000

    sr, a = read_wav(path)
    audio_ms = len(a) / sr * 1000
    lead_sil, trail_sil = edge_silence_ms(a, sr)
    f0 = f0_track(a, sr)
    if len(f0) >= 5:
        head = float(f0[: len(f0) // 5].mean())
        tail = float(f0[-(len(f0) // 5) :].mean())
    else:
        head = tail = 0.0
    words = max(1, len(text.split()))
    return Render(
        label=label,
        synth_ms=round(synth_ms, 1),
        audio_ms=round(audio_ms, 1),
        lead_silence_ms=round(lead_sil, 1),
        trail_silence_ms=round(trail_sil, 1),
        f0_start_hz=round(head, 1),
        f0_end_hz=round(tail, 1),
        terminal_fall_hz=round(head - tail, 1),
        words_per_minute=round(words / (audio_ms / 1000) * 60, 1),
    )


def _stats(renders: list[Render], attr: str) -> tuple[float, float]:
    values = [getattr(r, attr) for r in renders]
    return statistics.mean(values), statistics.pstdev(values)


def build_report(voice, repeats: int, voice_name: str) -> dict:
    out_dir = Path(mkdtemp())
    leads = [measure(voice, LEAD, f"lead{i}", out_dir) for i in range(repeats)]
    tails = [measure(voice, TAIL, f"tail{i}", out_dir) for i in range(repeats)]

    # What a listener hears between the lead and the tail when nothing is
    # prefetched and no edge silence is trimmed.
    gaps = [
        lead.trail_silence_ms + tail.synth_ms + tail.lead_silence_ms
        for lead, tail in zip(leads, tails, strict=True)
    ]

    def side(rs: list[Render]) -> dict:
        audio_mean, audio_sd = _stats(rs, "audio_ms")
        return {
            "audio_ms_mean": round(audio_mean, 1),
            "audio_ms_sd": round(audio_sd, 1),
            "synth_ms_mean": round(_stats(rs, "synth_ms")[0], 1),
            "rtf_mean": round(statistics.mean([r.rtf for r in rs]), 3),
            "terminal_fall_hz_mean": round(_stats(rs, "terminal_fall_hz")[0], 1),
            "wpm_mean": round(_stats(rs, "words_per_minute")[0], 1),
        }

    return {
        "voice": voice_name,
        "repeats": repeats,
        "lead": side(leads),
        "tail": side(tails),
        "edge_silence_ms": {
            "lead_in_mean": round(_stats(leads, "lead_silence_ms")[0], 1),
            "lead_out_mean": round(_stats(leads, "trail_silence_ms")[0], 1),
            "tail_in_mean": round(_stats(tails, "lead_silence_ms")[0], 1),
            "tail_out_mean": round(_stats(tails, "trail_silence_ms")[0], 1),
        },
        "inter_chunk_gap_ms": {
            "mean": round(statistics.mean(gaps), 1),
            "min": round(min(gaps), 1),
            "max": round(max(gaps), 1),
        },
        "renders": [asdict(r) for r in leads + tails],
    }


def print_report(report: dict) -> None:
    lead, tail = report["lead"], report["tail"]
    edge, gap = report["edge_silence_ms"], report["inter_chunk_gap_ms"]
    print(f"voice: {report['voice']}  n={report['repeats']}")
    for name, side in (("lead", lead), ("tail", tail)):
        print(
            f"  {name}  {side['audio_ms_mean']:7.1f}ms audio  "
            f"rtf={side['rtf_mean']:.3f}  "
            f"fall={side['terminal_fall_hz_mean']:+6.1f}Hz  "
            f"{side['wpm_mean']:5.1f}wpm"
        )
    print(
        f"  edge silence   lead {edge['lead_in_mean']:.0f}/{edge['lead_out_mean']:.0f}ms"
        f"   tail {edge['tail_in_mean']:.0f}/{edge['tail_out_mean']:.0f}ms"
    )
    print(
        f"  INTER-CHUNK GAP  mean={gap['mean']:.0f}ms  "
        f"min={gap['min']:.0f}  max={gap['max']:.0f}"
    )
    print(
        f"  duration sd      lead={lead['audio_ms_sd']:.0f}ms  "
        f"tail={tail['audio_ms_sd']:.0f}ms"
    )


def main() -> None:
    ap = argparse.ArgumentParser(description="Measure rendered delivery")
    ap.add_argument("--voice", default="voices/en_US-john-medium.onnx")
    ap.add_argument("--repeats", type=int, default=5)
    ap.add_argument("--json", type=Path)
    args = ap.parse_args()

    from piper import PiperVoice

    voice = PiperVoice.load(args.voice, use_cuda=False)
    report = build_report(voice, args.repeats, args.voice)

    if args.json:
        args.json.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print_report(report)


if __name__ == "__main__":
    main()
