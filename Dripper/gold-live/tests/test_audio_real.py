"""Real audio file handling, with the actual libraries.

The audio path had only been tested against a fake sink. These use the real
soundfile and sounddevice packages to check the things that break on a machine
that is not this one: wav files that are readable by the playback library,
sample rates that survive the round trip, and device enumeration.

Actual playback is opt-in (GOLDLIVE_TEST_PLAYBACK=1) because a test suite that
makes noise on a developer's machine is a test suite people stop running.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from platform_.audio.devices import (
    describe_audio_setup,
    find_virtual_cable,
    list_output_devices,
)
from shared.mocks.tts import FileTTS

sf = pytest.importorskip("soundfile")


async def test_generated_wav_is_readable_by_the_playback_library(tmp_path):
    """FileTTS writes wav with the wave module; playback reads with soundfile.
    Two different libraries, and they have to agree."""
    tts = FileTTS()
    out = tmp_path / "segment.wav"
    result = await tts.synthesize(
        "Price closed above the prior high, which is the level worth watching.",
        "voice", out,
    )

    assert out.exists()
    data, rate = sf.read(str(out))
    assert rate == FileTTS.SAMPLE_RATE
    assert len(data) > 0

    actual_ms = len(data) / rate * 1000
    assert abs(actual_ms - result.duration_ms) < 50, "reported duration must match"


async def test_duration_tracks_word_count(tmp_path):
    """The Director marks itself speaking for this long. If it is wrong the
    host talks over itself or leaves dead air."""
    tts = FileTTS()
    short = await tts.synthesize("Short.", "v", tmp_path / "a.wav")
    long = await tts.synthesize(" ".join(["word"] * 40), "v", tmp_path / "b.wav")
    assert long.duration_ms > short.duration_ms * 3


async def test_sidecar_transcript_is_written(tmp_path):
    """Reading the transcript is how you judge output without listening to
    hours of audio."""
    tts = FileTTS()
    text = "One scenario I am watching is acceptance above that level."
    out = tmp_path / "seg.wav"
    await tts.synthesize(text, "v", out)
    assert out.with_suffix(".txt").read_text(encoding="utf-8") == text


async def test_empty_text_produces_no_file(tmp_path):
    from platform_.tts.piper import PiperTTS

    result = await PiperTTS(voices_dir=tmp_path).synthesize("   ", "v", tmp_path / "x.wav")
    assert result.path is None
    assert result.duration_ms == 0


def test_device_enumeration_returns_real_devices():
    devices = list_output_devices()
    if not devices:
        pytest.skip("no audio backend on this machine")
    assert all(d.channels > 0 for d in devices)
    assert all(d.sample_rate > 0 for d in devices)
    assert len({d.index for d in devices}) == len(devices), "indices must be unique"


def test_setup_description_guides_a_missing_cable():
    text = describe_audio_setup()
    if find_virtual_cable() is None and list_output_devices():
        assert "VB-CABLE" in text
        # The single most damaging misconfiguration: desktop audio sends every
        # system sound to the audience.
        assert "MICROPHONE" in text or "desktop audio" in text


def test_piper_reports_absence_rather_than_crashing(tmp_path):
    from platform_.tts.piper import PiperTTS

    tts = PiperTTS(voices_dir=tmp_path)
    assert isinstance(tts.available(), bool)


def test_piper_names_the_fix_when_no_voice_exists(tmp_path):
    from platform_.tts.piper import PiperTTS

    with pytest.raises(FileNotFoundError, match=r"get_voices"):
        PiperTTS(voices_dir=tmp_path)._model_path("en_GB-alba-medium")


@pytest.mark.skipif(
    os.environ.get("GOLDLIVE_TEST_PLAYBACK") != "1",
    reason="set GOLDLIVE_TEST_PLAYBACK=1 to play audio out loud",
)
async def test_actual_playback(tmp_path):
    from platform_.audio.router import AudioSink

    tts = FileTTS()
    out = tmp_path / "tone.wav"
    await tts.synthesize("Testing one two three.", "v", out)
    await AudioSink().play(out)  # audible


def test_sink_survives_a_missing_backend(tmp_path: Path):
    """A machine with no sound card must not crash the session -- audio is not
    the only reason the process exists."""
    import asyncio

    from platform_.audio.router import AudioSink

    missing = tmp_path / "nope.wav"
    sink = AudioSink(device_index=999_999)
    with pytest.raises(Exception):
        asyncio.run(sink.play(missing))


# -- piper latency ---------------------------------------------------------


def test_piper_holds_loaded_voices():
    """The whole latency fix. A subprocess per segment reloaded a 60-114MB
    model every utterance: 4.5s of wall time for 2s of audio, with no
    improvement across calls. Loading once took it to ~225ms."""
    from platform_.tts.piper import PiperTTS

    tts = PiperTTS(voices_dir=Path("voices"))
    assert hasattr(tts, "_voices"), "loaded voices must be cached"
    assert hasattr(tts, "warmup"), "voices must be loadable before going live"


def test_piper_default_voice_is_public_domain():
    """The shipped default must be redistributable. Several Piper English
    voices are CC BY-NC-SA; two of the first three picked here were."""
    from platform_.tts.piper import PiperTTS

    assert PiperTTS().default_voice == "en_US-ljspeech-high"


def test_shipped_personas_use_audited_voices():
    """Persona voice ids must name real, licence-checked models rather than
    placeholder labels like 'voice_warm'."""
    from intelligence.personas import load_personas
    from scripts.get_voices import CATALOGUE
    from shared.paths import config_dir

    for persona in load_personas(config_dir("personas")).values():
        assert persona.voice_id in CATALOGUE, (
            f"{persona.persona_id} uses {persona.voice_id!r}, which is not a "
            "known voice model"
        )
