"""Readiness gates.

Every gate here exists to catch a fallback that would otherwise let the system
report READY while producing nothing: the synthetic feed, the offline text
generator, the placeholder tone, the audio sink that pretends to have played.
So the important tests are the ones that assert a gate FAILS -- a gate that
only ever passes is indistinguishable from no gate at all.

CI-safe: no network, no model server, no audio hardware.
"""

from __future__ import annotations

import wave

import pytest

from runtime.readiness import (
    BROADCAST_GATES,
    SESSION_GATES,
    GateResult,
    Level,
    Readiness,
    _inspect_wav,
    _peak_amplitude,
    gate_comments,
    gate_market_live,
    gate_model_real,
    gate_voice_real,
)


def ok(name: str) -> GateResult:
    return GateResult(name, True, evidence="fine")


def bad(name: str) -> GateResult:
    return GateResult(name, False, reason="nope")


# -- the two levels --------------------------------------------------------


def test_all_gates_passing_is_broadcast_ready():
    r = Readiness(gates=[ok(n) for n in (*SESSION_GATES, *BROADCAST_GATES)])
    assert r.level is Level.BROADCAST_READY


def test_a_missing_cable_still_leaves_the_session_ready():
    """The core AI session works perfectly well before VB-CABLE exists;
    conflating the two would block a working session on a broadcast detail."""
    gates = [ok(n) for n in SESSION_GATES] + [bad("broadcast_route"), ok("comments")]
    r = Readiness(gates=gates)
    assert r.level is Level.SESSION_READY
    assert "Broadcasting is not set up" in r.render()


def test_missing_comments_does_not_block_a_session():
    gates = [ok(n) for n in SESSION_GATES] + [ok("broadcast_route"), bad("comments")]
    assert Readiness(gates=gates).level is Level.SESSION_READY


@pytest.mark.parametrize("failing", SESSION_GATES)
def test_any_core_gate_failing_blocks_everything(failing):
    gates = [ok(n) if n != failing else bad(n) for n in SESSION_GATES]
    gates += [ok(n) for n in BROADCAST_GATES]
    r = Readiness(gates=gates)
    assert r.level is Level.NOT_READY
    assert failing in r.render()


def test_a_failed_gate_renders_an_action_not_just_FAIL():
    g = GateResult("model_real", False, reason="no models installed",
                   action="run: ollama pull llama3.2:3b")
    text = g.render()
    assert "no models installed" in text and "ollama pull" in text


# -- market_live -----------------------------------------------------------


@pytest.mark.parametrize("market", ["synthetic", "replay"])
async def test_simulated_feeds_can_never_satisfy_market_live(market):
    """Both emit well-formed ticks, so the engine would report LIVE and
    may_quote_price() would be True. Rejecting them by name is the only way to
    stop the host quoting a price that never existed."""
    result = await gate_market_live(market)
    assert not result.ok
    assert "simulated" in result.reason
    assert "--market gold" in result.action


async def test_market_live_fails_when_no_tick_arrives(monkeypatch):
    import runtime.readiness as mod

    class SilentFeed:
        def staleness_thresholds(self):
            return 6.0, 20.0, 120.0

        async def ticks(self):
            import asyncio

            await asyncio.sleep(10)
            yield None  # never reached

        async def close(self):
            return None

    monkeypatch.setattr("runtime.live.build_feed", lambda *_a, **_k: SilentFeed())
    result = await mod.gate_market_live("gold", timeout_s=0.2)
    assert not result.ok and "no tick" in result.reason


# -- model_real ------------------------------------------------------------


async def test_model_real_fails_when_nothing_is_serving(monkeypatch):
    async def no_server(*_a, **_kw):
        return None

    monkeypatch.setattr("platform_.llm.discovery.discover", no_server)
    result = await gate_model_real()
    assert not result.ok
    assert "no models installed" in result.reason or "no local model server" in result.reason
    assert "ollama" in result.action.lower()


async def test_model_real_rejects_the_placeholder_model_name(monkeypatch):
    """LocalLLM falls back to the literal "local-model" when nothing is
    configured. A server answering under that name has not proved anything."""
    class FakeLLM:
        model = "local-model"

        async def close(self):
            return None

    async def fake_discover(*_a, **_kw):
        return FakeLLM()

    monkeypatch.setattr("platform_.llm.discovery.discover", fake_discover)
    result = await gate_model_real()
    assert not result.ok and "did not name an installed model" in result.reason


async def test_model_real_rejects_a_too_short_answer(monkeypatch):
    """A server that responds but generates nothing useful is not ready."""
    class Result:
        text = "Sure."

    class FakeLLM:
        model = "llama3.2:3b"

        async def complete(self, *_a, **_kw):
            return Result()

        async def close(self):
            return None

    async def fake_discover(*_a, **_kw):
        return FakeLLM()

    monkeypatch.setattr("platform_.llm.discovery.discover", fake_discover)
    result = await gate_model_real(min_words=20)
    assert not result.ok and "1 words" in result.reason


async def test_model_real_passes_on_a_real_answer(monkeypatch):
    class Result:
        text = " ".join(["word"] * 40)

    class FakeLLM:
        model = "llama3.2:3b"

        async def complete(self, *_a, **_kw):
            return Result()

        async def close(self):
            return None

    async def fake_discover(*_a, **_kw):
        return FakeLLM()

    monkeypatch.setattr("platform_.llm.discovery.discover", fake_discover)
    result = await gate_model_real(min_words=20)
    assert result.ok and "40 words" in result.evidence


# -- voice_real ------------------------------------------------------------


async def test_voice_real_fails_when_the_voice_is_not_installed(tmp_path):
    result = await gate_voice_real("en_US-john-medium", tmp_path)
    assert not result.ok
    assert "not installed" in result.reason
    assert "provision" in result.action


async def test_voice_real_rejects_a_fallback_to_a_different_voice(tmp_path, monkeypatch):
    """PiperTTS._model_path silently substitutes the default voice when the
    requested one is absent. Without this check a machine holding one voice
    passes for every voice, then broadcasts in the wrong one."""
    pytest.importorskip("piper")
    (tmp_path / "wanted.onnx").write_bytes(b"not really a model")

    from platform_.tts import piper as piper_mod

    monkeypatch.setattr(
        piper_mod.PiperTTS, "_model_path",
        lambda self, voice_id: tmp_path / "something-else.onnx",
    )
    result = await gate_voice_real("wanted", tmp_path)
    assert not result.ok
    assert "resolved" in result.reason


# -- wav inspection --------------------------------------------------------


def write_wav(path, seconds: float, amplitude: int, rate: int = 22050):
    import struct

    frames = int(rate * seconds)
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(rate)
        w.writeframes(b"".join(struct.pack("<h", amplitude) for _ in range(frames)))


def test_silence_is_rejected(tmp_path):
    path = tmp_path / "silent.wav"
    write_wav(path, seconds=4.5, amplitude=0)
    ok_, reason = _inspect_wav(path)
    assert not ok_ and "silent" in reason


def test_an_empty_wav_is_rejected(tmp_path):
    path = tmp_path / "empty.wav"
    write_wav(path, seconds=0, amplitude=0)
    ok_, reason = _inspect_wav(path)
    assert not ok_ and "no audio" in reason


def test_an_implausibly_short_render_is_rejected(tmp_path):
    """Thirteen words cannot be spoken in a fifth of a second."""
    path = tmp_path / "tiny.wav"
    write_wav(path, seconds=0.2, amplitude=8000)
    ok_, reason = _inspect_wav(path)
    assert not ok_ and "implausible" in reason


def test_a_plausible_render_passes(tmp_path):
    path = tmp_path / "good.wav"
    write_wav(path, seconds=4.5, amplitude=12000)
    ok_, detail = _inspect_wav(path)
    assert ok_ and "peak" in detail


def test_a_corrupt_file_is_rejected(tmp_path):
    path = tmp_path / "junk.wav"
    path.write_bytes(b"this is not a wav file at all")
    ok_, reason = _inspect_wav(path)
    assert not ok_ and "unreadable" in reason


def test_peak_amplitude_is_normalised():
    import struct

    loud = b"".join(struct.pack("<h", 32000) for _ in range(50))
    quiet = b"".join(struct.pack("<h", 20) for _ in range(50))
    assert _peak_amplitude(loud, 2) > 0.9
    assert _peak_amplitude(quiet, 2) < 0.01


# -- comments --------------------------------------------------------------


async def test_the_mock_adapter_never_counts_as_a_comment_source():
    result = await gate_comments("mock")
    assert not result.ok and "development stub" in result.reason


async def test_screen_adapter_reports_missing_ocr():
    result = await gate_comments("screen")
    assert not result.ok
    assert "paddleocr" in result.reason or "calibrated" in result.reason


async def test_an_unknown_adapter_is_rejected():
    result = await gate_comments("carrier-pigeon")
    assert not result.ok and "unknown adapter" in result.reason
