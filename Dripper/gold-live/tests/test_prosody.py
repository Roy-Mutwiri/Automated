"""Delivery: how an utterance sounds, as opposed to what it says.

Two mechanisms, both aimed at the same failure -- a host that is intelligible
and completely, obviously synthetic.

Sentence splitting is right for streaming and wrong for prosody. A TTS call
sees only the text it is given, so a sentence rendered alone gets its own
complete intonation contour ending in a terminal pitch fall. Measured on
en_US-john-medium, a three-sentence utterance rendered per sentence fell ~40Hz
at each of the first two boundaries; the same text in one call ended within
5Hz of where it started. So the lead sentence is synthesised alone, for
latency, and the tail goes as one span.

And Piper renders with identical rate and rhythm every time unless told
otherwise, which across hundreds of utterances is its own tell.
"""

from __future__ import annotations

from uuid import uuid4

import pytest
from fakes import FakeSink, audio_request

from platform_.audio.router import LEAD_SEGMENTS, AudioRouter, group_for_prosody
from platform_.tts.base import (
    LENGTH_SCALE_BAND,
    NOISE_W_BAND,
    Prosody,
    TTSResult,
)
from shared.delivery import PACE_MAX, PACE_MIN


# -- grouping ---------------------------------------------------------------


@pytest.mark.parametrize(
    "segments, expected",
    [
        ([], []),
        (["One."], ["One."]),
        (["One.", "Two."], ["One.", "Two."]),
        (["One.", "Two.", "Three."], ["One.", "Two. Three."]),
        (["a", "b", "c", "d"], ["a", "b c d"]),
    ],
)
def test_only_the_lead_sentence_is_synthesised_alone(segments, expected):
    assert group_for_prosody(segments) == expected


def test_grouping_loses_no_words():
    """The property that must hold whatever the policy is."""
    segments = ["First one.", "Second one.", "Third one.", "Fourth."]
    assert " ".join(group_for_prosody(segments)) == " ".join(segments)


def test_grouping_drops_empty_segments_rather_than_synthesising_silence():
    assert group_for_prosody(["a", "", "  ", "b"]) == ["a", "b"]


def test_lead_is_configurable_without_touching_the_router():
    assert group_for_prosody(["a", "b", "c"], lead=2) == ["a", "b", "c"]
    assert group_for_prosody(["a", "b", "c", "d"], lead=2) == ["a", "b", "c d"]
    assert LEAD_SEGMENTS == 1


# -- per-utterance delivery -------------------------------------------------


def test_the_same_utterance_always_sounds_the_same():
    """Deterministic, so a recording can be explained after the fact."""
    u = uuid4()
    assert Prosody.for_utterance(u) == Prosody.for_utterance(u)


def test_different_utterances_are_delivered_differently():
    made = {Prosody.for_utterance(uuid4()) for _ in range(50)}
    assert len(made) > 45, "delivery must actually vary between utterances"


def test_delivery_stays_inside_the_measured_safe_bands():
    """The one that matters. NOISE_W_BAND is not a style preference.

    Above noise_w_scale 1.0 Piper's duration variance climbs sharply, and at
    1.6 it degenerates: a 3.7-second sentence measured 20.5 seconds. Live,
    that is a host who stops making sense for twenty seconds. If this test
    fails because someone widened the band, re-measure before believing it.
    """
    assert NOISE_W_BAND[1] <= 1.0, "measured degeneracy above 1.0"
    for _ in range(200):
        p = Prosody.for_utterance(uuid4())
        assert LENGTH_SCALE_BAND[0] <= p.length_scale <= LENGTH_SCALE_BAND[1]
        assert NOISE_W_BAND[0] <= p.noise_w_scale <= NOISE_W_BAND[1]


async def test_every_segment_of_one_utterance_shares_a_delivery(tmp_path):
    """Otherwise the speaking rate would change mid-utterance, which is a
    worse artefact than not varying at all."""
    seen: list[Prosody | None] = []

    class RecordingTTS:
        async def synthesize(self, text, voice_id, out_path, *, prosody=None):
            seen.append(prosody)
            out_path.parent.mkdir(parents=True, exist_ok=True)
            out_path.write_bytes(b"x")
            return TTSResult(path=out_path, duration_ms=10, first_audio_ms=1)

    r = AudioRouter("SESSION_001", RecordingTTS(), tmp_path, sink=FakeSink())
    r._running = True
    await r._speak(audio_request(segments=["One.", "Two.", "Three."]))

    assert len(seen) == 2, "lead alone, then the tail as one span"
    assert seen[0] is not None, "the router must supply a delivery"
    assert seen[0] == seen[1], "rate must not change mid-utterance"


async def test_two_utterances_get_different_deliveries(tmp_path):
    seen: list[Prosody | None] = []

    class RecordingTTS:
        async def synthesize(self, text, voice_id, out_path, *, prosody=None):
            seen.append(prosody)
            out_path.parent.mkdir(parents=True, exist_ok=True)
            out_path.write_bytes(b"x")
            return TTSResult(path=out_path, duration_ms=10, first_audio_ms=1)

    r = AudioRouter("SESSION_001", RecordingTTS(), tmp_path, sink=FakeSink())
    r._running = True
    await r._speak(audio_request(segments=["A."]))
    await r._speak(audio_request(segments=["B."]))
    assert seen[0] != seen[1]


# -- combining with the operator's setting ----------------------------------


class _StubConfig:
    """Stands in for piper.SynthesisConfig so this needs no voice model."""

    def __init__(self, length_scale=None, noise_w_scale=None):
        self.length_scale = length_scale
        self.noise_w_scale = noise_w_scale


def test_operator_rate_and_utterance_jitter_multiply():
    """`length_scale` on the provider is a deliberate setting -- "this host
    talks faster than default" -- so the per-utterance variation scales it
    rather than discarding it."""
    from platform_.tts.piper import PiperTTS

    tts = PiperTTS(length_scale=1.2)
    cfg = tts._config(Prosody(length_scale=0.95, noise_w_scale=0.8), _StubConfig)
    assert cfg.length_scale == pytest.approx(1.2 * 0.95)
    assert cfg.noise_w_scale == 0.8


def test_no_prosody_preserves_the_previous_behaviour_exactly():
    from platform_.tts.piper import PiperTTS

    assert PiperTTS()._config(None, _StubConfig) is None
    cfg = PiperTTS(length_scale=1.1)._config(None, _StubConfig)
    assert cfg.length_scale == 1.1 and cfg.noise_w_scale is None

# -- rate carries meaning, and in the right direction -----------------------


def test_a_faster_pace_produces_shorter_audio_not_longer():
    """The regression the whole suite missed.

    `pace` is a speaking-RATE multiplier; Piper's length_scale scales phoneme
    DURATION, so the two are inverses. Multiplying pace straight into
    length_scale made an excited host speak more slowly and an uncertain one
    speed up -- every test still passed, because nothing asserted a direction.
    """
    uid = uuid4()
    fast = Prosody.for_utterance(uid, pace=1.08)
    normal = Prosody.for_utterance(uid, pace=1.00)
    slow = Prosody.for_utterance(uid, pace=0.92)
    assert fast.length_scale < normal.length_scale < slow.length_scale


def test_neutral_pace_leaves_the_voice_at_its_natural_rate():
    uid = uuid4()
    p = Prosody.for_utterance(uid, pace=1.0)
    assert 0.98 <= p.length_scale <= 1.02


def test_the_emotional_signal_is_wider_than_the_incidental_jitter():
    """A signal narrower than its own noise floor is not a signal.

    The jitter used to be +-6% against an emotional range of +-5.5%, so
    rendering the same sentence quiet versus excited produced distributions
    overlapping across 62% of their span -- and a listener hears one
    utterance, not a distribution.
    """
    jitter_width = LENGTH_SCALE_BAND[1] - LENGTH_SCALE_BAND[0]
    uid = uuid4()
    signal_width = abs(
        Prosody.for_utterance(uid, PACE_MAX).length_scale
        - Prosody.for_utterance(uid, PACE_MIN).length_scale
    )
    assert signal_width > 3 * jitter_width, (
        f"signal {signal_width:.4f} vs jitter {jitter_width:.4f}: "
        "the incidental variation is swamping the intentional one"
    )
