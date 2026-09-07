"""TTS provider interface.

The conversation engine outputs text. This turns text into audio. It knows
nothing about Gold, markets, or why the words were chosen -- keeping that
boundary is what makes the vendor swappable, and you will swap it.

Choose a vendor on measured time-to-first-audio, not on MOS scores.
"""

from __future__ import annotations

import random
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from uuid import UUID

from shared.delivery import DeliveryCapabilities, NULL_CAPABILITIES


@dataclass(slots=True)
class TTSResult:
    path: Path | None
    duration_ms: int
    #: Time to first audible sample. This is the number that matters for the
    #: latency budget -- total synthesis time does not, because we stream.
    first_audio_ms: int


#: Incidental per-utterance variation in rate, to stop hundreds of utterances
#: being metronomically identical.
#:
#: DELIBERATELY TINY, and it used to be +-6%. At that width it was wider than
#: the entire range the host's emotional state could produce (0.952-1.060),
#: so the intentional signal was buried in the incidental one: rendering the
#: same sentence quiet versus excited gave distributions overlapping across
#: 62% of their combined span. A listener hears one utterance, not a
#: distribution, so a signal that only separates on average does not separate
#: at all. Rate now carries meaning; rhythm carries the variety, via
#: noise_w_scale below.
LENGTH_SCALE_BAND = (0.985, 1.015)

#: Phoneme-duration variability. The single most effective knob for making
#: Piper sound less metronomic, and the most dangerous one.
#:
#: MEASURED, do not widen without re-measuring. On a sentence that renders in
#: ~3.7s (n=6 per point):
#:
#:      noise_w_scale   mean     max      sd
#:      (model default) 3.661s   3.738s   0.058
#:      0.7             3.493s   3.611s   0.103
#:      1.0             3.725s   3.913s   0.134
#:      1.1             3.721s   4.110s   0.268
#:      1.3             3.961s   4.203s   0.331
#:      1.6             7.693s  20.515s   6.412   <-- degenerate
#:
#: Above 1.0 the variance climbs, and by 1.6 the model babbles: a 3.7-second
#: sentence rendered as 20.5 seconds of audio. Live, that is a host who stops
#: making sense for twenty seconds. The ceiling stays at 1.0.
NOISE_W_BAND = (0.75, 1.00)

#: Hard limit on the final speaking rate, after the per-utterance jitter and
#: the DeliveryPlan's pace have been multiplied together. Both inputs are
#: bounded already, so this is the backstop: whatever combination of settings
#: is in play, the rate that reaches the synthesiser stays inside the range
#: that was measured to behave monotonically (0.9 and 1.25 both render
#: cleanly; the failure mode above that is smeared consonants, not silence,
#: so it would ship unnoticed).
LENGTH_SCALE_ABSOLUTE = (0.80, 1.25)


@dataclass(frozen=True, slots=True)
class Prosody:
    """How one utterance is delivered, as distinct from what it says.

    Piper renders every call with identical rhythm and rate unless told
    otherwise, and identical rhythm across hundreds of utterances is one of
    the clearest tells that nobody is actually talking. This varies the
    delivery a little per utterance.

    Per UTTERANCE, not per segment: every segment of one utterance must share
    a rate, or the speed would visibly change mid-sentence, which is worse
    than no variation at all. Derived from the utterance id so it is
    deterministic -- the same utterance always sounds the same way, which
    keeps tests reproducible and makes a recording explainable after the fact.

    `None` on both fields means "use the voice's own defaults", which is what
    a provider that does not support prosody control should do anyway.
    """

    length_scale: float | None = None
    noise_w_scale: float | None = None

    @classmethod
    def for_utterance(cls, utterance_id: UUID, pace: float = 1.0) -> Prosody:
        """Jitter for this utterance, scaled by what the delivery plan asked for.

        `pace` is the deliberate part -- faster because something happened --
        and the jitter is the incidental part that stops every utterance from
        being metronomically identical. They multiply, then the product is
        clamped, so no combination of a plan and a seed can drive the rate
        outside the measured range.
        """
        rng = random.Random(utterance_id.int)
        # INVERTED, deliberately. `pace` is a speaking-rate multiplier -- 1.05
        # means "speaks 5% faster" -- but Piper's length_scale scales phoneme
        # DURATION, so larger is slower (measured: 0.85 -> 3206ms, 1.15 ->
        # 3965ms on the same sentence). Multiplying pace straight in made an
        # excited host speak more slowly and an uncertain one speed up, which
        # is exactly backwards and is the kind of error that is invisible in
        # a passing test suite.
        length = rng.uniform(*LENGTH_SCALE_BAND) / pace
        low, high = LENGTH_SCALE_ABSOLUTE
        return cls(
            length_scale=round(max(low, min(high, length)), 4),
            noise_w_scale=round(rng.uniform(*NOISE_W_BAND), 4),
        )


class TTSProvider(ABC):
    #: What this backend can express. The router degrades a DeliveryPlan to
    #: this before rendering, so a provider that cannot vary anything simply
    #: receives a neutral plan rather than silently ignoring an expressive one.
    capabilities: DeliveryCapabilities = NULL_CAPABILITIES

    @abstractmethod
    async def synthesize(
        self,
        text: str,
        voice_id: str,
        out_path: Path,
        *,
        prosody: Prosody | None = None,
    ) -> TTSResult:
        """Render one segment.

        A segment is not necessarily one sentence -- see
        `platform_.audio.router.group_for_prosody`. Providers that cannot vary
        delivery ignore `prosody`.
        """
