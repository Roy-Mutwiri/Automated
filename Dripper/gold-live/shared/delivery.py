"""The vocabulary of delivery: what a plan is, and what a backend can express.

This lives in shared/ for the same reason shared/contracts.py does -- it is
spoken by both halves. `intelligence` decides how the host should sound;
`platform_` renders it. Neither may import the other, so the types they both
need live here.

It is deliberately NOT in shared/contracts.py. Those eight contracts are the
serialised interface between processes and carry a schema version and a
change process; this is an in-process value type between two layers of one
session. Putting it there would mean a schema bump for what is an
implementation detail of how one process talks to itself.

## Why the schema is this short

Every field is one Piper can actually render. `emphasis`, `pitch_bias` and
`laugh` are carried but flagged unsupported, because Piper exposes rate and
noise scales and nothing else -- no per-word emphasis, no pitch target, no
non-speech vocalisation. A plan goes through `renderable_with()` before it
reaches a backend, which strips whatever that backend cannot do, so an
expressive plan is never silently assumed to have been honoured.

That is the whole point of the capability layer. When a streaming engine with
real expressive control replaces Piper, its capability table turns those
fields on and `DeliveryPlan` does not change.

## Bounds are not style choices

Every numeric bound below was measured. Widening one without re-measuring is
how this ends up sounding worse than the flat version it replaced. See
`scripts/prosody_probe.py`, and `platform_.tts.base.NOISE_W_BAND` for the
range where Piper degenerates outright.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

# ---------------------------------------------------------------------------
# Bounds
# ---------------------------------------------------------------------------

#: Speaking-rate multiplier. Narrower than it looks: below ~0.92 Piper starts
#: to sound like a slowed recording rather than someone speaking carefully,
#: and above ~1.08 consonants smear. The default synthesis rate is 1.0.
PACE_MIN, PACE_MAX = 0.92, 1.08

#: Silence around an utterance, in milliseconds. The ceiling exists because a
#: live host who leaves a full second before answering sounds like a dropout,
#: and the audience cannot see that it was deliberate.
PAUSE_MAX_MS = 600

#: Below this, a pause is not heard as a pause. A gap shorter than roughly
#: 50-80ms reads as the closure of a stop consonant rather than as silence, so
#: anything under the floor is rounded away to nothing. Observed in a real run:
#: mild uncertainty produced `before=10ms`, which cost a schema field and a
#: code path to render something no listener could perceive.
PAUSE_MIN_AUDIBLE_MS = 60

#: The gap between chunks of one utterance. Baseline is what a natural
#: sentence boundary sounds like inside continuous speech -- NOT zero, which
#: sounds rushed and glued. Measured baseline before prefetching was ~562ms,
#: which is why the host sounded assembled.
GAP_MIN_MS, GAP_DEFAULT_MS, GAP_MAX_MS = 40, 90, 320

#: A reaction is a word. Anything longer is a sentence pretending to be one,
#: and a long "reaction" is how a reaction unit turns into runaway audio.
REACTION_MAX_CHARS = 12


# ---------------------------------------------------------------------------
# Capabilities
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class DeliveryCapabilities:
    """What a TTS backend can actually express.

    The renderer degrades a plan to this before using it, so an expressive
    plan is never silently assumed to have been honoured.
    """

    pace: bool = False
    pauses: bool = False
    reactions: bool = False
    emphasis: bool = False
    pitch: bool = False
    laughter: bool = False


#: Piper: rate and silence, nothing else. Pauses and reactions are ours to
#: implement (we control the gaps between renders); emphasis, pitch targets
#: and laughter would need a different engine.
PIPER_CAPABILITIES = DeliveryCapabilities(
    pace=True, pauses=True, reactions=True, emphasis=False, pitch=False, laughter=False
)

#: What a mock or a file-writing stand-in can do: nothing expressive. Used so
#: tests and the dry run take the same code path without pretending.
NULL_CAPABILITIES = DeliveryCapabilities()


# ---------------------------------------------------------------------------
# The plan
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class DeliveryPlan:
    """How to say it. Immutable: a plan is decided once per utterance."""

    pace: float = 1.0
    pause_before_ms: int = 0
    pause_after_ms: int = 0
    gap_ms: int = GAP_DEFAULT_MS
    #: A short sound made before the sentence, when something warranted it.
    reaction: str | None = None
    #: 0-1, for logging and for a future backend that can act on it.
    energy: float = 0.0

    # -- carried, not currently renderable -------------------------------
    emphasis: tuple[str, ...] = ()
    pitch_bias: float = 0.0
    laugh: bool = False

    @classmethod
    def neutral(cls) -> DeliveryPlan:
        """Ordinary speech. What every utterance gets when nothing happened,
        which is most of them."""
        return cls()

    def renderable_with(self, caps: DeliveryCapabilities) -> DeliveryPlan:
        """Strip everything this backend cannot express, and clamp the rest.

        Clamping happens here rather than at construction so that a plan built
        by some future LLM-authored path cannot hand the renderer a pace of 4.0
        or a nine-second pause. Whatever produced the plan, only bounded values
        reach the synthesiser.
        """
        return replace(
            self,
            pace=_clamp(self.pace, PACE_MIN, PACE_MAX) if caps.pace else 1.0,
            pause_before_ms=_clamp_ms(self.pause_before_ms) if caps.pauses else 0,
            pause_after_ms=_clamp_ms(self.pause_after_ms) if caps.pauses else 0,
            gap_ms=(
                int(_clamp(self.gap_ms, GAP_MIN_MS, GAP_MAX_MS))
                if caps.pauses
                else GAP_DEFAULT_MS
            ),
            reaction=_clean_reaction(self.reaction) if caps.reactions else None,
            emphasis=self.emphasis if caps.emphasis else (),
            pitch_bias=self.pitch_bias if caps.pitch else 0.0,
            laugh=self.laugh if caps.laughter else False,
        )

    def describe(self) -> str:
        bits = [f"pace={self.pace:.2f}", f"gap={self.gap_ms}ms"]
        if self.pause_before_ms:
            bits.append(f"before={self.pause_before_ms}ms")
        if self.pause_after_ms:
            bits.append(f"after={self.pause_after_ms}ms")
        if self.reaction:
            bits.append(f"reaction={self.reaction!r}")
        return " ".join(bits)


def _clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def _clamp_ms(value: int) -> int:
    bounded = int(max(0, min(PAUSE_MAX_MS, value)))
    return 0 if bounded < PAUSE_MIN_AUDIBLE_MS else bounded


def _clean_reaction(text: str | None) -> str | None:
    """A reaction that is not short is not a reaction.

    This is the guard that stops a reaction unit becoming runaway audio: no
    matter what produced the string, what reaches the synthesiser is at most a
    few characters.
    """
    if not text:
        return None
    stripped = text.strip()
    if not stripped or len(stripped) > REACTION_MAX_CHARS:
        return None
    return stripped
