"""How the host state becomes a delivery plan.

The words come from the model. The delivery comes from `HostState`, which
comes from events that actually happened. Keeping them apart is what makes the
expression honest: the host speeds up because a level broke, not because a
random number said to.

    MarketEvent -> HostState -> DeliveryPlan -> Prosody -> Piper
                                     |
                                     +-> reaction unit, pauses, chunk gap

The plan type itself lives in `shared.delivery`, because `platform_` has to
read it and must not import from `intelligence`. This module is the policy:
what state produces what delivery, and which events are worth reacting to out
loud.
"""

from __future__ import annotations

from datetime import datetime, timedelta

from intelligence.host_state import AROUSAL_BASELINE, HostState
from shared.contracts import MarketEventKind, Priority
from shared.delivery import GAP_DEFAULT_MS, GAP_MIN_MS, DeliveryPlan

#: Never two reactions closer together than this. The novelty rule in
#: HostState already suppresses most repeats; this is the backstop that makes
#: "cannot create runaway audio" a property rather than a hope.
REACTION_COOLDOWN_S = 25.0

#: Below this surprise level nothing is startling enough to react to out loud.
REACTION_SURPRISE_FLOOR = 0.45


# ---------------------------------------------------------------------------
# Reactions
# ---------------------------------------------------------------------------

#: Deterministic, event-driven, and deliberately tiny. Chosen by what happened,
#: never sampled: a host who says "whoa" at random is worse than one who never
#: says it. Keyed by the event kind that startled the host.
#: Every entry must be within REACTION_MAX_CHARS or _clean_reaction will
#: discard it -- the reaction would simply never be spoken, silently. There
#: is a test asserting this.
REACTIONS: dict[MarketEventKind, str] = {
    MarketEventKind.BOS: "Wait.",
    MarketEventKind.CHOCH: "Hold on.",
    MarketEventKind.LIQUIDITY_SWEEP: "Oh.",
    MarketEventKind.VOL_EXPANSION: "Whoa.",
    MarketEventKind.LEVEL_BREAK: "There it is.",
    MarketEventKind.NEWS: "Okay.",
}

#: Used when surprise is high enough that the specific kind matters less than
#: the size of it.
BIG_SURPRISE_REACTION = "Whoa."
BIG_SURPRISE_LEVEL = 0.8


# ---------------------------------------------------------------------------
# Planning
# ---------------------------------------------------------------------------


class DeliveryPlanner:
    """Turns host state into a plan. One per session.

    Stateful only for the reaction cooldown, which cannot live in a pure
    function and must not live in HostState -- HostState describes how the
    host feels, not what it has recently said out loud.
    """

    def __init__(self, cooldown_s: float = REACTION_COOLDOWN_S) -> None:
        self.cooldown_s = cooldown_s
        self.last_reaction_at: datetime | None = None

    def plan(
        self,
        state: HostState,
        now: datetime,
        *,
        priority: Priority = Priority.MEDIUM,
        startling_event: MarketEventKind | None = None,
    ) -> DeliveryPlan:
        """Ordinary speech is the default and the common case.

        Only a genuinely startling event produces a reaction or a long pause.
        With no events at all this returns something very close to neutral,
        which is the behaviour that keeps the host from sounding permanently
        over-caffeinated.
        """
        energy = state.energy

        # Arousal above resting speeds the host up; uncertainty slows them
        # down. Anchored so a resting host speaks at exactly the voice's
        # natural rate -- "normal" has to be normal, and the previous formula
        # made an idle host permanently 1.8% slow for no reason at all.
        #
        # The span is wider than it was because it has to be: the incidental
        # jitter in Prosody is +-1.5%, and a signal narrower than its own
        # noise floor is not a signal.
        lift = (state.arousal - AROUSAL_BASELINE) / max(1e-9, 1.0 - AROUSAL_BASELINE)
        lift = max(0.0, min(1.0, lift))
        pace = 1.0 + 0.10 * lift - 0.06 * state.uncertainty
        # A surprise lands *before* the words: the beat of silence is what
        # sells it. Explaining something unclear earns a shorter beat.
        pause_before = int(260 * state.surprise + 120 * state.uncertainty)
        # Let a big moment breathe afterwards rather than running straight on.
        pause_after = int(180 * state.surprise)
        # An energised host leaves less air between clauses. Anchored to the
        # same resting point, so an idle host gets exactly GAP_DEFAULT_MS.
        #
        # This carries most of what arousal can express, because rate cannot.
        # MEASURED: across the whole reachable arousal range the rendered
        # duration of a line moves by 2-5%, and Piper's own run-to-run
        # duration spread on identical input is ~3.5% -- so the rate signal
        # sits inside the engine's noise floor and a listener cannot recover
        # it. Silence is the one expressive channel on Piper that is exact,
        # so arousal spends its budget there: the full band, not part of it.
        gap = int(GAP_DEFAULT_MS - (GAP_DEFAULT_MS - GAP_MIN_MS) * lift)

        reaction = self._reaction_for(state, now, startling_event)
        if reaction is not None:
            self.last_reaction_at = now

        return DeliveryPlan(
            pace=pace,
            pause_before_ms=pause_before,
            pause_after_ms=pause_after,
            gap_ms=gap,
            reaction=reaction,
            energy=energy,
            # Carried for a future backend; Piper strips these.
            pitch_bias=0.25 * state.surprise,
        )

    def _reaction_for(
        self,
        state: HostState,
        now: datetime,
        startling_event: MarketEventKind | None,
    ) -> str | None:
        if startling_event is None:
            return None
        if state.surprise < REACTION_SURPRISE_FLOOR:
            return None
        if self.last_reaction_at is not None:
            if now - self.last_reaction_at < timedelta(seconds=self.cooldown_s):
                return None
        if state.surprise >= BIG_SURPRISE_LEVEL:
            return BIG_SURPRISE_REACTION
        return REACTIONS.get(startling_event)
