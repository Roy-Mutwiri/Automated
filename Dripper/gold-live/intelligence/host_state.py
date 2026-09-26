"""How the host currently feels, derived from what has actually happened.

The point of this module is the word *derived*. Sprinkling random hesitation
and excitement into speech produces the "AI trying to sound human" effect --
the giveaway is not that the variation exists but that it does not correlate
with anything. A listener notices immediately that the host got excited about
nothing.

So there is no randomness here at all. Three dimensions, each moved only by a
signal the system already computes, each decaying back to baseline on a clock:

    arousal       how energised. Rises with market event severity, falls
                  slowly. Drives pace and the gap between chunks.
    surprise      spikes on a genuine *change*, decays fast. Drives the
                  pause before speaking and whether a reaction sound fires.
    uncertainty   rises when the data cannot be trusted. Drives a slower,
                  more hedged delivery.

Three, not nine, because Piper can express roughly two things -- rate and
silence -- and a dimension that cannot reach the listener is a dimension that
only exists in a log. `tension`, `amusement` and `confidence` are deliberately
absent: they would be indistinguishable in the rendered audio from the three
above. Add one when there is a renderer that can express it.

Determinism: every method takes `now` explicitly, like the Director does.
Given the same events at the same times, the state is identical, so a
recording can be reproduced and a test can assert on it.

Novelty matters more than magnitude. A break of structure is startling once;
the fourth one in two minutes is the market being busy, not a surprise. Only
a change of *kind* or a jump in severity produces surprise, which is what
stops reaction sounds from firing on every tick.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import datetime

from shared.contracts import MarketConfidence, MarketEvent, MarketEventKind

#: Seconds for a dimension to fall to ~37% of its excess over baseline.
#: Arousal outlives a single utterance on purpose -- a host who was excited
#: ten seconds ago is still a bit lively. Surprise does not: it is a reaction
#: to a moment, and a host still audibly startled half a minute later sounds
#: unhinged rather than alive.
AROUSAL_HALFLIFE_S = 45.0
SURPRISE_HALFLIFE_S = 8.0
UNCERTAINTY_HALFLIFE_S = 30.0

#: Where each dimension sits when nothing is happening. Not zero for arousal:
#: a live host at rest still has some energy, and a floor of zero would make
#: quiet stretches sound depressed.
AROUSAL_BASELINE = 0.35

#: severity 1-5 -> how much arousal a single event adds.
SEVERITY_AROUSAL = {1: 0.05, 2: 0.12, 3: 0.22, 4: 0.38, 5: 0.55}

#: Events that are inherently startling when they are new, versus events that
#: are simply information. A scheduled session open is never a surprise no
#: matter how important it is.
STARTLING_KINDS = frozenset(
    {
        MarketEventKind.BOS,
        MarketEventKind.CHOCH,
        MarketEventKind.LIQUIDITY_SWEEP,
        MarketEventKind.VOL_EXPANSION,
        MarketEventKind.LEVEL_BREAK,
        MarketEventKind.NEWS,
    }
)

#: Below this severity nothing is startling regardless of kind.
SURPRISE_MIN_SEVERITY = 3


def _decay(value: float, baseline: float, elapsed_s: float, halflife_s: float) -> float:
    """Exponential return to baseline. Monotonic and frame-rate independent,
    so a tick every 250ms and a tick every 4s converge to the same place."""
    if elapsed_s <= 0:
        return value
    keep = math.exp(-elapsed_s / halflife_s)
    return baseline + (value - baseline) * keep


@dataclass
class HostState:
    """One per session. Holds no reference to any other session."""

    arousal: float = AROUSAL_BASELINE
    surprise: float = 0.0
    uncertainty: float = 0.0

    #: What the last event was, so novelty can be judged. Not for display.
    last_event_kind: MarketEventKind | None = None
    last_event_severity: int = 0
    updated_at: datetime | None = None

    #: Events seen, for diagnostics and for tests that assert nothing fired.
    surprises: int = 0
    _history: list[str] = field(default_factory=list, repr=False)

    # -- time -------------------------------------------------------------

    def decay_to(self, now: datetime) -> None:
        """Bring every dimension forward to `now`. Idempotent for the same
        timestamp, which matters because tick() and on_event() may both run
        within one loop iteration."""
        if self.updated_at is None:
            self.updated_at = now
            return
        elapsed = (now - self.updated_at).total_seconds()
        if elapsed <= 0:
            return
        self.arousal = _decay(self.arousal, AROUSAL_BASELINE, elapsed, AROUSAL_HALFLIFE_S)
        self.surprise = _decay(self.surprise, 0.0, elapsed, SURPRISE_HALFLIFE_S)
        self.uncertainty = _decay(
            self.uncertainty, 0.0, elapsed, UNCERTAINTY_HALFLIFE_S
        )
        self.updated_at = now

    # -- inputs -----------------------------------------------------------

    def observe_event(self, event: MarketEvent, now: datetime) -> bool:
        """Fold a market event in. Returns True if it was genuinely startling.

        The caller uses the return value to decide whether a reaction sound is
        warranted, which is why novelty is judged here rather than at the
        point of speech: this is the only place that knows what came before.
        """
        self.decay_to(now)

        severity = max(1, min(5, int(event.severity)))
        self.arousal = min(1.0, self.arousal + SEVERITY_AROUSAL[severity])

        startling = self._is_startling(event, severity)
        if startling:
            # Scaled by severity so a 5 lands harder than a 3, and capped so
            # repeated events cannot stack into a permanent state of shock.
            self.surprise = min(1.0, self.surprise + 0.35 + 0.13 * (severity - 3))
            self.surprises += 1
            self._history.append(f"{event.kind.value}:{severity}")

        self.last_event_kind = event.kind
        self.last_event_severity = severity
        return startling

    def _is_startling(self, event: MarketEvent, severity: int) -> bool:
        if event.kind not in STARTLING_KINDS:
            return False
        if severity < SURPRISE_MIN_SEVERITY:
            return False
        # The novelty rule. The same kind of event repeating is the market
        # being busy; a host who gasps at each one is a host nobody believes.
        if event.kind == self.last_event_kind and severity <= self.last_event_severity:
            return False
        return True

    def observe_market(self, confidence: MarketConfidence, now: datetime) -> None:
        """Data quality drives uncertainty. The host genuinely does not know
        what the price is when the feed is stale, and should sound like it."""
        self.decay_to(now)
        target = {
            MarketConfidence.LIVE: 0.0,
        }.get(confidence, 0.6)
        # Rise quickly, fall by decay. Discovering the feed is stale is a step
        # change; regaining trust in it is gradual.
        self.uncertainty = max(self.uncertainty, target)

    # -- output -----------------------------------------------------------

    @property
    def energy(self) -> float:
        """A single 0-1 summary, for logging and for the dashboard. Surprise
        counts toward energy because a startled host is briefly livelier."""
        return max(0.0, min(1.0, 0.75 * self.arousal + 0.25 * self.surprise))

    def snapshot(self) -> dict:
        return {
            "arousal": round(self.arousal, 3),
            "surprise": round(self.surprise, 3),
            "uncertainty": round(self.uncertainty, 3),
            "energy": round(self.energy, 3),
            "surprises": self.surprises,
            "last_event": self.last_event_kind.value if self.last_event_kind else None,
        }
