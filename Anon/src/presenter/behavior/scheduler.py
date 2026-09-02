"""Duration-aware state scheduling and behavioural memory.

Two things the engine was missing, and they are related.

## Why a semi-Markov model rather than per-frame sampling

The state was previously set from outside and never changed on its own, so an
unattended presenter sat in one state indefinitely. The obvious repair - sample
a transition every frame - is worse, and worth being explicit about why: with a
per-frame transition probability the *dwell time in every state is geometric*.
Geometric means the most likely duration is the shortest one, and it means a
state that has lasted two minutes is exactly as likely to end in the next
second as one that started a second ago. Neither is true of people. Attention
has persistence and a characteristic length.

A semi-Markov model separates the two questions. **How long do I stay** is drawn
once, on entry, from a distribution with a real mode. **Where do I go next** is
a weighted choice made only when that time expires. Nothing is re-rolled per
frame, so a state cannot end early by bad luck, and the duration distribution is
whatever we say it is instead of an artefact of the sampling rate.

## Hysteresis and cooldowns

A state that has just ended is barred from re-entry for a cooldown period. This
is what stops the NEUTRAL -> HAPPY -> NEUTRAL -> HAPPY flicker the brief calls
out: not by damping the output, which would just make the flicker slower, but by
making the transition unavailable.

## Behavioural memory

The repetition the brief cares about is not a single repeated action - it is a
repeated *sequence*. `BehaviorMemory` keeps a rolling signature of recent
voluntary events and can report both how often an action has occurred lately
(used to suppress it) and whether short n-grams are recurring (used to fail a
test). Involuntary events are excluded: blinks and microsaccades recur by
definition and would swamp any n-gram statistic, which is a mistake this project
has already made once.
"""

from __future__ import annotations

from collections import Counter, deque
from dataclasses import dataclass, field

from .state import BehaviorState

__all__ = ["StateSpec", "IDLE_STATE_GRAPH", "StateScheduler", "BehaviorMemory"]


@dataclass(frozen=True)
class StateSpec:
    """How long a state lasts and where it can go next."""

    min_duration: float
    median_duration: float
    max_duration: float
    # Unnormalised weights over successor states.
    successors: dict[str, float]
    # Seconds before this state may be entered again.
    cooldown: float = 0.0
    shape: float = 0.5


# The silent-presenter graph. Speech states exist in `BehaviorState` and are
# reachable only when something external drives them; nothing here enters them
# autonomously, because a silent presenter who starts "speaking" is a bug.
#
# Durations come from what a seated presenter actually does: attentive stretches
# are long, reading a display is medium, thinking is short, and the positive
# state is brief because a smile that outlasts its cause reads as a mask.
IDLE_STATE_GRAPH: dict[str, StateSpec] = {
    "IDLE_ATTENTIVE": StateSpec(
        min_duration=12.0, median_duration=38.0, max_duration=150.0,
        successors={"IDLE_RELAXED": 2.2, "READING": 1.6, "FOCUSED": 1.0,
                    "THINKING": 0.7, "MILD_POSITIVE": 0.45},
    ),
    "IDLE_RELAXED": StateSpec(
        min_duration=14.0, median_duration=46.0, max_duration=180.0,
        successors={"IDLE_ATTENTIVE": 3.0, "READING": 1.1, "THINKING": 0.6,
                    "MILD_POSITIVE": 0.5},
    ),
    "READING": StateSpec(
        min_duration=4.0, median_duration=11.0, max_duration=34.0,
        successors={"IDLE_ATTENTIVE": 2.6, "FOCUSED": 1.4, "IDLE_RELAXED": 1.0,
                    "MILD_POSITIVE": 0.4},
        cooldown=20.0,
    ),
    "FOCUSED": StateSpec(
        min_duration=6.0, median_duration=17.0, max_duration=52.0,
        successors={"IDLE_ATTENTIVE": 2.4, "READING": 1.5, "IDLE_RELAXED": 1.2},
        cooldown=25.0,
    ),
    "THINKING": StateSpec(
        min_duration=2.2, median_duration=5.0, max_duration=13.0,
        successors={"IDLE_ATTENTIVE": 3.2, "READING": 1.0, "IDLE_RELAXED": 0.8},
        cooldown=45.0,
    ),
    "MILD_POSITIVE": StateSpec(
        min_duration=1.8, median_duration=4.2, max_duration=9.0,
        successors={"IDLE_ATTENTIVE": 3.0, "IDLE_RELAXED": 1.6},
        cooldown=70.0,
    ),
}


class StateScheduler:
    """Semi-Markov state machine over the idle graph."""

    def __init__(self, graph=None, start: str = "IDLE_ATTENTIVE") -> None:
        self.graph = graph or IDLE_STATE_GRAPH
        self.state = start
        self.entered_at = 0.0
        self.leave_at = 0.0
        self._last_exit: dict[str, float] = {}
        self.transitions = 0
        self._pending: str | None = None

    def _sample_duration(self, name: str, rng) -> float:
        spec = self.graph[name]
        return rng.lognormal_interval(
            median=spec.median_duration, shape=spec.shape,
            low=spec.min_duration, high=spec.max_duration,
        )

    def start(self, now: float, rng) -> None:
        self.entered_at = now
        self.leave_at = now + self._sample_duration(self.state, rng)

    def _choose_successor(self, now: float, rng) -> str:
        spec = self.graph[self.state]
        options = {}
        for name, w in spec.successors.items():
            if name not in self.graph:
                continue
            cd = self.graph[name].cooldown
            since = now - self._last_exit.get(name, -1e9)
            if since < cd:
                continue                      # still in hysteresis, unavailable
            options[name] = w
        if not options:
            return "IDLE_ATTENTIVE"
        total = sum(options.values())
        r = rng.uniform(0.0, total)
        acc = 0.0
        for name, w in options.items():
            acc += w
            if r <= acc:
                return name
        return next(iter(options))

    def adopt(self, name: str, now: float, rng) -> None:
        """Take an externally imposed state.

        When something outside drives the presenter - the content pipeline
        putting him into SPEAKING - that caller owns the state and the
        scheduler must not fight it. If the state is one this graph knows, the
        dwell clock restarts on it; if it is not (every speech state), the
        scheduler suspends until an idle state returns. Without this the
        successor lookup raised KeyError the moment anything set a state the
        idle graph had never heard of.
        """
        self.state = name
        self.entered_at = now
        if name in self.graph:
            self.leave_at = now + self._sample_duration(name, rng)
        else:
            self.leave_at = float("inf")

    def update(self, now: float, rng) -> str | None:
        """Advance. Returns the new state name if a transition happened."""
        if self.state not in self.graph:
            return None                    # externally driven; not ours to end
        if self.leave_at <= 0.0:
            self.start(now, rng)
            return None
        if now < self.leave_at:
            return None

        nxt = self._choose_successor(now, rng)
        self._last_exit[self.state] = now
        self.state = nxt
        self.entered_at = now
        self.leave_at = now + self._sample_duration(nxt, rng)
        self.transitions += 1
        return nxt

    @property
    def time_in_state(self) -> float:
        return self.leave_at - self.entered_at

    def as_behavior_state(self) -> BehaviorState:
        return BehaviorState(self.state)


class BehaviorMemory:
    """Rolling record of recent voluntary behaviour.

    Used two ways: live, to make a recently-used action less likely; and after
    the fact, to detect that a sequence has started recurring.
    """

    # Blinks and microsaccades are involuntary and recur by construction.
    # Including them in the n-gram statistic buries every real repetition under
    # a wall of `blink, blink, blink` - a mistake made once already in this
    # project's loop detector.
    INVOLUNTARY = {"blink", "microsaccade", "breath", "drift"}

    def __init__(self, window: int = 40) -> None:
        self.events: deque[tuple[float, str]] = deque(maxlen=window)
        self.last_at: dict[str, float] = {}
        self.counts: Counter[str] = Counter()

    def record(self, now: float, kind: str, detail: str = "") -> None:
        key = f"{kind}:{detail}" if detail else kind
        self.last_at[kind] = now
        self.counts[kind] += 1
        if kind.split(".")[0] not in self.INVOLUNTARY:
            self.events.append((now, key))

    def since(self, kind: str, now: float) -> float:
        return now - self.last_at.get(kind, -1e9)

    def recent_count(self, kind: str, now: float, window: float) -> int:
        return sum(1 for t, k in self.events
                   if k.startswith(kind) and now - t <= window)

    def cooldown_factor(self, kind: str, now: float, window: float = 45.0,
                        decay: float = 0.5) -> float:
        """Multiplier in (0, 1] that shrinks the more often `kind` fired lately."""
        return decay ** self.recent_count(kind, now, window)

    def repeated_ngrams(self, n: int = 3, min_repeats: int = 3):
        """n-grams of voluntary actions that recur. Evidence of a loop."""
        seq = [k.split(" ")[0] for _, k in self.events]
        if len(seq) < n * min_repeats:
            return []
        grams = Counter(tuple(seq[i:i + n]) for i in range(len(seq) - n + 1))
        return [(g, c) for g, c in grams.items() if c >= min_repeats]
