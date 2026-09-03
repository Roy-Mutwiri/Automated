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
    # The intention layer. A state is not a label, it is a *reason* for the
    # next few seconds of attention, posture and blink rate - which is why the
    # successors are not uniform: finishing a read is a reason to settle back,
    # and checking chat is a reason to come back to the audience.
    "IDLE_ATTENTIVE": StateSpec(
        min_duration=6.0, median_duration=16.0, max_duration=46.0,
        successors={"READING": 2.4, "CHECKING_CHAT": 1.5, "WAITING": 1.3,
                    "FOCUSED": 1.1, "IDLE_RELAXED": 1.0, "THINKING": 0.7,
                    "MILD_POSITIVE": 0.4},
    ),
    "IDLE_RELAXED": StateSpec(
        min_duration=8.0, median_duration=22.0, max_duration=60.0,
        successors={"IDLE_ATTENTIVE": 2.0, "WAITING": 1.6, "READING": 1.3,
                    "CHECKING_CHAT": 0.9, "THINKING": 0.6},
    ),
    "READING": StateSpec(
        min_duration=7.0, median_duration=19.0, max_duration=48.0,
        # Having read something, he comes back to the audience or sits back.
        successors={"IDLE_ATTENTIVE": 2.2, "FOCUSED": 1.6, "IDLE_RELAXED": 1.4,
                    "CHECKING_CHAT": 1.0, "MILD_POSITIVE": 0.5,
                    "THINKING": 0.5},
        cooldown=14.0,
    ),
    "CHECKING_CHAT": StateSpec(
        min_duration=2.5, median_duration=5.5, max_duration=13.0,
        successors={"IDLE_ATTENTIVE": 2.8, "READING": 1.2,
                    "MILD_POSITIVE": 0.9, "IDLE_RELAXED": 0.8},
        cooldown=22.0,
    ),
    "WAITING": StateSpec(
        min_duration=8.0, median_duration=20.0, max_duration=55.0,
        successors={"IDLE_ATTENTIVE": 2.2, "IDLE_RELAXED": 1.8,
                    "CHECKING_CHAT": 1.2, "READING": 1.0},
        cooldown=25.0,
    ),
    "FOCUSED": StateSpec(
        min_duration=6.0, median_duration=15.0, max_duration=40.0,
        # Concentration is tiring; it resolves into sitting back.
        successors={"IDLE_RELAXED": 2.0, "IDLE_ATTENTIVE": 1.8, "READING": 1.4,
                    "WAITING": 0.8},
        cooldown=20.0,
    ),
    "THINKING": StateSpec(
        min_duration=2.0, median_duration=4.5, max_duration=11.0,
        successors={"IDLE_ATTENTIVE": 2.6, "READING": 1.4, "FOCUSED": 0.9,
                    "IDLE_RELAXED": 0.8},
        cooldown=38.0,
    ),
    "MILD_POSITIVE": StateSpec(
        min_duration=1.8, median_duration=4.0, max_duration=9.0,
        successors={"IDLE_ATTENTIVE": 2.8, "IDLE_RELAXED": 1.4,
                    "CHECKING_CHAT": 0.6},
        cooldown=55.0,
    ),
}


# Intentions that point him at a screen. Chaining these indefinitely produces
# exactly the failure this whole cycle is about: measured pitch sat at -15 to
# -17 degrees for the last two minutes of a five-minute clip, because READING
# and FOCUSED kept handing off to each other and he never looked up.
#
# People surface. Whatever they are reading, they periodically lift their eyes.
SCREEN_STATES = frozenset({"READING", "FOCUSED", "CHECKING_CHAT"})
MAX_SCREEN_RUN = 42.0     # seconds before surfacing is forced


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
        self._screen_run = 0.0

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

        # Surface. After a long uninterrupted stretch of screen-directed
        # intention he looks up, whatever the weights say - which is what a
        # person does and what the graph on its own would not.
        if self._screen_run >= MAX_SCREEN_RUN:
            self._screen_run = 0.0
            surf = {k: v for k, v in
                    {"IDLE_ATTENTIVE": 3.0, "IDLE_RELAXED": 1.6,
                     "WAITING": 1.2, "THINKING": 0.5}.items()
                    if k in self.graph}
            total = sum(surf.values())
            r = rng.uniform(0.0, total)
            acc = 0.0
            for name, w in surf.items():
                acc += w
                if r <= acc:
                    return name

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

        held = now - self.entered_at
        self._screen_run = (self._screen_run + held
                            if self.state in SCREEN_STATES else 0.0)

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
    # a wall of `blink, blink, blink`.
    #
    # Matched by prefix, not by splitting on ".". The engine emits
    # `blink_partial` and `double_blink_second`, neither of which an exact-match
    # test on "blink" excludes - so the first version of this filter let every
    # blink variant through and the detector duly reported blink-anchored
    # sequences as loops.
    INVOLUNTARY = ("blink", "microsaccade", "breath", "drift")

    def __init__(self, window: int = 400) -> None:
        self.events: deque[tuple[float, str]] = deque(maxlen=window)
        self.last_at: dict[str, float] = {}
        self.counts: Counter[str] = Counter()

    @staticmethod
    def _label(kind: str, detail: str) -> str:
        """Coarse action identity: what he did, not by exactly how much.

        `head_yaw to=(+0.28,-1.39,-0.99)deg` and the same move a degree over are
        the same *behaviour*, and a detector keyed on the numbers will either
        miss the repetition or - as happened here - report spurious ones when a
        quantised value collides. Attention keeps its target name because
        looking at the lens and looking at chat are genuinely different acts;
        everything else keeps only its kind.
        """
        if kind in ("attention", "subfixation"):
            # Sub-fixations carry their target for the same reason major shifts
            # do. Without it every sub-fixation was the single token
            # "subfixation", so any two in a row looked like a repeat and the
            # detector reported "monitor, subfixation, subfixation" as a loop -
            # which is not a loop, it is a man reading a screen. Keyed by
            # target, two glances at the same sub-region still register and two
            # at different ones correctly do not.
            return f"{kind}:{detail.split(' ')[0]}"
        if kind == "state_change":
            return f"state:{detail.split(' -> ')[-1]}"
        return kind

    def record(self, now: float, kind: str, detail: str = "") -> None:
        self.last_at[kind] = now
        self.counts[kind] += 1
        if not kind.startswith(self.INVOLUNTARY):
            self.events.append((now, self._label(kind, detail)))

    def since(self, kind: str, now: float) -> float:
        return now - self.last_at.get(kind, -1e9)

    def recent_count(self, kind: str, now: float, window: float) -> int:
        return sum(1 for t, k in self.events
                   if k.startswith(kind) and now - t <= window)

    def cooldown_factor(self, kind: str, now: float, window: float = 45.0,
                        decay: float = 0.5) -> float:
        """Multiplier in (0, 1] that shrinks the more often `kind` fired lately."""
        return decay ** self.recent_count(kind, now, window)

    def repeated_ngrams(self, n: int = 4, min_count: int = 4,
                        excess: float = 4.0):
        """Sequences that recur *more than chance would produce*.

        A raw count is not a loop detector. If half of all attention events are
        the lens, then `LENS -> DISPLAY -> LENS` will appear often in any
        sequence whatsoever, including a perfectly natural one, and a
        count-based test flags it every time.

        What matters is *excess* structure: how often a gram appears against how
        often the marginal distribution alone predicts. A gram at four times its
        expected rate is a habit the sampler is not supposed to have; a gram at
        one times its expected rate is just the marginals showing through.
        """
        seq = [k for _, k in self.events]
        if len(seq) < n * min_count:
            return []

        marg = Counter(seq)
        total = len(seq)
        positions = total - n + 1
        grams = Counter(tuple(seq[i:i + n]) for i in range(positions))

        out = []
        for g, c in grams.items():
            if c < min_count:
                continue
            p = 1.0
            for token in g:
                p *= marg[token] / total
            expected = p * positions
            if expected <= 0 or c / expected >= excess:
                out.append((g, c, round(expected, 2)))
        return sorted(out, key=lambda r: -r[1])
