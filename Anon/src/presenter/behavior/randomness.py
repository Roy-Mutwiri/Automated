"""Stochastic primitives with the distributions human timing actually follows.

The brief is explicit that uniform `random()` everywhere looks chaotic, and it
is right: uniform sampling produces intervals that are too evenly spread and
motion that has no temporal correlation. Two things fix that.

**Interval distributions.** Waiting times between spontaneous behaviours are
positively skewed - many short gaps, a long tail of occasional long ones. That
is a log-normal, not a uniform, and it is what the blink literature reports for
inter-blink intervals (see docs/human_behavior.md for sources). Sampling a
skewed distribution is what produces the "clustering then a long quiet stretch"
texture that reads as alive.

**Correlated noise.** Continuous drift - head pose, posture - must wander
slowly and stay bounded. White noise jitters; a sine loops. The
Ornstein-Uhlenbeck process does neither: it is mean-reverting, temporally
correlated, and never repeats. That makes it the right generator for anything
the viewer should perceive as "not quite still" without perceiving a pattern.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field

__all__ = ["Rng", "OrnsteinUhlenbeck", "Cooldown"]


class Rng:
    """Seeded RNG exposing the distributions this project actually needs.

    Seeding matters here: a reproducible seed is what makes the 30-minute
    timeline analysis meaningful, and what lets a reported artefact be
    reproduced rather than argued about.
    """

    def __init__(self, seed: int | None = None) -> None:
        self._py = random.Random(seed)
        self.seed = seed

    # -- uniform ----------------------------------------------------------
    def uniform(self, low: float, high: float) -> float:
        return self._py.uniform(low, high)

    def chance(self, probability: float) -> bool:
        """True with the given probability."""
        if probability <= 0.0:
            return False
        if probability >= 1.0:
            return True
        return self._py.random() < probability

    def choice(self, items):
        return self._py.choice(items)

    # -- shaped ------------------------------------------------------------
    def gauss(self, mean: float, sigma: float) -> float:
        return self._py.gauss(mean, sigma)

    def truncated_gauss(
        self, mean: float, sigma: float, low: float, high: float
    ) -> float:
        """Gaussian resampled into [low, high].

        Used for amplitudes that have a natural centre but a hard anatomical
        limit - head yaw, for instance, must never exceed its range no matter
        how far into the tail a sample lands.
        """
        if high <= low:
            return low
        for _ in range(12):
            value = self._py.gauss(mean, sigma)
            if low <= value <= high:
                return value
        # Tail-heavy draw or a badly configured range: fall back rather than
        # loop forever. Clamping biases toward the bound, which is acceptable
        # for the rare case but would not be as the primary path.
        return min(max(self._py.gauss(mean, sigma), low), high)

    def lognormal_interval(
        self,
        median: float,
        shape: float = 0.55,
        low: float | None = None,
        high: float | None = None,
    ) -> float:
        """Positively skewed waiting time with the given median.

        Parameterised by median rather than by the underlying normal's mean,
        because the median is the quantity that is actually reported in the
        behavioural literature and the one a person tuning the config can
        reason about. For a log-normal, median = exp(mu), so mu = ln(median).

        `shape` is sigma of the underlying normal: larger means a longer tail
        and burstier behaviour. Around 0.5-0.6 gives the spread reported for
        spontaneous blinking.
        """
        if median <= 0.0:
            return 0.0
        value = math.exp(self._py.gauss(math.log(median), shape))
        if low is not None:
            value = max(value, low)
        if high is not None:
            value = min(value, high)
        return value

    def exponential_interval(self, mean: float) -> float:
        """Memoryless waiting time - a Poisson process's inter-arrival time.

        Appropriate where an event genuinely has no refractory structure.
        Most behaviours here do have one, so log-normal is usually the better
        choice; this exists for the ones that do not.
        """
        if mean <= 0.0:
            return 0.0
        return self._py.expovariate(1.0 / mean)

    def gamma_interval(self, shape: float, mean: float) -> float:
        """Gamma waiting time.

        A sum of `shape` exponential stages, so it has a refractory bump near
        zero rather than the exponential's peak at zero. Useful where an event
        cannot immediately re-fire but is otherwise memoryless - gaze
        fixations behave this way.
        """
        if mean <= 0.0 or shape <= 0.0:
            return 0.0
        return self._py.gammavariate(shape, mean / shape)

    def sign(self) -> float:
        return 1.0 if self._py.random() < 0.5 else -1.0

    def jitter(self, value: float, fraction: float) -> float:
        """Perturb `value` by +/- `fraction` of itself."""
        return value * (1.0 + self.uniform(-fraction, fraction))


@dataclass
class OrnsteinUhlenbeck:
    """Mean-reverting correlated noise: dx = theta*(mu - x)*dt + sigma*dW.

    This is the workhorse for "alive but not moving on purpose" - the residual
    sway of a head that is being held still, the slow settle of posture. Three
    properties earn it that job:

    * **Correlated in time.** Consecutive samples are close, so it looks like
      motion rather than jitter.
    * **Mean reverting.** It cannot wander away and leave the head at an
      absurd angle, so no clamping artefacts.
    * **Aperiodic.** Unlike a sine or a noise loop it never repeats, which is
      what the "must survive hours of viewing" requirement demands.

    `theta` is the reversion rate in 1/s: roughly, the reciprocal of how long
    an excursion persists. `sigma` sets the drive strength; the resulting
    stationary standard deviation is sigma / sqrt(2*theta), which is exposed as
    `stationary_sigma` so a caller can tune in units of visible amplitude.
    """

    theta: float
    sigma: float
    mu: float = 0.0
    value: float = field(default=0.0)

    @classmethod
    def from_amplitude(
        cls, stationary_sigma: float, correlation_time: float, mu: float = 0.0
    ) -> "OrnsteinUhlenbeck":
        """Construct from the two quantities that are actually intuitive.

        `stationary_sigma` is the typical excursion size in output units;
        `correlation_time` is how many seconds an excursion tends to last.
        """
        theta = 1.0 / max(correlation_time, 1e-3)
        sigma = stationary_sigma * math.sqrt(2.0 * theta)
        return cls(theta=theta, sigma=sigma, mu=mu, value=mu)

    @property
    def stationary_sigma(self) -> float:
        return self.sigma / math.sqrt(2.0 * self.theta)

    def step(self, dt: float, rng: Rng) -> float:
        """Advance by `dt` seconds and return the new value.

        Uses exact discretisation rather than Euler-Maruyama, so the process
        keeps the correct stationary variance regardless of step size. That
        matters because dt here is a real frame time that fluctuates; an Euler
        step would make the motion amplitude depend on the frame rate.
        """
        if dt <= 0.0:
            return self.value
        decay = math.exp(-self.theta * dt)
        variance = (self.sigma * self.sigma) / (2.0 * self.theta) * (
            1.0 - decay * decay
        )
        self.value = (
            self.mu + (self.value - self.mu) * decay + rng.gauss(0.0, math.sqrt(variance))
        )
        return self.value


@dataclass
class Cooldown:
    """A refractory timer. The memory in "randomness must have memory".

    A behaviour that just fired should be unlikely to fire again immediately.
    Rather than rejecting samples after the fact, subsystems consult a cooldown
    and either suppress or attenuate their firing probability while it is warm.
    """

    duration: float
    remaining: float = 0.0

    def trigger(self, duration: float | None = None) -> None:
        self.remaining = self.duration if duration is None else duration

    def tick(self, dt: float) -> None:
        if self.remaining > 0.0:
            self.remaining = max(0.0, self.remaining - dt)

    @property
    def ready(self) -> bool:
        return self.remaining <= 0.0

    def gate(self) -> float:
        """A 0..1 multiplier that rises from 0 to 1 as the cooldown expires.

        Preferable to a hard boolean for probabilistic events: the behaviour
        becomes progressively more likely rather than switching on at an edge,
        which avoids a subtle rhythmic artefact where events cluster just after
        each cooldown boundary.
        """
        if self.duration <= 0.0:
            return 1.0
        if self.remaining <= 0.0:
            return 1.0
        return 1.0 - (self.remaining / self.duration)
