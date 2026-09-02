"""Respiration, in the torso where it belongs.

## What this replaces, and why it was wrong

The previous implementation wrote breathing into `pose.scale`, `pose.ty` and
`pose.pitch` - it made the *head* periodically larger. That is not a tuning
error, it is the wrong anatomy: breathing does not change the size of a head.
The head-scale version existed because head scale was the only channel the 2D
face renderer exposed, which is exactly the coupling between behaviour and
renderer that the canonical motion state now forbids.

Breathing originates in the rib cage. Everything above it is a consequence,
and the consequences get rapidly smaller as they travel up:

| | |
|---|---|
| rib cage / chest | primary |
| upper chest, mid spine | high |
| clavicle | low |
| shoulder | very low |
| neck | extremely low |
| head | near zero |

The neck term is *negative* relative to the chest on purpose. As the chest
rises and opens it would carry the head back with it; a real neck compensates
so the gaze stays level. Without that compensation the head nods gently in time
with the breath, which is the single most obvious way a breathing rig announces
itself.

## The waveform

Not a sine. Quiet respiration is markedly asymmetric and has a pause:

    rest -> inhale (active, shorter) -> brief transition at the top
         -> exhale (passive, longer) -> variable rest -> inhale

A raised cosine is used within the inhale and exhale segments so velocity is
zero at both ends; a linear ramp would produce a visible corner at the top of
every breath.

## Continuity

Rate and depth are **slowly varying**, not re-drawn per cycle. An earlier
version sampled a fresh period and depth for every breath, which is more random
but less physiological: real respiration drifts over tens of seconds and
successive breaths resemble each other. Two Ornstein-Uhlenbeck processes on
~50 s timescales carry that drift, with only a small per-cycle jitter on top so
no two breaths are identical.
"""

from __future__ import annotations

import math

from ..behavior.randomness import OrnsteinUhlenbeck
from .state import BreathingState, HumanMotionState

__all__ = ["RespirationSystem", "BREATH_COUPLING"]

# Peak joint response at drive = 1.0 and depth = 1.0, in degrees.
#
# Signs: -rx on the chest is extension - the chest opening and lifting. The
# neck's +rx is the compensation that keeps the head level while the chest
# carries it back.
#
# These are small. If a viewer can consciously see the shoulders rising, the
# whole effect is wrong, so the shoulder term is an order of magnitude below
# the chest.
BREATH_COUPLING = {
    "chest":       (-0.95, 0.0, 0.0),
    "spine_mid":   (-0.34, 0.0, 0.0),
    "spine_lower": (-0.10, 0.0, 0.0),
    "clavicle_l":  (-0.26, 0.0, +0.30),
    "clavicle_r":  (-0.26, 0.0, -0.30),
    "shoulder_l":  (-0.09, 0.0, +0.10),
    "shoulder_r":  (-0.09, 0.0, -0.10),
    "neck":        (+0.30, 0.0, 0.0),
    "head":        (+0.04, 0.0, 0.0),
}

# Rib-cage circumference change at full inhale, as a fraction. Applied by
# adapters that can express a shape change; the 2D face adapter ignores it.
RIB_EXPANSION = 0.013


class RespirationSystem:
    """Produces a `BreathingState` and writes its consequences into the body."""

    def __init__(self, profile) -> None:
        base_period = getattr(profile, "breath_period", 4.1)
        self.base_rate = 60.0 / max(base_period, 0.5)

        # Slow drift. Time constants deliberately much longer than one breath,
        # so successive breaths resemble each other.
        self._rate_drift = OrnsteinUhlenbeck.from_amplitude(0.085, 52.0)
        self._depth_drift = OrnsteinUhlenbeck.from_amplitude(0.16, 44.0)

        self._phase = 0.0
        self._period = 60.0 / self.base_rate
        self.breath_count = 0

        # Segment fractions: inhale, transition, exhale, rest. Re-jittered a
        # little each cycle around these.
        self._segments = (0.34, 0.07, 0.44, 0.15)
        self._seg = self._segments

    # -- waveform ------------------------------------------------------------
    @staticmethod
    def _ramp(t: float) -> float:
        """Raised cosine, zero velocity at both ends."""
        return 0.5 - 0.5 * math.cos(math.pi * min(max(t, 0.0), 1.0))

    def _drive_at(self, phase: float) -> float:
        inh, trans, exh, _rest = self._seg
        if phase < inh:
            return self._ramp(phase / max(inh, 1e-6))
        if phase < inh + trans:
            return 1.0
        if phase < inh + trans + exh:
            return 1.0 - self._ramp((phase - inh - trans) / max(exh, 1e-6))
        return 0.0

    def _new_cycle(self, rng) -> None:
        self.breath_count += 1
        inh, trans, exh, rest = self._segments
        j = lambda v, s: max(v * (1.0 + rng.gauss(0.0, s)), 0.02)
        seg = [j(inh, 0.10), j(trans, 0.28), j(exh, 0.09), j(rest, 0.35)]
        total = sum(seg)
        self._seg = tuple(s / total for s in seg)

    # -- per frame -----------------------------------------------------------
    def update(self, drives) -> BreathingState:
        rate_mod = 1.0 + self._rate_drift.step(drives.dt, drives.rng)
        depth = 1.0 + self._depth_drift.step(drives.dt, drives.rng)

        # Arousal breathes a little faster and shallower; a subdued presenter
        # slower and deeper. Small - this is not panting.
        rate = self.base_rate * rate_mod * (1.0 + 0.10 * drives.arousal)
        rate *= getattr(drives.mod, "breath_rate", 1.0)
        depth *= 1.0 - 0.08 * drives.arousal

        self._period = 60.0 / max(rate, 4.0)
        self._phase += drives.dt / self._period
        while self._phase >= 1.0:
            self._phase -= 1.0
            self._new_cycle(drives.rng)

        return BreathingState(
            phase=self._phase,
            depth=max(depth, 0.35),
            rate=rate,
            drive=self._drive_at(self._phase),
        )

    def apply(self, breath: BreathingState, motion: HumanMotionState) -> None:
        """Write the breath's consequences into the body's joints."""
        # Centred on zero: at rest the torso is neutral, at full inhale it is
        # extended. Driving 0..1 instead would leave the chest permanently
        # half-inflated at rest.
        amount = (breath.drive - 0.5) * 2.0 * breath.depth
        joints = motion.joints()
        for name, (rx, ry, rz) in BREATH_COUPLING.items():
            j = joints.get(name)
            if j is None:
                continue
            j.rx += rx * amount
            j.ry += ry * amount
            j.rz += rz * amount
        motion.breathing = breath
