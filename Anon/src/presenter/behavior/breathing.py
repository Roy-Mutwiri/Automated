"""Quiet respiration.

Breathing is the one behaviour here that is genuinely near-periodic, so it is
also the one most at risk of reading as a loop. The defence is that no two
breaths are identical: period, depth and the inhale/exhale split are re-drawn
every cycle, so the signal is quasi-periodic rather than periodic and never
lines up with itself.

Quiet seated respiration runs about 12-18 breaths per minute - a 3.3-5.0 s
cycle - and is markedly asymmetric: a shorter active inhale followed by a
longer passive exhale, often with a brief pause before the next breath. A pure
sine gets all three wrong.

Amplitude is the critical parameter and the easy one to overdo. On a
head-and-shoulders framing, visible chest pumping is wrong; what a viewer
actually perceives is a fractional scale change and a sub-pixel vertical
drift. The defaults here are deliberately near the threshold of perception -
the brief's word for it is "subconscious". If breathing is *visible* as
breathing, it is turned up too far.
"""

from __future__ import annotations

import math

from ..types import AvatarPose
from .context import Drives
from .curves import clamp

__all__ = ["BreathingSystem"]


class BreathingSystem:
    """Writes scale, vertical offset and a small pitch component."""

    def __init__(self, profile) -> None:
        self._phase = 0.0             # 0..1 through the current breath
        self._period = profile.breath_period
        self._depth = 1.0
        self._inhale_fraction = profile.breath_inhale_fraction
        self._pause = 0.0             # end-expiratory pause, seconds
        self._pause_left = 0.0
        self.breath_count = 0
        self._new_breath(None)

    def _new_breath(self, drives: Drives | None) -> None:
        """Re-draw this breath's parameters. Called once per cycle."""
        if drives is None:
            return
        p = drives.profile
        rng = drives.rng

        rate_gain = drives.mod.breathing_rate * (1.0 + 0.12 * drives.arousal)
        base_period = p.breath_period / max(rate_gain, 0.2)

        self._period = rng.truncated_gauss(
            base_period, p.breath_period_sigma, 2.2, 8.5
        )
        # Depth and period are positively correlated in real breathing - a
        # longer breath is usually a deeper one. Independent draws produce
        # occasional long shallow breaths that look wrong.
        period_ratio = self._period / max(base_period, 1e-3)
        self._depth = clamp(
            rng.truncated_gauss(period_ratio, 0.16, 0.55, 1.7), 0.5, 1.8
        )
        self._inhale_fraction = rng.truncated_gauss(
            p.breath_inhale_fraction, 0.045, 0.28, 0.55
        )
        # A short pause at the end of expiration. Present in quiet breathing,
        # and its absence is part of why a sine reads as mechanical.
        self._pause = max(0.0, rng.gauss(0.28, 0.18))
        self.breath_count += 1

    def _excursion(self) -> float:
        """Chest excursion for the current phase, 0 at rest to 1 at full inhale.

        Inhale is the active phase and rises faster; exhale is passive elastic
        recoil and decays more slowly. Modelled as two half-cosines with
        different widths, which is closer to a real respiratory trace than a
        single sinusoid and costs nothing.
        """
        t = self._phase
        f = self._inhale_fraction
        if t <= f:
            return 0.5 - 0.5 * math.cos(math.pi * (t / f))
        return 0.5 + 0.5 * math.cos(math.pi * ((t - f) / (1.0 - f)))

    def update(self, drives: Drives, pose: AvatarPose) -> None:
        if self._pause_left > 0.0:
            self._pause_left = max(0.0, self._pause_left - drives.dt)
            excursion = 0.0
        else:
            self._phase += drives.dt / max(self._period, 1e-3)
            if self._phase >= 1.0:
                self._phase = 0.0
                self._pause_left = self._pause
                self._new_breath(drives)
            excursion = self._excursion()

        p = drives.profile
        amount = excursion * self._depth

        # Additive, so breathing rides on whatever head/posture is doing rather
        # than overwriting it.
        pose.scale += amount * p.breath_scale_amount
        pose.ty += amount * p.breath_ty_amount
        # The head tips back a fraction of a degree as the chest rises. This is
        # what sells breathing on a head-and-shoulders crop, more than the
        # scale change does.
        pose.pitch += amount * p.breath_pitch_amount

        pose.breathing_phase = self._phase

    @property
    def phase(self) -> float:
        return self._phase
