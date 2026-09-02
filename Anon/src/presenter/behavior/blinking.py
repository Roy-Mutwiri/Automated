"""Spontaneous blink generation.

The brief bans fixed-interval blinking, and rightly - a metronomic blink is
detectable within about twenty seconds of watching. What replaces it has to
reproduce four properties of real spontaneous blinking:

1. **Skewed intervals.** Inter-blink intervals are positively skewed: a mode
   around 2-4 s with a long tail. Modelled here as a log-normal.
2. **A refractory floor.** The lid cannot re-fire immediately. Enforced as a
   hard minimum plus a soft cooldown gate.
3. **Bursts.** Real blinking clusters - double blinks are common and are not
   two independent draws from the interval distribution.
4. **Asymmetric kinematics.** Closing is much faster than opening, and the two
   lids are not perfectly synchronised. Handled by `curves.blink_profile` and
   a small per-eye offset.

Rate is state-dependent rather than constant: the literature reports roughly
double the resting rate during active conversation and roughly half during
reading or sustained focus. Those multipliers live in `state.py`.

See docs/human_behavior.md for the measurements behind the defaults.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from ..types import AvatarPose, BehaviorEvent
from .context import Drives
from .curves import blink_profile, clamp
from .randomness import Cooldown

__all__ = ["BlinkSystem"]


@dataclass
# How far visual demand can stretch the inter-blink interval. At demand 1.0 the
# interval is 2.2x the baseline, which turns a ~15/min conversational rate into
# ~7/min while reading - inside the measured range for both regimes.
BLINK_DEMAND_SPAN = 1.2


class _ActiveBlink:
    """One blink in flight."""

    start: float
    duration: float
    depth_l: float          # peak closure, 0..1
    depth_r: float
    offset_r: float         # right lid lags/leads the left, seconds
    close_fraction: float
    kind: str
    brow_coupling: float = 0.0

    def finished(self, now: float) -> bool:
        # The right lid may lag, so the blink is not over until both are back.
        return now >= self.start + self.duration + max(self.offset_r, 0.0)


class BlinkSystem:
    """Generates blinks and writes eyelid aperture into the pose."""

    def __init__(self) -> None:
        self._next_at: float | None = None
        self._active: _ActiveBlink | None = None
        self._pending_double: float | None = None
        self._cooldown = Cooldown(duration=0.85)
        self.last_blink_time: float = -999.0
        self.blink_count: int = 0

    # -- scheduling --------------------------------------------------------
    def _schedule_next(self, drives: Drives) -> None:
        p = drives.profile
        median = drives.rate(p.blink_median_interval, drives.mod.blink_rate)

        interval = drives.rng.lognormal_interval(
            median=median,
            shape=p.blink_interval_shape,
            low=p.blink_min_interval,
            high=p.blink_max_interval,
        )

        # Blink probability rises the longer the gaze has been held. This is
        # the documented interaction between fixation and blinking, and it is
        # also what stops the eyes reading as glassy during a long hold: a
        # blink arrives naturally at the end of a fixation rather than at an
        # unrelated moment.
        fixation = drives.time_since_gaze_shift
        if fixation > 4.0:
            interval *= clamp(1.0 - 0.16 * (fixation - 4.0) / 6.0, 0.62, 1.0)

        # Visual demand suppresses blinking. Measured spontaneous blink rate
        # runs 1.4-14.4/min while reading against 10.5-32.5/min in
        # conversation; the eye holds the lid open to avoid losing visual
        # information it is actively using. The span below reproduces roughly
        # that factor of three across the range of targets a streamer looks at,
        # so blink rate falls out of *what he is doing* rather than being a
        # constant of his personality.
        demand = clamp(drives.visual_demand, 0.0, 1.0)
        interval *= 1.0 + BLINK_DEMAND_SPAN * demand

        self._next_at = drives.now + interval

    def _begin_blink(self, drives: Drives, kind: str = "blink") -> None:
        p = drives.profile
        rng = drives.rng

        duration = rng.truncated_gauss(
            p.blink_duration_mean, p.blink_duration_sigma, 0.09, 0.40
        )

        # Guarantee the blink is *sampled* often enough to read as movement.
        #
        # Frame-rate-independent timing is not enough on its own. A 145 ms blink
        # at 13 FPS lands on a single intermediate frame - measured sequence
        # 0.00 -> 0.88 -> 0.00 - so the lid appears to teleport shut and back.
        # That single-frame flash is the specific thing that reads as synthetic;
        # it is a sampling artefact, not a timing error.
        #
        # Stretching the blink so it spans at least `blink_min_frames` rendered
        # frames costs realism in the strictest sense - a real blink does not
        # slow down because a camera is slow - but a slightly long blink that
        # is *seen* beats a physiologically exact one that is not. The result
        # stays inside the 100-400 ms range reported for spontaneous blinks at
        # any frame rate above about 10 FPS.
        #
        # This is an accommodation, not a fix. Raising the render rate removes
        # the need for it: above ~25 FPS the floor rarely binds.
        min_duration = p.blink_min_frames * max(drives.frame_interval, 1e-3)
        if min_duration > duration:
            duration = min(min_duration, p.blink_max_duration)

        # Most blinks close fully. A minority are incomplete - a real and very
        # common behaviour that, included sparingly, breaks the uniformity of
        # a face that always blinks identically.
        if rng.chance(p.blink_partial_probability):
            depth = rng.uniform(0.55, 0.85)
            kind = "blink_partial"
        else:
            depth = rng.uniform(0.96, 1.0)

        # Asymmetry: never mirror the two lids exactly. Kept small - a large
        # difference reads as a wink or as facial palsy, not as realism.
        asym = p.blink_asymmetry
        depth_l = clamp(depth * (1.0 + rng.uniform(-asym, asym)), 0.0, 1.0)
        depth_r = clamp(depth * (1.0 + rng.uniform(-asym, asym)), 0.0, 1.0)
        offset_r = rng.uniform(-0.012, 0.012)

        self._active = _ActiveBlink(
            start=drives.now,
            duration=duration,
            depth_l=depth_l,
            depth_r=depth_r,
            offset_r=offset_r,
            close_fraction=rng.jitter(p.blink_close_fraction, 0.12),
            kind=kind,
            brow_coupling=rng.jitter(p.blink_brow_coupling, 0.3),
        )

        self.last_blink_time = drives.now
        self.blink_count += 1
        self._cooldown.trigger()

        drives.emit(
            BehaviorEvent(
                time=drives.now,
                kind=kind,
                detail=f"dur={duration * 1000:.0f}ms depth={depth:.2f}",
                magnitude=depth,
                metadata={"duration": duration, "depth": depth},
            )
        )

        # Decide now whether this becomes a double blink. Doubles are a burst
        # from one trigger, not two independent samples, so the second is
        # scheduled directly instead of going through the interval draw.
        if kind != "double_blink_second" and rng.chance(p.double_blink_probability):
            self._pending_double = drives.now + duration + rng.uniform(0.06, 0.19)
        else:
            self._pending_double = None
            self._next_at = None  # re-drawn once the blink completes

    # -- per-frame ---------------------------------------------------------
    def update(self, drives: Drives, pose: AvatarPose) -> None:
        self._cooldown.tick(drives.dt)

        if self._next_at is None and self._active is None and self._pending_double is None:
            self._schedule_next(drives)

        # A queued second blink of a double takes priority over the schedule.
        if (
            self._active is None
            and self._pending_double is not None
            and drives.now >= self._pending_double
        ):
            self._pending_double = None
            self._begin_blink(drives, kind="double_blink_second")

        elif (
            self._active is None
            and self._next_at is not None
            and drives.now >= self._next_at
        ):
            # The soft gate makes a blink progressively more likely as the
            # cooldown decays rather than switching on at a hard edge, which
            # would put a faint rhythm into the blink train.
            if self._cooldown.ready or drives.rng.chance(self._cooldown.gate()):
                self._begin_blink(drives)

        self._apply(drives, pose)

    def _apply(self, drives: Drives, pose: AvatarPose) -> None:
        blink = self._active
        if blink is None:
            pose.eye_open_l = 1.0
            pose.eye_open_r = 1.0
            return

        if blink.finished(drives.now):
            self._active = None
            pose.eye_open_l = 1.0
            pose.eye_open_r = 1.0
            if self._pending_double is None:
                self._schedule_next(drives)
            return

        elapsed = drives.now - blink.start
        tau_l = elapsed / blink.duration
        tau_r = (elapsed - blink.offset_r) / blink.duration

        closure_l = blink_profile(tau_l, blink.close_fraction) * blink.depth_l
        closure_r = blink_profile(tau_r, blink.close_fraction) * blink.depth_r

        pose.eye_open_l = clamp(1.0 - closure_l, 0.0, 1.0)
        pose.eye_open_r = clamp(1.0 - closure_r, 0.0, 1.0)

        # A blink is not just an eyelid. Orbicularis oculi wraps the whole eye
        # socket, so a real blink pulls the brow down slightly and lifts the
        # cheek a little, and the face relaxes back afterwards. Isolating the
        # lid - moving nothing else on the face - is a large part of why a
        # synthetic blink reads as a shutter rather than as a person blinking.
        #
        # Small, and proportional to how far the lid actually closed, so a
        # partial blink gets a partial brow movement.
        coupling = 0.5 * (closure_l + closure_r)
        if coupling > 0.01:
            pose.brow_l -= coupling * blink.brow_coupling
            pose.brow_r -= coupling * blink.brow_coupling * 0.88
            pose.cheek += coupling * blink.brow_coupling * 0.35

    # -- introspection -----------------------------------------------------
    @property
    def is_blinking(self) -> bool:
        return self._active is not None

    def time_to_next(self, now: float) -> float:
        if self._next_at is None:
            return 0.0
        return max(0.0, self._next_at - now)
