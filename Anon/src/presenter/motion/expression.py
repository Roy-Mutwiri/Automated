"""Facial emotion: coordinated expressions with a temporal profile.

Separate from `behavior/expression.py`, which is the frozen micro-expression and
brow-tone layer. This is the emotion layer, and it writes into the canonical
motion state rather than into a renderer's pose.

## A smile is not a mouth-corner slider

The hard fail named in the brief is corners that move while the cheeks stay dead
and the eyes stay identical. That is a rictus, and it is what
`mouth_corner += 1` produces every time.

A real smile is *zygomaticus major* pulling the corners up and back, which
raises the cheek mass, which in turn narrows the eye aperture from below - the
lower lid rises, it does not squint from above. So every smile here writes
corners, cheeks and lower lids together, in fixed proportion, and the
proportions differ by smile type: a polite smile is nearly all mouth, an amused
one is mostly eyes.

## Temporal profile

Never neutral -> value -> neutral.

    onset (fast, ~250 ms)
      -> peak
      -> hold (variable, and this is what distinguishes types)
      -> decay (slower than onset, always)
      -> residual (a fraction that lingers for seconds)
      -> baseline

The decay being slower than the onset is not a stylistic choice: a smile that
falls as fast as it rises reads as switched off, which is one of the most
reliable tells of a rig. The residual is why a person who has just been amused
still looks faintly amused a few seconds later.

## Asymmetry

Stable within one expression, not re-rolled per frame. The side that leads and
the amount it leads by are drawn once at onset, so the smile is lopsided the way
a face is lopsided rather than flickering.
"""

from __future__ import annotations

import math

from ..types import BehaviorEvent
from .state import FaceParameters, HumanMotionState

__all__ = ["FacialExpressionSystem", "EXPRESSIONS"]


# peak: mouth-corner amplitude. The other fields are *proportions of that
# peak*, which is what keeps a smile coordinated at every intensity.
EXPRESSIONS: dict[str, dict] = {
    "MICRO_SMILE": dict(
        peak=0.16, cheek=0.55, squint=0.28, brow=0.04,
        onset=0.30, hold=(0.7, 1.8), decay=0.85, residual=0.10,
        valence=+0.25,
    ),
    "SMALL_SMILE": dict(
        peak=0.34, cheek=0.75, squint=0.46, brow=0.06,
        onset=0.26, hold=(1.1, 2.8), decay=1.15, residual=0.18,
        valence=+0.45,
    ),
    "WARM_SMILE": dict(
        peak=0.52, cheek=0.88, squint=0.62, brow=0.10,
        onset=0.30, hold=(1.6, 3.6), decay=1.5, residual=0.22,
        valence=+0.65,
    ),
    "AMUSED": dict(
        # Mostly eyes. An amused expression with a wide mouth and flat eyes is
        # the classic uncanny smile.
        peak=0.44, cheek=0.95, squint=0.80, brow=0.14,
        onset=0.22, hold=(1.2, 3.0), decay=1.35, residual=0.26,
        valence=+0.7, head_tilt=1.6, shoulder_relax=0.7,
    ),
    "SMIRK": dict(
        peak=0.30, cheek=0.45, squint=0.30, brow=0.05,
        onset=0.28, hold=(0.9, 2.2), decay=1.0, residual=0.16,
        valence=+0.3, asymmetry=0.75,
    ),
    # Concentration is genuinely a quiet expression, but it was quiet enough
    # to be absent: at the 0.55 the THINKING state triggers it with, it moved
    # the face 1.8 px at 1080p. Strengthened until its two trigger levels land
    # above the ~3 px the plate needs before a change is legible, which is
    # still far short of a scowl.
    "FOCUSED": dict(
        peak=0.0, cheek=0.0, squint=0.38, brow=-0.34, furrow=0.46,
        onset=0.6, hold=(2.0, 6.0), decay=1.8, residual=0.20,
        valence=0.0,
    ),
    # `brow` was 0.0 here while `brow_split` was 0.55. The split is a
    # *proportion of the brow raise*, so a zero brow made it meaningless and
    # the one raised eyebrow that defines scepticism never rendered at all.
    # Measured on the plate, the whole expression moved 0.9 px at full
    # intensity - below anything a viewer can see. The mouth alone cannot
    # carry it; the brow has to do the work.
    "SKEPTICAL": dict(
        peak=-0.16, cheek=0.10, squint=0.30, brow=0.34, furrow=0.22,
        onset=0.45, hold=(1.0, 2.6), decay=1.3, residual=0.14,
        valence=-0.15, asymmetry=0.85, brow_split=0.90, head_tilt=2.4,
    ),
    "SURPRISED": dict(
        peak=0.10, cheek=0.05, squint=-0.55, brow=0.85,
        onset=0.14, hold=(0.35, 0.9), decay=0.75, residual=0.08,
        valence=+0.1, head_recoil=-1.2,
    ),
}


def _smoothstep(t: float) -> float:
    t = min(max(t, 0.0), 1.0)
    return t * t * (3.0 - 2.0 * t)


class _Instance:
    """One expression, in flight."""

    __slots__ = ("name", "spec", "t0", "onset", "hold", "decay",
                 "lead_left", "asym", "scale", "latency")

    def __init__(self, name, spec, now, rng, scale, latency):
        self.name = name
        self.spec = spec
        self.t0 = now + latency
        self.latency = latency
        self.onset = spec["onset"] * rng.uniform(0.85, 1.20)
        lo, hi = spec["hold"]
        self.hold = rng.uniform(lo, hi)
        self.decay = spec["decay"] * rng.uniform(0.9, 1.3)
        self.scale = scale
        # Which side leads, and by how much. Drawn once: a smile that changes
        # its lopsidedness every frame is noise, not a face.
        self.lead_left = rng.chance(0.5)
        base = spec.get("asymmetry", 0.14)
        self.asym = base * rng.uniform(0.6, 1.25)

    def level(self, now: float) -> float:
        t = now - self.t0
        if t < 0.0:
            return 0.0                       # still inside the reaction latency
        if t < self.onset:
            return _smoothstep(t / self.onset)
        t -= self.onset
        if t < self.hold:
            return 1.0
        t -= self.hold
        if t < self.decay:
            # Decay always slower than onset, and to the residual rather than
            # to zero.
            k = 1.0 - _smoothstep(t / self.decay)
            return self.spec["residual"] + (1.0 - self.spec["residual"]) * k
        return self.spec["residual"]

    def finished(self, now: float) -> bool:
        return (now - self.t0) > (self.onset + self.hold + self.decay + 6.0)


class FacialExpressionSystem:
    """Chooses expressions, and writes them into the face and the body."""

    def __init__(self, profile, persona=None) -> None:
        persona = persona or {}
        self.strength = float(persona.get("smile_strength", 0.30)) / 0.30
        self.frequency = float(persona.get("smile_frequency", 0.35))
        self.latency_median = float(persona.get("reaction_latency_median", 0.42))
        self.latency_shape = float(persona.get("reaction_latency_shape", 0.35))
        self._active: _Instance | None = None
        self._residual = 0.0
        self._residual_name = None
        self.count = 0

    # -- external trigger ----------------------------------------------------
    def trigger(self, drives, name: str, intensity: float = 1.0) -> None:
        """Start an expression, after a human reaction delay.

        Nothing reacts at frame zero. The delay is drawn per event rather than
        fixed, and it is carried *inside* the instance so the expression's own
        clock starts late rather than the whole system stalling.
        """
        spec = EXPRESSIONS.get(name)
        if spec is None:
            return
        latency = drives.rng.lognormal_interval(
            median=self.latency_median, shape=self.latency_shape,
            low=0.09, high=1.4)
        scale = intensity * self.strength * drives.rng.uniform(0.8, 1.15)
        self._active = _Instance(name, spec, drives.now, drives.rng, scale, latency)
        self.count += 1
        drives.emit(BehaviorEvent(
            time=drives.now, kind="expression_emotion", detail=name,
            magnitude=scale,
            metadata={"name": name, "latency": latency, "scale": scale}))

    # -- per frame -----------------------------------------------------------
    def update(self, drives, motion: HumanMotionState) -> None:
        inst = self._active
        if inst is None:
            return
        if inst.finished(drives.now):
            self._active = None
            return

        level = inst.level(drives.now) * inst.scale
        if level <= 1e-4:
            return

        spec = inst.spec
        f: FaceParameters = motion.face
        peak = spec["peak"] * level

        # Asymmetry: one side leads. Stable for this instance.
        a = inst.asym
        l_gain = 1.0 + (a if inst.lead_left else -a)
        r_gain = 1.0 + (-a if inst.lead_left else a)

        f.mouth_corner_l += peak * l_gain
        f.mouth_corner_r += peak * r_gain

        # The cheek is raised *by* the corner pull, so it is proportional to it
        # rather than independent. This is the coupling that stops the smile
        # being a mouth-only rictus.
        cheek = abs(peak) * spec.get("cheek", 0.0)
        f.cheek_l += cheek * l_gain
        f.cheek_r += cheek * r_gain

        # Lower lid rises with the cheek. Duchenne. Negative squint on
        # SURPRISED widens the aperture instead.
        squint = abs(peak) * spec.get("squint", 0.0) if peak >= 0 else \
            level * spec.get("squint", 0.0)
        if spec.get("squint", 0.0) < 0:
            squint = level * spec["squint"]
        f.eye_squint_l += squint * l_gain
        f.eye_squint_r += squint * r_gain

        brow = spec.get("brow", 0.0) * level
        # Which brow leads follows the same coin as the head tilt, rather than
        # always the left. Two reasons: a person does not raise the same
        # eyebrow every time, and the renderer's two brow channels are not
        # equally strong, so a permanently left-leading split would ship the
        # weaker of the two every time.
        split = spec.get("brow_split", 0.0)
        if not inst.lead_left:
            split = -split
        f.brow_outer_l += brow * (1.0 + split)
        f.brow_outer_r += brow * (1.0 - split)
        f.brow_inner += brow * 0.35
        f.brow_furrow += spec.get("furrow", 0.0) * level

        # --- face/body coupling -------------------------------------------
        # Emotion cannot stay only in the face. Amusement tilts the head and
        # lets the shoulders down; scepticism tilts the head the other way.
        tilt = spec.get("head_tilt", 0.0) * level
        if tilt:
            motion.head.rz += tilt * (1.0 if inst.lead_left else -1.0)
        relax = spec.get("shoulder_relax", 0.0) * level
        if relax:
            motion.clavicle_l.rz -= relax * 0.5
            motion.clavicle_r.rz += relax * 0.5
        recoil = spec.get("head_recoil", 0.0) * level
        if recoil:
            motion.neck.rx += recoil

        motion.emotion.valence += spec.get("valence", 0.0) * level
        motion.emotion.label = inst.name

    @property
    def active(self) -> str | None:
        return self._active.name if self._active else None
