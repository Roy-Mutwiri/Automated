"""Mapping from semantic pose channels into LivePortrait expression space.

**Read this before trusting any number in this file.**

LivePortrait's expression vector `exp` is 21 keypoints x 3 axes of *learned,
undocumented* latent deformation. There is no published semantic index map. The
model was trained to consume deltas extracted from real driving frames, not
hand-authored ones, so which dimension moves an iris — and by how much before
the face distorts — is an empirical question, not something to be reasoned out
from the architecture.

Two channels avoid this problem entirely and should be preferred wherever
possible:

* **Head pose** — `pitch`/`yaw`/`roll` are explicit, in degrees, outside `exp`.
* **Eyelids** — LivePortrait ships a trained `retarget_eye` network. The
  renderer uses it. Never nudge "eyelid keypoints" by hand when a purpose-built
  network exists.

What is left — gaze direction and brow movement — has no such escape hatch, and
that is what lives here.

Every entry carries a `verified` flag. `False` means the index and sign are a
**starting hypothesis for calibration, not a measurement**, and the amplitude
is set very low so that a wrong guess produces an ineffective avatar rather
than a distorted one. Run `tools/calibrate_expression.py` to sweep a channel
and inspect the rendered result, then record what you actually observed.

Failing subtly is the point: an uncalibrated gaze axis that visibly warps the
face is worse than one that does nothing, because the brief ranks identity and
eye integrity above every other quality.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import torch

__all__ = ["ExpressionChannel", "CHANNELS", "apply_expression_deltas"]


@dataclass
class ExpressionChannel:
    """One semantic control mapped onto expression-space dimensions.

    `targets` is a list of `(keypoint_index, axis, weight)`. Axis is 0=x, 1=y,
    2=z in the model's canonical space.
    """

    name: str
    targets: list[tuple[int, int, float]]
    gain: float
    verified: bool
    note: str = ""


# ---------------------------------------------------------------------------
# MEASURED, 2026-09-02, on assets/master/master_v04_final.png.
#
# Method and raw numbers: tools/calibrate_expression.py, which drives every one
# of the 21 latent keypoints on each axis and reports where the face actually
# moved, followed by visual inspection of the eye region at 4x.
#
# What the measurement changed:
#
# * The inherited map spread gaze_x across indices 11, 13, 15 and 16 at equal
#   weight. Driven individually, **only 15 produces clean bilateral horizontal
#   iris movement**; 11 on this axis deforms the eyelid and 16 drags the brow.
#   Selectivity (effect in the eye box / mean effect elsewhere) rose from 1.4 to
#   4.2 by using 15 alone.
# * `brow_r` was mapped to index 9, which is inert: 1.31 mean change against
#   12-19 for the working brow dimensions. **The right brow channel did nothing
#   at all.**
# * Lateralisation was measured by splitting the brow band in half. Index 16 is
#   strongly biased to the subject's left brow (image-right/image-left ratio
#   0.19-0.27); indices 10 and 0 to the subject's right (ratio 3.0-5.1).
#
# The two brows are not equally expressive in this latent. The subject-left
# dimension is roughly 6x stronger than the best subject-right pair at the same
# amplitude. `brow_r`'s gain is scaled up to compensate, but not all the way -
# pushing it further starts to distort. A small residual imbalance remains and
# is left in deliberately: it reads as facial asymmetry, which is wanted.
#
# Amplitudes are set against the behaviour engine's own scale, where
# `saccade_max_amplitude = 0.42` is a large voluntary gaze shift and
# `microsaccade_amplitude = 0.018` must stay near perceptual threshold.
# ---------------------------------------------------------------------------
CHANNELS: dict[str, ExpressionChannel] = {
    "gaze_x": ExpressionChannel(
        name="gaze_x",
        targets=[(15, 0, 1.0)],
        gain=0.043,
        verified=True,
        note="horizontal iris. index 15 alone; 11/13/16 added distortion, not "
             "gaze. 0.018 latent = eyes near their natural lateral limit, so "
             "gain maps gaze_x=+-0.42 onto that limit",
    ),
    "gaze_y": ExpressionChannel(
        name="gaze_y",
        targets=[(13, 1, 1.0)],
        gain=0.024,
        verified=True,
        note="vertical iris. couples with lid aperture, which is anatomically "
             "correct - lids follow vertical gaze - but caps how far this can "
             "be driven before the eye reads as widened rather than raised",
    ),
    "brow_l": ExpressionChannel(
        name="brow_l",
        targets=[(16, 1, 1.0)],
        gain=0.012,
        verified=True,
        note="subject's LEFT brow. strongest brow dimension in the latent",
    ),
    "brow_r": ExpressionChannel(
        name="brow_r",
        targets=[(10, 1, 1.0), (0, 1, 0.9)],
        gain=0.036,
        verified=True,
        note="subject's RIGHT brow. weaker pair, gain scaled ~3x to partly "
             "compensate; a residual left/right imbalance is left in as "
             "facial asymmetry",
    ),
    "brow_furrow": ExpressionChannel(
        name="brow_furrow",
        targets=[(16, 1, -0.55), (10, 1, -0.55), (0, 1, -0.5)],
        gain=0.014,
        verified=False,
        note="both brows driven down together. the lowering is measured; "
             "whether this reads as a medial *pull* rather than a flat drop "
             "has not been confirmed at expressive amplitude",
    ),
    "smile": ExpressionChannel(
        name="smile",
        targets=[(17, 0, 1.0), (14, 1, -0.64)],
        gain=0.026,
        verified=True,
        note="symmetric smile. NO single dimension lifts the corners - the scan "
             "found the mouth dims either open/close it (19, 20 axis 1) or "
             "shear it sideways (17, 20 axis 0). A smile needs a combination, "
             "and this pair was chosen by rendering candidates and looking. "
             "Driving it also produces the cheek lift and lower-lid narrowing "
             "on its own, because the model learned faces holistically - which "
             "is why cheek and squint are deliberately NOT mapped separately",
    ),
    "smile_asym": ExpressionChannel(
        name="smile_asym",
        targets=[(20, 0, 1.0)],
        gain=0.010,
        verified=True,
        note="lateral pull, carrying the left/right difference so one corner "
             "can lead. Measured as the most laterally-biased mouth dimension",
    ),
    "mouth_open": ExpressionChannel(
        name="mouth_open",
        targets=[(20, 1, 1.0), (17, 1, 0.85), (19, 1, 0.6)],
        gain=0.010,
        verified=False,
        note="reserved for lip-sync, unused while mouth_open stays 0. indices "
             "are the three most mouth-selective on axis 1 (27.1, 18.9, 17.0) "
             "but sign and articulation quality are uncalibrated",
    ),
}


def apply_expression_deltas(exp: torch.Tensor, pose) -> torch.Tensor:
    """Add semantic deltas to a source expression tensor.

    `exp` is (1, 21, 3) and is modified in place and returned.

    Only channels with a non-negligible pose value are touched, so an
    uncalibrated channel that the behaviour engine leaves at zero costs
    nothing and changes nothing.
    """
    # The two mouth corners are carried as a symmetric part and a difference.
    #
    # Mapping each corner to its own dimension is not possible here: the latent
    # has no per-corner control, and the pair that produces a smile is
    # bilateral. So the mean drives the smile and the difference drives a
    # lateral pull, which is what makes one corner lead without the asymmetry
    # cancelling itself out.
    smile = 0.5 * (pose.mouth_corner_l + pose.mouth_corner_r)
    values = {
        "gaze_x": pose.gaze_x,
        "gaze_y": pose.gaze_y,
        "brow_l": pose.brow_l,
        "brow_r": pose.brow_r,
        "brow_furrow": pose.brow_furrow,
        "smile": smile,
        "smile_asym": pose.mouth_corner_l - pose.mouth_corner_r,
        "mouth_open": pose.mouth_open,
    }

    for name, value in values.items():
        if abs(value) < 1e-4:
            continue
        channel = CHANNELS.get(name)
        if channel is None:
            continue
        amount = value * channel.gain
        for index, axis, weight in channel.targets:
            exp[0, index, axis] += amount * weight

    return exp


def calibration_report() -> str:
    """Human-readable status, surfaced in the debug overlay and the README."""
    lines = ["expression channel calibration:"]
    for channel in CHANNELS.values():
        mark = "VERIFIED" if channel.verified else "UNVERIFIED"
        lines.append(
            f"  [{mark:10}] {channel.name:12} gain={channel.gain:.4f}  {channel.note}"
        )
    return "\n".join(lines)
