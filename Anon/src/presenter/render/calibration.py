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
# UNVERIFIED hypotheses.
#
# Indices below follow the mapping used by community expression-editor tooling
# built on LivePortrait, which converged on keypoints 11, 13, 15 and 16 as
# eye-region and 1, 2, 9 as brow-region. That is circumstantial evidence from
# third-party tools, not a measurement on this checkpoint, so every gain is
# deliberately conservative and every entry is marked unverified.
# ---------------------------------------------------------------------------
CHANNELS: dict[str, ExpressionChannel] = {
    "gaze_x": ExpressionChannel(
        name="gaze_x",
        targets=[(11, 0, 1.0), (13, 0, 1.0), (15, 0, 1.0), (16, 0, 1.0)],
        gain=0.004,
        verified=False,
        note="horizontal iris movement; sign and index need calibration",
    ),
    "gaze_y": ExpressionChannel(
        name="gaze_y",
        targets=[(11, 1, 1.0), (13, 1, 1.0), (15, 1, 1.0), (16, 1, 1.0)],
        gain=0.003,
        verified=False,
        note="vertical iris movement; couples with eyelid, calibrate together",
    ),
    "brow_l": ExpressionChannel(
        name="brow_l",
        targets=[(1, 1, 1.0), (2, 1, 0.6)],
        gain=0.004,
        verified=False,
        note="left brow raise",
    ),
    "brow_r": ExpressionChannel(
        name="brow_r",
        targets=[(9, 1, 1.0), (2, 1, 0.6)],
        gain=0.004,
        verified=False,
        note="right brow raise",
    ),
    "brow_furrow": ExpressionChannel(
        name="brow_furrow",
        targets=[(1, 0, -0.5), (9, 0, 0.5), (2, 1, -0.8)],
        gain=0.003,
        verified=False,
        note="medial brow pull",
    ),
    "mouth_open": ExpressionChannel(
        name="mouth_open",
        targets=[(14, 1, -1.0), (17, 1, -1.0), (19, 1, -0.6), (20, 1, -0.6)],
        gain=0.006,
        verified=False,
        note="reserved for lip-sync; unused while mouth_open stays 0",
    ),
}


def apply_expression_deltas(exp: torch.Tensor, pose) -> torch.Tensor:
    """Add semantic deltas to a source expression tensor.

    `exp` is (1, 21, 3) and is modified in place and returned.

    Only channels with a non-negligible pose value are touched, so an
    uncalibrated channel that the behaviour engine leaves at zero costs
    nothing and changes nothing.
    """
    values = {
        "gaze_x": pose.gaze_x,
        "gaze_y": pose.gaze_y,
        "brow_l": pose.brow_l,
        "brow_r": pose.brow_r,
        "brow_furrow": pose.brow_furrow,
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
