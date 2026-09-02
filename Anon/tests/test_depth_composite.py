"""Tests for the depth compositor's contract.

These are deliberately synthetic and Blender-free. The compositor's job is a
per-pixel comparison, and a comparison is exactly the kind of thing that can be
wrong in a way no single rendered frame reveals: a constant ordering bug draws
a plausible picture every time.

Two of these tests exist because of failures that actually happened rather than
failures that were imagined. `test_all_far_room_is_rejected` encodes the
OpenEXR incident - cv2 returned None for a depth file, the buffer became
all-infinity, and the occlusion self-test passed while comparing nothing
against nothing. `test_alpha_zero_never_contributes` encodes the fact that a
Gaussian renderer emits depth for pixels it did not cover.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))

from depth_composite import (  # noqa: E402
    ALPHA_EPS,
    FAR,
    composite,
    validate_depth,
)

H, W = 64, 256
ROOM_COLOUR = (0, 0, 0)
HUMAN_COLOUR = (200, 100, 50)


def room(depth_m=2.0):
    """A flat wall at a constant distance, painted black."""
    rgb = np.zeros((H, W, 3), np.uint8)
    rgb[:] = ROOM_COLOUR
    return rgb, np.full((H, W), depth_m, np.float32)


def human(depth, alpha=255):
    """A human layer with the given depth map and uniform or per-pixel alpha."""
    rgba = np.zeros((H, W, 4), np.uint8)
    rgba[..., :3] = HUMAN_COLOUR
    rgba[..., 3] = alpha
    return rgba, np.asarray(depth, np.float32)


# --------------------------------------------------------------------------
# The contract


def test_nan_depth_is_rejected():
    d = np.full((H, W), 2.0, np.float32)
    d[0, 0] = np.nan
    with pytest.raises(ValueError, match="NaN"):
        validate_depth(d, "probe")


@pytest.mark.parametrize("bad", [0.0, -1.0, -1e-6])
def test_zero_and_negative_depth_are_rejected(bad):
    # These do not merely look wrong, they invert the result: a depth <= 0 wins
    # every comparison and draws that surface in front of the entire room.
    d = np.full((H, W), 2.0, np.float32)
    d[10, 10] = bad
    with pytest.raises(ValueError, match="<= 0"):
        validate_depth(d, "probe")


def test_all_far_room_is_rejected():
    """The vacuous pass, as a test.

    An all-FAR buffer is what an unread depth image looks like. Every occlusion
    test against it succeeds, because nothing is ever compared.
    """
    with pytest.raises(ValueError, match="no geometry"):
        validate_depth(np.full((H, W), FAR, np.float32), "room depth")

    # ...and it must be rejected through the compositor too, not just directly.
    rgb = np.zeros((H, W, 3), np.uint8)
    rgba, hd = human(np.full((H, W), 1.0))
    with pytest.raises(ValueError, match="no geometry"):
        composite(rgb, np.full((H, W), FAR, np.float32), rgba, hd)


def test_shape_mismatch_is_rejected():
    rgb, rd = room()
    rgba, hd = human(np.full((H, W), 1.0))
    with pytest.raises(ValueError, match="shape mismatch"):
        composite(rgb, rd, rgba, hd[:, :10])


# --------------------------------------------------------------------------
# Depth order reversal


def test_depth_order_reverses_across_the_image():
    """A human who crosses from in front of the wall to behind it.

    A compositor with a constant ordering bug - always human, always room, or a
    flipped comparison - cannot produce a crossover in the right place. Testing
    only a human standing in front would pass on all three.
    """
    rgb, rd = room(2.0)
    ramp = np.linspace(1.0, 3.0, W, dtype=np.float32)
    ramp = np.broadcast_to(ramp, (H, W)).copy()
    rgba, hd = human(ramp)

    out, _ = composite(rgb, rd, rgba, hd)

    crossover = int(np.argmin(np.abs(ramp[0] - 2.0)))
    human_px = np.all(out == np.array(HUMAN_COLOUR, np.uint8), axis=-1)
    room_px = np.all(out == np.array(ROOM_COLOUR, np.uint8), axis=-1)

    # Both outcomes must actually occur, or the test proves nothing.
    assert human_px.any(), "the human never wins anywhere"
    assert room_px.any(), "the room never wins anywhere"

    # Near side is the human's, far side is the room's, with the boundary where
    # the ramp crosses the wall.
    assert human_px[:, : crossover - 1].all()
    assert room_px[:, crossover + 2 :].all()

    boundary = int(np.argmax(~human_px[0]))
    assert abs(boundary - crossover) <= 1, (
        f"crossover at column {boundary}, expected {crossover}")


def test_equal_depth_gives_the_room_the_pixel():
    """A tie is not a win.

    Coplanar surfaces are a coin toss physically, but the choice must be
    consistent: `<` rather than `<=` keeps a human exactly level with the wall
    from punching through it.
    """
    rgb, rd = room(2.0)
    rgba, hd = human(np.full((H, W), 2.0))
    out, _ = composite(rgb, rd, rgba, hd)
    assert np.all(out == np.array(ROOM_COLOUR, np.uint8))


# --------------------------------------------------------------------------
# Alpha and depth together


def test_alpha_zero_never_contributes():
    """Coverage beats depth.

    A Gaussian renderer emits a depth value for pixels it never covered. Here
    the whole layer claims to be 1 m in front of a 2 m wall - the only thing
    keeping the human out of the empty band is its alpha.
    """
    rgb, rd = room(2.0)
    alpha = np.zeros((H, W), np.uint8)
    alpha[:, : W // 2] = 255
    rgba, hd = human(np.full((H, W), 1.0), alpha=alpha)

    out, _ = composite(rgb, rd, rgba, hd)

    assert np.all(out[:, : W // 2] == np.array(HUMAN_COLOUR, np.uint8))
    assert np.all(out[:, W // 2 :] == np.array(ROOM_COLOUR, np.uint8)), (
        "the human drew into pixels he does not cover")


def test_semi_transparent_edge_blends_with_the_room():
    """The silhouette is a blend, not a hard choice.

    Gaussian edges are genuinely partial coverage. Rounding them to a binary
    decision is what produces a cut-out look along hair and shoulders.
    """
    rgb, rd = room(2.0)
    alpha = np.full((H, W), 128, np.uint8)
    rgba, hd = human(np.full((H, W), 1.0), alpha=alpha)

    out, _ = composite(rgb, rd, rgba, hd)

    expected = np.array(HUMAN_COLOUR, np.float32) * (128 / 255.0)
    assert np.allclose(out[0, 0], expected, atol=1.5), out[0, 0]


def test_alpha_below_epsilon_is_treated_as_empty():
    rgb, rd = room(2.0)
    alpha = np.full((H, W), max(int(ALPHA_EPS * 255) - 1, 0), np.uint8)
    rgba, hd = human(np.full((H, W), 1.0), alpha=alpha)
    out, _ = composite(rgb, rd, rgba, hd)
    assert np.all(out == np.array(ROOM_COLOUR, np.uint8))


# --------------------------------------------------------------------------
# Metres, not normalised units


def test_depth_stays_metric():
    """Comparison happens in camera-space metres on both sides.

    If either side were normalised to 0-1, a human at 1 m against a wall at
    2 m would still order correctly by luck, but the same human against a wall
    at 40 m would not. Scaling only the room must therefore change the result.
    """
    rgb, rd = room(2.0)
    rgba, hd = human(np.full((H, W), 1.0))
    near, _ = composite(rgb, rd, rgba, hd)
    assert np.all(near == np.array(HUMAN_COLOUR, np.uint8))

    # Same human, wall moved closer than he is. Nothing else changes.
    _, rd_close = room(0.5)
    far, _ = composite(rgb, rd_close, rgba, hd)
    assert np.all(far == np.array(ROOM_COLOUR, np.uint8)), (
        "moving the wall in metres did not change the outcome - one side is "
        "probably normalised")
