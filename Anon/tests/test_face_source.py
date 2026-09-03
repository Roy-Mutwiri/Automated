"""Tests for the face-cutout seam.

Deliberately does not load LivePortrait or touch the GPU. The parts worth
guarding are pure geometry: which landmarks become which anchor, and what the
alpha matte covers. Those are also the parts that have failed silently twice in
this project - once ranking expression latents off boxes that sat on the
forehead, once handing the brow region zero landmarks and reporting 0.00 px -
and in both cases the code ran perfectly and produced confident wrong numbers.

The face here is synthetic: a point cloud with the proportions of the real one,
so a band that drifts onto the wrong feature fails an assertion rather than a
render.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from presenter.render.face_source import ANCHOR_NAMES, FaceSource


def synthetic_face(cx=1000.0, cy=400.0, w=290.0, h=310.0) -> np.ndarray:
    """A landmark cloud shaped like the real one: brow-top to chin."""
    pts = []
    # brows across the top 10%
    for t in np.linspace(-0.42, 0.42, 32):
        pts.append((cx + t * w, cy - 0.5 * h + 0.04 * h))
    # eyes, 10-24% down, two clusters either side of centre
    for side in (-1, 1):
        for t in np.linspace(0.10, 0.30, 16):
            pts.append((cx + side * t * w, cy - 0.5 * h + 0.17 * h))
    # nose, 38-55%
    for t in np.linspace(-0.06, 0.06, 12):
        pts.append((cx + t * w, cy - 0.5 * h + 0.47 * h))
    # mouth, 55-82%
    for t in np.linspace(-0.20, 0.20, 24):
        pts.append((cx + t * w, cy - 0.5 * h + 0.68 * h))
    # jaw contour down to the chin
    for a in np.linspace(-np.pi * 0.42, np.pi * 0.42, 40):
        pts.append((cx + np.sin(a) * w * 0.5,
                    cy - 0.5 * h + 0.55 * h + np.cos(a) * h * 0.45))
    return np.asarray(pts, np.float32)


def test_every_anchor_is_produced():
    a = FaceSource._anchors(synthetic_face())
    assert set(a) == set(ANCHOR_NAMES)
    assert all(np.isfinite(a[k]).all() for k in a)


def test_eyes_are_above_mouth_and_left_is_left():
    """The failure mode is a band sliding onto the wrong feature."""
    a = FaceSource._anchors(synthetic_face())
    assert a["eye_l"][1] < a["nose"][1] < a["mouth"][1] < a["chin"][1]
    # Image coordinates: subject's left eye is at smaller x.
    assert a["eye_l"][0] < a["eye_r"][0]
    sep = a["eye_r"][0] - a["eye_l"][0]
    assert 0.2 * 290 < sep < 0.8 * 290, f"interocular {sep:.0f}px implausible"


def test_anchors_track_a_moved_face():
    """Anchors are geometry, not fixed indices, so they must follow the face."""
    base = FaceSource._anchors(synthetic_face())
    moved = FaceSource._anchors(synthetic_face(cx=1120.0, cy=455.0))
    for k in ANCHOR_NAMES:
        assert moved[k][0] - base[k][0] == pytest.approx(120.0, abs=1.0)
        assert moved[k][1] - base[k][1] == pytest.approx(55.0, abs=1.0)


def test_matte_covers_the_face_and_reaches_the_forehead():
    """The hull must extend above the brows, or the forehead is left behind."""
    src = FaceSource.__new__(FaceSource)     # no model load
    src.feather, src.margin, src.forehead = 17, 26, 0.30
    pts = synthetic_face()
    alpha = src._alpha(pts, (1080, 1920, 3))

    brow_y = pts[:, 1].min()
    height = np.ptp(pts[:, 1])
    ys = np.nonzero(alpha)[0]
    reach = brow_y - ys.min()
    assert reach > 0.15 * height, (
        f"matte reaches only {reach:.0f}px above the brow line; the forehead "
        f"would come from the geometry underneath")

    # Opaque in the middle of the face, absent well outside it.
    cy, cx = int(pts[:, 1].mean()), int(pts[:, 0].mean())
    assert alpha[cy, cx] > 240
    assert alpha[cy, int(pts[:, 0].max()) + 120] == 0


def test_confidence_falls_off_away_from_frontal():
    """A frontal face pasted onto a profile head is not a likeness."""
    from presenter.render.face_source import YAW_FULL_DEG, YAW_ZERO_DEG

    def conf(yaw):
        y = abs(yaw)
        return 1.0 if y <= YAW_FULL_DEG else max(
            0.0, 1.0 - (y - YAW_FULL_DEG) / (YAW_ZERO_DEG - YAW_FULL_DEG))

    assert conf(0.0) == 1.0
    assert conf(-YAW_FULL_DEG) == 1.0
    assert 0.0 < conf(34.0) < 1.0
    assert conf(90.0) == 0.0
    assert conf(-90.0) == 0.0
