"""The camera system must work without a human renderer, and must not lie.

Two failures are being locked out here. The first is the coupling that started
this: the rig was created inside the LivePortrait branch, so with any other
renderer there were no cameras, no buttons and no working keys - and that read
as broken camera switching rather than a feature that had never been turned on.

The second is subtler and worse. The legacy 2D rig's cam1/cam2/cam3 are three
crops of a single photograph. Labelling those "LEFT" and "RIGHT" would promise
a viewpoint the pixels do not contain, and the preview exists precisely to tell
real camera moves from reframings.
"""

from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from presenter.render.camera_manager import CameraManager  # noqa: E402
from presenter.render.camera_preview import CameraPreviewRenderer  # noqa: E402


@pytest.fixture
def physical():
    return CameraManager.load("config/cameras.yaml")


def test_loads_without_any_renderer(physical):
    """No renderer, no GPU, no avatar model - the rig still loads."""
    assert len(physical.views) == 7
    assert physical.keys() == [f"cam{i}" for i in range(1, 8)]
    assert physical.physical


def test_disabled_cameras_are_still_selectable(physical):
    """`enabled: false` blocks a production shot, not looking at where it points.

    Cameras 4-7 are blocked as deliverables. Hiding them from the preview would
    make the camera plan unreviewable, which is the one thing preview is for.
    """
    blocked = [v.key for v in physical.ordered() if not v.enabled]
    assert blocked == ["cam4", "cam5", "cam6", "cam7"]
    for key in blocked:
        assert physical.select(key)
        assert physical.current == key


def test_number_keys_map_to_their_own_camera(physical):
    for n in range(1, 8):
        assert physical.by_number(n) == f"cam{n}"
    assert physical.by_number(9) is None


def test_step_wraps_both_ways(physical):
    physical.select("cam1")
    assert physical.step(-1) == "cam7"
    physical.select("cam7")
    assert physical.step(1) == "cam1"


def test_describe_reports_the_real_lens_and_position(physical):
    physical.select("cam6")
    text = " ".join(physical.describe())
    assert "CAM6" in text and "HIGH DIAGONAL" in text
    assert "28 mm" in text
    # Aim is derived from position and look_at, so it cannot drift out of step
    # with the transform the renderer uses.
    assert "pitch" in text and "yaw" in text


def test_legacy_2d_rig_is_never_called_left_or_right():
    """The honesty test.

    cam2 and cam3 in the legacy rig are punch-ins of cam1's photograph. If this
    ever starts describing them as viewpoints, the preview has begun lying
    about what a camera cut actually contains.
    """
    legacy = CameraManager.load("config/cameras_2d_legacy.yaml")
    assert not legacy.physical
    for key in ("cam1", "cam2", "cam3"):
        intent = legacy.view(key).intent
        assert "LEGACY 2D" in intent, intent
        assert "LEFT" not in intent.upper()
        assert "RIGHT" not in intent.upper()

    legacy.select("cam2")
    described = " ".join(legacy.describe())
    assert "not a viewpoint" in described
    # A crop has no lens of its own; claiming one would be an invented fact.
    assert "FOCAL" not in described


# --------------------------------------------------------------------------
# The preview renderer


def test_seven_cameras_give_seven_different_frames(physical):
    """The acceptance test: cam2/cam3 must not be cam1 again.

    Compared as whole images. Two physically distinct viewpoints of the same
    room cannot be near-identical, so equality here would mean the switch is
    not reaching the renderer.
    """
    if physical.missing_previews():
        pytest.skip("run tools/render_camera_previews.py first")

    r = CameraPreviewRenderer(physical, 640, 360)
    frames = {}
    for key in physical.keys():
        physical.select(key)
        frames[key] = r.render(None)

    keys = list(frames)
    for i, a in enumerate(keys):
        for b in keys[i + 1:]:
            diff = float(np.abs(frames[a].astype(np.int16)
                                - frames[b].astype(np.int16)).mean())
            assert diff > 2.0, f"{a} and {b} are nearly identical ({diff:.2f})"


def test_missing_preview_never_substitutes_another_camera(tmp_path, physical):
    """A placeholder, not somebody else's shot.

    Showing cam1 when cam5 was asked for would look entirely plausible and
    answer the wrong question - the exact failure mode of every silent
    fallback in this project so far.
    """
    # cam1 gets a real image written here rather than whichever PNGs happen to
    # be on disk. renders/camera_preview/ is gitignored, so on a fresh worktree
    # every camera returns a placeholder and the old version of this test
    # compared one placeholder's variance against another's - it passed here
    # and failed for anyone who had not rendered yet, which is a test measuring
    # the machine instead of the code.
    stand_in = tmp_path / "cam1.png"
    rng = np.random.default_rng(0)
    cv2.imwrite(str(stand_in), rng.integers(0, 255, (360, 640, 3), dtype=np.uint8))
    physical.views["cam1"].preview = stand_in
    physical.views["cam5"].preview = tmp_path / "does_not_exist.png"

    r = CameraPreviewRenderer(physical, 640, 360)
    physical.select("cam1")
    cam1 = r.render(None)
    physical.select("cam5")
    cam5 = r.render(None)

    assert not np.array_equal(cam1, cam5)
    # cam1 is the noise image that was just written; cam5 must be the flat
    # placeholder card, not a copy of it.
    assert cam5.std() < cam1.std()
    assert cam1.std() > 20.0, "the stand-in image did not load"


def test_render_ignores_pose(physical):
    """Stills of one frozen timestamp; a pose must not appear to drive them."""
    if physical.missing_previews():
        pytest.skip("run tools/render_camera_previews.py first")
    r = CameraPreviewRenderer(physical, 320, 180)
    physical.select("cam3")
    assert np.array_equal(r.render(None), r.render(object()))
