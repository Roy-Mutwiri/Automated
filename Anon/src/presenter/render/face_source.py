"""The photoreal face, as an RGBA cutout with the landmarks it is registered to.

This is the seam between the two halves of the presenter. The motion side owns
the face - the master frame, the LivePortrait drive, the expression calibration
- and the camera side owns the body, the room, the seven cameras and the
compositing. Neither needs to import the other's internals: this hands over an
image and the points it should be pinned to, and nothing else.

    src = FaceSource("assets/master/master_v04_final.png")
    f = src.frame(pose)
    f.rgba        # H x W x 4 uint8, straight alpha, face only
    f.anchors     # {"eye_l": (x, y), "eye_r": ..., "mouth": ..., ...}
    f.landmarks   # 203 x 2, in f.rgba's own pixel coordinates

## Coordinate and pose conventions

Angles are `AvatarPose`'s, unchanged, and the definitions live there:

* `yaw`   - **+ turns the head to the subject's right** (viewer's left)
* `pitch` - **+ is up**
* `roll`  - **+ is clockwise from the viewer's side**

Applied as intrinsic yaw, then pitch, then roll. Degrees, absolute rather than
incremental - the engine emits a whole pose every frame precisely so that a
dropped frame cannot accumulate error.

Anchor coordinates are pixels in the returned cutout, origin top-left, x right
and y **down**, which is image convention and not the world's. A consumer
projecting 3D landmarks into a camera gets the same handedness from OpenCV or
from Blender's `world_to_camera_view` after the usual y flip.

## What the anchors are, and what they are not

Registration should use `eye_l`, `eye_r` and `mouth`. Those three points define
a similarity transform - translation, rotation, uniform scale - which is the
standard and stable way to align a face, and they are the landmarks this model
localises best.

**There are no ear anchors, and there is no skull anchor.** The landmark model
returns 203 points covering the face oval, brows, eyes, nose and mouth: brow
line down to chin, jaw contour at the sides. Ears, hairline and cranium are
outside it. A consumer that wants to pin ear-to-ear is asking for something this
cannot supply, and the honest answer is to register on the eye/mouth triangle
and accept that the silhouette comes from the geometry underneath.

`head_centre` is provided as the centroid of the face oval. It is a convenience,
not a skull centre, and it moves with expression.

## The hard limit

The master frame contains **one view of this man's face**. Everything here is
that view, re-posed. Within roughly +-20 degrees of frontal the re-pose is
convincing because LivePortrait is genuinely rotating a learned head. Past that
it is increasingly a frontal face on a shape that disagrees with it, and from
behind there is no face at all. This module cannot fix that and does not try:
`frame()` reports `confidence` from the requested yaw so a caller can fade it
out rather than composite something that has quietly stopped being a likeness.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

__all__ = ["FaceSource", "FaceFrame", "ANCHOR_NAMES"]

ANCHOR_NAMES = ("eye_l", "eye_r", "mouth", "nose", "chin", "head_centre")

# Beyond this the single frontal view stops being a likeness. Not a cliff: the
# reported confidence ramps from 1.0 to 0.0 between these two.
YAW_FULL_DEG = 20.0
YAW_ZERO_DEG = 48.0


@dataclass
class FaceFrame:
    """One rendered face, ready to composite."""

    rgba: np.ndarray                    # H x W x 4, uint8, straight alpha
    anchors: dict                       # name -> (x, y) in rgba pixels
    landmarks: np.ndarray               # 203 x 2, in rgba pixels
    origin: tuple                       # (x, y) of the cutout in the full plate
    confidence: float                   # 1.0 frontal -> 0.0 past the usable cone
    render_ms: float = 0.0
    full: np.ndarray | None = field(default=None, repr=False)


class FaceSource:
    """Renders the photoreal face and cuts it out with an alpha matte."""

    def __init__(self, source_image: str,
                 liveportrait_root: str = "third_party/LivePortrait",
                 head_box: tuple = (790, 60, 1330, 620),
                 feather: int = 17,
                 margin: int = 26,
                 forehead: float = 0.30) -> None:
        from .liveportrait import LivePortraitRenderer

        self.head_box = head_box
        self.feather = feather
        self.margin = margin
        # Fraction of face height the matte reaches above the brow line.
        self.forehead = forehead
        self._renderer = LivePortraitRenderer(
            source_image=source_image, liveportrait_root=liveportrait_root,
            output_size=(1920, 1080), framing="full", environment="source",
            neutralize_pose=0.0)

        root = Path(liveportrait_root).resolve()
        if str(root) not in sys.path:
            sys.path.insert(0, str(root))
        from src.utils.human_landmark_runner import LandmarkRunner
        self._lmk = LandmarkRunner(
            ckpt_path=str(root / "pretrained_weights/liveportrait/landmark.onnx"),
            onnx_provider="cpu", device_id=0)
        self._lmk.warmup()

    # -- landmarks ---------------------------------------------------------
    def _landmarks(self, frame: np.ndarray) -> np.ndarray:
        x0, y0, x1, y1 = self.head_box
        crop = cv2.cvtColor(frame[y0:y1, x0:x1], cv2.COLOR_BGR2RGB)
        s = 512.0 / max(crop.shape[:2])
        crop = cv2.resize(crop, (int(crop.shape[1] * s), int(crop.shape[0] * s)))
        pts = self._lmk.run(crop, self._lmk.run(crop)).astype(np.float32) / s
        pts[:, 0] += x0
        pts[:, 1] += y0
        return pts

    @staticmethod
    def _anchors(pts: np.ndarray) -> dict:
        """Named points, derived from the landmark cloud's own geometry.

        Bands rather than hard indices: the index layout of this model is
        undocumented, and a hard-coded index that silently points at the
        forehead is a mistake this project has already made twice. Geometry
        cannot drift that way - the topmost band of a face is its brows
        whatever the numbering says.
        """
        y0, h = pts[:, 1].min(), np.ptp(pts[:, 1])
        x_mid = (pts[:, 0].min() + pts[:, 0].max()) * 0.5

        lids = pts[(pts[:, 1] >= y0 + 0.10 * h) & (pts[:, 1] < y0 + 0.24 * h)]
        left = lids[lids[:, 0] < x_mid]
        right = lids[lids[:, 0] >= x_mid]
        mouth = pts[(pts[:, 1] >= y0 + 0.55 * h) & (pts[:, 1] < y0 + 0.82 * h)]
        nose = pts[(pts[:, 1] >= y0 + 0.38 * h) & (pts[:, 1] < y0 + 0.55 * h)]

        def mid(a, fallback):
            return (float(a[:, 0].mean()), float(a[:, 1].mean())) if len(a) else fallback

        centre = (float(pts[:, 0].mean()), float(pts[:, 1].mean()))
        return {
            "eye_l": mid(left, centre),
            "eye_r": mid(right, centre),
            "mouth": mid(mouth, centre),
            "nose": mid(nose, centre),
            "chin": (float(x_mid), float(pts[:, 1].max())),
            "head_centre": centre,
        }

    # -- matte -------------------------------------------------------------
    def _alpha(self, pts: np.ndarray, shape: tuple) -> np.ndarray:
        """Feathered convex hull of the face oval.

        The hull is the face, not the head: no hair, no ears, no cranium,
        because the landmark model does not see them. Feathering matters more
        than it looks - a hard edge on a composited face reads as a mask
        instantly, and a wide soft edge lets the geometry underneath carry the
        silhouette, which is exactly the division of labour intended here.
        """
        m = np.zeros(shape[:2], np.uint8)
        # The landmark cloud stops at the brow line, so a hull of it alone
        # hands the forehead back to whatever geometry is underneath - which,
        # for a mannequin, means a grey band between the eyebrows and the hair.
        # The master frame has those pixels, so the hull is extended upward.
        # Only as far as the hairline: past it the extension starts taking hair,
        # and hair pinned to an eye/mouth transform will not agree with the
        # silhouette it is sitting on.
        pts = np.asarray(pts, np.float32)
        top = pts[:, 1].min()
        height = np.ptp(pts[:, 1])
        brow = pts[pts[:, 1] < top + 0.12 * height].copy()
        brow[:, 1] -= self.forehead * height
        hull = cv2.convexHull(np.vstack([pts, brow]).astype(np.int32))
        cv2.fillConvexPoly(m, hull, 255)
        k = self.feather | 1
        m = cv2.erode(m, np.ones((k, k), np.uint8), iterations=1)
        return cv2.GaussianBlur(m, (0, 0), self.feather * 0.75)

    # -- public ------------------------------------------------------------
    def frame(self, pose, want_full: bool = False) -> FaceFrame:
        """Render one face for `pose` and return it as an RGBA cutout."""
        import time

        t0 = time.perf_counter()
        full = self._renderer.render(pose)
        render_ms = (time.perf_counter() - t0) * 1000.0

        pts = self._landmarks(full)
        alpha = self._alpha(pts, full.shape)

        # Crop to the *matte*, not to the landmarks. The matte reaches above the
        # brow line to take in the forehead, so a crop sized to the landmark
        # extent sliced that extension off and left a hard horizontal edge
        # across the forehead - the one thing the feathering exists to avoid.
        ys_nz, xs_nz = np.nonzero(alpha)
        x0 = max(int(xs_nz.min()) - self.margin, 0)
        y0 = max(int(ys_nz.min()) - self.margin, 0)
        x1 = min(int(xs_nz.max()) + self.margin + 1, full.shape[1])
        y1 = min(int(ys_nz.max()) + self.margin + 1, full.shape[0])

        rgba = np.dstack([full[y0:y1, x0:x1], alpha[y0:y1, x0:x1]])
        local = pts - np.array([x0, y0], np.float32)
        anchors = {k: (v[0] - x0, v[1] - y0)
                   for k, v in self._anchors(pts).items()}

        yaw = abs(float(getattr(pose, "yaw", 0.0)))
        conf = 1.0 if yaw <= YAW_FULL_DEG else max(
            0.0, 1.0 - (yaw - YAW_FULL_DEG) / (YAW_ZERO_DEG - YAW_FULL_DEG))

        return FaceFrame(rgba=rgba, anchors=anchors, landmarks=local,
                         origin=(x0, y0), confidence=conf,
                         render_ms=render_ms,
                         full=full if want_full else None)
