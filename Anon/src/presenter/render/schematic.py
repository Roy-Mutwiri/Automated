"""A diagnostic rig preview. Explicitly NOT the photoreal deliverable.

This renderer exists to answer one question that statistics cannot: *does the
behaviour look right?* The 30-minute audit proves the event distributions are
human-shaped, but a distribution that passes every test can still animate a
face that reads as wrong, and the only way to find that out is to watch it.

Drawing the rig schematically rather than photorealistically is deliberate.
Blink kinematics, saccade amplitude and breathing depth are all far easier to
judge against clean geometry than against a photograph, where skin texture and
lighting hide exactly the timing errors being looked for. Tuning happens here;
the photoreal backend then consumes the same, already-validated pose stream.

Nothing in this file feeds the final output path.
"""

from __future__ import annotations

import math

import cv2
import numpy as np

from ..types import AvatarPose
from .base import RendererInfo

__all__ = ["SchematicRenderer"]

_BG = (26, 24, 22)
_SKIN = (150, 168, 188)
_LINE = (90, 104, 122)
_SCLERA = (232, 236, 240)
_IRIS = (92, 74, 58)
_PUPIL = (18, 16, 15)
_BROW = (58, 62, 76)
_MOUTH = (96, 92, 128)
_ACCENT = (120, 190, 235)


class SchematicRenderer:
    """Draws a wireframe head driven by an `AvatarPose`."""

    def __init__(self, width: int = 1280, height: int = 720) -> None:
        self.width = width
        self.height = height
        self._info = RendererInfo(
            name="schematic",
            resolution=(width, height),
            device="cpu",
            photoreal=False,
            notes="rig preview for behaviour tuning; not the delivery renderer",
        )

    @property
    def info(self) -> RendererInfo:
        return self._info

    # -- geometry helpers ---------------------------------------------------
    def _project(
        self, x: float, y: float, z: float, pose: AvatarPose, cx: float, cy: float,
        scale: float,
    ) -> tuple[int, int]:
        """Rotate a head-local point and project it with a weak perspective.

        A real head is not a billboard: yawing it moves the near eye outward
        and the far eye inward, and that parallax is a large part of why a
        rotation reads as a head turning rather than an image shearing. A weak
        perspective divide is enough to get that cue at negligible cost.
        """
        yaw = math.radians(pose.yaw)
        pitch = math.radians(-pose.pitch)
        roll = math.radians(pose.roll)

        # Y (yaw), then X (pitch), then Z (roll).
        cos_y, sin_y = math.cos(yaw), math.sin(yaw)
        x1 = x * cos_y + z * sin_y
        z1 = -x * sin_y + z * cos_y

        cos_p, sin_p = math.cos(pitch), math.sin(pitch)
        y2 = y * cos_p - z1 * sin_p
        z2 = y * sin_p + z1 * cos_p

        cos_r, sin_r = math.cos(roll), math.sin(roll)
        x3 = x1 * cos_r - y2 * sin_r
        y3 = x1 * sin_r + y2 * cos_r

        # Weak perspective: camera at a portrait-lens distance so the
        # foreshortening is gentle, matching an 85mm-equivalent framing rather
        # than a wide-angle caricature.
        camera_distance = 7.5
        persp = camera_distance / max(camera_distance - z2, 0.35)

        px = cx + x3 * scale * persp
        py = cy - y3 * scale * persp
        return int(round(px)), int(round(py))

    # -- rendering ----------------------------------------------------------
    def render(self, pose: AvatarPose) -> np.ndarray:
        frame = np.full((self.height, self.width, 3), _BG, dtype=np.uint8)

        base = min(self.width, self.height) * 0.30
        scale = base * pose.scale
        cx = self.width * 0.5 + pose.tx * self.width
        cy = self.height * 0.52 - pose.ty * self.height

        P = lambda x, y, z=0.0: self._project(x, y, z, pose, cx, cy, scale)

        # -- head outline: an ellipse sampled in head-local space so it rotates
        # correctly rather than being drawn flat and then sheared.
        outline = []
        for i in range(72):
            a = 2.0 * math.pi * i / 72
            hx = 0.78 * math.cos(a)
            hy = 1.0 * math.sin(a)
            hz = 0.30 * math.cos(a)  # cheeks recede
            outline.append(P(hx, hy, hz))
        cv2.fillPoly(frame, [np.array(outline, np.int32)], _SKIN)
        cv2.polylines(frame, [np.array(outline, np.int32)], True, _LINE, 2, cv2.LINE_AA)

        # -- neck and shoulders. Breathing shows most clearly here, and having
        # them present stops the head reading as a floating object.
        shoulder_y = -1.55
        shoulders = [
            P(-2.05, shoulder_y - 0.55, -0.25), P(-1.15, shoulder_y + 0.10, 0.0),
            P(-0.42, shoulder_y + 0.52, 0.18), P(0.42, shoulder_y + 0.52, 0.18),
            P(1.15, shoulder_y + 0.10, 0.0), P(2.05, shoulder_y - 0.55, -0.25),
            P(2.05, shoulder_y - 1.30, -0.25), P(-2.05, shoulder_y - 1.30, -0.25),
        ]
        cv2.fillPoly(frame, [np.array(shoulders, np.int32)], (58, 62, 70))

        self._draw_eyes(frame, pose, P, scale)
        self._draw_brows(frame, pose, P, scale)
        self._draw_mouth(frame, pose, P, scale)

        return frame

    def _draw_eyes(self, frame, pose: AvatarPose, P, scale: float) -> None:
        for side, eye_open in (("l", pose.eye_open_l), ("r", pose.eye_open_r)):
            sign = -1.0 if side == "l" else 1.0
            ex = sign * 0.30
            ey = 0.16
            ez = 0.20

            half_w = 0.155
            half_h = 0.082

            centre = P(ex, ey, ez)
            corner_in = P(ex - sign * half_w, ey, ez * 0.8)
            corner_out = P(ex + sign * half_w, ey, ez * 0.8)
            eye_w = max(abs(corner_out[0] - corner_in[0]) * 0.5, 3)
            eye_h = max(half_h * scale, 3)

            squint = pose.squint_l if side == "l" else pose.squint_r
            # The lower lid rises under squint; the upper lid carries the blink.
            aperture = max(0.0, eye_open - 0.55 * squint)

            cv2.ellipse(frame, centre, (int(eye_w), int(eye_h)), 0, 0, 360,
                        _SCLERA, -1, cv2.LINE_AA)

            # Iris follows gaze. Clamped inside the sclera so it can never
            # detach from the eye, which is the artefact that makes synthetic
            # eyes read as pasted on.
            gx = pose.gaze_x * eye_w * 0.62
            gy = -pose.gaze_y * eye_h * 0.72
            iris_r = max(int(eye_h * 0.88), 3)
            ic = (int(centre[0] + gx), int(centre[1] + gy))
            cv2.circle(frame, ic, iris_r, _IRIS, -1, cv2.LINE_AA)
            cv2.circle(frame, ic, max(int(iris_r * 0.42), 1), _PUPIL, -1, cv2.LINE_AA)
            # Catchlight. Fixed relative to the eye, not the iris: a specular
            # highlight comes from the light source and must not slide around
            # with gaze, which is a classic tell in generated faces.
            cv2.circle(frame, (int(centre[0] - eye_w * 0.22), int(centre[1] - eye_h * 0.38)),
                       max(int(iris_r * 0.22), 1), (250, 250, 250), -1, cv2.LINE_AA)

            # Eyelid occlusion, drawn as a skin-coloured cap descending from
            # the top of the eye.
            lid_drop = (1.0 - aperture) * (eye_h * 2.0)
            if lid_drop > 0.5:
                top = int(centre[1] - eye_h - 2)
                bottom = int(centre[1] - eye_h + lid_drop)
                cv2.rectangle(
                    frame,
                    (int(centre[0] - eye_w - 2), top),
                    (int(centre[0] + eye_w + 2), bottom),
                    _SKIN, -1,
                )
                cv2.line(frame, (int(centre[0] - eye_w), bottom),
                         (int(centre[0] + eye_w), bottom), _LINE, 2, cv2.LINE_AA)

            cv2.ellipse(frame, centre, (int(eye_w), int(eye_h)), 0, 0, 360,
                        _LINE, 1, cv2.LINE_AA)

    def _draw_brows(self, frame, pose: AvatarPose, P, scale: float) -> None:
        for side, lift in (("l", pose.brow_l), ("r", pose.brow_r)):
            sign = -1.0 if side == "l" else 1.0
            raise_amt = lift * 0.12
            furrow = pose.brow_furrow * 0.05
            inner = P(sign * (0.14 + furrow), 0.35 + raise_amt - furrow * 0.4, 0.20)
            mid = P(sign * 0.30, 0.40 + raise_amt * 1.15, 0.22)
            outer = P(sign * 0.46, 0.355 + raise_amt * 0.8, 0.16)
            cv2.polylines(frame, [np.array([inner, mid, outer], np.int32)],
                          False, _BROW, max(int(scale * 0.035), 2), cv2.LINE_AA)

    def _draw_mouth(self, frame, pose: AvatarPose, P, scale: float) -> None:
        open_amt = pose.mouth_open * 0.10 + pose.jaw * 0.03
        cl = pose.mouth_corner_l * 0.05
        cr = pose.mouth_corner_r * 0.05
        left = P(-0.24, -0.44 + cl, 0.14)
        right = P(0.24, -0.44 + cr, 0.14)
        top = P(0.0, -0.42 - open_amt * 0.35, 0.20)
        bottom = P(0.0, -0.46 - open_amt, 0.20)
        pts = np.array([left, top, right, bottom], np.int32)
        if open_amt > 0.012:
            cv2.fillPoly(frame, [pts], (38, 30, 34))
        cv2.polylines(frame, [pts], True, _MOUTH, 2, cv2.LINE_AA)

        nose = [P(0.0, 0.02, 0.42), P(-0.08, -0.20, 0.30), P(0.08, -0.20, 0.30)]
        cv2.polylines(frame, [np.array(nose, np.int32)], False, _LINE, 2, cv2.LINE_AA)

    def close(self) -> None:
        pass
