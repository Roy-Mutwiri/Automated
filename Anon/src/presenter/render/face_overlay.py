"""Register the photoreal face onto the 3D head, and fade it where it lies.

The 3D world has the body, the room and seven real cameras but a mannequin
face. The master frame has the real man but only one view of him. This puts the
second onto the first, through the camera that is actually looking.

## The gate is geometric, and that is the point

`FaceSource` reports a confidence from the head's own yaw - how far he has
turned away from the plate's viewpoint. That is necessary but not sufficient
here, because a head at yaw 0 is still turned ~40 degrees away from cam2. The
angle that matters is between the direction his face points in world space and
the direction to the camera doing the looking, and it is computed per camera.

The consequence is deliberate and worth stating plainly: cam1 gets the real
man, cam2 and cam3 get a partial fade, and cam4-7 get nothing at all because
they are behind him. Pasting a frontal face onto the back of a head would be a
far worse artefact than the mannequin, and no amount of blending fixes a view
the photograph does not contain. The fade is the honest behaviour, not a
limitation to be tuned away.

## Registration

Two points, not five. The face model localises the eyes best; `head_centre` is
a face-oval centroid that moves with expression, and there are no ear or skull
anchors at all - so eyes give translation, rotation and uniform scale, which is
exactly a similarity transform and nothing more is claimed.

Interocular distance carries about 2% frame-to-frame jitter, which would read
as a pulsing face. It is smoothed over a few frames; the lean it is riding on
moves on a 3.2 s time constant, so a 5-frame smooth cannot introduce visible
lag.
"""

from __future__ import annotations

import math
from collections import deque

import cv2
import numpy as np

# Matches FaceSource's own ramp: full inside 20 degrees, gone by 48.
FULL_CONE_DEG = 20.0
ZERO_CONE_DEG = 48.0
# Below this the face contributes nothing worth the milliseconds.
MIN_GATE = 0.02


def _unit(v):
    n = float(np.linalg.norm(v))
    return v / n if n > 1e-9 else v


class FaceOverlay:
    """Composites the photoreal face onto the rendered 3D head."""

    def __init__(self, master_path: str, smooth: int = 5, quantise_deg: float = 0.5):
        from presenter.render.face_source import FaceSource

        self.source = FaceSource(master_path)
        self.smooth = deque(maxlen=smooth)
        self.quantise = quantise_deg
        self._cache_key = None
        self._cache_frame = None
        self.calls = 0
        self.hits = 0

    # -- geometry -----------------------------------------------------------
    @staticmethod
    def _project(scene, camera, point):
        """World point -> pixel coordinates, y down."""
        from bpy_extras.object_utils import world_to_camera_view
        from mathutils import Vector

        uv = world_to_camera_view(scene, camera, Vector(tuple(point)))
        w = scene.render.resolution_x
        h = scene.render.resolution_y
        return np.array([uv.x * w, (1.0 - uv.y) * h]), uv.z

    def gate(self, eye_mid, face_forward, camera_position, confidence):
        """How much of the photoreal face this camera has any right to see."""
        to_cam = _unit(np.asarray(camera_position, float) - eye_mid)
        cos = float(np.clip(np.dot(_unit(face_forward), to_cam), -1.0, 1.0))
        off = math.degrees(math.acos(cos))
        if off <= FULL_CONE_DEG:
            geo = 1.0
        elif off >= ZERO_CONE_DEG:
            geo = 0.0
        else:
            geo = 1.0 - (off - FULL_CONE_DEG) / (ZERO_CONE_DEG - FULL_CONE_DEG)
        return geo * float(confidence), off

    def _face(self, pose):
        """FaceSource frame, cached on a quantised head pose.

        The face costs ~91 ms and EEVEE ~110 ms; serially that is 5 fps rather
        than 9. At 30 Hz behaviour the head is often near-static between frames,
        so quantising the pose turns most frames into a cache hit and the cost
        back into a rounding error.
        """
        q = self.quantise
        key = (round(pose.yaw / q), round(pose.pitch / q), round(pose.roll / q),
               round(pose.eye_open_l, 2), round(pose.eye_open_r, 2),
               round(pose.mouth_open, 2))
        self.calls += 1
        if key == self._cache_key and self._cache_frame is not None:
            self.hits += 1
            return self._cache_frame
        self._cache_frame = self.source.frame(pose)
        self._cache_key = key
        return self._cache_frame

    # -- compositing --------------------------------------------------------
    def composite(self, frame_bgr, scene, camera, eye_l, eye_r, skull, pose):
        """Draw the face onto `frame_bgr`. Returns (frame, diagnostics)."""
        eye_l = np.asarray(eye_l, float)
        eye_r = np.asarray(eye_r, float)
        eye_mid = (eye_l + eye_r) / 2.0
        # The eyes sit forward of the skull centre by construction, so this
        # follows the head's rotation without re-deriving it from Euler angles
        # and getting a sign convention wrong.
        face_forward = _unit(eye_mid - np.asarray(skull, float))

        f = self._face(pose)
        gate, off_deg = self.gate(eye_mid, face_forward,
                                  camera.matrix_world.translation, f.confidence)
        info = {"off_axis_deg": round(off_deg, 1), "gate": round(gate, 3),
                "confidence": round(float(f.confidence), 3),
                "cache_hit_rate": round(self.hits / max(self.calls, 1), 3)}
        if gate < MIN_GATE:
            info["skipped"] = "camera is outside the cone the photograph covers"
            return frame_bgr, info

        pl, zl = self._project(scene, camera, eye_l)
        pr, zr = self._project(scene, camera, eye_r)
        if zl <= 0 or zr <= 0:
            info["skipped"] = "head is behind the camera"
            return frame_bgr, info

        # Smooth only the scale. Position and rotation track the head exactly;
        # it is the interocular measurement that jitters.
        target = float(np.linalg.norm(pr - pl))
        self.smooth.append(target)
        target = float(np.mean(self.smooth))

        # Left and right mean different things on the two sides of this seam.
        # The 3D eyes are named for the SUBJECT - eye_l is the eye on his left,
        # at -X. In a frontal photograph of him that eye appears on the IMAGE's
        # right, and FaceSource names its anchors by image position. Pairing
        # eye_l with eye_l therefore reverses the interocular vector, and a
        # similarity transform fitted to a reversed vector is a 180 degree
        # rotation: the face renders upside down, mouth above eyes.
        src_l = np.asarray(f.anchors["eye_r"], float)   # subject's left
        src_r = np.asarray(f.anchors["eye_l"], float)   # subject's right
        src_d = float(np.linalg.norm(src_r - src_l))
        if src_d < 1e-6 or target < 1e-6:
            info["skipped"] = "degenerate interocular distance"
            return frame_bgr, info

        scale = target / src_d
        ang = (math.atan2(pr[1] - pl[1], pr[0] - pl[0])
               - math.atan2(src_r[1] - src_l[1], src_r[0] - src_l[0]))
        # A near-180 degree fit means the eye correspondence has been reversed
        # again, not that he has tilted his head. Say so rather than rendering
        # an upside-down face that someone has to notice by eye.
        deg = abs(math.degrees(math.atan2(math.sin(ang), math.cos(ang))))
        if deg > 90.0:
            info["warning"] = (f"registration rotated {deg:.0f} deg - the eye "
                               f"anchors are probably reversed")
        ca, sa = math.cos(ang) * scale, math.sin(ang) * scale
        src_mid = (src_l + src_r) / 2.0
        dst_mid = (pl + pr) / 2.0
        m = np.array([[ca, -sa, dst_mid[0] - (ca * src_mid[0] - sa * src_mid[1])],
                      [sa, ca, dst_mid[1] - (sa * src_mid[0] + ca * src_mid[1])]],
                     dtype=np.float32)

        h, w = frame_bgr.shape[:2]
        rgba = f.rgba
        warped = cv2.warpAffine(rgba, m, (w, h), flags=cv2.INTER_LINEAR,
                                borderMode=cv2.BORDER_CONSTANT,
                                borderValue=(0, 0, 0, 0))
        # Straight alpha, so the colour is used as-is and only the coverage is
        # scaled by the gate.
        alpha = (warped[..., 3:4].astype(np.float32) / 255.0) * gate
        face_bgr = warped[..., :3].astype(np.float32)
        out = face_bgr * alpha + frame_bgr.astype(np.float32) * (1.0 - alpha)
        info["covered_px"] = int((alpha[..., 0] > 0.02).sum())
        return out.astype(np.uint8), info
