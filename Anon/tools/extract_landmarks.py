"""Measure the approved identity: facial landmarks on the locked plate.

These are the numbers the 3D head is fitted *to*. The rule for this whole stage
is that the mesh adapts to the identity and never the reverse, so the identity
has to exist as measurements rather than as an impression.

MediaPipe Face Mesh is used because it is Apache-2.0 and already a dependency -
the licence audit rules out anything that would put a non-commercial component
into asset creation, and this runs on the CPU in a second.

Two coordinate sets come out of it and both matter:

* **image landmarks** - where each feature is in the plate, in pixels. These are
  the fitting target: project the 3D mesh through Camera 1 and these are what
  the projection must land on.
* **canonical landmarks** - MediaPipe's own 3D reconstruction of the face, in a
  metric-ish canonical space. Not accurate enough to *be* the head, but very
  useful for the ratios that describe a face: eye spacing over face width, nose
  length, jaw width. Those ratios are what distinguish this man from a generic
  head, and they are what the morph targets get driven by.

Writes `config/avatar_landmarks.json`.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]

# MediaPipe Face Mesh indices. Named so the fitting code reads as anatomy
# rather than as magic numbers.
NAMED = {
    "eye_l_outer": 33, "eye_l_inner": 133,
    "eye_r_inner": 362, "eye_r_outer": 263,
    "eye_l_top": 159, "eye_l_bottom": 145,
    "eye_r_top": 386, "eye_r_bottom": 374,
    "iris_l": 468, "iris_r": 473,
    "brow_l_inner": 55, "brow_l_outer": 46,
    "brow_r_inner": 285, "brow_r_outer": 276,
    "nose_bridge": 168, "nose_tip": 1,
    "nostril_l": 129, "nostril_r": 358,
    "nose_base": 2,
    "mouth_l": 61, "mouth_r": 291,
    "lip_top": 13, "lip_bottom": 14,
    "chin": 152,
    "jaw_l": 172, "jaw_r": 397,
    "cheek_l": 234, "cheek_r": 454,
    "forehead": 10,
    "temple_l": 127, "temple_r": 356,
}


def measure(pts3: dict) -> dict:
    """Ratios that describe *this* face, independent of overall scale.

    Absolute sizes are useless for driving morph targets - the head is scaled
    to the plate separately. What identifies a person is proportion: how far
    apart the eyes sit relative to face width, how far the nose projects, how
    wide the jaw is against the cheekbones.
    """
    def d(a, b):
        return float(np.linalg.norm(np.array(pts3[a]) - np.array(pts3[b])))

    face_w = d("cheek_l", "cheek_r")
    face_h = d("forehead", "chin")
    return {
        "face_width": face_w,
        "face_height": face_h,
        "aspect_w_over_h": face_w / max(face_h, 1e-6),
        "interpupillary_over_width": d("iris_l", "iris_r") / max(face_w, 1e-6),
        "eye_width_over_width": d("eye_l_outer", "eye_l_inner") / max(face_w, 1e-6),
        "nose_length_over_height": d("nose_bridge", "nose_base") / max(face_h, 1e-6),
        "nose_width_over_width": d("nostril_l", "nostril_r") / max(face_w, 1e-6),
        "mouth_width_over_width": d("mouth_l", "mouth_r") / max(face_w, 1e-6),
        "jaw_width_over_width": d("jaw_l", "jaw_r") / max(face_w, 1e-6),
        "chin_to_mouth_over_height": d("chin", "lip_bottom") / max(face_h, 1e-6),
        "brow_to_eye_over_height": d("brow_l_inner", "eye_l_top") / max(face_h, 1e-6),
        "upper_face_over_height": d("forehead", "nose_bridge") / max(face_h, 1e-6),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--plate", default="assets/reference/avatar_identity_camera1.png")
    ap.add_argument("--out", default="config/avatar_landmarks.json")
    args = ap.parse_args()

    img = cv2.imread(str(ROOT / args.plate))
    if img is None:
        print(f"[landmarks] cannot read {args.plate}")
        return 2
    h, w = img.shape[:2]

    # MediaPipe 1.x is tasks-only; the legacy `solutions.face_mesh` module is
    # gone, so this uses FaceLandmarker with a downloaded model bundle
    # (Apache-2.0, recorded in docs/avatar_dependency_licenses.md).
    import mediapipe as mp
    from mediapipe.tasks.python import vision
    from mediapipe.tasks.python.core.base_options import BaseOptions

    model = ROOT / "assets/models/face_landmarker.task"
    options = vision.FaceLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=str(model)),
        running_mode=vision.RunningMode.IMAGE,
        num_faces=1,
        output_face_blendshapes=True,
        output_facial_transformation_matrixes=True,
    )
    with vision.FaceLandmarker.create_from_options(options) as landmarker:
        mp_img = mp.Image(image_format=mp.ImageFormat.SRGB,
                          data=cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        res = landmarker.detect(mp_img)

    if not res.face_landmarks:
        print("[landmarks] no face found")
        return 2
    lm = res.face_landmarks[0]
    print(f"[landmarks] {len(lm)} raw landmarks on a {w}x{h} plate")

    image_pts, canon_pts = {}, {}
    for name, i in NAMED.items():
        if i >= len(lm):
            continue
        p = lm[i]
        image_pts[name] = [round(p.x * w, 2), round(p.y * h, 2)]
        canon_pts[name] = [p.x * w, p.y * h, p.z * w]

    ratios = measure(canon_pts)

    # Head box in image space, for the scale fit.
    xs = [p.x * w for p in lm]
    ys = [p.y * h for p in lm]
    box = {"x0": min(xs), "x1": max(xs), "y0": min(ys), "y1": max(ys)}

    payload = {
        "_comment": "Measured from the locked identity plate by "
                    "tools/extract_landmarks.py. The 3D head is fitted to "
                    "these; they are never adjusted to suit the mesh.",
        "plate": args.plate,
        "plate_size": [w, h],
        "image_landmarks": image_pts,
        "face_box": {k: round(v, 2) for k, v in box.items()},
        "ratios": {k: round(v, 5) for k, v in ratios.items()},
        "all_image_landmarks": [[round(p.x * w, 2), round(p.y * h, 2)] for p in lm],
    }
    out = ROOT / args.out
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    print(f"[landmarks] face box {box['x1'] - box['x0']:.0f} x "
          f"{box['y1'] - box['y0']:.0f} px")
    for k, v in ratios.items():
        print(f"  {k:34s} {v:.4f}")
    print(f"[landmarks] -> {out}")

    # Visual check: the fit is only as good as these points.
    vis = img.copy()
    for p in lm:
        cv2.circle(vis, (int(p.x * w), int(p.y * h)), 1, (60, 200, 60), -1)
    for name, (x, y) in image_pts.items():
        cv2.circle(vis, (int(x), int(y)), 3, (0, 200, 255), -1)
    x0, y0 = int(box["x0"]) - 40, int(box["y0"]) - 40
    x1, y1 = int(box["x1"]) + 40, int(box["y1"]) + 40
    crop = vis[max(y0, 0):y1, max(x0, 0):x1]
    (ROOT / "renders").mkdir(exist_ok=True)
    cv2.imwrite(str(ROOT / "renders/landmarks_debug.png"), crop)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
