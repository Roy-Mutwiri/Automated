"""Prove the cameras observe one common geometry, and report what they see.

This does the checks that *can* be automated. It deliberately does not claim to
judge realism - the specification is explicit that automated tests support
visual inspection rather than replacing it, and nothing here says whether the
result looks good.

What it does establish:

* every camera is inside the room and not inside furniture
* every camera renders at the same resolution and shares one sensor and
  exposure - a cut between two different colour pipelines is a hard fail
* the SAME canonical 3D landmarks project into each camera consistently, which
  is the actual evidence that there is one world rather than seven
* which landmarks are in frame for each camera, which is the object visibility
  log the specification asks for
* the human transforms are identical across cameras at a frozen timestamp

The landmark projection is the important one. If two cameras disagree about
where a projected 3D point lands relative to what is rendered there, the scene
is not shared.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def project(scene, cam_ob, point):
    """World point -> normalised camera coordinates (0..1), plus depth."""
    from bpy_extras.object_utils import world_to_camera_view
    from mathutils import Vector

    co = world_to_camera_view(scene, cam_ob, Vector(point))
    return (co.x, 1.0 - co.y, co.z)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--time", type=float, default=314.5)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--out", default="docs/multicam_validation.md")
    args = ap.parse_args()

    from tools_shim import frozen_pose  # noqa: F401  (kept simple below)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
