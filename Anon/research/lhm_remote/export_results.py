#!/usr/bin/env python3
"""Collect LHM's output into one portable package, and render the identity gate.

Two jobs:

1. **Turntable first.** Render the reconstruction at 0, +/-20 and +/-40 degrees
   and write `outputs/lhm_identity_turntable.png`. This is the only thing worth
   looking at before deciding whether to continue - a reconstruction that is not
   recognisably the same man does not deserve an export pipeline.

2. **Package second.** Gather whatever LHM actually produced - mesh, Gaussian
   ply, textures, SMPL-X parameters - into `outputs/avatar_v01/` with a
   `metadata.json` describing every file, so the Windows side can consume it
   without needing LHM installed.

LHM's output layout varies by version, so this discovers files rather than
assuming paths, and reports honestly what it did and did not find.
"""

from __future__ import annotations

import argparse
import json
import math
import shutil
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent

# What we hope to find, and what each thing is for. Order matters: earlier
# patterns are preferred when several match.
WANTED = [
    ("mesh", ("*.obj", "*.glb", "*.gltf", "*.ply"), "geometry"),
    ("gaussians", ("*.ply",), "3D Gaussian representation, if produced"),
    ("texture", ("*.png", "*.jpg"), "texture maps"),
    ("smplx", ("*.npz", "*.npy", "*.json"), "SMPL-X / pose parameters"),
]


def find_outputs(root: Path) -> dict[str, list[Path]]:
    """Discover what LHM wrote. Paths differ between versions."""
    found: dict[str, list[Path]] = {}
    search_roots = [root / "exps", root / "outputs", root / "LHM" / "outputs", root]
    seen: set[Path] = set()
    for base in search_roots:
        if not base.is_dir():
            continue
        for kind, patterns, _ in WANTED:
            for pat in patterns:
                for p in base.rglob(pat):
                    if p in seen or ".git" in p.parts or "pretrained_models" in p.parts:
                        continue
                    if p.stat().st_mtime < time.time() - 86400:
                        continue          # not from this run
                    seen.add(p)
                    found.setdefault(kind, []).append(p)
    return found


def turntable(mesh_path: Path, out: Path, angles=(-40, -20, 0, 20, 40)) -> bool:
    """Render the reconstruction from five yaw angles.

    Uses trimesh + pyrender, both already in LHM's dependency set, rather than
    requiring Blender on the remote box. This is a look-at-it check, not a
    beauty render - flat lighting on purpose, so geometry is judged rather than
    shading.
    """
    try:
        import numpy as np
        import pyrender
        import trimesh
        from PIL import Image
    except Exception as exc:  # noqa: BLE001
        print(f"[export] cannot render turntable ({type(exc).__name__}: {exc})")
        return False

    try:
        mesh = trimesh.load(str(mesh_path), force="mesh")
        if mesh.vertices.shape[0] == 0:
            print("[export] mesh has no vertices")
            return False
    except Exception as exc:  # noqa: BLE001
        print(f"[export] cannot load {mesh_path.name}: {exc}")
        return False

    mesh.vertices -= mesh.vertices.mean(axis=0)
    radius = float(np.linalg.norm(mesh.vertices, axis=1).max())

    tiles = []
    for deg in angles:
        scene = pyrender.Scene(bg_color=[0.10, 0.10, 0.11, 1.0],
                               ambient_light=[0.35, 0.35, 0.35])
        scene.add(pyrender.Mesh.from_trimesh(mesh, smooth=True))
        a = math.radians(deg)
        dist = radius * 2.6
        eye = np.array([dist * math.sin(a), 0.0, dist * math.cos(a)])
        # look-at, +Y up
        fwd = -eye / np.linalg.norm(eye)
        right = np.cross(np.array([0.0, 1.0, 0.0]), fwd)
        right /= np.linalg.norm(right)
        up = np.cross(fwd, right)
        pose = np.eye(4)
        pose[:3, 0], pose[:3, 1], pose[:3, 2], pose[:3, 3] = right, up, -fwd, eye
        cam = pyrender.PerspectiveCamera(yfov=math.radians(35))
        scene.add(cam, pose=pose)
        scene.add(pyrender.DirectionalLight(color=[1, 1, 1], intensity=3.0),
                  pose=pose)
        r = pyrender.OffscreenRenderer(512, 512)
        colour, _ = r.render(scene)
        r.delete()
        tiles.append(np.asarray(colour))

    strip = np.hstack(tiles)
    Image.fromarray(strip).save(out)
    print(f"[export] turntable -> {out}")
    return True


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model-name", default="LHM-1B-HF")
    ap.add_argument("--image", default="inputs/avatar_rgba.png")
    ap.add_argument("--out", default="outputs/avatar_v01")
    args = ap.parse_args()

    pkg = HERE / args.out
    pkg.mkdir(parents=True, exist_ok=True)

    found = find_outputs(HERE)
    if not found:
        print("[export] NOTHING FOUND. Check outputs/reconstruction.log - the "
              "run probably failed before writing anything.")
        return 2

    manifest = {
        "model": args.model_name,
        "source_image": args.image,
        "created": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "files": [],
        "note": "Reconstructed from an original synthetic identity. "
                "Research/personal use. Do not redistribute checkpoints or "
                "body models.",
    }

    for kind, paths in found.items():
        for p in sorted(paths)[:40]:
            dest = pkg / kind / p.name
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(p, dest)
            manifest["files"].append({
                "kind": kind,
                "name": f"{kind}/{p.name}",
                "bytes": p.stat().st_size,
                "from": str(p.relative_to(HERE)) if HERE in p.parents else str(p),
            })
        print(f"[export] {kind}: {len(paths)} file(s)")

    meshes = [p for p in found.get("mesh", [])
              if p.suffix.lower() in (".obj", ".glb", ".gltf", ".ply")]
    if meshes:
        biggest = max(meshes, key=lambda p: p.stat().st_size)
        print(f"[export] turntable from {biggest.name}")
        turntable(biggest, HERE / "outputs" / "lhm_identity_turntable.png")
        manifest["turntable_source"] = biggest.name
    else:
        print("[export] no mesh found - cannot render the identity turntable. "
              "If LHM produced only Gaussians, view its own preview renders in "
              "outputs/ instead.")

    (pkg / "metadata.json").write_text(json.dumps(manifest, indent=2),
                                       encoding="utf-8")
    shutil.copy2(HERE / args.image, pkg / "source_input.png")
    print(f"\n[export] package -> {pkg}  ({len(manifest['files'])} files)")
    print("[export] LOOK AT outputs/lhm_identity_turntable.png BEFORE anything else.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
