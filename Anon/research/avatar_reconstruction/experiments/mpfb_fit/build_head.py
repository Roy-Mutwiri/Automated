"""Build a fitted human mesh from the CC0 MakeHuman base mesh and targets.

**No MPFB/MakeHuman code runs here.** Only the CC0 *data* is used - `base.obj`
and the `.target.gz` morph files. The MPFB add-on itself is GPL-3.0, and it is a
Blender UI tool that cannot be driven meaningfully headless anyway. The data is
the valuable part and it carries a licence we can ship under
(see `docs/avatar_dependency_licenses.md`).

## The morph format

A MakeHuman target is a sparse list of vertex displacements:

    <vertex index> <dx> <dy> <dz>

Applying a target at weight w adds w * delta to those vertices. Targets compose
linearly, which is what makes fitting tractable: the whole space of faces this
base mesh can reach is a weighted sum, and fitting is choosing weights.

## What is fitted, and what is not

Fitted from `config/avatar_landmarks.json`, measured off the locked plate:
overall head shape, face aspect, jaw width, nose width and length, mouth width,
eye spacing, chin. Those are the proportions that identify a face.

Not fitted: anything the plate cannot show. The back of the skull, the ear
interior, and the body below the shoulders are *designed once* from plausible
defaults and locked, exactly as the room's hidden geometry was.

Usage
-----
    python build_head.py --out ../../outputs/fitted_head.obj
"""

from __future__ import annotations

import argparse
import glob
import gzip
import json
import os
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[4] if (HERE.parents[4] / "config").exists() else HERE.parents[3]
DATA = HERE.parents[1] / "env/mpfb/data"


def load_base(path):
    """Read base.obj, separating the body from everything else.

    `base.obj` is not just a human. It contains, in groups:

    * `body` - 13,378 faces, the actual figure
    * `helper-*` - clothing simulation shells that *envelope* the body
    * `joint-*` - 6-face marker cubes at every skeleton joint

    Exporting the file wholesale renders the helper shells, which look like a
    smooth featureless mass wrapped around the person - and that is exactly
    what the first attempt produced. The helpers are why MPFB ships a
    `deletehelpers` operator.

    The joint cubes are kept separately rather than discarded: their centroids
    are the skeleton positions, which is what a rig will be built from.
    """
    verts, lines, keep_face, joints = [], [], [], {}
    group = None
    with open(path, encoding="utf-8", errors="ignore") as fh:
        for line in fh:
            lines.append(line)
            if line[:2] in ("g ", "o "):
                group = line.strip().split(None, 1)[1]
            elif line.startswith("v "):
                verts.append([float(x) for x in line.split()[1:4]])
            elif line.startswith("f "):
                keep_face.append(group == "body")
                if group and group.startswith("joint-"):
                    idx = [int(p.split("/")[0]) - 1 for p in line.split()[1:]]
                    joints.setdefault(group, set()).update(idx)
    return np.array(verts, np.float64), lines, keep_face, joints


def load_target(path):
    """Return (indices, deltas) for one .target.gz."""
    idx, d = [], []
    with gzip.open(path, "rt") as fh:
        for line in fh:
            parts = line.split()
            if len(parts) < 4:
                continue
            try:
                idx.append(int(parts[0]))
                d.append([float(parts[1]), float(parts[2]), float(parts[3])])
            except ValueError:
                continue
    return np.array(idx, np.int64), np.array(d, np.float64)


class Morpher:
    def __init__(self, data_dir: Path):
        self.data = data_dir
        self._cache: dict[str, tuple] = {}
        self.index = {}
        for f in glob.glob(str(data_dir / "targets/**/*.target.gz"), recursive=True):
            key = os.path.basename(f).replace(".target.gz", "")
            self.index.setdefault(key, f)

    def get(self, name):
        if name not in self._cache:
            path = self.index.get(name)
            if path is None:
                return None
            self._cache[name] = load_target(path)
        return self._cache[name]

    def apply(self, verts, weights: dict[str, float], verbose=True):
        out = verts.copy()
        applied, missing = 0, []
        for name, w in weights.items():
            if abs(w) < 1e-6:
                continue
            t = self.get(name)
            if t is None:
                missing.append(name)
                continue
            idx, d = t
            out[idx] += d * w
            applied += 1
        if verbose:
            print(f"[build] applied {applied} targets"
                  + (f", MISSING {missing}" if missing else ""))
        return out


def identity_weights(ratios: dict) -> dict[str, float]:
    """Turn measured face ratios into morph weights.

    Each line is a deliberate reading of one measurement, not a tuned constant.
    The reference values are the base mesh's own proportions, so a weight is
    "how far is this man from the average face in this dimension".
    """
    w: dict[str, float] = {}

    # Macro: an adult male. Ethnicity in MakeHuman is a three-way blend and a
    # Middle Eastern face is not one of the three axes, so it is expressed as a
    # blend rather than forced onto the nearest single label.
    w["caucasian-male-young"] = 0.65
    w["african-male-young"] = 0.20
    w["asian-male-young"] = 0.15

    # Face aspect: measured 0.82 wide-over-tall. Below ~0.86 is a longer face.
    aspect = ratios.get("aspect_w_over_h", 0.85)
    longness = np.clip((0.86 - aspect) / 0.12, -1, 1)
    w["head-oval"] = float(max(longness, 0.0)) * 0.6
    w["head-scale-vert-incr"] = float(max(longness, 0.0)) * 0.45
    w["head-scale-horiz-decr"] = float(max(longness, 0.0)) * 0.25

    # Jaw: 0.849 of face width is a wide, square jaw.
    jaw = ratios.get("jaw_width_over_width", 0.78)
    jawness = np.clip((jaw - 0.78) / 0.12, -1, 1)
    w["chin-width-incr" if jawness >= 0 else "chin-width-decr"] = abs(float(jawness)) * 0.7
    w["chin-bones-incr"] = float(max(jawness, 0.0)) * 0.6
    w["head-square"] = float(max(jawness, 0.0)) * 0.35

    # Nose.
    nw = ratios.get("nose_width_over_width", 0.30)
    nwness = np.clip((nw - 0.29) / 0.08, -1, 1)
    w["nose-scale-horiz-incr" if nwness >= 0 else "nose-scale-horiz-decr"] = \
        abs(float(nwness)) * 0.7
    nl = ratios.get("nose_length_over_height", 0.28)
    nlness = np.clip((nl - 0.28) / 0.06, -1, 1)
    w["nose-scale-vert-incr" if nlness >= 0 else "nose-scale-vert-decr"] = \
        abs(float(nlness)) * 0.6

    # Mouth.
    mw = ratios.get("mouth_width_over_width", 0.38)
    mwness = np.clip((mw - 0.38) / 0.08, -1, 1)
    w["mouth-scale-horiz-incr" if mwness >= 0 else "mouth-scale-horiz-decr"] = \
        abs(float(mwness)) * 0.6

    # Eye spacing.
    ipd = ratios.get("interpupillary_over_width", 0.45)
    ipdness = np.clip((ipd - 0.45) / 0.06, -1, 1)
    for side in "lr":
        key = f"{side}-eye-trans-out" if ipdness >= 0 else f"{side}-eye-trans-in"
        w[key] = abs(float(ipdness)) * 0.5

    # Chin height.
    ch = ratios.get("chin_to_mouth_over_height", 0.27)
    chness = np.clip((ch - 0.27) / 0.06, -1, 1)
    w["chin-height-incr" if chness >= 0 else "chin-height-decr"] = \
        abs(float(chness)) * 0.5

    return {k: v for k, v in w.items() if abs(v) > 1e-6}


def write_obj(path, verts, source_lines, keep_face):
    """Write the fitted BODY only, with vertices replaced.

    Vertex indices stay as they are. OBJ indices are global, so dropping faces
    needs no renumbering, and keeping the original numbering means the morph
    targets and any future vertex-index mapping remain valid.
    """
    vi = fi = written = 0
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("# MakeHuman base mesh (CC0), body group only, "
                 "morph-fitted to the approved identity.\n")
        for line in source_lines:
            if line.startswith("v "):
                x, y, z = verts[vi]
                fh.write(f"v {x:.6f} {y:.6f} {z:.6f}\n")
                vi += 1
            elif line.startswith("f "):
                if keep_face[fi]:
                    fh.write(line)
                    written += 1
                fi += 1
            elif line[:2] in ("g ", "o "):
                continue          # one object; group names are not needed
            else:
                fh.write(line)
    return vi, written


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--landmarks", default=None)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    lm_path = Path(args.landmarks) if args.landmarks else \
        ROOT / "config/avatar_landmarks.json"
    out_path = Path(args.out) if args.out else \
        HERE.parents[1] / "outputs/fitted_head.obj"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    lm = json.loads(lm_path.read_text(encoding="utf-8"))
    ratios = lm["ratios"]
    print(f"[build] identity ratios from {lm_path.name}")

    verts, lines, keep_face, joints = load_base(DATA / "3dobjs/base.obj")
    print(f"[build] body faces {sum(keep_face)} of {len(keep_face)}; "
          f"{len(joints)} joint markers")
    lo, hi = verts.min(axis=0), verts.max(axis=0)
    print(f"[build] base mesh {len(verts)} verts, bounds "
          f"x[{lo[0]:.2f},{hi[0]:.2f}] y[{lo[1]:.2f},{hi[1]:.2f}] "
          f"z[{lo[2]:.2f},{hi[2]:.2f}]")

    morpher = Morpher(DATA)
    print(f"[build] {len(morpher.index)} targets available")
    weights = identity_weights(ratios)
    for k, v in sorted(weights.items()):
        print(f"    {k:34s} {v:+.3f}")

    fitted = morpher.apply(verts, weights)
    moved = int((np.linalg.norm(fitted - verts, axis=1) > 1e-6).sum())
    print(f"[build] {moved} of {len(verts)} vertices moved by the fit")

    n, nf = write_obj(out_path, fitted, lines, keep_face)
    print(f"[build] wrote {n} verts, {nf} body faces -> {out_path}")

    # Skeleton joint positions, from the marker cubes, after the same morphs.
    # These are what a rig gets built from, so they must come from the fitted
    # mesh rather than the base one.
    joint_pos = {name: fitted[sorted(idx)].mean(axis=0).tolist()
                 for name, idx in joints.items()}
    (out_path.parent / "fitted_joints.json").write_text(
        json.dumps(joint_pos, indent=2), encoding="utf-8")
    print(f"[build] {len(joint_pos)} fitted joint positions -> "
          f"fitted_joints.json")

    meta = out_path.with_suffix(".json")
    meta.write_text(json.dumps({
        "source_base_mesh": "MakeHuman base.obj (CC0), via mpfb-2.0.8",
        "targets_licence": "CC0",
        "addon_code_used": False,
        "weights": weights,
        "ratios": ratios,
    }, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
