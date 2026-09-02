#!/usr/bin/env python3
"""Report what is actually inside a 3DGS PLY. Used to record Gaussian count."""
import sys
from pathlib import Path


def main() -> int:
    p = Path(sys.argv[1])
    if not p.exists():
        print("missing"); return 2
    n, props, binary = 0, [], False
    with open(p, "rb") as fh:
        for raw in fh:
            line = raw.decode("ascii", "ignore").strip()
            if line.startswith("format"):
                binary = "binary" in line
            elif line.startswith("element vertex"):
                n = int(line.split()[-1])
            elif line.startswith("property"):
                props.append(line.split()[-1])
            elif line == "end_header":
                break
    # A 3DGS ply is identifiable by its spherical-harmonic and opacity fields.
    sh = [x for x in props if x.startswith(("f_dc", "f_rest"))]
    kind = "3DGS" if ("opacity" in props and sh) else "point cloud / mesh"
    print(f"{n} gaussians | {kind} | {len(props)} properties | "
          f"{'binary' if binary else 'ascii'} | {p.stat().st_size/1e6:.1f} MB")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
