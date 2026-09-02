#!/usr/bin/env python3
"""Minimal, recorded patches to make LHM run on a modern GPU stack.

The rule is: **fix only what prevents inference.** Nothing here changes model
behaviour, weights, architecture or output. Every edit is printed so the diff
between "what upstream says" and "what we ran" is never invisible.

Idempotent - safe to run twice.

    python patches/apply_patches.py --repo LHM
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

APPLIED: list[str] = []


def patch_requirements(repo: Path) -> None:
    """Relax pins that cannot hold on a modern/Blackwell stack.

    * `torch==2.3.0` / `torchvision==0.18.0` - installed separately by
      `setup.sh`, which chooses the build that actually has kernels for the GPU
      present. Leaving them here makes pip downgrade torch back to a version
      with no sm_120 support, silently undoing the decision.
    * `numpy==1.23.0` - predates the numpy 2 ABI split and will not build
      against a current torch.
    * `chumpy` - unmaintained, imports `numpy.bool`/`numpy.float` which were
      removed. Only needed by legacy SMPL loaders; `smplx` works without it for
      `.npz` models.
    """
    req = repo / "requirements.txt"
    if not req.exists():
        return
    lines = req.read_text(encoding="utf-8").splitlines()
    out, dropped = [], []
    for line in lines:
        name = re.split(r"[=<>!\[]", line.strip(), 1)[0].strip().lower()
        if name in {"torch", "torchvision", "torchaudio"}:
            dropped.append(line)
            continue
        if name == "numpy":
            out.append("numpy>=1.26")
            dropped.append(f"{line} -> numpy>=1.26")
            continue
        if name == "chumpy":
            dropped.append(f"{line} (removed; breaks on modern numpy)")
            continue
        out.append(line)
    if dropped:
        req.write_text("\n".join(out) + "\n", encoding="utf-8")
        APPLIED.append("requirements.txt: " + "; ".join(dropped))


def patch_numpy_aliases(repo: Path) -> None:
    """Restore the numpy aliases removed in numpy 1.24.

    Several vendored research files still use `np.float`, `np.int`, `np.bool`.
    Rewriting every call site would be a large diff for no benefit, so a tiny
    shim is installed at package import instead - and only for names that are
    genuinely gone.
    """
    init = repo / "LHM" / "__init__.py"
    if not init.exists():
        return
    text = init.read_text(encoding="utf-8")
    marker = "# --- lhm_remote numpy alias shim"
    if marker in text:
        return
    shim = f'''{marker} ------------------------------------------
# numpy >= 1.24 removed np.float / np.int / np.bool / np.object. Some vendored
# research code still uses them. Restoring the aliases is far less invasive
# than editing every call site, and affects nothing else.
import numpy as _np
for _alias, _real in (("float", float), ("int", int), ("bool", bool),
                      ("object", object), ("str", str)):
    if not hasattr(_np, _alias):
        setattr(_np, _alias, _real)
del _np, _alias, _real
# --- end shim ---------------------------------------------------------------

'''
    init.write_text(shim + text, encoding="utf-8")
    APPLIED.append("LHM/__init__.py: numpy alias shim")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--repo", default="LHM")
    args = ap.parse_args()
    repo = Path(args.repo)
    if not repo.exists():
        print(f"[patch] {repo} not found - clone LHM first")
        return 2

    patch_requirements(repo)
    patch_numpy_aliases(repo)

    if APPLIED:
        print("[patch] applied:")
        for a in APPLIED:
            print(f"  - {a}")
    else:
        print("[patch] nothing to do (already patched)")

    log = Path(__file__).with_name("APPLIED_PATCHES.md")
    log.write_text(
        "# Patches applied to upstream LHM\n\n"
        "Recorded so the difference between upstream and what we ran is never\n"
        "invisible. None of these change model behaviour or output.\n\n"
        + ("".join(f"- {a}\n" for a in APPLIED) if APPLIED
           else "- (none on the last run; already patched)\n"),
        encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
