#!/usr/bin/env python3
"""Validate this package without a GPU, before it is copied to a rented box.

Everything here is checkable locally, and every check corresponds to a way the
remote run could waste an hour of paid GPU time on something that was knowable
in advance: a path that only exists on the Windows workstation, an input
filename that drifted from the frozen asset, a script that reports success
after its inference step failed.

    python dryrun_check.py

Exit code 0 means the package is safe to copy across.
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
FAILURES: list[str] = []
NOTES: list[str] = []


def check(name, ok, detail=""):
    print(f"  [{'OK  ' if ok else 'FAIL'}] {name:<44} {detail}")
    if not ok:
        FAILURES.append(f"{name}: {detail}")
    return ok


def main() -> int:
    print("\n=== required contents ===")
    required = [
        "README.md", "RUNBOOK.md", "environment_manifest.json",
        "setup_lhmpp.sh", "run_lhmpp.sh", "verify_environment.py",
        "inspect_ply.py", "export_results.py",
        "setup.sh", "run_reconstruction.sh",
        "patches/apply_patches.py",
    ]
    for rel in required:
        p = HERE / rel
        check(rel, p.exists(), f"{p.stat().st_size} bytes" if p.exists()
              else "MISSING")
    for d in ("inputs", "patches"):
        check(f"{d}/", (HERE / d).is_dir())

    print("\n=== frozen inputs ===")
    # The identity is locked. A silently renamed input is how a run ends up
    # reconstructing the wrong person, or nobody.
    expected_inputs = ["avatar_identity_camera1.png", "avatar_rgba.png",
                       "avatar_mask.png"]
    for name in expected_inputs:
        p = HERE / "inputs" / name
        check(f"inputs/{name}", p.exists(),
              f"{p.stat().st_size / 1024:.0f} KB" if p.exists() else "MISSING")

    print("\n=== manifest ===")
    try:
        man = json.loads((HERE / "environment_manifest.json").read_text())
    except Exception as exc:  # noqa: BLE001
        check("manifest parses", False, str(exc)[:80])
        man = {}
    if man:
        commit = man.get("target", {}).get("commit", "")
        check("commit is a full sha", bool(re.fullmatch(r"[0-9a-f]{40}", commit)),
              commit or "missing")
        check("model name pinned", bool(man.get("model", {}).get("name")),
              man.get("model", {}).get("name", ""))
        check("post-run fields still blank",
              all(v is None for v in man.get("recorded_after_the_run", {}).values()),
              "ready to be filled in on the box")

    print("\n=== scripts reference only files that exist ===")
    # run_lhmpp.sh calls three helpers by name; a typo there fails after the
    # checkpoint download rather than before it.
    run_sh = (HERE / "run_lhmpp.sh").read_text(encoding="utf-8")
    for helper in ("verify_environment.py", "inspect_ply.py", "export_results.py"):
        check(f"run_lhmpp.sh -> {helper}",
              helper not in run_sh or (HERE / helper).exists())
    check("run_lhmpp.sh uses --lhmpp", "--lhmpp" in run_sh)
    check("run_lhmpp.sh passes --out", "--out" in run_sh)

    print("\n=== no workstation-local paths ===")
    # A path from this machine is the single most likely thing to survive into
    # a script and break only once it is somewhere else.
    local = re.compile(r"[A-Za-z]:[\\/]|/mnt/[a-z]/|Users[\\/]mutwi|Documents[\\/]Automated",
                       re.IGNORECASE)
    for p in sorted(HERE.glob("*.sh")) + sorted(HERE.glob("*.py")) + \
            [HERE / "README.md", HERE / "RUNBOOK.md"]:
        hits = [f"line {i}" for i, line in
                enumerate(p.read_text(encoding="utf-8", errors="ignore").splitlines(), 1)
                if local.search(line)]
        check(f"{p.name} has no local paths", not hits, "; ".join(hits[:3]))

    print("\n=== failures exit nonzero ===")
    for name in ("setup_lhmpp.sh", "run_lhmpp.sh"):
        body = (HERE / name).read_text(encoding="utf-8")
        check(f"{name}: set -euo pipefail", "set -euo pipefail" in body)

    # `cmd | tee log` reports tee's exit status, not the command's, so an
    # inference that dies mid-run looks like a success to `set -e`. Blocking
    # this is the difference between finding out now and finding out after the
    # box is destroyed.
    piped = [i for i, line in enumerate(run_sh.splitlines(), 1)
             if "| tee" in line]
    check("run_lhmpp.sh: piped commands keep their exit status",
          not piped or "pipefail" in run_sh,
          f"tee on line(s) {piped}" if piped else "")

    print("\n=== logs and output directories ===")
    check("run_lhmpp.sh creates its output dirs", "mkdir -p" in run_sh)
    check("run_lhmpp.sh captures a log", "reconstruction.log" in run_sh)
    check("outputs/ is git-ignored", _ignored("outputs"))
    check("private_models/ is git-ignored", _ignored("private_models"))

    print("\n=== python helpers are syntactically valid ===")
    import ast
    for p in sorted(HERE.glob("*.py")) + sorted(HERE.glob("patches/*.py")):
        try:
            ast.parse(p.read_text(encoding="utf-8"))
            check(p.name, True)
        except SyntaxError as exc:
            check(p.name, False, f"line {exc.lineno}: {exc.msg}")

    print("\n" + "=" * 68)
    for note in NOTES:
        print(f"note: {note}")
    if FAILURES:
        print(f"NOT READY - {len(FAILURES)} problem(s):")
        for f in FAILURES:
            print(f"  - {f}")
        return 1
    print("Package is self-consistent and safe to copy to the GPU box.")
    print("Next: RUNBOOK.md step 1.")
    return 0


def _ignored(name: str) -> bool:
    """True if `name` is covered by a .gitignore at or above this directory."""
    for base in (HERE, *HERE.parents):
        gi = base / ".gitignore"
        if gi.exists() and any(
                name in line for line in
                gi.read_text(encoding="utf-8", errors="ignore").splitlines()):
            return True
        if (base / ".git").exists():
            break
    NOTES.append(f"{name}/ is not in any .gitignore - check before committing")
    return False


if __name__ == "__main__":
    raise SystemExit(main())
