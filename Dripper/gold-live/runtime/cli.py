"""Single entry point for the packaged application.

One exe, subcommands, so a customer has one thing to run rather than a
directory of scripts:

    GoldLive.exe setup                          seed config on this machine
    GoldLive.exe doctor                         preflight every dependency
    GoldLive.exe calibrate --session SESSION_001
    GoldLive.exe run --session SESSION_001      one session
    GoldLive.exe supervise                      all configured sessions
    GoldLive.exe dashboard                      operator UI
    GoldLive.exe bench                          benchmark the local model
    GoldLive.exe version

Argument parsing is delegated to each module by rewriting sys.argv, so the same
modules work identically from source and frozen.
"""

from __future__ import annotations

import sys

from shared.paths import data_root, describe

VERSION = "0.4.0"

USAGE = f"""Gold Live {VERSION}

  setup       Seed configuration into this machine's data directory
  doctor      Check every dependency and report what is missing
  calibrate   Locate the comment panel on this screen
  run         Run one session
  supervise   Run every configured session with restart supervision
  dashboard   Operator dashboard
  bench       Benchmark the local model against this workload
  paths       Show where configuration and data live
  version

Data directory: {data_root()}

Start with:  setup, then doctor.
"""


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if not argv or argv[0] in ("-h", "--help", "help"):
        print(USAGE)
        return 0

    command, rest = argv[0], argv[1:]
    sys.argv = [f"goldlive-{command}", *rest]

    if command == "version":
        print(f"Gold Live {VERSION}")
        return 0

    if command == "paths":
        print(describe())
        return 0

    if command == "setup":
        from runtime.bootstrap import setup

        return setup(force="--force" in rest)

    if command == "doctor":
        from runtime.bootstrap import doctor, seed_config

        seed_config()  # a doctor run on a fresh machine should not fail on config
        return doctor()

    if command == "calibrate":
        from scripts.calibrate_capture import main as calibrate

        calibrate()
        return 0

    if command == "run":
        from runtime.bootstrap import seed_config
        from runtime.live import main as live

        seed_config()
        live()
        return 0

    if command == "supervise":
        from runtime.bootstrap import seed_config
        from runtime.supervisor import main as supervise

        seed_config()
        supervise()
        return 0

    if command == "dashboard":
        from dashboard.server import main as dashboard

        dashboard()
        return 0

    if command == "bench":
        from scripts.bench_llm import main as bench

        bench()
        return 0

    if command == "dryrun":
        from runtime.dryrun import main as dryrun

        dryrun()
        return 0

    if command == "soak":
        from runtime.soak import main as soak

        soak()
        return 0

    print(f"Unknown command: {command}\n")
    print(USAGE)
    return 2


if __name__ == "__main__":
    sys.exit(main())
