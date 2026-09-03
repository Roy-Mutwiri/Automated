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
from shared.version import app_version

#: Resolved from pyproject.toml -- see shared/version.py. Kept as a module
#: attribute because existing callers and the USAGE banner reference it.
VERSION = app_version()

USAGE = f"""Gold Live {VERSION}

  panel       Open the control panel (START / STOP)
  setup-wizard  Guided first-time setup, then stop
  stop        Stop a running GoldLive
  provision   Set this PC up: check it, download the model and voices
  ready       Check whether SESSION_001 can actually start right now
  selftest    Exercise the real components (--imports for a quick check)
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

Start with:  provision, then ready.
"""


def _code(result: object) -> int:
    """Turn a subcommand's return value into a process exit code.

    Every subcommand used to be called for its side effects and followed by a
    bare `return 0`, so a session that refused to start, a failed selftest or a
    readiness gate that did not pass all exited 0. A scheduled task or a shell
    script had no way to tell success from failure.

    None means "ran to completion without a verdict" and stays 0; an int is
    passed through; anything else is a programming error, not a status.
    """
    if result is None:
        return 0
    if isinstance(result, bool):  # bool is an int subclass; catch it first
        return 0 if result else 1
    if isinstance(result, int):
        return result
    return 0


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

    if command in ("panel", "gui", "control"):
        from runtime.panel import main as panel

        return _code(panel())

    if command in ("setup-wizard", "firstrun"):
        from runtime.setup_wizard import main as wizard

        return _code(wizard())

    if command == "stop":
        from runtime.panel import stop_goldlive

        ok, why = stop_goldlive()
        print(f"\n  {'stopped' if ok else 'STOP INCOMPLETE: ' + why}\n")
        return 0 if ok else 1

    if command == "provision":
        from runtime.provision import main as provision

        return _code(provision())

    if command == "selftest":
        from runtime.selftest import main as selftest

        return _code(selftest())

    if command == "ready":
        import argparse
        import asyncio

        from runtime.bootstrap import seed_config
        from runtime.readiness import check

        ap = argparse.ArgumentParser(prog="GoldLive ready")
        ap.add_argument("--session", default="SESSION_001")
        ap.add_argument("--market", default="gold")
        ap.add_argument("--adapter", default="file")
        ap.add_argument("--session-only", action="store_true",
                        help="skip the broadcast gates")
        ready_args = ap.parse_args(rest)

        seed_config()
        result = asyncio.run(check(
            session_id=ready_args.session, market=ready_args.market,
            adapter=ready_args.adapter,
            include_broadcast=not ready_args.session_only,
        ))
        print()
        print(result.render())
        print()
        # 0 session-ready or better, 1 not ready. Broadcast-only failures are
        # deliberately not an error: a session that can speak about real prices
        # is a success even before VB-CABLE is installed.
        from runtime.readiness import Level

        return 0 if result.level is not Level.NOT_READY else 1

    if command == "setup":
        from runtime.bootstrap import setup

        return setup(force="--force" in rest)

    if command == "doctor":
        from runtime.bootstrap import doctor, seed_config

        seed_config()  # a doctor run on a fresh machine should not fail on config
        return doctor()

    if command == "calibrate":
        from scripts.calibrate_capture import main as calibrate

        return _code(calibrate())

    if command == "run":
        from runtime.bootstrap import seed_config
        from runtime.live import main as live

        seed_config()
        return _code(live())

    if command == "supervise":
        from runtime.bootstrap import seed_config
        from runtime.supervisor import main as supervise

        seed_config()
        return _code(supervise())

    if command == "dashboard":
        from dashboard.server import main as dashboard

        return _code(dashboard())

    if command == "bench":
        from scripts.bench_llm import main as bench

        return _code(bench())

    if command == "dryrun":
        from runtime.dryrun import main as dryrun

        return _code(dryrun())

    if command == "soak":
        from runtime.soak import main as soak

        return _code(soak())

    print(f"Unknown command: {command}\n")
    print(USAGE)
    return 2


if __name__ == "__main__":
    sys.exit(main())
