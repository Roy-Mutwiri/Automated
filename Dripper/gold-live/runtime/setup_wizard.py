"""GoldLive Setup: prepare this PC, then stop.

The one rule this file exists to enforce:

    Setup INSTALLS. It never RUNS.

Provisioning downloads a model and a voice, writes configuration, and proves
the pieces work. It deliberately ends at "READY TO START" and hands control
back to the user. Nothing here starts a session, and nothing here arranges for
one to start later -- no service, no scheduled task, no registry key, no
Startup folder entry.

It reuses the existing provisioning and readiness systems rather than
duplicating them; the only new thing is a face a non-developer can read.
"""

from __future__ import annotations

import asyncio
import logging
import sys

from runtime import lifecycle
from shared.paths import data_root
from shared.version import app_version

log = logging.getLogger(__name__)

LINE = "=" * 64


def _say(text: str = "") -> None:
    print(text, flush=True)


def _step(ok: bool | None, label: str, detail: str = "") -> None:
    mark = {True: "[ok]  ", False: "[!]   ", None: "[..]  "}[ok]
    _say(f"  {mark}{label:<30}{detail}")


def run_setup(skip_model: bool = False) -> int:
    """Provision this machine and report readiness. Returns a process exit code."""
    from runtime.provision import ProvisionError, provision
    from runtime.readiness import Level, check

    _say()
    _say(LINE)
    _say(f"  GOLDLIVE {app_version()}  --  first-time setup")
    _say(LINE)
    _say()
    _say("  This prepares your PC to run GoldLive.")
    _say("  It will NOT start GoldLive. You do that yourself afterwards.")
    _say()
    _say(f"  Everything is installed under:\n    {data_root()}")
    _say()

    try:
        provision(session_id="SESSION_001", include_model=not skip_model, echo=_say)
    except ProvisionError as exc:
        _say()
        _say(LINE)
        _say(f"  SETUP COULD NOT FINISH  --  stage: {exc.stage}")
        _say(LINE)
        _say()
        _say(f"  {exc.reason}")
        if exc.action:
            _say()
            _say("  Manual action required:")
            for line in exc.action.splitlines():
                _say(f"    {line}")
        _say()
        _say("  Fix the above and run GoldLive Setup again.")
        _say()
        return 1
    except Exception as exc:  # setup must fail readably, not with a traceback
        log.exception("setup failed")
        _say()
        _say(f"  SETUP FAILED: {exc}")
        _say()
        return 1

    _say()
    _say("  Checking that everything actually works ...")
    _say()
    result = asyncio.run(check(session_id="SESSION_001", market="gold", adapter="file"))
    for gate in result.gates:
        _step(gate.ok, gate.name, gate.evidence if gate.ok else gate.reason)

    manual = [g for g in result.gates if not g.ok and g.action]
    if manual:
        _say()
        _say("  Manual action required for full broadcasting:")
        for gate in manual:
            _say(f"    - {gate.name}: {gate.action}")

    # Setup must never leave the machine believing it should be running.
    lifecycle.mark_stopped()

    _say()
    _say(LINE)
    if result.level is Level.NOT_READY:
        _say("  SETUP INCOMPLETE  --  GoldLive cannot start yet")
        _say(LINE)
        _say()
        _say("  Fix the items marked [!] above, then run GoldLive Setup again.")
        _say()
        return 1

    _say("  READY TO START")
    _say(LINE)
    _say()
    _say(f"  Readiness level: {result.level.value}")
    if result.level is Level.SESSION_READY:
        _say("  The host can speak about real prices, but broadcast audio")
        _say("  is not configured yet, so nothing reaches a stream.")
    _say()
    _say("  GoldLive is NOT running. To start it:")
    _say("    open GoldLive.exe and press START")
    _say()
    _say("  It will never start on its own, and it does not start with Windows.")
    _say()
    return 0


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    skip_model = "--skip-model" in sys.argv
    code = run_setup(skip_model=skip_model)

    # Setup is usually launched by double-click, so the window would vanish
    # before the result could be read.
    if sys.stdout.isatty():
        try:
            _say()
            input("  Press Enter to close this window, then open GoldLive.exe ... ")
        except (EOFError, KeyboardInterrupt):
            pass
    return code
