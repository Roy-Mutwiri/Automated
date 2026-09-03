"""The watchdog, run as a test.

A diagnostic that has to be remembered is a diagnostic that will not be run.
Every check in `presenter.behavior.watchdog` corresponds to a defect that
shipped into a rendered video at least once while the whole test suite was
green, so the watchdog belongs in the suite rather than in a tool somebody has
to think to invoke.

Kept to a few seeds and a few minutes so it stays runnable; the longer sweep
lives in `tools/state_watchdog.py`.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from presenter.behavior.engine import BehaviorEngine
from presenter.behavior.watchdog import Watchdog


def _run(seed: int, minutes: float = 8.0, fps: float = 30.0) -> Watchdog:
    engine = BehaviorEngine(seed=seed)
    watch = Watchdog()
    dt = 1.0 / fps
    for _ in range(int(minutes * 60 * fps)):
        engine.update(dt)
        watch.update(dt, engine.motion, engine.attention)
    return watch


@pytest.mark.parametrize("seed", [20260902, 7, 33])
def test_no_watchdog_warnings(seed):
    """No stare, no drift, no frozen posture, no absent camera contact."""
    watch = _run(seed)
    warnings = watch.check()
    assert not warnings, f"seed {seed}: " + "; ".join(warnings)


def test_watchdog_catches_a_frozen_head():
    """The watchdog must fail on the failure it was written for.

    A check that has never been seen to fire is not evidence of anything. This
    replays the actual regression - a head whose yaw barely moves, which is
    what "an image that sometimes moves" measures like - and asserts the
    watchdog says so.
    """
    watch = _run(20260902, minutes=2.0)
    # Flatten the head trajectory, leave everything else as it was.
    watch._yaw = [0.4] * len(watch._yaw)
    warnings = watch.check()
    assert any("barely turns" in w for w in warnings), warnings
    assert any("no large head turns" in w for w in warnings), warnings


def test_watchdog_catches_a_stuck_fixation():
    """A four-minute stare at one target is a defect, not a long think."""
    watch = _run(20260902, minutes=2.0)
    watch._longest_fixation = 240.0
    watch._longest_target = "MAIN_MONITOR_CENTER"
    warnings = watch.check()
    assert any("stuck fixation" in w for w in warnings), warnings


def test_watchdog_catches_parked_pitch():
    """Pitch that sits outside the comfort band for minutes is the drift bug."""
    watch = _run(20260902, minutes=2.0)
    watch._max_out_of_band = 130.0
    warnings = watch.check()
    assert any("pitch parked" in w for w in warnings), warnings


def test_watchdog_catches_accumulating_posture():
    """Posture must revert to neutral, not wander off and stay there."""
    watch = _run(20260902, minutes=2.0)
    watch._eng = [0.8] * len(watch._eng)
    warnings = watch.check()
    assert any("not mean-reverting" in w for w in warnings), warnings
