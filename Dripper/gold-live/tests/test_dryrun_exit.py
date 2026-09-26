"""The dry run's exit code, which is what makes it usable as a CI gate.

`run()` counts two things that must not have happened -- utterances the safety
gate had to block, and any cross-session trace leakage -- and returns the total.
That return value was computed and then dropped on the floor by `main()`, so a
dry run could print

    FAIL: SESSION_002 shares traces with SESSION_003

and still exit 0. Nothing downstream could tell a clean run from a leaking one,
which is the same silent-success failure `runtime/readiness.py` was written to
prevent, and it is why the harness could not be wired into CI before now.

These tests pin the contract rather than the wording: whatever `run()` counts,
`main()` must turn a non-zero count into a non-zero exit.
"""

from __future__ import annotations

import sys

import pytest

from runtime import dryrun


@pytest.fixture
def argv(monkeypatch, tmp_path):
    monkeypatch.setattr(
        sys, "argv",
        ["dryrun", "--mode", "offline", "--beats", "1", "--out", str(tmp_path)],
    )


def _returning(value: int):
    async def fake_run(beats, n_sessions, mode, out_dir):
        return value
    return fake_run


@pytest.mark.parametrize("failures", [1, 2, 7])
def test_a_failing_run_exits_non_zero(monkeypatch, argv, failures):
    """The regression. Any non-zero count must reach the shell."""
    monkeypatch.setattr(dryrun, "run", _returning(failures))
    with pytest.raises(SystemExit) as exc:
        dryrun.main()
    assert exc.value.code == 1, "a detected failure must not exit 0"


def test_a_clean_run_exits_zero(monkeypatch, argv):
    monkeypatch.setattr(dryrun, "run", _returning(0))
    with pytest.raises(SystemExit) as exc:
        dryrun.main()
    assert exc.value.code == 0


async def test_a_clean_two_session_run_counts_zero(tmp_path):
    """The integration end: a real two-session run, exercising the isolation
    check for real, must count nothing.

    This is deliberately not a test that a leak *is* counted -- forcing two
    runtimes to share a trace_id from outside `run()` would test the stub, not
    the system. The leak-detection logic itself is covered adversarially in
    tests/test_isolation.py; what is pinned here is that the real path returns
    a number, and that a healthy system's number is zero.
    """
    failures = await dryrun.run(2, 2, "offline", tmp_path)
    assert failures == 0, f"a clean run must count no failures, got {failures}"
