"""Supervisor: keeps N session processes alive.

systemd is the right answer on a Linux box and there are units in ops/ for it.
This exists because the machine described is a Windows PC running LIVE Studio,
where there is no systemd, and because it makes the fault-isolation claim
concrete and testable: kill one session and watch the others carry on.

Policy:
  - restart on exit, with exponential backoff and jitter
  - a session that fails `crash_loop_threshold` times inside an hour is left
    DOWN and alerted rather than restarted forever. A crash loop that restarts
    silently is worse than one that stops, because nobody finds out.
  - /ready returning 503 for longer than `unready_grace_s` is treated as a
    hang and the process is killed so it can be restarted. A wedged process
    that never exits is the failure supervision usually misses.
  - SIGTERM first, SIGKILL after a grace period.

    python -m runtime.supervisor
    python -m runtime.supervisor --sessions SESSION_001 SESSION_002
"""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import logging
import os
import random
import signal
import subprocess
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import yaml

from runtime import lifecycle
from shared.paths import config_path, data_root, is_frozen

ROOT = Path(__file__).resolve().parent.parent
log = logging.getLogger("supervisor")


def _errored(reason: str) -> "lifecycle.Lifecycle":
    lc = lifecycle.load(reconcile=False)
    lc.mark_error(reason)
    return lc


def session_command(session_id: str, health_port: int, args: list[str]) -> list[str]:
    """Build the argv that starts one session, correctly in both modes.

    Frozen builds were spawning `GoldLive.exe -m runtime.live ...`, because
    sys.executable is the exe rather than a Python interpreter. The exe has its
    own subcommand dispatcher and answered "Unknown command: -m", so every
    supervised session died instantly and the restart logic turned that into a
    spawn loop. `GoldLive.exe supervise` had therefore never worked in the
    artifact intended for distribution.

    Pure function on purpose: the argv is the thing that was wrong, so it needs
    to be assertable without starting a process.
    """
    tail = ["--session", session_id, "--health-port", str(health_port), *args]
    if is_frozen():
        return [sys.executable, "run", *tail]
    return [sys.executable, "-m", "runtime.live", *tail]



@dataclass
class ManagedSession:
    session_id: str
    health_port: int
    args: list[str]
    process: subprocess.Popen | None = None
    restarts: list[float] = field(default_factory=list)
    backoff_s: float = 1.0
    state: str = "starting"
    started_at: float | None = None
    last_ready_at: float | None = None
    give_up: bool = False

    def restarts_last_hour(self) -> int:
        cutoff = time.time() - 3600
        self.restarts = [t for t in self.restarts if t > cutoff]
        return len(self.restarts)

    @property
    def alive(self) -> bool:
        return self.process is not None and self.process.poll() is None

    @property
    def uptime_s(self) -> float:
        return time.time() - self.started_at if self.started_at else 0.0


class Supervisor:
    def __init__(
        self,
        sessions: list[ManagedSession],
        crash_loop_threshold: int = 6,
        max_backoff_s: float = 60.0,
        unready_grace_s: float = 120.0,
        term_grace_s: float = 10.0,
        allow_degraded: bool = False,
        market: str = "gold",
        adapter: str = "file",
        ignore_consent: bool = False,
    ) -> None:
        self.sessions = {s.session_id: s for s in sessions}
        self.allow_degraded = allow_degraded
        #: Tests and `--no-consent-gate` bypass the recorded intent.
        self.ignore_consent = ignore_consent
        self.market = market
        self.adapter = adapter
        self.crash_loop_threshold = crash_loop_threshold
        self.max_backoff_s = max_backoff_s
        self.unready_grace_s = unready_grace_s
        self.term_grace_s = term_grace_s
        self.stopping = asyncio.Event()

    def consent_to_run(self) -> bool:
        """Has the user asked for GoldLive to be running?

        Checked before every spawn and every restart, not once at startup, so
        pressing STOP takes effect on the next supervision tick rather than
        whenever the current session happens to end.
        """
        if self.ignore_consent:
            return True
        try:
            return lifecycle.load(reconcile=False).may_run
        except Exception:
            # If the intent cannot be read, do not run. The safe default when
            # consent is unknown is no.
            log.exception("could not read lifecycle state; refusing to start")
            return False

    # -- startup readiness ------------------------------------------------

    async def preflight(self) -> bool:
        """Refuse to start sessions this machine cannot actually run.

        Without this, a machine with no model and no voices spawns sessions
        that come up "healthy" and broadcast canned text over a placeholder
        tone -- and the supervisor dutifully keeps them alive. Failing here,
        loudly, is the whole point of the readiness work.

        Development modes bypass it explicitly via --allow-degraded, which is
        logged so a degraded run is never mistaken for a real one.
        """
        if self.allow_degraded:
            log.warning(
                "--allow-degraded: starting without readiness checks. "
                "Market data, model output or audio may be simulated."
            )
            return True

        from runtime.readiness import Level, check

        log.info("checking readiness before starting any session ...")
        result = await check(
            session_id=next(iter(self.sessions), "SESSION_001"),
            market=self.market, adapter=self.adapter, include_broadcast=True,
        )
        print()
        print(result.render())
        print()

        if result.level is Level.NOT_READY:
            log.error(
                "not starting: %s",
                ", ".join(g.name for g in result.failed(("config_valid", "market_live",
                                                         "model_real", "voice_real",
                                                         "audio_out"))),
            )
            log.error("run `GoldLive.exe provision` to fix, or pass --allow-degraded "
                      "to start anyway (development only).")
            return False

        if result.level is Level.SESSION_READY:
            log.warning("session ready, but NOT broadcast ready -- "
                        "audio will not reach LIVE Studio.")
        return True

    # -- process control --------------------------------------------------

    def spawn(self, session: ManagedSession) -> None:
        # The consent gate. Without it, "the child is missing" and "the user
        # pressed STOP" are indistinguishable from in here, and the restart
        # logic would resurrect a session the user just shut down.
        if not self.consent_to_run():
            log.info("not starting %s: the user has not asked GoldLive to run",
                     session.session_id)
            return

        cmd = session_command(session.session_id, session.health_port, session.args)
        log.info("starting %s: %s", session.session_id, " ".join(cmd[1:]))

        # Children skip their own provisioning and readiness work: the
        # supervisor has already done it once, and N sessions each probing the
        # market feed and generating a test utterance would be both slow and
        # pointless.
        env = {**os.environ, "GOLDLIVE_SUPERVISED": "1"}

        # cwd must be writable and must outlive the process. ROOT resolves
        # inside PyInstaller's temporary _MEIPASS directory when frozen, which
        # is deleted on exit and read-only in practice.
        session.process = subprocess.Popen(
            cmd, cwd=str(data_root()), env=env, stdout=None, stderr=None
        )
        session.started_at = time.time()
        session.last_ready_at = time.time()
        session.state = "running"

    async def terminate(self, session: ManagedSession) -> None:
        """SIGTERM, then SIGKILL. The session flushes traces on SIGTERM, so
        give it a real chance before killing it."""
        if not session.alive:
            return
        assert session.process is not None
        log.info("stopping %s (pid %d)", session.session_id, session.process.pid)
        with contextlib.suppress(Exception):
            session.process.terminate()

        deadline = time.time() + self.term_grace_s
        while time.time() < deadline and session.alive:
            await asyncio.sleep(0.2)
        if session.alive:
            log.warning("%s ignored SIGTERM, killing", session.session_id)
            with contextlib.suppress(Exception):
                session.process.kill()
        session.state = "stopped"

    # -- readiness --------------------------------------------------------

    async def check_ready(self, session: ManagedSession) -> bool:
        try:
            import httpx

            async with httpx.AsyncClient(timeout=3.0) as client:
                resp = await client.get(f"http://127.0.0.1:{session.health_port}/ready")
                return resp.status_code == 200
        except Exception:
            return False

    # -- the loop ---------------------------------------------------------

    async def _watch(self, session: ManagedSession) -> None:
        while not self.stopping.is_set():
            if session.give_up:
                await asyncio.sleep(30)
                continue

            if not self.consent_to_run():
                if session.state != "stopped":
                    log.info("user stopped GoldLive; not restarting %s",
                             session.session_id)
                    session.state = "stopped"
                self.stopping.set()
                return

            if not session.alive:
                code = session.process.returncode if session.process else "n/a"
                if session.state == "running":
                    log.warning("%s exited (code %s)", session.session_id, code)
                    session.restarts.append(time.time())

                if session.restarts_last_hour() >= self.crash_loop_threshold:
                    session.give_up = True
                    session.state = "crash_loop"
                    log.error(
                        "%s crash-looped %d times in an hour; leaving it DOWN. "
                        "A loop that restarts silently is worse than one that stops.",
                        session.session_id, session.restarts_last_hour(),
                    )
                    continue

                # Jitter so seven sessions do not all restart in lockstep and
                # hammer the model server at the same instant.
                delay = session.backoff_s * (0.7 + 0.6 * random.random())
                log.info("restarting %s in %.1fs", session.session_id, delay)
                await asyncio.sleep(delay)
                session.backoff_s = min(session.backoff_s * 2, self.max_backoff_s)
                self.spawn(session)
                await asyncio.sleep(5)
                continue

            # Alive: reset backoff once it has been up a while, then check that
            # it is actually serving rather than merely running.
            if session.uptime_s > 120:
                session.backoff_s = 1.0

            if await self.check_ready(session):
                session.last_ready_at = time.time()
                session.state = "running"
            else:
                unready_for = time.time() - (session.last_ready_at or time.time())
                session.state = "unready"
                if unready_for > self.unready_grace_s:
                    log.error(
                        "%s not ready for %.0fs; killing so it can restart",
                        session.session_id, unready_for,
                    )
                    await self.terminate(session)
            await asyncio.sleep(10)

    async def run(self) -> int:
        if not await self.preflight():
            if not self.ignore_consent:
                lifecycle.save(_errored("readiness gates failed"))
            return 2

        # Record that a supervisor now owns the run, with its pid, so a hard
        # kill leaves evidence of a crash rather than a stale RUNNING that the
        # control panel would show as a healthy green light.
        if not self.ignore_consent:
            lc = lifecycle.load(reconcile=False)
            lc.mark_running(os.getpid())
            lifecycle.save(lc)

        for session in self.sessions.values():
            self.spawn(session)
            await asyncio.sleep(2)  # stagger startup; model load is expensive

        watchers = [
            asyncio.create_task(self._watch(s), name=f"watch:{s.session_id}")
            for s in self.sessions.values()
        ]
        reporter = asyncio.create_task(self._report(), name="report")

        await self.stopping.wait()
        for task in [*watchers, reporter]:
            task.cancel()
        await asyncio.gather(
            *(self.terminate(s) for s in self.sessions.values()), return_exceptions=True
        )
        log.info("all sessions stopped")
        if not self.ignore_consent:
            lifecycle.mark_stopped()
        return 0

    async def _report(self) -> None:
        while not self.stopping.is_set():
            await asyncio.sleep(60)
            rows = [
                f"{s.session_id}={s.state}"
                f"(up {s.uptime_s / 60:.0f}m, {s.restarts_last_hour()} restarts/h)"
                for s in self.sessions.values()
            ]
            log.info("status: %s", "  ".join(rows))

    def snapshot(self) -> dict:
        return {
            "at": datetime.now(timezone.utc).isoformat(),
            "sessions": [
                {
                    "session_id": s.session_id,
                    "state": s.state,
                    "alive": s.alive,
                    "uptime_s": round(s.uptime_s),
                    "restarts_1h": s.restarts_last_hour(),
                    "health_port": s.health_port,
                    "given_up": s.give_up,
                }
                for s in self.sessions.values()
            ],
        }


def build_sessions(only: list[str] | None, passthrough: list[str]) -> list[ManagedSession]:
    cfg = yaml.safe_load(config_path("sessions.yaml").read_text(encoding="utf-8"))
    sessions = []
    for i, spec in enumerate(cfg["sessions"]):
        if only and spec["session_id"] not in only:
            continue
        sessions.append(
            ManagedSession(
                session_id=spec["session_id"],
                health_port=9101 + i,
                args=passthrough,
            )
        )
    if not sessions:
        raise SystemExit("no sessions selected")
    return sessions


def main() -> None:
    ap = argparse.ArgumentParser(description="Supervise session processes")
    ap.add_argument("--sessions", nargs="*", help="defaults to all in sessions.yaml")
    # Production defaults. `supervise` with no arguments must run the real
    # thing: a real feed, real speech, a real model. synthetic/mock/file are
    # development modes and have to be asked for by name.
    ap.add_argument("--mode", default="auto")
    ap.add_argument("--market", default="gold")
    ap.add_argument("--adapter", default="file")
    ap.add_argument("--tts", default="piper")
    ap.add_argument(
        "--allow-degraded", action="store_true",
        help="start even when readiness gates fail (development only)",
    )
    ap.add_argument(
        "--no-consent-gate", action="store_true",
        help="ignore the recorded start/stop intent (tests and debugging only)",
    )
    args = ap.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)-7s %(name)s: %(message)s",
    )

    passthrough = [
        "--mode", args.mode, "--market", args.market,
        "--adapter", args.adapter, "--tts", args.tts,
    ]
    if not args.no_consent_gate:
        # Typing `GoldLive.exe supervise` is itself an explicit start request.
        # The control panel sets the same state before launching this process;
        # doing it here too means the command works standalone without ever
        # letting the supervisor run without a recorded intent.
        lifecycle.request_start()

    supervisor = Supervisor(
        build_sessions(args.sessions, passthrough),
        ignore_consent=args.no_consent_gate,
        allow_degraded=args.allow_degraded,
        market=args.market,
        adapter=args.adapter,
    )

    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

    def request_stop(*_a) -> None:
        loop.call_soon_threadsafe(supervisor.stopping.set)

    for sig in (signal.SIGINT, signal.SIGTERM):
        with contextlib.suppress(NotImplementedError, AttributeError):
            signal.signal(sig, request_stop)

    try:
        sys.exit(loop.run_until_complete(supervisor.run()))
    finally:
        loop.close()


if __name__ == "__main__":
    main()
