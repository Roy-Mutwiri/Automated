"""Health endpoints and Prometheus metrics.

Every service exposes /health, /ready and /metrics on its own port. Deliberately
stdlib-only: this must keep answering when the rest of the process is unhealthy,
which is exactly when a heavyweight framework is most likely to be the thing
that is broken.

  /health   liveness. 200 while the process is running at all.
  /ready    readiness. 503 when DEGRADED or worse, so a supervisor can act.
  /metrics  Prometheus text format.

The distinction matters for a 24/7 system: a session with a stale market feed
is alive and should NOT be restarted -- it is correctly degrading and still
broadcasting. A session whose capture adapter has been silent for ten minutes
is a different problem. /ready separates them.
"""

from __future__ import annotations

import asyncio
import json
import logging
import threading
import time
from collections import defaultdict
from http.server import BaseHTTPRequestHandler, HTTPServer
from typing import Callable

from shared.contracts import HealthState, ServiceHealth

log = logging.getLogger(__name__)


class Metrics:
    """Minimal Prometheus registry: counters, gauges and histograms."""

    def __init__(self) -> None:
        self._counters: dict[tuple[str, tuple], float] = defaultdict(float)
        self._gauges: dict[tuple[str, tuple], float] = {}
        self._hist: dict[tuple[str, tuple], list[float]] = defaultdict(list)
        self._help: dict[str, str] = {}
        self._lock = threading.Lock()

    @staticmethod
    def _key(name: str, labels: dict[str, str] | None) -> tuple[str, tuple]:
        return name, tuple(sorted((labels or {}).items()))

    def describe(self, name: str, text: str) -> None:
        self._help[name] = text

    def inc(self, name: str, labels: dict[str, str] | None = None, by: float = 1.0) -> None:
        with self._lock:
            self._counters[self._key(name, labels)] += by

    def gauge(self, name: str, value: float, labels: dict[str, str] | None = None) -> None:
        with self._lock:
            self._gauges[self._key(name, labels)] = value

    def observe(self, name: str, value: float, labels: dict[str, str] | None = None) -> None:
        with self._lock:
            bucket = self._hist[self._key(name, labels)]
            bucket.append(value)
            if len(bucket) > 2000:
                del bucket[:-2000]

    @staticmethod
    def _fmt(name: str, labels: tuple, value: float) -> str:
        if labels:
            rendered = ",".join(f'{k}="{v}"' for k, v in labels)
            return f"{name}{{{rendered}}} {value}"
        return f"{name} {value}"

    def render(self) -> str:
        lines: list[str] = []
        with self._lock:
            for name, text in self._help.items():
                lines.append(f"# HELP {name} {text}")

            for (name, labels), value in sorted(self._counters.items()):
                lines.append(f"# TYPE {name} counter")
                lines.append(self._fmt(name, labels, value))

            for (name, labels), value in sorted(self._gauges.items()):
                lines.append(f"# TYPE {name} gauge")
                lines.append(self._fmt(name, labels, value))

            for (name, labels), values in sorted(self._hist.items()):
                if not values:
                    continue
                ordered = sorted(values)
                lines.append(f"# TYPE {name} summary")
                for q in (0.5, 0.95, 0.99):
                    idx = min(len(ordered) - 1, int(len(ordered) * q))
                    lines.append(
                        self._fmt(name, labels + (("quantile", str(q)),), ordered[idx])
                    )
                lines.append(self._fmt(f"{name}_sum", labels, sum(ordered)))
                lines.append(self._fmt(f"{name}_count", labels, len(ordered)))
        return "\n".join(lines) + "\n"


METRICS = Metrics()
METRICS.describe("goldlive_utterances_total", "Utterances spoken")
METRICS.describe("goldlive_blocked_total", "Utterances blocked, by reason")
METRICS.describe("goldlive_first_token_ms", "Time to first generated token")
METRICS.describe("goldlive_first_audio_ms", "Time to first audible sample")
METRICS.describe("goldlive_market_staleness_ms", "Age of the market snapshot")
METRICS.describe("goldlive_queue_depth", "Pending items, by queue")
METRICS.describe("goldlive_comments_total", "Comments ingested")
METRICS.describe("goldlive_session_up", "1 when the session is live")


HealthProvider = Callable[[], list[ServiceHealth]]


class HealthServer:
    """Threaded HTTP server so health keeps answering under event-loop stalls.

    If the loop is blocked, an asyncio-based endpoint stops responding at the
    exact moment a supervisor most needs an answer.
    """

    def __init__(
        self,
        provider: HealthProvider,
        port: int = 9101,
        host: str = "127.0.0.1",
        metrics: Metrics | None = None,
    ) -> None:
        self.provider = provider
        self.port = port
        self.host = host
        self.metrics = metrics or METRICS
        self._server: HTTPServer | None = None
        self._thread: threading.Thread | None = None
        self.started_at = time.time()

    def _snapshot(self) -> tuple[dict, bool]:
        try:
            components = self.provider()
        except Exception as exc:  # noqa: BLE001 - never fail the health endpoint
            return {"error": str(exc), "state": "failing"}, False

        worst = HealthState.OK
        order = [HealthState.OK, HealthState.DEGRADED, HealthState.FAILING, HealthState.DOWN]
        for c in components:
            if order.index(c.state) > order.index(worst):
                worst = c.state

        body = {
            "state": worst.value,
            "uptime_s": round(time.time() - self.started_at, 1),
            "components": [
                {
                    "component": c.component,
                    "session_id": c.session_id,
                    "state": c.state.value,
                    "degraded_reason": c.degraded_reason,
                    "restart_count_1h": c.restart_count_1h,
                    "checks": [
                        {"name": k.name, "ok": k.ok, "detail": k.detail} for k in c.checks
                    ],
                }
                for c in components
            ],
        }
        # DEGRADED is deliberately still "ready". A session with a stale feed is
        # working as designed -- it has stopped quoting prices and is still
        # broadcasting. Restarting it would be strictly worse.
        ready = worst in (HealthState.OK, HealthState.DEGRADED)
        return body, ready

    def _handler(self) -> type[BaseHTTPRequestHandler]:
        server = self

        class Handler(BaseHTTPRequestHandler):
            protocol_version = "HTTP/1.1"

            def _send(self, code: int, body: bytes, content_type: str) -> None:
                self.send_response(code)
                self.send_header("Content-Type", content_type)
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def do_GET(self) -> None:  # noqa: N802 - stdlib naming
                path = self.path.split("?")[0]
                if path == "/health":
                    body, _ = server._snapshot()
                    self._send(200, json.dumps(body).encode(), "application/json")
                elif path == "/ready":
                    body, ready = server._snapshot()
                    self._send(
                        200 if ready else 503,
                        json.dumps(body).encode(),
                        "application/json",
                    )
                elif path == "/metrics":
                    self._send(
                        200, server.metrics.render().encode(), "text/plain; version=0.0.4"
                    )
                else:
                    self._send(404, b"not found", "text/plain")

            def log_message(self, *args) -> None:
                pass  # health probes must not fill the journal

        return Handler

    def start(self) -> None:
        self._server = HTTPServer((self.host, self.port), self._handler())
        self._thread = threading.Thread(
            target=self._server.serve_forever, name="health", daemon=True
        )
        self._thread.start()
        log.info("health server on http://%s:%d/health", self.host, self.port)

    def stop(self) -> None:
        if self._server:
            self._server.shutdown()
            self._server.server_close()
            self._server = None


async def heartbeat(
    provider: HealthProvider, store, interval_s: float = 30.0
) -> None:
    """Persist health periodically so the dashboard can show history."""
    while True:
        await asyncio.sleep(interval_s)
        try:
            for component in provider():
                store.record_health(component)
        except asyncio.CancelledError:
            raise
        except Exception:  # noqa: BLE001
            log.exception("heartbeat failed")
