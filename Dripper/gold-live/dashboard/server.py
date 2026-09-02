"""Operator dashboard.

Aggregates every session's /health and the trace store into one page, and
provides the controls an operator actually needs at 3am. Stdlib only, for the
same reason as the health server: the monitoring surface must not depend on the
thing it is monitoring being healthy.

    python -m dashboard.server --port 8080

Controls are deliberately limited to what is safe to expose without auth on
localhost. Bind to 127.0.0.1 and reach it over SSH; do not put this on a public
interface as it stands -- there is no authentication.
"""

from __future__ import annotations

import argparse
import json
import logging
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

import yaml

from shared.store import TraceStore

from shared.paths import config_path, data_path

ROOT = Path(__file__).resolve().parent.parent
log = logging.getLogger("dashboard")


def session_ports() -> dict[str, int]:
    cfg = yaml.safe_load(config_path("sessions.yaml").read_text(encoding="utf-8"))
    return {s["session_id"]: 9101 + i for i, s in enumerate(cfg["sessions"])}


def fetch_health(port: int, timeout: float = 2.0) -> dict:
    try:
        with urllib.request.urlopen(  # noqa: S310 - fixed localhost URL
            f"http://127.0.0.1:{port}/health", timeout=timeout
        ) as resp:
            return json.loads(resp.read())
    except (urllib.error.URLError, TimeoutError, OSError) as exc:
        return {"state": "down", "error": str(exc), "components": []}


PAGE = """<!doctype html>
<meta charset="utf-8">
<title>Gold Live</title>
<meta http-equiv="refresh" content="10">
<style>
:root{--bg:#f7f6f2;--fg:#17181a;--muted:#6b6b64;--rule:#e1dfd6;--card:#fff;
--ok:#3d6b4f;--warn:#8a6a12;--bad:#9e3527}
@media(prefers-color-scheme:dark){:root{--bg:#121316;--fg:#e9e7e1;--muted:#9c988d;
--rule:#2c2e33;--card:#1a1b1f;--ok:#6faf85;--warn:#d4a83c;--bad:#e0705c}}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--fg);
font:14px/1.5 ui-monospace,SFMono-Regular,Consolas,monospace;padding:24px}
h1{font-size:18px;margin:0 0 4px;letter-spacing:-.01em}
.sub{color:var(--muted);font-size:12px;margin-bottom:22px}
.grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(300px,1fr));gap:12px}
.card{background:var(--card);border:1px solid var(--rule);border-radius:5px;padding:14px}
.card.ok{border-left:3px solid var(--ok)}
.card.degraded{border-left:3px solid var(--warn)}
.card.down,.card.failing{border-left:3px solid var(--bad)}
.name{font-weight:600;display:flex;justify-content:space-between;align-items:center}
.pill{font-size:10px;padding:2px 7px;border-radius:3px;letter-spacing:.06em;
text-transform:uppercase}
.pill.ok{background:color-mix(in srgb,var(--ok) 18%,transparent);color:var(--ok)}
.pill.degraded{background:color-mix(in srgb,var(--warn) 18%,transparent);color:var(--warn)}
.pill.down,.pill.failing{background:color-mix(in srgb,var(--bad) 18%,transparent);
color:var(--bad)}
table{width:100%;border-collapse:collapse;margin-top:9px;font-size:12px}
td{padding:3px 0;border-bottom:1px solid var(--rule);vertical-align:top}
td:last-child{text-align:right;color:var(--muted)}
.reason{margin-top:9px;font-size:12px;color:var(--warn)}
h2{font-size:13px;margin:28px 0 8px;color:var(--muted);letter-spacing:.06em;
text-transform:uppercase}
.utt{background:var(--card);border:1px solid var(--rule);border-radius:5px;
padding:11px 13px;margin-bottom:7px}
.meta{color:var(--muted);font-size:11px;margin-bottom:4px;
display:flex;gap:12px;flex-wrap:wrap}
.empty{color:var(--muted);padding:20px 0}
</style>
<h1>Gold Live</h1>
<div class="sub">%(when)s &middot; refreshes every 10s</div>
<div class="grid">%(cards)s</div>
<h2>Recent utterances</h2>
%(utterances)s
"""


def render_card(session_id: str, port: int, health: dict) -> str:
    state = health.get("state", "down")
    rows = []
    for component in health.get("components", []):
        for check in component.get("checks", []):
            mark = "ok" if check["ok"] else "FAIL"
            rows.append(
                f"<tr><td>{component['component']}.{check['name']}</td>"
                f"<td>{check.get('detail') or mark}</td></tr>"
            )
    reasons = [
        c["degraded_reason"] for c in health.get("components", []) if c.get("degraded_reason")
    ]
    reason_html = f'<div class="reason">{"; ".join(reasons)}</div>' if reasons else ""
    uptime = health.get("uptime_s")
    uptime_html = f"{uptime / 60:.0f}m" if isinstance(uptime, (int, float)) else "-"

    return (
        f'<div class="card {state}">'
        f'<div class="name">{session_id}<span class="pill {state}">{state}</span></div>'
        f'<table><tr><td>uptime</td><td>{uptime_html}</td></tr>'
        f'<tr><td>health port</td><td>{port}</td></tr>{"".join(rows)}</table>'
        f"{reason_html}</div>"
    )


class Dashboard:
    def __init__(self, store: TraceStore) -> None:
        self.store = store
        self.ports = session_ports()
        self.pool = ThreadPoolExecutor(max_workers=8)

    def render(self) -> str:
        from datetime import datetime, timezone

        healths = dict(
            zip(
                self.ports,
                self.pool.map(fetch_health, self.ports.values()),
                strict=True,
            )
        )
        cards = "".join(
            render_card(sid, self.ports[sid], healths[sid]) for sid in self.ports
        )

        try:
            recent = self.store.recent_utterances(limit=25)
        except Exception as exc:  # noqa: BLE001 - dashboard must still render
            recent = []
            log.warning("trace read failed: %s", exc)

        if recent:
            blocks = []
            for u in recent:
                latency = (
                    f"{u['first_token_ms']}ms to first token"
                    if u.get("first_token_ms")
                    else ""
                )
                blocks.append(
                    f'<div class="utt"><div class="meta">'
                    f'<span>{u["session_id"]}</span>'
                    f'<span>{u["created_at"][11:19]}</span>'
                    f'<span>{u["trigger_type"]}</span>'
                    f'<span>{u.get("market_confidence") or "-"}</span>'
                    f"<span>{latency}</span></div>{u['text']}</div>"
                )
            utterances = "".join(blocks)
        else:
            utterances = '<div class="empty">Nothing recorded yet.</div>'

        return PAGE % {
            "when": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC"),
            "cards": cards,
            "utterances": utterances,
        }

    def api(self) -> dict:
        return {
            "sessions": {
                sid: fetch_health(port) for sid, port in self.ports.items()
            },
            "stats": {
                sid: self.store.stats(sid) for sid in self.ports
            },
        }


def make_handler(dash: Dashboard) -> type[BaseHTTPRequestHandler]:
    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def _send(self, code: int, body: bytes, ctype: str) -> None:
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self) -> None:  # noqa: N802 - stdlib naming
            path = self.path.split("?")[0]
            try:
                if path in ("/", "/index.html"):
                    self._send(200, dash.render().encode(), "text/html; charset=utf-8")
                elif path == "/api/status":
                    self._send(
                        200, json.dumps(dash.api()).encode(), "application/json"
                    )
                elif path.startswith("/api/explain/"):
                    trace = path.rsplit("/", 1)[-1]
                    found = dash.store.explain(trace)
                    self._send(
                        200 if found else 404,
                        json.dumps(found or {"error": "not found"}).encode(),
                        "application/json",
                    )
                else:
                    self._send(404, b"not found", "text/plain")
            except Exception as exc:  # noqa: BLE001 - never 500 silently
                log.exception("dashboard request failed")
                self._send(500, str(exc).encode(), "text/plain")

        def log_message(self, *args) -> None:
            pass

    return Handler


def main() -> None:
    ap = argparse.ArgumentParser(description="Gold Live operator dashboard")
    ap.add_argument("--port", type=int, default=8080)
    ap.add_argument("--host", default="127.0.0.1",
                    help="keep on localhost; there is no authentication")
    ap.add_argument("--db", default=str(data_path("data", "gold-live.db", create_parent=False)))
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    store = TraceStore(args.db)
    store.migrate()

    server = HTTPServer((args.host, args.port), make_handler(Dashboard(store)))
    log.info("dashboard on http://%s:%d", args.host, args.port)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
