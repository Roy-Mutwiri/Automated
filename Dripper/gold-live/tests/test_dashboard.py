"""Dashboard rendering and operator controls.

The page silently returned HTTP 500 for a while: the CSS contains percentage
values (color-mix) which collided with %-formatting in the template. It went
unnoticed because the JSON API kept working and that is what got checked.
These render the actual HTML, which is the only way that class of bug shows up.
"""

from __future__ import annotations

import json


from dashboard.server import (
    Dashboard,
    post_control,
    render_card,
    render_controls,
    session_ports,
)
from shared.store import TraceStore


def healthy(paused: bool = False) -> dict:
    return {
        "state": "ok",
        "uptime_s": 3600,
        "components": [
            {
                "component": "session",
                "session_id": "SESSION_001",
                "state": "ok",
                "degraded_reason": "paused" if paused else None,
                "checks": [{"name": "director_queue", "ok": True, "detail": "depth=0"}],
            }
        ],
    }


# -- rendering -------------------------------------------------------------


def test_page_renders_without_a_format_error(tmp_path):
    """The regression that mattered: CSS percentages breaking the template."""
    store = TraceStore(tmp_path / "d.db")
    store.migrate()
    html = Dashboard(store).render()

    assert len(html) > 1000
    assert "<h1>Gold Live</h1>" in html
    # No unsubstituted placeholders left behind.
    for leftover in ("__JS__", "__WHEN__", "__CARDS__", "__UTTERANCES__"):
        assert leftover not in html, f"{leftover} was not substituted"


def test_css_percentages_survive_rendering(tmp_path):
    store = TraceStore(tmp_path / "d.db")
    store.migrate()
    html = Dashboard(store).render()
    assert "color-mix" in html and "18%" in html


def test_card_shows_state_and_checks():
    html = render_card("SESSION_001", 9101, healthy())
    assert "SESSION_001" in html
    assert "session.director_queue" in html
    assert 'class="card ok"' in html


def test_down_session_renders_rather_than_erroring():
    html = render_card("SESSION_004", 9104, {"state": "down", "components": []})
    assert 'class="card down"' in html
    assert "SESSION_004" in html


def test_degraded_reason_is_surfaced():
    health = healthy()
    health["components"][0]["degraded_reason"] = "market data stale; price quoting disabled"
    assert "price quoting disabled" in render_card("SESSION_001", 9101, health)


# -- controls --------------------------------------------------------------


def test_controls_offer_pause_when_running():
    html = render_controls(9101, paused=False)
    assert "ctl(9101,'pause')" in html
    assert "ctl(9101,'resume')" not in html


def test_controls_offer_resume_when_paused():
    html = render_controls(9101, paused=True)
    assert "ctl(9101,'resume')" in html
    assert "ctl(9101,'pause')" not in html


def test_every_control_is_reachable_from_the_page():
    html = render_controls(9101, paused=False)
    for action in ("pause", "mute", "unmute", "skip"):
        assert f"ctl(9101,'{action}')" in html
    assert "say(9101)" in html


def test_page_warns_that_controls_are_live_and_unauthenticated(tmp_path):
    store = TraceStore(tmp_path / "d.db")
    store.migrate()
    html = Dashboard(store).render()
    assert "No authentication" in html
    assert "immediately" in html


def test_control_to_a_dead_session_returns_an_error_not_an_exception():
    # Port 1 is reserved and never listening.
    result = post_control(1, "pause")
    assert "error" in result


# -- config ----------------------------------------------------------------


def test_each_session_gets_a_distinct_port():
    ports = session_ports()
    assert len(set(ports.values())) == len(ports)
    assert all(p >= 9101 for p in ports.values())


def test_api_status_shape(tmp_path):
    store = TraceStore(tmp_path / "d.db")
    store.migrate()
    payload = Dashboard(store).api()
    assert "sessions" in payload and "stats" in payload
    assert set(payload["sessions"]) == set(session_ports())
    json.dumps(payload)  # must be serialisable


def test_explain_returns_none_for_an_unknown_trace(tmp_path):
    store = TraceStore(tmp_path / "d.db")
    store.migrate()
    assert store.explain("no-such-trace") is None
