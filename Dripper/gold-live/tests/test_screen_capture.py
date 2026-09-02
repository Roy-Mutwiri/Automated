"""Screen-capture adapter: parsing, dedupe, the hash gate, and stall detection.

The pipeline is deliberately synchronous and pure at its core (`process_frame`)
so all of this is testable without a screen, a GPU, or LIVE Studio.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from platform_.adapters.ocr import FakeOCR, OCRLine
from platform_.adapters.screen import (
    ScreenCaptureAdapter,
    frame_hash,
    parse_row,
)
from shared.contracts import CaptureCalibration, HealthState

np = pytest.importorskip("numpy")


def cal(**kw) -> CaptureCalibration:
    return CaptureCalibration(
        crop_x=100, crop_y=200, crop_w=400, crop_h=600,
        fps=5, min_ocr_confidence=0.55, **kw
    )


def adapter(frames: list[list[OCRLine]], **kw) -> ScreenCaptureAdapter:
    return ScreenCaptureAdapter(
        session_id="SESSION_001", calibration=cal(), ocr=FakeOCR(frames),
        author_salt="salt", frame_source=object(), **kw  # type: ignore[arg-type]
    )


def img(seed: int, h: int = 64, w: int = 64):
    rng = np.random.default_rng(seed)
    return rng.integers(0, 255, size=(h, w, 3), dtype=np.uint8)


def line(text: str, conf: float = 0.9, y: int = 0) -> OCRLine:
    return OCRLine(text=text, confidence=conf, y=y)


# -- row parsing -----------------------------------------------------------


@pytest.mark.parametrize(
    "raw,author,text",
    [
        ("zaraFX: where is resistance?", "zaraFX", "where is resistance?"),
        ("trader_mike : gm everyone", "trader_mike", "gm everyone"),
        ("kev • what is a sweep", "kev", "what is a sweep"),
        ("sam — nice stream", "sam", "nice stream"),
    ],
)
def test_parses_author_and_text(raw, author, text):
    assert parse_row(line(raw)) == (author, text)


@pytest.mark.parametrize(
    "raw",
    [
        "someone joined",
        "goldbug99 liked the stream",
        "welcome",
        "x99",
        "no separator here",
        "",
        "   ",
    ],
)
def test_rejects_non_comment_rows(raw):
    assert parse_row(line(raw)) is None


def test_author_is_length_bounded():
    """A mis-segmented row must not become a 300-character 'author'."""
    assert parse_row(line("x" * 200 + ": hello")) is None


# -- frame hashing ---------------------------------------------------------


def test_identical_frames_hash_identically():
    a = img(1)
    assert frame_hash(a) == frame_hash(a.copy())


def test_different_frames_hash_differently():
    assert frame_hash(img(1)) != frame_hash(img(2))


def test_unchanged_frame_skips_ocr_entirely():
    """The gate that stops seven OCR pipelines running continuously."""
    a = adapter([[line("zara: hi")], [line("zara: hi")]])
    same = img(7)

    first = a.process_frame(same)
    second = a.process_frame(same.copy())

    assert len(first) == 1
    assert second == []
    assert a.stats.frames == 2
    assert a.stats.ocr_runs == 1, "second frame must not reach OCR"
    assert a.stats.skip_rate == 0.5


# -- extraction ------------------------------------------------------------


def test_emits_comment_events_scoped_to_the_session():
    a = adapter([[line("zaraFX: where is resistance?")]])
    events = a.process_frame(img(1))
    assert len(events) == 1
    e = events[0]
    assert e.session_id == "SESSION_001"
    assert e.text_raw == "where is resistance?"
    assert e.text_norm == "where is resistance?"
    assert e.platform == "tiktok"


def test_author_name_is_never_stored():
    a = adapter([[line("realusername123: hello")]])
    e = a.process_frame(img(1))[0]
    assert "realusername123" not in e.author_hash
    assert "realusername123" not in e.text_raw


def test_low_confidence_rows_are_dropped_not_guessed():
    a = adapter([[line("zara: legible", 0.9), line("xx: g4rbl3d", 0.2)]])
    events = a.process_frame(img(1))
    assert len(events) == 1
    assert a.stats.rows_dropped_confidence == 1


def test_overlay_rows_are_filtered():
    a = adapter([[line("someone joined"), line("zara: real comment")]])
    assert len(a.process_frame(img(1))) == 1
    assert a.stats.rows_dropped_overlay == 1


# -- dedupe ----------------------------------------------------------------


def test_same_comment_across_scroll_positions_emits_once():
    """The core problem: no message ids, and each comment appears in many
    frames at different vertical positions."""
    a = adapter([
        [line("zara: where is resistance?", y=100)],
        [line("zara: where is resistance?", y=70), line("kev: new one", y=100)],
        [line("zara: where is resistance?", y=40), line("kev: new one", y=70)],
    ])
    total = []
    for i in range(3):
        total.extend(a.process_frame(img(i)))

    assert len(total) == 2
    assert {e.text_raw for e in total} == {"where is resistance?", "new one"}
    assert a.stats.duplicates == 3


def test_dedupe_window_is_bounded():
    a = adapter([[line(f"user{i}: message {i}")] for i in range(60)],
                dedupe_window=16)
    for i in range(60):
        a.process_frame(img(i))
    assert len(a._seen) <= 16


# -- health ----------------------------------------------------------------


async def test_health_reports_down_before_connect():
    a = adapter([[]])
    health = await a.health()
    assert health.state is HealthState.DOWN
    assert health.session_id == "SESSION_001"


async def test_health_flags_a_stalled_panel():
    """A LIVE Studio update that moves the panel yields zero comments and no
    error. Elapsed time is the only available signal."""
    a = adapter([[]], stall_alert_after_s=60.0)
    a._running = True
    a.started_at = datetime.now(timezone.utc) - timedelta(seconds=300)

    health = await a.health()
    assert health.state is HealthState.DEGRADED
    assert "recalibrate" in (health.degraded_reason or "")


async def test_health_ok_while_comments_flow():
    a = adapter([[line("zara: hi")]], stall_alert_after_s=60.0)
    a._running = True
    a.started_at = datetime.now(timezone.utc)
    a.process_frame(img(1))
    health = await a.health()
    assert health.state is HealthState.OK


async def test_connect_refuses_uncalibrated_device():
    a = ScreenCaptureAdapter(
        session_id="SESSION_001",
        calibration=CaptureCalibration(),  # zero crop
        ocr=FakeOCR(), author_salt="s", frame_source=object(),  # type: ignore[arg-type]
    )
    with pytest.raises(ValueError, match="calibrate"):
        await a.connect()
