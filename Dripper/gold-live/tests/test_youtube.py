"""YouTube Live adapter.

The hedge against an account-level problem on the primary platform. Tested
against a fake transport rather than the live API, because the behaviour that
matters is quota discipline and failure handling, not Google's JSON.

Quota is the real constraint and gets the most attention here: burning a day's
allowance by mid-morning leaves the host silently deaf while otherwise looking
healthy, which is a far worse failure than polling more slowly.
"""

from __future__ import annotations

import asyncio

import pytest

from platform_.adapters.youtube import (
    COST_LIST_MESSAGES,
    QuotaBudget,
    YouTubeLiveAdapter,
)
from shared.contracts import HealthState


def video_response(chat_id: str | None = "chat-123") -> dict:
    details = {"activeLiveChatId": chat_id} if chat_id else {}
    return {"items": [{"liveStreamingDetails": details}]}


def message(msg_id: str, text: str, author: str = "viewer") -> dict:
    return {
        "id": msg_id,
        "snippet": {
            "displayMessage": text,
            "publishedAt": "2026-09-01T08:00:00Z",
        },
        "authorDetails": {"displayName": author, "channelId": f"UC{author}"},
    }


def chat_response(items: list[dict], token: str = "next", interval_ms: int = 2000) -> dict:
    return {
        "items": items,
        "nextPageToken": token,
        "pollingIntervalMillis": interval_ms,
    }


class FakeTransport:
    """Scripts responses per endpoint and records what was asked for."""

    def __init__(self, video=None, pages=None, fail_with=None) -> None:
        self.video = video if video is not None else video_response()
        self.pages = list(pages or [])
        self.fail_with = fail_with
        self.calls: list[tuple[str, dict]] = []

    async def __call__(self, path: str, params: dict) -> dict:
        self.calls.append((path, params))
        if self.fail_with:
            raise self.fail_with
        if path == "/videos":
            return self.video
        if self.pages:
            return self.pages.pop(0)
        return chat_response([])


def adapter(transport, **kw) -> YouTubeLiveAdapter:
    return YouTubeLiveAdapter(
        session_id="SESSION_001", video_id="vid123", author_salt="salt",
        api_key="fake-key", transport=transport, **kw,
    )


# -- quota budgeting -------------------------------------------------------


def test_default_quota_allows_a_poll_every_43_seconds():
    """The number that decides the design. Polling every 5s would exhaust a
    day's allowance in under three hours."""
    budget = QuotaBudget()
    assert budget.calls_per_day == 2000
    assert 40 < budget.sustainable_interval_s < 46


def test_a_quota_increase_buys_faster_polling():
    fast = QuotaBudget(daily_units=1_000_000)
    assert fast.sustainable_interval_s < 1.0


def test_budget_reports_exhaustion():
    budget = QuotaBudget(daily_units=COST_LIST_MESSAGES * 2)
    assert not budget.exhausted
    budget.spend()
    budget.spend()
    assert budget.exhausted
    assert budget.remaining_units == 0


def test_budget_rolls_over_after_a_day():
    budget = QuotaBudget(daily_units=10)
    budget.spend(10)
    assert budget.exhausted
    budget.window_started -= 86_401
    assert not budget.exhausted


# -- connecting ------------------------------------------------------------


async def test_connect_resolves_the_live_chat_id():
    t = FakeTransport()
    a = adapter(t)
    await a.connect()
    assert a.live_chat_id == "chat-123"
    assert t.calls[0][0] == "/videos"


async def test_connect_refuses_without_an_api_key():
    a = YouTubeLiveAdapter(
        session_id="S1", video_id="v", author_salt="s", api_key="",
        transport=FakeTransport(),
    )
    with pytest.raises(ValueError, match="YOUTUBE_API_KEY"):
        await a.connect()


async def test_connect_explains_a_stream_with_no_chat():
    a = adapter(FakeTransport(video=video_response(chat_id=None)))
    with pytest.raises(ValueError, match="no active live chat"):
        await a.connect()


async def test_connect_explains_a_missing_video():
    a = adapter(FakeTransport(video={"items": []}))
    with pytest.raises(ValueError, match="not found"):
        await a.connect()


# -- polling ---------------------------------------------------------------


async def test_converts_messages_to_comment_events():
    t = FakeTransport(pages=[chat_response([
        message("m1", "where is resistance?", "zaraFX"),
        message("m2", "gm all", "mike"),
    ])])
    a = adapter(t)
    await a.connect()
    comments, _wait = await a.poll_once()

    assert len(comments) == 2
    assert comments[0].session_id == "SESSION_001"
    assert comments[0].platform == "youtube"
    assert comments[0].text_raw == "where is resistance?"
    assert comments[0].platform_msg_id == "m1", "YouTube ids are stable; use them"


async def test_author_names_are_hashed():
    t = FakeTransport(pages=[chat_response([message("m1", "hi", "realname123")])])
    a = adapter(t)
    await a.connect()
    comments, _ = await a.poll_once()
    assert "realname123" not in comments[0].author_hash


async def test_messages_without_text_are_skipped():
    """Superchats and membership events arrive with no displayMessage."""
    item = message("m1", "")
    t = FakeTransport(pages=[chat_response([item, message("m2", "real comment")])])
    a = adapter(t)
    await a.connect()
    comments, _ = await a.poll_once()
    assert [c.text_raw for c in comments] == ["real comment"]


async def test_duplicate_ids_are_not_re_emitted():
    """A page replayed after a retry must not replay the whole chat."""
    t = FakeTransport(pages=[
        chat_response([message("m1", "hello")]),
        chat_response([message("m1", "hello")]),
    ])
    a = adapter(t)
    await a.connect()

    first, _ = await a.poll_once()
    second, _ = await a.poll_once()
    assert len(first) == 1
    assert second == [], "the same message id must only be emitted once"


async def test_page_token_is_carried_forward():
    t = FakeTransport(pages=[
        chat_response([message("m1", "a")], token="tok2"),
        chat_response([message("m2", "b")], token="tok3"),
    ])
    a = adapter(t)
    await a.connect()
    await a.poll_once()
    await a.poll_once()

    second_poll = [c for c in t.calls if c[0] == "/liveChat/messages"][1]
    assert second_poll[1]["pageToken"] == "tok2"


# -- pacing ----------------------------------------------------------------


async def test_never_polls_faster_than_the_quota_allows():
    """YouTube may say 'poll in 2 seconds'. At default quota that would run
    the allowance out before lunch."""
    t = FakeTransport(pages=[chat_response([], interval_ms=2000)])
    a = adapter(t)
    await a.connect()
    _comments, wait_s = await a.poll_once()
    assert wait_s >= a.budget.sustainable_interval_s
    assert wait_s > 40


async def test_respects_a_slower_interval_from_youtube():
    """Ignoring their guidance is how an application gets throttled."""
    t = FakeTransport(pages=[chat_response([], interval_ms=120_000)])
    a = adapter(t, daily_quota=10_000_000)  # quota not the constraint here
    await a.connect()
    _comments, wait_s = await a.poll_once()
    assert wait_s == pytest.approx(120.0)


async def test_exhausted_quota_backs_off_instead_of_hammering():
    t = FakeTransport(pages=[chat_response([])])
    a = adapter(t, daily_quota=COST_LIST_MESSAGES)
    await a.connect()
    a.budget.spend(a.budget.daily_units)

    calls_before = len(t.calls)
    comments, wait_s = await a.poll_once()  # must not reach the transport
    assert comments == []
    assert wait_s >= 60
    assert len(t.calls) == calls_before, "must not spend quota it does not have"


# -- failure ---------------------------------------------------------------


async def test_poll_failures_do_not_end_the_stream():
    a = adapter(FakeTransport())
    await a.connect()
    a._transport = FakeTransport(fail_with=ConnectionError("network gone"))

    collected = []

    async def run():
        async for c in a.comments():
            collected.append(c)

    task = asyncio.create_task(run())
    await asyncio.sleep(0.05)
    a._running = False
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert a.errors >= 1, "errors are counted, not raised"


# -- health ----------------------------------------------------------------


async def test_health_is_ok_when_connected():
    a = adapter(FakeTransport())
    await a.connect()
    health = await a.health()
    assert health.state is HealthState.OK
    assert health.session_id == "SESSION_001"


async def test_health_degrades_as_quota_runs_low():
    a = adapter(FakeTransport())
    await a.connect()
    a.budget.spend(int(a.budget.daily_units * 0.95))
    health = await a.health()
    assert health.state is HealthState.DEGRADED
    assert "quota" in (health.degraded_reason or "").lower()


async def test_health_names_the_fix_when_quota_is_gone():
    a = adapter(FakeTransport())
    await a.connect()
    a.budget.spend(a.budget.daily_units)
    health = await a.health()
    assert "quota increase" in (health.degraded_reason or "")


async def test_health_is_down_before_connecting():
    a = adapter(FakeTransport())
    assert (await a.health()).state is HealthState.DOWN


# -- interface parity ------------------------------------------------------


def test_satisfies_the_platform_adapter_interface():
    """Nothing above the adapter should know which platform it is talking to."""
    from platform_.adapters.base import PlatformAdapter

    assert issubclass(YouTubeLiveAdapter, PlatformAdapter)
    for method in ("connect", "comments", "health", "disconnect"):
        assert callable(getattr(YouTubeLiveAdapter, method))


async def test_emits_the_same_shape_as_screen_capture():
    """A session must not care which adapter produced its comments."""
    t = FakeTransport(pages=[chat_response([message("m1", "where is resistance?")])])
    a = adapter(t)
    await a.connect()
    comments, _ = await a.poll_once()

    c = comments[0]
    assert c.session_id and c.platform_msg_id and c.author_hash
    assert c.text_norm == c.text_raw.lower()
    assert c.classification is None, "classification happens in the pipeline, not the adapter"
