"""YouTube Live comment adapter.

The hedge. Everything else in this system is platform-neutral, and this is what
makes that worth something: if an account-level problem takes a TikTok stream
off air, the same session config points here instead and keeps broadcasting.
Roughly a day of work for the difference between an enforcement action being
survivable and being terminal.

Unlike screen capture this is an officially supported API, which removes the
terms-of-service question entirely for this platform.

QUOTA IS THE DESIGN CONSTRAINT, and it is tighter than it looks:

    liveChatMessages.list costs 5 units per call
    default daily quota is 10,000 units
    -> 2,000 calls per day -> one call roughly every 43 seconds

Polling every five seconds would exhaust a day's quota in under three hours and
then go silent, which is a far worse failure than being slightly slower: the
host would appear to work all morning and be deaf by lunchtime. So this budgets
quota explicitly, adapts its interval to make the allowance last a full day, and
reports how much is left. Ask Google for a quota increase if you need faster
than that -- it is a form, not a payment.

The API also returns `pollingIntervalMillis`, which is YouTube telling you how
often it expects to be polled. Ignoring it is how an application gets throttled.
We honour whichever is slower: their number or our budget.
"""

from __future__ import annotations

import asyncio
import logging
import os
import time
from collections.abc import AsyncIterator
from datetime import datetime, timezone

from platform_.adapters.base import PlatformAdapter
from shared.contracts import (
    CommentEvent,
    HealthCheck,
    HealthState,
    ServiceHealth,
)

log = logging.getLogger(__name__)

API_ROOT = "https://www.googleapis.com/youtube/v3"

# Documented quota costs, in units.
COST_LIST_MESSAGES = 5
COST_VIDEO_LOOKUP = 1
DEFAULT_DAILY_QUOTA = 10_000

SECONDS_PER_DAY = 86_400


class QuotaBudget:
    """Spends a daily allowance evenly rather than all at once.

    The failure this prevents is specific and nasty: burning the day's quota by
    late morning, after which the host silently stops seeing comments while
    otherwise appearing healthy. Running slower all day beats running fast and
    going deaf.
    """

    def __init__(self, daily_units: int = DEFAULT_DAILY_QUOTA, cost_per_call: int = COST_LIST_MESSAGES) -> None:
        self.daily_units = daily_units
        self.cost_per_call = cost_per_call
        self.used = 0
        self.window_started = time.time()

    def _roll(self) -> None:
        if time.time() - self.window_started >= SECONDS_PER_DAY:
            self.used = 0
            self.window_started = time.time()

    @property
    def calls_per_day(self) -> int:
        return max(1, self.daily_units // self.cost_per_call)

    @property
    def sustainable_interval_s(self) -> float:
        return SECONDS_PER_DAY / self.calls_per_day

    @property
    def remaining_units(self) -> int:
        self._roll()
        return max(0, self.daily_units - self.used)

    @property
    def exhausted(self) -> bool:
        return self.remaining_units < self.cost_per_call

    def spend(self, units: int | None = None) -> None:
        self._roll()
        self.used += self.cost_per_call if units is None else units


class YouTubeLiveAdapter(PlatformAdapter):
    def __init__(
        self,
        session_id: str,
        video_id: str,
        author_salt: str,
        api_key: str | None = None,
        daily_quota: int = DEFAULT_DAILY_QUOTA,
        max_backoff_s: float = 300.0,
        transport=None,
    ) -> None:
        self.session_id = session_id
        self.video_id = video_id
        self.author_salt = author_salt
        self.api_key = api_key or os.environ.get("YOUTUBE_API_KEY", "")
        self.budget = QuotaBudget(daily_quota)
        self.max_backoff_s = max_backoff_s
        self.platform = "youtube"

        self._transport = transport  # injected for tests
        self.live_chat_id: str | None = None
        self.page_token: str | None = None
        self._running = False
        self._seen: set[str] = set()
        self.started_at: datetime | None = None
        self.last_message_at: datetime | None = None
        self.messages_seen = 0
        self.errors = 0
        self.quota_exhausted_at: datetime | None = None

    # -- transport --------------------------------------------------------

    async def _get(self, path: str, params: dict) -> dict:
        if self._transport is not None:
            return await self._transport(path, params)

        import httpx

        if not hasattr(self, "_client"):
            self._client = httpx.AsyncClient(
                base_url=API_ROOT, timeout=httpx.Timeout(20.0, connect=5.0)
            )
        resp = await self._client.get(
            path, params={**params, "key": self.api_key}
        )
        resp.raise_for_status()
        return resp.json()

    # -- lifecycle --------------------------------------------------------

    async def connect(self) -> None:
        if not self.api_key:
            raise ValueError(
                "No YouTube API key. Set YOUTUBE_API_KEY, or create one at "
                "console.cloud.google.com with the YouTube Data API v3 enabled."
            )

        body = await self._get(
            "/videos", {"part": "liveStreamingDetails", "id": self.video_id}
        )
        self.budget.spend(COST_VIDEO_LOOKUP)

        items = body.get("items") or []
        if not items:
            raise ValueError(f"video {self.video_id!r} not found, or not public")

        details = items[0].get("liveStreamingDetails") or {}
        chat_id = details.get("activeLiveChatId")
        if not chat_id:
            raise ValueError(
                f"video {self.video_id!r} has no active live chat. It is either "
                "not live, has chat disabled, or the broadcast has ended."
            )

        self.live_chat_id = chat_id
        self._running = True
        self.started_at = datetime.now(timezone.utc)
        log.info(
            "youtube chat connected for %s (quota allows a poll every %.0fs)",
            self.session_id, self.budget.sustainable_interval_s,
        )

    async def disconnect(self) -> None:
        self._running = False
        client = getattr(self, "_client", None)
        if client is not None:
            await client.aclose()
            del self._client

    # -- ingest -----------------------------------------------------------

    def _to_comment(self, item: dict) -> CommentEvent | None:
        snippet = item.get("snippet") or {}
        author = item.get("authorDetails") or {}
        text = (snippet.get("displayMessage") or "").strip()
        if not text:
            return None  # superchats and membership events carry no message

        # YouTube gives a stable message id, so unlike screen capture there is
        # no need to synthesise one from the content.
        msg_id = str(item.get("id") or CommentEvent.synth_msg_id(
            author.get("displayName", ""), text
        ))
        if msg_id in self._seen:
            return None
        self._seen.add(msg_id)
        if len(self._seen) > 5000:
            self._seen = set(list(self._seen)[-2500:])

        published = snippet.get("publishedAt")
        try:
            at = (
                datetime.fromisoformat(str(published).replace("Z", "+00:00"))
                if published
                else datetime.now(timezone.utc)
            )
        except ValueError:
            at = datetime.now(timezone.utc)

        return CommentEvent(
            session_id=self.session_id,
            platform="youtube",
            platform_msg_id=msg_id,
            author_hash=CommentEvent.hash_author(
                author.get("displayName", "") or author.get("channelId", ""),
                self.author_salt,
            ),
            text_raw=text,
            text_norm=" ".join(text.lower().split()),
            received_at=at,
        )

    async def poll_once(self) -> tuple[list[CommentEvent], float]:
        """One page of chat. Returns (comments, seconds to wait before the next)."""
        if self.budget.exhausted:
            self.quota_exhausted_at = self.quota_exhausted_at or datetime.now(timezone.utc)
            log.warning(
                "[%s] YouTube quota exhausted; comments will resume when it resets",
                self.session_id,
            )
            return [], 300.0

        params: dict = {
            "liveChatId": self.live_chat_id,
            "part": "snippet,authorDetails",
            "maxResults": 200,
        }
        if self.page_token:
            params["pageToken"] = self.page_token

        body = await self._get("/liveChat/messages", params)
        self.budget.spend()

        self.page_token = body.get("nextPageToken") or self.page_token
        comments = [c for c in (self._to_comment(i) for i in body.get("items", [])) if c]
        if comments:
            self.messages_seen += len(comments)
            self.last_message_at = datetime.now(timezone.utc)

        # Honour whichever is slower: YouTube's own guidance, or the interval
        # that makes the daily quota last a full day.
        suggested = float(body.get("pollingIntervalMillis") or 0) / 1000.0
        return comments, max(suggested, self.budget.sustainable_interval_s)

    async def comments(self) -> AsyncIterator[CommentEvent]:  # type: ignore[override]
        backoff = 1.0
        while self._running:
            try:
                batch, wait_s = await self.poll_once()
                backoff = 1.0
                for comment in batch:
                    yield comment
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                self.errors += 1
                log.warning(
                    "[%s] youtube poll failed (%s); retrying in %.0fs",
                    self.session_id, exc, backoff,
                )
                wait_s = backoff
                backoff = min(backoff * 2, self.max_backoff_s)
            await asyncio.sleep(wait_s)

    # -- health -----------------------------------------------------------

    async def health(self) -> ServiceHealth:
        remaining = self.budget.remaining_units
        pct = remaining / self.budget.daily_units if self.budget.daily_units else 0.0

        checks = [
            HealthCheck(name="connected", ok=self._running and bool(self.live_chat_id),
                        detail=f"chat_id={'set' if self.live_chat_id else 'missing'}"),
            HealthCheck(name="quota", ok=pct > 0.1,
                        detail=f"{remaining}/{self.budget.daily_units} units left"),
            HealthCheck(name="poll_interval", ok=True,
                        detail=f"{self.budget.sustainable_interval_s:.0f}s sustainable"),
        ]

        state = HealthState.OK
        reason = None
        if not self._running:
            state, reason = HealthState.DOWN, "adapter stopped"
        elif self.budget.exhausted:
            state = HealthState.DEGRADED
            reason = (
                "YouTube quota exhausted; comments pause until it resets. "
                "Request a quota increase in the Google Cloud console."
            )
        elif pct < 0.1:
            state = HealthState.DEGRADED
            reason = f"only {remaining} quota units left today"

        return ServiceHealth(
            component="youtube_adapter", session_id=self.session_id,
            state=state, checks=checks, degraded_reason=reason,
        )
