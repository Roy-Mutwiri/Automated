"""LocalLLM against a real HTTP server.

This path had never executed against a socket -- only an in-process fake. These
tests exercise the wire protocol: SSE framing, chunked transfer, keep-alive
reuse, junk lines mid-stream, dropped connections, error statuses.
"""

from __future__ import annotations

import pytest

from intelligence.generation import LocalGenerator
from intelligence.personas import Persona
from platform_.llm.base import ChatMessage
from platform_.llm.local import LocalLLM
from tests.stub_llm_server import StubLLMServer

pytest.importorskip("httpx")


def messages() -> list[ChatMessage]:
    return [
        ChatMessage(role="system", content="You are a live trading host."),
        ChatMessage(role="user", content="React to the break of structure."),
    ]


# -- health ---------------------------------------------------------------


async def test_health_true_when_serving():
    with StubLLMServer() as (url, _cfg):
        llm = LocalLLM(base_url=url, model="stub-model")
        assert await llm.health()
        await llm.close()


async def test_health_false_when_nothing_listening():
    # Port 1 is reserved and never listening.
    llm = LocalLLM(base_url="http://127.0.0.1:1/v1", model="x")
    assert not await llm.health()
    await llm.close()


# -- streaming ------------------------------------------------------------


async def test_streams_deltas_over_real_http():
    with StubLLMServer() as (url, cfg):
        cfg.reply = "Price closed above the prior high. That is the level to watch."
        llm = LocalLLM(base_url=url, model="stub-model")
        chunks = [c async for c in llm.stream(messages())]
        await llm.close()

    assert len(chunks) > 1, "should arrive incrementally, not in one lump"
    assert "".join(chunks) == cfg.reply


async def test_survives_keepalive_and_malformed_lines():
    """Real servers emit keep-alive comments and occasionally a bad line. A
    client that dies on those is broken."""
    with StubLLMServer() as (url, cfg):
        cfg.inject_garbage = True
        llm = LocalLLM(base_url=url, model="stub-model")
        chunks = [c async for c in llm.stream(messages())]
        await llm.close()
    assert "".join(chunks) == cfg.reply


async def test_dropped_stream_fails_fast_not_after_a_minute():
    """A server that drops mid-generation must be detected in seconds.

    With a single flat timeout this blocked for the full 60s, which on a live
    broadcast is a minute of dead air before anything notices.
    """
    import time

    with StubLLMServer() as (url, cfg):
        cfg.truncate_after = 3
        llm = LocalLLM(base_url=url, model="stub-model", read_timeout_s=3.0)
        started = time.perf_counter()
        chunks = []
        try:
            async for chunk in llm.stream(messages()):
                chunks.append(chunk)
        except Exception:
            pass
        elapsed = time.perf_counter() - started
        await llm.close()

    assert len(chunks) <= 3, "partial output should still have arrived"
    assert elapsed < 10, f"took {elapsed:.1f}s; a dropped stream must fail fast"


def test_timeouts_are_separated():
    """One flat timeout is the bug. Connect, read and total are distinct."""
    llm = LocalLLM(base_url="http://x/v1", model="m")
    assert llm.connect_timeout_s < llm.read_timeout_s < llm.total_timeout_s


# -- completion -----------------------------------------------------------


async def test_complete_returns_text_and_usage():
    with StubLLMServer() as (url, cfg):
        cfg.reply = "One scenario worth watching."
        llm = LocalLLM(base_url=url, model="stub-model")
        result = await llm.complete(messages())
        await llm.close()

    assert result.text == cfg.reply
    assert result.prompt_tokens == 120
    assert result.completion_tokens == 24
    assert result.total_ms is not None and result.total_ms >= 0


async def test_reports_prefix_cache_hits():
    """Prefix caching is most of the latency win, so a zero hit rate must be
    visible rather than silent."""
    with StubLLMServer() as (url, cfg):
        llm = LocalLLM(base_url=url, model="stub-model")
        cold = await llm.complete(messages())
        cfg.cached_tokens = 110
        warm = await llm.complete(messages())
        await llm.close()

    assert not cold.prefix_cached
    assert warm.prefix_cached


async def test_server_error_raises():
    with StubLLMServer() as (url, cfg):
        cfg.status = 500
        llm = LocalLLM(base_url=url, model="stub-model")
        with pytest.raises(Exception):
            await llm.complete(messages())
        await llm.close()


# -- request shape --------------------------------------------------------


async def test_sends_penalties_that_reduce_verbal_tics():
    with StubLLMServer() as (url, cfg):
        llm = LocalLLM(base_url=url, model="stub-model")
        await llm.complete(messages())
        await llm.close()

    sent = cfg.requests[-1]
    assert sent["model"] == "stub-model"
    assert sent["presence_penalty"] > 0
    assert sent["frequency_penalty"] > 0
    assert [m["role"] for m in sent["messages"]] == ["system", "user"]


async def test_connection_is_reused_across_calls():
    """A fresh connection per utterance costs a handshake on the critical path."""
    with StubLLMServer() as (url, _cfg):
        llm = LocalLLM(base_url=url, model="stub-model")
        first = await llm._http()
        await llm.complete(messages())
        second = await llm._http()
        await llm.complete(messages())
        assert first is second
        await llm.close()


# -- generator integration -------------------------------------------------


async def test_local_generator_produces_segments():
    from intelligence.director import SpeechIntent
    from shared.contracts import (
        MarketConfidence,
        MarketState,
        Price,
        Priority,
        TradingSession,
        TriggerType,
        utcnow,
    )

    with StubLLMServer() as (url, cfg):
        cfg.reply = (
            "Host: *clears throat* Price closed above the prior high. "
            "The level to watch now is the session high."
        )
        llm = LocalLLM(base_url=url, model="stub-model")
        generator = LocalGenerator(llm)

        now = utcnow()
        result = await generator.generate(
            persona=Persona(
                persona_id="p", display_name="P", audience="traders",
                primary_timeframe="5m", focus=["structure"], avoid=[],
                voice_id="v", style="terse",
            ),
            intent=SpeechIntent(
                trigger=TriggerType.MARKET_EVENT, priority=Priority.HIGH,
                topic="mkt:bos", created_at=now,
            ),
            market=MarketState(
                as_of=now, computed_at=now, confidence=MarketConfidence.LIVE,
                price=Price(bid=3652.2, ask=3652.6), session=TradingSession.LONDON,
            ),
            transcript=[],
        )
        await llm.close()

    # Stage directions and speaker labels must never reach a spoken stream.
    assert "Host:" not in result.text
    assert "*clears throat*" not in result.text
    assert len(result.segments) == 2, "split into sentences for streaming TTS"
    assert result.provenance.first_token_ms is not None
    assert result.provenance.model == "local:stub-model"
