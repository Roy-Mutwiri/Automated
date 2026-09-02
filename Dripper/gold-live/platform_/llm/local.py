"""Locally hosted LLM on the central machine.

Speaks the OpenAI-compatible chat-completions protocol, which vLLM, llama.cpp's
server, Ollama, TGI, LM Studio and SGLang all expose. That means the server can
be swapped without touching this file -- only the base URL and model name change.

Recommended server: vLLM. It has continuous batching and automatic prefix
caching, which are the two things that matter here:

  Continuous batching -- seven sessions share one GPU. They do not speak at the
  same instant (the Director rate-limits each to roughly one utterance every
  20-40s), so the real load is ~10 generations per minute, one every six
  seconds. A batching server absorbs that on a single card; a naive one-request-
  at-a-time server will not.

  Prefix caching -- each session's persona prompt is identical on every request
  for the life of the session. With prefix caching that prompt is processed once
  rather than on every utterance. Keep the persona prompt BYTE-STABLE: no
  timestamps, no prices, no counters. Volatile market state goes in the trailing
  user message, never in the system prompt.

Sizing, roughly, for a ~70B model at 4-bit on one 48GB card:
    weights ~38GB, KV cache for 7 concurrent sequences at 8k ctx ~6GB,
    leaving headroom. ~30-50 tok/s single-stream, ~200ms to first token warm.
A 3-4 sentence utterance is ~80 tokens, so ~2s to complete -- but with
sentence-chunked streaming, audio starts after the first ~20 tokens.

Benchmark before committing to hardware. `scripts/bench_llm.py` measures the
numbers that matter: time to first token, tokens/sec under concurrency, and
prefix-cache hit rate.
"""

from __future__ import annotations

import json
import logging
import os
import time
from collections.abc import AsyncIterator
from typing import Any

from platform_.llm.base import ChatMessage, LLMBackend, LLMResult

log = logging.getLogger(__name__)


class LocalLLM(LLMBackend):
    def __init__(
        self,
        base_url: str | None = None,
        model: str | None = None,
        api_key: str = "not-needed",
        connect_timeout_s: float = 5.0,
        read_timeout_s: float = 15.0,
        total_timeout_s: float = 60.0,
    ) -> None:
        self.base_url = (
            base_url or os.environ.get("LLM_BASE_URL", "http://127.0.0.1:8000/v1")
        ).rstrip("/")
        self.model = model or os.environ.get("LLM_MODEL", "local-model")
        self.api_key = os.environ.get("LLM_API_KEY", api_key)
        # Separate read timeout, and a short one. It applies between reads, so
        # once tokens are flowing the gaps are milliseconds -- the only window
        # it really bounds is time-to-first-token. A single flat 60s timeout
        # means a server that drops a stream mid-generation (OOM, restart, load
        # shedding) freezes the session for a full minute of dead air before
        # anyone notices. Detect it in seconds instead.
        self.connect_timeout_s = connect_timeout_s
        self.read_timeout_s = read_timeout_s
        self.total_timeout_s = total_timeout_s
        self.name = f"local:{self.model}"
        self._client: Any = None

    async def _http(self) -> Any:
        """One pooled client for the process.

        A fresh connection per utterance costs a TCP and TLS handshake on every
        generation -- around 100ms of pure waste on the critical path.
        """
        if self._client is None:
            import httpx

            self._client = httpx.AsyncClient(
                base_url=self.base_url,
                timeout=httpx.Timeout(
                    self.total_timeout_s,
                    connect=self.connect_timeout_s,
                    read=self.read_timeout_s,
                ),
                headers={"Authorization": f"Bearer {self.api_key}"},
                limits=httpx.Limits(max_keepalive_connections=16, max_connections=32),
            )
        return self._client

    def _payload(
        self,
        messages: list[ChatMessage],
        max_tokens: int,
        temperature: float,
        stop: list[str] | None,
        stream: bool,
    ) -> dict[str, Any]:
        return {
            "model": self.model,
            "messages": [{"role": m.role, "content": m.content} for m in messages],
            "max_tokens": max_tokens,
            "temperature": temperature,
            # Discourage the tics that make synthetic speech obvious: repeated
            # openers, the same connective every sentence, stock phrases.
            "presence_penalty": 0.6,
            "frequency_penalty": 0.35,
            "top_p": 0.92,
            "stop": stop or [],
            "stream": stream,
        }

    async def stream(  # type: ignore[override]
        self,
        messages: list[ChatMessage],
        *,
        max_tokens: int = 300,
        temperature: float = 0.85,
        stop: list[str] | None = None,
    ) -> AsyncIterator[str]:
        client = await self._http()
        payload = self._payload(messages, max_tokens, temperature, stop, stream=True)

        async with client.stream("POST", "/chat/completions", json=payload) as resp:
            resp.raise_for_status()
            async for line in resp.aiter_lines():
                if not line or not line.startswith("data:"):
                    continue
                data = line[5:].strip()
                if data == "[DONE]":
                    break
                try:
                    chunk = json.loads(data)
                except json.JSONDecodeError:
                    continue
                choices = chunk.get("choices") or []
                if not choices:
                    continue
                delta = choices[0].get("delta", {}).get("content")
                if delta:
                    yield delta

    async def complete(
        self,
        messages: list[ChatMessage],
        *,
        max_tokens: int = 300,
        temperature: float = 0.85,
        stop: list[str] | None = None,
    ) -> LLMResult:
        client = await self._http()
        t0 = time.perf_counter()
        resp = await client.post(
            "/chat/completions",
            json=self._payload(messages, max_tokens, temperature, stop, stream=False),
        )
        resp.raise_for_status()
        body = resp.json()
        usage = body.get("usage", {}) or {}
        return LLMResult(
            text=body["choices"][0]["message"]["content"].strip(),
            model=self.model,
            total_ms=int((time.perf_counter() - t0) * 1000),
            prompt_tokens=usage.get("prompt_tokens"),
            completion_tokens=usage.get("completion_tokens"),
            prefix_cached=bool(
                (usage.get("prompt_tokens_details") or {}).get("cached_tokens")
            ),
        )

    async def health(self) -> bool:
        try:
            client = await self._http()
            resp = await client.get("/models", timeout=3.0)
            return resp.status_code == 200
        except Exception as exc:  # noqa: BLE001 - health must never raise
            log.warning("local LLM health check failed: %s", exc)
            return False

    async def first_model(self) -> str | None:
        """Whatever the server is actually serving.

        Saves the recipient of a shared build from having to know the exact
        model string -- if the server has one model loaded, use it.
        """
        try:
            client = await self._http()
            resp = await client.get("/models", timeout=3.0)
            data = resp.json().get("data") or []
            return data[0].get("id") if data else None
        except Exception:  # noqa: BLE001
            return None

    async def close(self) -> None:
        if self._client is not None:
            await self._client.aclose()
            self._client = None
