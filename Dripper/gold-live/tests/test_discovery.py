"""Model server discovery.

Whoever you share the build with should not need to know port numbers or model
identifiers. These check that discovery actually removes that step.
"""

from __future__ import annotations

import pytest

from platform_.llm import discovery
from tests.stub_llm_server import StubLLMServer

pytest.importorskip("httpx")


async def test_finds_a_running_server_and_adopts_its_model(monkeypatch):
    monkeypatch.delenv("LLM_BASE_URL", raising=False)
    monkeypatch.delenv("LLM_MODEL", raising=False)

    with StubLLMServer() as (url, _cfg):
        monkeypatch.setattr(discovery, "KNOWN_ENDPOINTS", [("stub", url)])
        llm = await discovery.discover()
        assert llm is not None
        # Taken from the server, not from config.
        assert llm.model == "stub-model"
        assert llm.name == "local:stub-model"
        await llm.close()


async def test_skips_dead_ports_and_keeps_looking(monkeypatch):
    monkeypatch.delenv("LLM_BASE_URL", raising=False)
    monkeypatch.delenv("LLM_MODEL", raising=False)

    with StubLLMServer() as (url, _cfg):
        monkeypatch.setattr(
            discovery, "KNOWN_ENDPOINTS",
            [("dead", "http://127.0.0.1:1/v1"), ("stub", url)],
        )
        llm = await discovery.discover()
        assert llm is not None and llm.base_url == url.rstrip("/")
        await llm.close()


async def test_returns_none_when_nothing_is_serving(monkeypatch):
    monkeypatch.delenv("LLM_BASE_URL", raising=False)
    monkeypatch.setattr(
        discovery, "KNOWN_ENDPOINTS", [("dead", "http://127.0.0.1:1/v1")]
    )
    assert await discovery.discover() is None


async def test_explicit_url_wins_over_probing(monkeypatch):
    monkeypatch.delenv("LLM_MODEL", raising=False)
    with StubLLMServer() as (url, _cfg):
        # Probing would find nothing; the explicit URL must still be used.
        monkeypatch.setattr(
            discovery, "KNOWN_ENDPOINTS", [("dead", "http://127.0.0.1:1/v1")]
        )
        llm = await discovery.discover(explicit_url=url)
        assert llm is not None
        await llm.close()


async def test_pinned_model_is_not_overridden(monkeypatch):
    monkeypatch.setenv("LLM_MODEL", "my-specific-model")
    with StubLLMServer() as (url, _cfg):
        monkeypatch.setattr(discovery, "KNOWN_ENDPOINTS", [("stub", url)])
        llm = await discovery.discover()
        assert llm is not None
        assert llm.model == "my-specific-model"
        await llm.close()


def test_hints_name_the_easy_option_first():
    hints = discovery.endpoint_hints()
    assert "Ollama" in hints
    assert "vllm serve" in hints
    assert hints.index("Ollama") < hints.index("vLLM")


async def test_a_server_with_no_models_is_not_usable(monkeypatch):
    """A running server with nothing loaded passes a health check but cannot
    generate. Accepting it means the first utterance fails with model-not-found
    and the offline fallback never engages, because health already said yes."""
    monkeypatch.delenv("LLM_BASE_URL", raising=False)
    monkeypatch.delenv("LLM_MODEL", raising=False)

    with StubLLMServer() as (url, cfg):
        cfg.no_models = True
        monkeypatch.setattr(discovery, "KNOWN_ENDPOINTS", [("stub", url)])
        assert await discovery.discover() is None


async def test_a_pinned_model_is_used_even_if_none_are_listed(monkeypatch):
    """An operator who names a model knows better than the listing -- some
    servers load on first use."""
    monkeypatch.setenv("LLM_MODEL", "my-model")
    with StubLLMServer() as (url, cfg):
        cfg.no_models = True
        monkeypatch.setattr(discovery, "KNOWN_ENDPOINTS", [("stub", url)])
        llm = await discovery.discover()
        assert llm is not None and llm.model == "my-model"
        await llm.close()
