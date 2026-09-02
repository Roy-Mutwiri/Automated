"""Find whatever model server is already running on this machine.

Someone you share the build with should not have to know that vLLM defaults to
port 8000 and Ollama to 11434, nor type the exact model identifier. If a server
is running, find it and use what it is serving.

Order is deliberate: an explicit LLM_BASE_URL always wins, then the ports in
descending order of how likely they are to be a deliberate choice.
"""

from __future__ import annotations

import logging
import os

from platform_.llm.local import LocalLLM

log = logging.getLogger(__name__)

# (label, base url). vLLM first: someone running it chose it deliberately.
KNOWN_ENDPOINTS: list[tuple[str, str]] = [
    ("vLLM / OpenAI-compatible", "http://127.0.0.1:8000/v1"),
    ("Ollama", "http://127.0.0.1:11434/v1"),
    ("LM Studio", "http://127.0.0.1:1234/v1"),
    ("llama.cpp server", "http://127.0.0.1:8080/v1"),
    ("Text generation WebUI", "http://127.0.0.1:5000/v1"),
]


async def discover(explicit_url: str | None = None) -> LocalLLM | None:
    """Return a ready LocalLLM, or None if nothing is serving.

    Resolves the model name from the server when the caller has not pinned one,
    so a shared build works without the recipient editing any configuration.
    """
    configured = explicit_url or os.environ.get("LLM_BASE_URL")
    candidates = (
        [("configured", configured)] if configured
        else list(KNOWN_ENDPOINTS)
    )

    for label, url in candidates:
        llm = LocalLLM(base_url=url)
        if not await llm.health():
            await llm.close()
            continue

        if not os.environ.get("LLM_MODEL"):
            served = await llm.first_model()
            if served:
                llm.model = served
                llm.name = f"local:{served}"
        log.info("found %s at %s serving %s", label, url, llm.model)
        return llm

    log.warning(
        "no model server found. Tried: %s",
        ", ".join(url for _label, url in candidates),
    )
    return None


def endpoint_hints() -> str:
    lines = ["No model server is running. Start one of these:", ""]
    lines += [
        "  Ollama (easiest)",
        "    ollama serve",
        "    ollama pull <model>",
        "",
        "  vLLM (best for several sessions at once)",
        "    vllm serve <model> --port 8000 --enable-prefix-caching",
        "",
        "Then re-run. Ports checked automatically:",
    ]
    lines += [f"  {label:<28} {url}" for label, url in KNOWN_ENDPOINTS]
    return "\n".join(lines)
