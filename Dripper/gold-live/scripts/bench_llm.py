"""Benchmark the local model against this system's actual workload.

Do not size hardware from spec sheets. The numbers that matter here are not
peak tokens/sec on a single long generation -- they are:

  time to first token       decides when audio starts, so it decides whether
                            the host feels responsive or laggy
  tokens/sec under load     with N sessions generating concurrently
  prefix cache hit rate     the persona prompt is identical every request; if
                            it is being reprocessed each time you are paying
                            hundreds of milliseconds for nothing

Usage:
    python -m scripts.bench_llm                  # 1, 3, 7 concurrent
    python -m scripts.bench_llm --concurrency 7 --runs 20

Target for the latency budget in the architecture doc: p95 time-to-first-token
under 900ms at your intended session count. If it is worse, the options are a
smaller model, heavier quantisation, or a second GPU -- in that order.
"""

from __future__ import annotations

import argparse
import asyncio
import statistics
import time

from platform_.llm.base import ChatMessage
from platform_.llm.local import LocalLLM

SYSTEM = """You are the live host of a Gold (XAUUSD) trading stream.

Your audience: active intraday traders watching the chart right now.
Style: quick and observational, short sentences, no hype.

You speak out loud on a live stream. At most 3 sentences. No markdown, no
headings, no emoji. You describe scenarios, never predictions. You never give
personalised trading advice and never state a guaranteed outcome."""

TURN = """CURRENT MARKET STATE
  data confidence : live (400ms old)
  5m trend        : bullish
  5m structure    : higher_high
  session range   : 3646.20 - 3655.90
  detected        : Break of structure above prior swing high

WHY YOU ARE SPEAKING NOW
  Market event: price broke structure to the upside after sweeping the session low.
  React to it the way a host watching the chart would."""


async def one(llm: LocalLLM) -> tuple[float, float, int]:
    """Returns (time_to_first_token_ms, total_ms, tokens_approx)."""
    messages = [
        ChatMessage(role="system", content=SYSTEM),
        ChatMessage(role="user", content=TURN),
    ]
    t0 = time.perf_counter()
    first: float | None = None
    chars = 0
    async for delta in llm.stream(messages, max_tokens=120, temperature=0.9):
        if first is None:
            first = (time.perf_counter() - t0) * 1000
        chars += len(delta)
    total = (time.perf_counter() - t0) * 1000
    return (first or total), total, max(1, chars // 4)


async def bench(llm: LocalLLM, concurrency: int, runs: int) -> dict:
    ttft: list[float] = []
    totals: list[float] = []
    tokens = 0

    # Warm the prefix cache first; the first request pays for prompt processing
    # and would otherwise skew every number here.
    await one(llm)

    t0 = time.perf_counter()
    for _ in range(runs):
        results = await asyncio.gather(*(one(llm) for _ in range(concurrency)))
        for f, t, n in results:
            ttft.append(f)
            totals.append(t)
            tokens += n
    wall = time.perf_counter() - t0

    ttft.sort()
    return {
        "concurrency": concurrency,
        "requests": runs * concurrency,
        "ttft_p50": round(statistics.median(ttft)),
        "ttft_p95": round(ttft[int(len(ttft) * 0.95) - 1]),
        "total_p50": round(statistics.median(totals)),
        "tok_per_s": round(tokens / wall, 1),
        "wall_s": round(wall, 1),
    }


async def main_async(levels: list[int], runs: int) -> None:
    llm = LocalLLM()
    if not await llm.health():
        raise SystemExit(
            f"No model server at {llm.base_url}.\n"
            "Start one, e.g.:\n"
            "  vllm serve <model> --port 8000 --enable-prefix-caching\n"
            "  ollama serve   (then LLM_BASE_URL=http://127.0.0.1:11434/v1)"
        )

    print(f"\n  model: {llm.model}   endpoint: {llm.base_url}\n")
    print(f"  {'conc':>5} {'reqs':>5} {'ttft p50':>9} {'ttft p95':>9} "
          f"{'total p50':>10} {'tok/s':>8}")
    print("  " + "-" * 54)

    rows = []
    for c in levels:
        r = await bench(llm, c, runs)
        rows.append(r)
        print(f"  {r['concurrency']:>5} {r['requests']:>5} {r['ttft_p50']:>8}ms "
              f"{r['ttft_p95']:>8}ms {r['total_p50']:>9}ms {r['tok_per_s']:>8}")

    print("\n  VERDICT")
    for r in rows:
        ok = r["ttft_p95"] < 900
        note = "within budget" if ok else "TOO SLOW for the 1-3s target"
        print(f"    {r['concurrency']} sessions: p95 first token {r['ttft_p95']}ms - {note}")

    worst = rows[-1]
    if worst["ttft_p95"] >= 900:
        print(
            "\n  Options, cheapest first: smaller model, heavier quantisation\n"
            "  (AWQ/GPTQ 4-bit), shorter prompts, then a second GPU.\n"
            "  Confirm --enable-prefix-caching is on: the persona prompt is\n"
            "  identical on every request and should never be reprocessed."
        )
    await llm.close()


def main() -> None:
    ap = argparse.ArgumentParser(description="Benchmark the local LLM")
    ap.add_argument("--concurrency", type=int, nargs="*", default=[1, 3, 7])
    ap.add_argument("--runs", type=int, default=8)
    args = ap.parse_args()
    asyncio.run(main_async(args.concurrency, args.runs))


if __name__ == "__main__":
    main()
