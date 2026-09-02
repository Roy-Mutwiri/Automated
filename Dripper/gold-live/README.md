# Gold Live

A 24/7 multi-session AI broadcasting system for XAUUSD commentary.

Architecture and rationale: see the [architecture review](https://claude.ai/code/artifact/d7887018-f727-4728-b354-d204ed440fe3).
Read §00 before building anything on top of this.

**Status: M0 + M1 complete.** Contracts, mocks, Director, memory, safety gate
and a runnable dry run. No live market feed, no platform adapter, no device
agent yet.

## Run it

```bash
python -m venv .venv
.venv/Scripts/python -m pip install -e ".[dev]"

# Offline generator, no API key needed
PYTHONPATH=. .venv/Scripts/python -m runtime.dryrun

# Real Claude generation
export ANTHROPIC_API_KEY=sk-ant-...
PYTHONPATH=. .venv/Scripts/python -m runtime.dryrun --live

# Three sessions, with the cross-session isolation check
PYTHONPATH=. .venv/Scripts/python -m runtime.dryrun --sessions 3

PYTHONPATH=. .venv/Scripts/python -m pytest
```

Output (audio + text transcripts) lands in `out/<SESSION_ID>/`.

## What M1 is for

It answers the one question no architecture can: **does the output sound like
something a person would listen to for an hour?** Run it with `--live`, read
the transcript, listen to the audio. If it is boring, everything downstream is
wasted effort — and you have found that out in week one rather than month four.

The offline generator uses fixed templates and is deliberately repetitive. That
is not the product; it exists so the whole pipeline runs with no API key and so
the repetition detector has something real to catch.

## Layout

```
shared/         contracts, event bus, mocks          BOTH OWN
intelligence/   director, memory, safety, generation  ROY
platform_/      market, adapters, tts, audio          BROTHER
runtime/        session process, dry run              BOTH
configs/        personas and session config           BOTH
tests/
```

`shared/contracts.py` is the interface between the two halves. Changing it gets
its own PR, both approvals, and a version bump. Everything else is free to move.

Pydantic models are the source of truth; JSON Schema is generated from them:

```bash
PYTHONPATH=. .venv/Scripts/python -m shared.contracts   # writes shared/schemas/
```

## The parts that carry the design

**`intelligence/director.py`** — decides whether to speak, about what, and
whether to interrupt. Deterministic scoring: base priority, decayed by age,
penalised by topic cooldown and semantic repetition, boosted by silence. The
LLM writes the words; it does not choose the moment.

**`intelligence/safety.py`** — a control with tests, not a tone instruction.
Blocks numeric price claims when `MarketState.confidence != LIVE`, and blocks
outcome-certainty language. Runs after generation, before audio.

**`intelligence/memory.py`** — four layers plus repetition detection. Ships
with character n-gram cosine (no dependencies); swap in embeddings behind
`SimilarityIndex` when soak tests show it is needed.

**Session isolation** is enforced in four independent layers — process, Redis
namespace, `session_id` on every record, and a runtime assertion in the comment
pipeline. `tests/test_isolation.py` includes the adversarial vocabulary test.

## Soak testing — the test that actually matters

Ten minutes of good output proves nothing. The failures that kill a 24/7 product
only appear over days: content exhaustion, repetition drift, dead air, and the
~48 hours a week that spot gold is closed.

```bash
PYTHONPATH=. .venv/Scripts/python -m runtime.soak --hours 24
PYTHONPATH=. .venv/Scripts/python -m runtime.soak --hours 72 --start-friday
```

Runs on a simulated clock, so 72 hours takes about a minute. Current results:

| Run | Utterances | Longest silence | Drift | Result |
|-----|-----------:|----------------:|-------|--------|
| 24h weekday | 401 (16.7/h) | 6.0 min | 337 -> 332 | PASS |
| 72h with weekend | 920 (12.8/h) | 9.3 min | 2641 -> 1813 | content exhausted |

The weekend run still reports exhaustion, and that is accurate rather than a
bug: 46 topics x 8 angles = 368 beats cannot fill a 49-hour close without
reuse. The planner degrades rather than falling silent (see `degraded_level`),
but **the inventory needs to be roughly 3x larger** — target ~120 topics, so
nothing repeats inside 6-8 hours. That is writing, not engineering.

Offline soak numbers understate throughput badly: templates cannot produce
hundreds of distinct utterances, so most planned content is correctly blocked
as repetitive. Offline runs validate STRUCTURE. Use `--live` to judge content.

## Two behaviours worth knowing

**Repetition policy is priority-dependent.** A severity-5 market event
regenerates once and then speaks *anyway* rather than being silently dropped
for sounding similar to something earlier. High priority regenerates once then
drops; anything lower drops immediately. Found by running the dry run — the
break of structure was being swallowed.

**Clocks are injected everywhere.** `SpeechIntent.created_at`, `Director.tick`,
and the runtime's `on_*` methods all take an explicit `now`. Mixing wall-clock
intent creation with a simulated scoring clock makes every intent look instantly
expired.

## Not built yet

M2 real market engine · M3 screen-capture adapter and classification · M4 device
agent and audio routing · M5 supervision and observability · M6 scale to 3 ·
M7 dashboard and 7 sessions.

## Notes

- Copy `.env.example` to `.env`. Never commit `.env`.
- This project should live in its own repository with the `Automated` auto-push
  watcher **off**. Per-file auto-commits bypass PRs and branch protection, which
  is the opposite of what the two-owner contract workflow needs.
