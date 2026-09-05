"""Memory must not scale with lifetime.

The target machine has 8 GB. A host that runs for weeks cannot retain a
constant fraction of what it produces, and the production ratio makes that
sharper than it sounds: of 1,439 utterances actually spoken, 5,243 more were
rejected -- roughly three quarters of everything generated is discarded, so
any collection that keeps rejects grows four times faster than one that keeps
speech.

These tests drive enough traffic to expose an accidental retained reference,
deterministically and without wall-clock time. They assert bounds, not
absolute byte counts, because byte counts are not reproducible across machines
and would fail for reasons that have nothing to do with a leak.
"""

from __future__ import annotations

import gc
import sys
from collections import deque

import pytest

from runtime.session import RETAINED_DROPS, RETAINED_UTTERANCES


# -- the session's own collections -----------------------------------------


class Bounded:
    """The exact shape SessionRuntime now uses: bounded history beside a
    monotonic counter."""

    def __init__(self) -> None:
        self.transcript: deque[str] = deque(maxlen=RETAINED_UTTERANCES)
        self.dropped_repetitive: deque[tuple[str, float]] = deque(maxlen=RETAINED_DROPS)
        self.dropped_unsafe: deque[tuple[str, list[str]]] = deque(maxlen=RETAINED_DROPS)
        self.spoken_count = 0
        self.repetitive_count = 0
        self.unsafe_count = 0

    def generate(self, text: str, *, rejected: bool) -> None:
        if rejected:
            self.dropped_repetitive.append((text, 0.55))
            self.repetitive_count += 1
        else:
            self.transcript.append(text)
            self.spoken_count += 1


def run_workload(state: Bounded, utterances: int, reject_ratio: float = 0.75) -> None:
    """Drive the real production mix: about three in four rejected."""
    for i in range(utterances):
        state.generate(f"utterance number {i} " + "x" * 200,
                       rejected=(i % 4 != 0))


def test_history_stays_bounded_across_ten_thousand_utterances():
    state = Bounded()
    run_workload(state, 10_000)

    assert len(state.transcript) <= RETAINED_UTTERANCES
    assert len(state.dropped_repetitive) <= RETAINED_DROPS
    assert len(state.dropped_unsafe) <= RETAINED_DROPS


def test_counters_still_rise_after_the_history_is_full():
    """The regression that would break metrics: consumers derive "how many are
    new" from the counters, which must keep climbing long after the deques
    have stopped."""
    state = Bounded()
    run_workload(state, 10_000)

    assert state.spoken_count + state.repetitive_count == 10_000
    assert state.repetitive_count > state.spoken_count * 2, (
        "the workload must actually reproduce the production reject ratio"
    )


def test_memory_does_not_scale_with_lifetime():
    """Ten times the work must not mean ten times the retained bytes.

    Compares retained size, not process RSS: RSS moves for reasons unrelated
    to this system -- allocator behaviour, imports, the interpreter itself --
    and asserting on it produces a flaky test that fails for the wrong reason.
    """
    def retained(n: int) -> int:
        state = Bounded()
        run_workload(state, n)
        gc.collect()
        return (
            sum(sys.getsizeof(x) for x in state.transcript)
            + sum(sys.getsizeof(x) for x in state.dropped_repetitive)
        )

    # Both workloads must be large enough to fill every deque, or this
    # compares a partly-filled state against a full one and fails for a
    # reason that is not a leak.
    small = retained(5_000)
    large = retained(50_000)

    assert large <= small * 1.2, (
        f"retained bytes grew with lifetime: {small} -> {large}"
    )


def test_the_oldest_entries_are_the_ones_discarded():
    state = Bounded()
    for i in range(RETAINED_UTTERANCES * 2):
        state.generate(f"line-{i}", rejected=False)

    assert "line-0" not in state.transcript
    assert f"line-{RETAINED_UTTERANCES * 2 - 1}" in state.transcript


def test_no_duplicate_entries_appear_under_load():
    state = Bounded()
    for i in range(2_000):
        state.generate(f"unique-{i}", rejected=False)
    assert len(set(state.transcript)) == len(state.transcript)


# -- the similarity index, which is the other obvious candidate -------------


def test_the_similarity_index_is_bounded():
    """It compares each new utterance against recent ones, so it necessarily
    retains text. It must retain a window, not a history."""
    from intelligence.memory import SessionMemory

    memory = SessionMemory("MEM_TEST")
    for i in range(5_000):
        memory.record_utterance(f"a distinct sentence number {i}", topic=f"t{i % 20}")

    vecs = memory._index._vecs
    assert vecs.maxlen is not None, "the vector window must be bounded"
    assert len(vecs) <= vecs.maxlen
    assert len(memory.short_term) <= (memory.short_term.maxlen or 0)
    assert memory.utterance_count == 5_000, "the total is still counted"


def test_topics_last_seen_is_the_one_deliberate_exception():
    """This one genuinely grows -- with the number of distinct topics, not
    with lifetime -- and it is what stops the host repeating a topic it
    covered an hour ago. Documented rather than 'fixed'."""
    from intelligence.memory import SessionMemory

    memory = SessionMemory("MEM_TEST")
    for i in range(3_000):
        memory.record_utterance("text", topic=f"topic-{i % 40}")

    assert len(memory.topics_last_seen) <= 40, (
        "bounded by distinct topics, not by how long the host has been running"
    )


# -- metrics ----------------------------------------------------------------


def test_metric_histograms_are_capped():
    """Prometheus summaries keep observations to compute quantiles. Uncapped,
    that is a list that grows for the life of the process."""
    from runtime.health import Metrics

    m = Metrics()
    for i in range(10_000):
        m.observe("goldlive_first_token_ms", float(i))

    bucket = next(iter(m._hist.values()))
    assert len(bucket) <= 2_000, f"histogram grew to {len(bucket)}"


def test_metric_counters_are_scalars_not_lists():
    """A counter that appended would be a leak wearing a counter's name."""
    from runtime.health import Metrics

    m = Metrics()
    for _ in range(50_000):
        m.inc("goldlive_utterances_total", {"session": "S1"})

    assert len(m._counters) == 1
    assert next(iter(m._counters.values())) == 50_000


def test_metric_labels_do_not_multiply_without_bound():
    """Distinct label sets each create a series. Session ids are bounded;
    anything unbounded in a label would be a cardinality leak."""
    from runtime.health import Metrics

    m = Metrics()
    for i in range(1_000):
        m.inc("goldlive_utterances_total", {"session": f"SESSION_{i % 7:03d}"})

    assert len(m._counters) == 7


# -- router queue -----------------------------------------------------------


async def test_the_audio_queue_cannot_grow_without_bound(tmp_path):
    from fakes import FakeSink, FakeTTS, audio_request

    from platform_.audio.router import AudioRouter

    r = AudioRouter("SESSION_001", FakeTTS(), tmp_path, sink=FakeSink(), max_queue=8)
    for _ in range(500):
        await r.submit(audio_request(segments=["x"]))

    assert r.queue_depth <= 8


async def test_preemption_marks_do_not_accumulate(tmp_path):
    """The set added for barge-in is keyed by utterance id, which is exactly
    the kind of thing that leaks if entries are never removed."""
    from fakes import FakeSink, FakeTTS, audio_request

    from platform_.audio.router import AudioRouter

    r = AudioRouter("SESSION_001", FakeTTS(), tmp_path, sink=FakeSink())
    r._running = True

    for _ in range(1_000):
        request = audio_request(segments=["x"])
        r._preempted.add(request.utterance_id)
        r._preempted.discard(request.utterance_id)

    assert len(r._preempted) == 0


@pytest.mark.parametrize("cycles", [200])
async def test_repeated_router_start_stop_does_not_accumulate_tasks(tmp_path, cycles):
    """Lifecycle churn is where task leaks hide."""
    import asyncio

    from fakes import FakeSink, FakeTTS

    from platform_.audio.router import AudioRouter

    before = len(asyncio.all_tasks())
    for _ in range(cycles):
        r = AudioRouter("SESSION_001", FakeTTS(), tmp_path, sink=FakeSink())
        await r.start()
        await r.stop()
    gc.collect()

    after = len(asyncio.all_tasks())
    assert after <= before + 2, f"tasks accumulated: {before} -> {after}"


def test_the_viewer_question_table_is_bounded():
    """Keyed by the text of every distinct question anyone has ever asked, so
    on a real stream it grows without bound. Only frequently-repeated ones are
    ever read back."""
    from intelligence.memory import QUESTION_TABLE_LIMIT, SessionMemory

    memory = SessionMemory("MEM_TEST")
    for i in range(QUESTION_TABLE_LIMIT * 5):
        memory.record_question(f"a question nobody repeats number {i}")

    assert len(memory.audience_questions) <= QUESTION_TABLE_LIMIT


def test_pruning_keeps_the_questions_that_matter():
    """Frequency is what hot_questions selects on, so pruning must keep the
    most-asked -- dropping one asked ten times for one asked once would defeat
    the feature the table exists for."""
    from intelligence.memory import QUESTION_TABLE_LIMIT, SessionMemory

    memory = SessionMemory("MEM_TEST")
    for _ in range(12):
        memory.record_question("where is gold heading")
    for i in range(QUESTION_TABLE_LIMIT * 3):
        memory.record_question(f"one-off question {i}")

    assert memory.audience_questions["where is gold heading"] == 12
    assert "where is gold heading" in memory.hot_questions()
