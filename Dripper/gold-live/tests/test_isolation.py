"""Session isolation -- the requirement that must hold architecturally.

A comment received by SESSION_003 must never reach SESSION_005, and memory from
one session must never become context for another. Tested adversarially: two
sessions with deliberately distinctive vocabularies, then grep for leakage.
"""

from __future__ import annotations

import pytest

from intelligence.comments import CommentPipeline
from intelligence.memory import SessionMemory
from shared.contracts import CommentEvent, Envelope, utcnow
from shared.events import InMemoryBus, wrap


def comment(session_id: str, text: str, author: str = "viewer") -> CommentEvent:
    return CommentEvent(
        session_id=session_id,
        platform="mock",
        platform_msg_id=CommentEvent.synth_msg_id(author, text),
        author_hash=CommentEvent.hash_author(author, "salt"),
        text_raw=text,
        text_norm=text.lower(),
    )


async def test_pipeline_rejects_foreign_session_comment():
    """Isolation layer 4: assert, don't trust."""
    pipe = CommentPipeline("SESSION_001")
    foreign = comment("SESSION_005", "where is resistance")
    with pytest.raises(ValueError, match="SESSION_005.*SESSION_001"):
        await pipe.process(foreign)


async def test_pipeline_accepts_own_session_comment():
    pipe = CommentPipeline("SESSION_001")
    result = await pipe.process(comment("SESSION_001", "where is resistance"))
    assert result is not None
    assert result.comment.session_id == "SESSION_001"


def test_memory_instances_do_not_share_state():
    """Isolation layer 1: separate objects, separate memory."""
    a = SessionMemory("SESSION_002")
    b = SessionMemory("SESSION_007")
    a.record_utterance("The Tokyo range is compressing.", "mkt:range")
    assert a.recent_transcript() == ["The Tokyo range is compressing."]
    assert b.recent_transcript() == []
    assert not b.topic_on_cooldown("mkt:range")
    repetitive, _ = b.is_repetitive("The Tokyo range is compressing.")
    assert not repetitive, "session B must not see session A's utterances"


async def test_bus_does_not_deliver_across_sessions():
    """Isolation layer 2: the bus filters on session_id."""
    bus = InMemoryBus()
    received: list[Envelope] = []

    sub = bus.subscribe(["viewer.comment_received"], session_id="SESSION_002")

    async def collect():
        async for env in sub:
            received.append(env)
            if len(received) >= 1:
                return

    import asyncio

    task = asyncio.create_task(collect())
    await asyncio.sleep(0)

    await bus.publish(
        wrap("viewer.comment_received", {"text": "for seven"}, "t1", session_id="SESSION_007")
    )
    await bus.publish(
        wrap("viewer.comment_received", {"text": "for two"}, "t2", session_id="SESSION_002")
    )
    await asyncio.wait_for(task, timeout=1.0)

    assert len(received) == 1
    assert received[0].payload["text"] == "for two"


async def test_shared_plane_events_reach_every_session():
    """Market state is shared; only conversation is isolated."""
    bus = InMemoryBus()
    got: list[Envelope] = []
    sub = bus.subscribe(["market.state_updated"], session_id="SESSION_004")

    import asyncio

    async def collect():
        async for env in sub:
            got.append(env)
            return

    task = asyncio.create_task(collect())
    await asyncio.sleep(0)
    await bus.publish(wrap("market.state_updated", {"price": 3652.4}, "t3", session_id=None))
    await asyncio.wait_for(task, timeout=1.0)
    assert got[0].payload["price"] == 3652.4


async def test_adversarial_vocabulary_leak():
    """The test that actually matters: distinctive vocab must not cross over."""
    a_pipe, b_pipe = CommentPipeline("SESSION_A"), CommentPipeline("SESSION_B")
    a_mem, b_mem = SessionMemory("SESSION_A"), SessionMemory("SESSION_B")

    for i in range(50):
        ca = comment("SESSION_A", f"what about the tokyo range {i}", f"a{i}")
        cb = comment("SESSION_B", f"what about the frankfurt open {i}", f"b{i}")
        ra = await a_pipe.process(ca)
        rb = await b_pipe.process(cb)
        if ra:
            a_mem.record_question(ra.comment.text_norm)
        if rb:
            b_mem.record_question(rb.comment.text_norm)

    a_text = " ".join(a_mem.audience_questions)
    b_text = " ".join(b_mem.audience_questions)
    assert "frankfurt" not in a_text
    assert "tokyo" not in b_text
