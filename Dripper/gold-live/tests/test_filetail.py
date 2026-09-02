"""File-tailing comment adapter.

A development tool, but it is about to become the only comment source that runs
inside the live process -- so it needs to behave correctly or it will produce
misleading results when used to debug the rest of the pipeline.
"""

from __future__ import annotations

import asyncio

import pytest

from platform_.adapters.filetail import FileTailAdapter
from shared.contracts import HealthState


def adapter(tmp_path, **kw) -> FileTailAdapter:
    return FileTailAdapter(
        session_id="SESSION_001", path=tmp_path / "comments.txt", **kw
    )


# -- parsing ---------------------------------------------------------------


def test_bare_line_becomes_a_comment(tmp_path):
    """Typing a question should just work, with no format to remember."""
    comment = adapter(tmp_path).parse("Where is resistance on Gold?")
    assert comment is not None
    assert comment.text_raw == "Where is resistance on Gold?"
    assert comment.session_id == "SESSION_001"


def test_handle_prefix_is_honoured(tmp_path):
    comment = adapter(tmp_path).parse("zaraFX: where is resistance?")
    assert comment is not None
    assert comment.text_raw == "where is resistance?"


def test_a_colon_mid_sentence_is_not_treated_as_an_author(tmp_path):
    """Otherwise 'One thing: gold is quiet' loses its first two words."""
    comment = adapter(tmp_path).parse("One thing: gold looks quiet today")
    assert comment is not None
    assert comment.text_raw == "One thing: gold looks quiet today"


@pytest.mark.parametrize("line", ["", "   ", "# a comment about the file"])
def test_blank_and_hash_lines_are_ignored(tmp_path, line):
    assert adapter(tmp_path).parse(line) is None


def test_author_is_hashed(tmp_path):
    comment = adapter(tmp_path).parse("realname99: hello")
    assert comment is not None
    assert "realname99" not in comment.author_hash


# -- tailing ---------------------------------------------------------------


async def test_starts_at_the_end_by_default(tmp_path):
    """An old file must not replay its whole history into a live session."""
    path = tmp_path / "comments.txt"
    path.write_text("old one\nold two\n", encoding="utf-8")

    a = FileTailAdapter(session_id="S1", path=path)
    await a.connect()
    assert a.read_new() == []

    path.open("a", encoding="utf-8").write("a new one\n")
    assert [c.text_raw for c in a.read_new()] == ["a new one"]


async def test_from_start_replays_the_file(tmp_path):
    path = tmp_path / "comments.txt"
    path.write_text("one\ntwo\n", encoding="utf-8")
    a = FileTailAdapter(session_id="S1", path=path, from_start=True)
    await a.connect()
    assert [c.text_raw for c in a.read_new()] == ["one", "two"]


async def test_creates_the_file_if_absent(tmp_path):
    a = adapter(tmp_path)
    await a.connect()
    assert (tmp_path / "comments.txt").exists()


async def test_reads_only_what_is_new(tmp_path):
    a = adapter(tmp_path)
    await a.connect()
    path = tmp_path / "comments.txt"

    path.open("a", encoding="utf-8").write("first\n")
    assert len(a.read_new()) == 1
    assert a.read_new() == [], "already-read lines must not repeat"

    path.open("a", encoding="utf-8").write("second\n")
    assert [c.text_raw for c in a.read_new()] == ["second"]


async def test_truncation_is_survived(tmp_path):
    """Clearing the file mid-session must not leave the adapter seeking past
    the end and permanently silent."""
    a = adapter(tmp_path)
    await a.connect()
    path = tmp_path / "comments.txt"

    path.open("a", encoding="utf-8").write("one\ntwo\nthree\n")
    assert len(a.read_new()) == 3

    path.write_text("fresh\n", encoding="utf-8")
    assert [c.text_raw for c in a.read_new()] == ["fresh"]


async def test_async_iteration_yields_appended_lines(tmp_path):
    a = adapter(tmp_path, poll_interval_s=0.05)
    await a.connect()
    path = tmp_path / "comments.txt"

    got = []

    async def collect():
        async for comment in a.comments():
            got.append(comment.text_raw)
            if len(got) >= 2:
                return

    task = asyncio.create_task(collect())
    await asyncio.sleep(0.1)
    path.open("a", encoding="utf-8").write("where is resistance?\nshould i buy now?\n")
    await asyncio.wait_for(task, timeout=3.0)
    await a.disconnect()

    assert got == ["where is resistance?", "should i buy now?"]


async def test_a_read_failure_does_not_kill_the_session(tmp_path):
    a = adapter(tmp_path)
    await a.connect()
    a.path = tmp_path / "gone" / "missing.txt"  # unreadable
    assert a.read_new() == []


# -- health ----------------------------------------------------------------


async def test_health_down_before_connect(tmp_path):
    assert (await adapter(tmp_path).health()).state is HealthState.DOWN


async def test_health_ok_once_running(tmp_path):
    a = adapter(tmp_path)
    await a.connect()
    health = await a.health()
    assert health.state is HealthState.OK
    assert health.session_id == "SESSION_001"


# -- interface parity ------------------------------------------------------


def test_satisfies_the_platform_adapter_interface():
    from platform_.adapters.base import PlatformAdapter

    assert issubclass(FileTailAdapter, PlatformAdapter)


async def test_emits_the_same_shape_as_the_real_adapters(tmp_path):
    """The point of the tool: everything downstream must be unable to tell the
    difference between this and a real platform."""
    a = adapter(tmp_path)
    await a.connect()
    (tmp_path / "comments.txt").open("a", encoding="utf-8").write("where is resistance?\n")

    comment = a.read_new()[0]
    assert comment.session_id and comment.platform_msg_id and comment.author_hash
    assert comment.text_norm == comment.text_raw.lower()
    assert comment.classification is None, "classification happens in the pipeline"


async def test_pipeline_accepts_its_output(tmp_path):
    """Straight into the real comment pipeline, unmodified."""
    from intelligence.comments import CommentPipeline

    a = adapter(tmp_path)
    await a.connect()
    (tmp_path / "comments.txt").open("a", encoding="utf-8").write(
        "where is resistance on gold?\n"
    )
    comment = a.read_new()[0]

    scored = await CommentPipeline("SESSION_001").process(comment)
    assert scored is not None
    assert scored.comment.classification is not None
