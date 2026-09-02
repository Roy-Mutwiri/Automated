"""Local-model comment classification.

The classifier previously called a hosted API and had no callers at all. It now
runs on the same local model as generation, and is wired into the live session.

The property that matters most: a model verdict may RAISE the risk flag but
never lower one the regex already set. A classifier failure must not be the
reason the host starts giving trading advice.
"""

from __future__ import annotations


from intelligence.comments import CommentPipeline, build_classifier
from platform_.llm.base import ChatMessage, LLMResult
from shared.contracts import CommentEvent, CommentIntent


class ScriptedLLM:
    name = "scripted"

    def __init__(self, reply: str = "", fail: bool = False) -> None:
        self.reply = reply
        self.fail = fail
        self.calls: list[str] = []

    async def stream(self, *a, **k):
        yield self.reply

    async def complete(self, messages: list[ChatMessage], **kw) -> LLMResult:
        if self.fail:
            raise ConnectionError("model down")
        self.calls.append(messages[-1].content)
        return LLMResult(text=self.reply, model=self.name)

    async def health(self) -> bool:
        return not self.fail


def comment(text: str, session_id: str = "S1") -> CommentEvent:
    return CommentEvent(
        session_id=session_id, platform="mock",
        platform_msg_id=CommentEvent.synth_msg_id("u", text),
        author_hash="h", text_raw=text, text_norm=" ".join(text.lower().split()),
    )


async def test_model_may_raise_the_risk_flag():
    llm = ScriptedLLM('{"intent": "trade_advice_req", "is_risk_sensitive": true, "relevance": 0.9}')
    classify = build_classifier(llm)
    # Phrasing the regex does not catch, but a model recognises.
    result = await classify(comment("gold gonna moon or nah, what would you do"))
    assert result is not None and result.is_risk_sensitive


async def test_model_cannot_lower_a_regex_risk_flag():
    """The safety-critical property."""
    llm = ScriptedLLM('{"intent": "greeting", "is_risk_sensitive": false, "relevance": 0.1}')
    classify = build_classifier(llm, escalate_only=False)
    result = await classify(comment("should i buy now?"))
    assert result is not None
    assert result.is_risk_sensitive, "a model must never clear a code-detected risk flag"


async def test_obvious_cases_never_reach_the_model():
    """Re-confirming that 'gm' is a greeting is GPU time spent on nothing."""
    llm = ScriptedLLM('{"intent": "market_q", "is_risk_sensitive": false, "relevance": 0.5}')
    classify = build_classifier(llm, escalate_only=True)

    for text in ["gm everyone", "FREE SIGNALS DM ME", "where is resistance", "should i buy now?"]:
        await classify(comment(text))
    assert llm.calls == [], "clear-cut comments must be decided in code"


async def test_ambiguous_cases_do_reach_the_model():
    llm = ScriptedLLM('{"intent": "education_q", "is_risk_sensitive": false, "relevance": 0.8}')
    classify = build_classifier(llm, escalate_only=True)
    result = await classify(comment("what do you make of the dollar here"))
    assert llm.calls, "an ambiguous comment should be escalated"
    assert result is not None and result.intent is CommentIntent.EDUCATION_Q


async def test_model_failure_falls_back_to_the_heuristic():
    classify = build_classifier(ScriptedLLM(fail=True))
    pipeline = CommentPipeline("S1", classifier=classify)
    scored = await pipeline.process(comment("what about the dollar today"))
    assert scored is not None, "a dead classifier must not drop comments"
    assert scored.comment.classification is not None


async def test_garbage_model_output_falls_back():
    classify = build_classifier(ScriptedLLM("I think this is probably a question?"))
    result = await classify(comment("what about the dollar today"))
    assert result is not None, "unparseable output keeps the heuristic verdict"


async def test_relevance_is_clamped():
    llm = ScriptedLLM('{"intent": "market_q", "is_risk_sensitive": false, "relevance": 7.5}')
    classify = build_classifier(llm, escalate_only=False)
    result = await classify(comment("what about the dollar"))
    assert result is not None and 0.0 <= result.relevance <= 1.0


async def test_temperature_is_zero():
    """A classifier that varies run to run is not a classifier."""
    llm = ScriptedLLM('{"intent": "market_q", "is_risk_sensitive": false, "relevance": 0.5}')
    captured = {}
    original = llm.complete

    async def spy(messages, **kw):
        captured.update(kw)
        return await original(messages, **kw)

    llm.complete = spy  # type: ignore[method-assign]
    await build_classifier(llm, escalate_only=False)(comment("what about the dollar"))
    assert captured.get("temperature") == 0.0
