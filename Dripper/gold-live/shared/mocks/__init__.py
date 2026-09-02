"""Mocks for every contract.

CI rule: a public interface under platform_/ without a corresponding mock here
fails the build. This is what lets the intelligence half be built without
waiting for the platform half, and vice versa.
"""

from shared.mocks.comments import MockCommentSource
from shared.mocks.market import MockMarketEngine
from shared.mocks.tts import FileTTS, MockTTS

__all__ = ["FileTTS", "MockCommentSource", "MockMarketEngine", "MockTTS"]
