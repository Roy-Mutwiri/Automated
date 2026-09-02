"""OCR backends for reading the LIVE Studio comment panel.

Kept behind an interface for two reasons: the engine choice is a benchmark
result rather than a decision (PaddleOCR and RapidOCR trade accuracy against
startup cost differently on different hardware), and a fake backend makes the
whole capture pipeline testable without a screen.

Row-wise, not panel-wise. Whole-panel OCR merges adjacent comments and mangles
the author/text boundary; segmenting into rows first is markedly more accurate
and lets a single bad row be dropped instead of the whole frame.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass

log = logging.getLogger(__name__)


@dataclass(slots=True)
class OCRLine:
    text: str
    confidence: float
    #: Vertical position in the crop, used to order rows top to bottom.
    y: int = 0


class OCREngine(ABC):
    name: str

    @abstractmethod
    def read(self, image) -> list[OCRLine]:
        """Return one OCRLine per detected text row, ordered top to bottom."""

    def warmup(self) -> None:  # pragma: no cover - trivial
        """Load models ahead of first use. First-call latency is otherwise
        paid during a live stream."""
        return None


class FakeOCR(OCREngine):
    """Scripted OCR for tests. `frames` is a list of per-frame line lists."""

    name = "fake"

    def __init__(self, frames: list[list[OCRLine]] | None = None) -> None:
        self.frames = frames or []
        self.calls = 0

    def read(self, image) -> list[OCRLine]:
        if not self.frames:
            return []
        result = self.frames[min(self.calls, len(self.frames) - 1)]
        self.calls += 1
        return result


class PaddleOCREngine(OCREngine):
    """PaddleOCR. Strong on small anti-aliased UI text, which is what this is.

    Runs on the device beside LIVE Studio, not centrally -- shipping frames to
    the server would waste bandwidth for no benefit and add a network hop to
    the ingest path.
    """

    name = "paddle"

    def __init__(self, lang: str = "en", use_gpu: bool = False) -> None:
        self.lang = lang
        self.use_gpu = use_gpu
        self._ocr = None

    def warmup(self) -> None:
        if self._ocr is not None:
            return
        try:
            from paddleocr import PaddleOCR
        except ImportError as exc:  # pragma: no cover - depends on host install
            raise RuntimeError(
                "paddleocr not installed. pip install paddleocr paddlepaddle"
            ) from exc
        self._ocr = PaddleOCR(use_angle_cls=False, lang=self.lang, show_log=False)

    def read(self, image) -> list[OCRLine]:
        if self._ocr is None:
            self.warmup()
        assert self._ocr is not None
        try:
            raw = self._ocr.ocr(image, cls=False)
        except Exception as exc:
            log.warning("OCR read failed: %s", exc)
            return []

        lines: list[OCRLine] = []
        for block in raw or []:
            for entry in block or []:
                try:
                    box, (text, confidence) = entry
                    y = int(min(p[1] for p in box))
                    lines.append(
                        OCRLine(text=str(text), confidence=float(confidence), y=y)
                    )
                except (ValueError, TypeError, IndexError):
                    continue
        lines.sort(key=lambda line: line.y)
        return lines


def build_ocr(engine: str = "paddle", **kwargs) -> OCREngine:
    if engine == "paddle":
        return PaddleOCREngine(**kwargs)
    if engine == "fake":
        return FakeOCR(**kwargs)
    raise ValueError(f"unknown OCR engine: {engine}")
