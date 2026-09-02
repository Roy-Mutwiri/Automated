"""Personas: config, not code.

Differentiation is by AUDIENCE and TIMEFRAME, not by voice affectation. Seven
hosts with different accents saying the same thing about the same chart at the
same moment is both boring and a coordinated-behaviour signal. Seven hosts
covering different timeframes for different experience levels is a real product.

Adding session 8 means adding a YAML file. It must never mean changing code.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass(slots=True)
class Persona:
    persona_id: str
    display_name: str
    audience: str
    primary_timeframe: str
    focus: list[str]
    avoid: list[str]
    voice_id: str
    #: Free-text style guidance. Kept short on purpose -- long style prompts
    #: produce parody. Variation comes from the Director's topic selection.
    style: str
    max_sentences: int = 4

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> Persona:
        return cls(
            persona_id=d["persona_id"],
            display_name=d["display_name"],
            audience=d["audience"],
            primary_timeframe=d["primary_timeframe"],
            focus=list(d.get("focus", [])),
            avoid=list(d.get("avoid", [])),
            voice_id=d.get("voice_id", "default"),
            style=d.get("style", ""),
            max_sentences=int(d.get("max_sentences", 4)),
        )

    def system_prompt(self) -> str:
        """Stable across the whole session -- this is the cached prefix.

        Nothing volatile may appear here. No timestamps, no prices, no counts.
        A single changing byte invalidates the cache on every request.
        """
        focus = ", ".join(self.focus)
        avoid = ", ".join(self.avoid) if self.avoid else "nothing in particular"
        return f"""You are the live host of a Gold (XAUUSD) trading stream.

Your audience: {self.audience}
Your primary timeframe: {self.primary_timeframe}
What you cover: {focus}
What you leave to others: {avoid}

Style: {self.style}

How you speak:
- You are talking out loud on a live stream. Write speech, not prose. No
  headings, no bullet points, no markdown, no emoji.
- At most {self.max_sentences} sentences. Usually fewer. Short is better.
- Vary how you open. Never start consecutive turns the same way. Do not say
  "guys", do not stack rhetorical questions, do not use hype.
- Weave viewer questions into the conversation naturally. Never say
  "Viewer John asked" -- just answer as though it came up in conversation.

How you handle the market:
- You describe scenarios and conditions, never predictions or certainties.
  "If price accepts above that level, the bullish case gets more interesting"
  is right. "Gold will go up" is forbidden.
- Distinguish what you observe from what you infer. Say which is which when
  it matters.
- You never give personalised trading advice, never tell anyone to buy or
  sell, and never state or imply a guaranteed outcome.
- If asked whether to buy, redirect to process: how someone would frame the
  decision, what would invalidate it, how risk is managed. Never the answer.
- You never invent a price, a level, a news event or an economic release. If
  you do not have current data, you say so plainly and talk about something
  else."""


def load_personas(directory: str | Path) -> dict[str, Persona]:
    out: dict[str, Persona] = {}
    for path in sorted(Path(directory).glob("*.yaml")):
        data = yaml.safe_load(path.read_text(encoding="utf-8"))
        p = Persona.from_dict(data)
        out[p.persona_id] = p
    return out
