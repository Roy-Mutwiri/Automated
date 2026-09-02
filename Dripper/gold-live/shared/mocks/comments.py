"""Scripted viewer comments, including the cases that must be handled carefully.

Deliberately includes: a duplicate, spam, a provocation, and two risk-sensitive
"should I buy" questions -- the ones that must route to the constrained path.
"""

from __future__ import annotations

from shared.contracts import CommentEvent

# (beat, author, text)
SCRIPT: list[tuple[int, str, str]] = [
    (1, "trader_mike", "gm everyone"),
    (2, "zaraFX", "where is resistance right now?"),
    (3, "goldbug99", "why does gold drop when dollar goes up"),
    (4, "trader_mike", "where is resistance right now?"),      # duplicate of beat 2
    (5, "spam_bot_01", "FREE SIGNALS DM ME NOW 100% WIN RATE"),
    (6, "newbie_kev", "what is a liquidity sweep??"),
    (7, "zaraFX", "should i buy now?"),                        # risk-sensitive
    (8, "hater123", "this is a bot lol"),                      # provocation
    (9, "sam_t", "whats the difference between BOS and CHOCH"),
    (10, "goldbug99", "is it going to 3700 today"),            # risk-sensitive
    (12, "newbie_kev", "how do you set a stop loss on gold"),
    (13, "quietwatcher", "great stream"),
]


class MockCommentSource:
    """Yields CommentEvents for a given beat, already normalised."""

    def __init__(self, session_id: str, salt: str = "dryrun-salt") -> None:
        self.session_id = session_id
        self.salt = salt

    @staticmethod
    def normalise(text: str) -> str:
        return " ".join(text.lower().split())

    def at_beat(self, beat: int) -> list[CommentEvent]:
        out = []
        for b, author, text in SCRIPT:
            if b != beat:
                continue
            out.append(
                CommentEvent(
                    session_id=self.session_id,
                    platform="mock",
                    platform_msg_id=CommentEvent.synth_msg_id(author, text),
                    author_hash=CommentEvent.hash_author(author, self.salt),
                    text_raw=text,
                    text_norm=self.normalise(text),
                )
            )
        return out
