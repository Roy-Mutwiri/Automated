"""The renderer seam.

Any backend that can turn an `AvatarPose` into a BGR frame satisfies this.
The point of keeping it this narrow is that the rendering strategy is the part
of this project most likely to change - the research in
docs/avatar_model_research.md compares several viable approaches and explicitly
declines to lock into one before benchmarking. Everything upstream of this
interface (the behaviour engine, which is the hard and slow part to get right)
survives that decision either way.

It is also what makes the eventual lip-sync integration tractable: an
audio-driven mouth model becomes a renderer that consumes the same pose plus an
audio feature stream, rather than a rewrite.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

import numpy as np

from ..types import AvatarPose

__all__ = ["Renderer", "RendererInfo"]


class RendererInfo:
    """What a backend reports about itself, for the debug overlay and README."""

    def __init__(
        self,
        name: str,
        resolution: tuple[int, int],
        device: str = "cpu",
        photoreal: bool = False,
        notes: str = "",
    ) -> None:
        self.name = name
        self.resolution = resolution
        self.device = device
        self.photoreal = photoreal
        self.notes = notes

    def __str__(self) -> str:
        w, h = self.resolution
        return f"{self.name} {w}x{h} on {self.device}"


@runtime_checkable
class Renderer(Protocol):
    """Turns a pose into a frame."""

    @property
    def info(self) -> RendererInfo:
        ...

    def render(self, pose: AvatarPose) -> np.ndarray:
        """Return an HxWx3 uint8 BGR frame for this pose.

        Implementations must not raise for a recoverable problem. The
        application loop keeps the last good frame on failure, per the brief's
        requirement that one bad frame never black-frames or crashes the
        output, but a backend that raises on every call will still stall the
        stream - so backends should degrade internally where they can.
        """
        ...

    def close(self) -> None:
        """Release GPU memory and any other resources."""
        ...
