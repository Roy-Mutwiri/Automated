"""A renderer whose only job is to show which camera you are on.

It loads nothing: no LivePortrait, no diffusion, no face model, no CUDA. It
reads seven PNGs off disk and blits whichever one is selected. Switching is a
hard cut because it is an array copy.

This exists because judging a camera plan and judging a human are different
questions, and the second one was blocking the first. The mannequin in these
frames is the debug proxy; identity is not being evaluated here and the overlay
says so rather than letting the picture imply otherwise.

The preview source is chosen by an explicit hierarchy, and the last rung is a
labelled placeholder - never another camera's image. Silently showing cam1 when
cam5 was requested would answer the question wrongly and look like a success.
"""

from __future__ import annotations

import cv2
import numpy as np

from ..types import AvatarPose
from .base import RendererInfo
from .camera_manager import CameraManager


class CameraPreviewRenderer:
    """Shows the selected camera's still. Costs a memcpy per frame."""

    def __init__(self, manager: CameraManager, width: int = 1280,
                 height: int = 720) -> None:
        self.manager = manager
        self.width = width
        self.height = height
        self._cache: dict[str, np.ndarray] = {}
        self._info = RendererInfo(
            name="camera-preview",
            resolution=(width, height),
            device="cpu",
            photoreal=False,
            notes="camera angle visualisation with the debug proxy human; "
                  "loads no avatar model",
        )

    @property
    def info(self) -> RendererInfo:
        return self._info

    def _placeholder(self, key: str, reason: str) -> np.ndarray:
        """A frame that admits what it is.

        Substituting a different camera's picture here would be the worst
        available option: the shot would look plausible and answer the wrong
        question.
        """
        frame = np.full((self.height, self.width, 3), 26, np.uint8)
        cv2.rectangle(frame, (40, 40), (self.width - 40, self.height - 40),
                      (58, 58, 66), 2)
        cv2.putText(frame, key.upper(), (70, 150), cv2.FONT_HERSHEY_SIMPLEX,
                    2.4, (90, 150, 230), 3, cv2.LINE_AA)
        cv2.putText(frame, "NO PREVIEW RENDERED", (72, 210),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (200, 210, 225), 2, cv2.LINE_AA)
        for i, line in enumerate((reason,
                                  "",
                                  "Generate previews with:",
                                  "  python tools/render_camera_previews.py")):
            cv2.putText(frame, line, (72, 270 + i * 34),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.62, (150, 160, 178), 1,
                        cv2.LINE_AA)
        return frame

    def _load(self, key: str) -> np.ndarray:
        cached = self._cache.get(key)
        if cached is not None:
            return cached

        view = self.manager.view(key)
        if view is None:
            frame = self._placeholder(key, "This camera is not in the rig.")
        elif not view.physical:
            frame = self._placeholder(
                key, "Legacy 2D rig: this camera has no rendered viewpoint.")
        elif not view.has_preview():
            frame = self._placeholder(key, "No still has been rendered yet.")
        else:
            img = cv2.imread(str(view.preview))
            if img is None:
                frame = self._placeholder(key, f"Could not read {view.preview.name}.")
            else:
                if img.shape[1::-1] != (self.width, self.height):
                    img = cv2.resize(img, (self.width, self.height),
                                     interpolation=cv2.INTER_AREA)
                frame = img
        self._cache[key] = frame
        return frame

    def invalidate(self) -> None:
        """Drop cached stills, after previews have been re-rendered."""
        self._cache.clear()

    def render(self, pose: AvatarPose) -> np.ndarray:  # noqa: ARG002
        """The pose is ignored on purpose: these are stills of a fixed moment.

        Animating the proxy here would imply the preview tracks the behaviour
        engine, which it does not - it is one frozen timestamp per camera.
        """
        key = self.manager.current
        if key is None:
            return self._placeholder("cam?", "No camera is selected.")
        return self._load(key).copy()
