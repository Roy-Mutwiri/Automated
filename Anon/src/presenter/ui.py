"""On-frame dropdown menus, drawn with OpenCV.

The preview window is a single `cv2.imshow` surface, so there is no widget
toolkit to hang a menu off - the menu has to be drawn into the frame and driven
by a mouse callback. That is the whole of this module.

Two details are worth knowing:

**Window coordinates are not image coordinates.** The preview window is created
`WINDOW_NORMAL` and is therefore resizable, and OpenCV reports mouse positions
in *window* pixels. Once someone drags the corner, every hit test is wrong by
the scale factor. `_to_image` corrects for it using the window's actual image
rect, so the menus keep working at any window size.

**Selection is polled, not called back.** The callback runs on OpenCV's UI
thread and switching an outfit takes seconds - blocking in there would freeze
the window mid-redraw with no way to show progress. Instead the callback only
records the choice, and the render loop picks it up with `take_selection()`
when it is ready to act on it.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np

__all__ = ["Option", "Menu", "DropdownBar"]

FONT = cv2.FONT_HERSHEY_SIMPLEX

BG = (34, 30, 28)
BG_OPEN = (44, 39, 36)
EDGE = (86, 78, 72)
TEXT = (222, 226, 232)
TEXT_DIM = (120, 124, 132)
TITLE = (150, 160, 175)
HILITE = (196, 148, 62)      # the room's teal accent, so the UI belongs to it


@dataclass
class Option:
    key: str
    label: str
    enabled: bool = True


@dataclass
class Menu:
    ident: str
    title: str
    options: list[Option]
    selected: str
    width: int = 230
    open: bool = False
    x: int = 0
    y: int = 0

    ROW = 30
    HEAD = 34

    @property
    def label(self) -> str:
        for o in self.options:
            if o.key == self.selected:
                return o.label
        return self.selected

    def head_rect(self) -> tuple[int, int, int, int]:
        return self.x, self.y, self.width, self.HEAD

    def row_rect(self, i: int) -> tuple[int, int, int, int]:
        return self.x, self.y + self.HEAD + i * self.ROW, self.width, self.ROW


def _inside(rect, x, y) -> bool:
    rx, ry, rw, rh = rect
    return rx <= x < rx + rw and ry <= y < ry + rh


class DropdownBar:
    """A row of dropdowns drawn into the frame."""

    def __init__(self, menus: list[Menu], origin: tuple[int, int] = (14, 14),
                 gap: int = 10) -> None:
        self.origin = origin
        self.gap = gap
        self.menus: list[Menu] = []
        self._pending: tuple[str, str] | None = None
        self._window: str | None = None
        self._frame_size = (0, 0)
        self.set_menus(menus)

    def set_menus(self, menus: list[Menu]) -> None:
        """Replace the menus, preserving which one is open.

        Rebuilding rather than mutating is what keeps the enabled flags honest:
        whether an outfit is pickable depends on the *other* dropdown's current
        value, so the lists have to be recomputed after every change. Laying
        them out here rather than at the call site is not tidiness - a menu that
        skips layout draws at the origin of the frame, on top of the presenter.
        """
        x, y = self.origin
        for menu in menus:
            menu.x, menu.y = x, y
            x += menu.width + self.gap
        was_open = {m.ident for m in self.menus if m.open}
        for menu in menus:
            menu.open = menu.ident in was_open
        self.menus = menus

    # -- input --------------------------------------------------------------
    def attach(self, window: str, frame_size: tuple[int, int]) -> None:
        self._window = window
        self._frame_size = frame_size
        cv2.setMouseCallback(window, self._on_mouse)

    def _to_image(self, x: int, y: int) -> tuple[int, int]:
        """Map window coordinates onto frame coordinates."""
        try:
            _, _, ww, wh = cv2.getWindowImageRect(self._window)
        except cv2.error:
            return x, y
        fw, fh = self._frame_size
        if ww <= 0 or wh <= 0 or not fw or not fh:
            return x, y
        return int(x * fw / ww), int(y * fh / wh)

    def _on_mouse(self, event, x, y, flags, param) -> None:  # noqa: ARG002
        if event != cv2.EVENT_LBUTTONDOWN:
            return
        x, y = self._to_image(x, y)

        for menu in self.menus:
            if _inside(menu.head_rect(), x, y):
                was = menu.open
                self.close_all()
                menu.open = not was
                return

        for menu in self.menus:
            if not menu.open:
                continue
            for i, option in enumerate(menu.options):
                if _inside(menu.row_rect(i), x, y):
                    if option.enabled:
                        self._pending = (menu.ident, option.key)
                        menu.open = False
                    return

        # A click anywhere else dismisses. Without this the only way to close a
        # menu opened by accident is to pick something from it.
        self.close_all()

    def close_all(self) -> None:
        for menu in self.menus:
            menu.open = False

    @property
    def is_open(self) -> bool:
        return any(m.open for m in self.menus)

    def take_selection(self) -> tuple[str, str] | None:
        """Return and clear the pending choice, if any."""
        pending, self._pending = self._pending, None
        return pending

    def set_selected(self, ident: str, key: str) -> None:
        for menu in self.menus:
            if menu.ident == ident:
                menu.selected = key

    def cycle(self, ident: str, step: int = 1) -> str | None:
        """Advance a menu to its next enabled option, for keyboard use."""
        for menu in self.menus:
            if menu.ident != ident:
                continue
            usable = [o for o in menu.options if o.enabled]
            if not usable:
                return None
            keys = [o.key for o in usable]
            i = keys.index(menu.selected) if menu.selected in keys else -step
            return keys[(i + step) % len(keys)]
        return None

    # -- drawing ------------------------------------------------------------
    def draw(self, frame: np.ndarray) -> None:
        for menu in self.menus:
            self._draw_head(frame, menu)
        # Open lists are drawn in a second pass so one menu's list is never
        # painted over by the next menu's header.
        for menu in self.menus:
            if menu.open:
                self._draw_list(frame, menu)

    def _panel(self, frame, rect, colour, alpha=0.88) -> None:
        x, y, w, h = rect
        h_img, w_img = frame.shape[:2]
        x0, y0 = max(x, 0), max(y, 0)
        x1, y1 = min(x + w, w_img), min(y + h, h_img)
        if x1 <= x0 or y1 <= y0:
            return
        region = frame[y0:y1, x0:x1].astype(np.float32)
        frame[y0:y1, x0:x1] = (
            region * (1 - alpha) + np.array(colour, np.float32) * alpha
        ).astype(np.uint8)

    def _draw_head(self, frame, menu: Menu) -> None:
        x, y, w, h = menu.head_rect()
        self._panel(frame, (x, y, w, h), BG_OPEN if menu.open else BG)
        cv2.rectangle(frame, (x, y), (x + w - 1, y + h - 1), EDGE, 1, cv2.LINE_AA)
        cv2.putText(frame, menu.title.upper(), (x + 10, y + 13),
                    FONT, 0.32, TITLE, 1, cv2.LINE_AA)
        cv2.putText(frame, menu.label, (x + 10, y + 27),
                    FONT, 0.46, TEXT, 1, cv2.LINE_AA)
        # Caret, pointing the way the list will go.
        cx, cy = x + w - 16, y + h // 2 + 2
        pts = ([(cx - 5, cy - 2), (cx + 5, cy - 2), (cx, cy + 4)] if not menu.open
               else [(cx - 5, cy + 3), (cx + 5, cy + 3), (cx, cy - 3)])
        cv2.fillPoly(frame, [np.array(pts, np.int32)], TITLE, cv2.LINE_AA)

    def _draw_list(self, frame, menu: Menu) -> None:
        x, y, w, h = menu.x, menu.y + menu.HEAD, menu.width, menu.ROW * len(menu.options)
        self._panel(frame, (x, y, w, h), BG, alpha=0.94)
        cv2.rectangle(frame, (x, y), (x + w - 1, y + h - 1), EDGE, 1, cv2.LINE_AA)
        for i, option in enumerate(menu.options):
            rx, ry, rw, rh = menu.row_rect(i)
            if option.key == menu.selected:
                self._panel(frame, (rx + 1, ry, rw - 2, rh), BG_OPEN, alpha=0.9)
                cv2.rectangle(frame, (rx + 1, ry + 4), (rx + 3, ry + rh - 4),
                              HILITE, -1)
            colour = TEXT if option.enabled else TEXT_DIM
            cv2.putText(frame, option.label, (rx + 12, ry + 20),
                        FONT, 0.44, colour, 1, cv2.LINE_AA)
            if not option.enabled:
                # Says *why* it cannot be picked. A greyed row with no
                # explanation reads as a bug.
                cv2.putText(frame, "not generated", (rx + rw - 84, ry + 20),
                            FONT, 0.32, TEXT_DIM, 1, cv2.LINE_AA)


def draw_busy(frame: np.ndarray, message: str) -> None:
    """Centre banner for the seconds an outfit change takes to prepare."""
    h, w = frame.shape[:2]
    bw, bh = 420, 62
    x, y = (w - bw) // 2, (h - bh) // 2
    region = frame[y:y + bh, x:x + bw].astype(np.float32)
    frame[y:y + bh, x:x + bw] = (region * 0.18 + np.array(BG, np.float32) * 0.82
                                 ).astype(np.uint8)
    cv2.rectangle(frame, (x, y), (x + bw - 1, y + bh - 1), HILITE, 1, cv2.LINE_AA)
    size = cv2.getTextSize(message, FONT, 0.56, 1)[0]
    cv2.putText(frame, message, (x + (bw - size[0]) // 2, y + 38),
                FONT, 0.56, TEXT, 1, cv2.LINE_AA)
