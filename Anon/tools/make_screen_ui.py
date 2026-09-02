"""Draw the monitor content templates. Deterministic vector-ish drawing, no diffusion.

These are the *source* images that get perspective-mapped onto the panels. They
are stored separately from the plate on purpose: the application can swap the
screen later without touching a single pixel of the room.

## Design rules, and why

**Environmental storytelling, not information.** The screen has to say "serious
workstation" from across the frame. It never has to be read - at final scale the
primary panel occupies roughly 430 x 250 px and is behind the focal plane, so a
legible interface is not merely unnecessary, it is a liability.

**No generated prose, no prices, no candlesticks.** Fake text is the single most
reliable way to make an image read as AI, and fake numbers invite a viewer to
check them. Everything here is abstract geometry: panels, bars, blocks, one
restrained line. The only glyphs in the whole design are the TRADE FIX wordmark,
drawn as stroked geometry rather than typeset copy.

**Low contrast, dark, low saturation.** The screen is the cool counterpoint to a
warm tungsten room; it is not a light source competing with the face. The
palette tops out well below white, and the exposure match in
`monitor_replace.py` pulls it to the plate's own LCD luminance afterwards.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np

# Charcoal through to a single muted cyan accent.
#
# The greys are exactly neutral (B == G == R) on purpose. The exposure fit in
# monitor_replace.py lifts this template with a gain around 4-6 and a negative
# offset, and any tint in the base is multiplied by that gain: an earlier
# palette that was only three units bluer than neutral came back as a navy
# panel. Neutral greys mean the accent is the only thing on screen carrying
# colour, which is what section 20 asks for.
BG        = (17, 17, 17)        # BGR, near-black charcoal
GRID      = (26, 26, 26)
PANEL     = (25, 25, 26)
PANEL_HI  = (34, 34, 35)
EDGE      = (46, 46, 47)
INK       = (112, 112, 112)     # dim neutral marks
ACCENT    = (146, 124, 74)      # muted cyan-teal, the single cool accent
ACCENT_DIM = (92, 80, 50)


def rr(img, p0, p1, colour, r=8, thickness=-1):
    """Rounded rectangle. Sharp corners read as a chart, not an interface."""
    x0, y0 = p0
    x1, y1 = p1
    r = int(min(r, (x1 - x0) // 2, (y1 - y0) // 2))
    if thickness < 0:
        cv2.rectangle(img, (x0 + r, y0), (x1 - r, y1), colour, -1)
        cv2.rectangle(img, (x0, y0 + r), (x1, y1 - r), colour, -1)
        for cx, cy in ((x0 + r, y0 + r), (x1 - r, y0 + r),
                       (x0 + r, y1 - r), (x1 - r, y1 - r)):
            cv2.circle(img, (cx, cy), r, colour, -1)
    else:
        cv2.rectangle(img, (x0, y0), (x1, y1), colour, thickness, cv2.LINE_AA)


def wordmark(img, x, y, h, colour, gap=None):
    """TRADE FIX, drawn as strokes.

    Deliberately not typeset. The brief allows a small genuine wordmark and
    forbids AI-generated text; drawing the letterforms from line segments means
    the glyphs are exactly what was specified and cannot drift into gibberish.
    """
    w = int(h * 0.62)
    gap = gap if gap is not None else int(h * 0.30)
    t = max(int(h * 0.13), 2)

    def seg(a, b):
        cv2.line(img, (int(a[0]), int(a[1])), (int(b[0]), int(b[1])),
                 colour, t, cv2.LINE_AA)

    def letter(ch, ox):
        L, R, T, B = ox, ox + w, y, y + h
        M = y + h // 2
        if ch == "T":
            seg((L, T), (R, T)); seg(((L + R) // 2, T), ((L + R) // 2, B))
        elif ch == "R":
            seg((L, T), (L, B)); seg((L, T), (R, T)); seg((R, T), (R, M))
            seg((L, M), (R, M)); seg((L, M), (R, B))
        elif ch == "A":
            seg((L, B), ((L + R) // 2, T)); seg(((L + R) // 2, T), (R, B))
            seg((L + w // 5, M + h // 6), (R - w // 5, M + h // 6))
        elif ch == "D":
            seg((L, T), (L, B)); seg((L, T), (R - w // 4, T))
            seg((R - w // 4, T), (R, T + h // 4)); seg((R, T + h // 4), (R, B - h // 4))
            seg((R, B - h // 4), (R - w // 4, B)); seg((R - w // 4, B), (L, B))
        elif ch == "E":
            seg((L, T), (L, B)); seg((L, T), (R, T)); seg((L, M), (R - w // 5, M))
            seg((L, B), (R, B))
        elif ch == "F":
            seg((L, T), (L, B)); seg((L, T), (R, T)); seg((L, M), (R - w // 5, M))
        elif ch == "I":
            seg((L + t, T), (L + t, B))
        elif ch == "X":
            seg((L, T), (R, B)); seg((R, T), (L, B))
        return ox + (w if ch != "I" else w // 2) + gap

    ox = x
    for ch in "TRADE":
        ox = letter(ch, ox)
    ox += gap * 2
    for ch in "FIX":
        ox = letter(ch, ox)
    return ox


def sparkline(img, box, seed, colour, fill=True, points=90):
    """One restrained line. A smooth random walk, not a market."""
    x0, y0, x1, y1 = box
    rng = np.random.default_rng(seed)
    v = np.cumsum(rng.normal(0, 1.0, points))
    v = cv2.GaussianBlur(v.astype(np.float32).reshape(-1, 1), (0, 0), 2.2).ravel()
    v -= v.min()
    v /= max(v.ptp() if hasattr(v, "ptp") else np.ptp(v), 1e-6)
    xs = np.linspace(x0, x1, points)
    ys = y1 - v * (y1 - y0) * 0.82 - (y1 - y0) * 0.09
    pts = np.stack([xs, ys], 1).astype(np.int32)

    if fill:
        poly = np.vstack([pts, [[x1, y1], [x0, y1]]]).astype(np.int32)
        layer = img.copy()
        cv2.fillPoly(layer, [poly], tuple(int(c * 0.30) for c in colour))
        cv2.addWeighted(layer, 0.5, img, 0.5, 0, img)
    cv2.polylines(img, [pts], False, colour, 2, cv2.LINE_AA)


def grid(img, step, colour):
    h, w = img.shape[:2]
    for x in range(0, w, step):
        cv2.line(img, (x, 0), (x, h), colour, 1)
    for y in range(0, h, step):
        cv2.line(img, (0, y), (w, y), colour, 1)


def primary(w=1920, h=1080) -> np.ndarray:
    img = np.full((h, w, 3), BG, np.uint8)
    grid(img, 60, GRID)

    # Left nav rail. Icon blocks, one active.
    cv2.rectangle(img, (0, 0), (96, h), (24, 20, 17), -1)
    for i in range(6):
        y = 150 + i * 84
        c = ACCENT_DIM if i == 1 else PANEL_HI
        rr(img, (30, y), (66, y + 36), c, 8)

    # Top bar with the wordmark and three small status dots.
    cv2.rectangle(img, (96, 0), (w, 74), (15, 13, 11), -1)
    cv2.line(img, (96, 74), (w, 74), EDGE, 1)
    wordmark(img, 132, 26, 24, INK)
    for i, c in enumerate((ACCENT, INK, PANEL_HI)):
        cv2.circle(img, (w - 60 - i * 34, 37), 6, c, -1, cv2.LINE_AA)

    # Main chart panel. One line, faint gridlines, no axis labels.
    rr(img, (130, 104), (1180, 566), PANEL, 10)
    rr(img, (130, 104), (1180, 566), EDGE, 10, 1)
    for i in range(1, 5):
        y = 104 + i * (566 - 104) // 5
        cv2.line(img, (150, y), (1160, y), GRID, 1)
    rr(img, (152, 126), (280, 146), PANEL_HI, 6)      # a title placeholder block
    rr(img, (152, 156), (214, 172), ACCENT_DIM, 4)
    sparkline(img, (154, 200, 1158, 544), seed=7, colour=ACCENT)

    # Right column: three cards, each a couple of blocks and a small bar.
    for i in range(3):
        y0 = 104 + i * 156
        rr(img, (1210, y0), (1880, y0 + 138), PANEL, 10)
        rr(img, (1210, y0), (1880, y0 + 138), EDGE, 10, 1)
        rr(img, (1234, y0 + 24), (1360, y0 + 40), PANEL_HI, 5)
        rr(img, (1234, y0 + 56), (1300 + i * 90, y0 + 74), ACCENT_DIM if i == 0 else PANEL_HI, 5)
        for b in range(9):
            bh = 10 + ((b * 37 + i * 11) % 40)
            x = 1234 + b * 34
            rr(img, (x, y0 + 118 - bh), (x + 22, y0 + 118), PANEL_HI, 3)

    # Waveform strip. Vertical bars, symmetric about a centreline.
    rr(img, (130, 590), (900, 1000), PANEL, 10)
    rr(img, (130, 590), (900, 1000), EDGE, 10, 1)
    rng = np.random.default_rng(3)
    amp = cv2.GaussianBlur(rng.random((1, 76)).astype(np.float32), (0, 0), 1.6).ravel()
    amp /= amp.max()
    mid = 795
    for i, a in enumerate(amp):
        x = 156 + i * 9
        hgt = int(12 + a * 150)
        cv2.line(img, (x, mid - hgt), (x, mid + hgt),
                 ACCENT_DIM if i % 7 == 0 else PANEL_HI, 4)
    cv2.line(img, (150, mid), (880, mid), GRID, 1)

    # Status block grid. Reads as a monitoring surface at a glance.
    rr(img, (930, 590), (1880, 1000), PANEL, 10)
    rr(img, (930, 590), (1880, 1000), EDGE, 10, 1)
    # Seeded random, not a modular formula: an arithmetic pattern lays a visible
    # diagonal across the grid, which reads as decoration rather than status.
    srng = np.random.default_rng(11)
    for r in range(6):
        for c in range(16):
            x, y = 958 + c * 56, 620 + r * 60
            u = srng.random()
            col = ACCENT_DIM if u > 0.94 else (PANEL_HI if u > 0.55 else (34, 29, 25))
            rr(img, (x, y), (x + 44, y + 44), col, 6)
    return img


def secondary(w=1280, h=720) -> np.ndarray:
    """The angled panel. Only its left third is ever visible, so that is where
    the content lives - the rest is deliberately quiet."""
    img = np.full((h, w, 3), BG, np.uint8)
    grid(img, 48, GRID)
    cv2.rectangle(img, (0, 0), (w, 56), (15, 13, 11), -1)
    cv2.line(img, (0, 56), (w, 56), EDGE, 1)
    wordmark(img, 28, 18, 18, INK)

    for i in range(7):
        y = 86 + i * 62
        rr(img, (28, y), (300, y + 48), PANEL, 8)
        rr(img, (28, y), (300, y + 48), EDGE, 8, 1)
        rr(img, (46, y + 14), (46 + 60 + (i * 29) % 90, y + 30),
           ACCENT_DIM if i == 2 else PANEL_HI, 4)
        cv2.circle(img, (280, y + 24), 5, ACCENT if i == 2 else PANEL_HI, -1, cv2.LINE_AA)

    rr(img, (330, 86), (900, 400), PANEL, 10)
    rr(img, (330, 86), (900, 400), EDGE, 10, 1)
    sparkline(img, (352, 120, 878, 378), seed=19, colour=ACCENT_DIM)

    rr(img, (330, 424), (900, 680), PANEL, 10)
    rr(img, (330, 424), (900, 680), EDGE, 10, 1)
    for r in range(4):
        for c in range(9):
            x, y = 352 + c * 58, 448 + r * 56
            rr(img, (x, y), (x + 44, y + 40),
               PANEL_HI if (r + c) % 4 else (34, 29, 25), 5)
    return img


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out-dir", default="assets/screens")
    args = ap.parse_args()
    d = Path(args.out_dir)
    d.mkdir(parents=True, exist_ok=True)

    for name, img in (("default_streamer_ui.png", primary()),
                      ("secondary_panel.png", secondary())):
        cv2.imwrite(str(d / name), img)
        lum = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        print(f"[screen] {name}  {img.shape[1]}x{img.shape[0]}  "
              f"luma mean {lum.mean():.1f} p99 {np.percentile(lum, 99):.0f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
