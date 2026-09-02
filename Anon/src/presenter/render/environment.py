"""Procedurally generated streaming-room background.

## Why generate it rather than use a photograph

A real streaming setup shot on a portrait lens is almost entirely *defocused
light*: bokeh from fairy lights, an LED strip washing a wall, monitor spill,
the soft silhouette of a shelf. At f/1.8 there is essentially no sharp detail
behind the subject. That is exactly the kind of image that can be synthesised
convincingly, because there is no fine structure to give it away - the thing
that normally betrays a generated background is edge detail, and here there
isn't any.

Doing it this way also solves three problems the brief cares about at once:

* **Temporal stability.** Computed once, static forever. It cannot warp,
  wobble, or flicker, which the brief lists as unacceptable.
* **Licensing.** Nothing is downloaded or copied; there is no stock-photo
  provenance to track.
* **Lighting match.** The background is built to match the *subject's* existing
  key light rather than the other way round. Mismatched lighting between
  subject and background is the single loudest tell in any composite, and it
  cannot be fixed afterwards.

## The lighting constraint

The source portrait is lit warm from the front-left, with warm golden bokeh
behind. A cold blue-and-purple gaming wash - the obvious "streamer" look -
would fight that and immediately read as a cut-out.

So the room is warm-dominant, with the RGB accent restrained and pushed to the
edges where a real LED strip would sit. `led_strength` is deliberately capped
low. Turning it up produces a more striking image and a less believable one.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import cv2
import numpy as np

from ..behavior.randomness import Rng

__all__ = ["RoomStyle", "render_streaming_room", "render_desk_foreground"]


@dataclass
class RoomStyle:
    """Appearance of the generated room. All colours are BGR."""

    # Wall behind the subject. Warm neutral, dark enough that the lit face
    # separates from it without needing a rim light the source does not have.
    wall_dark: tuple[int, int, int] = (26, 24, 26)
    wall_warm: tuple[int, int, int] = (58, 62, 78)

    # Fairy / string lights. Warm, matching the source's existing bokeh.
    bokeh_warm: tuple[int, int, int] = (150, 196, 235)
    bokeh_count: int = 44
    bokeh_radius: tuple[int, int] = (9, 42)

    # RGB accent. Kept low - see the module docstring. Two hues on opposite
    # sides is the common LED-strip arrangement.
    led_left: tuple[int, int, int] = (168, 74, 196)    # magenta
    led_right: tuple[int, int, int] = (196, 148, 62)   # teal/cyan
    led_strength: float = 0.28

    # Cool spill from the monitor the presenter is facing. Comes from the
    # camera side, so it brightens the lower-centre of the wall slightly.
    monitor_glow: tuple[int, int, int] = (176, 150, 120)
    monitor_strength: float = 0.20

    # Shelf silhouette. Very low contrast; it reads as "a room" without
    # asserting any detail the blur would have destroyed anyway.
    shelf: bool = True
    shelf_y: float = 0.30          # fraction of height
    shelf_darkness: float = 0.40

    # Desk edge across the bottom of frame, closer to camera than the wall and
    # therefore less blurred. This is what actually sells "seated at a desk".
    desk: bool = True
    desk_y: float = 0.89
    desk_color: tuple[int, int, int] = (38, 40, 48)

    # Defocus. The wall is far behind the subject, so it is blurred hard.
    blur: float = 0.055            # fraction of the short side
    vignette: float = 0.42
    grain: float = 3.0             # sensor noise; without it the gradients band


def _radial_sprite(size: int) -> np.ndarray:
    """A soft disc with a slightly brighter rim - a defocused point light.

    A real out-of-focus highlight is not a gaussian blob: the lens aperture
    projects a disc with a marginally hot edge. Reproducing that rim is what
    makes synthetic bokeh look like glass rather than like an airbrush.
    """
    yy, xx = np.mgrid[0:size, 0:size].astype(np.float32)
    c = (size - 1) * 0.5
    r = np.sqrt((xx - c) ** 2 + (yy - c) ** 2) / max(c, 1e-3)
    disc = np.clip(1.0 - r, 0.0, 1.0)
    disc = np.where(r < 1.0, 0.72 + 0.28 * np.clip((r - 0.55) / 0.45, 0, 1), 0.0)
    disc *= np.clip(1.0 - (r - 0.86) / 0.14, 0.0, 1.0)   # soft outer falloff
    return np.clip(disc, 0.0, 1.0)


def render_streaming_room(
    width: int,
    height: int,
    style: RoomStyle | None = None,
    seed: int | None = 7,
) -> np.ndarray:
    """Render the room. Called once at startup; cost is irrelevant.

    Built at 2x and downsampled, which is what gives the gradients and bokeh
    edges their smoothness - generating at final resolution leaves visible
    stepping in the large soft areas.
    """
    style = style or RoomStyle()
    rng = Rng(seed)

    w, h = width * 2, height * 2
    short = min(w, h)

    # -- wall: vertical gradient, warmer and brighter toward the middle -----
    ramp = np.linspace(0.0, 1.0, h, dtype=np.float32)[:, None]
    falloff = 1.0 - np.abs(ramp - 0.42) * 1.55
    falloff = np.clip(falloff, 0.0, 1.0) ** 1.4
    dark = np.array(style.wall_dark, np.float32)
    warm = np.array(style.wall_warm, np.float32)
    room = dark[None, None, :] + (warm - dark)[None, None, :] * falloff[..., None]
    room = np.repeat(room, w, axis=1) if room.shape[1] == 1 else room
    room = np.ascontiguousarray(np.broadcast_to(room, (h, w, 3)).copy())

    # -- LED strips: broad washes from either edge --------------------------
    xs = np.linspace(0.0, 1.0, w, dtype=np.float32)[None, :, None]
    left_wash = np.clip(1.0 - xs / 0.42, 0.0, 1.0) ** 2.1
    right_wash = np.clip((xs - 0.58) / 0.42, 0.0, 1.0) ** 2.1
    room += np.array(style.led_left, np.float32)[None, None, :] * left_wash * style.led_strength
    room += np.array(style.led_right, np.float32)[None, None, :] * right_wash * style.led_strength

    # -- shelf silhouette ---------------------------------------------------
    if style.shelf:
        sy = int(h * style.shelf_y)
        thickness = max(int(h * 0.014), 2)
        # Drawn as a *darkening* of the wall rather than as black. A hard black
        # bar survives even heavy blur as a visible band, which reads as a
        # graphic element rather than as a piece of furniture.
        shelf_layer = np.zeros((h, w), np.float32)
        cv2.rectangle(shelf_layer, (0, sy), (w, sy + thickness), 1.0, -1)
        x = int(w * 0.04)
        while x < w * 0.96:
            ow = rng.uniform(0.025, 0.07) * w
            oh = rng.uniform(0.03, 0.09) * h
            if rng.chance(0.6):
                cv2.rectangle(
                    shelf_layer,
                    (int(x), int(sy - oh)), (int(x + ow), sy),
                    float(rng.uniform(0.35, 0.8)), -1,
                )
            x += ow + rng.uniform(0.02, 0.07) * w
        shelf_layer = cv2.GaussianBlur(
            shelf_layer, (max(int(short * 0.012) | 1, 3),) * 2, 0
        )
        room *= (1.0 - style.shelf_darkness * shelf_layer)[..., None]

    # -- defocus the wall ---------------------------------------------------
    # Everything above is at wall distance and goes out of focus together.
    # Bokeh is added *after* this: a defocused highlight already carries its
    # own softness and its characteristic hot rim, and blurring it a second
    # time turns it into a shapeless smear. Getting this order wrong is what
    # makes synthetic bokeh look like a stock overlay.
    k = int(short * style.blur) | 1
    room = cv2.GaussianBlur(room, (k, k), 0)

    # -- bokeh: defocused fairy lights --------------------------------------
    glow = np.zeros((h, w, 3), np.float32)
    base = np.array(style.bokeh_warm, np.float32)
    for _ in range(style.bokeh_count):
        radius = int(rng.uniform(*style.bokeh_radius) * (short / 720.0))
        radius = max(radius, 6)
        sprite = _radial_sprite(radius * 2)

        # Clustered along the upper wall, the way a string of lights hangs,
        # rather than scattered uniformly - uniform placement reads as a
        # particle effect.
        cx = int(rng.uniform(0.02, 0.98) * w)
        cy = int(rng.truncated_gauss(0.30 * h, 0.16 * h, 0.02 * h, 0.72 * h))

        intensity = rng.uniform(0.35, 1.0)
        tint = base * rng.uniform(0.82, 1.12)
        tint = np.clip(tint, 0, 255)

        x0, y0 = cx - radius, cy - radius
        x1, y1 = x0 + radius * 2, y0 + radius * 2
        sx0, sy0 = max(0, -x0), max(0, -y0)
        sx1, sy1 = radius * 2 - max(0, x1 - w), radius * 2 - max(0, y1 - h)
        if sx1 <= sx0 or sy1 <= sy0:
            continue
        patch = sprite[sy0:sy1, sx0:sx1, None] * tint[None, None, :] * intensity
        gy0, gx0 = max(y0, 0), max(x0, 0)
        glow[gy0:gy0 + patch.shape[0], gx0:gx0 + patch.shape[1]] += patch

    # Only a whisper of blur, to seat the sprites in the image without
    # destroying the aperture rim that makes them read as glass.
    glow = cv2.GaussianBlur(glow, (max(int(short * 0.006) | 1, 3),) * 2, 0)
    room += glow

    # -- monitor spill ------------------------------------------------------
    if style.monitor_strength > 0:
        yy = np.linspace(0.0, 1.0, h, dtype=np.float32)[:, None, None]
        xx2 = np.linspace(-1.0, 1.0, w, dtype=np.float32)[None, :, None]
        spill = np.clip(1.0 - np.abs(xx2) * 1.15, 0.0, 1.0) * np.clip(
            (yy - 0.35) / 0.65, 0.0, 1.0
        )
        room += (
            np.array(style.monitor_glow, np.float32)[None, None, :]
            * spill * style.monitor_strength
        )

    # NOTE: the desk is not drawn here. It belongs in *front* of the presenter,
    # not behind, and is produced separately by render_desk_foreground().

    # -- vignette -----------------------------------------------------------
    if style.vignette > 0:
        yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
        nx = (xx / w - 0.5) * 2.0
        ny = (yy / h - 0.5) * 2.0
        radial = np.sqrt(nx * nx + ny * ny) / math.sqrt(2.0)
        room *= (1.0 - style.vignette * np.clip(radial, 0, 1) ** 2.0)[..., None]

    room = cv2.resize(room, (width, height), interpolation=cv2.INTER_AREA)

    # -- grain --------------------------------------------------------------
    # Every real camera has noise. Its absence is subtle but perceptible: clean
    # synthetic gradients band, and banding against a photographic subject is a
    # giveaway.
    if style.grain > 0:
        noise = np.random.default_rng(seed).normal(0.0, style.grain,
                                                   (height, width, 1))
        room += noise

    return np.clip(room, 0, 255).astype(np.uint8)


def render_desk_foreground(
    width: int,
    height: int,
    style: RoomStyle | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """The desk edge, which sits in FRONT of the presenter.

    Returns ``(bgr, alpha)`` at the output resolution.

    Two reasons this is a separate foreground layer rather than part of the
    wall:

    **Depth.** A seated presenter is behind their desk. Painting the desk into
    the background and then drawing the person over it puts the furniture in
    the wrong place, and the eye notices even when it cannot say why. Occluding
    the bottom of the subject is what actually communicates "seated at a desk"
    rather than "photograph of a person with a desk drawn behind them".

    **It hides a real defect.** The source portrait is a fixed crop, so the
    torso is cut off at the image boundary, leaving hard vertical edges at the
    bottom of frame where the body simply stops. A foreground desk crossing
    above that line occludes it completely. Solving a composition problem and a
    clipping artefact with the same element is the reason it is worth doing
    properly.

    It is nearer the camera than the wall, so it takes a *weaker* blur - it
    sits at a different point on the focus ramp. Blurring foreground and
    background identically is what flattens a render into a painted backdrop.
    """
    style = style or RoomStyle()
    w, h = width * 2, height * 2
    short = min(w, h)

    desk = np.zeros((h, w, 3), np.float32)
    alpha = np.zeros((h, w), np.float32)

    dy = int(h * style.desk_y)
    cv2.rectangle(desk, (0, dy), (w, h), style.desk_color, -1)
    cv2.rectangle(alpha, (0, dy), (w, h), 1.0, -1)

    # Specular sheen along the front edge, picking up the monitor the presenter
    # is facing. A flat slab reads as a rectangle; the highlight reads as a
    # surface.
    edge = max(int(h * 0.008), 2)
    cv2.line(desk, (0, dy), (w, dy),
             tuple(float(min(c * 2.6, 255)) for c in style.desk_color), edge)
    # Warm falloff toward the sides, matching the LED washes on the wall.
    xs = np.linspace(-1.0, 1.0, w, dtype=np.float32)[None, :, None]
    desk[dy:] *= (1.0 - 0.35 * np.abs(xs) ** 1.6)

    k = max(int(short * style.blur * 0.35) | 1, 3)
    desk = cv2.GaussianBlur(desk, (k, k), 0)
    alpha = cv2.GaussianBlur(alpha, (k, k), 0)

    desk = cv2.resize(desk, (width, height), interpolation=cv2.INTER_AREA)
    alpha = cv2.resize(alpha, (width, height), interpolation=cv2.INTER_AREA)
    return (
        np.clip(desk, 0, 255).astype(np.uint8),
        np.clip(alpha, 0.0, 1.0)[..., None].astype(np.float32),
    )
