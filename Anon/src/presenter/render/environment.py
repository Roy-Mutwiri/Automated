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

__all__ = [
    "RoomStyle",
    "render_streaming_room",
    "render_desk_foreground",
    "render_mic_foreground",
]


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

    # Optical vignetting. Toward the frame edges the lens barrel clips the
    # exit pupil, so a highlight that is a circle in the middle of the frame
    # becomes a tangentially-oriented lens shape - "cat's eye" - at the
    # corners. It is strongest wide open, which is exactly the aperture this
    # room claims to have been shot at. Perfectly round discs everywhere is
    # the clearest signature of *rendered* rather than photographed bokeh.
    bokeh_cats_eye: float = 0.42

    # Lateral chromatic aberration: the channels focus at slightly different
    # scales, so out-of-focus highlights carry a colour fringe that grows
    # toward the corners. Fast lenses always show some.
    chromatic: float = 0.0022

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
    # Objects on it. Set-dressing convention is a small, odd number with
    # varied heights, grouped rather than spread evenly: an even rank of
    # similar blocks reads as a graphic pattern, not as someone's shelf.
    shelf_objects: int = 5
    # The shelf stands proud of the wall, so it is fractionally nearer the
    # camera and takes a correspondingly weaker blur. This is the third plane
    # in the focus ramp - wall, shelf, desk - and it is what stops the room
    # collapsing into a single painted backdrop.
    shelf_focus: float = 0.72

    # Desk edge across the bottom of frame, closer to camera than the wall and
    # therefore less blurred. This is what actually sells "seated at a desk".
    desk: bool = True
    desk_y: float = 0.89
    desk_color: tuple[int, int, int] = (38, 40, 48)

    # Boom microphone edging into frame. Foreground, so it is blurred harder
    # than the wall - see render_mic_foreground.
    mic: bool = True
    mic_side: str = "right"          # "right" or "left"
    mic_scale: float = 1.0
    mic_color: tuple[int, int, int] = (30, 31, 36)
    mic_opacity: float = 0.92

    # Defocus. The wall is far behind the subject, so it is blurred hard.
    blur: float = 0.055            # fraction of the short side
    vignette: float = 0.42
    grain: float = 3.0             # sensor noise; without it the gradients band

    # Exposure relative to the subject's face, as a luminance ratio. The
    # broadcast convention is a background 1-2 stops under the key; 0.35 is
    # 1.5 stops. Below this band the subject floats in a void, above it the
    # image goes flat no matter how good the key light is.
    exposure_ratio: float = 0.35
    exposure_limits: tuple[float, float] = (0.45, 1.9)

    # Light wrap: how far the plate's light bleeds back onto the silhouette,
    # as a fraction of the short side, and how strongly.
    wrap_width: float = 0.022
    wrap_strength: float = 0.55


def _radial_sprite(size: int, cats_eye: float = 0.0, angle: float = 0.0) -> np.ndarray:
    """A defocused point light: a soft disc with a slightly brighter rim.

    A real out-of-focus highlight is not a gaussian blob - the lens aperture
    projects a disc with a marginally hot edge. Reproducing that rim is what
    makes synthetic bokeh look like glass rather than like an airbrush.

    ``cats_eye`` adds optical vignetting. Away from the optical axis the barrel
    clips the exit pupil, and the highlight becomes the *intersection of two
    circles* offset along the radial direction - which is what physically
    happens, and why the resulting lens shape is elongated tangentially rather
    than radially. Modelling it as two circles rather than as a squashed
    ellipse costs nothing and gets the orientation right for free:
    ``max(r1, r2)`` is exactly 1.0 on the clipped boundary, so the aperture rim
    follows the new outline instead of the original circle.

    At ``cats_eye == 0`` the two circles coincide and this reduces exactly to
    the unclipped disc.
    """
    yy, xx = np.mgrid[0:size, 0:size].astype(np.float32)
    c = (size - 1) * 0.5
    nx = (xx - c) / max(c, 1e-3)
    ny = (yy - c) / max(c, 1e-3)

    if cats_eye > 1e-3:
        ox = math.cos(angle) * cats_eye
        oy = math.sin(angle) * cats_eye
        r1 = np.sqrt((nx - ox) ** 2 + (ny - oy) ** 2)
        r2 = np.sqrt((nx + ox) ** 2 + (ny + oy) ** 2)
        r = np.maximum(r1, r2)
    else:
        r = np.sqrt(nx * nx + ny * ny)

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

    # -- defocus the wall ---------------------------------------------------
    # Everything above is at wall distance and goes out of focus together.
    # Bokeh is added *after* this: a defocused highlight already carries its
    # own softness and its characteristic hot rim, and blurring it a second
    # time turns it into a shapeless smear. Getting this order wrong is what
    # makes synthetic bokeh look like a stock overlay.
    k = int(short * style.blur) | 1
    room = cv2.GaussianBlur(room, (k, k), 0)

    # -- shelf silhouette, one plane nearer than the wall -------------------
    # Applied *after* the wall blur and with a weaker kernel of its own. The
    # shelf stands proud of the wall, so it is not at the same focus distance,
    # and giving both the same blur is what flattens a room into a backdrop.
    if style.shelf:
        sy = int(h * style.shelf_y)
        thickness = max(int(h * 0.014), 2)
        # Drawn as a *darkening* of the wall rather than as black. A hard black
        # bar survives even heavy blur as a visible band, which reads as a
        # graphic element rather than as a piece of furniture.
        shelf_layer = np.zeros((h, w), np.float32)
        cv2.rectangle(shelf_layer, (0, sy), (w, sy + thickness), 1.0, -1)

        # A small odd number of objects in two loose groups, heights varied.
        # An even rank of similarly-sized blocks spread across the whole width
        # reads as a graphic pattern rather than as somebody's belongings, and
        # the eye picks that up even through this much blur.
        n = max(int(style.shelf_objects), 0)
        groups = 2 if n >= 4 else 1
        for g in range(groups):
            x = (0.10 + g * 0.44 + rng.uniform(0.0, 0.10)) * w
            count = n // groups + (1 if g < n % groups else 0)
            for _ in range(count):
                ow = rng.uniform(0.022, 0.055) * w
                oh = rng.uniform(0.025, 0.10) * h
                cv2.rectangle(
                    shelf_layer,
                    (int(x), int(sy - oh)), (int(x + ow), sy),
                    float(rng.uniform(0.35, 0.8)), -1,
                )
                x += ow + rng.uniform(0.008, 0.03) * w

        k_shelf = max(int(short * style.blur * style.shelf_focus) | 1, 3)
        shelf_layer = cv2.GaussianBlur(shelf_layer, (k_shelf, k_shelf), 0)
        room *= (1.0 - style.shelf_darkness * shelf_layer)[..., None]

    # -- bokeh: defocused fairy lights --------------------------------------
    glow = np.zeros((h, w, 3), np.float32)
    base = np.array(style.bokeh_warm, np.float32)
    for _ in range(style.bokeh_count):
        radius = int(rng.uniform(*style.bokeh_radius) * (short / 720.0))
        radius = max(radius, 6)

        # Clustered along the upper wall, the way a string of lights hangs,
        # rather than scattered uniformly - uniform placement reads as a
        # particle effect.
        cx = int(rng.uniform(0.02, 0.98) * w)
        cy = int(rng.truncated_gauss(0.30 * h, 0.16 * h, 0.02 * h, 0.72 * h))

        # Optical vignetting scales with distance from the optical axis. The
        # clip is applied along the radial direction, which leaves the surviving
        # lens shape elongated tangentially - the way real cat's-eye bokeh sits.
        dx = (cx / w - 0.5) * 2.0
        dy = (cy / h - 0.5) * 2.0
        r_norm = min(math.hypot(dx, dy) / math.sqrt(2.0), 1.0)
        sprite = _radial_sprite(
            radius * 2,
            style.bokeh_cats_eye * r_norm ** 1.6,
            math.atan2(dy, dx),
        )

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

    # -- lateral chromatic aberration --------------------------------------
    # Each channel focuses at a fractionally different scale, so a highlight
    # picks up a colour fringe that is zero on axis and grows toward the
    # corners. Applying it as a per-channel zoom about the frame centre is
    # exactly what the aberration is, and it lands only on the glow layer -
    # the wall has no edges left for a fringe to show up on.
    if style.chromatic > 0:
        for ch, s in ((0, 1.0 - style.chromatic), (2, 1.0 + style.chromatic)):
            M = np.array(
                [[s, 0.0, (1.0 - s) * w * 0.5], [0.0, s, (1.0 - s) * h * 0.5]],
                np.float32,
            )
            glow[:, :, ch] = cv2.warpAffine(
                glow[:, :, ch], M, (w, h),
                flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE,
            )

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


def render_mic_foreground(
    width: int,
    height: int,
    style: RoomStyle | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """A boom microphone intruding into frame, in FRONT of the presenter.

    Returns ``(bgr, alpha)`` at the output resolution.

    Nothing says "live stream" faster than a mic edging into the shot, and it is
    the cheapest realism cue available here - but only if it behaves like a real
    one, which means two things:

    **It is foreground, not scenery.** A boom mic is clamped to the desk, not to
    the presenter. Prompting it into the source portrait would weld it to his
    head, so it would swing every time he turned - the kind of error that is
    invisible in a still and glaring in motion. Composited as a static layer, it
    stays put while he moves behind it, which is what a real one does.

    **It is blurred *more* than the background, not less.** This is the part
    that is easy to get backwards. The mic sits nearer the lens than the
    subject, so it falls on the opposite side of the focal plane and goes
    heavily out of focus - much softer than the wall behind. A crisp mic in the
    corner reads instantly as a pasted-on graphic.

    Kept to one corner and well clear of the face; it is a framing cue, not a
    subject.
    """
    style = style or RoomStyle()
    if not style.mic:
        return (
            np.zeros((height, width, 3), np.uint8),
            np.zeros((height, width, 1), np.float32),
        )

    w, h = width * 2, height * 2
    short = min(w, h)
    layer = np.zeros((h, w, 3), np.float32)
    alpha = np.zeros((h, w), np.float32)

    # Boom arm entering from a bottom corner and angling up toward the subject,
    # with the capsule ending short of centre frame.
    sign = 1.0 if style.mic_side == "right" else -1.0
    cx = w * (0.86 if sign > 0 else 0.14)
    base = (int(cx + sign * w * 0.18), int(h * 1.02))
    tip = (int(cx - sign * w * 0.05), int(h * 0.60))

    arm_w = max(int(short * 0.016 * style.mic_scale), 3)
    for canvas, colour in ((layer, style.mic_color), (alpha, 1.0)):
        cv2.line(canvas, base, tip, colour, arm_w)

    # Shock mount ring and capsule body.
    body_len = int(h * 0.20 * style.mic_scale)
    body_w = max(int(short * 0.055 * style.mic_scale), 8)
    ang = math.atan2(tip[1] - base[1], tip[0] - base[0])
    bx = int(tip[0] - math.cos(ang) * body_len * 0.1)
    by = int(tip[1] - math.sin(ang) * body_len * 0.1)
    box = ((bx, by), (body_w, body_len), math.degrees(ang) + 90.0)
    pts = cv2.boxPoints(box).astype(np.int32)
    cv2.fillPoly(layer, [pts], style.mic_color)
    cv2.fillPoly(alpha, [pts], 1.0)

    # A soft highlight down one side of the capsule so it reads as a cylinder
    # rather than a flat slab once blurred.
    hi_box = ((bx - int(sign * body_w * 0.22), by), (max(body_w // 4, 2), body_len),
              math.degrees(ang) + 90.0)
    cv2.fillPoly(layer, [cv2.boxPoints(hi_box).astype(np.int32)],
                 tuple(float(min(c * 3.0, 190)) for c in style.mic_color))

    # Foreground defocus: stronger than the wall's, because this is nearer the
    # lens than the subject is.
    k = max(int(short * style.blur * 1.5) | 1, 9)
    layer = cv2.GaussianBlur(layer, (k, k), 0)
    alpha = cv2.GaussianBlur(alpha, (k, k), 0)

    layer = cv2.resize(layer, (width, height), interpolation=cv2.INTER_AREA)
    alpha = cv2.resize(alpha, (width, height), interpolation=cv2.INTER_AREA)
    alpha = np.clip(alpha * style.mic_opacity, 0.0, 1.0)[..., None]
    return np.clip(layer, 0, 255).astype(np.uint8), alpha.astype(np.float32)
