"""Regression tests for the properties that make the avatar look alive.

These lock in the specific bugs already found, so they cannot come back
silently. They are fast - the engine has no I/O and simulates faster than real
time - so they run on every change.
"""

from __future__ import annotations

import statistics

import pytest

from presenter.behavior import PROFILES, BehaviorEngine, BehaviorState
from presenter.behavior.curves import blink_profile, min_jerk

VOLUNTARY = {
    "gaze_shift", "gaze_left", "gaze_right", "gaze_down", "gaze_return",
    "head_yaw", "head_pitch", "head_roll", "expression", "posture_shift",
}


def run(minutes=10.0, profile="PRESENTER_CALM", seed=0, fps=30.0,
        state=BehaviorState.IDLE_ATTENTIVE):
    engine = BehaviorEngine(profile=profile, state=state, seed=seed)
    dt = 1.0 / fps
    for _ in range(int(minutes * 60.0 * fps)):
        engine.update(dt)
    return engine


def intervals(events, kinds):
    times = [e.time for e in events if e.kind in kinds]
    return [b - a for a, b in zip(times, times[1:])]


def cv(values):
    mean = statistics.fmean(values)
    return statistics.pstdev(values) / mean if mean else 0.0


# -- curves -----------------------------------------------------------------
def test_min_jerk_endpoints_are_stationary():
    assert min_jerk(0.0) == pytest.approx(0.0)
    assert min_jerk(1.0) == pytest.approx(1.0)
    # Near-zero velocity at both ends is what removes the mechanical snap.
    assert min_jerk(0.02) < 0.001
    assert min_jerk(0.98) > 0.999


def test_blink_closes_faster_than_it_opens():
    """The asymmetry is the single most recognisable feature of a real blink."""
    closure = [blink_profile(i / 200.0) for i in range(201)]
    peak = max(range(len(closure)), key=lambda i: closure[i])
    # Peak closure well before the midpoint means the closing phase is shorter.
    assert peak / 200.0 < 0.45
    assert closure[0] == pytest.approx(0.0, abs=1e-6)
    assert closure[-1] == pytest.approx(0.0, abs=1e-6)


# -- timing distributions ---------------------------------------------------
def test_blink_intervals_are_not_fixed():
    """Regression: the brief forbids fixed-interval blinking outright."""
    engine = run()
    iv = intervals(engine.peek_events(),
                   {"blink", "blink_partial", "double_blink_second"})
    assert len(iv) > 50
    assert cv(iv) > 0.35, "blink intervals look metronomic"
    assert max(iv) / min(iv) > 5.0, "blink interval range is too narrow"


def test_head_move_rate_matches_profile():
    """Regression for the suppression-baked-into-intervals bug.

    The motion budget was folded into the sampled interval, which stretched the
    observed median to ~3x the profile base and dropped the rate to 1.87/min.
    """
    engine = run(minutes=20.0)
    rate = engine.stats.head_moves / 20.0
    assert 2.0 <= rate <= 20.0, f"head move rate {rate:.2f}/min off target"


def test_frame_rate_invariance():
    """Behaviour must be identical at 25 and 60 FPS - it is time-driven."""
    slow = run(minutes=15.0, fps=25.0, seed=11)
    fast = run(minutes=15.0, fps=60.0, seed=11)
    for attr in ("blinks", "breaths"):
        a = getattr(slow.stats, attr) / 15.0
        b = getattr(fast.stats, attr) / 15.0
        assert abs(a - b) / max(a, b) < 0.15, f"{attr} varies with frame rate"


# -- the core requirement ---------------------------------------------------
def test_genuine_stillness_occurs():
    """Regression: an earlier tuning passed a weak bar while visibly fidgeting."""
    engine = run(minutes=20.0)
    times = [e.time for e in engine.peek_events() if e.kind in VOLUNTARY]
    gaps = [b - a for a, b in zip(times, times[1:])]
    assert statistics.median(gaps) >= 3.0, "avatar fidgets - median gap too short"
    assert sum(1 for g in gaps if g >= 3.0) / len(gaps) >= 0.50
    assert max(gaps) >= 10.0, "no genuinely still stretch occurs"
    assert len(times) / 20.0 <= 17.0, "too many voluntary movements per minute"


def test_involuntary_motion_never_stops():
    """Still must not mean frozen: eyes and breath run even during quiet."""
    engine = BehaviorEngine(profile="PRESENTER_FOCUSED", seed=3)
    gaze_positions, scales = [], []
    for _ in range(600):  # 20 s
        pose = engine.update(1.0 / 30.0)
        gaze_positions.append((pose.gaze_x, pose.gaze_y))
        scales.append(pose.scale)
    assert len(set(gaze_positions)) > 500, "gaze is frozen"
    assert max(scales) - min(scales) > 1e-4, "breathing is not moving the frame"


def test_face_is_never_perfectly_symmetric_during_a_blink():
    """Regression: perfect symmetry is listed as a bug in the brief."""
    engine = BehaviorEngine(seed=17)
    asymmetric = False
    for _ in range(30 * 60 * 2):  # 2 minutes
        pose = engine.update(1.0 / 30.0)
        if 0.05 < pose.eye_open_l < 0.95 and pose.eye_open_l != pose.eye_open_r:
            asymmetric = True
            break
    assert asymmetric, "both eyelids move identically"


# -- states and profiles ----------------------------------------------------
def test_focus_suppresses_and_speech_raises_blink_rate():
    """Matches the reported halving under focus / doubling in conversation."""
    idle = run(minutes=15.0, seed=5).stats.blinks / 15.0
    focused = run(minutes=15.0, seed=5, state=BehaviorState.FOCUSED).stats.blinks / 15.0
    speaking = run(minutes=15.0, seed=5, state=BehaviorState.SPEAKING).stats.blinks / 15.0
    assert focused < idle < speaking


@pytest.mark.parametrize("profile", sorted(PROFILES))
def test_every_profile_stays_within_anatomical_limits(profile):
    engine = BehaviorEngine(profile=profile, seed=2)
    p = PROFILES[profile]
    for _ in range(30 * 60 * 5):  # 5 minutes
        pose = engine.update(1.0 / 30.0)
        assert abs(pose.yaw) <= p.head_max_yaw * 1.35
        assert abs(pose.pitch) <= p.head_max_pitch * 1.35
        assert abs(pose.roll) <= p.head_max_roll * 1.35
        assert 0.0 <= pose.eye_open_l <= 1.0
        assert 0.0 <= pose.eye_open_r <= 1.0
        assert abs(pose.gaze_x) <= 0.56
        assert abs(pose.gaze_y) <= 0.46


def test_seeds_diverge():
    a = run(minutes=10.0, seed=1)
    b = run(minutes=10.0, seed=2)
    ka = [e.kind for e in a.peek_events()]
    kb = [e.kind for e in b.peek_events()]
    assert ka[:40] != kb[:40], "different seeds produce identical behaviour"


def test_long_stall_does_not_teleport_the_avatar():
    """A GPU hiccup must not integrate as one enormous step."""
    engine = BehaviorEngine(seed=9)
    for _ in range(300):
        engine.update(1.0 / 30.0)
    before = engine.update(1.0 / 30.0)
    after = engine.update(5.0)  # a 5-second stall
    assert abs(after.yaw - before.yaw) < 3.0
    assert abs(after.gaze_x - before.gaze_x) < 0.5


# -- blink visibility -------------------------------------------------------
def sample_blink(fps: float, seed: int = 4) -> list[float]:
    """Return eyelid closure sampled at one blink's rendered frames."""
    engine = BehaviorEngine(seed=seed)
    dt = 1.0 / fps
    for _ in range(int(3 * fps)):      # let the frame-interval estimate settle
        engine.update(dt)
    frames: list[float] = []
    capturing = False
    for _ in range(int(120 * fps)):
        pose = engine.update(dt)
        closed = 1.0 - min(pose.eye_open_l, pose.eye_open_r)
        if closed > 0.01:
            capturing = True
            frames.append(closed)
        elif capturing:
            if len(frames) >= 2:
                return frames
            frames, capturing = [], False
    return frames


@pytest.mark.parametrize("fps", [12.0, 15.0, 24.0, 30.0, 60.0])
def test_blink_spans_enough_frames_to_read_as_motion(fps):
    """Regression: at low frame rates a blink was a single-frame flash.

    Frame-rate-independent *timing* is not sufficient. A 145 ms blink sampled
    at 13 FPS produced closure 0.00 -> 0.88 -> 0.00: the lid appeared to
    teleport shut, which is precisely what reads as synthetic. The blink now
    stretches so it is sampled across several frames at any realistic rate.
    """
    frames = sample_blink(fps)
    assert len(frames) >= 4, (
        f"blink at {fps} FPS spans only {len(frames)} frames ({frames}) - "
        "it will read as a flash, not a movement"
    )
    # And it must actually pass through intermediate closure, not jump.
    assert any(0.15 < f < 0.85 for f in frames), (
        f"blink at {fps} FPS has no partially-closed frame: {frames}"
    )


def test_blink_stays_physiological_at_every_frame_rate():
    """The low-frame-rate stretch must not push blinks outside human range."""
    for fps in (10.0, 13.0, 30.0, 60.0):
        engine = BehaviorEngine(seed=8)
        dt = 1.0 / fps
        for _ in range(int(3 * fps)):
            engine.update(dt)
        durations = []
        start = None
        for _ in range(int(180 * fps)):
            pose = engine.update(dt)
            closed = 1.0 - min(pose.eye_open_l, pose.eye_open_r)
            if closed > 0.01 and start is None:
                start = engine.now
            elif closed <= 0.01 and start is not None:
                durations.append(engine.now - start)
                start = None
        assert durations, f"no blinks at {fps} FPS"
        longest = max(durations)
        assert longest <= 0.45, (
            f"blink of {longest * 1000:.0f} ms at {fps} FPS exceeds the "
            "100-400 ms physiological range"
        )


def test_blink_moves_the_brow():
    """A blink that moves only the eyelid reads as a shutter, not a person."""
    engine = BehaviorEngine(seed=21)
    coupled = False
    for _ in range(30 * 90):
        pose = engine.update(1.0 / 30.0)
        closure = 1.0 - min(pose.eye_open_l, pose.eye_open_r)
        if closure > 0.5:
            if pose.brow_l < -0.01 and pose.brow_r < -0.01:
                coupled = True
                break
    assert coupled, "brow does not follow the eyelid during a blink"


# -- environment ------------------------------------------------------------
def test_streaming_room_is_deterministic_and_plausible():
    """The room must be identical every run: it is the background, and the
    brief rules out any background that changes between frames."""
    from presenter.render.environment import render_streaming_room

    a = render_streaming_room(320, 180, seed=7)
    b = render_streaming_room(320, 180, seed=7)
    assert (a == b).all(), "room is not deterministic for a fixed seed"

    c = render_streaming_room(320, 180, seed=8)
    assert not (a == c).all(), "seed does not change the room"

    assert a.shape == (180, 320, 3) and a.dtype.name == "uint8"
    # Dim enough that a lit face separates from it, not crushed to black.
    assert 15 < a.mean() < 120, f"room mean luminance {a.mean():.1f} implausible"


def test_desk_foreground_covers_only_the_bottom():
    """The desk occludes the presenter's lower edge; it must not creep up
    over the face."""
    from presenter.render.environment import render_desk_foreground

    _, alpha = render_desk_foreground(320, 180)
    rows = alpha[:, 0, 0]
    covered = rows > 0.5
    assert covered.any(), "desk covers nothing"
    assert not covered[: int(180 * 0.6)].any(), "desk reaches into the face area"
    assert covered[-1], "desk does not reach the bottom of frame"


def test_bokeh_is_clipped_tangentially_off_axis():
    """Optical vignetting turns edge-of-frame highlights into cat's eyes, and
    the long axis is *tangential*. Getting the orientation backwards is the
    sort of thing that looks fine in a thumbnail and wrong at full size."""
    from presenter.render.environment import _radial_sprite

    round_ = _radial_sprite(64)
    clipped = _radial_sprite(64, cats_eye=0.42, angle=0.0)   # offset along +x

    rm = round_ > 0.1
    assert abs(int(rm.any(0).sum()) - int(rm.any(1).sum())) <= 1, (
        "an unvignetted highlight must be circular"
    )

    cm = clipped > 0.1
    across = int(cm.any(0).sum())     # extent along the radial direction
    along = int(cm.any(1).sum())      # extent tangential to it
    assert along > across * 1.2, (
        f"cat's eye is not elongated tangentially ({along} vs {across})"
    )
    assert clipped.sum() < round_.sum(), "clipping did not remove any light"


def test_key_luminance_reads_the_lit_skin_not_the_beard():
    """A landmark box is half skin and half hair, beard and shadow. The median
    lands in the shadow, and a background fitted to it comes out most of a stop
    too dark - worst on exactly the faces where it matters."""
    import numpy as np

    from presenter.render.environment import key_luminance, luminance

    # Deliberately shadow-majority, which is what a bearded, dark-haired
    # subject in low key actually looks like inside a landmark box.
    face = np.zeros((100, 100, 3), np.uint8)
    face[:40] = 160          # lit skin
    face[40:] = 30           # beard, hair, shadow side
    median = float(np.median(luminance(face)))
    assert median < 100, "fixture is not shadow-majority"
    assert key_luminance(face) > 150, (
        "key_luminance is being dragged into the shadow like a median would be"
    )


def test_background_sits_one_to_two_stops_under_the_face():
    """The one measured property of the background. If the room and the face
    are the same brightness the image is flat no matter how good the key is."""
    from presenter.render.environment import fit_exposure, render_streaming_room

    room = render_streaming_room(320, 180, seed=7)
    for face_luma in (80.0, 120.0, 160.0, 200.0):
        _, scale, stops = fit_exposure(room, face_luma)
        assert 1.0 <= stops <= 2.0, (
            f"face={face_luma}: background is {stops:.2f} stops down, "
            f"outside the 1-2 stop band (scale {scale:.2f})"
        )


def test_light_wrap_stays_inside_the_silhouette():
    """The wrap softens the subject's edge. Spilling *outward* would put a
    halo on the background, which is the failure mode it exists to avoid."""
    import numpy as np

    from presenter.render.environment import light_wrap

    plate = np.full((120, 120, 3), 200, np.uint8)
    alpha = np.zeros((120, 120), np.float32)
    alpha[:, 60:] = 1.0                      # hard edge at x = 60

    wrap = light_wrap(plate, alpha, width=0.15)
    assert wrap[:, :60].max() == 0.0, "wrap leaked outside the subject"
    assert wrap[:, 60:72].max() > 1.0, "no wrap just inside the edge"
    assert wrap[:, 90:].max() == 0.0, "wrap reaches deep into the subject"


# -- wardrobe ---------------------------------------------------------------
def test_source_state_is_complete():
    """Switching outfits swaps one source portrait for another, and it has to
    swap *everything* that belongs to a source.

    A missing entry does not crash - it leaves one stale array behind, so the
    new face renders against the old silhouette's mask or the old background.
    That is a far worse failure than an exception, so the list is checked
    against the code rather than maintained by hand and hoped over."""
    import ast
    from pathlib import Path

    from presenter.render.liveportrait import _SOURCE_STATE

    src = Path("src/presenter/render/liveportrait.py")
    if not src.exists():                      # running from another cwd
        src = Path(__file__).resolve().parents[1] / src
    tree = ast.parse(src.read_text(encoding="utf-8"))
    cls = next(n for n in tree.body
               if isinstance(n, ast.ClassDef) and n.name == "LivePortraitRenderer")

    per_source = {"_prepare_source", "_prepare_compositing", "_finish_blend_setup"}
    assigned = set()
    for fn in cls.body:
        if not (isinstance(fn, ast.FunctionDef) and fn.name in per_source):
            continue
        for node in ast.walk(fn):
            if not isinstance(node, (ast.Assign, ast.AugAssign, ast.AnnAssign)):
                continue
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            flat = []
            for t in targets:
                flat.extend(t.elts if isinstance(t, (ast.Tuple, ast.List)) else [t])
            for t in flat:
                if (isinstance(t, ast.Attribute)
                        and isinstance(t.value, ast.Name) and t.value.id == "self"):
                    assigned.add(t.attr)

    missing = assigned - set(_SOURCE_STATE)
    assert not missing, (
        f"_SOURCE_STATE does not cover {sorted(missing)} - an outfit switch "
        f"would leave these stale from the previous portrait"
    )


def test_wardrobe_resolves_outfits_to_portraits():
    """The base portrait is the no-edit combination, and it is not duplicated
    on disk to make the naming scheme uniform."""
    from presenter.render.wardrobe import Wardrobe

    w = Wardrobe.load()
    assert w.clothing and w.headwear, "wardrobe declares nothing"

    plain_c = next(k for k, i in w.clothing.items() if not i.edits)
    plain_h = next(k for k, i in w.headwear.items() if not i.edits)
    assert w.path(plain_c, plain_h) == w.base

    dressed = next(k for k, i in w.clothing.items() if i.edits)
    assert w.path(dressed, plain_h).name == f"{dressed}__{plain_h}.png"
    assert w.path(dressed, plain_h).parent == w.directory

    try:
        w.path("no_such_shirt", plain_h)
    except KeyError as exc:
        assert "no_such_shirt" in str(exc)
    else:
        raise AssertionError("unknown clothing key was accepted")


def test_wardrobe_prompts_fit_the_clip_token_limit():
    """CLIP truncates at 77 tokens silently. An earlier revision lost the whole
    photographic-direction block off the end of every prompt and produced
    costume-shop results with no error anywhere."""
    from presenter.render.wardrobe import Wardrobe

    w = Wardrobe.load()
    # Word count is a deliberate stand-in: it needs no tokenizer download to
    # run in CI, and CLIP tokens outnumber words, so passing here is necessary
    # rather than sufficient. tools/generate_wardrobe.py does the real check
    # with the actual tokenizer before it generates anything.
    for section in (w.clothing, w.headwear):
        for item in section.values():
            if not item.edits:
                continue
            words = len(w.prompt_for(item).split())
            assert words <= 60, f"{item.key} prompt is {words} words, too long"
    for headwear in (False, True):
        words = len(w.negative_for(headwear=headwear).split())
        assert words <= 60, f"negative prompt is {words} words, too long"


# -- dropdown menus ---------------------------------------------------------
def _bar():
    from presenter.ui import DropdownBar, Menu, Option

    return DropdownBar([
        Menu("clothing", "Clothing",
             [Option("tee", "Grey tee"), Option("thobe", "Thobe", enabled=False)],
             "tee"),
        Menu("headwear", "Head attire",
             [Option("none", "Bare head"), Option("ghutra", "Ghutra")],
             "none"),
    ], origin=(100, 10))


def click(bar, x, y):
    import cv2

    bar.on_mouse(cv2.EVENT_LBUTTONDOWN, x, y)


def test_dropdown_opens_selects_and_dismisses():
    bar = _bar()
    clothing, headwear = bar.menus

    click(bar, clothing.x + 20, clothing.y + 10)
    assert clothing.open and not headwear.open

    # Opening the other menu closes this one; two open lists would overlap.
    click(bar, headwear.x + 20, headwear.y + 10)
    assert headwear.open and not clothing.open

    x, y, w, h = headwear.row_rect(1)
    click(bar, x + 20, y + h // 2)
    assert bar.take_selection() == ("headwear", "ghutra")
    assert bar.take_selection() is None, "a selection was reported twice"
    assert not headwear.open

    click(bar, clothing.x + 20, clothing.y + 10)
    click(bar, 5, 700)                      # anywhere else
    assert not bar.is_open, "clicking away did not dismiss the menu"


def test_dropdown_ignores_options_with_no_portrait():
    """A greyed row means that combination was never generated. Selecting it
    would hand the renderer a path that does not exist."""
    bar = _bar()
    clothing = bar.menus[0]
    click(bar, clothing.x + 20, clothing.y + 10)

    x, y, w, h = clothing.row_rect(1)        # the disabled entry
    click(bar, x + 20, y + h // 2)
    assert bar.take_selection() is None
    assert clothing.open, "a dead click closed the menu"

    assert bar.cycle("clothing") == "tee", "cycle offered a disabled option"


def test_replacing_menus_keeps_layout_and_open_state():
    """Rebuilt menus that skip layout draw at the origin of the frame, on top
    of the presenter."""
    from presenter.ui import Menu, Option

    bar = _bar()
    bar.menus[0].open = True
    before = [(m.x, m.y) for m in bar.menus]

    bar.set_menus([
        Menu("clothing", "Clothing", [Option("tee", "Grey tee")], "tee"),
        Menu("headwear", "Head attire", [Option("none", "Bare head")], "none"),
    ])
    assert [(m.x, m.y) for m in bar.menus] == before
    assert bar.menus[0].open and not bar.menus[1].open


# -- cameras ----------------------------------------------------------------
def test_every_camera_showing_his_face_is_the_same_photograph():
    """The rule that keeps it one man rather than seven.

    Generating each camera from a prompt produced a different person every time
    - prompts share a description, not a face. So any camera whose face is
    visible enough to animate must *derive* from the one master frame, and only
    cameras with no recognisable face may be separately generated.

    This is the invariant that failure would be most embarrassing to ship, and
    it is cheap to check."""
    from presenter.render.cameras import CameraRig

    rig = CameraRig.load()
    assert len(rig.cameras) == 7, "the buttons are camera 1 to camera 7"
    assert [c.index for c in rig.ordered()] == [1, 2, 3, 4, 5, 6, 7]

    derived = [c for c in rig.ordered() if c.derive]
    generated = [c for c in rig.ordered() if not c.derive]
    assert derived and generated, "a rig that is all one kind is suspect"

    for cam in derived:
        assert cam.animated, f"{cam.key} derives from the master but is a still"
        assert not cam.subject, f"{cam.key} derives; its prompt would be ignored"
        assert rig.path(cam.key) == rig.master
        assert cam.framing in ("full", "shoulders", "close")

    for cam in generated:
        assert not cam.animated, (
            f"{cam.key} is separately generated *and* animated - that is the "
            f"combination that puts a different man on screen"
        )
        assert cam.negative, (
            f"{cam.key} is a still but inherits the shared negative, which "
            f"forbids exactly the shot it is trying to make"
        )

    # Every derived camera must show a distinct framing, or two buttons do the
    # same thing.
    framings = [c.framing for c in derived]
    assert len(set(framings)) == len(framings), f"duplicate framings {framings}"


def test_camera_default_prefers_a_live_camera():
    """Opening on a still would show a frozen presenter, which is the one thing
    this project exists to avoid."""
    from presenter.render.cameras import Camera, CameraRig

    rig = CameraRig.load()
    rig.cameras = {
        "cam6": Camera("cam6", "6 Still", False, "s"),
        "cam1": Camera("cam1", "1 Hero", True, "looking into the lens"),
    }
    rig.exists = lambda key: True                      # pretend both generated
    assert rig.default() == "cam1"

    rig.cameras.pop("cam1")
    assert rig.default() == "cam6", "with no live camera, a still is fine"


def test_camera_buttons_press_and_step():
    from presenter.ui import Button, ButtonRow
    import cv2

    row = ButtonRow([
        Button("cam1", "1 Hero"),
        Button("cam2", "2 Close", enabled=False),
        Button("cam3", "3 Wide", live=False),
    ], origin=(10, 600))
    assert row.selected == "cam1"

    b = row.buttons[2]
    assert row.on_mouse(cv2.EVENT_LBUTTONDOWN, b.x + 5, b.y + 5) is True
    assert row.take_selection() == "cam3"
    assert row.take_selection() is None

    dead = row.buttons[1]
    assert row.on_mouse(cv2.EVENT_LBUTTONDOWN, dead.x + 5, dead.y + 5) is True
    assert row.take_selection() is None, "a camera with no frame was selectable"

    assert row.on_mouse(cv2.EVENT_LBUTTONDOWN, 5, 5) is False, "claimed a click "\
        "outside the row, which would stop the dropdowns ever seeing one"

    # Stepping skips the ungenerated camera rather than stalling on it.
    row.selected = "cam1"
    assert row.step(1) == "cam3"
    assert row.step(-1) == "cam3"
