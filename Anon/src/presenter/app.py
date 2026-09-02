"""Application loop: clock, renderer, debug overlay, graceful degradation.

Three things here are load-bearing rather than incidental:

**The clock is real time, not frames.** The engine is advanced by measured
elapsed seconds, so a frame that arrives late produces a correspondingly larger
step and the behaviour stays on schedule. Driving animation by frame count -
the obvious shortcut - makes every motion speed up and slow down with the
render load, which is immediately visible as breathing that changes rate when
the GPU is busy.

**One bad frame never kills the stream.** A render failure keeps the last good
frame and increments a counter. The brief is explicit that a black frame or a
crash is unacceptable, and for a system intended to run for hours unattended,
this is the difference between a glitch and a dead broadcast.

**Debug is off by default and complete when on.** Presentation mode shows only
the avatar.

Run:
    python -m presenter.app                     # schematic rig preview
    python -m presenter.app --profile PRESENTER_FOCUSED
    python -m presenter.app --debug --duration 60
    python -m presenter.app --headless --duration 30   # benchmark, no window
"""

from __future__ import annotations

import argparse
import statistics
import sys
import time
from pathlib import Path

import cv2
import numpy as np

if __package__ in (None, ""):  # allow `python src/presenter/app.py`
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from presenter.behavior import PROFILES, BehaviorEngine, BehaviorState
from presenter.render.schematic import SchematicRenderer
from presenter.types import AvatarPose

WINDOW = "AI Presenter"

# Keys 1-9 switch state, for exercising the interface Developer A will drive.
_STATE_KEYS = {
    ord("1"): BehaviorState.IDLE_ATTENTIVE,
    ord("2"): BehaviorState.IDLE_RELAXED,
    ord("3"): BehaviorState.LISTENING,
    ord("4"): BehaviorState.THINKING,
    ord("5"): BehaviorState.PRE_SPEECH,
    ord("6"): BehaviorState.SPEAKING,
    ord("7"): BehaviorState.POST_SPEECH,
    ord("8"): BehaviorState.FOCUSED,
    ord("9"): BehaviorState.READING,
}


class FrameTimer:
    """Rolling frame-time statistics.

    Reports percentiles, not just a mean: a 30 FPS average hiding a 90 ms
    p99 stutter is not a 30 FPS system, and the mean alone would conceal
    exactly the latency spikes the brief asks to be flagged.
    """

    def __init__(self, window: int = 240) -> None:
        self._samples: list[float] = []
        self._window = window
        self.worst = 0.0

    def add(self, seconds: float) -> None:
        self._samples.append(seconds)
        self.worst = max(self.worst, seconds)
        if len(self._samples) > self._window:
            self._samples.pop(0)

    @property
    def fps(self) -> float:
        if not self._samples:
            return 0.0
        mean = statistics.fmean(self._samples)
        return 1.0 / mean if mean > 0 else 0.0

    def percentile(self, p: float) -> float:
        if not self._samples:
            return 0.0
        ordered = sorted(self._samples)
        idx = min(int(p * len(ordered)), len(ordered) - 1)
        return ordered[idx]


def draw_debug(frame: np.ndarray, engine: BehaviorEngine, pose: AvatarPose,
               timer: FrameTimer, render_ms: float, failures: int,
               renderer_name: str) -> None:
    """Overlay engine state. Deliberately dense - this is a diagnostic view."""
    lines = [
        f"{renderer_name}   {frame.shape[1]}x{frame.shape[0]}",
        f"FPS {timer.fps:5.1f}   render {render_ms:5.2f}ms   "
        f"p95 {timer.percentile(0.95) * 1000:5.1f}ms   worst {timer.worst * 1000:5.1f}ms",
        f"state {pose.state}   profile {engine.profile.name}",
        f"elapsed {engine.stats.elapsed:7.1f}s   frames {engine.stats.frames}",
        "",
        f"head   yaw {pose.yaw:+6.2f}  pitch {pose.pitch:+6.2f}  roll {pose.roll:+6.2f}",
        f"gaze   x {pose.gaze_x:+6.3f}  y {pose.gaze_y:+6.3f}",
        f"lids   L {pose.eye_open_l:5.3f}  R {pose.eye_open_r:5.3f}",
        f"brow   L {pose.brow_l:+5.3f}  R {pose.brow_r:+5.3f}  "
        f"furrow {pose.brow_furrow:+5.3f}",
        f"breath phase {pose.breathing_phase:4.2f}   scale {pose.scale:6.4f}",
        f"arousal {engine.arousal:+5.2f}   motion budget {engine.motion_budget:4.2f}",
        "",
        f"blinks {engine.stats.blinks:4d} ({engine.stats.blinks_per_minute():5.1f}/min)"
        f"   next in {engine.blink.time_to_next(engine.now):4.1f}s",
        f"saccades {engine.stats.saccades:4d} "
        f"({engine.stats.saccades_per_minute():5.1f}/min)   "
        f"micro {engine.stats.microsaccades}",
        f"head moves {engine.stats.head_moves:3d} "
        f"({engine.stats.head_moves_per_minute():5.1f}/min)   "
        f"expr {engine.stats.expressions}   posture {engine.stats.posture_shifts}",
        f"breaths {engine.stats.breaths:4d}   expression: {engine.expression.active_name}",
    ]
    if failures:
        lines.append(f"RENDER FAILURES: {failures} (holding last good frame)")

    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (620, 26 + 19 * len(lines)), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.62, frame, 0.38, 0, frame)

    y = 26
    for line in lines:
        colour = (120, 210, 255) if "FAILURES" in line else (215, 225, 235)
        cv2.putText(frame, line, (14, y), cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                    colour, 1, cv2.LINE_AA)
        y += 19

    cv2.putText(frame, "1-9 state   d debug   s screenshot   q quit",
                (14, frame.shape[0] - 14), cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                (150, 160, 175), 1, cv2.LINE_AA)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--profile", default="PRESENTER_CALM", choices=sorted(PROFILES))
    ap.add_argument("--state", default="IDLE_ATTENTIVE",
                    choices=[s.value for s in BehaviorState])
    ap.add_argument("--width", type=int, default=1280)
    ap.add_argument("--height", type=int, default=720)
    ap.add_argument("--fps", type=float, default=30.0, help="target frame rate")
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--debug", action="store_true")
    ap.add_argument("--duration", type=float, default=0.0,
                    help="seconds to run, 0 = until quit")
    ap.add_argument("--headless", action="store_true",
                    help="no window; renders and benchmarks only")
    ap.add_argument("--renderer", default="schematic",
                    choices=["schematic", "liveportrait"],
                    help="schematic = behaviour-tuning rig preview; "
                         "liveportrait = photoreal")
    ap.add_argument("--source", default="assets/presenter_source.png",
                    help="source portrait for the photoreal renderer")
    ap.add_argument("--liveportrait-root", default="third_party/LivePortrait")
    ap.add_argument("--framing", default="shoulders",
                    choices=["shoulders", "close", "full"],
                    help="full = master frame used as shot (streaming room); "
                         "shoulders = head and shoulders, blurred side fill; "
                         "close = tight 16:9 crop")
    ap.add_argument("--environment", default="streaming_room",
                    choices=["streaming_room", "source"],
                    help="streaming_room = generated room behind the presenter "
                         "(needs a person matte, adds ~30s to startup); "
                         "source = keep the portrait's own background")
    ap.add_argument("--room-seed", type=int, default=7,
                    help="changes the generated room's light placement")
    ap.add_argument("--neutralize-pose", type=float, default=0.85,
                    help="0 keeps the portrait's own head angle, 1 squares it "
                         "to the lens; a little residual is left on purpose")
    ap.add_argument("--compile", action="store_true",
                    help="torch.compile the warping/generator modules; large "
                         "first-run cost, needed because the workload is "
                         "launch-bound rather than compute-bound")
    ap.add_argument("--save-frames", type=str, default="",
                    help="directory to write periodic frames for inspection")
    ap.add_argument("--frame-interval", type=float, default=5.0,
                    help="seconds between saved frames")
    args = ap.parse_args()

    engine = BehaviorEngine(profile=args.profile,
                            state=BehaviorState(args.state), seed=args.seed)

    if args.renderer == "liveportrait":
        import torch

        from presenter.render.calibration import calibration_report
        from presenter.render.liveportrait import LivePortraitRenderer

        # The workload is launch-bound, not compute-bound (measured: 14-18% GPU
        # utilisation at 30 W). cudnn.benchmark picks fixed-shape algorithms
        # once instead of re-selecting every call.
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

        print(f"[app] loading LivePortrait, source={args.source}")
        renderer = LivePortraitRenderer(
            source_image=args.source,
            liveportrait_root=args.liveportrait_root,
            output_size=(args.width, args.height),
            framing=args.framing,
            environment=args.environment,
            room_seed=args.room_seed,
            neutralize_pose=args.neutralize_pose,
        )
        if args.compile:
            print("[app] torch.compile - first frames will be slow")
            renderer.wrapper.warping_module = torch.compile(
                renderer.wrapper.warping_module)
            renderer.wrapper.spade_generator = torch.compile(
                renderer.wrapper.spade_generator)
        print(calibration_report())
    else:
        renderer = SchematicRenderer(args.width, args.height)

    timer = FrameTimer()

    save_dir = Path(args.save_frames) if args.save_frames else None
    if save_dir:
        save_dir.mkdir(parents=True, exist_ok=True)

    debug = args.debug
    failures = 0
    last_good: np.ndarray | None = None
    next_save = 0.0
    saved = 0

    if not args.headless:
        cv2.namedWindow(WINDOW, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(WINDOW, args.width, args.height)

    # Warm up before the clock starts. The first render is dramatically slower
    # than steady state - measured at 31.5 s with the LivePortrait backend,
    # because cudnn.benchmark autotunes every 3D convolution shape on its first
    # call (and torch.compile, if enabled, compiles here too). Without this the
    # first frame's cost lands in the frame-time statistics and, worse, a short
    # --duration run can exit having rendered two frames.
    warm_frames = 3 if args.renderer == "liveportrait" else 1
    print(f"[app] warming up ({warm_frames} frames)...", flush=True)
    warm_start = time.perf_counter()
    warm_pose = engine.pose if engine.stats.frames else engine.update(1.0 / 30.0)
    for _ in range(warm_frames):
        try:
            renderer.render(warm_pose)
        except Exception as exc:  # noqa: BLE001
            print(f"[app] warmup render failed: {exc}", file=sys.stderr)
            break
    print(f"[app] warmup took {time.perf_counter() - warm_start:.1f}s", flush=True)

    target_dt = 1.0 / max(args.fps, 1.0)
    previous = time.perf_counter()
    start = previous

    try:
        while True:
            now = time.perf_counter()
            dt = now - previous
            previous = now

            pose = engine.update(dt)

            render_start = time.perf_counter()
            try:
                frame = renderer.render(pose)
                last_good = frame
            except Exception as exc:  # noqa: BLE001 - a bad frame must not kill the loop
                failures += 1
                if failures <= 3:
                    print(f"[render] frame failed ({exc}); holding last good frame",
                          file=sys.stderr)
                if last_good is None:
                    continue
                frame = last_good.copy()
            render_ms = (time.perf_counter() - render_start) * 1000.0

            elapsed = now - start
            timer.add(max(dt, 1e-6))

            if save_dir and elapsed >= next_save:
                cv2.imwrite(str(save_dir / f"frame_{saved:04d}_t{elapsed:07.2f}s.png"),
                            frame)
                saved += 1
                next_save = elapsed + args.frame_interval

            if not args.headless:
                shown = frame
                if debug:
                    shown = frame.copy()
                    draw_debug(shown, engine, pose, timer, render_ms, failures,
                               renderer.info.name)
                cv2.imshow(WINDOW, shown)

                key = cv2.waitKey(1) & 0xFF
                if key in (ord("q"), 27):
                    break
                if key == ord("d"):
                    debug = not debug
                if key == ord("s"):
                    path = Path(f"screenshot_{int(time.time())}.png")
                    cv2.imwrite(str(path), frame)
                    print(f"[app] wrote {path}")
                if key in _STATE_KEYS:
                    engine.set_state(_STATE_KEYS[key])
                    print(f"[app] state -> {_STATE_KEYS[key].value}")

            if args.duration and elapsed >= args.duration:
                break

            # Sleep off the remainder of the budget rather than spinning. The
            # engine is time-based, so an imprecise sleep costs nothing.
            slack = target_dt - (time.perf_counter() - now)
            if slack > 0.001:
                time.sleep(slack)

    except KeyboardInterrupt:
        print("\n[app] interrupted")
    finally:
        renderer.close()
        if not args.headless:
            cv2.destroyAllWindows()

    total = time.perf_counter() - start
    stats = engine.stats
    print(f"\n--- session ---")
    print(f"  renderer        {renderer.info}")
    print(f"  duration        {total:.1f}s over {stats.frames} frames")
    print(f"  mean FPS        {stats.frames / max(total, 1e-6):.1f}")
    print(f"  p95 frame time  {timer.percentile(0.95) * 1000:.2f}ms")
    print(f"  worst frame     {timer.worst * 1000:.2f}ms")
    print(f"  render failures {failures}")
    print(f"  blinks          {stats.blinks} ({stats.blinks_per_minute():.1f}/min)")
    print(f"  saccades        {stats.saccades} "
          f"({stats.saccades_per_minute():.1f}/min)")
    print(f"  head moves      {stats.head_moves} "
          f"({stats.head_moves_per_minute():.1f}/min)")
    print(f"  breaths         {stats.breaths}")
    if saved:
        print(f"  frames written  {saved} -> {save_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
