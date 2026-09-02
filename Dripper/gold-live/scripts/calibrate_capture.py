"""Calibrate the comment-panel crop for this device.

Run once per machine, with LIVE Studio open and a few comments visible. Writes
the crop region into the device config so the capture adapter knows where to
look.

    python -m scripts.calibrate_capture --session SESSION_001
    python -m scripts.calibrate_capture --session SESSION_001 --verify

Two modes:

  interactive  drag a box around the comment panel in a preview window
  manual       pass --crop X,Y,W,H if you already know the coordinates

--verify re-runs OCR on the saved region and prints what it reads, which is the
only honest way to know the calibration is right. A crop that is ten pixels too
narrow clips the author name off every row and still "works".

Re-run this after any LIVE Studio update that moves the panel. The adapter's
health check will tell you when that has happened -- it reports DEGRADED after
a few minutes with no comments read.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from platform_.adapters.ocr import build_ocr
from platform_.adapters.screen import FrameSource, parse_row
from shared.contracts import CaptureCalibration

from shared.paths import data_path

ROOT = Path(__file__).resolve().parent.parent
DEVICES = data_path("configs", "devices.json", create_parent=False)


def load_devices() -> dict:
    if DEVICES.exists():
        return json.loads(DEVICES.read_text(encoding="utf-8"))
    return {}


def save_devices(data: dict) -> None:
    DEVICES.parent.mkdir(parents=True, exist_ok=True)
    DEVICES.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")


def pick_region_interactive() -> tuple[int, int, int, int]:
    try:
        import cv2
        import mss
        import numpy as np
    except ImportError:
        sys.exit(
            "Interactive mode needs opencv and mss:\n"
            "    pip install opencv-python mss\n"
            "Or pass the region directly: --crop X,Y,W,H"
        )

    with mss.mss() as sct:
        shot = np.array(sct.grab(sct.monitors[0]))[:, :, :3]

    print("\n  Drag a box around the COMMENT PANEL only.")
    print("  Include the author names. Exclude the video and any overlays.")
    print("  Press ENTER to confirm, C to cancel.\n")

    scale = min(1.0, 1400 / shot.shape[1])
    preview = cv2.resize(shot, None, fx=scale, fy=scale) if scale < 1.0 else shot
    box = cv2.selectROI("Select the comment panel", preview, showCrosshair=False)
    cv2.destroyAllWindows()

    x, y, w, h = (int(v / scale) for v in box)
    if w <= 0 or h <= 0:
        sys.exit("Cancelled - no region selected.")
    return x, y, w, h


def verify(cal: CaptureCalibration, engine: str) -> int:
    ocr = build_ocr(engine)
    ocr.warmup()
    source = FrameSource(cal)
    try:
        image = source.grab()
    finally:
        source.close()

    lines = ocr.read(image)
    print(f"\n  Region {cal.crop_w}x{cal.crop_h} at ({cal.crop_x}, {cal.crop_y})")
    print(f"  OCR returned {len(lines)} rows\n")

    parsed = 0
    for line in lines:
        ok = parse_row(line)
        flag = "OK  " if ok else "skip"
        if ok:
            parsed += 1
            author, text = ok
            print(f"    {flag} [{line.confidence:.2f}] {author} | {text[:52]}")
        else:
            print(f"    {flag} [{line.confidence:.2f}] {line.text[:64]}")

    print()
    if parsed == 0:
        print("  FAIL: no rows parsed as comments.")
        print("  Either the crop is wrong, or the panel is empty right now.")
        print("  Post a test comment on the stream and run --verify again.")
        return 1
    if parsed < len(lines) / 2:
        print(f"  WARNING: only {parsed}/{len(lines)} rows parsed as comments.")
        print("  The crop may be including overlays or clipping author names.")
        return 0
    print(f"  PASS: {parsed}/{len(lines)} rows parsed cleanly.")
    return 0


def main() -> None:
    ap = argparse.ArgumentParser(description="Calibrate the comment-panel crop")
    ap.add_argument("--session", required=True, help="e.g. SESSION_001")
    ap.add_argument("--device", help="device id (defaults to DEVICE_<session number>)")
    ap.add_argument("--crop", help="X,Y,W,H instead of selecting interactively")
    ap.add_argument("--monitor", type=int, default=0)
    ap.add_argument("--fps", type=int, default=5)
    ap.add_argument("--min-confidence", type=float, default=0.55)
    ap.add_argument("--engine", default="paddle")
    ap.add_argument("--verify", action="store_true",
                    help="re-run OCR on the saved region and show what it reads")
    args = ap.parse_args()

    devices = load_devices()
    device_id = args.device or f"DEVICE_{args.session.split('_')[-1]}"

    if args.verify:
        existing = devices.get(device_id, {}).get("capture")
        if not existing:
            sys.exit(f"No saved calibration for {device_id}. Run without --verify first.")
        raise SystemExit(verify(CaptureCalibration(**existing), args.engine))

    if args.crop:
        try:
            x, y, w, h = (int(v) for v in args.crop.split(","))
        except ValueError:
            sys.exit("--crop must be X,Y,W,H")
    else:
        x, y, w, h = pick_region_interactive()

    cal = CaptureCalibration(
        monitor=args.monitor, crop_x=x, crop_y=y, crop_w=w, crop_h=h,
        fps=args.fps, min_ocr_confidence=args.min_confidence,
    )
    devices.setdefault(device_id, {})["bound_session"] = args.session
    devices[device_id]["capture"] = cal.model_dump()
    save_devices(devices)

    print(f"\n  Saved {device_id} -> {args.session}")
    print(f"  Region: {w}x{h} at ({x}, {y}) on monitor {args.monitor}")
    print(f"  Written to {DEVICES}")
    print("\n  Verify it actually reads comments:")
    print(f"    python -m scripts.calibrate_capture --session {args.session} --verify\n")


if __name__ == "__main__":
    main()
