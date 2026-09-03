"""The avatar's own basis, and a top-down plan of the whole rig.

## Why directions are derived here and never written down

"Behind him" is not a world axis. It is a fact about which way he faces, and
this project has already put two cameras in front of the subject by reasoning
about +Y from memory. So nothing in this file, or in anything that imports it,
states a semantic direction as a constant. `facing` is read from
config/room_geometry.yaml and turned into a basis:

    forward   the way he looks
    up        room up
    right     forward x up

and every placement is expressed as a combination of those:

    behind  = position - forward * distance
    lateral = position + right   * offset

Change `facing` in the config and every camera derived through here moves with
him, which is the property that was missing when cameras 4 and 5 were first
placed.

    python tools/camera_layout.py          # writes the floorplan, prints the basis
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]

AXES = {
    "+x": (1.0, 0.0, 0.0), "-x": (-1.0, 0.0, 0.0),
    "+y": (0.0, 1.0, 0.0), "-y": (0.0, -1.0, 0.0),
    "+z": (0.0, 0.0, 1.0), "-z": (0.0, 0.0, -1.0),
}


def load(geometry="config/room_geometry.yaml", cameras="config/cameras.yaml"):
    g = yaml.safe_load((ROOT / geometry).read_text(encoding="utf-8"))
    c = yaml.safe_load((ROOT / cameras).read_text(encoding="utf-8"))
    return g, c


class AvatarBasis:
    """Where he is and which way he faces, as vectors rather than adjectives."""

    def __init__(self, g: dict) -> None:
        human = g["human"]
        self.hip = np.array(human["hip"], float)
        self.eye_height = float(human["eye_height"])
        facing = str(human.get("facing", "+y")).strip().lower()
        if facing not in AXES:
            raise ValueError(f"human.facing must be one of {sorted(AXES)}, "
                             f"got {facing!r}")
        self.forward = np.array(AXES[facing], float)
        self.up = np.array(AXES[str(g.get("up_axis", "+z")).lower()], float) \
            if g.get("up_axis") else np.array([0.0, 0.0, 1.0])
        # Right-handed: with forward +Y and up +Z this gives +X, which is his
        # right hand. Derived, so a change of facing rotates it correctly.
        self.right = np.cross(self.forward, self.up)
        n = np.linalg.norm(self.right)
        if n < 1e-9:
            raise ValueError("facing is parallel to up; no lateral axis exists")
        self.right /= n
        # Eye is a height above the floor, not above the hip.
        self.eye = np.array([self.hip[0], self.hip[1], self.eye_height])

    def place(self, behind=0.0, lateral=0.0, height=None, about=None) -> np.ndarray:
        """A point relative to him. Positive `behind` is away from his gaze."""
        base = self.eye if about is None else np.asarray(about, float)
        p = base - self.forward * behind + self.right * lateral
        if height is not None:
            p = np.array([p[0], p[1], float(height)])
        return p

    def describe(self) -> str:
        return (f"avatar_position (eye) {self.eye.round(3).tolist()}\n"
                f"avatar_forward        {self.forward.round(3).tolist()}\n"
                f"avatar_up             {self.up.round(3).tolist()}\n"
                f"avatar_right          {self.right.round(3).tolist()}")


# ---------------------------------------------------------------------------
# Floorplan


def _fov_deg(focal_mm: float, sensor_mm: float = 36.0) -> float:
    return math.degrees(2.0 * math.atan(sensor_mm / (2.0 * focal_mm)))


def floorplan(out=None, geometry="config/room_geometry.yaml",
              cameras="config/cameras.yaml", scale=210):
    """Top-down plan: room, furniture, avatar heading, and every camera cone.

    Drawn so the slat wall (Y=0) is at the top and the front of the room at the
    bottom, which is how the cameras 1-3 see it. The avatar's forward arrow is
    drawn from the derived basis, not from an assumption, so a camera cone that
    does not contain what its name claims is visible at a glance.
    """
    import cv2

    g, c = load(geometry, cameras)
    basis = AvatarBasis(g)
    room = g["room"]
    W, D = float(room["width"]), float(room["depth"])
    m = 70
    img = np.full((int(D * scale) + 2 * m, int(W * scale) + 2 * m, 3), 22, np.uint8)

    def px(x, y):
        """World XY -> pixels. X right, Y down (slat wall on top)."""
        return (int(m + (x + W / 2) * scale), int(m + y * scale))

    def rect(cx, cy, sx, sy, colour, thickness=2, label=None):
        a = px(cx - sx / 2, cy - sy / 2)
        b = px(cx + sx / 2, cy + sy / 2)
        cv2.rectangle(img, a, b, colour, thickness)
        if label:
            cv2.putText(img, label, (a[0] + 5, a[1] - 6),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.42, colour, 1, cv2.LINE_AA)

    # Room
    cv2.rectangle(img, px(-W / 2, 0.0), px(W / 2, D), (95, 95, 105), 2)
    cv2.putText(img, "SLAT WALL  Y=0   (display wall, monitors face +Y)",
                (m + 6, m - 42), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                (150, 190, 235), 1, cv2.LINE_AA)
    cv2.putText(img, f"FRONT OF ROOM  Y={D}", (m + 6, m + int(D * scale) + 28),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150, 150, 160), 1, cv2.LINE_AA)

    # Monitors, with the direction they face - the whole point of the diagram.
    for mon in g.get("monitors", []):
        cx, cy, _ = mon["centre"]
        sx, sy, _ = mon["size"]
        colour = (110, 220, 140) if mon.get("powered") else (90, 120, 100)
        rect(cx, cy, sx, max(sy, 0.06), colour, 2, mon["id"].replace("monitor_", "mon "))
        nrm = np.array(AXES[str(mon.get("normal", "+y")).lower()], float)
        tip = px(cx + nrm[0] * 0.34, cy + nrm[1] * 0.34)
        cv2.arrowedLine(img, px(cx, cy), tip, colour, 2, tipLength=0.35)

    d = g["desk"]
    rect(d["centre"][0], d["centre"][1], d["top_size"][0], d["top_size"][1],
         (150, 175, 210), 2, "desk")
    ch = g["chair"]
    rect(ch["base_centre"][0], ch["base_centre"][1], ch["seat_size"][0],
         ch["seat_size"][1], (170, 150, 120), 2, "chair")
    for sp in g.get("speakers", []):
        rect(sp["centre"][0], sp["centre"][1], sp["size"][0], sp["size"][1],
             (120, 120, 150), 1, "spk")
    for lm in g.get("landmarks", []):
        rect(lm["centre"][0], lm["centre"][1], lm["size"][0], lm["size"][1],
             (110, 110, 125), 1, lm["id"])

    mic = g.get("microphone")
    if mic:
        pts = [px(p[0], p[1]) for p in mic["joints"]]
        for a, b in zip(pts, pts[1:]):
            cv2.line(img, a, b, (210, 170, 90), 2)
        cv2.circle(img, px(*mic["capsule_centre"][:2]), 5, (210, 170, 90), -1)
        cv2.putText(img, "mic", (px(*mic["capsule_centre"][:2])[0] + 8,
                                 px(*mic["capsule_centre"][:2])[1]),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, (210, 170, 90), 1, cv2.LINE_AA)

    # The avatar and, crucially, the way he actually faces.
    hp = px(basis.eye[0], basis.eye[1])
    cv2.circle(img, hp, 13, (235, 240, 250), -1)
    fwd = basis.eye + basis.forward * 0.85
    cv2.arrowedLine(img, hp, px(fwd[0], fwd[1]), (90, 200, 255), 3, tipLength=0.3)
    cv2.putText(img, "AVATAR forward", (hp[0] + 16, hp[1] + 4),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (90, 200, 255), 1, cv2.LINE_AA)
    rt = basis.eye + basis.right * 0.5
    cv2.arrowedLine(img, hp, px(rt[0], rt[1]), (120, 160, 200), 1, tipLength=0.3)
    cv2.putText(img, "right", (px(rt[0], rt[1])[0] + 4, px(rt[0], rt[1])[1] + 4),
                cv2.FONT_HERSHEY_SIMPLEX, 0.38, (120, 160, 200), 1, cv2.LINE_AA)

    # Cameras: position, aim ray, and horizontal field of view.
    sensor = c["defaults"]["sensor_width_mm"]
    for spec in c["cameras"]:
        p = np.array(spec["position"], float)
        t = np.array(spec["look_at"], float)
        live = spec.get("enabled", True)
        colour = (90, 200, 120) if live else (95, 130, 235)
        a = px(p[0], p[1])
        half = math.radians(_fov_deg(spec["focal_length_mm"], sensor) / 2.0)
        v = t[:2] - p[:2]
        if np.linalg.norm(v) > 1e-6:
            v = v / np.linalg.norm(v)
            reach = 3.2
            for s in (-1, 1):
                ca, sa = math.cos(s * half), math.sin(s * half)
                e = p[:2] + np.array([v[0] * ca - v[1] * sa,
                                      v[0] * sa + v[1] * ca]) * reach
                cv2.line(img, a, px(e[0], e[1]), colour, 1, cv2.LINE_AA)
            cv2.arrowedLine(img, a, px(*(p[:2] + v * 0.6)), colour, 2, tipLength=0.3)
        cv2.circle(img, a, 8, colour, -1)
        cv2.putText(img, f"{spec['id'].upper()} {spec['focal_length_mm']:.0f}mm",
                    (a[0] + 11, a[1] - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.46,
                    colour, 1, cv2.LINE_AA)
        cv2.putText(img, f"z={p[2]:.2f}", (a[0] + 11, a[1] + 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.38, colour, 1, cv2.LINE_AA)

    cv2.putText(img, "green = production-enabled    blue = production-blocked",
                (m + 6, img.shape[0] - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.46,
                (170, 170, 180), 1, cv2.LINE_AA)

    dest = Path(out) if out else ROOT / "renders/camera_preview/camera_floorplan_debug.png"
    dest.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(dest), img)
    return dest


def main() -> int:
    g, _ = load()
    basis = AvatarBasis(g)
    print(basis.describe())
    print()
    # Stated as measurements, because this is the fact that invalidated the
    # first cam4/cam5 placement.
    mons = [m["centre"][1] for m in g["monitors"]]
    print(f"monitor wall at Y={min(mons):.3f}..{max(mons):.3f}  "
          f"(normal +Y, i.e. facing the front cameras)")
    print(f"desk centre     Y={g['desk']['centre'][1]:.3f}")
    print(f"avatar          Y={basis.eye[1]:.3f}, forward {basis.forward.tolist()}")
    behind = basis.place(behind=1.0)
    print(f"1.0 m BEHIND him is {behind.round(3).tolist()} "
          f"-> that is toward the monitor wall")
    dest = floorplan()
    print(f"\nfloorplan -> {dest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
