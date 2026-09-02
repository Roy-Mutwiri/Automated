"""Camera selection, independent of whichever renderer draws the human.

The camera rig used to be created inside the LivePortrait branch of the
application. That coupling was wrong in a way that cost real time: with the
default renderer no rig existed, so no buttons were built, the bracket keys
were inert, and camera switching looked broken when in fact it had never been
turned on.

Cameras are world configuration. Where a camera stands, what lens it carries
and where it aims are facts about the room, not features of a face renderer.
So the manager loads whenever the application starts, and any renderer may ask
it what the current camera is.

Two rigs exist and they are not the same kind of thing:

* **`config/cameras.yaml`** - seven *physical* cameras in the canonical 3D
  world. cam2 and cam3 are genuinely to his left and right.

* **`config/cameras_2d_legacy.yaml`** - the older 2D rig, where cam1, cam2 and
  cam3 are three *crops of one photograph*. Those are punch-ins, not
  viewpoints, and this module refuses to describe them as left and right. A
  label that promises a camera move the pixels do not contain is how the whole
  multicam illusion got debugged from the wrong end once already.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[3]

# What each physical camera is for. The label describes the *viewpoint*.
PHYSICAL_INTENT = {
    "cam1": "HERO FRONT",
    "cam2": "LEFT 3/4",
    "cam3": "RIGHT 3/4",
    "cam4": "OVER SHOULDER",
    "cam5": "REAR WIDE",
    "cam6": "HIGH DIAGONAL",
    "cam7": "ROOM WIDE",
}

# The legacy rig's first three cameras are one photograph at three framings.
# Naming them for their crop is the only honest option available.
LEGACY_INTENT = {
    "cam1": "LEGACY 2D MASTER",
    "cam2": "LEGACY 2D PUNCH-IN",
    "cam3": "LEGACY 2D CLOSE-UP",
}


@dataclass
class CameraView:
    """One camera, as the UI and the debug overlay need to see it."""

    key: str
    label: str
    intent: str
    position: tuple[float, float, float]
    look_at: tuple[float, float, float]
    focal_mm: float
    enabled: bool = True
    preview: Path | None = None
    physical: bool = True

    @property
    def index(self) -> int:
        digits = "".join(c for c in self.key if c.isdigit())
        return int(digits) if digits else 0

    @property
    def rotation_deg(self) -> tuple[float, float, float]:
        """Aim as yaw/pitch in degrees - readable, unlike a Blender Euler.

        Derived from position and look_at rather than stored, so it cannot
        disagree with the transform the renderer actually uses.
        """
        dx = self.look_at[0] - self.position[0]
        dy = self.look_at[1] - self.position[1]
        dz = self.look_at[2] - self.position[2]
        yaw = math.degrees(math.atan2(dx, dy))
        pitch = math.degrees(math.atan2(dz, math.hypot(dx, dy)))
        return (round(pitch, 1), 0.0, round(yaw, 1))

    def has_preview(self) -> bool:
        return self.preview is not None and self.preview.exists()


@dataclass
class CameraManager:
    """Every camera in the rig, and which one is live."""

    views: dict[str, CameraView] = field(default_factory=dict)
    current: str | None = None
    source: str = ""
    physical: bool = True

    # -- loading ------------------------------------------------------------
    @classmethod
    def load(cls, path="config/cameras.yaml",
             preview_dir="renders/camera_preview") -> "CameraManager":
        cfg = Path(path)
        if not cfg.is_absolute():
            cfg = ROOT / cfg
        if not cfg.exists():
            raise FileNotFoundError(f"no camera rig at {cfg}")

        data = yaml.safe_load(cfg.read_text(encoding="utf-8")) or {}
        previews = Path(preview_dir)
        if not previews.is_absolute():
            previews = ROOT / previews

        views: dict[str, CameraView] = {}
        # The physical rig is a list of camera dicts; the legacy 2D rig is a
        # mapping. The shape of the file is what tells them apart.
        specs = data.get("cameras")
        physical = isinstance(specs, list)

        if physical:
            for spec in specs:
                key = spec["id"]
                intent = PHYSICAL_INTENT.get(key, spec.get("composition", ""))
                views[key] = CameraView(
                    key=key,
                    label=f"{key.upper()}  {intent}",
                    intent=intent,
                    position=tuple(spec["position"]),
                    look_at=tuple(spec["look_at"]),
                    focal_mm=float(spec["focal_length_mm"]),
                    enabled=bool(spec.get("enabled", True)),
                    preview=previews / f"{key}.png",
                    physical=True,
                )
        else:
            for key, spec in (specs or {}).items():
                intent = LEGACY_INTENT.get(key, (spec.get("label") or key))
                views[key] = CameraView(
                    key=key,
                    label=f"{key.upper()}  {intent}",
                    intent=intent,
                    position=(0.0, 0.0, 0.0),
                    look_at=(0.0, 0.0, 0.0),
                    focal_mm=0.0,
                    enabled=True,
                    preview=None,
                    physical=False,
                )

        mgr = cls(views=views, source=str(cfg), physical=physical)
        mgr.current = mgr.ordered()[0].key if mgr.views else None
        return mgr

    # -- access -------------------------------------------------------------
    def ordered(self) -> list[CameraView]:
        return sorted(self.views.values(), key=lambda v: (v.index, v.key))

    def keys(self) -> list[str]:
        return [v.key for v in self.ordered()]

    def view(self, key: str) -> CameraView | None:
        return self.views.get(key)

    @property
    def active(self) -> CameraView | None:
        return self.views.get(self.current) if self.current else None

    def select(self, key: str) -> bool:
        if key not in self.views or key == self.current:
            return False
        self.current = key
        return True

    def step(self, delta: int) -> str | None:
        keys = self.keys()
        if not keys:
            return None
        if self.current not in keys:
            return keys[0]
        return keys[(keys.index(self.current) + delta) % len(keys)]

    def by_number(self, n: int) -> str | None:
        """Camera for a number key. 1 means cam1, not 'the first available'."""
        for v in self.ordered():
            if v.index == n:
                return v.key
        return None

    def missing_previews(self) -> list[str]:
        return [v.key for v in self.ordered() if not v.has_preview()]

    def describe(self) -> list[str]:
        """Debug overlay lines for the active camera."""
        v = self.active
        if v is None:
            return ["ACTIVE CAMERA: none"]
        lines = [f"ACTIVE CAMERA: {v.key.upper()} - {v.intent}"]
        if v.physical:
            lines.append(f"FOCAL: {v.focal_mm:.0f} mm")
            lines.append("POSITION: [" +
                         ", ".join(f"{c:+.2f}" for c in v.position) + "]")
            p, _, y = v.rotation_deg
            lines.append(f"ROTATION: pitch {p:+.1f} yaw {y:+.1f} deg")
        else:
            # Never let the overlay imply a viewpoint change that is really a
            # crop of the same photograph.
            lines.append("SOURCE: one 2D photograph, reframed - not a viewpoint")
        return lines
