"""Camera positions: which master frame each button shows, and whether it lives.

## A camera is a photograph, not a viewpoint

There is no 3D scene to move a virtual camera through. LivePortrait warps a
face crop out of one photograph; it cannot invent the back of a head or a
ceiling. So each camera is generated separately as its own master frame, and
switching cameras switches the prepared source - the same mechanism the
wardrobe uses, for the same reason.

## Live cameras and still cameras

`tools/evaluate_scenes.py` rejects any master frame whose head yaw exceeds 10
degrees, because past that LivePortrait hallucinates the parts of the head the
rotation reveals. That gate is what divides the list:

* A camera he **looks into** is near-frontal to *that* lens whatever angle his
  body sits at, so it passes the gate and can be animated. Real multi-camera
  presenters look at whichever camera is live, so the constraint and the
  realism agree.
* A camera showing his **back**, or far enough away that his face is a handful
  of pixels, has no drivable face and is a still.

`Camera.animated` carries that, and the application takes a different path for
each: a live camera goes through `LivePortraitRenderer.set_source`, a still is
displayed directly without the renderer being involved at all.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import yaml

__all__ = ["Camera", "CameraRig"]


@dataclass(frozen=True)
class Camera:
    key: str
    label: str
    animated: bool
    subject: str = ""
    hint: str = ""
    # A derived camera has no image of its own: it prepares the rig's master
    # frame at a different framing. That is what makes the man identical rather
    # than merely similar across the cameras that show his face - it is one
    # photograph, punched in and out of.
    derive: bool = False
    framing: str = "full"
    room: str | None = None            # None = the generator's shared room text
    negative: str | None = None
    negative_2: str | None = None

    @property
    def index(self) -> int:
        """1-based position, taken from the key (`cam3` -> 3)."""
        digits = "".join(c for c in self.key if c.isdigit())
        return int(digits) if digits else 0


@dataclass
class CameraRig:
    cameras: dict[str, Camera]
    directory: Path
    room: str = ""

    @classmethod
    def load(cls, path: str | Path = "config/cameras.yaml",
             root: str | Path | None = None) -> CameraRig:
        root = Path(root) if root else Path(__file__).resolve().parents[3]
        path = Path(path)
        if not path.is_absolute():
            path = root / path
        with open(path, encoding="utf-8") as fh:
            data = yaml.safe_load(fh)

        cameras = {}
        for key, spec in (data.get("cameras") or {}).items():
            spec = spec or {}
            cameras[key] = Camera(
                key=key,
                label=str(spec.get("label", key)),
                animated=bool(spec.get("animated", True)),
                subject=(spec.get("subject") or "").strip(),
                hint=(spec.get("hint") or "").strip(),
                derive=bool(spec.get("derive", False)),
                framing=str(spec.get("framing", "full")),
                room=(spec.get("room") or None),
                negative=(spec.get("negative") or None),
                negative_2=(spec.get("negative_2") or None),
            )

        directory = Path(data.get("directory", "assets/cameras"))
        return cls(
            cameras=cameras,
            directory=directory if directory.is_absolute() else root / directory,
            room=(data.get("room") or "").strip(),
        )

    # -- resolution ---------------------------------------------------------
    def path(self, key: str) -> Path:
        if key not in self.cameras:
            raise KeyError(f"unknown camera {key!r}; have {sorted(self.cameras)}")
        return self.directory / f"{key}.png"

    def exists(self, key: str) -> bool:
        return self.path(key).exists()

    def available(self) -> list[str]:
        return [k for k in self.cameras if self.exists(k)]

    def missing(self) -> list[str]:
        return [k for k in self.cameras if not self.exists(k)]

    def default(self) -> str | None:
        """The first generated camera, preferring an animated one.

        A still is a shot to cut *to*. Opening the session on one would show a
        frozen presenter, which is the single thing this project exists to
        avoid, so a live camera wins even if a still comes first in the file.
        """
        live = [k for k in self.available() if self.cameras[k].animated]
        return live[0] if live else next(iter(self.available()), None)

    def ordered(self) -> list[Camera]:
        return sorted(self.cameras.values(), key=lambda c: (c.index, c.key))
