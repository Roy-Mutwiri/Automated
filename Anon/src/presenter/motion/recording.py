"""Record and replay a `HumanMotionState` sequence.

The point is not archiving. It is that a recorded performance can be pushed
through *any* adapter and produce the same human, which is the only real proof
that behaviour and renderer are separate - and the mechanism by which seven
cameras will eventually render one take rather than seven near-misses.

## Format

A flat float32 array per frame, plus a header naming the columns. Deliberately
not pickle: a pickled dataclass is unreadable by anything that is not this exact
Python, and a motion recording is precisely the artefact another terminal might
want to read.

Columns are generated from the state's own structure, so adding a joint changes
the file format without anyone having to remember to update a writer.
"""

from __future__ import annotations

import json
from dataclasses import fields
from pathlib import Path

import numpy as np

from .state import (AttentionState, BreathingState, EmotionState,
                    FaceParameters, HandPose, HumanMotionState, JointRotation,
                    PostureState)

__all__ = ["MotionRecording", "columns"]

_JOINTS = ("pelvis", "spine_lower", "spine_mid", "chest", "clavicle_l",
           "clavicle_r", "shoulder_l", "shoulder_r", "elbow_l", "elbow_r",
           "wrist_l", "wrist_r", "neck", "head", "eye_l", "eye_r")
_ROOT = ("root_x", "root_y", "root_z", "root_yaw")
_SCALARS = {
    "face": FaceParameters,
    "breathing": BreathingState,
    "posture": PostureState,
    "attention": AttentionState,
    "emotion": EmotionState,
}


def columns() -> list[str]:
    """Every numeric channel of the state, in a fixed order."""
    cols = ["timestamp", *_ROOT]
    for j in _JOINTS:
        cols += [f"{j}.rx", f"{j}.ry", f"{j}.rz"]
    for side in ("l", "r"):
        cols += [f"hand_{side}.curl{i}" for i in range(5)]
        cols += [f"hand_{side}.spread", f"hand_{side}.contact_weight"]
    for name, cls in _SCALARS.items():
        for f in fields(cls):
            if f.type in ("float", float) or isinstance(
                    getattr(cls(), f.name), (int, float)) and not isinstance(
                    getattr(cls(), f.name), bool):
                cols.append(f"{name}.{f.name}")
    return cols


class MotionRecording:
    """A sequence of motion states, writable and replayable."""

    def __init__(self, cols: list[str] | None = None) -> None:
        self.columns = cols or columns()
        self._index = {c: i for i, c in enumerate(self.columns)}
        self.rows: list[np.ndarray] = []
        # Non-numeric channels, kept alongside rather than dropped: the
        # attention target's *name* is the reason for a movement, and a
        # recording that loses it cannot explain itself.
        self.labels: list[dict] = []

    # -- writing -----------------------------------------------------------
    def append(self, m: HumanMotionState) -> None:
        row = np.zeros(len(self.columns), dtype=np.float32)
        idx = self._index
        row[idx["timestamp"]] = m.timestamp
        for name in _ROOT:
            row[idx[name]] = getattr(m, name)
        joints = m.joints()
        for j in _JOINTS:
            r = joints[j]
            row[idx[f"{j}.rx"]] = r.rx
            row[idx[f"{j}.ry"]] = r.ry
            row[idx[f"{j}.rz"]] = r.rz
        for side, hand in (("l", m.hand_l), ("r", m.hand_r)):
            for i, c in enumerate(hand.curl):
                row[idx[f"hand_{side}.curl{i}"]] = c
            row[idx[f"hand_{side}.spread"]] = hand.spread
            row[idx[f"hand_{side}.contact_weight"]] = hand.contact_weight
        for name in _SCALARS:
            obj = getattr(m, name)
            for col in self.columns:
                if col.startswith(name + "."):
                    attr = col.split(".", 1)[1]
                    v = getattr(obj, attr, 0.0)
                    if isinstance(v, (int, float)) and not isinstance(v, bool):
                        row[idx[col]] = float(v)
        self.rows.append(row)
        self.labels.append(dict(
            behavior_state=m.behavior_state,
            attention_target=m.attention.target,
            emotion_label=m.emotion.label,
            hand_l_contact=m.hand_l.contact,
            hand_r_contact=m.hand_r.contact,
        ))

    def save(self, path: str | Path) -> None:
        path = Path(path)
        np.savez_compressed(
            path,
            data=np.stack(self.rows) if self.rows else np.zeros((0, len(self.columns))),
            columns=np.array(self.columns),
            labels=np.array([json.dumps(l) for l in self.labels]),
        )

    # -- reading -----------------------------------------------------------
    @classmethod
    def load(cls, path: str | Path) -> "MotionRecording":
        z = np.load(Path(path), allow_pickle=False)
        rec = cls(list(z["columns"]))
        rec.rows = [r for r in z["data"].astype(np.float32)]
        rec.labels = [json.loads(s) for s in z["labels"]]
        return rec

    def __len__(self) -> int:
        return len(self.rows)

    def state(self, i: int) -> HumanMotionState:
        """Rebuild a motion state. Byte-for-byte for every numeric channel."""
        row, lab, idx = self.rows[i], self.labels[i], self._index
        m = HumanMotionState()
        m.timestamp = float(row[idx["timestamp"]])
        for name in _ROOT:
            setattr(m, name, float(row[idx[name]]))
        joints = m.joints()
        for j in _JOINTS:
            r = joints[j]
            r.rx = float(row[idx[f"{j}.rx"]])
            r.ry = float(row[idx[f"{j}.ry"]])
            r.rz = float(row[idx[f"{j}.rz"]])
        for side, hand in (("l", m.hand_l), ("r", m.hand_r)):
            hand.curl = [float(row[idx[f"hand_{side}.curl{i2}"]]) for i2 in range(5)]
            hand.spread = float(row[idx[f"hand_{side}.spread"]])
            hand.contact_weight = float(row[idx[f"hand_{side}.contact_weight"]])
            hand.contact = lab[f"hand_{side}_contact"]
        for name in _SCALARS:
            obj = getattr(m, name)
            for col in self.columns:
                if col.startswith(name + "."):
                    setattr(obj, col.split(".", 1)[1], float(row[idx[col]]))
        m.behavior_state = lab["behavior_state"]
        m.attention.target = lab["attention_target"]
        m.emotion.label = lab["emotion_label"]
        return m

    def __iter__(self):
        for i in range(len(self)):
            yield self.state(i)
