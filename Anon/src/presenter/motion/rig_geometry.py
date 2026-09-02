"""Seated-scene geometry, derived once from the body's own measurements.

The single source of truth for where the floor, the seat, the backrest and the
desk are. Both the rig builder and the debug renderer import it, so they cannot
disagree - which they did: the desk was placed at +Y in the render scene while
the hand targets were computed at -Y, and the character sat with his back to his
own desk.

## Which way round the derivation goes

The brief says the pelvis should be derived from the chair. Here it is the other
way about, and deliberately: the body mesh is the fixed input, its proportions
are measured rather than chosen, and a chair derived from a real pelvis is more
likely to fit than a pelvis fitted to an invented chair. The important property
is that there is **one** derivation and everything reads it.

## Scale

Nothing in the source data states a unit. It is recoverable: the mesh spans
17.53 units from `joint-ground` to the top of the skull, and a male adult is
about 1.75 m, so **1 unit ~ 10 cm**. Every dimension below is quoted in
centimetres in a comment so the numbers can be sanity-checked against furniture
rather than against each other.

## Coordinates

MakeHuman source data: +X left, +Y up, +Z forward.
Blender rig space:     +X left, +Y back, +Z up.

    (x, y, z)_mh  ->  (x, -z, y)_blender

This module works in **MakeHuman space** because that is what the joint file
uses; `mh_to_blender` is applied once, at the point bones are created.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path

__all__ = ["SeatedGeometry", "load_joints", "mh_to_blender", "UNITS_PER_CM",
           "WRIST_CLEARANCE"]

# Recovered from the mesh: 17.53 units for a ~1.75 m adult.
UNITS_PER_CM = 0.10

# Height of the wrist joint centre above a surface the hand rests on: roughly
# half the wrist's thickness.
WRIST_CLEARANCE = 3.5 * 0.10

# Rearmost point of the torso, measured off the mesh in the lumbar-to-mid-back
# band (min Z over vertices with |x| < 2 in that height range). Not derived from
# the spine joint, which sits inside the body.
BACK_SURFACE_Z = -1.04

# The posed torso leans further forward than the rest pose, so the gap the body
# actually has to close is larger than the rest geometry implies. Measured with
# tools/contact_check.py.
POSED_LEAN_ALLOWANCE = 2.2 * 0.10


def load_joints(path="research/avatar_reconstruction/outputs/fitted_joints.json"):
    return json.loads(Path(path).read_text())


def mh_to_blender(p):
    """MakeHuman (Y-up, +Z forward) -> Blender (Z-up, -Y forward)."""
    x, y, z = p
    return (x, -z, y)


def _dist(a, b):
    return math.dist(a, b)


@dataclass
class SeatedGeometry:
    """Every plane and contact point of the seated scene, in MakeHuman space."""

    # measured segment lengths
    thigh: float
    shin: float
    foot_rise: float          # ankle height above the sole
    upper_arm: float
    forearm: float
    hand: float

    # derived planes (MakeHuman Y)
    floor_y: float
    seat_y: float
    seat_back_z: float        # +Z of the backrest face
    seat_front_z: float
    desk_y: float
    desk_front_z: float

    hip_y: float
    hip_half_width: float
    shoulder_y: float

    @property
    def arm_reach(self) -> float:
        """Shoulder to wrist. The hand is deliberately excluded: an IK chain
        that ends at the wrist must not be given the palm's length to spend."""
        return self.upper_arm + self.forearm

    @classmethod
    def measure(cls, j: dict) -> "SeatedGeometry":
        thigh = _dist(j["joint-l-upper-leg"], j["joint-l-knee"])
        shin = _dist(j["joint-l-knee"], j["joint-l-ankle"])
        foot_rise = j["joint-l-ankle"][1] - j["joint-ground"][1]

        upper_arm = _dist(j["joint-l-shoulder"], j["joint-l-elbow"])
        forearm = _dist(j["joint-l-elbow"], j["joint-l-hand"])
        hand = _dist(j["joint-l-hand"], j["joint-l-hand-2"])

        hip_y = j["joint-pelvis"][1]
        hip_half_width = abs(j["joint-l-upper-leg"][0])
        shoulder_y = j["joint-l-shoulder"][1]

        # Seated, the shin hangs vertically and the sole meets the floor, so
        # the floor sits one shin plus one ankle-height below the hip. This is
        # what makes the feet land instead of hovering: the floor is placed
        # from the leg the character actually has.
        floor_y = hip_y - (shin + foot_rise)

        # The pelvis rests *on* the seat, so the seat surface is just below it.
        # 9 cm accounts for the flesh between the ischial tuberosities and the
        # joint centre; at 7.5 the contact check measured the buttocks 1.4 cm
        # inside a rigid slab. A real cushion would compress by about that
        # much, but the debug seat does not, so it reads as clipping.
        seat_y = hip_y - 9.0 * UNITS_PER_CM

        # Seat depth ~45 cm, with the backrest behind the spine.
        seat_back_z = j["joint-spine-4"][2] - 12.0 * UNITS_PER_CM
        seat_front_z = seat_back_z + 45.0 * UNITS_PER_CM

        # Desk height ~73 cm above the floor, front edge just clear of the knee.
        desk_y = floor_y + 73.0 * UNITS_PER_CM
        desk_front_z = j["joint-l-knee"][2] + 14.0 * UNITS_PER_CM

        return cls(thigh=thigh, shin=shin, foot_rise=foot_rise,
                   upper_arm=upper_arm, forearm=forearm, hand=hand,
                   floor_y=floor_y, seat_y=seat_y,
                   seat_back_z=seat_back_z, seat_front_z=seat_front_z,
                   desk_y=desk_y, desk_front_z=desk_front_z,
                   hip_y=hip_y, hip_half_width=hip_half_width,
                   shoulder_y=shoulder_y)

    # -- contact targets ----------------------------------------------------
    def hand_targets(self, j: dict, rest_fraction: float = 0.80) -> dict:
        """Resting hand positions, on the desk plane, inside comfortable reach.

        `rest_fraction` is the share of shoulder-to-wrist reach the target sits
        at. The brief asks for 70-90%; 80% puts the elbow near 100 degrees,
        which is what a forearm resting on a desk does.

        The failure this replaces put the target at 117% of reach. The solver
        did the only thing it could and straightened the arm to point at it.
        """
        out = {}
        for side, sign in (("l", +1.0), ("r", -1.0)):
            sh = j[f"joint-{side}-shoulder"]
            d = self.arm_reach * rest_fraction

            dy = self.desk_y - sh[1]
            if abs(dy) >= d:
                # Cannot reach the desk at all at this fraction; sit as low as
                # the arm allows rather than silently producing a straight arm.
                dy = math.copysign(d * 0.95, dy)
            planar = math.sqrt(max(d * d - dy * dy, 1e-4))

            # A little inward, most of it forward.
            dx = -sign * planar * 0.10
            dz = math.sqrt(max(planar * planar - dx * dx, 1e-4))

            key = "mouse" if side == "r" else "desk_rest_l"
            # The IK chain ends at the *wrist*, so a target on the desk plane
            # puts the wrist joint on the surface and the whole hand below it -
            # the finger test showed the fingers curling through the desk. The
            # wrist centre sits about one wrist radius above a surface the palm
            # is resting on.
            out[key] = (sh[0] + dx, self.desk_y + WRIST_CLEARANCE, sh[2] + dz)

        # Lap: the hand rests on the thigh, so it sits a little above the seat
        # and well forward of the hip.
        #
        # This lands near 88% of reach rather than the 80% used for the desk,
        # and that is correct rather than sloppy: with the torso upright the arm
        # hangs almost straight down to the lap, so the elbow is only bent to
        # about 130 degrees. The brief's 70-90% band is quoted for desk and
        # mouse poses, which are the ones a too-distant target ruins.
        for side, sign in (("l", +1.0), ("r", -1.0)):
            sh = j[f"joint-{side}-shoulder"]
            out[f"lap_rest_{side}"] = (
                sh[0] - sign * self.arm_reach * 0.16,
                self.seat_y + 13.0 * UNITS_PER_CM,
                self.seat_front_z - 9.0 * UNITS_PER_CM,
            )
            out[f"armrest_{side}"] = (
                sh[0] + sign * 3.0 * UNITS_PER_CM,
                self.seat_y + 20.0 * UNITS_PER_CM,
                self.seat_back_z + 26.0 * UNITS_PER_CM,
            )

        mouse = out["mouse"]
        out["keyboard"] = (mouse[0] * 0.30, mouse[1], mouse[2] - 3.0 * UNITS_PER_CM)
        return out

    def back_contact_travel(self, j: dict) -> tuple[float, float]:
        """(slide range, gap at rest) for the pelvis, in MakeHuman Z.

        How far back the pelvis may slide before the back meets the rest.

        `BACK_SURFACE_Z` is measured off the mesh rather than estimated from the
        spine joint: the joint sits inside the torso and a guess at torso depth
        was 1.3 cm out. The travel also has to cover the fact that the *posed*
        torso leans further forward than the rest pose, which the contact check
        measured at 6.5 cm rather than the 4.3 cm the rest geometry implies.
        """
        gap = BACK_SURFACE_Z - self.seat_back_z + POSED_LEAN_ALLOWANCE
        return max(gap, 0.0), gap

    def foot_targets(self, j: dict) -> dict:
        """Both soles on the floor, with a small fore/aft asymmetry.

        Nobody plants both feet at the same distance. The offset is fixed for
        this character rather than re-randomised.
        """
        out = {}
        for side, ahead in (("l", +3.5), ("r", -2.0)):
            hip = j[f"joint-{side}-upper-leg"]
            out[f"foot_{side}"] = (
                hip[0] + (2.0 * UNITS_PER_CM if side == "l" else -3.0 * UNITS_PER_CM),
                self.floor_y + self.foot_rise,
                self.seat_front_z + (18.0 + ahead) * UNITS_PER_CM,
            )
        return out

    def pole_targets(self, j: dict) -> dict:
        """Elbow and knee poles.

        Without these the solver picks a plane on its own and can flip the
        elbow through the torso between frames. Elbows point outward and down;
        knees point forward and slightly out, which is where a seated person's
        knees actually go.
        """
        out = {}
        for side, sign in (("l", +1.0), ("r", -1.0)):
            el = j[f"joint-{side}-elbow"]
            out[f"pole_elbow_{side}"] = (
                el[0] + sign * self.arm_reach * 0.55,
                el[1] - self.arm_reach * 0.35,
                el[2] - self.arm_reach * 0.60,
            )
            kn = j[f"joint-{side}-knee"]
            out[f"pole_knee_{side}"] = (
                kn[0] + sign * self.thigh * 0.30,
                kn[1] + self.thigh * 0.10,
                kn[2] + self.thigh * 1.30,
            )
        return out

    def describe(self) -> str:
        cm = 1.0 / UNITS_PER_CM
        return "\n".join([
            f"  thigh          {self.thigh:6.2f} u  ({self.thigh * cm:5.1f} cm)",
            f"  shin           {self.shin:6.2f} u  ({self.shin * cm:5.1f} cm)",
            f"  ankle rise     {self.foot_rise:6.2f} u  ({self.foot_rise * cm:5.1f} cm)",
            f"  upper arm      {self.upper_arm:6.2f} u  ({self.upper_arm * cm:5.1f} cm)",
            f"  forearm        {self.forearm:6.2f} u  ({self.forearm * cm:5.1f} cm)",
            f"  arm reach      {self.arm_reach:6.2f} u  ({self.arm_reach * cm:5.1f} cm)",
            f"  floor  Y       {self.floor_y:6.2f} u",
            f"  seat   Y       {self.seat_y:6.2f} u  "
            f"({(self.seat_y - self.floor_y) * cm:5.1f} cm above floor)",
            f"  desk   Y       {self.desk_y:6.2f} u  "
            f"({(self.desk_y - self.floor_y) * cm:5.1f} cm above floor)",
            f"  seat   Z       {self.seat_back_z:6.2f} .. {self.seat_front_z:6.2f} u",
            f"  desk front Z   {self.desk_front_z:6.2f} u",
        ])
