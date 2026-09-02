# The body: canonical motion state, rig adapter, and what changed

## The architectural rule, and how it is enforced

There is **one** canonical `HumanMotionState` (`src/presenter/motion/state.py`)
and there are adapters. There is no `MPFBBehaviorEngine` and no
`LivePortraitBehaviorEngine`, and there cannot be one: nothing in
`presenter/behavior/` or `presenter/motion/` imports a renderer, a rig, a
camera or an identity. The two adapters that exist import `bpy` and `AvatarPose`
respectively, and neither is reachable from a decision.

```
                    HumanMotionState
                          │
        ┌─────────────────┴─────────────────┐
        ▼                                   ▼
  adapters/face2d.py                 adapters/mpfb.py
  (2D LivePortrait plate)            (MPFB test rig, Blender)
        │                                   │
        ▼                                   ▼
   AvatarPose                       48 posed bones + IK
```

The clearest evidence that the boundary is real is that the two adapters want
**opposite things from the same state**. The 2D renderer drives deltas from a
photograph of a man who is already sitting, so it subtracts the seated neutral;
the rig rests in a T-pose and wants the absolute value. Feeding the 2D path the
absolute chain tipped his head down 7.8 degrees. Neither adapter's need belongs
in the motion state, and neither leaks into the other.

## What the test rig is, and is not

`MPFB_BODY_RIG_FOUNDATION_V1` — a fitted CC0 MakeHuman mesh plus 125 joint
markers, tagged by the Camera Terminal with "Body/rig foundation: PROMISING.
Face identity: FAILED - do not resume face fitting."

`tools/build_body_rig.py` turns that into 48 bones — pelvis through four spine
segments, a separate `chest_top` for the rib cage, neck, head, clavicles, arms,
wrists, five fingers of three segments per hand, and eyes — skinned with
automatic weights, with two-bone IK on each forearm.

**Its face is not our avatar's face and is not meant to be.** Identity belongs
to the Camera Terminal.

## Breathing, moved out of the head

The previous implementation wrote breathing into `pose.scale`: the head
periodically grew. That existed because head scale was the only channel the 2D
renderer exposed, which is exactly the behaviour-to-renderer coupling the
canonical state now forbids.

Coupling, in degrees at full inhale:

| | |
|---|---|
| chest | −0.95 (primary) |
| spine_mid | −0.34 |
| clavicle | −0.26 / ±0.30 |
| shoulder | −0.09 (very low) |
| neck | **+0.30** |
| head | +0.04 (near zero) |

Plus a 1.3% circumference change on `chest_top` as a scale, which no rotation
can express.

The neck term is positive against a negative chest on purpose. As the chest
opens it would carry the head back with it; a real neck compensates so the gaze
stays level. Without that the head nods gently in time with the breath, which
is how a breathing rig announces itself.

The waveform is not a sine: inhale (active, shorter) → brief transition →
exhale (passive, longer) → variable rest, with raised-cosine ramps so velocity
is zero at both ends. Rate and depth drift on ~50 s Ornstein-Uhlenbeck
processes rather than being re-drawn per cycle — real respiration drifts slowly
and successive breaths resemble each other.

**The honest cost:** through the 2D face renderer this produces almost nothing,
because a rib cage is outside the animated crop. That clip is measurably less
alive than when the head was scaling. The fix is the body, not putting the head
scaling back.

## Eye-head recruitment

The review found a 19° glance recruiting 3° of head — side-eye. The cause was a
fixed ratio. Recruitment now depends on three things:

* **eccentricity**, on the full angle rather than azimuth alone. Azimuth alone
  meant the desk — 6° across but 17° down — recruited no head at all, so he
  looked at his own hands by rolling his eyes down.
* **how long he means to look**, scaled by the target's dwell. This is the one
  that fixes the side-eye: nobody turns their head to check something for half
  a second, everybody turns it to read for five.
* **what he is doing** — reading and focusing recruit more head at the same
  angle.

| | head yaw / pitch |
|---|---|
| quick 0.6 s glance at chat (18.3° ecc) | +3.6° / −1.0° |
| normal 3.3 s look at chat | +5.7° / −1.5° |
| 2.8 s read of second display (21.3°) | −11.4° / −1.6° |
| desk (6° across, 17° down) | −1.7° / **−4.0°** |
| main display (4.9°, next to the lens) | 0 / 0 |

The eyes lead by 20–60 ms and the head's share then ramps in over ~1.8 s. The
eyes *recenter in their sockets* as it does, with nothing animating that: the
eye angle is always the residual `target − head`, so it shrinks as the head
grows.

## Seated posture

`SEATED_NEUTRAL` is deliberately asymmetric and fixed — left clavicle lower,
unequal shoulders and elbows, head a degree off square. Fixed matters: a body
asymmetric differently every second reads as noise, not as a person's habitual
way of sitting.

Posture is a continuum (`engagement`, −1 settled back to +1 forward focus), not
four clips. Comfort shifts — shoulder settle, pelvis shift, lean, settle back,
hand reposition, torso rotate — fire on a ~3 min median with recency
suppression, ease in over seconds, and then **persist**, decaying back over a
4-minute time constant. A settle that springs back was not a settle.

## Hands

One arrangement, locked: dominant hand on the mouse, other on the desk, both
pinned by IK. Contact targets are placed at 0.72 of the arm's *measured* reach.

The first version placed them by fractions of shoulder width and put the mouse
6.09 units from the shoulder against a total reach of 5.20 — out of reach. The
IK solved that correctly by straightening the arm and pointing at the target,
which is why the character sat at his desk with both arms locked straight out
to the sides.

Finger curl is on bone-local **X**, established by driving each axis and
looking; Z was the first guess and fanned the fingers apart instead of closing
them. Resting curls are non-zero and unequal per finger.

## Smiles

`FacialExpressionSystem` writes corners, cheeks and lower lids together in
fixed proportion, because zygomaticus major raises cheek mass which narrows the
eye from below. Measured at peak: cheek/corner **0.75**, squint/corner **0.46**,
left/right asymmetry **30%**, drawn once per instance so the smile is lopsided
the way a face is rather than flickering.

Profile is onset → peak → hold → decay → residual, with decay always slower
than onset, and a residual that lingers for seconds. Every expression is
triggered through a drawn reaction latency (measured 0.46–0.64 s), so nothing
reacts on the frame it was decided.

## Background drift

The report said 0.14 levels/frame inside the animated crop. Diagnosed before
fixing: **it was not recursion.** `_composite` already starts from an immutable
master every frame, so there was no feedback loop to break.

The real cause: LivePortrait's paste mask is a rounded box around the *crop*,
and a crop containing a head also contains a good deal of the room behind it.
Those pixels were being regenerated every frame. With `environment="source"`
the subject restriction was never applied.

The fix is a static dynamic-human region — the segmented silhouette, grown by
2.2% of frame width to cover the head's full excursion plus hair. Outside it the
room comes from the plate. Generous on purpose: over-growing costs a slightly
larger repaint region, under-growing clips hair, and the brief is explicit that
this must not blur the hair or jaw boundary.

---

# Seated mechanics milestone

## Coordinate contract

`docs/rig_coordinate_contract.md`, regenerated by `tools/rig_axes.py` — the
whole file, so nothing hand-added can be silently lost on the next run.

Bone axes are **measured**, not assumed. A Blender bone's local Y runs along its
own length; X and Z come from the bone's roll, which here derives from fitted
joint data rather than from anything a person chose. The tool reads all 56
bones, computes each hinge's anatomical axis as the normal of the plane its two
segments lie in, and reports how closely the bone's own axes line up with it.

| joint | hinge | quality |
|---|---|---|
| knee | X | 0.997 |
| elbow | Z | 0.805 |
| thumb, index | X | 0.744–0.998 |
| middle, ring, little | **Z** | 0.830–0.981 |

That last row is the finding. **Finger axes are not uniform across the hand**,
and the signs differ between hands. A single hardcoded curl axis was right for
two fingers out of five and drove the other three sideways — which is why the
hand fanned open instead of closing. `adapters/mpfb.py` now reads
`config/rig_axes.json` and contains no axis index of its own.

## Arm reach

Everything is placed as a fraction of **measured** reach (shoulder→wrist,
52 cm):

| target | % of reach |
|---|---|
| mouse, desk rest | 80% |
| lap | 88% |
| armrest | 67% |

The failure this replaces put the mouse at **117%** of reach. The solver did the
only thing available and straightened the arm to point at it, which is why the
character sat with both arms locked out sideways.

Pole targets were tried and made things worse: a pole angle is measured from the
bone's roll, the elbow's roll is only 0.805 aligned with its hinge, and leaving
the angle at zero twisted both arms out of their sockets. The rig instead locks
the two axes a hinge cannot rotate about and limits the third — equivalent, no
angle to get wrong, and the solver has nowhere to flip to.

## Lower body

Thigh, shin, foot and toe per side, from `joint-*-upper-leg` through
`joint-*-foot-2`. Two-bone IK from the shin to a floor target, knee hinge locked
to its measured axis. Feet are planted with a deliberate fore/aft asymmetry that
belongs to this character rather than being re-randomised.

The floor is placed one shin plus one ankle-height below the hip — from the leg
the character actually has, which is what makes the feet land rather than
hover.

## Contact, measured not asserted

`tools/contact_check.py` evaluates the **deformed** mesh — armature applied —
against every surface. Both failure directions matter and look identical in a
still: a hand 2 cm above a desk and a hand 2 cm inside it are both wrong.

| contact | upright | settled back |
|---|---|---|
| pelvis / seat | +0.1 cm | +0.1 cm |
| back / backrest | +6.6 cm (clear) | **+0.2 cm (resting)** |
| feet / floor | −0.5 / −0.3 cm | −0.7 / −0.6 cm |
| hands / desk | +0.3 / +0.2 cm | −0.2 / −0.3 cm |

Chair contact is now physical: settling slides the pelvis back and closes a
6.6 cm gap to 0.2 cm. `back_contact` is *derived* from where the spine ends up,
so it cannot claim the back is resting while the torso is leaning away.

## Three bugs the checks caught

**The wrist clearance.** The IK chain ends at the wrist, so a target on the desk
plane put the wrist joint on the surface and the whole hand below it. The finger
test showed the fingers curling through the desk.

**Bone-local translation.** `pose_bone.location` is in bone space, and the root
bone points up the spine — so a world-space slide assigned to it moved the body
*along the spine*. Settling back lifted the pelvis 4.7 cm off the seat.

**Coordinate bands that stop covering the body.** Both the hand and pelvis
checks first used fixed coordinate bands. The hand band caught the elbow hanging
below the desk edge and reported the hands 14 cm inside the desk; the pelvis band
stopped covering the buttocks the moment they slid back, and measured the thigh
instead. Every region is now anchored to a posed bone.

## Seated joint limits

`SEATED_JOINT_LIMITS` in `constraints.py` bounds all sixteen driven joints to
what a man in a task chair actually does — not full anatomical range, which
would never catch anything. Over five minutes of ordinary idle behaviour
**zero** axes are clamped: the limits are a backstop for generated motion, not a
working constraint.

## Motion recording

`presenter/motion/recording.py` writes a `HumanMotionState` sequence as a flat
float32 array with a generated column header — not a pickle, because a motion
recording is exactly the artefact another terminal might want to read. 60 s is
1800 frames × 101 channels in 215 KB, and round-trips with **max abs difference
0.0**. Non-numeric channels (attention target, behaviour state, hand contacts)
travel alongside, because they carry the *reason* for a movement.
