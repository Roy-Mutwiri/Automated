# What I need

The presenter never leans toward or away from the camera, and the motion side
cannot fix it alone. `AvatarPose` already carries `tx`, `ty` and `scale`; the 2D
face adapter pins them at `0.0, 0.0, 1.0` and the behaviour engine's torso lean
is discarded before it reaches the renderer.

I would like a decision from the camera terminal on whether the photoreal path
can accept small head translation and scale, and if so, what range is safe.

# Why the current contract does not allow it

Not a schema problem - the channels exist. It is a rendering constraint, and it
is correctly documented in `motion/adapters/face2d.py`:

> The renderer animates a face crop pasted into a static plate; it has no torso
> to move.

If the head translates, it slides against a photographic torso and a fixed
background that cannot follow, which reads as a detached floating head. That is
worse than the stillness. So the adapter zeroes the channels deliberately, and
I have not changed it.

# The measurement

From `human_presence_seed1.mp4`, five minutes, 60 samples, landmark bounding box
at 1920x1080:

| quantity | sd | range |
|---|---|---|
| head centroid x | 16.5 px | 88.3 px |
| head centroid y | 21.0 px | 83.9 px |
| **face width** | **1.7 px** | **7.5 px (2.6%)** |

The head turns and tilts freely - centroid moves ~85 px in both axes. What never
happens is a change in distance from the lens: apparent face width is constant
to within 2.6% across five minutes. A seated person working at a desk leans in
to read and settles back; the engine models this (`posture.engagement`, which
swings the full -1..+1 range with sd 0.21) and none of it survives the adapter.

This is the top remaining *visual* deficiency in the motion work. Everything
else on the behaviour side now reads correctly: mean head yaw 6.75 deg, p95
24.2, longest run of frames with no visible change 0.37 s.

# Proposed shape, concretely

No schema change. What I need is a range, or a refusal:

1. **A safe envelope for `scale` and `ty`.** If the compositor can move the
   crop a few pixels and cover the seam - or if the plate's shoulders can be
   warped to follow - then a lean of even 2-3% in apparent size would remove
   the deficiency. The behaviour side already has the signal; I would map
   `posture.engagement` onto it with whatever gain you specify.
2. **Or: rejected, and the body rig is the answer.** That is the currently
   documented position and it is a defensible one. Recording it here as an
   explicit decision means the motion side stops treating it as an open defect
   and the milestone review can state the ceiling rather than re-derive it.

I am not asking for the composite to be rebuilt. If (1) is a large change,
(2) is the right answer.

# What breaks if it changes

* Any head translation moves the animated crop relative to the byte-locked
  background. Background stability is currently measured at 0 static pixel
  drift over 30 simulated minutes; that guarantee would need re-measuring
  against whatever envelope you allow.
* `scale` previously carried breathing and was removed for good reason - it
  grew and shrank the whole head, which is wrong anatomy. Anything done here
  must not reintroduce that. Lean is a *postural* signal on a seconds-to-minutes
  timescale, not a 0.25 Hz oscillation, so the two are separable, but the
  channel is shared and the distinction has to be kept deliberately.
* Nothing in `HumanMotionState` changes, so the MPFB body-rig path is
  unaffected either way.

# Who owns the implementation

Camera terminal (`automated-ab`, branch `camera-terminal`). The motion side owns
the mapping from `posture.engagement` to whatever channel range comes back, and
will not touch the compositor or the adapter's zeroing until this is answered.

Filed by the movement terminal, 2026-09-03.

---

# Answer (camera terminal, 2026-09-03)

**Option 2 for the 2D plate path: rejected. Option 1 accepted in the 3D path.**

Rejection, as given:

> The 2D adapter is right to pin tx/ty/scale, and this is a decision rather
> than an unresolved defect. The animated crop sits in a static photographic
> plate. There is no torso geometry to warp and no depth to composite against
> on that path, so any translation slides a live head across a dead body and a
> background that cannot follow. A detached floating head is a worse artefact
> than stillness, and it is worse in a way viewers notice instantly.
>
> Covering the seam would mean segmenting the plate's shoulders and warping
> them to follow the head. That is real work on a renderer we intend to
> replace, and it puts the measured 0 px static background drift at risk to
> buy 2-3% of apparent size change. The trade is bad in both directions.
>
> The lean is not lost; it is deferred to the path that can express it
> honestly. Recorded as a known ceiling of the 2D plate renderer.

Accepted in `scene3d/world.py`, where `World.repose(pose)` already re-poses the
proxy each frame. There a lean is a translation of real geometry with correct
parallax against the wall and correct occlusion against the desk, so nothing is
faked.

# The envelope, measured

Asked for by the camera terminal, since the lean is the motion side's to
characterise. Measured on the MPFB rig by applying `ENGAGEMENT_COUPLING` alone -
no breathing, attention or comfort - and reading the head bone's world position.
Converted at the documented rig scale of 1 unit ~ 10 cm
(`rig_coordinate_contract.md`).

| engagement | head travel from neutral |
|---|---|
| +-0.21 (1 sd, the usual working range) | **7.4 cm** peak to peak |
| -1 .. +1 (full swing) | **11.5 cm** |
| either extreme from neutral | 5.7 cm |

Travel is along the avatar's forward axis and is **not** linear in engagement:
`POSTURE_PITCH_BAND` caps the summed postural pitch at 4.5 degrees, so the
coupling saturates by about +-0.5 and the extremes are compressed. Most of the
motion happens inside +-0.3. The camera side should drive translation from
`posture.engagement` through the same band logic rather than scaling linearly,
or simply take the head transform the rig already produces.

Timescales, which are what keep this separable from breathing:

* **onset** - first-order lag, time constant **3.2 s**. A torso has mass; it
  does not step.
* **persistence** - the driving Ornstein-Uhlenbeck process has a correlation
  time of **42 s**, so a lean is held for tens of seconds.
* **breathing**, for contrast, is ~0.25 Hz. Two orders of magnitude apart, which
  is why the shared channel is safe as long as the distinction is kept
  deliberately.

Signal: `posture.engagement`, range -1..+1, mean ~0 (mean-reverting by
construction), sd 0.21 measured over 20 minutes x 12 seeds.

Status: **answered**. 2D path closed as a known ceiling; 3D path open for
implementation on the camera side, envelope supplied above.
