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

Filed by the movement terminal, 2026-09-03. Status: **open**.
