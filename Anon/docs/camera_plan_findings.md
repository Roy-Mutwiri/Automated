# Camera plan findings

What the preview renders exposed, and what was done about it.

## The defect in CAM4 and CAM5

Both cameras were standing in the right place and looking at the wrong thing.

`behind him` was already correct in the config — he faces `+Y`, so behind is
low `Y`, and a comment in `cameras.yaml` said exactly that. The failure was the
aim point:

| | old `look_at` | what is there |
|---|---|---|
| cam4 | `[-0.10, 2.30, 1.00]` | empty floor, 0.30 m past the desk |
| cam5 | `[ 0.10, 2.60, 1.00]` | empty floor, 0.60 m past the desk |

The desk front edge is at `Y = 2.36`. Both cameras aimed *beyond* it, into the
undeveloped front half of the room, so each frame was the back of a head against
darkness. Nothing was wrong with the position, the lens, or the lighting. They
were pointed at nothing.

Corrected by aiming at scene landmarks instead of at round numbers:

| | new position | new `look_at` | lens |
|---|---|---|---|
| cam4 | `[ 0.44, 0.44, 1.64]` | `[0.00, 2.00, 0.80]` desk working surface | 40 mm |
| cam5 | `[-0.25, 0.24, 1.95]` | `[0.05, 1.62, 1.01]` midpoint of eye line and desk | 30 mm |

Both were chosen by `tools/search_camera_placement.py`, which scores candidates
on landmark visibility measured by projection and ray-cast rather than by eye.
cam4 measures desk 3/3 landmarks visible, head 27% of frame, 2% black; cam5
measures desk 3/3, head 9%, 2% black.

Neither is the top-scoring candidate on raw score alone. cam4's scoring winner
sat 0.34 m directly in front of the primary monitor, where a real camera body
would obstruct the display wall that cameras 1–3 exist to show; `x = +0.44` sits
in the gap between `monitor_main` and `monitor_right` instead, for four points
of score. cam5 is mounted at `z = 1.95`, clearing the tallest monitor's top edge
at 1.82, so it is bracketed *above* the panels rather than floating among them.

## CAM5 cannot show the monitors, and no placement fixes it

This was a hard acceptance requirement. It cannot be met in this room, and the
reason is geometry rather than framing.

    monitors      Y = 0.075, normal +Y
    avatar        Y = 1.240, facing +Y
    desk          Y = 2.000
    front cameras Y = 2.80 .. 2.94

The display wall is **behind him**, with the screens facing away from his back
and toward cameras 1–3. Any camera positioned to see his back is, by definition,
on the same side of him as that wall — so the monitors are behind the camera.
The `composition` field in the original config said `monitor rears`, which was
the honest reading all along.

So "streamer → desk → screens" is not a relationship this room contains. What he
actually faces is a bare desk: its only geometry is the top, a cable tray, and
the microphone clamp. There is no work surface content in his forward direction
at all, which is the deeper reason the rear shots looked empty.

Three ways to change that, none of them taken here:

1. **Put something on the desk** — a display, keyboard, deck. Cheapest, and it
   would make cam4 and cam5 read as a workstation instead of a bare slab.
2. **Move the display wall to what he faces** and make the slat wall behind him
   pure backdrop. This inverts the approved plate and would invalidate cam1–3.
3. **Accept it.** The display wall is a backdrop for the audience, which is a
   real and common streamer arrangement.

Option 1 is the recommendation, but it is new room geometry and the room is the
reference. Not fabricating it silently is the whole point of writing this down.

## The rear space is 1.0 m

He sits at `Y = 1.24` and the wall is at `Y = 0`. After clearance, a rear camera
has about one metre to stand in. That is why cam5 is 30 mm and why it reads as a
rear *medium* rather than the rear *wide* it is named. A genuinely wide
battlestation shot needs either a shorter lens, with the distortion that brings,
or more room behind him than the room has.

## Still true

Cameras 4–7 remain blocked as production shots. This was a correction of two
physically invalid designs, not the start of their development. `avatar_transform`,
the desk, the monitors and the room were not touched, and the camera bridge
re-verified at 0.0000 px after the extrinsics changed.
