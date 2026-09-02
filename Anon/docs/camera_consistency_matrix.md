# Camera consistency matrix

What each camera should see, derived from the actual layout in
`config/room_geometry.yaml` and `config/cameras.yaml` - not aspiration.

`Y` clearly visible · `P` partially visible or at the frame edge ·
`N` normally not visible.

The subject faces **+Y**. The slat wall is at **y = 0**, *behind* him. Cameras
1, 2, 3 and 7 are in front of him (high Y); cameras 4 and 5 stand in the 1.24 m
between his chair and the display wall; camera 6 is high on the same side as 3.

That geometry is the whole matrix. An earlier draft of the rig put cameras 4 and
5 at high Y - in front of him - and called them "over shoulder" and "rear".
`docs/room_floorplan.png` made the error obvious, which is the reason the
floorplan is generated from config rather than drawn.

## Subject

| | C1 | C2 | C3 | C4 | C5 | C6 | C7 |
|---|---|---|---|---|---|---|---|
| Face | Y | Y | Y | N | N | P | Y |
| Back of head | N | N | N | Y | Y | P | N |
| Left ear | P | Y | N | P | Y | P | P |
| Right ear | P | N | Y | Y | P | P | P |
| Hair crown | P | P | P | P | P | **Y** | P |
| Shoulders | Y | Y | Y | Y | Y | Y | Y |
| Hands on desk | P | P | P | Y | Y | **Y** | Y |
| Chest / breathing | Y | Y | Y | N | N | P | Y |

Cameras 4 and 5 are the identity test that matters: they see hair, ears and the
back of the neck, and nothing about those may differ from what 1-3 imply.

## Room

| | C1 | C2 | C3 | C4 | C5 | C6 | C7 |
|---|---|---|---|---|---|---|---|
| `walnut_wall_01` | Y | Y | Y | N | N | P | Y |
| `monitor_main` front | Y | Y | Y | N | N | P | Y |
| `monitor_main` rear | N | N | N | Y | Y | P | N |
| `monitor_left` | Y | Y | P | N | N | P | Y |
| `monitor_right` | P | N | Y | N | N | P | Y |
| `desk_main` top | N | N | N | Y | Y | **Y** | Y |
| `desk_main` front edge | P | P | P | N | N | P | Y |
| `chair_main` front | P | P | P | N | N | P | P |
| `chair_main` back | N | P | P | Y | **Y** | Y | P |
| `mic_main` boom | P | P | P | Y | Y | Y | Y |
| `speaker_right` | P | N | Y | N | N | P | Y |
| `pc_tower` | N | N | P | N | P | P | Y |
| `shelf_right` + `plant_shelf` | P | Y | P | N | N | P | Y |
| Floor | N | N | N | P | P | **Y** | Y |
| Ceiling | N | N | N | N | N | P | P |

## How this is used

`tools/validate_multicam.py` projects the canonical landmarks through every
camera and prints the *measured* visibility. Where the measurement disagrees
with this table, one of the two is wrong and it must be resolved by checking
occlusion - never by moving an object for one camera's benefit.

The measured table for the current rig is in `docs/multicam_validation.md`.

## Rows that are deliberately empty of promises

`monitor_main` rear, `desk_main` top and the floor are marked visible from
cameras 4-6, and that geometry exists (real monitor shells with arms, a real
desk with legs and a cable tray). It has never been *looked at*, because cameras
4-7 are disabled until 1-3 pass. When they are enabled, those surfaces are the
first thing to inspect - designed-once hidden geometry is exactly where a
canonical world quietly fails.
