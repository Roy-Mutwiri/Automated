# Camera 1: approved plate vs canonical 3D

Side by side: `renders/camera1_match.png`.

Left is `assets/reference/camera1_approved.png` - the approved 2D plate, and the
quality benchmark. Right is Camera 1 of the canonical 3D world at the same
notional viewpoint. **They are not close, and that is the expected result of
Stage 1, not a bug.**

## What matches

| | Status |
|---|---|
| Slat wall present, correct pitch and tone | **yes** - global slat coordinate system carried into 3D |
| Wall is fronto-parallel behind the subject | yes |
| Display wall behind him, panels facing camera | yes |
| Eye-level camera, ~50 mm, subject facing lens | yes |
| Warm key from camera-left, cooler fill right | yes |
| Overall palette: charcoal, walnut, matte black | yes |

## What does not match

| | Status | Why |
|---|---|---|
| The human | **hard mismatch** | Proxy geometry. Correct proportions and articulation, no photoreal skin, hair, beard or eyes. This is Stage 2. |
| Subject scale in frame | too large | The 3D subject fills more of the frame than the plate. Camera distance needs another pass against the plate. |
| Chair | not visible | Present in the world at the right place; the current framing crops it. |
| Microphone | not visible from cam1 | Present and correctly placed - the boom is below and left of this framing. |
| Desk and desk gear | **absent from frame** | The plate's dark foreground object is gear standing on the desk near the lens, not the desk top. The canonical room has the desk but not that gear yet. |
| Monitor content | flat emissive panel | `assets/screens/*.png` exist and are not yet mapped onto the 3D panels. |
| Room detail density | much lower | The plate has depth from many small objects; the canonical room has four landmarks by design. |
| Material richness | much lower | Flat Principled BSDF, no textures, no grain, no imperfection maps. |

## The honest summary

The 3D Camera 1 reproduces the **layout, the lighting direction and the wall
system**. It does not reproduce the photograph. Anyone comparing them will see
a previsualisation next to a photoreal plate.

That trade was made deliberately and is argued in
`docs/multicam_architecture.md`: the alternative - keeping the photoreal plate
and generating the other cameras - was tried in this project and produced seven
different men in seven different rooms. Geometric consistency had to come
first because nothing downstream can repair a different person.

## What would close each gap, in order of value

1. **The human.** The single dominant gap. An SVAD-shaped pipeline (single
   image -> video diffusion -> synthetic multi-view -> 3DGS avatar), or a
   photoreal scanned human re-textured toward the approved identity. Weeks, and
   genuinely risky.
2. **Framing.** Cheap: another pass matching camera distance and height against
   the plate, with the plate loaded as a reference overlay.
3. **Desk gear and monitor content.** Cheap and mechanical. The screen textures
   already exist; they need UV-mapping onto the panels rather than an emissive
   colour.
4. **Materials.** Textures, roughness variation and grain. Moderate work, large
   perceived return, no architectural risk.

Items 2-4 are ordinary work with predictable outcomes. Item 1 is a research
project. They should not be scheduled as if they were the same kind of task.

## Lock status

- `ROOM_GEOMETRY_LOCKED` - **not yet.** Framing and prop pass still pending.
- `HUMAN_IDENTITY_LOCKED` - **not yet.** Proxy only.
- `MATERIALS_LOCKED` - no.
- `CAMERAS_LOCKED` - no. cam1-3 placed and validated; cam4-7 defined, corrected
  after the floorplan caught them on the wrong side of the subject, still
  disabled.
