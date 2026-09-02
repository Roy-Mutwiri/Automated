# Didimo: technical assessment

Reviewed 2026-09-02 from public developer documentation. **Nothing uploaded, no
account created, no API call made.** Every figure below is quoted from
`developer.didimo.co`, not inferred.

The question this answers: *if Didimo granted every right we wanted, would the
technology actually solve our identity problem?*

## Verdict

**TECHNICAL GO** — with two named gaps that we would fill ourselves, and one
unknown that no amount of documentation can settle.

The rig is better than expected and is very close to the interface I had
already proposed for the behaviour terminal. The geometry is real, shared-topology
3D. The gaps are **hair and beard**, which are preset or absent — and for this
particular man the beard is a substantial part of his identity.

## Input

One image. Didimo's pipeline takes "inputs as simple as a selfie". This is the
single requirement that made Didimo preferable to Avaturn, which wants a frontal
plus two profiles we cannot produce without fabricating them.

## Geometry

Exact counts, FBX package:

| Mesh | Vertices | Triangles |
|---|---|---|
| Head | 3,181 | 6,280 |
| Mouth + tongue | 1,060 | 2,036 |
| Eyes (both) | 290 | 544 |
| Eyelashes | 176 | 240 |
| Body | 4,957 | 9,884 |
| **Total** | **9,664** | **18,984** |

glTF is the same triangle count with slightly higher vertex counts from split
UVs (10,556).

**Six separate meshes**: head, mouth/tongue, left eye, right eye, eyelashes,
body. Real 3D eyes and a real mouth cavity with tongue, as separate objects —
exactly what the specification demanded and what the proxy faked.

**Topology is shared and parametric**: "Shared Body topology which can
blendshape between Male and Female". So the head is **generic topology deformed
toward the input**, not a free-form scan.

That is the right answer for us, not a compromise. Fixed topology means the rig,
the UVs and any corrective shapes we author transfer to a regenerated avatar,
and it means a head can be swapped or combined predictably. A free-form
reconstruction would give better raw likeness and be far harder to build a
production pipeline on.

**The concern is polygon budget.** 6,280 triangles is a game-LOD head. Camera 1
puts the face at roughly a third of frame height at 1080p. Smooth shading, a
2048 normal map and a tileable micro-normal will carry most of it, and the
topology is regular enough to subdivide, but this is not a film-resolution head
and should not be described as one.

## Skin and materials

Head and body each ship **nine** maps at 2048×2048 PNG:

Albedo · Normal · Ambient Occlusion · Cavity · Roughness · Specular ·
Translucency · Tileable Micro Normal (1024) · SSS + AO Mask

Eyes: albedo 2048 + normal 1024. Mouth: albedo, normal, AO at 512. Eyelashes:
albedo-opacity 512.

This is a **proper PBR skin setup with subsurface scattering and a micro-normal**
— materially better than I expected and well beyond a diffuse-only export.

Profiles: **Standard** is the maximum, 2048, "recommended for close-up and hero
shots". Optimized and Minimal drop to 1024 and are explicitly "not ideal for
close ups". So 2048 is the ceiling; there is no 4K tier.

**Delighting is not documented.** Whether the albedo is delit or carries the
input photo's lighting is unstated, and it matters enormously for us: a baked
cheek highlight would rotate with the head and destroy cameras 2 and 3. This is
a question for them, not something to assume either way.

## Hair — weak

- **Preset library, not reconstructed.** 12 hairstyles: 3 short, 4 medium,
  4 long.
- ~2,000 triangles each (1,684–2,280). Mobile/VR budget.
- 8 preset colours, applied by tinting a blonde base albedo in shader.
- Six maps at 1024: albedo, alpha, AO, flow, ID, ramp.

Twelve mobile-budget hairstyles will not reproduce this man's hair, and were
never going to. Hair would be our own groom in Blender regardless of vendor.

## Facial hair — absent

**No beard, moustache or stubble system appears anywhere in the documentation.**

This is the single biggest technical gap for *this* subject. Our approved
identity has a full dark beard that carries a large share of his recognisability
— it defines his jaw line, which is one of the top identity features in the
specification's own priority list.

The consequence is concrete and acceptable: we would take Didimo's head geometry
and build the beard ourselves as Blender hair curves or cards. That is additive
work on top of a good base, not a blocker — but it must be planned for, and it
means a raw Didimo output will *never* look like him out of the box.

## Face rig — strong

- **138 facial joints**, plus pose packs layered on them.
- **ARKit: 51 of the 52** BlendShapeLocation variables. The only omission is
  `tongueOut`, which "does not exist on the didimo".
- Independent `eyeBlinkLeft` / `eyeBlinkRight`.
- Jaw: `jawOpen`, `jawForward`, `jawLeft`, `jawRight`.
- Eye look: eight poses, up/down/in/out for each eye independently.
- Brows: five poses. Mouth: 26+ poses.
- Alternative packs: AWS Polly (21 poses), Oculus Lipsync (15), Simple
  (6 expressions + blinks + jaw).

**This is almost exactly the interface I proposed for the behaviour terminal.**
`docs/avatar_architecture_stages.md` sketched an ARKit-style control layer as an
internal standard; Didimo ships 51/52 of it natively. Every channel the
behaviour engine currently produces maps onto a named pose:

| `AvatarPose` | Didimo control |
|---|---|
| `eye_open_l` / `eye_open_r` | `eyeBlinkLeft` / `eyeBlinkRight` — independent |
| `gaze_x` / `gaze_y` | the eight `eyeLook*` poses |
| `brow_l` / `brow_r` / `brow_furrow` | `browOuterUpLeft/Right`, `browInnerUp`, `browDownLeft/Right` |
| `mouth_open` (reserved for lip-sync) | `jawOpen` + mouth poses |
| head `yaw`/`pitch`/`roll` | neck/head joints in the body skeleton |

## Body rig

- **70 body joints** (208 total with the 138 facial).
- "Our body rig is fully compatible with Mixamo."
- Unity: compatible with the Mecanim animation system.
- Unreal: import against the supplied Female/Male `DidimoUnrealSkeleton`.

**Finger articulation is not explicitly documented.** 70 joints is consistent
with a fully fingered humanoid rig and Mixamo compatibility effectively requires
fingers, but it is not stated, so it stays an open question rather than an
assumption.

## Export

- **FBX** and **glTF**, both fully specified with per-mesh counts.
- Unreal import yields Textures and Materials folders, a Skeletal Mesh asset, a
  Physics Asset and a Character Blueprint.
- The Asset Fitter tool operates on glTF.

**Blender is not officially documented.** It does not need to be: FBX and glTF
carrying mesh, skeleton, skin weights, blendshapes and PBR textures are exactly
what Blender imports natively. Low risk, but unverified.

## Local runtime

Not explicitly documented, and it is a contract question as much as a technical
one. Technically, an FBX/glTF with a skeleton and named blendshapes is a
self-contained asset: nothing about setting `mouthSmileLeft = 0.15` every frame
requires a server. The Unity and Unreal SDKs animate locally.

Treat as **technically yes, contractually unconfirmed** — it is question 4 in the
commercial enquiry.

## Multi-view likeness — the real unknown

**No published multi-angle examples of a single generated avatar were found.**
Front, three-quarter and profile views of the *same* didimo are exactly what
would answer our question, and the marketing shows frontal results.

This cannot be resolved from documentation. What can be said is structural: the
output is genuine 3D geometry with fixed topology, so cameras 2 and 3 are
projections of one mesh and *cannot* show a different man — which is the failure
mode that killed the earlier generated-per-camera approach. Whether the
three-quarter view *resembles him specifically* is unverified and is precisely
what a single test avatar would settle.

## Scores

| | Score | Note |
|---|---|---|
| CAM1 identity potential | **7/10** | good topology, full PBR with SSS; capped by 6.3k-tri head, no beard |
| CAM2/3 identity potential | **7/10** | consistent by construction; likeness at angle unverified |
| Face rig | **9/10** | 51 ARKit poses, 138 joints, independent blinks |
| Body rig | **8/10** | 70 joints, Mixamo compatible; fingers undocumented |
| Hair | **3/10** | 12 mobile-budget presets, not reconstructed |
| Beard | **1/10** | no system documented at all |
| Eyes | **8/10** | separate L/R meshes, 8 look poses, 2048 albedo |
| Materials | **8/10** | 9 maps at 2048 incl. SSS and micro-normal |
| Export | **8/10** | FBX + glTF, complete contents |
| Local runtime | **7/10** | standard assets; not explicitly documented |
| Behaviour terminal compatibility | **9/10** | ARKit 51 is the interface we wanted |

## Limitations, stated plainly

1. **No facial hair system.** We build the beard ourselves.
2. **Hair is 12 mobile presets.** We build the groom ourselves.
3. **2048 is the texture ceiling**; no 4K tier for hero work.
4. **6,280-triangle head** — game LOD, not film. Subdividable, but not free.
5. **Delighting unknown** — a baked highlight would break cameras 2 and 3.
6. **Finger articulation undocumented.**
7. **No multi-angle sample of one avatar published.**
8. Blender path unofficial (though standard formats).

## Why this is still a GO

The two big gaps — hair and beard — are ones we were always going to own. No
vendor was going to reconstruct this man's beard from one photograph, and our
plan already had a custom groom in it.

What Didimo supplies is the part we could *not* build: a topologically clean,
fully rigged, PBR-textured 3D head derived from our one image, with 51 ARKit
poses and 208 joints that plug straight into the behaviour engine. That is
Stage A and most of Stage B of the architecture in one asset.

And it fits the hybrid that was already on the table:

    DIDIMO head geometry + rig  →  our beard and hair groom  →  MPFB/custom body
                                →  one canonical animatable human

which is why `MPFB_BODY_RIG_FOUNDATION_V1` was preserved rather than discarded.

**Technical GO. Proceed to the commercial enquiry — which remains the hard
part, since the standard agreement is still a NO-GO on ownership grounds.**
