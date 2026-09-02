# Didimo commercial enquiry — DRAFT, NOT SENT

Status: **awaiting review. Nothing has been sent. Nothing has been uploaded.**

Send to: Didimo sales/licensing (via `didimo.co/contact`), asking to be routed
to whoever handles negotiated Order Forms.

## Requirement classification

| Requirement | Status | Why |
|---|---|---|
| Synthetic portrait accepted as Initial Image | **MUST** | No real person exists. If refused, nothing else matters. |
| Monetised livestream / recorded video use | **MUST** | This is the product. |
| Offline runtime after generation | **MUST** | No per-frame cloud dependency in a live broadcast. |
| Export of the character asset | **MUST** | Without it there is nothing to animate. |
| Modification (mesh, textures, materials, rig) | **MUST** | Hair and beard must be added by us. |
| Derivative works (head + our own body/rig) | **MUST** | The hybrid architecture depends on it. |
| Continued use of the asset after termination | **MUST** | A presenter that expires with a subscription is not a brand. |
| Commercial rights to rendered output | **MUST** | Streams and clips must be exploitable. |
| Ownership **or** a broad perpetual licence | **MUST** (either) | Practical control is what matters, not title to their platform. |
| No ML/training retention of the Initial Image | **MUST** | Their published terms currently assert this without consent. |
| Acceptable pricing, ideally fixed rather than revenue share | **MUST** (acceptable), NICE (fixed) | |
| Application bundling / distribution rights | **NICE** | Not committed to shipping software yet; need to know the option. |
| Contractor / subprocessor use | **NICE** | Useful, not blocking. |
| Confidential treatment of uploaded assets | **NICE** | Ask, do not demand an NDA up front. |
| Fully articulated fingers | **NICE** | Solvable by us if absent. |
| Delit albedo | **NICE**, high value | Correctable, but expensive if badly baked. |
| 4K facial texture | **NICE** | 2048 is workable for our framing. |
| Safe subdivision of head topology | **NICE** | |
| Beard / hair reconstruction | **NICE** | Already assumed to be our own work. |

**GREEN** = every MUST obtained. **YELLOW** = one MUST unresolved but
negotiable. **RED** = any of: synthetic identity refused, offline use refused,
commercial livestreaming refused, modification/derivatives refused, or use
ceasing on termination.

## Notes on tone

Framed as evaluation, not commitment. It does not ask for ownership of Didimo's
platform, only of the generated character *or* a licence broad enough to be
equivalent in practice. It concedes up front that hair and beard are our own
work, which removes a whole category of back-and-forth and signals we have
actually read the documentation.

Nothing about our architecture, tooling, pipeline or other vendors is disclosed.

---

## The message

**Subject:** Commercial licensing enquiry — single persistent virtual presenter from a synthetic portrait

Hello,

We are evaluating Didimo for a specific use case and would like to establish
whether it can be supported under a negotiated commercial agreement before we
proceed any further.

**What we are building.** We are developing a real-time virtual presenter based
on an original photorealistic synthetic character. The character would be a
single, persistent, branded presenter appearing in monetised livestreams and
recorded video. Our intended workflow is to generate the character once, export
it, add our own hair, facial hair, materials and animation work, and then
animate and render it locally in our own real-time pipeline.

This is deliberately not the pattern your standard API terms describe. We are
not building a platform in which many end users each generate their own avatar.
We need **one** character, under our long-term control.

**1. Synthetic identity — our first and most important question.**

Our source portrait is a fully synthetic, original photorealistic character. It
is **not intended to represent or impersonate any actual person**, and no
physical person exists from whom additional photographs, selfies, scans or video
could be captured. We hold the rights to the source artwork.

Please confirm explicitly:

- Is an AI-generated or otherwise synthetically created photorealistic human
  portrait accepted as an Initial Image, as a matter of policy?
- Will your face-validation pipeline accept such an image in practice?
- Are fictional characters and virtual presenters a permitted commercial use?

We would rather have a clear no than an inference, so please treat this as a
policy question rather than a question about whether the pipeline usually works
on selfies.

**2. Commercial rights.**

We note that the published API terms place ownership of Animation Files with
Didimo, and restrict exploitation beyond sharing on end users' personal social
media without approval. Our use case requires long-term control of the resulting
presenter, so we would like to know whether a negotiated Order Form can provide
either:

1. customer ownership of the generated character asset; **or**
2. a perpetual, irrevocable, worldwide, commercial and modifiable licence to it.

Either would work for us. Specifically, we need to be able to use the character
perpetually and worldwide in monetised livestreams, recorded and promotional
video; to download, export and store it locally; to modify it — including
retopology, materials, textures, replacement hair, added facial hair, geometry
optimisation, LODs, rig changes, corrective blendshapes and animation
retargeting; and to create derivative versions, including combining the head
with our own body and rig if that proves necessary.

Three points we would like addressed directly, because they are the ones that
would end the evaluation if unavailable:

- **Offline runtime.** Once the character is generated and exported, may it be
  animated and rendered entirely on our own hardware, with no per-frame
  dependency on Didimo services?
- **Continued use after termination.** If our agreement with Didimo later ends,
  may we continue using a character already generated under it, indefinitely?
- **Rendered output.** Are streams, recordings, clips, screenshots and
  promotional renders produced with the character ours to exploit commercially,
  without per-render fees, royalties or revenue share?

We would also like to understand whether the licence can extend to contractors
and service providers working on our behalf, and separately — as an option we
are not committing to now — whether the asset could later be embedded in a
distributed desktop application, or whether that requires a different licence.

**3. Technical confirmations.**

Your documentation indicates 138 facial joints, 51 ARKit-compatible facial
poses, independent left/right blinks, jaw and eye-look controls, and a
Mixamo-compatible body rig. Please confirm these remain available in a
commercially exported asset, and clarify four points:

- **Facial albedo and lighting.** Does the process produce a broadly
  illumination-neutral facial albedo, or is the source photograph's lighting
  baked substantially into the face texture? We will view the character from
  several physical 3D camera angles under our own lighting, so a strong baked
  directional highlight would be a problem. We are asking about the output
  characteristic, not your implementation.
- **Multi-angle consistency.** The character will be seen from three-quarter and
  profile angles as well as frontal. Are single-image avatars intended to hold
  up from those angles, and could you share sample renders of one generated
  avatar from front, three-quarter left, three-quarter right and profile?
- **Hands and texture resolution.** Does the exported body skeleton include
  fully articulated fingers with multiple joints per digit? And is 2048 the
  current maximum facial texture resolution, or is a higher-resolution
  enterprise output — or an approved workflow for us to replace the face maps
  ourselves — available?
- **Our own hair and beard.** We expect to replace the hair entirely and to
  create our own facial hair, since our character has a full beard. Is that
  supported, and does it interfere with the facial rig? We would also like to
  know whether the head topology can be safely subdivided for closer shots.

Finally, is there a documented or recommended route for importing a complete
exported character into Blender with rig, skin weights, facial poses and
materials intact — or does the standard FBX/glTF export carry everything needed?

**4. Data handling.**

Can a commercial agreement provide that our Initial Image is not used for model
training or machine-learning development, is deleted after character creation or
on request, and is subject to a defined retention period? We would also like to
know whether generated geometry and textures are used to train your systems, and
whether that can be excluded contractually. Temporary processing to deliver the
service is entirely reasonable; we simply need clarity. If commercial
confidentiality terms are available for uploaded assets, please let us know.

**5. Pricing and next steps.**

Could you outline the commercial structure for a low-volume, high-value case of
this kind — one persistent character rather than high-volume generation? We are
interested in the character generation fee, commercial licensing, any platform
or API fees, export and offline-runtime rights, and whether any recurring fees
or revenue share apply. Our preference would be a fixed licensing structure
rather than revenue share, if that is available.

We would appreciate a written response on the rights questions in particular. If
any of this requires a custom Order Form, please point us to the right contact
and we will take it from there.

Thank you,

[name] — [company]

---

*Word count: ~980.*

---

## PARKED — 2026-09-02

**Not sent, and no longer the active path.** The project was reclassified as
personal / non-commercial, so the commercial-rights blockers this enquiry exists
to resolve are not currently blockers at all.

Kept because the analysis stays valid if the project is ever commercialised: the
standard agreement's ownership clause, the no-download restriction and the
ML-retention term are all still true, and this draft already asks the right
questions in the right order.

The technical assessment in `docs/didimo_technical_assessment.md` remains
independently useful - 51 ARKit poses and 208 joints is a good benchmark to
judge any other candidate against.
