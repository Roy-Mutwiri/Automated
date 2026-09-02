# Didimo: commercial and technical due diligence

Reviewed 2026-09-02, from public documentation only. **Nothing has been
uploaded.** No identity asset has left this machine.

**This is an engineering reading of public contract text, not legal advice.**
Quoted clauses are from Didimo's published terms; anything not quoted is marked
as an open question rather than inferred.

## Verdict, up front

**NO-GO for upload on the current public terms.**

The blocker is not the synthetic-identity question, which is genuinely
unanswered. It is **ownership and licence scope**, and on the face of the
published agreement it fails four of our nine requirements outright. Those
terms may be variable by Order Form - that is exactly what to ask - but they
cannot be assumed away, and an upload cannot be undone.

## Service status

Active. Cloud pipeline, single image in, rigged digital human out, described as
accepting "inputs as simple as a selfie". Portal at `app.didimo.co`, developer
docs at `developer.didimo.co`.

Note for the decision tree: **Meshcapade is gone.** Epic Games acquired it in
February 2026, the team moved into Epic's AI Research division to work on
Unreal and MetaHuman, and the platform has been shut down. It is correctly
removed from consideration. A side effect worth tracking: Epic now owns the
SMPL-adjacent tooling, which makes the MetaHuman route more interesting later,
not less.

## Input requirements

| | |
|---|---|
| Images needed | One. "Inputs as simple as a selfie." |
| Our situation | One frontal synthetic portrait, 1344x768, face box 207x247 px |
| Fit | **Good** - this is the one requirement Didimo clearly satisfies |

This is why Didimo was preferred over Avaturn, which expects a frontal plus two
side photographs we cannot produce without fabricating them.

## Synthetic identity acceptance

**UNANSWERED IN PUBLIC DOCUMENTATION. Must be asked before any upload.**

The terms are written entirely around real people. Section 4.5 requires the
Customer warrant it has "obtained prior, explicit consent from the User or any
other third parties", and ownership of the Initial Image is addressed as
"as between Didimo and Customer, all rights in any Initial Image shall belong
to Customer".

Neither sentence tells us whether their **face validator** accepts a rendered
synthetic portrait, nor whether their policy permits fictional identities and
virtual presenters. Those are different questions and both need a yes.

Our position is clean and should be stated plainly to them: the Initial Image is
an original synthetic identity created for this project, generated locally with
SDXL under CreativeML Open RAIL++-M, depicting no real person. It must never be
represented as a photograph of a real individual.

## Commercial rights - where it fails

| # | Requirement | Finding | Status |
|---|---|---|---|
| 7 | We own the generated asset | "Didimo shall own all intellectual property rights in and to the Didimo API **and all Animation Files**" | **FAIL** |
| 2 | Export / receive a full character package | Licence is to "store the Animation Files on its own servers" and "use the Animation Files in connection with any Customer Online Platform"; **"Customer shall not make the Animation Files available for download by Users"** | **RESTRICTED** |
| 8 | Distribute an application containing the character | Same no-download clause; "non-transferable... non-sublicensable" | **FAIL by default** |
| 1 | Monetised livestream use | "Customer shall not license and/or exploit the Animation Files for any reason whatsoever beyond sharing on Users' personal social media profiles... **without Didimo's approval**" | **REQUIRES WRITTEN APPROVAL** |
| 4 | Modify: retopology, materials, hair, rig | Not addressed. Licence grants *store* and *use*, not *adapt* | **NOT GRANTED ON ITS FACE** |
| 5 | Derivative asset rights (Didimo head + our body) | Not addressed; follows from ownership sitting with Didimo | **OPEN - assume no** |
| 3 | Offline runtime after generation | Not addressed; licence is tied to "Customer Online Platform" | **OPEN** |
| 9 | Rendered video unrestricted | Not addressed. The exploitation clause is about Animation Files; whether renders are separately free is unstated | **OPEN** |
| 6 | Data retention / training opt-out | Didimo may retain the Initial Image "for machine learning purposes", perpetual, surviving termination, and is "authorized by its privacy policy to retain such personal data... **for which no consent is required**" | **RETAINED, NO DOCUMENTED OPT-OUT** |

### The structural problem, stated plainly

These terms describe a **consumer avatar SaaS**: a Customer runs an Online
Platform, each of the Customer's *Users* generates their own didimo from their
own selfie, and it stays inside that platform.

Our use is the opposite shape in every dimension. One avatar, not many. Our
identity, not a user's. A persistent brand presenter, not a per-user asset.
Rendered offline in our own runtime, not served from a platform. Modified
heavily. Potentially shipped inside a distributed application.

That mismatch is why so many rows above are FAIL or OPEN. It is not a
technicality to be argued around; it means the standard agreement is not
written for us, and anything workable would have to be a negotiated Order Form.

## Technical - largely unverified

The public "Digital Human Specification" page is marketing-level. It confirms
single-image input and claims "high-fidelity animation and control" but does
**not** itemise:

- output formats beyond a reference to `.fbx` ("a 3-D animation file in an .fbx
  or other format")
- package contents: mesh, skeleton, skin weights, textures, blendshapes, eyes,
  teeth
- polygon counts or LODs
- **ARKit blendshape compatibility** - critical for the behaviour terminal
- body rig, joint hierarchy, finger support, IK compatibility
- **hair and beard handling** - identity-critical for this subject, who has a
  full beard and thick hair

Deeper specs exist behind `developer.didimo.co/llms.txt` and the SDK docs and
should be read before any commercial conversation, so the technical questions
can be asked once rather than twice.

## Pricing

**CONTACT SALES REQUIRED.** No public pricing. The agreement refers throughout
to "the fees specified in each Order Form", 30 days net. No per-avatar,
per-credit or volume figures are published, and none should be invented.

## Open questions to put to Didimo

Ordered so the deal-breakers come first. If the first three are not
satisfactory, the rest do not matter.

1. **Ownership.** The terms state Didimo owns all Animation Files. For a single
   avatar used as our own brand presenter, can ownership or a perpetual,
   irrevocable, sublicensable licence be granted by Order Form?
2. **Monetised livestreaming.** Is use as a persistent presenter in monetised
   live broadcast (TikTok LIVE, YouTube LIVE, OBS) permitted, and on what terms
   - business plan, per-avatar fee, royalty or revenue share?
3. **Synthetic identity.** Does the pipeline, and the policy, accept an
   AI-generated photorealistic portrait as the Initial Image? A clear yes or no.
4. **Offline runtime.** After generation, may the exported avatar run entirely
   in our own application with no per-frame cloud dependency?
5. **Modification.** May we retopologise, re-shade, replace hair and beard,
   re-rig, and combine the head with our own body mesh?
6. **Derivative works.** What rights attach to the result of (5)?
7. **Application distribution.** May a compiled desktop application containing
   the avatar be distributed, given the no-download clause?
8. **Rendered output.** Are stream recordings, uploads, clips and screenshots
   unrestricted once rendered?
9. **Data.** Can our Initial Image be excluded from machine-learning retention?
   Can we require deletion? What is the retention period?

### Draft enquiry

Send this, and nothing about our implementation:

> We are building a commercial real-time virtual presenter using an original
> photorealistic synthetic identity that does not depict any real person. We
> would like to use one frontal image of that synthetic character to generate a
> single rigged digital human.
>
> Before uploading anything we need to confirm four points:
>
> 1. Does your service accept AI-generated/synthetic portraits as the Initial
>    Image, both as policy and through your face validation?
> 2. Your published API terms state that Didimo owns all Animation Files and
>    that they may not be made available for download, and restrict exploitation
>    beyond personal social media without approval. For a single brand-owned
>    avatar, can an Order Form grant us a perpetual licence to export, modify and
>    use the character in monetised livestream and video production?
> 3. After generation, can the exported character run entirely offline in our own
>    application, with no per-frame cloud dependency?
> 4. Can the Initial Image be excluded from retention for machine learning, and
>    deleted on request?
>
> We would also appreciate the technical specification of the export package:
> formats, mesh/skeleton/blendshape contents, ARKit compatibility, and how hair
> and facial hair are handled.

## GO / NO-GO

**NO-GO for upload.**

The decision tree said: proceed if synthetic identity is permitted **and**
commercial livestream rights are available **and** export/offline/modification
rights are acceptable. On public terms, the second and third are not satisfied
and the first is unanswered. So the gate is not passed, and per the rule, no
identity asset leaves this machine.

This is a "not yet, and here is exactly what to ask", not a rejection of
Didimo. Their input requirement - one image - remains the best fit of anything
surveyed, and every blocker above is contractual rather than technical, which
means it is the kind of thing an Order Form can change.

### If Didimo answers unfavourably

Return to local reconstruction research, with the MPFB body/rig foundation
preserved (`MPFB_BODY_RIG_FOUNDATION_V1`) as the body beneath whatever head we
end up with. The remaining candidates then are, in order: a commercially
licensed photoreal human asset re-shaped toward the identity; MetaHuman via
custom mesh, now more interesting because Epic has absorbed Meshcapade; and
last, LHM as a research-only benchmark to establish what is recoverable at all.
