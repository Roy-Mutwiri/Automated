# Research-only dependencies — PERSONAL / NON-COMMERCIAL USE ONLY

The project is currently a **personal, non-commercial** research project, so
technologies under research/non-commercial licences are acceptable **where their
own terms permit this use**. That is a statement about today, not a permanent
position.

## Rules that still apply

- **Do not redistribute** restricted checkpoints or body models.
- **Do not claim ownership** of third-party assets.
- **Keep attribution** where a licence requires it.
- **Keep every restricted dependency documented** in
  `docs/avatar_dependency_licenses.md`.
- **Keep them isolated** from anything that might later be commercialised.
  Nothing under this directory is imported by `src/presenter/`.

## If this ever becomes commercial

The whole dependency tree gets reassessed then. The two documents that make that
possible are already written: `docs/avatar_dependency_licenses.md` records what
is restricted and why, and `docs/didimo_commercial_enquiry_draft.md` records what
a commercial route would have to ask for.

The key fact to carry forward: **SMPL-X MODEL is non-commercial** (shape
blendshapes), while **SMPL-X BODY is CC BY 4.0** and commercially usable. The
distinction falls between reconstruction and runtime, so a commercial runtime
may still be reachable even if the reconstruction stage is not.

## The identity is fictional

`assets/reference/avatar_identity_camera1.png` is an original synthetic
character. No real person is depicted, and no consent, biometric or
impersonation workflow applies unless a specific service's own terms demand one.
