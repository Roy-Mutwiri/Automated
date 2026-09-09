# Releasing GoldLive

A release is cut by pushing a tag. Nothing is built or uploaded by hand: the
bytes on the Releases page are the bytes CI tested, and the only way to publish
is to tag a commit that passed.

## The one command

```bash
git tag v0.5.0
git push origin v0.5.0
```

`.github/workflows/ci.yml` does the rest:

| Job | Runner | What it proves |
|---|---|---|
| `version-guard` | `ubuntu-latest` | the tag matches `version` in `pyproject.toml` |
| `test` (3.12, 3.13) | `ubuntu-latest` | lint and the full suite on Linux |
| `windows-test` | `windows-latest` | the same suite on the platform that ships |
| `integration` | `ubuntu-latest` | dry run and accelerated soak, end to end |
| `build` | `windows-latest` | PyInstaller build, ZIP, and the packaged exe runs |
| `release` | `ubuntu-latest` | publishes the artifact `build` produced |
| `release-published` | `ubuntu-latest` | a tag that published nothing fails loudly |

## Version numbers live in one place

`pyproject.toml` is the single source of truth (`shared/version.py` reads it,
and `build_tools/make_zip.py` names the archive from it). A tag that disagrees
with it fails `version-guard` in seconds, before the Windows build starts —
because the archive would be named after the pyproject version and would
contradict the release it was attached to.

To release a new version, bump `pyproject.toml` first, commit, then tag.

## What "publish" is allowed to mean

- `build` runs under `contents: read` and can only upload a workflow artifact.
  A pull-request build therefore cannot publish anything, by permission rather
  than by convention.
- `release` is a separate job purely so `contents: write` is scoped to it, and
  it never rebuilds — it downloads what `build` produced.
- `gh release create --verify-tag` refuses to invent a tag, so a release can
  only ever describe a commit somebody tagged.
- After publishing, the workflow reads the release back and fails if the ZIP is
  not attached. A release with no download is worse than no release, because
  the install docs point straight at it.
- `release-published` runs with `if: always()` on any `v*` tag. If `release` was
  skipped — because a job it needed failed — the run goes red and says so. A
  green run with an empty Releases page is the one outcome this file exists to
  prevent.

## Verifying a release by hand

```bash
gh release view v0.5.0 --repo Roy-Mutwiri/Automated
gh release download v0.5.0 --repo Roy-Mutwiri/Automated
```

Extract it and run the entry point a user would double-click:

```powershell
.\GoldLive\GoldLive.exe selftest --imports
```

That is the same check CI runs against the extracted archive. It exists because
presence is not proof: an earlier layout shipped both executables at the top of
the folder with the runtime in `app/`, so every structural check passed while
the exe the README told people to double-click died at launch with *"Failed to
load Python DLL"*.

## Requirements outside this repository

GitHub Actions must be enabled and the repository owner's account must not be
in a billing-locked state. Standard runners (`ubuntu-latest`, `windows-latest`)
are free for public repositories and this workflow uses nothing else — no
larger runners, no runner groups, no self-hosted labels, no GitHub Packages, no
paid marketplace actions. If jobs report *"The job was not started because your
account is locked due to a billing issue"*, that is account-level and is cleared
in GitHub billing settings; no change to this workflow can affect it.
