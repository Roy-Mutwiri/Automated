# Enabling CI

The workflow now lives at `.github/workflows/ci.yml`, where GitHub will find
it. One step remains, and it needs a browser: the token on this machine has
scopes `gist, read:org, repo` but **not `workflow`**, so pushing a file under
`.github/workflows/` is rejected.

```
gh auth refresh --hostname github.com --scopes workflow
git push
```

If you would rather not widen the token, create the file through the GitHub
web UI instead (**Add file → Create new file**, named
`.github/workflows/ci.yml`) and paste the contents. Either way the commit is
already made locally.

## Verify it actually runs

A CI that has never failed has not been shown to work. After the first green
run, push a deliberately broken commit — delete an assert, or break an import —
confirm the run goes **red**, then revert. Until you have seen it fail, a green
badge only proves the workflow parsed.

## What it does

- **test** — lint and the full suite on Python 3.12 and 3.13
- **integration** — runs the two harnesses that were built and then never
  executed: a three-session dry run (feed → engine → director → generator →
  safety → TTS, with the cross-session isolation check active) and a 24-hour
  accelerated soak that fails on dead air, content exhaustion or repetition
  drift. Both are hermetic — mock TTS, offline generator, simulated time — so
  they need no model server, audio device, network or API key, and together
  they take well under a minute.
- **contracts** — fails a PR that edits `shared/contracts.py` without bumping
  `SCHEMA_VERSION` or regenerating schemas. That file is the interface between
  both halves of the system; drift there is invisible until integration.
- **mocks** — fails if a platform interface ships with no test double, which
  would block the other side of the project
- **build** — on `main`, builds the Windows distributable and uploads the zip.
  Gated on both `test` and `integration`: a build whose pipeline did not run
  is not a release candidate.

`.github/CODEOWNERS` and the PR template are already in place and need no
special scope — they take effect as soon as branch protection is turned on in
repository settings.
