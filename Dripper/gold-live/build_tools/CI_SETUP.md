# Enabling CI

`ci.yml` in this directory is the workflow. It is not at `.github/workflows/`
yet because pushing a file there requires a GitHub token with `workflow` scope,
which the token on this machine does not have.

Two ways to enable it, either is fine.

## Via the GitHub web UI (no scope needed)

1. Open the repository on github.com
2. **Add file → Create new file**
3. Name it `.github/workflows/ci.yml`
4. Paste the contents of `Dripper/gold-live/build_tools/ci.yml`
5. Commit

## Via the CLI (grants the scope once)

```
gh auth refresh --hostname github.com --scopes workflow
git mv Dripper/gold-live/build_tools/ci.yml .github/workflows/ci.yml
git commit -m "Enable CI"
git push
```

## What it does

- **test** — lint and the full suite on Python 3.12 and 3.13
- **contracts** — fails a PR that edits `shared/contracts.py` without bumping
  `SCHEMA_VERSION` or regenerating schemas. That file is the interface between
  both halves of the system; drift there is invisible until integration.
- **mocks** — fails if a platform interface ships with no test double, which
  would block the other side of the project
- **build** — on `main`, builds the Windows distributable and uploads the zip

`.github/CODEOWNERS` and the PR template are already in place and need no
special scope — they take effect as soon as branch protection is turned on in
repository settings.
