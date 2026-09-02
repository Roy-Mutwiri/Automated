## What and why

<!-- One or two sentences. What changes, and what problem it solves. -->

## Contracts

- [ ] This PR does **not** touch `shared/contracts.py`
- [ ] It does, and: `SCHEMA_VERSION` is bumped, schemas regenerated, the other
      owner is reviewing, and this PR contains **only** the contract change

> Contract changes get their own PR. Bundling them with implementation is how
> the two halves of this system drift apart.

## Verification

<!-- What you actually ran, not what you intended to run. -->

- [ ] `PYTHONPATH=. pytest -q`
- [ ] Ran it: `python -m runtime.dryrun` / `python -m runtime.live --session ...`
- [ ] For anything touching timing, memory or content: `python -m runtime.soak --hours 24`

## Risk

- [ ] Touches the **safety gate** (price quoting, certainty language) — say how
      it was tested. This is a control, not a feature.
- [ ] Touches **session isolation** — the adversarial vocabulary test still passes
- [ ] Touches the **speech path** — no new synchronous I/O added to it
- [ ] Adds a platform interface — a mock ships in the same PR

## Notes for the reviewer

<!-- Anything you are unsure about, or deliberately left for later. -->
