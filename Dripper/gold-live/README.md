# GoldLive

An AI host that talks about the live gold market — real prices, real
generated commentary, real synthesised speech.

**GoldLive never starts on its own.** It runs only while you have pressed
START, and STOP always stops it. No service, no scheduled task, no registry
entry, no Startup shortcut.

## Install

1. Download `GoldLive-Windows-x64-v0.5.0.zip` from Releases
2. Extract it
3. Run **`GoldLive Setup.exe`**

Setup checks your PC, downloads the AI model and voice, and proves each piece
works before reporting `READY TO START`.

## Start

Open **`GoldLive.exe`** and press **START**.

## Stop

Press **STOP**.

Full instructions: [docs/INSTALL_WINDOWS.md](docs/INSTALL_WINDOWS.md) ·
[docs/USER_GUIDE.md](docs/USER_GUIDE.md)

---

## You install these yourself

| | Why | Where |
|---|---|---|
| **Ollama** | runs the AI model | <https://ollama.com> |
| **VB-CABLE** | only for broadcasting audio | <https://vb-audio.com/Cable> |

Both register system-level components, so GoldLive detects them and asks
rather than installing them behind your back.

## What is proven, and what is not

Verified on the development machine, in the packaged build:

- Provisioning from scratch, idempotent on re-run
- Real live gold price driving real AI commentary through real Piper speech
- A viewer comment producing a comment-triggered spoken reply
- START → RUNNING → STOP → STOPPED with no orphaned processes
- Recovery from a killed session (21 s) and from the model server dying (15 s)
- One-hour soak: stable threads, handles and queue depth

**Not proven — do not rely on these yet:**

- **Never installed on a clean PC.** Only this development machine.
- **Broadcasting to TikTok LIVE has never been tested end to end.**
- **Comment ingestion from a real TikTok stream is untested.** Comments work
  from a local file; screen-capture OCR is unproven.
- No soak longer than one hour.
- Multi-session operation is not built.

Current readiness level: `SESSION_READY` — the host genuinely speaks about
real prices, but nothing has been shown to reach an audience.

## For developers

```bash
pip install -e ".[dev]"
pytest -q
python -m PyInstaller build_tools/GoldLive.spec --noconfirm
python -m build_tools.make_zip
```

Architecture and design decisions: [docs/PORTABILITY_SPEC.md](docs/PORTABILITY_SPEC.md)

## Disclaimer

GoldLive discusses market scenarios. It is **not financial advice**, it does
not tell anyone when to buy or sell, and it refuses to quote a price it cannot
currently verify. Prices come from tokenised gold (PAXG/XAUT) on a public
exchange feed — close to spot, but not an official XAUUSD fix.
