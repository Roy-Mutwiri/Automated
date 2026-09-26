# GoldLive

An AI host that talks about the live gold market — real prices, real
generated commentary, real synthesised speech.

**GoldLive never starts on its own.** It runs only while you have pressed
START, and STOP always stops it. No service, no scheduled task, no registry
entry, no Startup shortcut.

There are two ways in, and they are not the same thing:

| | Who it is for | What you need |
|---|---|---|
| [**Install the Windows build**](#install-the-windows-build) | anyone who wants to run GoldLive | nothing — Python is inside the app |
| [**Run it from source**](#run-it-from-source) | anyone changing the code | Python 3.12+, a checkout |

---

## Install the Windows build

1. Download **`GoldLive-Windows-x64-v0.5.0.zip`** from the
   [Releases page](https://github.com/Roy-Mutwiri/Automated/releases/latest).
   Direct link once v0.5.0 is published:
   [`GoldLive-Windows-x64-v0.5.0.zip`](https://github.com/Roy-Mutwiri/Automated/releases/download/v0.5.0/GoldLive-Windows-x64-v0.5.0.zip)
   (~103 MB)
2. Right-click the ZIP → **Extract All**
3. Open the extracted `GoldLive` folder and run **`GoldLive Setup.exe`**

Setup checks your PC, downloads the AI model and voice, and proves each piece
works before reporting `READY TO START`. Then open **`GoldLive.exe`** and press
**START**. **STOP** always stops it.

The ZIP is built and published by CI from the tagged commit — the same bytes
that were tested, never a hand-made upload. If the Releases page is empty,
no version has been tagged yet; build it yourself with the steps below.

Full instructions: [docs/INSTALL_WINDOWS.md](docs/INSTALL_WINDOWS.md) ·
[docs/USER_GUIDE.md](docs/USER_GUIDE.md)

## Run it from source

For development. This gives you the engine and the test suite, not an
installed application.

```bash
git clone https://github.com/Roy-Mutwiri/Automated.git
cd Automated/Dripper/gold-live

python -m venv .venv
.venv/Scripts/python -m pip install -e ".[dev]"    # Windows
# .venv/bin/python -m pip install -e ".[dev]"      # Linux/macOS

.venv/Scripts/python -m pytest -q                  # the suite
.venv/Scripts/python -m runtime.dryrun --mode offline   # the pipeline, no model
```

`--mode offline` needs no model server, no audio device, no network and no API
key. Output lands in `out/<SESSION_ID>/`.

To produce the same ZIP that Releases carries:

```bash
.venv/Scripts/python -m build_tools.build --skip-tests
.venv/Scripts/python -m build_tools.make_zip
# -> dist/GoldLive-Windows-x64-v0.5.0.zip
```

Building the Windows executable requires Windows. Those are the commands CI
runs, so they are the ones documented here.

> **Windows path length.** `pip` puts PortAudio's DLL five directories deep
> inside `site-packages`. Under a long checkout path the total exceeds
> `MAX_PATH` and the DLL fails to load with `OSError 0xce`, which looks like a
> broken install and is only a long name. Clone somewhere short (`C:\src\...`),
> or enable long paths:
> `git config --system core.longpaths true`.

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

Commands are under [Run it from source](#run-it-from-source). A bare `pytest -q`
works (see `pythonpath` in `pyproject.toml`); CI runs the same suite on Linux
and on Windows, because the Windows-only failures — rename races, `MAX_PATH`,
real device enumeration — are exactly the ones that reach users.

Architecture and design decisions: [docs/PORTABILITY_SPEC.md](docs/PORTABILITY_SPEC.md)

Release process: [docs/RELEASING.md](docs/RELEASING.md)

## Disclaimer

GoldLive discusses market scenarios. It is **not financial advice**, it does
not tell anyone when to buy or sell, and it refuses to quote a price it cannot
currently verify. Prices come from tokenised gold (PAXG/XAUT) on a public
exchange feed — close to spot, but not an official XAUUSD fix.
