# GoldLive — User Guide

GoldLive is an AI host that talks about the live gold market. It speaks out
loud through your PC's audio, so **you** decide when it runs.

---

## The one thing to know

**GoldLive never starts on its own.**

- Extracting the ZIP does not start it.
- Running Setup does not start it.
- Opening the control panel does not start it.
- Restarting Windows does not start it.

It runs only while you have pressed **START**, and **STOP** always stops it.
There is no service, no scheduled task, no registry entry and no Startup
folder shortcut. Nothing about GoldLive is hidden from you.

---

## Install

1. Download `GoldLive-Windows-x64-v0.5.0.zip`.
2. Right-click → **Extract All**.
3. Open the extracted `GoldLive` folder.
4. Run **`GoldLive Setup.exe`**.

Setup checks your PC, downloads the AI model (about 2 GB) and the voice
(about 60 MB), writes your configuration, and then *tests that each piece
actually works* — it plays a short tone, synthesises a real sentence, fetches
a real gold price, and asks the model to write something.

It takes roughly 15 minutes, mostly downloading, and needs about 3 GB free.

When it finishes you will see one of two things:

```
  READY TO START
```

or

```
  SETUP INCOMPLETE  --  GoldLive cannot start yet
```

Setup never reports ready when it is not. If something is missing it names
the item and what to do about it.

### Running Setup again

Safe, and expected. It re-downloads nothing that is already present and
intact, and repairs anything that is missing or corrupt. Use it whenever
something looks wrong.

---

## Start

Open **`GoldLive.exe`** and press **START**.

You will see `STARTING` for 10–30 seconds while it verifies the market feed,
the model, the voice and the audio device. It becomes `RUNNING` only once
those actually work — never merely because a process launched.

## Stop

Press **STOP**, then confirm. GoldLive shuts down cleanly, stops speaking,
closes its connections, saves its state and confirms every process is gone.

Closing the control panel window is **not** a stop. If GoldLive is running
when you close it, you are asked what you want.

---

## What the control panel shows

| State | Meaning |
|---|---|
| `NOT INSTALLED` | Setup has not completed. Run `GoldLive Setup.exe`. |
| `STOPPED` | Installed and idle. Press START. |
| `STARTING` | Verifying that everything works. |
| `RUNNING` | Live and speaking. |
| `DEGRADED` | Running, but something is impaired — check the rows below. |
| `STOPPING` | Shutting down. |
| `ERROR` | The last start failed, or GoldLive stopped unexpectedly. |

The rows underneath show each part separately, so when something is wrong you
can see *which* thing is wrong rather than a single unhelpful red light.

---

## Readiness levels

GoldLive is deliberately precise about what it has proven.

| Level | Meaning |
|---|---|
| `NOT_READY` | Cannot run a real session. |
| `SESSION_READY` | **Genuinely working**: real live gold price, real AI text, real synthesised speech, real audio device. Nothing simulated. Not reaching an audience. |
| `BROADCAST_READY` | Also routed to a device streaming software can capture. |
| `FULL_LIVE_READY` | Also confirmed **by a person** that a viewer actually heard it. |

`FULL_LIVE_READY` is never set automatically. No check on your PC can prove a
viewer heard something, so GoldLive does not pretend otherwise.

---

## Things GoldLive cannot install for you

**Ollama** — runs the AI model. It installs a background service, so you
install it yourself: <https://ollama.com>. Setup detects it and tells you if
it is missing.

**VB-CABLE** — only needed to send GoldLive's audio into streaming software.
It is a kernel audio driver, so GoldLive will not install it silently:
<https://vb-audio.com/Cable>. Afterwards, point your streaming software's
**microphone** at CABLE Output — not its desktop audio.

Without VB-CABLE GoldLive still works; it just plays to your speakers.

---

## Where your files live

Not in the folder you extracted. Settings, logs, the model and the voices go
under your user profile, so you can replace the program folder when updating
without losing anything.

```
GoldLive.exe paths
```

shows the exact locations. Logs are also reachable from the panel's **Logs**
button.

---

## Honest limitations

- **Broadcasting to TikTok LIVE has never been verified end to end.** The
  audio path beyond your PC's speakers is unproven.
- **Comment ingestion from a real TikTok stream is unproven.** Comments work
  through a local file today; screen-capture OCR exists but has never been
  tested against a real LIVE Studio window.
- **GoldLive has never been installed on a clean PC.** Everything above was
  verified on the development machine only.
- Speech is slower than conversation on a CPU-only machine: roughly 6–12
  seconds to produce an utterance.

The gold price is derived from tokenised gold (PAXG/XAUT) on a public
exchange feed. It tracks spot closely but is not an official XAUUSD fix, and
GoldLive says so rather than implying otherwise.

**GoldLive discusses scenarios. It is not financial advice and does not tell
anyone when to buy or sell.**
