# Gold Live — setup

A 24/7 AI host that talks about Gold (XAUUSD) on a live stream. It watches the
market, reads viewer comments, decides when there is something worth saying,
and says it out loud.

Everything runs on your machine. Nothing is sent to a third party.

---

## What you need

| | Required? | Why |
|---|---|---|
| A model server | **Yes** | Writes what the host says. Not bundled — it is tens of GB. |
| Piper + a voice | For audio | Turns text into speech. Without it you get a placeholder tone. |
| VB-CABLE | For streaming | Carries audio into LIVE Studio. |
| A market feed | For live prices | Without one, the host talks about concepts and never quotes a price. |

Only the model server is genuinely required. Everything else degrades rather
than failing — you can run it and read the transcript before setting up audio.

**Hardware:** the model is the demanding part. A 7–8B model at 4-bit runs on an
8GB GPU. Larger models sound better and need more. Check with `GoldLive.exe bench`.

---

## 1. Set up

```
GoldLive.exe setup
```

Creates your configuration folder. `GoldLive.exe paths` shows where.

## 2. Start a model

**Ollama** is the easiest. Install it from ollama.com, then:

```
ollama serve
ollama pull <model>
```

That is all — Gold Live finds Ollama automatically on its usual port. You do
not need to configure a URL or type the model name.

**vLLM** is better if you plan to run several sessions at once, because it
batches requests and caches the prompt prefix:

```
vllm serve <model> --port 8000 --enable-prefix-caching
```

> On model choice: bigger sounds better, and licences vary a lot. Some models
> forbid commercial use. If you are sharing this on, check the licence of the
> model you recommend — see THIRD_PARTY_NOTICES.md.

## 3. Check everything

```
GoldLive.exe doctor
```

Reports each dependency. **FAIL** blocks; **warn** means it runs with that
capability missing. Fix the FAILs and ignore the warns until you need them.

## 4. Hear it

```
GoldLive.exe dryrun
```

Runs a simulated market session and writes transcripts and audio to your data
folder. **Read the transcript before setting anything else up** — if the output
is not interesting, no amount of audio routing will fix that, and that is the
thing worth knowing early.

## 5. Audio

1. Download a voice from the Piper voices list into the `voices` folder next to
   the exe. Each voice is an `.onnx` plus an `.onnx.json`.
2. Install **VB-CABLE**.
3. In LIVE Studio, set the **microphone** to `CABLE Output`.

> Do **not** point LIVE Studio at desktop audio. That sends every system sound
> to your audience — notifications, alerts, a stray browser tab. The virtual
> cable exists so only the host's voice goes out.

```
GoldLive.exe run --session SESSION_001 --tts piper
```

## 6. Comments

```
GoldLive.exe calibrate --session SESSION_001
```

Drag a box around the comment panel — the names and the messages, nothing else.
Then check it actually reads them:

```
GoldLive.exe calibrate --session SESSION_001 --verify
```

Re-run this after any LIVE Studio update that moves the panel. The system tells
you when that has happened: the dashboard shows the session as degraded after a
few minutes with no comments read.

## 7. Run it

```
GoldLive.exe supervise          all configured sessions, with restarts
GoldLive.exe dashboard          http://127.0.0.1:8080
```

---

## Customising

Everything is in your config folder (`GoldLive.exe paths`):

- **`configs/personas/*.yaml`** — who each host is. Differentiate by *audience
  and timeframe*, not by accent. Several hosts saying the same thing in
  different voices is worse than one host.
- **`configs/sessions.yaml`** — which sessions exist. Adding one is a config
  entry, never a code change.
- **`configs/content.yaml`** — fallback topics for when the model is
  unreachable. Not the content plan; the host normally generates its own.

Edits survive upgrades — config lives outside the program folder.

---

## When something is wrong

| Symptom | Cause |
|---|---|
| Host says nothing | No model server. `GoldLive.exe doctor`. |
| Audio is a beeping tone | Piper missing or no voice file. |
| Audience hears nothing | LIVE Studio is not listening to the virtual cable. |
| No comments picked up | Panel moved. Re-run `calibrate`. |
| Never quotes a price | Working as intended — no live market feed, so it refuses to invent one. |
| Repeats itself | Expected on the fallback topic list. Means the model is unreachable. |

`GoldLive.exe dashboard` shows every session's health and the last thing each
one said.

---

## What it will not do

- **Give trading advice.** Asked "should I buy?", it redirects to process. This
  is enforced in code, not by instruction, and cannot be configured off.
- **Quote a price it is not sure of.** If market data is stale, price quoting is
  disabled automatically and it says so.
- **Claim certainty.** Outcome-certainty language is blocked before anything is
  spoken.

These are deliberate. It is financial content going to an audience, and being
confidently wrong is the failure that actually costs people money.

Before streaming publicly, check your platform's rules on unattended broadcasts
and AI-generated voices, and your own jurisdiction's rules on financial
promotion. Neither is something this software can decide for you.
