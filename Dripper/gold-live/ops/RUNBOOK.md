# Runbook

What to do when something is wrong at 3am. Symptom first, because that is what
you have.

Start here every time:

```
GoldLive.exe doctor          what is broken right now
GoldLive.exe dashboard       per-session health, and the last thing each said
```

---

## The host has gone silent

**Check `/ready` for the session.** Ports start at 9101 and increment per session.

```
curl localhost:9101/ready
```

| | Meaning |
|---|---|
| 200 | The session thinks it is fine. The problem is downstream — audio, or the platform. |
| 503 | The session knows it is unwell. `/health` says which component. |
| no response | The process is gone or wedged. The supervisor should be restarting it. |

**If the process is up and `/ready` is 200 but nobody hears anything,** the fault
is in the audio path, not the AI. Check in this order:

1. `GoldLive.exe doctor` — is the virtual cable still present? A Windows update
   or a reconnected USB device can reassign it.
2. Is LIVE Studio's **microphone** still set to the cable? Not desktop audio —
   desktop audio sends every system sound to your audience.
3. `curl localhost:9101/metrics | grep first_audio` — if TTS is producing audio,
   the break is between the cable and LIVE Studio.

**If it is genuinely generating nothing,** the model server is usually down.
`doctor` names every port it checked.

---

## It is repeating itself

Expected in one specific case: **the model is unreachable and it has fallen back
to the scripted topic list.** Check `fallback_used` on the dashboard — if it is
climbing, fix the model server and the repetition stops.

If the model is up and it still repeats, that is a real fault. Capture a
transcript before restarting:

```
curl localhost:9101/metrics | grep blocked_total
```

A high `reason="repetition"` count means the gate is working and generation is
producing similar text — a model or prompt problem. A low count with obvious
repetition means the gate is not catching it, which is a bug worth reporting
with examples.

---

## It quoted a price that looks wrong

It should be structurally impossible for it to quote a price on stale data.
Verify:

```
curl localhost:9101/health | grep -o '"confidence=[a-z]*"'
```

If confidence is `live`, the price came from the feed and the feed is wrong —
a provider problem. If it is anything else and a price was still spoken, that
is a **serious bug in the safety gate**. Stop the session and report it with the
trace id:

```
curl localhost:8080/api/explain/<trace_id>
```

---

## A session is restarting over and over

The supervisor gives up after 6 failures in an hour and leaves it DOWN
deliberately — a crash loop that restarts silently is worse than one that
stops, because nobody finds out. The dashboard shows `crash_loop`.

Read the actual error:

```
journalctl --user -u goldlive-session@SESSION_003 -n 100     # Linux
Get-Content "$env:LOCALAPPDATA\GoldLive\logs\*.log" -Tail 100 # Windows
```

Common causes, in order of likelihood: a config edit with invalid YAML, the
capture calibration pointing off-screen after a monitor change, and the model
server refusing connections.

---

## Comments stopped arriving

The panel has almost certainly moved — a LIVE Studio update, a window resize, a
resolution change. The adapter reports DEGRADED after a few minutes of silence
because that is the only signal available: it produces no error, just nothing.

```
GoldLive.exe calibrate --session SESSION_001
GoldLive.exe calibrate --session SESSION_001 --verify
```

Always run `--verify`. A crop ten pixels too narrow clips the author name off
every row and still looks like it is working.

---

## The market feed keeps dropping

Reconnects are normal and handled — backoff, then retry, while the host stops
quoting prices. Watch the rate rather than individual events:

```
curl localhost:9101/metrics | grep market_staleness
```

Sustained staleness above ~15s means the host has stopped quoting levels and is
talking about concepts instead. That is correct behaviour, not a bug — but it
is a worse stream, so chase the provider.

---

## The machine rebooted

Everything should come back on its own: the logon task or systemd units restart
the sessions, and each one resumes from its last checkpoint knowing what it had
already covered. Confirm:

```
GoldLive.exe dashboard
```

Look for `resumed:` in the logs. If sessions log `starting with no prior state`
after a reboot, checkpointing is broken and they will repeat old material.

---

## Stopping everything, now

```
Stop-ScheduledTask -TaskName GoldLive                    # Windows
systemctl --user stop 'goldlive-session@*'               # Linux
```

SIGTERM lets each session checkpoint and flush before exiting. Killing them
outright means the restarted sessions forget what they covered and repeat it.

---

## Before you restart anything

Restarting hides evidence. If the fault is not obvious, capture first:

```
curl localhost:9101/health  > health.json
curl localhost:9101/metrics > metrics.txt
curl "localhost:8080/api/status" > status.json
```

The trace store keeps every utterance with the market state, trigger and model
that produced it, so `/api/explain/<trace_id>` answers "why did it say that?"
long after the fact.
