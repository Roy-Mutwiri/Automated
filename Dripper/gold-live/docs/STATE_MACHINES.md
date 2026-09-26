# State machines

Seven of them, deliberately separate. They are documented together because the
failure mode that matters is not any one of them being wrong — it is two of
them disagreeing, so that the UI shows RUNNING while the session is dead, or
health reports OK while nothing can be produced.

Nothing here is invented. Every state below exists in the implementation.

**Audio-only.** There is no animation, avatar or lip-sync layer, and none is
planned in this milestone. The synchronization surface is:

```
MODEL → UTTERANCE → TTS → AUDIO QUEUE → PLAYBACK → INTERRUPTION → COMPLETION → STATE
```

---

## 1. `RunState` — did the user ask for this?

`runtime/lifecycle.py`. Persisted atomically to `lifecycle.json`. This is the
consent record, and it is the only thing permitted to authorise a session.

| State | Meaning | May the supervisor run? |
|---|---|---|
| `stopped` | Idle. The default, and the answer whenever intent is unknown. | no |
| `start_requested` | The user pressed START; readiness is running. | **yes** |
| `running` | A supervisor owns the run; its pid is recorded. | **yes** |
| `stop_requested` | STOP pressed; shutdown in progress. | no |
| `crashed` | Was running, supervisor process is gone. | no |
| `error` | The last start failed, with a reason. | no |

```
stopped ──START──> start_requested ──supervisor up──> running
   ^                      │                              │
   │                      │                            STOP
   │                      ▼                              ▼
   └──────────── stop_requested <─────────────────────────┘
   │
   ├── crashed   (running, but the pid is gone — found on load)
   └── error     (readiness gates failed at startup)
```

**Invalid by construction:** `may_run` is true for exactly
`{start_requested, running}`. Every other state, including `crashed`, denies
permission. A crash does not authorise a restart in this milestone — the user
is shown a START button instead.

**Persistence:** every transition is written atomically (temp file, fsync,
`os.replace`). An unreadable file is read as `stopped`, because when consent
cannot be determined the safe answer is no.

**Recovery:** `load()` reconciles. `running` with a dead pid becomes `crashed`,
and the correction is written back — without that the panel rediscovered the
same crash every 1.5 s forever.

---

## 2. `SessionState` — what the supervisor believes about a child

`runtime/supervisor.py`. In memory only; the supervisor is the authority while
it lives. Previously bare strings, now an enum, because a typo would have
created a new state silently and every comparison against it would have
returned False.

| State | Entry | Exit |
|---|---|---|
| `starting` | spawned | first successful `/ready` |
| `running` | `/ready` returned 200 | process exits, or `/ready` fails |
| `unready` | `/ready` failing | recovers, or `unready_grace_s` elapses → killed |
| `stopped` | terminated deliberately, or consent withdrawn | respawn |
| `crash_loop` | `crash_loop_threshold` failures in an hour | manual intervention only |

`crash_loop` is terminal on purpose. A loop that restarts silently is worse
than one that stops, because nobody finds out.

---

## 3. `PlaybackState` — the audio router

`platform_/audio/router.py`. In memory, one per session.

```
idle ──submit──> synthesising ──audio ready──> playing ──┐
 ^                    │                           │      │
 │                    └──── failure / empty ──────┤      │
 └────────────────────────────────────────────────┴──────┘
                              stopped  (router shut down)
```

**The invariant that matters:** the router must never be left claiming it is
speaking. `_speak` is wrapped in `try/finally`, and the `finally` clears the
current utterance and returns to `idle` after success, TTS failure, playback
failure and empty synthesis alike. A stuck `playing` is the state a wedged host
reports forever.

**Interruption:** preemption is recorded per utterance id in `_preempted`, not
as a shared flag. `sink.stop()` alone aborts only the segment currently
sounding — the loop then continued to the next one, so a preempted utterance
kept talking underneath whatever interrupted it. The id is checked between
segments and before each play, and discarded when the utterance ends.

---

## 4. `HealthState` — per component

`shared/contracts.py`. Ordered: `ok < degraded < failing < down`. The endpoint
reports the worst component.

| State | Meaning | `/ready` |
|---|---|---|
| `ok` | working | 200 |
| `degraded` | working, impaired — e.g. a stale market feed | **200** |
| `failing` | cannot do its job | 503 |
| `down` | not running | 503 |

`degraded` returning 200 is deliberate and load-bearing. A session with a stale
feed is behaving correctly: it has stopped quoting prices and is still
broadcasting. Restarting it would be strictly worse. Only `failing` and `down`
invite the supervisor to act.

Components published: `tasks`, `generation`, `market_engine`, `audio_router`,
`<adapter>_comment_adapter`, `session`.

---

## 5. `Level` — startup readiness

`runtime/readiness.py`. Recomputed every launch, never cached.

| Level | Requires |
|---|---|
| `not_ready` | any core gate failing |
| `session_ready` | `config_valid`, `market_live`, `model_real`, `voice_real`, `audio_out` |
| `broadcast_ready` | + `broadcast_route`, `comments` |
| `full_live_ready` | + a human confirmed a viewer heard it |

`full_live_ready` is never inferred. No local check can prove a viewer heard
anything, so it is set only by explicit human verification and is otherwise
absent.

---

## 6. `ProvisionState` — is this machine installed?

`shared/provisioning.py`. Persisted atomically to `provisioning.json`.

```
new ──> provisioning ──> ready
             │             │
             ▼             │
          failed <─────────┘  (repair)
```

`degraded` is deliberately **not** a persisted provisioning state. Whether a
session can run right now depends on the network, the model server and the
audio device — none of which this file knows about. Installation state is
cached; liveness never is.

---

## 7. Panel state — what the user sees

`runtime/panel.py`. Derived, never stored. Computed from provisioning state,
then recorded intent, then live health — in that order.

| Panel | Derived from |
|---|---|
| `NOT INSTALLED` | provisioning incomplete |
| `STOPPED` | `RunState.stopped` |
| `STARTING` | `start_requested`, or `running` with no health yet |
| `RUNNING` | `running` **and** health answers |
| `DEGRADED` | `running` and health reports degraded/failing |
| `STOPPING` | `stop_requested` |
| `ERROR` | `crashed` or `error` |

**The rule:** `RUNNING` requires intent and health to agree. A process existing
is not evidence — a process that is alive and mute is a failure this project
has already been bitten by.

---

## Cross-machine invariants

These are the disagreements worth testing, and each has a test:

1. The supervisor may spawn only when `RunState.may_run`.
2. `STOP` withdraws permission immediately, not when the current session ends.
3. A dead supervisor process cannot leave `RunState.running` readable.
4. The panel cannot show `RUNNING` without live health.
5. The router cannot remain `playing` after any terminal outcome.
6. `degraded` never turns `/ready` to 503; `failing` always does.
7. `full_live_ready` is unreachable without explicit human verification.
