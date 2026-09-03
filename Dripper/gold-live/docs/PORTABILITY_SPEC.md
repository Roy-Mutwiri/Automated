# Portable / Self-Installing GoldLive — Implementation Specification

Status: **for review, not implemented.** Derived from the cold-start audit (2 Sep 2026).

Product requirement: a user takes GoldLive to a supported clean Windows PC, launches
it, and the application guides and provisions that machine until it is *genuinely*
ready to run SESSION_001. Subsequent launches are fast and re-download nothing.

Governing principles, in priority order:

1. **Readiness is proven by execution, not by presence.** Every gate does the real
   thing once. Every defect found in this project so far was found by executing
   rather than reading; the readiness design encodes that lesson.
2. **Installation state is cached; liveness is never cached.** This is what makes
   the fast path fast without making it dishonest.
3. **Degradation is opt-in, not automatic.** Today every subsystem degrades
   silently. That is why a clean PC produces a beeping shell with no error.
4. **Never silently install system-level components.** Drivers and services are
   detected and explained, never installed behind the user's back.

---

## 1. Stage specifications

### STAGE 1 — BOOTSTRAP

| | |
|---|---|
| **Responsibility** | Resolve paths, seed config, load provisioning state, decide fast path vs full provisioning. Nothing else. Must complete in well under a second on the fast path. |
| **Inputs** | `argv`, environment (`GOLDLIVE_DATA_DIR`, `GOLDLIVE_SUPERVISED`), `resource_root()`, existing `provisioning.json` |
| **Outputs** | A `BootstrapResult`: `{data_root, state, needs_provisioning: bool, reason}` |
| **Persists** | Seeded config files (unchanged behaviour). Creates `provisioning.json` with `state_version` and nothing else if absent. |
| **Failure conditions** | Data directory not writable; `resource_root()` missing bundled config; `provisioning.json` unreadable *and* unbackupable |
| **Fatal?** | Data dir not writable → **fatal** (nothing can proceed). Corrupt state file → **recoverable**: back up to `provisioning.json.bad`, treat as first run. |
| **User sees** | Fast path: nothing. First run: `Gold Live — first time setup on this PC.` |
| **Integrates with** | `shared/paths.py` (unchanged), `runtime/bootstrap.py::seed_config` (unchanged), new `shared/provisioning.py` |
| **Unchanged** | `resource_path` / `data_path` / `config_path` semantics, the `SEED_FILES` / `SEED_DIRS` mechanism, the `GOLDLIVE_DATA_DIR` override |

### STAGE 2 — CAPABILITY PROBE

| | |
|---|---|
| **Responsibility** | Measure the machine once. Produce a profile that later stages *decide* from. Collect nothing that does not drive a decision. |
| **Inputs** | The OS, `shutil.disk_usage`, `sounddevice` device list, a network probe, a market-feed probe |
| **Outputs** | `HardwareProfile` (§3) + a `profile_hash` over its *stable* fields only |
| **Persists** | The profile and hash into `provisioning.json`. Re-probed when the hash's inputs change or on `--reprobe`. |
| **Failure conditions** | GPU enumeration unavailable; audio subsystem absent; network unreachable; market feed unreachable |
| **Fatal?** | All **recoverable** and recorded as facts, not errors. Unsupported OS or architecture → **fatal**. A probe that throws must yield `unknown`, never crash the launch. |
| **User sees** | A plain-English summary: `Windows 11 · 8 cores · 16 GB RAM · no usable GPU for AI (will use the processor) · 40 GB free · speakers found · internet OK · gold feed reachable (82 ms)` |
| **Integrates with** | `platform_/audio/devices.py::list_output_devices`, `find_virtual_cable`; `platform_/market/exchange_feed.py` constants for the reachability probe |
| **Unchanged** | The audio device module's API; the market feed itself (probe uses its URL, does not modify it) |

The probe never blocks longer than a hard 10 s budget in total. Anything slower is
recorded as `unknown` with a reason.

### STAGE 3 — DEPENDENCY MANAGER

| | |
|---|---|
| **Responsibility** | Determine, for each of the four dependency classes (§4), whether the requirement is met. Fetch nothing itself — downloads belong to stage 4. |
| **Inputs** | `HardwareProfile`, the static `BUNDLED` manifest compiled into the build, the state file |
| **Outputs** | `DependencyReport`: per dependency `{name, class, status, version, detail}` where status ∈ `ok / missing / unusable / skipped` |
| **Persists** | The report into `provisioning.json` under `dependencies`, with `detected_at` |
| **Failure conditions** | A **bundled** dependency fails to import; a **user-guided** dependency is absent; an **optional** one is absent |
| **Fatal?** | Bundled import failure → **fatal and non-repairable at runtime**. It means the download is damaged or the build is broken. GoldLive must say exactly that and stop. It must never attempt `pip install` at runtime. User-guided absent → **recoverable**, blocks specific readiness gates. Optional absent → **not a failure**. |
| **User sees** | For user-guided items only: what it is, why it is needed, the official link, and the one-line check to confirm. Then a re-check prompt. |
| **Integrates with** | `runtime/bootstrap.py` check functions (extended, not replaced), `platform_/llm/discovery.py` for model-server detection |
| **Unchanged** | The `Check` dataclass and its rendering; the existing check functions' remediation text, which is already good |

### STAGE 4 — MODEL / VOICE MANAGER

| | |
|---|---|
| **Responsibility** | The only stage that moves large files. Select, download, verify, and record models and voices. Resumable and idempotent. |
| **Inputs** | `HardwareProfile`, the model catalogue (`configs/models.yaml`), the configured personas' voices, the state file's `artifacts` |
| **Outputs** | Verified artifacts on disk; `artifacts` entries with sha256 and `verified_at`; a recorded model choice with measured tokens/sec |
| **Persists** | Per artifact: `{kind, id, path, bytes, sha256, source, licence, verified_at}`. Plus `selected_model` and its benchmark. |
| **Failure conditions** | No consent; insufficient disk; download interrupted; checksum mismatch; licence not permissive; Ollama pull fails; no catalogue entry fits the machine |
| **Fatal?** | All **recoverable**. Checksum mismatch → delete and retry once, then stop and explain. No fitting model → **fatal for readiness** but not for the process: GoldLive states honestly that this machine cannot run it acceptably. |
| **User sees** | A consent prompt naming exact sizes before anything downloads, then per-artifact progress, then `verified`. |
| **Integrates with** | `scripts/get_voices.py` — its download, `MODEL_CARD` fetch and licence-audit logic is extracted into a reusable core and promoted to a real subcommand. Ollama's `/api/pull` for weights. |
| **Unchanged** | The voice `CATALOGUE` structure and the persona→voice mapping; the licence-audit policy (public-domain voices only for the shipped personas) |

### STAGE 5 — HEALTH CHECK

| | |
|---|---|
| **Responsibility** | Answer one question: *is this machine genuinely able to run SESSION_001 right now?* Distinct from stage 3, which asks whether things are installed. |
| **Inputs** | The provisioned state, plus live probes of market, model, TTS and audio |
| **Outputs** | A `Readiness` verdict (§9): `SESSION_READY`, `BROADCAST_READY`, or `NOT_READY` with the failing gates named |
| **Persists** | `readiness.last_ready_at` and the per-gate results with timestamps |
| **Failure conditions** | Any gate in §9 failing |
| **Fatal?** | **Recoverable**, but it *blocks* `run` and `supervise` unless `--allow-degraded` is passed explicitly |
| **User sees** | Each gate as it passes, with the evidence: `Live gold price: 4381.82 (live) ✓`, `Model wrote 41 words in 1.4 s ✓`, `Voice spoke a test line, 2.1 s of audio ✓`, `You should have just heard it ✓` |
| **Integrates with** | `runtime/bootstrap.py::doctor` — `doctor` becomes the human-readable face of this stage; the gates are the machine-readable core |
| **Unchanged** | `doctor`'s output format, the fatal/warn distinction, `HealthState` and `ServiceHealth` contracts, the per-session `/ready` endpoint |

### STAGE 6 — RUNTIME

| | |
|---|---|
| **Responsibility** | Spawn and supervise session processes correctly in both source and frozen mode. Refuse to spawn when not ready. |
| **Inputs** | `sessions.yaml`, the readiness verdict |
| **Outputs** | Running session processes; restart/backoff bookkeeping |
| **Persists** | Nothing new. Existing per-session state checkpointing is unchanged. |
| **Failure conditions** | Not ready; spawn fails; crash loop threshold reached; `/ready` 503 beyond the grace period |
| **Fatal?** | Not ready → **fatal for the supervisor's start**, by design. Crash loop → the existing give-up policy, unchanged. |
| **User sees** | Either the readiness failure and how to fix it, or normal supervisor logs |
| **Integrates with** | `runtime/supervisor.py` — one new `session_command()` helper and a corrected `cwd`; a pre-spawn readiness call |
| **Unchanged** | Restart policy, backoff and jitter, crash-loop give-up, SIGTERM-then-kill, `/ready` hang detection, health ports, `ManagedSession` |

### STAGE 7 — SESSION

| | |
|---|---|
| **Responsibility** | Unchanged. Market feed, Director, generation, safety gates, TTS, audio routing, comments, checkpointing. |
| **Inputs / Outputs / Persists** | Unchanged |
| **Failure conditions** | Unchanged |
| **Fatal?** | Unchanged |
| **User sees** | Unchanged, plus one new line at startup naming the readiness level it started under |
| **Integrates with** | Receives `GOLDLIVE_SUPERVISED=1` so it skips its own provisioning |
| **Unchanged** | **Everything.** This specification proposes no change to the market engine, conversation engine, event contracts, TTS architecture, or session architecture. The only addition is the `--allow-degraded` flag and refusing to start without it when not ready. |

---

## 2. Provisioning state file

Location: `data_root() / "provisioning.json"`. Written atomically (temp file +
`os.replace`). Never partially written. Corruption is survivable by design.

```json
{
  "state_version": 1,
  "goldlive_version": "0.5.0",
  "contracts_schema_version": "1.0.0",
  "bundle_id": "sha256:9f2c...",

  "first_run": {
    "started_at": "2026-09-02T14:10:02Z",
    "completed_at": "2026-09-02T14:31:44Z",
    "completed": true
  },

  "hardware": {
    "profile_hash": "sha256:1a7b...",
    "probed_at": "2026-09-02T14:10:07Z",
    "os_name": "Windows 11 Pro",
    "os_build": "10.0.26200",
    "architecture": "AMD64",
    "cpu_model": "AMD Ryzen 5 3600",
    "cpu_physical_cores": 6,
    "cpu_logical_cores": 12,
    "ram_total_gb": 16.0,
    "ram_available_gb": 9.4,
    "gpu_vendor": "AMD",
    "gpu_model": "Radeon RX 580",
    "vram_gb": 4.0,
    "gpu_usable_for_inference": false,
    "gpu_unusable_reason": "Polaris architecture; ROCm support withdrawn",
    "disk_data_total_gb": 465.0,
    "disk_data_free_gb": 39.8,
    "audio_output_devices": ["Speakers (Hisense HS2100)"],
    "virtual_cable": null,
    "network_online": true,
    "market_feed_reachable": true,
    "market_feed_latency_ms": 82,
    "model_server": { "kind": "ollama", "version": "0.33.2", "url": "http://127.0.0.1:11434/v1" }
  },

  "dependencies": {
    "websockets":  { "class": "bundled",   "status": "ok",      "version": "13.1",  "detected_at": "..." },
    "piper":       { "class": "bundled",   "status": "ok",      "version": "1.2.0", "detected_at": "..." },
    "sounddevice": { "class": "bundled",   "status": "ok",      "version": "0.4.7", "detected_at": "..." },
    "ollama":      { "class": "guided",    "status": "ok",      "version": "0.33.2","detected_at": "..." },
    "vb_cable":    { "class": "guided",    "status": "missing", "version": null,    "detected_at": "...",
                     "detail": "no virtual cable among 1 output device(s)" },
    "paddleocr":   { "class": "optional",  "status": "skipped", "version": null,    "detected_at": "..." }
  },

  "artifacts": {
    "model:llama3.2:3b": {
      "kind": "model", "path": "<ollama-managed>", "bytes": 2019377152,
      "digest": "sha256:8eeb...", "source": "ollama://llama3.2:3b",
      "licence": "Llama 3.2 Community License",
      "verified_at": "2026-09-02T14:29:10Z",
      "benchmark": { "tokens_per_s": 11.4, "measured_at": "...", "device": "cpu" }
    },
    "voice:en_US-john-medium": {
      "kind": "voice", "path": "voices/en_US-john-medium.onnx", "bytes": 63201792,
      "sha256": "4c1f...", "source": "https://huggingface.co/rhasspy/piper-voices/...",
      "licence": "public domain", "verified_at": "2026-09-02T14:27:55Z"
    }
  },

  "selection": {
    "model_id": "llama3.2:3b",
    "tier": "baseline",
    "device": "cpu",
    "reason": "no usable GPU; 9.4 GB available RAM; measured 11.4 tok/s",
    "max_recommended_sessions": 1
  },

  "last_provision": {
    "attempt": 3,
    "started_at": "...", "finished_at": "...",
    "failed_stage": null,
    "failed_reason": null
  },

  "readiness": {
    "last_ready_at": "2026-09-02T14:31:40Z",
    "level": "session_ready",
    "gates": {
      "market_live":   { "ok": true,  "at": "...", "evidence": "4381.82 confidence=live" },
      "model_real":    { "ok": true,  "at": "...", "evidence": "41 words in 1412 ms" },
      "voice_real":    { "ok": true,  "at": "...", "evidence": "2.1 s audio, peak 0.94" },
      "audio_out":     { "ok": true,  "at": "...", "evidence": "Speakers (Hisense HS2100)" },
      "broadcast_route": { "ok": false, "at": "...", "evidence": "no virtual cable" },
      "comments":      { "ok": false, "at": "...", "evidence": "paddleocr not installed" }
    }
  }
}
```

**Why each field exists** — mapped to the questions it must answer:

| Question | Field |
|---|---|
| What GoldLive version is installed? | `goldlive_version` |
| What schema version? | `state_version` (this file) and `contracts_schema_version` (the data) |
| Has first-run provisioning completed? | `first_run.completed` |
| What dependencies are installed? | `dependencies.*.status` |
| What versions? | `dependencies.*.version` |
| What models? | `artifacts` where `kind == "model"` |
| What voice models? | `artifacts` where `kind == "voice"` |
| What checksums were verified? | `artifacts.*.sha256` / `.digest` |
| When was each last verified? | `artifacts.*.verified_at` |
| What hardware was detected? | `hardware` |
| What step failed? | `last_provision.failed_stage` / `failed_reason` |

`bundle_id` is the hash of the build's dependency lock. It is how an updated
application knows whether its *bundled* set changed, independently of the version
string.

**Idempotency contract.** Provisioning is a function of (state, profile, catalogue).
Running it twice with nothing changed must perform zero downloads and zero writes
beyond timestamps. Every step checks its own postcondition before acting.

---

## 3. Hardware capability profile

Every field below drives a decision. Nothing is collected for information.

| Field | Why it exists — the decision it drives |
|---|---|
| `os_name`, `os_build` | Supported-platform gate; some dependencies need Win10 1809+. Also makes bug reports actionable. |
| `architecture` | A bundle built for x64 cannot run on arm64. Wrong arch is fatal and must be said clearly, not discovered as a crash. |
| `cpu_model` | Instruction-set support (AVX2) for CPU inference; identifies known-slow parts. |
| `cpu_physical_cores` | Sets inference thread count and caps concurrent sessions on CPU. |
| `cpu_logical_cores` | Distinguishes SMT from real cores so thread count is not over-set. |
| `ram_total_gb` | Upper bound on model tier when inference is on CPU. |
| `ram_available_gb` | The real constraint at provisioning time; a 16 GB machine with 3 GB free cannot load a 5 GB model today. |
| `gpu_vendor`, `gpu_model` | Determines whether an inference backend exists at all for this part. |
| `vram_gb` | Model tier ceiling when on GPU. |
| `gpu_usable_for_inference` | **The actual decision input.** A GPU can be present, capable on paper, and still unusable — this machine's RX 580 is Polaris, which ROCm dropped. Recording only vendor/VRAM would produce a wrong recommendation. |
| `gpu_unusable_reason` | Told to the user, so "it's using my processor" is explained rather than mysterious. |
| `disk_data_free_gb` | Pre-flight for downloads; must be measured on the *data* drive, which `GOLDLIVE_DATA_DIR` may relocate. |
| `disk_data_total_gb` | Distinguishes "small disk" from "full disk" in the message. |
| `audio_output_devices` | Whether any real audio is possible; without one, TTS can only write files. |
| `virtual_cable` | Broadcast routing. Its absence is the difference between SESSION_READY and BROADCAST_READY. |
| `network_online` | Whether downloading is possible at all; changes the first-run flow. |
| `market_feed_reachable`, `market_feed_latency_ms` | **The most product-specific probe.** The system's core input is a WebSocket to Binance. Corporate, captive and geo-filtered networks fail here and nowhere else, and today that surfaces only as an endless reconnect loop. |
| `model_server` | Whether Ollama is already installed and serving, and at what version — decides whether stage 3 guides the user or skips ahead. |

**Deliberately not collected:** screen resolution (calibration is interactive and
handles it), MAC address or machine serial (no purpose, privacy cost), installed
software inventory, user identity.

**`profile_hash` covers only stable fields:** OS, architecture, CPU model, core
counts, total RAM, GPU model, VRAM, virtual-cable presence. It deliberately excludes
free disk, available RAM, network state and latency — otherwise the hash changes on
every launch and the fast path never triggers.

---

## 4. Dependency classification

### Class 1 — Bundled inside GoldLive

Pinned in the lock, forced into `hiddenimports`, verified by post-build selftest.

| Dependency | Why bundled |
|---|---|
| `pydantic` (+submodules) | Every contract. Already bundled. |
| `pyyaml` | Session, persona and content config. Already bundled. |
| **`websockets`** | The gold market feed. Pure Python, tiny. Without it there is no live price and therefore no product. Its absence is one of the two hard blockers. |
| **`piper` (+`onnxruntime`)** | The voice. Without it every utterance is a placeholder tone. The second hard blocker. |
| `httpx` | LLM client and REST feeds. |
| `numpy` | Audio buffers and the similarity index. |
| `sounddevice`, `soundfile` | Audio output and WAV I/O. Native PortAudio/libsndfile DLLs ship via `collect_dynamic_libs`. |
| `mss` | Screen capture, needed by `calibrate` even before OCR exists. |

Common rationale: all are small, licence-clean, and each is a hard dependency of a
core capability. Bundling removes an entire class of user-machine failure. None
require elevation or system modification.

### Class 2 — Downloaded automatically (with consent, checksum, resume)

| Dependency | Why downloaded rather than bundled |
|---|---|
| Piper voice models | ~60 MB each and **licences differ per voice** — the existing audit already caught a non-commercial voice downloaded by mistake. A blanket bundle would ship licence risk. Only the configured personas' voices are fetched. |
| LLM weights (via Ollama's API) | Gigabytes, and the correct choice depends on the target machine's hardware. Bundling one model would be exactly the hard-coding this spec forbids. |
| *(later)* PaddleOCR models | Only if OCR is opted into. |

### Class 3 — Detected and user-guided (never silently installed)

| Dependency | Why the user must act |
|---|---|
| **Ollama runtime** | Installs a background service and modifies PATH. It persists after GoldLive is deleted. Detect, explain, link, then re-check. |
| **VB-CABLE** | A **kernel-mode audio driver** behind a UAC prompt. Installing a driver on someone's machine without their knowledge is not acceptable under any framing. Detect, explain what it is for, link, and then explain the LIVE Studio microphone setting — because installing it and not pointing Studio at it is the common failure. |

GoldLive may *offer to open the download page*. It may not download and execute an
installer on the user's behalf.

### Class 4 — Optional

| Dependency | Why optional |
|---|---|
| `paddleocr` | Drags a large ML stack. Comments work through the file adapter without it. Explicit opt-in via `provision --with-ocr`. |
| `anthropic` | Hosted-model comparison only; never used in production. Move from base dependencies to an extra. |
| `redis` | **Remove from `pyproject.toml` entirely.** It is declared and imported nowhere the runtime uses. The event contracts remain bus-ready — that property was verified separately and is what actually matters — but the client library is dead weight in the dependency surface. |

---

## 5. Model selection

The catalogue lives in **`configs/models.yaml`**, not in code, mirroring the voice
catalogue. It can be updated without a rebuild.

```yaml
tiers:
  - id: llama3.2:1b   tier: minimum     bytes: 1.3e9  ram_gb: 3   vram_gb: 2
  - id: llama3.2:3b   tier: baseline    bytes: 2.0e9  ram_gb: 6   vram_gb: 4
  - id: qwen2.5:7b    tier: good        bytes: 4.7e9  ram_gb: 12  vram_gb: 8
  - id: qwen2.5:14b   tier: best        bytes: 9.0e9  ram_gb: 24  vram_gb: 12
```

**Selection algorithm**

1. **Choose the device.** GPU if `gpu_usable_for_inference` — which is a derived
   flag, not merely "a GPU exists". Otherwise CPU.
2. **Apply the memory budget.** GPU: model ≤ 80 % of VRAM. CPU: model ≤ 60 % of
   *available* RAM, leaving headroom for the OS, TTS and the audio pipeline, which
   run in the same machine and are not free.
3. **Apply the disk budget.** Require `bytes × 1.3` free, so a pull cannot fill the
   disk mid-download.
4. **Measure, do not assume.** Pull the highest tier that fits, then run a real
   generation benchmark once and record tokens/sec. This is where
   `scripts/bench_llm.py` — written but never run — becomes load-bearing.
5. **Enforce the latency budget.** If measured p95 for a typical 40-word utterance
   exceeds the target, drop one tier and re-measure. At most one downgrade per
   provisioning run, to bound first-run time.
6. **Report sessions honestly.** `max_recommended_sessions = floor(budget /
   per_session_footprint)`, capped by core count. On a CPU-only machine this will
   often be 1, and saying so plainly is the entire point.

**The three concepts the requirement asks to distinguish**

- **Minimum to function** — tier `minimum`. Below this GoldLive states that this
  machine cannot run it acceptably rather than delivering a bad experience quietly.
- **Recommended for this machine** — the highest tier passing steps 2–5.
- **Multi-session capable** — a tier whose footprint × N fits the budget. Prefix
  caching makes the shared persona prompt cheap, which is why the per-session
  marginal cost is lower than the first session's — but only if the persona prompt
  stays byte-stable, which the existing architecture already guarantees.

No model is hard-coded anywhere. The only constant is the catalogue file.

---

## 6. First launch — the user journey

Written for someone who has never opened a terminal.

1. **Double-click `GoldLive.exe`.** *(Until the build is signed, Windows shows
   "Windows protected your PC" → More info → Run anyway. This is documented in the
   README and is the strongest argument for code signing.)*
2. **A window opens:** `Gold Live — first time setup on this PC. This takes about
   15 minutes, mostly downloading.`
3. **Checking this PC** *(~10 s)* — prints the plain-English profile from §3.
4. **What's needed** — a checklist with ✓/✗ and exact sizes:
   ```
   ✓ Everything built into Gold Live
   ✗ An AI model                 2.0 GB   will download
   ✗ A voice                      60 MB   will download
   ✗ Ollama (runs the AI model)           you install this — I'll show you
   ✗ VB-CABLE (sends audio to TikTok)     you install this — I'll show you
   ```
   `Download 2.1 GB now? [Y/n]`
5. **Two things you install yourself** — one at a time, each with what it is, why
   it is needed, the official link, and `Press Enter when done and I'll check.`
   GoldLive re-checks and does not advance until present. VB-CABLE may be skipped;
   skipping is recorded and means broadcast is not ready.
6. **Downloading** — per-artifact progress, resumable, checksum verified on
   completion.
7. **Testing everything for real** — the readiness gates, each printing its
   evidence as it passes:
   ```
   Live gold price ........ 4381.82  (live)          ✓
   AI model ............... wrote 41 words in 1.4 s  ✓
   Voice .................. spoke a test line, 2.1 s ✓
   Speakers ............... you should have heard it ✓
   TikTok audio route ..... no virtual cable          ✗
   ```
8. **Result:**
   ```
   SESSION_001 is ready to run.
   Broadcasting is not set up yet (VB-CABLE missing).

   Start it with:  GoldLive.exe run --session SESSION_001
   Start it now? [Y/n]
   ```

At no point does the user need to know what a virtualenv, a checksum or a WebSocket
is. At every point they are told what will happen before it happens.

---

## 7. Subsequent launches — the fast path

Target: **under 2 seconds** to a readiness decision.

1. Read `provisioning.json`. Missing or corrupt → full first run.
2. Compare `state_version`, `goldlive_version`, `contracts_schema_version`,
   `bundle_id`. Any mismatch → the update path (§8), **not** a reinstall.
3. Recompute `profile_hash` from cheap sources only. Match → skip the full probe
   entirely.
4. **Verify artifacts cheaply:** existence, byte size, mtime. Compute sha256 only
   when size or mtime changed, when `verified_at` is older than 30 days, or on an
   explicit `--verify`. A full hash of a 2 GB model on every launch would defeat the
   purpose.
5. **Run the live gates anyway.** Market, model and TTS liveness are properties of
   *now*, not of installation. They take seconds, and caching them would reintroduce
   exactly the dishonesty this design exists to remove.

The invariant, stated once: **installation state is cached; liveness is never
cached.**

---

## 8. Repair

Universal rules: repair is idempotent; never destructive without a backup; every
attempt is recorded in `last_provision`; attempts are bounded — after the limit
GoldLive stops and explains rather than looping.

| Situation | Detected by | Action | Fatal? |
|---|---|---|---|
| **Model deleted** | Ollama no longer lists it | Re-pull. Consent required again if > 500 MB. | Recoverable |
| **Voice corrupted** | Size or sha256 mismatch | Delete and re-download that one file. Second failure → stop and report. | Recoverable |
| **Bundled dependency missing** | Import fails inside the frozen process | **Stop.** The download is damaged or the build is broken. `"This copy of Gold Live is incomplete — please download it again."` Never attempt a runtime install. | **Fatal** |
| **Config file invalid** | YAML/JSON parse error | Back up to `<name>.bad`, re-seed from the bundled default, tell the user which file and what was preserved. | Recoverable |
| **Application updated** | `goldlive_version` or `bundle_id` mismatch | Run state migrations, re-verify artifact checksums, re-run gates. **Re-download nothing whose checksum still matches.** This is the case the current design gets silently wrong. | Recoverable |
| **Hardware changed** | `profile_hash` mismatch | Re-probe, re-evaluate the model tier. If the installed model no longer fits, recommend a different one and **keep the old until the new is verified**. | Recoverable |
| **Network unavailable** | Probe fails | If already provisioned: only the live market gate fails — say precisely that, not "setup failed". If not yet provisioned: first run cannot proceed; offer retry. | Recoverable |
| **Model server not running** | Connection refused on the known ports | If Ollama is installed *and* the user consented previously, attempt `ollama serve`. Otherwise instruct. | Recoverable |
| **Audio / VB-CABLE missing** | Not among output devices | BROADCAST_READY fails, SESSION_READY unaffected. Explicit message distinguishing the two. | Recoverable |

---

## 9. Readiness — what READY actually means

This is the core of the specification. **GoldLive must never report ready when it
can only produce canned text, placeholder audio, or synthetic market data.**

Two levels, because the distinction is real and the current code conflates them.

### SESSION_READY — the host can genuinely speak real words about real prices

All five gates must pass, each **by executing the real thing**:

| Gate | Passes only when | Not sufficient |
|---|---|---|
| `market_live` | A real feed is connected **and** a tick has been received **and** `MarketState.confidence is MarketConfidence.LIVE` **and** `may_quote_price()` is `True`, within the last 30 s | `websockets` importing; the feed object existing; `synthetic`/`replay` feeds — these **fail** this gate by definition |
| `model_real` | A discovered server serves a **named** model **and** a real completion returned ≥ 20 words within the latency budget | HTTP 200 from `/v1/models`; a non-empty model list alone; the offline fallback generator |
| `voice_real` | `piper` imports, the persona's voice file passes checksum, and a real synthesis produced non-silent audio of plausible duration | `shutil.which("piper")`; the file existing; the placeholder tone generator |
| `audio_out` | An output device exists and a test buffer played without error | Devices being enumerable |
| `config_valid` | `sessions.yaml` parses and defines SESSION_001 with a persona whose voice is provisioned | The file existing |

### BROADCAST_READY — additionally, it can actually go live on TikTok

| Gate | Passes only when |
|---|---|
| `broadcast_route` | A virtual cable is present **and** selected as the session's output device |
| `comments` | A comment adapter is connected and has read at least one line — the file adapter satisfies this today; OCR will later |

### Enforcement

- `run` and `supervise` **refuse to start** unless SESSION_READY, printing the
  failing gates and their remedies.
- `--allow-degraded` overrides this. It is explicit, it is logged loudly at startup,
  and the session banner states which capabilities are simulated.
- Anything that today silently falls back — synthetic market, offline generator,
  placeholder tone — becomes reachable **only** through that flag.
- Gate results are never cached across launches. They are cheap and they are about
  the present.

The practical consequence, stated plainly: some users will correctly be told
"not ready". That is the requirement working, and it makes the messaging quality a
first-class concern rather than an afterthought.

---

## 10. Packaging — a build that contains what it claims

The root cause is that the spec **probes the build virtualenv** with a
`try: __import__(package) except ImportError: skip` loop. The bundle therefore
depends on the developer's incidental machine state.

1. **`build_tools/requirements.lock`** — fully pinned with hashes, generated by
   `uv pip compile` (or `pip-compile`), committed to the repository.
2. **`pyproject.toml`** — declares the real runtime set; `redis` removed;
   `anthropic` and `paddleocr` moved to extras.
3. **The spec reads a static `BUNDLED` list.** No probing, no try/except. A package
   in that list that is missing at build time **fails the build loudly**. Shipping
   less than declared must be impossible, not merely unlikely.
4. **`build.py` validates the environment** against the lock before invoking
   PyInstaller and refuses to build from a mismatched venv.
5. **Post-build selftest** — `dist/GoldLive/GoldLive.exe selftest --imports` imports
   every name in `BUNDLED` *inside the frozen process* and exits non-zero on any
   failure. This is precisely the check that would have caught `websockets` and
   `piper`, and it is the single highest-value item in this specification.
6. **`bundle_id`** — the lock's hash, compiled in, recorded in `provisioning.json`.

Result: the same source and the same lock produce the same dependency set on any
build machine, and a build that would ship a silently reduced product cannot
complete.

---

## 11. Runtime — spawning sessions in both modes

The audit proved `GoldLive.exe supervise` fails on its first spawn:
`sys.executable -m runtime.live` becomes `GoldLive.exe -m …` → `Unknown command: -m`.

One helper, one place:

```
session_command(session_id, health_port, args) -> list[str]

  frozen:  [sys.executable, "run", "--session", <id>, "--health-port", <port>, *args]
  source:  [sys.executable, "-m", "runtime.live", "--session", <id>, "--health-port", <port>, *args]
```

Plus three corrections:

- **`cwd`** → `data_root()`, not `ROOT`. When frozen, `ROOT` resolves inside the
  temporary `_MEIPASS` directory, which is the wrong place to run from and is
  deleted on exit.
- **`GOLDLIVE_SUPERVISED=1`** in the child environment, so a spawned session skips
  its own provisioning and readiness gates — those belong to the supervisor, and
  running them N times in parallel would be both slow and wrong.
- **Pre-spawn readiness.** The supervisor runs the gates once before spawning
  anything and refuses to start when not ready. Without this, a broken machine
  produces a spawn loop of degraded sessions — today's behaviour, plus a bug.

Everything else in the supervisor is unchanged: restart policy, backoff and jitter,
crash-loop give-up, `/ready` hang detection, SIGTERM-then-kill.

---

## 12. OCR and comments — where they belong

Not the immediate focus. Their place in the final architecture:

- **Stage 3** classifies `paddleocr` as class 4 (optional), behind an explicit
  `provision --with-ocr`.
- **Stage 4** would fetch its models only under that flag.
- **Stage 5** keeps `check_ocr` as a non-fatal check, unchanged.
- **Readiness:** OCR belongs to `comments`, which is a **BROADCAST_READY** gate, not
  a SESSION_READY one. A session is genuinely ready to talk about gold without it.

**What must eventually be proven on a real LIVE Studio window** — none of which has
been demonstrated on any machine:

1. The calibrated crop region tracks the real comment panel across scrolling,
   window resizes and Studio layout changes.
2. OCR accuracy on real TikTok handles, emoji and the panel's actual font size.
3. Cross-frame deduplication — the same comment must not be re-ingested on every
   poll as it sits on screen.
4. The end-to-end latency budget: capture + OCR + classify + generate + speak must
   fit inside the window where answering a comment still feels responsive.
5. **The host must not read its own on-screen text as a viewer comment** — a
   feedback loop that no current test would catch.

---

## A. Files and modules that need to change

| File | Change |
|---|---|
| `build_tools/GoldLive.spec` | Static `BUNDLED` list; remove the venv probe; fail loudly on a missing package |
| `build_tools/build.py` | Validate venv against the lock; run the post-build selftest |
| `pyproject.toml` | Declare the real dependency set; drop `redis`; move `anthropic`/`paddleocr` to extras |
| `runtime/supervisor.py` | `session_command()`; `cwd=data_root()`; `GOLDLIVE_SUPERVISED`; pre-spawn readiness |
| `runtime/bootstrap.py` | `check_llm` via `discovery`; `check_tts` against the in-process API; add `check_imports`; `doctor` renders the readiness gates |
| `runtime/cli.py` | New `provision`, `ready`, `selftest` subcommands; first-run interception on `run`/`supervise` |
| `runtime/live.py` | `--allow-degraded`; refuse non-ready starts; banner naming the readiness level |
| `scripts/get_voices.py` | Extract download + verify + licence-audit into a reusable core |

## B. New files and modules

| File | Purpose |
|---|---|
| `shared/provisioning.py` | State file: atomic read/write, migration, corruption recovery |
| `shared/capability.py` | Hardware probe and `profile_hash` |
| `runtime/provision.py` | Orchestrates stages 1–4; the first-run and repair flows |
| `runtime/readiness.py` | The gates and the two-level verdict |
| `configs/models.yaml` | Model tier catalogue (config, not code) |
| `build_tools/requirements.lock` | Pinned, hashed dependency lock |
| `tests/test_provisioning.py`, `test_capability.py`, `test_readiness.py`, `test_spawn.py`, `test_bundle.py` | Per-stage tests |

## C. Implementation order

Ordered by what unblocks the most and what is cheapest to verify. Packaging is first
because until the bundle can be trusted, no test of anything else in the frozen build
means anything.

1. **Packaging determinism** — lock, static `BUNDLED`, fail-loud spec, `selftest --imports`. Clears both hard blockers.
2. **Supervisor spawn fix** — small, isolated, immediately verifiable against `dist/`.
3. **Readiness gates** + the `check_llm` / `check_tts` corrections. Define READY before anything is allowed to claim it.
4. **Provisioning state** — the file every later stage writes to.
5. **Capability probe.**
6. **Model / voice manager** — reuse the `get_voices` core; add the Ollama pull and the benchmark.
7. **Provision orchestrator, CLI wiring, first-run UX.**
8. **Repair paths.**
9. **OCR** — separate work, after a session has been proven end to end.

## D. Tests required after each stage

| Stage | Tests |
|---|---|
| 1 Packaging | `selftest --imports` passes in the frozen build; the spec **fails** when a `BUNDLED` package is uninstalled; two builds from the same lock produce the same import set |
| 2 Spawn | `session_command()` returns the right argv in both modes; `GoldLive.exe supervise` actually starts a session in `dist/`; `cwd` is writable |
| 3 Readiness | Each gate fails correctly when its subsystem is stubbed out; synthetic feed **fails** `market_live`; the offline generator **fails** `model_real`; the placeholder tone **fails** `voice_real`; `run` refuses without `--allow-degraded` |
| 4 State | Atomic write survives a simulated crash mid-write; corrupt file is backed up and treated as first run; round-trip preserves every field; migration from `state_version` 0 |
| 5 Probe | Profile parses on this machine; `profile_hash` is stable across launches and changes when a stable field changes; every probe failure yields `unknown` rather than an exception |
| 6 Models | Selection picks the right tier for fabricated profiles (low RAM, no GPU, big GPU); checksum mismatch triggers exactly one retry; a resumed download completes correctly |
| 7 Orchestrator | Full first run against stub servers; second run performs zero downloads; `--verify` forces re-hashing |
| 8 Repair | Each of the nine rows in §8 as its own test |

Existing suite (355 tests) must stay green throughout. Every new stage follows the
stub-server pattern that has already found five defects invisible to in-process fakes.

## E. Clean-PC acceptance test

On a fresh Windows VM with **no Python, no Ollama, no VB-CABLE, no voices**:

1. Copy `dist/GoldLive/`. Run `GoldLive.exe`. → First-run flow starts unprompted.
2. Follow the prompts exactly as written, installing nothing not offered. → Reaches SESSION_READY.
3. `GoldLive.exe run --session SESSION_001`. → **A human hears real generated speech about a real gold price.** This is the acceptance criterion; nothing else substitutes for it.
4. Close and relaunch. → Ready in **under 2 seconds**, zero downloads.
5. Delete a voice `.onnx`, relaunch. → Detects, re-downloads only that file, ready again.
6. Disconnect the network, relaunch. → Reports *"live gold price unavailable"* specifically, not "setup failed".
7. Stop Ollama, relaunch. → Reports the model server precisely and offers the fix.
8. Install VB-CABLE, relaunch. → Advances to BROADCAST_READY with no other change.
9. `GoldLive.exe supervise`. → Sessions actually start and stay up.

## F. Risks, and what not to do yet

**Risks**

- `onnxruntime` arrives with `piper` and is a large native wheel — bundle size and clean-machine CPU-provider behaviour must be verified early, in step 1.
- `sounddevice`/`soundfile` DLLs may still need the VC++ redistributable on a bare Windows install. Test on a genuinely clean VM, not a developer machine.
- The Binance WebSocket may be geo-blocked or throttled for some users. The reachability probe surfaces it honestly, but **no fallback free feed has been identified** — this is an open product risk, not merely a technical one.
- Benchmarking during provisioning adds time to first run. Cap it hard.
- A strict readiness bar means some users will legitimately be told "not ready". That is correct, and it makes message quality a first-class concern.
- First-run downloads are ~2 GB. On a slow connection this is a long wait that must be resumable and clearly communicated, or it will be interpreted as a hang.

**Do not do yet**

- Do not sign the code yet — it costs money and needs a decision. Document the SmartScreen path meanwhile.
- Do not auto-install Ollama or VB-CABLE. Ever, without explicit per-install consent.
- Do not build a GUI installer (MSI/Inno). Prove the console flow first.
- Do not bundle `paddleocr`.
- Do not build auto-update. Detect a version change and migrate; downloading updates is a separate decision.
- Do not build multi-session provisioning. Profile and prove one session first — the capability probe will report an honest `max_recommended_sessions`, and on the current development machine that number is likely 1.
- Do not introduce Docker, Kubernetes, Postgres or Redis. Nothing in this specification needs them.
