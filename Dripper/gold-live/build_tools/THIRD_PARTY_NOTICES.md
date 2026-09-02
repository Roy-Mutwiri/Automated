# Third-party notices

Gold Live bundles or depends on the components below. If you distribute this
software, you must ship this file with it. Several of these licences require
attribution; some restrict commercial use.

**This file lists what is bundled. It is not legal advice, and it is not a
substitute for checking the licence of every model and voice you actually
deploy — those are the ones with real restrictions.**

## Bundled Python dependencies

| Component | Licence | Notes |
|---|---|---|
| Python | PSF-2.0 | Interpreter and standard library |
| pydantic | MIT | Contract definitions |
| PyYAML | MIT | Configuration |
| httpx | BSD-3-Clause | Model server client |
| numpy | BSD-3-Clause | Frame hashing, audio |
| sounddevice | MIT | Audio output (optional) |
| soundfile | BSD-3-Clause | WAV reading (optional) |
| libsndfile | LGPL-2.1 | Via soundfile. **Dynamic linking only.** |
| mss | MIT | Screen capture (optional) |
| PyInstaller | GPL-2.0 with exception | Build tool. The exception permits distributing the packaged output under your own terms. |

## Not bundled — installed separately by the operator

These are the ones that matter commercially, because their licences vary and
some forbid what you may be planning.

| Component | Licence | Commercial use |
|---|---|---|
| **Language model weights** | **varies enormously** | **Check per model. See below.** |
| vLLM / Ollama / llama.cpp | Apache-2.0 / MIT | Permitted |
| Piper | MIT | Permitted |
| Piper voice models | **varies per voice** | Some are CC-BY-SA, some non-commercial. Check each. |
| PaddleOCR | Apache-2.0 | Permitted |
| PaddleOCR models | Apache-2.0 | Permitted |

### Language model weights — read before selling anything

There is no single answer, and the differences are not cosmetic:

- **Llama family** — community licence, not open source. Includes an
  acceptable-use policy, an attribution requirement ("Built with Llama"), and a
  clause requiring a separate licence above 700M monthly active users.
- **Qwen, Mistral, Gemma** — each has its own terms; some releases are Apache-2.0
  and some are research-only. The licence differs *between releases of the same
  family*.
- **Any model tagged "non-commercial" or "research only"** — cannot be used in a
  product you sell, at any scale, regardless of how it is deployed.

Because the model is downloaded by the operator rather than bundled, the
obligation may fall on them rather than on you — but if you *recommend* or
*preconfigure* a specific model, that argument weakens considerably. Get advice
before shipping a default.

## Market data

Most XAUUSD feeds licence for **internal display only**. Broadcasting a live
price to a public audience is redistribution and normally requires a different,
more expensive tier. If you sell this software to operators who will use their
own feeds, make that obligation explicit in your terms — you do not want to
inherit it.

## Platform terms

This software can be configured to read comments from a live-streaming platform
by capturing the operator's own screen. Whether a given deployment complies with
that platform's terms of service depends on how it is used, and is the
operator's responsibility. See the project architecture document for the
specific findings on unattended broadcasting.
