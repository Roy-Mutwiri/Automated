# Installing GoldLive on Windows

## Requirements

| | |
|---|---|
| OS | Windows 10 or 11, 64-bit |
| Disk | ~3 GB free (model 2 GB, voice 60 MB, working space) |
| RAM | 8 GB minimum, 16 GB comfortable |
| Network | Needed for setup, and for the live gold price while running |
| Python | **Not required.** It is inside the application. |

GoldLive runs the AI model on the processor. A graphics card is not used in
this version and is not required.

## Steps

1. Download `GoldLive-Windows-x64-v<version>.zip` from the Releases page.
2. Right-click the ZIP → **Extract All**.
3. Open the extracted `GoldLive` folder.
4. Run **`GoldLive Setup.exe`**.
5. Follow what it tells you. It ends at `READY TO START` or names what is missing.
6. Open **`GoldLive.exe`** and press **START**.

### SmartScreen

The executables are not code-signed yet, so Windows may show
*"Windows protected your PC"*. Choose **More info → Run anyway**. This will
stop once the build is signed; until then it is expected, and worth being
suspicious of in general.

## Two things you install yourself

**Ollama** — <https://ollama.com> — runs the AI model. GoldLive will not
install it for you because it registers a background service, and installing
services on your machine without asking is not something this application does.

**VB-CABLE** — <https://vb-audio.com/Cable> — only needed for broadcasting.
It is a kernel-mode audio driver. Same reasoning: you install it.

Setup detects both and tells you clearly which, if either, is missing.

## Verifying an installation

```
GoldLive.exe doctor              what is installed and what is missing
GoldLive.exe ready               can a session actually start right now
GoldLive.exe selftest --imports  is this build complete
GoldLive.exe paths               where your files live
```

`GoldLive.exe ready` exits 0 when a session can start and 1 when it cannot,
so it is usable from a script.

## Updating

Replace the extracted folder. Your settings, model, voices and logs live
under your user profile and are untouched. Run `GoldLive Setup.exe` once
afterwards — it revalidates and repairs, and downloads nothing that is
already intact.

## Uninstalling

1. Press **STOP** in the control panel.
2. Delete the extracted folder.
3. Delete `%LOCALAPPDATA%\GoldLive` to remove the model, voices, logs and settings.

Nothing else is left behind. GoldLive installs no service, no scheduled task,
no registry entries and no Startup shortcut, so there is nothing hidden to
clean up.
