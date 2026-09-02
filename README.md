# Automated

One repository, two contributor folders. Every change is committed and pushed
automatically, attributed to whichever contributor's folder it came from.

```
Automated/
  Anon/          <- Anon's work      (commits authored as Anon)
  Dripper/       <- Dripper's work   (commits authored as Dripper)
  tools/         <- the auto-sync watcher
```

## How attribution works

A background watcher notices any file change, waits ~2 seconds for things to
settle, then commits:

| What changed              | Commit author           |
| ------------------------- | ----------------------- |
| anything under `Anon/`    | Anon                    |
| anything under `Dripper/` | Dripper                 |
| anything else             | the shared identity     |

One save that touches both folders becomes **two commits**, one per author, so
`git log --author=Anon` cleanly separates the two. After committing, the
watcher runs `git pull --rebase --autostash` and then pushes.

Names and emails live in [`tools/identities.json`](tools/identities.json).
Edit that file to change how commits are attributed - no other change needed.

> **Do not use `@users.noreply.github.com` emails here.** GitHub resolves
> `someone@users.noreply.github.com` to whichever real account owns the login
> `someone`, which silently credits your commits to a stranger. The identities
> use the non-routable `.local` domain instead, so commits show the name `Anon`
> or `Dripper` with no account attached. That also means these commits do not
> appear in GitHub's contributor graph - only commits bound to a real account
> do.

> **Note on identity vs. access.** This attributes *authorship* by folder.
> Both folders still push through whatever credentials this machine has, so
> this is a bookkeeping split, not an access-control boundary. If Anon and
> Dripper are genuinely separate people who should not be able to overwrite
> each other, give each their own clone and GitHub account instead.

## Running the watcher

```powershell
# install: start automatically at every logon
powershell -ExecutionPolicy Bypass -File tools\autostart.ps1

# start it right now without waiting for a logon
Start-ScheduledTask -TaskName AutomatedRepoSync

# stop it
Stop-ScheduledTask -TaskName AutomatedRepoSync

# uninstall
powershell -ExecutionPolicy Bypass -File tools\autostart.ps1 -Remove
```

Run it in the foreground instead (Ctrl+C to stop):

```
tools\sync.cmd
```

Useful flags:

| Command                    | Effect                                     |
| -------------------------- | ------------------------------------------ |
| `sync.ps1 -Once`           | one commit+push pass, then exit            |
| `sync.ps1 -NoPush`         | commit locally, never push                 |
| `sync.ps1 -DebounceSeconds 5` | wait longer for changes to settle       |

Activity is logged to `tools/sync.log` (gitignored).

## Things worth knowing

- **Nothing is reviewed.** Every save lands on `main` and goes straight to the
  remote. That is the point, but it means a broken or secret-containing file is
  pushed the moment it is written. Keep secrets out of both folders, or add
  them to `.gitignore` *before* creating them.
- **Push failures are not silent, but they are not loud either.** If a rebase
  hits a conflict the watcher logs it and stops pushing until you resolve it by
  hand. Check `tools/sync.log` if the remote looks stale.
- **The junctions are ignored.** `Anon/Dripper` and `Dripper/Anon` are Windows
  directory junctions to their sibling. `.gitignore` excludes them so git does
  not commit every file twice or chase the infinite path loop.
