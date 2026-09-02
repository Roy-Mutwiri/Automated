# Anon

Anon's working folder in the [`Automated`](../README.md) repo.

## Purpose

_TODO: describe what Anon does._

## Auto-commit

Everything saved in this folder is committed automatically, authored as
**Anon**, and pushed. No `git add` or `git commit` needed - just save.

- Identity used: see `folders.Anon` in [`tools/identities.json`](../tools/identities.json)
- Commits appear as `Anon: update <file>` in the log
- Check `tools/sync.log` if a change does not reach the remote

Anything written here is pushed within a couple of seconds, so do not put
secrets, keys, or scratch data in this folder unless it is in `.gitignore`
first.

## Linked folder

`Anon/Dripper` is a Windows directory junction to the sibling folder:

```
Anon\Dripper  ->  C:\Users\mutwi\Documents\Automated\Dripper
```

It is the same folder on disk, not a copy. Git ignores it (see `.gitignore`)
so files are not committed twice. To remove the link without touching the real
folder:

```powershell
Remove-Item "C:\Users\mutwi\Documents\Automated\Anon\Dripper"
```

Never delete it with a recursive delete that follows reparse points - that
would empty the real `Dripper` folder.
