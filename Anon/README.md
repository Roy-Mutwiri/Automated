# Anon

Part of the `Automated` workspace.

## Purpose

_TODO: describe what Anon does._

## Linked folders

This folder is cross-linked with its sibling `Dripper` using a Windows
directory junction:

```
Anon\Dripper  ->  C:\Users\mutwi\Documents\Automated\Dripper
```

`Anon\Dripper` is not a copy — it is the same folder on disk as
`..\Dripper`. Files written on either side appear on both immediately.

### Notes

- The link is a junction, not a symlink, so no admin rights are required.
- The pairing is mutual, which creates an infinite path loop
  (`Anon\Dripper\Anon\Dripper\...`). Most tools skip reparse points when
  recursing, but if a backup, sync, or recursive search ever hangs here,
  this is the cause.
- To remove the link without touching the real folder:

  ```powershell
  Remove-Item "C:\Users\mutwi\Documents\Automated\Anon\Dripper"
  ```

  Never delete it with a recursive delete that follows reparse points —
  that would empty the real `Dripper` folder.
