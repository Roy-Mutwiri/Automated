# Dripper

Part of the `Automated` workspace.

## Purpose

_TODO: describe what Dripper does._

## Linked folders

This folder is cross-linked with its sibling `Anon` using a Windows
directory junction:

```
Dripper\Anon  ->  C:\Users\mutwi\Documents\Automated\Anon
```

`Dripper\Anon` is not a copy — it is the same folder on disk as
`..\Anon`. Files written on either side appear on both immediately.

### Notes

- The link is a junction, not a symlink, so no admin rights are required.
- The pairing is mutual, which creates an infinite path loop
  (`Dripper\Anon\Dripper\Anon\...`). Most tools skip reparse points when
  recursing, but if a backup, sync, or recursive search ever hangs here,
  this is the cause.
- To remove the link without touching the real folder:

  ```powershell
  Remove-Item "C:\Users\mutwi\Documents\Automated\Dripper\Anon"
  ```

  Never delete it with a recursive delete that follows reparse points —
  that would empty the real `Anon` folder.
dripper test line
