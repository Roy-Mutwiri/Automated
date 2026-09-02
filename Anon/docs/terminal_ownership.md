# Terminal ownership: branches and worktrees

Two terminals work on this project at the same time. **The branch and worktree
decide ownership — not the directory.** Both terminals legitimately have files
under `Anon/`, which is exactly why the previous folder-based scheme could not
work.

## Layout

| Terminal | Branch | Worktree |
|---|---|---|
| Camera | `camera-terminal` | `C:\Users\mutwi\Documents\Automated-camera` |
| Movement | `movement-terminal` | its own worktree |
| — | `main` | `C:\Users\mutwi\Documents\Automated` (integration) |

The camera worktree is deliberately **outside** the repository root. The sync
watcher scopes every git call to its own worktree path and watches that
directory recursively, so a worktree placed outside it is invisible to it and
cannot be swept into someone else's commit.

`Anon/.venv` in the camera worktree is a junction to the one in the main
worktree. Both terminals therefore run identical dependency versions, which is
what lets a validated number reproduce across worktrees rather than merely
appearing to.

**It has one sharp edge.** That venv contains an editable install of
`presenter` pointing at the *main* worktree, so `python -m presenter.app` run
from here silently executes main's source, not yours. Edits appear to do
nothing. Prefix the command:

    PYTHONPATH=src .venv/Scripts/python.exe -m presenter.app --renderer liveportrait

`PYTHONPATH` precedes site-packages, so the worktree's own `src/` wins. Anything
under `tools/` is unaffected — those scripts insert their own `src` path at
import time, which is why the camera bridge and depth tests were genuinely
running this worktree's code.

## Ownership

| Camera terminal | Movement terminal |
|---|---|
| scene geometry | `HumanMotionState` |
| cameras, camera configuration | gaze, attention, blinking |
| camera bridge, projection | head / neck behaviour |
| Blender ↔ external renderer transforms | breathing, shoulders, posture |
| human reconstruction integration | hands, expressions, emotion |
| Gaussian renderer integration, depth adapter | body animation |
| depth, compositing, occlusion | behaviour tests |
| camera 1–7 rendering | |
| avatar world placement | |

`tools/background_stability.py` is **movement-owned**. It measures room pixels,
but it exists to prove the animated-human pipeline does not progressively alter
the immutable environment — that is a claim about the animation, not about the
room. The camera terminal may consume its result and must not change its
implementation without coordination.

## The interface

One object crosses the boundary: **the pose**.

    BehaviorEngine.update(dt) -> pose        (movement owns)
                 |
                 v
    build_world(pose)                        (camera consumes)
    tools/render_multicam.frozen_pose()

Camera reads `pose` and never writes it. Movement produces it and imports no
renderer. The second shared surface is `config/room_geometry.yaml`: camera owns
the file, movement reads `gaze_targets` from it so that where he looks and where
things are cannot drift apart.

### Shared-interface changes go through a request, not an edit

Shared interfaces are the `HumanMotionState` schema, the avatar rig adapter
contract, the camera/world coordinate contract, and the `avatar_transform`
schema.

If the camera terminal needs one changed, it writes the request to
`docs/integration_requests/` and does **not** rewrite the movement terminal's
implementation. The reverse applies equally. A silent cross-edit lands on
another branch and reappears as a merge conflict in work nobody was expecting to
touch.

## Attribution

`tools/sync.ps1` resolves the author from the current branch via
`tools/identities.json` → `branches`. When a branch is mapped, the whole
worktree is committed as that identity in one commit and file paths are never
consulted.

The one remaining path-based rule is the fallback for branches with **no**
mapping — in practice `main`, where `Anon/` and `Dripper/` are two genuinely
separate projects and folder splitting is the correct attribution. Removing it
would collapse Dripper's contributor history, which is a different problem from
the one branch ownership solves.

A terminal that wants a specific commit message writes it to `.sync-message` in
its worktree; the watcher uses it verbatim and deletes it.

No watcher currently runs on the camera worktree, and that is intentional:
camera commits are written deliberately. To start one:

    powershell -ExecutionPolicy Bypass -File tools\sync.ps1 `
        -Worktree C:\Users\mutwi\Documents\Automated-camera

The scheduled task `AutomatedRepoSync` still watches the main worktree. It runs
the old in-memory copy of the script until it is restarted.

## Frozen infrastructure

Frozen unless a real integration failure occurs. Each was validated against
something external rather than asserted, and each number below was reproduced
after the worktree migration.

| Frozen | Validated by |
|---|---|
| `tools/camera_bridge.py` | 0.0000 px against Blender's `world_to_camera_view`, 6 points × 7 cameras |
| camera projection convention | the same reprojection test |
| Blender ↔ external renderer matrices | the same reprojection test |
| `tools/depth_composite.py` | median room depth 2.935 m where config puts the wall at 2.94 m |
| metric ray-cast room depth | chosen over the EXR pass, which cannot be read back headless |
| cached depth per camera | key carries a sha256 of `cameras.yaml` + `room_geometry.yaml` |
| depth validation | finite geometry and finite contested overlap both required |
| invalid depth | NaN / zero / negative rejected, not repaired |
| `FAR` | empty space only, never a surface |
| occlusion | tested in both directions |
| mutation tests | inverted comparison, always-human and dropped-alpha all fail the suite |

Regenerate cached room depth only when scene geometry, a camera transform, or a
focal length / sensor setting changes — the cache key does this automatically.
Do not optimise the ~12 s generation.

## Current status

Camera mathematics, camera bridge, depth, occlusion: **solved**. Room and
cameras 1–3: **ready**. Identity reconstruction: **blocking input**.

Cameras 4–7 are blocked. `config/avatar_transform.yaml` stays unset until the
reconstruction exists, and is then fitted **once**, globally — never per camera.

Continuation point: `research/lhm_remote/RUNBOOK.md`, step 4.
