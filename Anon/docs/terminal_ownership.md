# Terminal ownership and the interface between them

Two terminals work in this repository at the same time. This file records who
owns what, and the narrow surface where they meet.

## Ownership

| Camera terminal | Movement terminal |
|---|---|
| scene / room geometry | behaviour |
| cameras, projection, camera bridge | attention |
| depth, compositing | eye / head |
| identity reconstruction integration | body, face |
| `config/cameras.yaml`, `config/room_geometry.yaml`, `config/avatar_transform.yaml` | motion tests |
| `src/presenter/scene3d/`, `tools/camera_bridge.py`, `tools/depth_composite.py`, `tools/render_multicam.py` | `src/presenter/behavior/`, `src/presenter/motion/` |
| `tests/test_depth_composite.py` | `tests/test_motion.py` |
| `research/lhm_remote/` | `tools/eye_head_test.py` |

`tools/background_stability.py` is unassigned. Its subject is the room, which
would make it camera work, but it was authored by the movement terminal and the
camera terminal has not reviewed it. It is filed with movement rather than
silently claimed.

## The interface

One object crosses the boundary: **the pose**.

    BehaviorEngine.update(dt) -> pose        (movement terminal owns this)
                 |
                 v
    build_world(pose)                        (camera terminal consumes it)
    tools/render_multicam.frozen_pose()

The camera terminal reads `pose` and never writes it. The movement terminal
produces it and knows nothing about cameras, depth, or the renderer. Nothing in
`presenter/behavior/` or `presenter/motion/` imports a renderer, and nothing in
`presenter/scene3d/` decides behaviour.

The second, smaller shared surface is `config/room_geometry.yaml`: the camera
terminal owns the file, and the movement terminal reads `gaze_targets` from it
so that where he looks and where things are cannot drift apart.

Changing the shape of `pose`, or the meaning of a `gaze_target`, is a change to
a shared interface. Everything else in each column is one terminal's business.

## The branch is the ownership boundary

Each terminal works in its own git worktree, on its own branch. The directory is
the isolation and the branch is the record:

| Worktree | Branch | Owner |
|---|---|---|
| `Automated/` | `main` | shared / integration |
| `Automated-camera/` | `camera-terminal` | camera terminal |
| `Automated-movement/` | `movement-terminal` | movement terminal |

Confirm which one you are in before doing anything:

    git branch --show-current

Both terminals legitimately edit files under `Anon/`, so a path can never say
who wrote a change. A branch can, and does.

## Automated commits follow the branch, not the folder

One scheduled task per worktree, `AutomatedRepoSync_<worktree>`, runs
`tools/sync.ps1 -Worktree <that directory>`. Each watcher:

    detect the current branch
        -> commit only its own worktree, as that branch's owner, in one commit
        -> push only that branch

Branch to identity is mapped in `tools/identities.json` under `branches`. A
watcher never reads a source path to decide an owner, and never touches a branch
another terminal is sitting on. Registration lives in `tools/autostart.ps1`;
`-Remove` unregisters every watcher.

The old behaviour attributed commits by top-level folder — `Anon/` versus
`Dripper/` — which could not separate two terminals working in the same folder,
and merged their work into single commits like
`Anon: 12 files (9 added, 3 modified)`. Folder attribution now survives only as
the fallback for a branch with no mapping, in practice `main`, where `Anon/` and
`Dripper/` really are two separate projects.

To give a commit a real message instead of a generated one, write the message to
`.sync-message` in your worktree before saving. The watcher uses it verbatim and
deletes it.

History still cannot be rewritten while your watcher runs: a `git reset --soft`
is re-committed within seconds. Stop that one task first —
`Stop-ScheduledTask -TaskName AutomatedRepoSync_<worktree>` — which now affects
only your own branch.

## Frozen infrastructure

Do not rewrite these unless a concrete reconstruction integration bug proves
them wrong. Each was validated against something external rather than asserted.

| Frozen | Validated by |
|---|---|
| `tools/camera_bridge.py` | 0.0000 px against Blender's `world_to_camera_view`, 6 probe points x 7 cameras |
| camera projection convention | the same reprojection test |
| Blender to external-renderer camera matrices | the same reprojection test |
| `tools/depth_composite.py` | median room depth 2.935 m where config puts the wall at 2.94 m |
| ray-cast room depth | as above; chosen over the EXR pass, which cannot be read back headless |
| cached depth per camera | cache is keyed to camera, resolution, and a hash of `config/cameras.yaml` + `config/room_geometry.yaml`, so moving a camera or changing the room invalidates it |
| near/far occlusion self-tests | two probes, one that must win and one that must lose |
| cam1/2/3 depth validation | mic and boom correctly occlude the human on cam2 |

Regenerate cached room depth only when scene geometry, a camera transform, or a
focal length / sensor setting changes. Otherwise reuse it indefinitely. Do not
optimise the ~12 s generation; it runs once per camera.
