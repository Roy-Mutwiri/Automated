# Integration requests

A shared interface is one both terminals depend on: the `HumanMotionState`
schema, the avatar rig adapter contract, the camera/world coordinate contract,
the `avatar_transform` schema.

When one terminal needs such a change, it files the request here instead of
editing the other terminal's implementation. A silent cross-edit lands on the
wrong branch and resurfaces as a merge conflict inside work nobody expected to
be touched.

One file per request, named `NNN-short-slug.md`:

    # What I need
    # Why the current contract does not allow it
    # Proposed shape, concretely
    # What breaks if it changes
    # Who owns the implementation

Open requests stay here until the owning terminal implements or rejects them.
A rejection is recorded in the same file, with the reason.

## Open

* [001-head-translation-channels](001-head-translation-channels.md) - the 2D
  face adapter pins `tx`/`ty`/`scale`, so the presenter's torso lean never
  reaches the picture and his apparent face width varies 2.6% over five
  minutes. Asks the camera terminal for a safe envelope, or an explicit
  rejection recording that the body rig is the answer. Owner: camera terminal.

The camera terminal has not needed a shared-interface change of its own: it
consumes `pose` read-only and owns everything else it touches.
