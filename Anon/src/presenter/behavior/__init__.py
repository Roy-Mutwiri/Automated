"""Behaviour generation: decides what the avatar does, and when it does nothing.

Exports are **lazy**, and that is load-bearing rather than a style choice.

`presenter.motion.breathing` needs `OrnsteinUhlenbeck` from
`presenter.behavior.randomness`. Importing a submodule runs its package's
`__init__` first, so an eager `from .engine import BehaviorEngine` here meant:

    motion.breathing -> behavior.randomness -> behavior/__init__
                     -> behavior.engine     -> motion.breathing   (half-built)

which raised `ImportError: cannot import name 'RespirationSystem' from
partially initialized module`. It only appeared when an adapter was imported
directly rather than through the engine, so it hid until the rig adapter was
loaded on its own.

Deferring the imports to first attribute access breaks the cycle without
scattering function-local imports through the motion package, and keeps
`from presenter.behavior import BehaviorEngine` working exactly as before.
"""

from __future__ import annotations

__all__ = [
    "BehaviorEngine",
    "EngineStats",
    "BehaviorState",
    "MotionProfile",
    "StateModulation",
    "PROFILES",
    "Drives",
    "Rng",
    "OrnsteinUhlenbeck",
    "Cooldown",
]

_SOURCES = {
    "BehaviorEngine": ".engine",
    "EngineStats": ".engine",
    "Drives": ".context",
    "Rng": ".randomness",
    "OrnsteinUhlenbeck": ".randomness",
    "Cooldown": ".randomness",
    "PROFILES": ".state",
    "BehaviorState": ".state",
    "MotionProfile": ".state",
    "StateModulation": ".state",
}


def __getattr__(name: str):
    module = _SOURCES.get(name)
    if module is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    from importlib import import_module

    return getattr(import_module(module, __name__), name)


def __dir__():
    return sorted(__all__)
