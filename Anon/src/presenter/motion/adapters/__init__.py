"""Adapters map one canonical HumanMotionState onto one renderer.

An adapter may drop anything its renderer cannot show - that is its job.
It may never invent behaviour, and nothing in the behaviour engine may
import one, because the moment a decision depends on which renderer is
attached the separation this package exists to enforce is gone.
"""
