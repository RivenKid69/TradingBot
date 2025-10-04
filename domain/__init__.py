"""Compatibility helpers expected by legacy training scripts.

This package provides adapters for translating Gym actions into the
`ActionProto` structure that the environment and execution stack work with.
The original codebase ships these utilities as part of an internal
``domain`` package; some open-source snapshots are missing them which leads
to ``ModuleNotFoundError: domain`` when the environment is created inside
vectorized workers.  We re-expose the minimal surface that is required by
:mod:`trading_patchnew` and training wrappers.
"""

from .adapters import gym_to_action_v1, action_v1_to_proto, normalize_volume

__all__ = ["gym_to_action_v1", "action_v1_to_proto", "normalize_volume"]
