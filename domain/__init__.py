"""Compatibility namespace for legacy ``domain`` package.

This repository historically exposed helpers under ``domain.adapters`` in the
internal monorepo.  Several public entry points (e.g. RL environments) still
import from that path, so we ship a small shim module that provides the
necessary adapters.
"""

from .adapters import ActionV1, action_v1_to_proto, gym_to_action_v1

__all__ = [
    "ActionV1",
    "action_v1_to_proto",
    "gym_to_action_v1",
]
