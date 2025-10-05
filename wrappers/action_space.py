"""Legacy compatibility wrappers used by downstream code.

The deterministic allocator pipeline no longer requires Gym action wrappers,
so this module exposes a minimal identity wrapper that forwards all operations
back to the underlying environment.  It exists solely to keep optional imports
from breaking in older integration scripts.
"""
from __future__ import annotations

from typing import Any


class IdentityActionWrapper:
    """No-op wrapper retaining a similar interface to gym.wrappers."""

    def __init__(self, env: Any, *args: Any, **kwargs: Any) -> None:
        self.env = env

    def action(self, action: Any) -> Any:  # pragma: no cover - passthrough
        return action

    def reverse_action(self, action: Any) -> Any:  # pragma: no cover - passthrough
        return action

    def __getattr__(self, name: str) -> Any:  # pragma: no cover - passthrough
        return getattr(self.env, name)


__all__ = ["IdentityActionWrapper"]
