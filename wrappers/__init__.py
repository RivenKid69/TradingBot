"""Utility wrappers for adapting environment interfaces."""

from .action_space import DictToMultiDiscreteActionWrapper, _wrap_action_space_if_needed

__all__ = ["DictToMultiDiscreteActionWrapper", "_wrap_action_space_if_needed"]
