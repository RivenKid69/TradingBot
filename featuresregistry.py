from __future__ import annotations
from typing import Callable, Dict, Any, Tuple

_FEATURES: Dict[str, Callable[[Any], Any]] = {}
_CACHE: Dict[Tuple[str, int], Any] = {}

def register(name: str) -> Callable[[Callable[[Any], Any]], Callable[[Any], Any]]:
    def _wrap(fn: Callable[[Any], Any]) -> Callable[[Any], Any]:
        _FEATURES[str(name)] = fn
        return fn
    return _wrap

def compute(name: str, state: Any, version: int = 1) -> Any:
    key = (str(name), int(version))
    if key in _CACHE:
        return _CACHE[key]
    fn = _FEATURES.get(str(name))
    if fn is None:
        raise KeyError(f"feature not registered: {name}")
    val = fn(state)
    _CACHE[key] = val
    return val

def clear_cache() -> None:
    _CACHE.clear()
