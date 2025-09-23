from __future__ import annotations

import math
import os
from datetime import datetime, timezone
from typing import Any, Mapping, MutableMapping, Sequence

__all__ = ["normalize_dataset_splits"]


def _coerce_str(value: Any) -> str | None:
    if value is None:
        return None
    if isinstance(value, str):
        text = value.strip()
        return text or None
    if hasattr(value, "__fspath__"):
        try:
            value = os.fspath(value)
        except TypeError:
            value = str(value)
    text = str(value).strip()
    return text or None


def _normalize_timestamp(value: Any) -> str | None:
    if value is None:
        return None
    if isinstance(value, datetime):
        dt = value if value.tzinfo is not None else value.replace(tzinfo=timezone.utc)
        return dt.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")
    if isinstance(value, str):
        text = value.strip()
        return text or None
    if isinstance(value, (int, float)):
        if isinstance(value, float) and math.isnan(value):
            return None
        number = float(value)
        if abs(number) >= 1e11:
            seconds = number / 1000.0
        else:
            seconds = number
        try:
            dt = datetime.fromtimestamp(seconds, tz=timezone.utc)
        except (OverflowError, OSError, ValueError):
            return _coerce_str(value)
        return dt.isoformat().replace("+00:00", "Z")
    return _coerce_str(value)


def _first_present(mapping: Mapping[str, Any], candidates: Sequence[str]) -> Any:
    for key in candidates:
        if key in mapping:
            value = mapping[key]
            if value is not None:
                return value
    return None


def _normalize_input_window(raw: Mapping[str, Any]) -> MutableMapping[str, str]:
    window: MutableMapping[str, str] = {}
    start = _normalize_timestamp(
        _first_present(raw, ("start", "from", "begin", "input_start"))
    )
    end = _normalize_timestamp(_first_present(raw, ("end", "until", "to", "stop", "input_end")))
    if not start or not end:
        nested = raw.get("input") or raw.get("range") or raw.get("window")
        if isinstance(nested, Mapping):
            if not start:
                start = _normalize_timestamp(
                    _first_present(nested, ("start", "from", "begin", "input_start"))
                )
            if not end:
                end = _normalize_timestamp(
                    _first_present(nested, ("end", "until", "to", "stop", "input_end"))
                )
    if start:
        window["start"] = start
    if end:
        window["end"] = end
    return window


def _normalize_artifact(raw: Any) -> MutableMapping[str, Any]:
    if not isinstance(raw, Mapping):
        return {}
    normalized: MutableMapping[str, Any] = {}
    window = _normalize_input_window(raw)
    if window:
        normalized["input"] = dict(window)
    output_path = _coerce_str(
        _first_present(raw, ("output_path", "output", "path", "destination", "out"))
    )
    if output_path:
        normalized["output_path"] = output_path
    hash_value = _coerce_str(
        _first_present(
            raw,
            ("verification_hash", "hash", "checksum", "sha256", "digest"),
        )
    )
    if hash_value is not None:
        normalized["verification_hash"] = hash_value
    return normalized


def _normalize_split(name: str, raw: Mapping[str, Any]) -> MutableMapping[str, Any]:
    normalized: MutableMapping[str, Any] = {"name": str(name)}
    version = _coerce_str(_first_present(raw, ("version", "tag", "revision")))
    if version:
        normalized["version"] = version
    start = _normalize_timestamp(
        _first_present(raw, ("start", "from", "begin", "window_start", "range_start"))
    )
    end = _normalize_timestamp(
        _first_present(raw, ("end", "until", "to", "window_end", "range_end"))
    )
    if not start or not end:
        nested = raw.get("range") or raw.get("window") or raw.get("period")
        if isinstance(nested, Mapping):
            if not start:
                start = _normalize_timestamp(
                    _first_present(nested, ("start", "from", "begin", "window_start"))
                )
            if not end:
                end = _normalize_timestamp(
                    _first_present(nested, ("end", "until", "to", "window_end"))
                )
    if start:
        normalized["start"] = start
    if end:
        normalized["end"] = end
    for artifact_name in ("seasonality", "adv", "fees"):
        artifact = _normalize_artifact(raw.get(artifact_name))
        if artifact:
            normalized[artifact_name] = dict(artifact)
    return normalized


def normalize_dataset_splits(raw: Any) -> dict[str, MutableMapping[str, Any]]:
    """Convert raw dataset split payload into a normalized mapping.

    The helper accepts dictionaries or sequences describing dataset splits and
    returns a canonical mapping keyed by split name. Each entry exposes the
    version tag, split boundaries (``start``/``end``) and nested artefact
    blocks for seasonality, ADV and fees, including their input windows and
    verification metadata when provided.
    """
    if raw is None:
        return {}
    normalized: dict[str, MutableMapping[str, Any]] = {}
    if isinstance(raw, Mapping):
        items = raw.items()
    elif isinstance(raw, Sequence):
        items = []
        for entry in raw:
            if not isinstance(entry, Mapping):
                continue
            name = _coerce_str(
                _first_present(entry, ("name", "split", "id", "key", "label"))
            )
            if not name:
                continue
            normalized[name] = _normalize_split(name, entry)
        return normalized
    else:
        return {}

    for name, value in items:  # type: ignore[assignment]
        if not isinstance(value, Mapping):
            continue
        key = _coerce_str(name) or _coerce_str(_first_present(value, ("name", "id")))
        if not key:
            continue
        normalized[key] = _normalize_split(key, value)
    return normalized
