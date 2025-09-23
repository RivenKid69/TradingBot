from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Mapping, MutableMapping, Sequence

import yaml

from offline_config import normalize_dataset_splits
from utils_time import parse_time_to_ms


_TAG_SAFE_RE = re.compile(r"[^A-Za-z0-9_.-]+")
_DAY_MS = 86_400_000


def _coerce_mapping(value: Any) -> Mapping[str, Any]:
    if isinstance(value, Mapping):
        return value
    raise KeyError("expected mapping")


def _parse_time_ms(raw: Any) -> int | None:
    if raw is None:
        return None
    text = str(raw).strip()
    if not text:
        return None
    try:
        return int(parse_time_to_ms(text))
    except Exception as exc:  # pragma: no cover - defensive
        raise ValueError(f"failed to parse timestamp '{raw}'") from exc


def ms_to_iso(ms: int | None) -> str | None:
    if ms is None:
        return None
    return datetime.fromtimestamp(ms / 1000.0, tz=timezone.utc).isoformat().replace("+00:00", "Z")


def sanitize_tag(tag: str | None, *, fallback: str) -> str:
    base = (tag or "").strip() or fallback
    sanitized = _TAG_SAFE_RE.sub("-", base).strip("-_.")
    return sanitized or fallback


def apply_split_tag(path: Path, tag: str) -> Path:
    sanitized = sanitize_tag(tag, fallback="split")
    suffix = "".join(path.suffixes)
    if suffix:
        stem = path.name[: -len(suffix)]
    else:
        stem = path.name
    if stem.endswith(f"_{sanitized}") or stem.endswith(f"-{sanitized}"):
        return path
    new_name = f"{stem}_{sanitized}{suffix}" if stem else f"{sanitized}{suffix}"
    return path.with_name(new_name)


def window_days(start_ms: int | None, end_ms: int | None) -> int | None:
    if start_ms is None or end_ms is None:
        return None
    if end_ms <= start_ms:
        return 0
    span = end_ms - start_ms
    return max(1, math.ceil(span / _DAY_MS))


def load_offline_payload(path: Path | str) -> dict[str, Any]:
    config_path = Path(path)
    with config_path.open("r", encoding="utf-8") as fh:
        payload = yaml.safe_load(fh) or {}
    if not isinstance(payload, MutableMapping):
        raise ValueError(f"offline config {config_path} must be a mapping")
    data = dict(payload)
    datasets_raw = data.get("datasets")
    if not isinstance(datasets_raw, Mapping):
        datasets_raw = data.get("dataset_splits")
    data["datasets"] = normalize_dataset_splits(datasets_raw)
    return data


@dataclass(frozen=True)
class SplitArtifact:
    split_name: str
    version: str | None
    split_start_ms: int | None
    split_end_ms: int | None
    artifact: Mapping[str, Any]
    config_start_ms: int | None
    config_end_ms: int | None
    output_path: Path | None

    @property
    def tag(self) -> str:
        return sanitize_tag(self.version, fallback=self.split_name)

    @property
    def split_metadata(self) -> dict[str, str]:
        meta = {"name": self.split_name}
        if self.version:
            meta["version"] = self.version
        return meta

    @property
    def configured_window(self) -> dict[str, str | int | None]:
        return {
            "start": ms_to_iso(self.config_start_ms),
            "end": ms_to_iso(self.config_end_ms),
            "start_ms": self.config_start_ms,
            "end_ms": self.config_end_ms,
        }


def resolve_split_artifact(
    payload: Mapping[str, Any],
    split_name: str,
    artifact_key: str,
) -> SplitArtifact:
    datasets = payload.get("datasets")
    if not isinstance(datasets, Mapping):
        raise KeyError("offline config does not define dataset splits")
    split_cfg = datasets.get(split_name)
    split_mapping = _coerce_mapping(split_cfg)
    version = split_mapping.get("version")
    split_start_ms = _parse_time_ms(split_mapping.get("start"))
    split_end_ms = _parse_time_ms(split_mapping.get("end"))

    artifact_cfg = split_mapping.get(artifact_key)
    artifact_mapping = _coerce_mapping(artifact_cfg)

    input_cfg = artifact_mapping.get("input") if isinstance(artifact_mapping.get("input"), Mapping) else None
    if isinstance(input_cfg, Mapping):
        config_start_ms = _parse_time_ms(input_cfg.get("start"))
        config_end_ms = _parse_time_ms(input_cfg.get("end"))
    else:
        config_start_ms = None
        config_end_ms = None
    if config_start_ms is None:
        config_start_ms = split_start_ms
    if config_end_ms is None:
        config_end_ms = split_end_ms

    output_path_raw = artifact_mapping.get("output_path")
    output_path = Path(str(output_path_raw)) if output_path_raw else None

    return SplitArtifact(
        split_name=split_mapping.get("name", split_name) or split_name,
        version=str(version) if version is not None else None,
        split_start_ms=split_start_ms,
        split_end_ms=split_end_ms,
        artifact=artifact_mapping,
        config_start_ms=config_start_ms,
        config_end_ms=config_end_ms,
        output_path=output_path,
    )


def _extract_metadata_block(payload: Any) -> Mapping[str, Any] | None:
    if isinstance(payload, Mapping):
        for key in ("metadata", "meta"):
            value = payload.get(key)
            if isinstance(value, Mapping):
                return value
        for value in payload.values():
            nested = _extract_metadata_block(value)
            if isinstance(nested, Mapping):
                return nested
    elif isinstance(payload, Sequence) and not isinstance(payload, (str, bytes, bytearray)):
        for item in payload:
            nested = _extract_metadata_block(item)
            if isinstance(nested, Mapping):
                return nested
    return None


def load_artifact_metadata(path: Path) -> Mapping[str, Any] | None:
    with Path(path).open("r", encoding="utf-8") as fh:
        try:
            payload = json.load(fh)
        except json.JSONDecodeError as exc:  # pragma: no cover - defensive
            raise ValueError(f"failed to parse metadata from {path}: {exc}") from exc
    metadata = _extract_metadata_block(payload)
    return metadata if isinstance(metadata, Mapping) else None


def _extract_window_bounds(entry: Mapping[str, Any] | None) -> tuple[int | None, int | None]:
    if not isinstance(entry, Mapping):
        return (None, None)
    start_ms = None
    end_ms = None
    for key in ("start_ms", "start_ts", "start", "from", "begin"):
        if key in entry and entry[key] is not None:
            try:
                start_ms = _parse_time_ms(entry[key])
            except Exception:  # pragma: no cover - defensive
                continue
            break
    for key in ("end_ms", "end_ts", "end", "to", "stop"):
        if key in entry and entry[key] is not None:
            try:
                end_ms = _parse_time_ms(entry[key])
            except Exception:  # pragma: no cover - defensive
                continue
            break
    return start_ms, end_ms


def validate_artifact_window(
    split_info: SplitArtifact,
    metadata: Mapping[str, Any],
    *,
    artifact_name: str,
) -> None:
    window_block = metadata.get("data_window")
    if not isinstance(window_block, Mapping):
        return
    actual_block = window_block.get("actual")
    actual_start, actual_end = _extract_window_bounds(actual_block if isinstance(actual_block, Mapping) else {})
    issues: list[str] = []
    split_start = split_info.split_start_ms
    split_end = split_info.split_end_ms
    if actual_start is not None and split_start is not None and actual_start < split_start:
        issues.append(
            f"start {ms_to_iso(actual_start)} precedes split start {ms_to_iso(split_start)}"
        )
    if actual_end is not None and split_end is not None and actual_end > split_end:
        issues.append(
            f"end {ms_to_iso(actual_end)} exceeds split end {ms_to_iso(split_end)}"
        )
    if issues:
        message = "; ".join(issues)
        raise ValueError(f"{artifact_name} window out of range: {message}")


@dataclass(frozen=True)
class ResolvedArtifact:
    name: str
    info: SplitArtifact
    path: Path
    metadata: Mapping[str, Any] | None


@dataclass(frozen=True)
class ResolvedSplitBundle:
    name: str
    version: str | None
    artifacts: Dict[str, ResolvedArtifact]


def resolve_artifact_path(
    split_info: SplitArtifact,
    *,
    artifact_name: str,
    require_metadata: bool = True,
    validate: bool = True,
) -> tuple[Path, Mapping[str, Any] | None]:
    base_path = split_info.output_path
    if base_path is None:
        raise ValueError(
            f"offline config for split {split_info.split_name!r} "
            f"is missing {artifact_name}.output_path"
        )
    resolved_path = apply_split_tag(base_path, split_info.tag)
    if not resolved_path.exists():
        raise FileNotFoundError(
            f"resolved {artifact_name} artifact not found: {resolved_path}"
        )
    metadata: Mapping[str, Any] | None = None
    if require_metadata or validate:
        metadata = load_artifact_metadata(resolved_path)
        if metadata is None and require_metadata:
            raise ValueError(
                f"{artifact_name} artifact {resolved_path} does not contain metadata"
            )
    if validate and metadata is not None:
        try:
            validate_artifact_window(split_info, metadata, artifact_name=artifact_name)
        except ValueError as exc:
            raise ValueError(f"{artifact_name} artifact {resolved_path}: {exc}") from exc
    elif validate and metadata is None:
        raise ValueError(
            f"{artifact_name} artifact {resolved_path} is missing metadata for validation"
        )
    return resolved_path, metadata


def resolve_split_bundle(
    payload_or_path: Mapping[str, Any] | str | Path,
    split_name: str,
    *,
    artifact_keys: Sequence[str] = ("seasonality", "adv", "fees"),
    require_metadata: bool = True,
    validate: bool = True,
) -> ResolvedSplitBundle:
    if isinstance(payload_or_path, (str, Path)):
        payload = load_offline_payload(payload_or_path)
    else:
        payload = payload_or_path
    datasets = payload.get("datasets")
    if not isinstance(datasets, Mapping):
        raise KeyError("offline config does not define dataset splits")
    split_cfg = datasets.get(split_name)
    split_mapping = _coerce_mapping(split_cfg)
    version = split_mapping.get("version")
    artifacts: Dict[str, ResolvedArtifact] = {}
    for key in artifact_keys:
        artifact_cfg = split_mapping.get(key)
        if artifact_cfg is None:
            continue
        try:
            split_info = resolve_split_artifact(payload, split_name, key)
        except KeyError:
            continue
        path, metadata = resolve_artifact_path(
            split_info,
            artifact_name=key,
            require_metadata=require_metadata,
            validate=validate,
        )
        artifacts[key] = ResolvedArtifact(
            name=key,
            info=split_info,
            path=path,
            metadata=metadata,
        )
        if version is None:
            version = split_info.version
    resolved_version = str(version) if version is not None else None
    return ResolvedSplitBundle(name=split_name, version=resolved_version, artifacts=artifacts)


__all__ = [
    "SplitArtifact",
    "ResolvedArtifact",
    "ResolvedSplitBundle",
    "apply_split_tag",
    "window_days",
    "load_offline_payload",
    "load_artifact_metadata",
    "resolve_artifact_path",
    "resolve_split_artifact",
    "resolve_split_bundle",
    "sanitize_tag",
    "validate_artifact_window",
]
