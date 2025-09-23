from __future__ import annotations

import math
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, MutableMapping

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
