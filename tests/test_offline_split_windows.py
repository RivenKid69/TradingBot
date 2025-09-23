from __future__ import annotations

from pathlib import Path
from typing import Iterable

import pytest

from scripts.offline_utils import (
    apply_split_tag,
    load_artifact_metadata,
    load_offline_payload,
    ms_to_iso,
    resolve_split_artifact,
    validate_artifact_window,
)

_ARTIFACT_KEYS: tuple[str, ...] = ("seasonality", "adv", "fees")


def _discover_offline_configs() -> list[Path]:
    root = Path(__file__).resolve().parents[1]
    config_dir = root / "configs"
    patterns = ("offline*.yml", "offline*.yaml")
    paths: list[Path] = []
    for pattern in patterns:
        paths.extend(sorted(config_dir.glob(pattern)))
    return paths


def _iter_split_artifacts(config_path: Path) -> Iterable[tuple[str, str, object]]:
    payload = load_offline_payload(config_path)
    datasets = payload.get("datasets", {})
    for split_name in sorted(datasets):
        for artifact_name in _ARTIFACT_KEYS:
            try:
                info = resolve_split_artifact(payload, split_name, artifact_name)
            except (KeyError, TypeError):
                continue
            yield split_name, artifact_name, info


@pytest.mark.parametrize("config_path", _discover_offline_configs())
def test_offline_split_metadata_within_bounds(config_path: Path) -> None:
    if not config_path.exists():  # pragma: no cover - defensive
        pytest.skip(f"offline config {config_path} is missing")

    issues: list[str] = []

    for split_name, artifact_name, info in _iter_split_artifacts(config_path):
        split_end = getattr(info, "split_end_ms", None)
        config_end = getattr(info, "config_end_ms", None)
        if split_end is not None and config_end is not None and config_end > split_end:
            issues.append(
                "::".join(
                    (
                        config_path.name,
                        split_name,
                        artifact_name,
                        (
                            f"config end {ms_to_iso(config_end)} "
                            f"exceeds split end {ms_to_iso(split_end)}"
                        ),
                    )
                )
            )

        output_path = getattr(info, "output_path", None)
        tag = getattr(info, "tag", None)
        if not output_path or not tag:
            continue
        resolved_path = apply_split_tag(Path(output_path), str(tag))
        if not resolved_path.exists():
            continue
        metadata = load_artifact_metadata(resolved_path)
        if metadata is None:
            continue
        try:
            validate_artifact_window(info, metadata, artifact_name=artifact_name)
        except ValueError as exc:
            issues.append(
                "::".join(
                    (
                        config_path.name,
                        split_name,
                        artifact_name,
                        f"metadata validation failed: {exc}",
                    )
                )
            )

    if not issues:
        return

    formatted = "\n".join(f"- {message}" for message in issues)
    pytest.fail(
        "Split metadata extends beyond configured window:\n" f"{formatted}",
        pytrace=False,
    )
