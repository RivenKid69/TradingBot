"""Helpers for persisting model artefact metadata and compatibility checks."""

from __future__ import annotations

import json
import platform
import warnings
from datetime import datetime, timezone
from hashlib import sha256
from pathlib import Path
from typing import Any, Mapping

try:  # Optional dependency: PyYAML is not guaranteed to be available.
    import yaml  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    yaml = None  # type: ignore

__all__ = ["save_sidecar_metadata", "check_model_compat"]


_SIDE_CAR_EXTENSIONS = (".sidecar.json", ".sidecar.yaml", ".sidecar.yml")
_METADATA_VERSION = 1


def _compute_sha256(path: Path) -> str:
    hasher = sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def _gather_versions() -> dict[str, str]:
    packages: dict[str, str] = {}
    for name in ("torch", "stable_baselines3", "numpy", "pandas", "optuna"):
        try:
            module = __import__(name)
        except Exception:  # pragma: no cover - package is optional
            continue
        version = getattr(module, "__version__", None)
        if isinstance(version, str):
            packages[name] = version
    return packages


def _major_minor_tuple(version: str | None) -> tuple[int, int] | None:
    if not version:
        return None
    parts: list[int] = []
    token = ""
    for ch in version:
        if ch.isdigit():
            token += ch
        else:
            if token:
                parts.append(int(token))
                token = ""
        if len(parts) >= 2:
            break
    if token and len(parts) < 2:
        parts.append(int(token))
    if len(parts) >= 2:
        return parts[0], parts[1]
    if parts:
        return parts[0], 0
    return None


def _sidecar_path(path: Path, *, preferred_ext: str = ".sidecar.json") -> Path:
    if preferred_ext not in _SIDE_CAR_EXTENSIONS:
        raise ValueError(f"Unsupported sidecar extension: {preferred_ext}")
    return path.with_suffix(path.suffix + preferred_ext)


def _find_existing_sidecar(path: Path) -> Path | None:
    for ext in _SIDE_CAR_EXTENSIONS:
        candidate = path.with_suffix(path.suffix + ext)
        if candidate.exists():
            return candidate
    return None


def save_sidecar_metadata(
    artifact_path: str | Path,
    *,
    extra: Mapping[str, Any] | None = None,
    format: str | None = None,
) -> Path:
    """Persist metadata alongside a model artefact.

    Parameters
    ----------
    artifact_path:
        Location of the artefact for which metadata should be generated.
    extra:
        Optional mapping with additional metadata to embed into the sidecar
        payload (for example, artefact type or role).
    format:
        Explicitly request the serialisation format (``"json"`` or ``"yaml"``).
        JSON is used by default and whenever YAML support is unavailable.
    """

    base_path = Path(artifact_path)
    if not base_path.exists():
        raise FileNotFoundError(f"Artefact not found: {base_path}")

    sidecar_format = (format or "json").lower()
    if sidecar_format not in {"json", "yaml", "yml"}:
        raise ValueError(f"Unsupported sidecar format: {sidecar_format}")
    if sidecar_format != "json" and yaml is None:
        warnings.warn(
            "PyYAML is not available; falling back to JSON for sidecar metadata",
            RuntimeWarning,
            stacklevel=2,
        )
        sidecar_format = "json"

    artefact_stats = base_path.stat()
    metadata: dict[str, Any] = {
        "metadata_version": _METADATA_VERSION,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "artifact": {
            "path": base_path.as_posix(),
            "name": base_path.name,
            "size": artefact_stats.st_size,
            "sha256": _compute_sha256(base_path),
            "modified_at": datetime.fromtimestamp(artefact_stats.st_mtime, tz=timezone.utc).isoformat(),
        },
        "runtime": {
            "python": platform.python_version(),
            "platform": platform.platform(),
        },
        "dependencies": _gather_versions(),
    }
    if extra:
        metadata["extra"] = dict(extra)

    extension = ".sidecar.yaml" if sidecar_format in {"yaml", "yml"} else ".sidecar.json"
    sidecar_path = _sidecar_path(base_path, preferred_ext=extension)
    sidecar_path.parent.mkdir(parents=True, exist_ok=True)

    if sidecar_format in {"yaml", "yml"} and yaml is not None:
        with sidecar_path.open("w", encoding="utf-8") as fh:
            yaml.safe_dump(metadata, fh, sort_keys=False)
    else:
        with sidecar_path.open("w", encoding="utf-8") as fh:
            json.dump(metadata, fh, indent=2, ensure_ascii=False)
    return sidecar_path


def check_model_compat(artifact_path: str | Path) -> None:
    """Validate that a saved artefact is compatible with the current runtime.

    The function inspects the sidecar metadata saved next to the artefact and
    verifies the integrity hash as well as major versions of critical
    dependencies. If metadata is missing, a warning is emitted and execution
    continues.
    """

    base_path = Path(artifact_path)
    if not base_path.exists():
        raise FileNotFoundError(f"Artefact not found: {base_path}")

    sidecar_path = _find_existing_sidecar(base_path)
    if sidecar_path is None:
        warnings.warn(
            f"Compatibility metadata not found for {base_path}. Proceeding without checks.",
            RuntimeWarning,
            stacklevel=2,
        )
        return

    try:
        with sidecar_path.open("r", encoding="utf-8") as fh:
            if sidecar_path.suffix.endswith("yaml") and yaml is not None:
                metadata = yaml.safe_load(fh)
            else:
                metadata = json.load(fh)
    except Exception as exc:  # pragma: no cover - defensive branch
        raise RuntimeError(f"Failed to read metadata from {sidecar_path}: {exc}") from exc

    if not isinstance(metadata, Mapping):
        raise RuntimeError(f"Invalid metadata payload in {sidecar_path}: expected mapping")

    artifact_info = metadata.get("artifact")
    if isinstance(artifact_info, Mapping):
        recorded_hash = artifact_info.get("sha256")
        if isinstance(recorded_hash, str):
            current_hash = _compute_sha256(base_path)
            if recorded_hash != current_hash:
                raise RuntimeError(
                    "Artefact hash mismatch: the saved file differs from the recorded metadata."
                )
        recorded_size = artifact_info.get("size")
        if isinstance(recorded_size, int) and recorded_size != base_path.stat().st_size:
            raise RuntimeError(
                f"Artefact size mismatch detected for {base_path}."
            )

    runtime_info = metadata.get("runtime")
    if isinstance(runtime_info, Mapping):
        stored_py = runtime_info.get("python")
        current_py = platform.python_version()
        stored_tuple = _major_minor_tuple(str(stored_py))
        current_tuple = _major_minor_tuple(current_py)
        if stored_tuple and current_tuple and stored_tuple[0] != current_tuple[0]:
            raise RuntimeError(
                "Python major version mismatch between artefact metadata "
                f"({stored_py}) and current runtime ({current_py})."
            )

    stored_deps = metadata.get("dependencies")
    if isinstance(stored_deps, Mapping):
        current_deps = _gather_versions()
        for name in ("torch", "stable_baselines3"):
            stored_version = stored_deps.get(name)
            current_version = current_deps.get(name)
            stored_tuple = _major_minor_tuple(str(stored_version))
            current_tuple = _major_minor_tuple(current_version)
            if stored_tuple and current_tuple and stored_tuple[0] != current_tuple[0]:
                raise RuntimeError(
                    f"{name} major version mismatch: metadata {stored_version}, runtime {current_version}."
                )

    # Additional metadata (such as `extra`) is intentionally ignored here –
    # the presence of the sidecar with consistent versions is sufficient.
    return None
