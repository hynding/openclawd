"""Configuration helpers for the OpenClaw local model project."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict


def _require_yaml() -> Any:
    """Import yaml lazily so users get a clear error when dependency is missing."""
    try:
        import yaml
    except ImportError as exc:  # pragma: no cover - exercised in runtime environments
        raise RuntimeError(
            "PyYAML is required to read config.yaml. Install dependencies from requirements.txt."
        ) from exc
    return yaml


def load_config(config_path: str | Path) -> Dict[str, Any]:
    """Load a YAML configuration file from disk.

    Args:
        config_path: Absolute or relative path to the YAML file.

    Returns:
        Parsed configuration dictionary.

    Raises:
        FileNotFoundError: If the config file does not exist.
        RuntimeError: If YAML parsing fails.
    """
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Configuration file not found: {path}")

    yaml = _require_yaml()

    try:
        with path.open("r", encoding="utf-8") as file:
            data = yaml.safe_load(file) or {}
    except Exception as exc:  # pragma: no cover - defensive runtime handling
        raise RuntimeError(f"Failed to parse configuration file {path}: {exc}") from exc

    if not isinstance(data, dict):
        raise RuntimeError(f"Configuration root must be a dictionary in {path}")

    return data
