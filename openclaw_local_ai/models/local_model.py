"""Local open-source model loader utilities for OpenClaw integration."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict


def _require_transformers_pipeline() -> Any:
    """Import transformers.pipeline lazily with a clear dependency error."""
    try:
        from transformers import pipeline
    except ImportError as exc:  # pragma: no cover - runtime environment dependent
        raise RuntimeError(
            "transformers dependency is missing. Install requirements.txt first."
        ) from exc
    return pipeline


def load_local_hf_pipeline(model_path: str | Path, task: str = "text-generation", device: int = -1) -> Any:
    """Load a Hugging Face model and tokenizer from a local directory.

    Args:
        model_path: Directory containing model weights and tokenizer files.
        task: transformers pipeline task name.
        device: Device index (-1 for CPU, >=0 for CUDA device).

    Returns:
        A callable transformers pipeline object.

    Raises:
        FileNotFoundError: If model directory does not exist.
        RuntimeError: If model fails to load.
    """
    path = Path(model_path)
    if not path.exists() or not path.is_dir():
        raise FileNotFoundError(f"Local model directory was not found: {path}")

    # A lightweight sanity check that catches empty or incomplete model folders.
    config_candidates = ["config.json", "tokenizer.json", "tokenizer_config.json"]
    if not any((path / candidate).exists() for candidate in config_candidates):
        raise FileNotFoundError(
            "Model directory exists but appears incomplete. "
            "Expected one of: config.json, tokenizer.json, tokenizer_config.json"
        )

    pipeline = _require_transformers_pipeline()

    try:
        return pipeline(task=task, model=str(path), tokenizer=str(path), device=device)
    except Exception as exc:  # pragma: no cover - runtime error surface
        raise RuntimeError(f"Failed to load local model from {path}: {exc}") from exc


def run_local_inference(model_callable: Any, prompt: str, generation_kwargs: Dict[str, Any] | None = None) -> Any:
    """Run inference against a loaded local model callable.

    Args:
        model_callable: Callable model object (for example HF pipeline).
        prompt: User prompt text for generation/inference.
        generation_kwargs: Optional generation parameters.

    Returns:
        Raw inference response from the model.

    Raises:
        RuntimeError: If prompt is invalid or inference call fails.
    """
    if not callable(model_callable):
        raise RuntimeError("Model object is not callable for inference.")

    if not isinstance(prompt, str) or not prompt.strip():
        raise RuntimeError("Prompt must be a non-empty string.")

    kwargs = generation_kwargs or {}

    try:
        return model_callable(prompt, **kwargs)
    except Exception as exc:  # pragma: no cover - runtime error surface
        raise RuntimeError(f"Local inference failed: {exc}") from exc
