"""OpenClaw client initialization and fallback wrapper utilities."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional

from openclaw_local_ai.models.local_model import run_local_inference


@dataclass
class OpenClawWrapper:
    """Small wrapper that keeps OpenClaw usage isolated from model logic.

    If OpenClaw is unavailable or explicitly disabled, this wrapper falls back to
    passthrough mode so local inference can still run.
    """

    enabled: bool = True
    endpoint: Optional[str] = None
    api_key: Optional[str] = None

    def __post_init__(self) -> None:
        """Try to initialize an OpenClaw client and keep failures explicit."""
        self.client = None

        if not self.enabled:
            return

        try:
            # Import lazily so missing dependency errors are clear and actionable.
            import openclaw  # type: ignore
        except ImportError as exc:
            raise RuntimeError(
                "openclaw dependency is missing. Install requirements.txt before running."
            ) from exc

        # We intentionally avoid tightly coupling to a specific OpenClaw SDK shape.
        # If a callable Client exists, we instantiate it; otherwise we keep the module.
        client_class = getattr(openclaw, "Client", None)
        if callable(client_class):
            kwargs: Dict[str, Any] = {}
            if self.endpoint:
                kwargs["endpoint"] = self.endpoint
            if self.api_key:
                kwargs["api_key"] = self.api_key
            self.client = client_class(**kwargs)
        else:
            self.client = openclaw

    def run_inference(self, prompt: str, model_callable: Any, **model_kwargs: Any) -> Any:
        """Run local model inference, optionally routed through OpenClaw later.

        Args:
            prompt: Input text to run through the local model.
            model_callable: Callable returned by model loader.
            **model_kwargs: Generation options forwarded to model callable.

        Returns:
            Model inference output in the callable's native format.

        Raises:
            RuntimeError: If inference fails.
        """
        try:
            # Delegate to shared inference helper so validation behavior is consistent.
            return run_local_inference(
                model_callable=model_callable,
                prompt=prompt,
                generation_kwargs=model_kwargs,
            )
        except Exception as exc:  # pragma: no cover - runtime error surface
            raise RuntimeError(f"Inference failed: {exc}") from exc
