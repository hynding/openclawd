"""Example entry point for running local text generation through OpenClaw wrapper."""

from __future__ import annotations

import json
from pathlib import Path

from openclaw_local_ai.models.local_model import load_local_hf_pipeline
from openclaw_local_ai.openclaw_integration.client import OpenClawWrapper
from openclaw_local_ai.utils.config import load_config


def main() -> None:
    """Load config, initialize OpenClaw, run local model inference, and print output."""
    try:
        config_path = Path(__file__).resolve().parents[2] / "config.yaml"
        config = load_config(config_path)

        openclaw_config = config.get("openclaw", {})
        model_config = config.get("model", {})
        runtime_config = config.get("runtime", {})

        wrapper = OpenClawWrapper(
            enabled=bool(openclaw_config.get("enabled", True)),
            endpoint=openclaw_config.get("endpoint"),
            api_key=openclaw_config.get("api_key"),
        )

        model = load_local_hf_pipeline(
            model_path=model_config.get("path", "./models/local-text-model"),
            task=model_config.get("task", "text-generation"),
            device=int(model_config.get("device", -1)),
        )

        prompt = "Explain why local open-source models are useful for private AI workflows."
        output = wrapper.run_inference(prompt, model, **runtime_config)

        # Pretty-print dict/list outputs while still supporting plain string outputs.
        if isinstance(output, (dict, list)):
            print(json.dumps(output, indent=2, ensure_ascii=False))
        else:
            print(output)
    except (FileNotFoundError, RuntimeError, ValueError) as exc:
        # Keep errors user-friendly for first-time setup and model wiring mistakes.
        print(f"Error: {exc}")


if __name__ == "__main__":
    main()
