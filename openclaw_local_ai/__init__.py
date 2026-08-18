"""OpenClaw local model integration package."""

from openclaw_local_ai.models.local_model import load_local_hf_pipeline, run_local_inference
from openclaw_local_ai.openclaw_integration.client import OpenClawWrapper

__all__ = ["OpenClawWrapper", "load_local_hf_pipeline", "run_local_inference"]
