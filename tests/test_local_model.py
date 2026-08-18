"""Focused tests for local model loading and inference validation paths."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from openclaw_local_ai.models.local_model import load_local_hf_pipeline, run_local_inference
from openclaw_local_ai.utils.config import load_config


class LocalModelTests(unittest.TestCase):
    """Validate error handling for missing model/config and prompt validation."""

    def test_missing_model_directory_raises(self) -> None:
        with self.assertRaises(FileNotFoundError):
            load_local_hf_pipeline("/path/that/does/not/exist")

    def test_incomplete_model_directory_raises(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            with self.assertRaises(FileNotFoundError):
                load_local_hf_pipeline(tmp_dir)

    def test_inference_requires_non_empty_prompt(self) -> None:
        fake_model = lambda prompt, **kwargs: [{"generated_text": prompt}]  # noqa: E731
        with self.assertRaises(RuntimeError):
            run_local_inference(fake_model, "")

    def test_inference_rejects_non_callable_model(self) -> None:
        with self.assertRaisesRegex(RuntimeError, "not callable"):
            run_local_inference({}, "hello")

    def test_successful_inference_returns_callable_output(self) -> None:
        fake_model = lambda prompt, **kwargs: [{"generated_text": f"{prompt} done"}]  # noqa: E731
        output = run_local_inference(fake_model, "task")
        self.assertEqual(output[0]["generated_text"], "task done")

    def test_missing_config_raises(self) -> None:
        with self.assertRaises(FileNotFoundError):
            load_config(Path("/path/that/does/not/exist.yaml"))

    def test_non_string_config_key_raises(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            config_path = Path(tmp_dir) / "config.yaml"
            config_path.write_text("1: invalid-key-type\n", encoding="utf-8")
            with self.assertRaises(RuntimeError):
                load_config(config_path)


if __name__ == "__main__":
    unittest.main()
