from __future__ import annotations

import json
from pathlib import Path

from object_classifier.config import ModelConfig, ROIBox
from object_classifier.export import export_onnx_artifacts
from object_classifier.features import PyTorchFeatureBackend, StatisticsFeatureSession


def test_export_onnx_artifacts_uses_backend_exporter(tmp_path) -> None:
    backend = PyTorchFeatureBackend(
        config=ModelConfig(input_size=(32, 32), roi_box=ROIBox(0, 0, 32, 32), embedding_dim=3),
        session=StatisticsFeatureSession(),
    )

    def fake_export(active_backend, output_dir: Path, *, validate: bool):
        embedding = output_dir / "embedding.onnx"
        patch_tokens = output_dir / "patch_tokens.onnx"
        embedding.write_bytes(b"embedding")
        patch_tokens.write_bytes(b"patch")
        assert validate is True
        return {
            "status": "ready",
            "embedding_onnx": embedding,
            "patch_tokens_onnx": patch_tokens,
            "notes": ["fake_export"],
            "validation_status": "passed",
            "validated_batches": [1, 2],
            "embedding_metrics": {
                "max_abs_err": 1e-6,
                "mean_abs_err": 1e-7,
            },
            "patch_tokens_metrics": {
                "max_abs_err": 2e-6,
                "mean_abs_err": 2e-7,
            },
        }

    artifacts = export_onnx_artifacts(tmp_path, backend=backend, exporter=fake_export)
    report = json.loads(artifacts.report_path.read_text(encoding="utf-8"))

    assert artifacts.status == "ready"
    assert artifacts.embedding_onnx == tmp_path / "embedding.onnx"
    assert artifacts.patch_tokens_onnx == tmp_path / "patch_tokens.onnx"
    assert artifacts.validation_status == "passed"
    assert artifacts.validated_batches == [1, 2]
    assert artifacts.embedding_metrics == {"max_abs_err": 1e-6, "mean_abs_err": 1e-7}
    assert artifacts.patch_tokens_metrics == {"max_abs_err": 2e-6, "mean_abs_err": 2e-7}
    assert report["notes"] == ["fake_export"]
    assert report["validation_status"] == "passed"
    assert report["validated_batches"] == [1, 2]
    assert report["embedding_metrics"] == {"max_abs_err": 1e-6, "mean_abs_err": 1e-7}
    assert report["patch_tokens_metrics"] == {"max_abs_err": 2e-6, "mean_abs_err": 2e-7}
