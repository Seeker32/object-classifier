from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace

from object_classifier.config import ModelConfig, ROIBox
from object_classifier.export import attempt_rknn_conversion, export_onnx_artifacts
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


def test_attempt_rknn_conversion_uses_resolved_input_size_for_symbolic_batch(tmp_path, monkeypatch) -> None:
    onnx_path = tmp_path / "embedding.onnx"
    onnx_path.write_bytes(b"onnx")
    output_dir = tmp_path / "export"
    captured: dict[str, object] = {}

    class FakeRKNN:
        def __init__(self, verbose: bool):
            captured["verbose"] = verbose

        def config(self, *, target_platform: str, dynamic_input: list[list[list[int]]]):
            captured["config"] = {"target_platform": target_platform, "dynamic_input": dynamic_input}

        def load_onnx(self, *, model: str, input_size_list: list[list[int]]):
            captured["load_onnx"] = {"model": model, "input_size_list": input_size_list}
            return 0

        def build(self, *, do_quantization: bool):
            captured["build"] = do_quantization
            return 0

        def export_rknn(self, path: str):
            captured["export_rknn"] = path
            return 0

        def release(self):
            captured["released"] = True

    class FakeDim:
        def __init__(self, *, dim_value: int = 0, dim_param: str = ""):
            self.dim_value = dim_value
            self.dim_param = dim_param

    dims = [
        FakeDim(dim_param="batch"),
        FakeDim(dim_value=3),
        FakeDim(dim_value=224),
        FakeDim(dim_value=224),
    ]
    fake_input = SimpleNamespace(
        name="pixel_values",
        type=SimpleNamespace(
            tensor_type=SimpleNamespace(shape=SimpleNamespace(dim=dims))
        ),
    )
    fake_model = SimpleNamespace(graph=SimpleNamespace(input=[fake_input]))

    monkeypatch.setitem(sys.modules, "rknn", SimpleNamespace(api=SimpleNamespace(RKNN=FakeRKNN)))
    monkeypatch.setitem(sys.modules, "rknn.api", SimpleNamespace(RKNN=FakeRKNN))
    monkeypatch.setitem(sys.modules, "onnx", SimpleNamespace(load=lambda path: fake_model))

    report = attempt_rknn_conversion(onnx_path, output_dir, "embedding")

    assert report["status"] == "ready"
    assert report["rknn_path"] == str(output_dir / "embedding.rknn")
    assert report["input_size_list"] == [[1, 3, 224, 224]]
    assert report["dynamic_input"] == [[[1, 3, 224, 224]]]
    assert captured["config"] == {
        "target_platform": "rk3588",
        "dynamic_input": [[[1, 3, 224, 224]]],
    }
    assert captured["load_onnx"] == {
        "model": str(onnx_path),
        "input_size_list": [[1, 3, 224, 224]],
    }
    assert captured["build"] is False
    assert captured["released"] is True


def test_attempt_rknn_conversion_falls_back_to_default_input_size_when_onnx_missing(tmp_path, monkeypatch) -> None:
    onnx_path = tmp_path / "embedding.onnx"
    onnx_path.write_bytes(b"onnx")
    output_dir = tmp_path / "export"

    class FakeRKNN:
        def __init__(self, verbose: bool):
            pass

        def config(self, *, target_platform: str, dynamic_input: list[list[list[int]]]):
            self.dynamic_input = dynamic_input

        def load_onnx(self, *, model: str, input_size_list: list[list[int]]):
            self.input_size_list = input_size_list
            return 0

        def build(self, *, do_quantization: bool):
            return 0

        def export_rknn(self, path: str):
            return 0

        def release(self):
            pass

    monkeypatch.setitem(sys.modules, "rknn", SimpleNamespace(api=SimpleNamespace(RKNN=FakeRKNN)))
    monkeypatch.setitem(sys.modules, "rknn.api", SimpleNamespace(RKNN=FakeRKNN))
    monkeypatch.delitem(sys.modules, "onnx", raising=False)

    report = attempt_rknn_conversion(onnx_path, output_dir, "embedding")

    assert report["status"] == "ready"
    assert report["input_size_list"] == [[1, 3, 224, 224]]
    assert report["dynamic_input"] == [[[1, 3, 224, 224]]]
