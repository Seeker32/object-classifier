from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from object_classifier.config import ModelConfig, ROIBox
from object_classifier.export import attempt_rknn_conversion, export_onnx_artifacts
from object_classifier.features import (
    TorchModuleSession,
    PyTorchFeatureBackend,
    StatisticsFeatureSession,
    _validate_onnx_export,
    _find_unsupported_onnx_ops,
    _forward_features_with_frozen_rope,
)


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


def test_torch_module_session_exports_static_batch_onnx(tmp_path, monkeypatch) -> None:
    captured: list[dict[str, object]] = []

    class FakeTensor:
        def __init__(self, shape: tuple[int, ...]):
            self.shape = shape

    class FakeModule:
        def eval(self):
            return self

    class FakeNoGrad:
        def __enter__(self):
            return None

        def __exit__(self, exc_type, exc, tb):
            return False

    def fake_export(model, sample, path, **kwargs):
        captured.append({"path": str(path), "sample_shape": sample.shape, "kwargs": kwargs})

    fake_torch = SimpleNamespace(
        float32="float32",
        nn=SimpleNamespace(Module=object),
        randn=lambda *shape, dtype=None: FakeTensor(shape),
        onnx=SimpleNamespace(export=fake_export),
        no_grad=lambda: FakeNoGrad(),
    )

    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    monkeypatch.setattr(
        "object_classifier.features._build_export_wrappers",
        lambda module, sample: ("embedding-wrapper", "patch-wrapper"),
    )
    monkeypatch.setattr(
        "object_classifier.features._validate_onnx_export",
        lambda **kwargs: {
            "validation_status": "passed",
            "validated_batches": [1],
            "embedding_metrics": {"max_abs_err": 0.0, "mean_abs_err": 0.0},
            "patch_tokens_metrics": {"max_abs_err": 0.0, "mean_abs_err": 0.0},
            "notes": ["validation_passed"],
        },
    )

    session = TorchModuleSession(FakeModule())
    payload = session.export_onnx(tmp_path, (224, 224), validate=True)

    assert payload["status"] == "ready"
    assert payload["validated_batches"] == [1]
    assert len(captured) == 2
    for export_call in captured:
        assert export_call["sample_shape"] == (1, 3, 224, 224)
        assert "dynamic_axes" not in export_call["kwargs"]


def test_validate_onnx_export_uses_static_batch_only(monkeypatch, tmp_path) -> None:
    captured_batch_sizes: list[int] = []

    class FakeTorchTensor:
        def __init__(self, array: np.ndarray):
            self.array = array

        def detach(self):
            return self

        def cpu(self):
            return self

        def numpy(self):
            return self.array

    class FakeNoGrad:
        def __enter__(self):
            return None

        def __exit__(self, exc_type, exc, tb):
            return False

    class FakeSession:
        def __init__(self, path: str, providers: list[str]):
            self.path = path
            self.providers = providers

        def run(self, _, ort_inputs):
            batch = ort_inputs["pixel_values"]
            captured_batch_sizes.append(int(batch.shape[0]))
            if self.path.endswith("embedding.onnx"):
                return [batch.mean(axis=(2, 3))]
            return [batch.reshape(batch.shape[0], batch.shape[1], -1).transpose(0, 2, 1)]

    class FakeEmbeddingWrapper:
        def eval(self):
            return self

        def __call__(self, sample):
            return FakeTorchTensor(sample.array.mean(axis=(2, 3)))

    class FakePatchWrapper:
        def eval(self):
            return self

        def __call__(self, sample):
            return FakeTorchTensor(sample.array.reshape(sample.array.shape[0], sample.array.shape[1], -1).transpose(0, 2, 1))

    class FakeModule:
        def eval(self):
            return self

    monkeypatch.setitem(
        sys.modules,
        "torch",
        SimpleNamespace(
            float32="float32",
            randn=lambda *shape, dtype=None: FakeTorchTensor(np.ones(shape, dtype=np.float32)),
            no_grad=lambda: FakeNoGrad(),
        ),
    )
    monkeypatch.setitem(
        sys.modules,
        "onnxruntime",
        SimpleNamespace(InferenceSession=FakeSession),
    )
    monkeypatch.setattr("object_classifier.features._find_unsupported_onnx_ops", lambda path: [])
    monkeypatch.setattr("object_classifier.features._EmbeddingExportWrapper", lambda module: FakeEmbeddingWrapper())
    monkeypatch.setattr("object_classifier.features._PatchTokenExportWrapper", lambda module: FakePatchWrapper())

    result = _validate_onnx_export(
        module=FakeModule(),
        input_size=(224, 224),
        embedding_path=tmp_path / "embedding.onnx",
        patch_tokens_path=tmp_path / "patch_tokens.onnx",
        enabled=True,
    )

    assert result["validation_status"] == "passed"
    assert result["validated_batches"] == [1]
    assert captured_batch_sizes == [1, 1]


def test_attempt_rknn_conversion_uses_resolved_input_size_for_symbolic_batch(tmp_path, monkeypatch) -> None:
    onnx_path = tmp_path / "embedding.onnx"
    onnx_path.write_bytes(b"onnx")
    output_dir = tmp_path / "export"
    captured: dict[str, object] = {}

    class FakeRKNN:
        def __init__(self, verbose: bool):
            captured["verbose"] = verbose

        def config(self, *, target_platform: str):
            captured["config"] = {"target_platform": target_platform}

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
    assert report["dynamic_input"] is None
    assert captured["config"] == {
        "target_platform": "rk3588",
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

        def config(self, *, target_platform: str):
            self.target_platform = target_platform

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
    assert report["dynamic_input"] is None


def test_forward_features_with_frozen_rope_uses_precomputed_rope() -> None:
    captured: dict[str, object] = {}

    class FakeBlock:
        def __call__(self, x, rope):
            captured.setdefault("ropes", []).append(rope)
            return x + 1

    class FakeModule:
        n_storage_tokens = 0
        blocks = [FakeBlock(), FakeBlock()]
        untie_cls_and_patch_norms = False
        untie_global_and_local_cls_norm = False

        @staticmethod
        def prepare_tokens_with_masks(pixel_values, masks=None):
            return pixel_values, (2, 2)

        @staticmethod
        def norm(x):
            return x * 2

    pixel_values = np.arange(24, dtype=np.float32).reshape(1, 3, 8)
    rope = ("sin", "cos")

    outputs = _forward_features_with_frozen_rope(FakeModule(), pixel_values, rope)

    assert captured["ropes"] == [rope, rope]
    assert outputs["x_norm_clstoken"].shape == (1, 8)
    assert outputs["x_norm_patchtokens"].shape == (1, 2, 8)


def test_find_unsupported_onnx_ops_reports_if_nodes(tmp_path, monkeypatch) -> None:
    onnx_path = tmp_path / "embedding.onnx"
    onnx_path.write_bytes(b"onnx")
    fake_model = SimpleNamespace(
        graph=SimpleNamespace(
            node=[
                SimpleNamespace(op_type="Identity"),
                SimpleNamespace(op_type="If"),
                SimpleNamespace(op_type="Add"),
            ]
        )
    )

    monkeypatch.setitem(sys.modules, "onnx", SimpleNamespace(load=lambda path: fake_model))

    assert _find_unsupported_onnx_ops(onnx_path) == ["If"]
