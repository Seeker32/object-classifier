from __future__ import annotations

import importlib.util
import json
from pathlib import Path

from object_classifier.config import ModelConfig, ROIBox
from object_classifier.schemas import ExportArtifacts


REPO_ROOT = Path(__file__).resolve().parents[1]


def load_module(module_name: str, relative_path: str):
    spec = importlib.util.spec_from_file_location(module_name, REPO_ROOT / relative_path)
    if spec is None or spec.loader is None:
        raise AssertionError(f"failed to load module: {relative_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_export_onnx_script_writes_onnx_payload(tmp_path, monkeypatch, capsys) -> None:
    module = load_module("tools_export_onnx", "tools/export_onnx.py")
    output_dir = tmp_path / "export"
    captured: dict[str, object] = {}

    class DummyBackend:
        pass

    def fake_create_backend(name: str, config: ModelConfig):
        captured["backend_name"] = name
        captured["config"] = config
        return DummyBackend()

    def fake_export(output: Path, backend, *, validate: bool):
        captured["output_dir"] = output
        captured["backend"] = backend
        captured["validate"] = validate
        return ExportArtifacts(
            status="ready",
            report_path=output / "export_report.json",
            embedding_onnx=output / "embedding.onnx",
            patch_tokens_onnx=output / "patch_tokens.onnx",
            notes=["fake_export"],
            validation_status="passed",
            validated_batches=[1, 2],
            embedding_metrics={"max_abs_err": 1e-6, "mean_abs_err": 1e-7},
            patch_tokens_metrics={"max_abs_err": 2e-6, "mean_abs_err": 2e-7},
        )

    monkeypatch.setattr(module, "create_backend", fake_create_backend)
    monkeypatch.setattr(module, "export_onnx_artifacts", fake_export)

    assert (
        module.main(
            [
                "--output-dir",
                str(output_dir),
                "--provider",
                "statistics",
                "--model-id",
                "dinov3_vits16",
            ]
        )
        == 0
    )

    payload = json.loads(capsys.readouterr().out)
    assert payload == {
        "status": "ready",
        "report_path": str(output_dir / "export_report.json"),
        "embedding_onnx": str(output_dir / "embedding.onnx"),
        "patch_tokens_onnx": str(output_dir / "patch_tokens.onnx"),
        "notes": ["fake_export"],
        "validation_status": "passed",
        "validated_batches": [1, 2],
        "embedding_metrics": {"max_abs_err": 1e-6, "mean_abs_err": 1e-7},
        "patch_tokens_metrics": {"max_abs_err": 2e-6, "mean_abs_err": 2e-7},
    }
    assert captured["output_dir"] == output_dir
    assert isinstance(captured["backend"], DummyBackend)
    assert captured["validate"] is True
    config = captured["config"]
    assert isinstance(config, ModelConfig)
    assert config.provider == "statistics"
    assert config.model_name == "dinov3_vits16"
    assert config.roi_box == ROIBox(0, 0, 224, 224)


def test_export_onnx_script_can_skip_validation(tmp_path, monkeypatch, capsys) -> None:
    module = load_module("tools_export_onnx_skip", "tools/export_onnx.py")
    output_dir = tmp_path / "export"
    captured: dict[str, object] = {}

    class DummyBackend:
        pass

    def fake_create_backend(name: str, config: ModelConfig):
        captured["backend_name"] = name
        captured["config"] = config
        return DummyBackend()

    def fake_export(output: Path, backend, *, validate: bool):
        captured["validate"] = validate
        return ExportArtifacts(
            status="ready",
            report_path=output / "export_report.json",
            embedding_onnx=output / "embedding.onnx",
            patch_tokens_onnx=output / "patch_tokens.onnx",
            notes=["validation_skipped:user_request"],
            validation_status="skipped",
            validated_batches=[],
            embedding_metrics=None,
            patch_tokens_metrics=None,
        )

    monkeypatch.setattr(module, "create_backend", fake_create_backend)
    monkeypatch.setattr(module, "export_onnx_artifacts", fake_export)

    assert module.main(["--output-dir", str(output_dir), "--skip-validate"]) == 0

    payload = json.loads(capsys.readouterr().out)
    assert captured["validate"] is False
    assert payload["validation_status"] == "skipped"
    assert payload["validated_batches"] == []


def test_export_rknn_script_converts_existing_onnx_only(tmp_path, monkeypatch, capsys) -> None:
    module = load_module("tools_export_rknn", "tools/export_rknn.py")
    output_dir = tmp_path / "export"
    output_dir.mkdir()
    embedding_onnx = output_dir / "embedding.onnx"
    embedding_onnx.write_bytes(b"embedding")
    patch_tokens_onnx = output_dir / "patch_tokens.onnx"
    patch_tokens_onnx.write_bytes(b"patch")
    calls: list[tuple[Path, Path, str, str]] = []

    def fake_attempt(onnx_path: Path, target_output_dir: Path, model_name: str, target: str = "rk3588"):
        calls.append((onnx_path, target_output_dir, model_name, target))
        return {
            "status": "ready",
            "model_name": model_name,
            "target": target,
            "onnx_path": str(onnx_path),
            "rknn_path": str(target_output_dir / f"{model_name}.rknn"),
            "notes": ["fake_rknn"],
        }

    monkeypatch.setattr(module, "attempt_rknn_conversion", fake_attempt)

    assert (
        module.main(
            [
                "--output-dir",
                str(output_dir),
                "--target",
                "rk3588",
                "--embedding-onnx",
                str(embedding_onnx),
                "--patch-tokens-onnx",
                str(patch_tokens_onnx),
            ]
        )
        == 0
    )

    payload = json.loads(capsys.readouterr().out)
    report = json.loads((output_dir / "rknn_report.json").read_text(encoding="utf-8"))

    assert [call[2] for call in calls] == ["embedding", "patch_tokens"]
    assert payload["status"] == "ready"
    assert payload["embedding_onnx"] == str(embedding_onnx)
    assert payload["patch_tokens_onnx"] == str(patch_tokens_onnx)
    assert payload["notes"] == []
    assert [item["model_name"] for item in payload["rknn"]] == ["embedding", "patch_tokens"]
    assert report == payload


def test_export_rknn_script_reports_missing_onnx_without_fallback(tmp_path, monkeypatch, capsys) -> None:
    module = load_module("tools_export_rknn", "tools/export_rknn.py")
    output_dir = tmp_path / "export"
    output_dir.mkdir()
    embedding_onnx = output_dir / "embedding.onnx"
    embedding_onnx.write_bytes(b"embedding")
    missing_patch_tokens = output_dir / "patch_tokens.onnx"
    calls: list[str] = []

    def fake_attempt(onnx_path: Path, target_output_dir: Path, model_name: str, target: str = "rk3588"):
        calls.append(model_name)
        return {
            "status": "ready",
            "model_name": model_name,
            "target": target,
            "onnx_path": str(onnx_path),
            "rknn_path": str(target_output_dir / f"{model_name}.rknn"),
            "notes": ["fake_rknn"],
        }

    def fail_create_backend(*args, **kwargs):
        raise AssertionError("RKNN export should not create a backend")

    monkeypatch.setattr(module, "attempt_rknn_conversion", fake_attempt)
    monkeypatch.setattr(module, "create_backend", fail_create_backend, raising=False)

    assert (
        module.main(
            [
                "--output-dir",
                str(output_dir),
                "--embedding-onnx",
                str(embedding_onnx),
                "--patch-tokens-onnx",
                str(missing_patch_tokens),
            ]
        )
        == 0
    )

    payload = json.loads(capsys.readouterr().out)

    assert calls == ["embedding"]
    assert payload["status"] == "partial"
    assert payload["embedding_onnx"] == str(embedding_onnx)
    assert payload["patch_tokens_onnx"] is None
    assert payload["notes"] == [f"missing_onnx:patch_tokens:{missing_patch_tokens}"]
    assert [item["model_name"] for item in payload["rknn"]] == ["embedding"]


def test_inspect_model_shapes_script_prints_onnx_and_pytorch_dimensions(tmp_path, monkeypatch, capsys) -> None:
    module = load_module("tools_inspect_model_shapes", "tools/inspect_model_shapes.py")
    embedding_onnx = tmp_path / "embedding.onnx"
    patch_tokens_onnx = tmp_path / "patch_tokens.onnx"
    embedding_onnx.write_bytes(b"embedding")
    patch_tokens_onnx.write_bytes(b"patch")
    captured: dict[str, object] = {}

    class DummyBackend:
        pass

    def fake_create_backend(name: str, config: ModelConfig):
        captured["backend_name"] = name
        captured["config"] = config
        return DummyBackend()

    def fake_inspect_onnx(path: Path):
        if path == embedding_onnx:
            return {
                "path": str(path),
                "inputs": [{"name": "pixel_values", "shape": [1, 3, 224, 224]}],
                "outputs": [{"name": "global_embedding", "shape": [1, 384]}],
            }
        if path == patch_tokens_onnx:
            return {
                "path": str(path),
                "inputs": [{"name": "pixel_values", "shape": [1, 3, 224, 224]}],
                "outputs": [{"name": "patch_tokens", "shape": [1, 196, 384]}],
            }
        raise AssertionError(f"unexpected onnx path: {path}")

    def fake_inspect_pytorch(backend):
        captured["backend"] = backend
        return {
            "input_shape": [1, 3, 224, 224],
            "outputs": [
                {"name": "global_embedding", "shape": [1, 384]},
                {"name": "patch_tokens", "shape": [1, 196, 384]},
            ],
        }

    monkeypatch.setattr(module, "create_backend", fake_create_backend)
    monkeypatch.setattr(module, "_inspect_onnx_model", fake_inspect_onnx)
    monkeypatch.setattr(module, "_inspect_pytorch_model", fake_inspect_pytorch)

    assert (
        module.main(
            [
                "--embedding-onnx",
                str(embedding_onnx),
                "--patch-tokens-onnx",
                str(patch_tokens_onnx),
                "--provider",
                "torchhub",
                "--model-id",
                "dinov3_vits16",
            ]
        )
        == 0
    )

    payload = json.loads(capsys.readouterr().out)
    assert payload == {
        "embedding_onnx": {
            "path": str(embedding_onnx),
            "inputs": [{"name": "pixel_values", "shape": [1, 3, 224, 224]}],
            "outputs": [{"name": "global_embedding", "shape": [1, 384]}],
        },
        "patch_tokens_onnx": {
            "path": str(patch_tokens_onnx),
            "inputs": [{"name": "pixel_values", "shape": [1, 3, 224, 224]}],
            "outputs": [{"name": "patch_tokens", "shape": [1, 196, 384]}],
        },
        "pytorch": {
            "input_shape": [1, 3, 224, 224],
            "outputs": [
                {"name": "global_embedding", "shape": [1, 384]},
                {"name": "patch_tokens", "shape": [1, 196, 384]},
            ],
        },
    }
    assert captured["backend_name"] == "pytorch"
    config = captured["config"]
    assert isinstance(config, ModelConfig)
    assert config.provider == "torchhub"
    assert config.model_name == "dinov3_vits16"
    assert config.roi_box == ROIBox(0, 0, 224, 224)
