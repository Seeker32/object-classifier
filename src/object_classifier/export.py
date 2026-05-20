from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .features import BaseFeatureBackend, export_backend_to_onnx
from .schemas import ExportArtifacts


def export_onnx_artifacts(
    output_dir: Path,
    backend: BaseFeatureBackend | None = None,
    exporter=None,
    *,
    validate: bool = True,
) -> ExportArtifacts:
    output_dir.mkdir(parents=True, exist_ok=True)
    report_path = output_dir / "export_report.json"
    payload = _default_report(output_dir)

    if backend is not None:
        active_exporter = exporter or export_backend_to_onnx
        try:
            payload = dict(active_exporter(backend, output_dir, validate=validate))
        except ModuleNotFoundError as exc:
            payload = {
                "status": "blocked",
                "embedding_onnx": None,
                "patch_tokens_onnx": None,
                "notes": [f"missing_dependency:{exc.name}"],
                "validation_status": "not_run",
                "validated_batches": [],
                "embedding_metrics": None,
                "patch_tokens_metrics": None,
            }

    report = _serialize_payload(payload)
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return ExportArtifacts(
        status=report["status"],
        report_path=report_path,
        embedding_onnx=Path(report["embedding_onnx"]) if report["embedding_onnx"] else None,
        patch_tokens_onnx=Path(report["patch_tokens_onnx"]) if report["patch_tokens_onnx"] else None,
        notes=report["notes"],
        validation_status=report["validation_status"],
        validated_batches=report["validated_batches"],
        embedding_metrics=report["embedding_metrics"],
        patch_tokens_metrics=report["patch_tokens_metrics"],
    )


def _default_report(output_dir: Path) -> dict[str, Any]:
    return {
        "status": "blocked",
        "embedding_onnx": None,
        "patch_tokens_onnx": None,
        "notes": ["backend_required_for_export"],
        "validation_status": "not_run",
        "validated_batches": [],
        "embedding_metrics": None,
        "patch_tokens_metrics": None,
    }


def _serialize_payload(payload: dict[str, Any]) -> dict[str, Any]:
    return {
        "status": payload["status"],
        "embedding_onnx": str(payload["embedding_onnx"]) if payload.get("embedding_onnx") else None,
        "patch_tokens_onnx": str(payload["patch_tokens_onnx"]) if payload.get("patch_tokens_onnx") else None,
        "notes": list(payload.get("notes", [])),
        "validation_status": payload.get("validation_status", "not_run"),
        "validated_batches": list(payload.get("validated_batches", [])),
        "embedding_metrics": payload.get("embedding_metrics"),
        "patch_tokens_metrics": payload.get("patch_tokens_metrics"),
    }


def attempt_rknn_conversion(
    onnx_path: Path,
    output_dir: Path,
    model_name: str,
    target: str = "rk3588",
) -> dict[str, Any]:
    report = {
        "status": "blocked",
        "model_name": model_name,
        "target": target,
        "onnx_path": str(onnx_path),
        "rknn_path": None,
        "notes": [],
    }
    try:
        from rknn.api import RKNN
    except ModuleNotFoundError:
        report["notes"].append("missing_dependency:rknn_toolkit2")
        return report

    output_dir.mkdir(parents=True, exist_ok=True)
    rknn_path = output_dir / f"{model_name}.rknn"
    log_path = output_dir / f"{model_name}.log"
    rknn = RKNN(verbose=False)
    try:
        rknn.config(target_platform=target)
        load_status = rknn.load_onnx(model=str(onnx_path))
        if load_status != 0:
            report["notes"].append(f"load_onnx_failed:{load_status}")
            return report
        build_status = rknn.build(do_quantization=False)
        if build_status != 0:
            report["notes"].append(f"build_failed:{build_status}")
            return report
        export_status = rknn.export_rknn(str(rknn_path))
        if export_status != 0:
            report["notes"].append(f"export_failed:{export_status}")
            return report
        report["status"] = "ready"
        report["rknn_path"] = str(rknn_path)
        report["notes"].append("rknn_export_complete")
        log_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
        return report
    except Exception as exc:  # pragma: no cover - exercised only with RKNN toolkit present
        report["notes"].append(f"exception:{type(exc).__name__}:{exc}")
        return report
    finally:
        rknn.release()
