from __future__ import annotations

import argparse
import json
from pathlib import Path

from object_classifier.config import ModelConfig, ROIBox
from object_classifier.export import attempt_rknn_conversion, export_onnx_artifacts
from object_classifier.features import create_backend


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="export-rknn")
    parser.add_argument("--output-dir", default="var/object-classifier/export")
    parser.add_argument("--provider", choices=("torchhub", "huggingface", "statistics"), default="torchhub")
    parser.add_argument("--model-id", default="dinov3_vits16")
    parser.add_argument("--repo-dir", default=None)
    parser.add_argument("--weights-dir", default=None)
    parser.add_argument("--target", default="rk3588")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    output_dir = Path(args.output_dir)
    config = ModelConfig(
        backend="pytorch",
        provider=args.provider,
        model_name=args.model_id,
        input_size=(224, 224),
        roi_box=ROIBox(0, 0, 224, 224),
        embedding_dim=384,
        repo_dir=Path(args.repo_dir) if args.repo_dir else None,
        weights_dir=Path(args.weights_dir) if args.weights_dir else None,
    )
    backend = create_backend("pytorch", config)
    artifacts = export_onnx_artifacts(output_dir, backend=backend)
    conversion_reports = []
    if artifacts.embedding_onnx:
        conversion_reports.append(
            attempt_rknn_conversion(artifacts.embedding_onnx, output_dir, "embedding", target=args.target)
        )
    if artifacts.patch_tokens_onnx:
        conversion_reports.append(
            attempt_rknn_conversion(artifacts.patch_tokens_onnx, output_dir, "patch_tokens", target=args.target)
        )
    payload = {
        "status": artifacts.status,
        "report_path": str(artifacts.report_path),
        "embedding_onnx": str(artifacts.embedding_onnx) if artifacts.embedding_onnx else None,
        "patch_tokens_onnx": str(artifacts.patch_tokens_onnx) if artifacts.patch_tokens_onnx else None,
        "notes": artifacts.notes,
        "rknn": conversion_reports,
    }
    (output_dir / "rknn_report.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
