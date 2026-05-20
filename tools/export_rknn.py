from __future__ import annotations

import argparse
import json
from pathlib import Path

from object_classifier.export import attempt_rknn_conversion


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="export-rknn")
    parser.add_argument("--output-dir", default="data/object-classifier/export")
    parser.add_argument("--target", default="rk3588")
    parser.add_argument("--embedding-onnx", default="data/object-classifier/export/embedding.onnx")
    parser.add_argument("--patch-tokens-onnx", default="data/object-classifier/export/patch_tokens.onnx")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    embedding_onnx = Path(args.embedding_onnx)
    patch_tokens_onnx = Path(args.patch_tokens_onnx)
    conversion_reports = []
    notes: list[str] = []

    if embedding_onnx.exists():
        conversion_reports.append(
            attempt_rknn_conversion(embedding_onnx, output_dir, "embedding", target=args.target)
        )
    else:
        notes.append(f"missing_onnx:embedding:{embedding_onnx}")

    if patch_tokens_onnx.exists():
        conversion_reports.append(
            attempt_rknn_conversion(patch_tokens_onnx, output_dir, "patch_tokens", target=args.target)
        )
    else:
        notes.append(f"missing_onnx:patch_tokens:{patch_tokens_onnx}")

    status = _resolve_status(conversion_reports, notes)
    report_path = output_dir / "rknn_report.json"
    payload = {
        "status": status,
        "report_path": str(report_path),
        "embedding_onnx": str(embedding_onnx) if embedding_onnx.exists() else None,
        "patch_tokens_onnx": str(patch_tokens_onnx) if patch_tokens_onnx.exists() else None,
        "notes": notes,
        "rknn": conversion_reports,
    }
    report_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload))
    return 0


def _resolve_status(conversion_reports: list[dict[str, object]], notes: list[str]) -> str:
    if not conversion_reports:
        return "blocked"
    if notes:
        return "partial"
    if all(report.get("status") == "ready" for report in conversion_reports):
        return "ready"
    if any(report.get("status") == "ready" for report in conversion_reports):
        return "partial"
    return "blocked"


if __name__ == "__main__":
    raise SystemExit(main())
