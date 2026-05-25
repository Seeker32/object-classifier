from __future__ import annotations

import argparse
import json
import shutil
import sys
from dataclasses import asdict, is_dataclass
from pathlib import Path

from .config import StorageConfig
from .export import export_onnx_artifacts
from .runtime import build_pipeline
from .web import create_app


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="object-classifier")
    parser.add_argument("--storage-root", default="data/object-classifier")
    parser.add_argument("--cache-dir", default=None)
    parser.add_argument("--backend", choices=("pytorch", "rknn"), default="pytorch")
    parser.add_argument("--provider", choices=("statistics", "huggingface", "torchhub"), default="statistics")
    parser.add_argument("--model-id", default=None)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--repo-dir", default=None)
    parser.add_argument("--weights-dir", default=None)
    parser.add_argument("--rknn-target", default="rk3588")
    subparsers = parser.add_subparsers(dest="command", required=True)

    register = subparsers.add_parser("register")
    register.add_argument("sku_name")
    register.add_argument("images", nargs="+")

    identify = subparsers.add_parser("identify")
    identify.add_argument("image")

    export = subparsers.add_parser("export")
    export.add_argument("--output-dir", default="data/object-classifier/export")

    reset = subparsers.add_parser("reset")
    reset.add_argument("--yes", action="store_true")

    serve = subparsers.add_parser("serve")
    serve.add_argument("--host", default="127.0.0.1")
    serve.add_argument("--port", type=int, default=9000)

    return parser


def handle_register(args: argparse.Namespace) -> int:
    pipeline = build_pipeline(
        storage_root=args.storage_root,
        cache_dir=args.cache_dir,
        backend=args.backend,
        provider=args.provider,
        model_id=args.model_id,
        device=args.device,
        repo_dir=args.repo_dir,
        weights_dir=args.weights_dir,
        rknn_target=args.rknn_target,
    )
    result = pipeline.register(args.sku_name, [Path(image) for image in args.images])
    payload = {
        "decision": result.decision,
        "sku_id": result.sku.sku_id if result.sku else None,
        "sku_name": result.sku.sku_name if result.sku else None,
        "sample_ids": [sample.sample_id for sample in result.samples],
        "warnings": result.warnings,
        "reasons": result.reasons,
    }
    print(json.dumps(payload))
    return 0


def handle_identify(args: argparse.Namespace) -> int:
    pipeline = build_pipeline(
        storage_root=args.storage_root,
        cache_dir=args.cache_dir,
        backend=args.backend,
        provider=args.provider,
        model_id=args.model_id,
        device=args.device,
        repo_dir=args.repo_dir,
        weights_dir=args.weights_dir,
        rknn_target=args.rknn_target,
    )
    result = pipeline.identify(Path(args.image))
    print(json.dumps(_to_jsonable(result)))
    return 0


def handle_export(args: argparse.Namespace) -> int:
    artifacts = export_onnx_artifacts(Path(args.output_dir))
    print(json.dumps(_to_jsonable(artifacts)))
    return 0


def handle_reset(args: argparse.Namespace) -> int:
    if not args.yes:
        print("reset is destructive; --yes is required", file=sys.stderr)
        return 1

    storage_root = Path(args.storage_root)
    if not storage_root.exists():
        print(
            json.dumps(
                {
                    "status": "noop",
                    "storage_root": str(storage_root),
                    "preserved_paths": [],
                    "removed_paths": [],
                }
            )
        )
        return 0

    config = StorageConfig(root=storage_root)
    preserved_paths: list[Path] = []
    removed_paths: list[Path] = []
    export_dir = storage_root / "export"
    if export_dir.exists():
        preserved_paths.append(export_dir)

    removal_targets = [
        config.database_path,
        config.metadata_root,
        config.feature_root,
        config.patch_token_root,
        storage_root / "cache",
        storage_root / config.faiss_index_file,
        storage_root / config.faiss_mapping_file,
    ]
    for target in removal_targets:
        if not target.exists():
            continue
        if target.is_dir():
            shutil.rmtree(target)
        else:
            target.unlink()
        removed_paths.append(target)

    print(
        json.dumps(
            {
                "status": "ok",
                "storage_root": str(storage_root),
                "preserved_paths": [str(path) for path in preserved_paths],
                "removed_paths": [str(path) for path in removed_paths],
            }
        )
    )
    return 0


def handle_serve(args: argparse.Namespace) -> int:
    import uvicorn

    app = create_app(
        storage_root=args.storage_root,
        cache_dir=args.cache_dir,
        backend=args.backend,
        provider=args.provider,
        model_id=args.model_id,
        device=args.device,
        repo_dir=args.repo_dir,
        weights_dir=args.weights_dir,
        rknn_target=args.rknn_target,
    )
    uvicorn.run(app, host=args.host, port=args.port)
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    handlers = {
        "register": handle_register,
        "identify": handle_identify,
        "export": handle_export,
        "reset": handle_reset,
        "serve": handle_serve,
    }
    return handlers[args.command](args)


def _to_jsonable(value):
    if is_dataclass(value):
        return {key: _to_jsonable(item) for key, item in asdict(value).items()}
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {key: _to_jsonable(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_to_jsonable(item) for item in value]
    return value
