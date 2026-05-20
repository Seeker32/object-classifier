from __future__ import annotations

import argparse
import json
from dataclasses import asdict, is_dataclass
from pathlib import Path

from .config import FeatureCacheConfig, PipelineConfig, StorageConfig, default_config
from .export import export_onnx_artifacts
from .features import create_backend
from .pipeline import ObjectClassifierPipeline


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
    subparsers = parser.add_subparsers(dest="command", required=True)

    register = subparsers.add_parser("register")
    register.add_argument("sku_name")
    register.add_argument("images", nargs="+")

    identify = subparsers.add_parser("identify")
    identify.add_argument("image")

    export = subparsers.add_parser("export")
    export.add_argument("--output-dir", default="data/object-classifier/export")

    return parser


def build_config(args: argparse.Namespace) -> PipelineConfig:
    base = default_config()
    cache_dir = Path(args.cache_dir) if args.cache_dir else Path(args.storage_root) / "cache"
    return PipelineConfig(
        model=base.model.__class__(
            backend=args.backend,
            provider=args.provider,
            input_size=base.model.input_size,
            roi_box=base.model.roi_box,
            model_name=args.model_id or base.model.model_name,
            embedding_dim=base.model.embedding_dim,
            device=args.device,
            repo_dir=Path(args.repo_dir) if args.repo_dir else None,
            weights_dir=Path(args.weights_dir) if args.weights_dir else None,
        ),
        quality=base.quality,
        decision=base.decision,
        storage=StorageConfig(root=Path(args.storage_root)),
        cache=FeatureCacheConfig(enabled=True, cache_dir=cache_dir),
        topk=base.topk,
    )


def build_pipeline(args: argparse.Namespace) -> ObjectClassifierPipeline:
    config = build_config(args)
    backend = create_backend(args.backend, config.model)
    return ObjectClassifierPipeline(config=config, backend=backend)


def handle_register(args: argparse.Namespace) -> int:
    pipeline = build_pipeline(args)
    result = pipeline.register(args.sku_name, [Path(image) for image in args.images])
    payload = {
        "sku_id": result.sku.sku_id,
        "sku_name": result.sku.sku_name,
        "sample_ids": [sample.sample_id for sample in result.samples],
        "warnings": result.warnings,
    }
    print(json.dumps(payload))
    return 0


def handle_identify(args: argparse.Namespace) -> int:
    pipeline = build_pipeline(args)
    result = pipeline.identify(Path(args.image))
    print(json.dumps(_to_jsonable(result)))
    return 0


def handle_export(args: argparse.Namespace) -> int:
    artifacts = export_onnx_artifacts(Path(args.output_dir))
    print(json.dumps(_to_jsonable(artifacts)))
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    handlers = {
        "register": handle_register,
        "identify": handle_identify,
        "export": handle_export,
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
