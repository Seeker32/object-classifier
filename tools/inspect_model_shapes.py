from __future__ import annotations

import argparse
import json
from pathlib import Path

from object_classifier.config import ModelConfig
from object_classifier.features import (
    _run_feature_forward,
    _unpack_model_outputs,
    create_backend,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="inspect-model-shapes")
    parser.add_argument("--embedding-onnx", default="data/object-classifier/export/embedding.onnx")
    parser.add_argument("--patch-tokens-onnx", default="data/object-classifier/export/patch_tokens.onnx")
    parser.add_argument("--provider", choices=("torchhub", "huggingface", "statistics"), default="torchhub")
    parser.add_argument("--model-id", default="dinov3_vits16")
    parser.add_argument("--repo-dir", default=None)
    parser.add_argument("--weights-dir", default=None)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    config = ModelConfig(
        backend="pytorch",
        provider=args.provider,
        model_name=args.model_id,
        input_size=(224, 224),
        embedding_dim=384,
        repo_dir=Path(args.repo_dir) if args.repo_dir else None,
        weights_dir=Path(args.weights_dir) if args.weights_dir else None,
    )
    backend = create_backend("pytorch", config)
    payload = {
        "embedding_onnx": _inspect_onnx_model(Path(args.embedding_onnx)),
        "patch_tokens_onnx": _inspect_onnx_model(Path(args.patch_tokens_onnx)),
        "pytorch": _inspect_pytorch_model(backend),
    }
    print(json.dumps(payload))
    return 0


def _inspect_onnx_model(path: Path) -> dict[str, object]:
    import onnx
    import onnxruntime as ort
    import numpy as np

    model = onnx.load(path)
    graph = model.graph
    session = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    inputs = []
    ort_inputs = {}

    for value_info in graph.input:
        shape = [_concrete_dim(dim) for dim in value_info.type.tensor_type.shape.dim]
        inputs.append({"name": value_info.name, "shape": shape})
        ort_inputs[value_info.name] = np.zeros(shape, dtype=np.float32)

    runtime_outputs = session.run(None, ort_inputs)
    outputs = []
    for value_info, output in zip(graph.output, runtime_outputs, strict=True):
        outputs.append({"name": value_info.name, "shape": list(output.shape)})

    return {
        "path": str(path),
        "inputs": inputs,
        "outputs": outputs,
    }


def _inspect_pytorch_model(backend) -> dict[str, object]:
    import torch

    session = backend.session or backend._load_default_session()
    module = session.module
    height, width = backend.config.input_size
    sample = torch.randn(1, 3, height, width, dtype=torch.float32)

    module.eval()
    with torch.no_grad():
        outputs = _run_feature_forward(module, sample)
    embedding, patch_tokens = _unpack_model_outputs(outputs)
    return {
        "input_shape": list(sample.shape),
        "outputs": [
            {"name": "global_embedding", "shape": list(embedding.shape)},
            {"name": "patch_tokens", "shape": list(patch_tokens.shape)},
        ],
    }


def _format_value_info(value_info) -> dict[str, object]:
    return {
        "name": value_info.name,
        "shape": [_format_dim(dim) for dim in value_info.type.tensor_type.shape.dim],
    }


def _format_dim(dim) -> int | str:
    if dim.dim_param:
        return dim.dim_param
    if dim.dim_value:
        return int(dim.dim_value)
    return "?"


def _concrete_dim(dim) -> int:
    value = _format_dim(dim)
    if isinstance(value, int):
        return value
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
