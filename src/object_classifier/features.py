from __future__ import annotations

import importlib
import sys
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
from typing import Any, Protocol

import numpy as np

from .config import ModelConfig
from .schemas import FeatureBundle, NormalizedROI


class InferenceSession(Protocol):
    def infer(self, batch: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Return global embedding and patch tokens."""


class ExportableSession(Protocol):
    def export_onnx(
        self,
        output_dir: Path,
        input_size: tuple[int, int],
        *,
        validate: bool = True,
    ) -> dict[str, Any]:
        """Export embedding and patch-token paths to ONNX artifacts."""


@dataclass
class BaseFeatureBackend:
    config: ModelConfig
    session: InferenceSession | None = None
    backend_name: str = "base"

    def extract(self, roi: NormalizedROI) -> FeatureBundle:
        batch = preprocess_roi(roi.image, self.config.input_size)
        session = self.session or self._load_default_session()
        embedding, patch_tokens = session.infer(batch)
        return FeatureBundle(
            global_embedding=l2_normalize(np.asarray(embedding, dtype=np.float32).reshape(-1)),
            patch_tokens=np.asarray(patch_tokens, dtype=np.float32),
            backend=self.backend_name,
        )

    def cache_key(self, roi: NormalizedROI) -> str:
        digest = sha256()
        digest.update(roi.image.tobytes())
        digest.update(self.backend_name.encode("utf-8"))
        digest.update(str(self.config.input_size).encode("utf-8"))
        return digest.hexdigest()

    def _load_default_session(self) -> InferenceSession:
        raise RuntimeError(f"{self.backend_name} backend requires an explicit inference session")


@dataclass
class PyTorchFeatureBackend(BaseFeatureBackend):
    backend_name: str = "pytorch"


@dataclass
class RKNNFeatureBackend(BaseFeatureBackend):
    backend_name: str = "rknn"


class TorchModuleSession:
    def __init__(self, module, device: str = "cpu") -> None:
        self.module = module
        self.device = device

    def infer(self, batch: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        import torch

        self.module.eval()
        tensor = torch.from_numpy(batch).to(self.device)
        with torch.no_grad():
            outputs = _run_feature_forward(self.module, tensor)
        embedding, patch_tokens = _unpack_model_outputs(outputs)
        return (
            embedding.detach().cpu().numpy().reshape(-1),
            patch_tokens.detach().cpu().numpy().reshape(-1, patch_tokens.shape[-1]),
        )

    def export_onnx(self, output_dir: Path, input_size: tuple[int, int], *, validate: bool = True) -> dict[str, Any]:
        import torch

        output_dir.mkdir(parents=True, exist_ok=True)
        embedding_path = output_dir / "embedding.onnx"
        patch_tokens_path = output_dir / "patch_tokens.onnx"
        sample = torch.randn(1, 3, input_size[0], input_size[1], dtype=torch.float32)

        embedding_wrapper, patch_wrapper = _build_export_wrappers(self.module, sample)

        torch.onnx.export(
            embedding_wrapper,
            sample,
            embedding_path,
            input_names=["pixel_values"],
            output_names=["global_embedding"],
            opset_version=18,
        )
        torch.onnx.export(
            patch_wrapper,
            sample,
            patch_tokens_path,
            input_names=["pixel_values"],
            output_names=["patch_tokens"],
            opset_version=18,
        )
        validation = _validate_onnx_export(
            module=self.module,
            input_size=input_size,
            embedding_path=embedding_path,
            patch_tokens_path=patch_tokens_path,
            enabled=validate,
        )
        notes = ["torch_onnx_export_complete", *validation["notes"]]
        return {
            "status": "partial" if validation["validation_status"] == "failed" else "ready",
            "embedding_onnx": embedding_path,
            "patch_tokens_onnx": patch_tokens_path,
            "notes": notes,
            "validation_status": validation["validation_status"],
            "validated_batches": validation["validated_batches"],
            "embedding_metrics": validation["embedding_metrics"],
            "patch_tokens_metrics": validation["patch_tokens_metrics"],
        }


class StatisticsFeatureSession:
    def infer(self, batch: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        channels = batch.mean(axis=(2, 3)).reshape(-1).astype(np.float32)
        quadrants = np.stack(
            [
                batch[:, :, : batch.shape[2] // 2, : batch.shape[3] // 2].mean(axis=(2, 3)).reshape(-1),
                batch[:, :, : batch.shape[2] // 2, batch.shape[3] // 2 :].mean(axis=(2, 3)).reshape(-1),
                batch[:, :, batch.shape[2] // 2 :, : batch.shape[3] // 2].mean(axis=(2, 3)).reshape(-1),
                batch[:, :, batch.shape[2] // 2 :, batch.shape[3] // 2 :].mean(axis=(2, 3)).reshape(-1),
            ]
        ).astype(np.float32)
        return channels, quadrants


class RKNNRuntimeSession:
    def __init__(
        self,
        embedding_model_path: Path,
        patch_model_path: Path | None,
        *,
        runtime_factory=None,
    ) -> None:
        self.embedding_model_path = Path(embedding_model_path)
        self.patch_model_path = Path(patch_model_path) if patch_model_path is not None else None
        self.runtime_factory = runtime_factory
        self._embedding_runtime = None
        self._patch_runtime = None

    def infer(self, batch: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        embedding_runtime = self._embedding_runtime or self._build_runtime(self.embedding_model_path)
        self._embedding_runtime = embedding_runtime
        embedding_outputs = embedding_runtime.inference(inputs=[batch])

        embedding = np.asarray(embedding_outputs[0], dtype=np.float32).reshape(-1)
        if self.patch_model_path is None:
            patch_tokens = np.asarray(batch.reshape(batch.shape[1], -1).T, dtype=np.float32)
            return embedding, patch_tokens

        if self.patch_model_path == self.embedding_model_path and len(embedding_outputs) > 1:
            patch_tokens = np.asarray(embedding_outputs[1], dtype=np.float32).reshape(-1, embedding_outputs[1].shape[-1])
            return embedding, patch_tokens

        patch_runtime = self._patch_runtime or self._build_runtime(self.patch_model_path)
        self._patch_runtime = patch_runtime
        patch_outputs = patch_runtime.inference(inputs=[batch])
        patch_tokens = np.asarray(patch_outputs[0], dtype=np.float32).reshape(-1, patch_outputs[0].shape[-1])
        return embedding, patch_tokens

    def _build_runtime(self, model_path: Path):
        runtime_cls = self.runtime_factory or _default_rknn_runtime_factory()
        runtime = runtime_cls()
        load_status = runtime.load_rknn(str(model_path))
        if load_status != 0:
            raise RuntimeError(f"Failed to load RKNN artifact: {model_path}")
        init_status = runtime.init_runtime()
        if init_status != 0:
            raise RuntimeError(f"Failed to init RKNN runtime: {model_path}")
        return runtime


def create_backend(backend_name: str, config: ModelConfig, session: InferenceSession | None = None) -> BaseFeatureBackend:
    active_session = session or _build_default_session(config)
    if backend_name == "pytorch":
        return PyTorchFeatureBackend(config=config, session=active_session)
    if backend_name == "rknn":
        return RKNNFeatureBackend(config=config, session=active_session)
    raise ValueError(f"Unsupported backend: {backend_name}")


def preprocess_roi(image: np.ndarray, output_size: tuple[int, int]) -> np.ndarray:
    height, width = output_size
    if image.shape[:2] != (height, width):
        from PIL import Image

        resized = Image.fromarray(image).resize((width, height), Image.Resampling.BILINEAR)
        image = np.asarray(resized)
    normalized = image.astype(np.float32) / 255.0
    chw = np.transpose(normalized, (2, 0, 1))
    return chw[None, ...]


def l2_normalize(vector: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(vector))
    if norm == 0.0:
        return vector
    return vector / norm


def save_feature_cache(path: Path, bundle: FeatureBundle) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        global_embedding=bundle.global_embedding,
        patch_tokens=bundle.patch_tokens,
        backend=bundle.backend,
    )


def load_feature_cache(path: Path) -> FeatureBundle:
    with np.load(path, allow_pickle=False) as data:
        return FeatureBundle(
            global_embedding=data["global_embedding"].astype(np.float32),
            patch_tokens=data["patch_tokens"].astype(np.float32),
            backend=str(data["backend"]),
        )


def export_backend_to_onnx(backend: BaseFeatureBackend, output_dir: Path, *, validate: bool = True) -> dict[str, Any]:
    session = backend.session or backend._load_default_session()
    if not hasattr(session, "export_onnx"):
        return {
            "status": "blocked",
            "embedding_onnx": None,
            "patch_tokens_onnx": None,
            "notes": [f"session_not_exportable:{type(session).__name__}"],
            "validation_status": "not_run",
            "validated_batches": [],
            "embedding_metrics": None,
            "patch_tokens_metrics": None,
        }
    return session.export_onnx(output_dir, backend.config.input_size, validate=validate)


def _unpack_model_outputs(outputs):
    if isinstance(outputs, tuple) and len(outputs) == 2:
        return outputs
    if hasattr(outputs, "ndim") and getattr(outputs, "ndim", None) == 2:
        return outputs, outputs.unsqueeze(1)
    if isinstance(outputs, dict):
        if "x_norm_clstoken" in outputs and "x_norm_patchtokens" in outputs:
            return outputs["x_norm_clstoken"], outputs["x_norm_patchtokens"]
        if "embedding" in outputs and "patch_tokens" in outputs:
            return outputs["embedding"], outputs["patch_tokens"]
        if "pooler_output" in outputs and "last_hidden_state" in outputs:
            hidden = outputs["last_hidden_state"]
            return outputs["pooler_output"], hidden[:, 1:, :]
    if hasattr(outputs, "pooler_output") and hasattr(outputs, "last_hidden_state"):
        hidden = outputs.last_hidden_state
        return outputs.pooler_output, hidden[:, 1:, :]
    raise ValueError("Unsupported model output format for feature extraction")


class _EmbeddingExportWrapper:
    def __new__(cls, module):
        import torch

        class _Wrapped(torch.nn.Module):
            def __init__(self, wrapped_module) -> None:
                super().__init__()
                self.module = wrapped_module

            def forward(self, pixel_values):
                embedding, _ = _unpack_model_outputs(_run_feature_forward(self.module, pixel_values))
                return embedding

        return _Wrapped(module)


class _PatchTokenExportWrapper:
    def __new__(cls, module):
        import torch

        class _Wrapped(torch.nn.Module):
            def __init__(self, wrapped_module) -> None:
                super().__init__()
                self.module = wrapped_module

            def forward(self, pixel_values):
                _, patch_tokens = _unpack_model_outputs(_run_feature_forward(self.module, pixel_values))
                return patch_tokens

        return _Wrapped(module)


class _FrozenRopeEmbeddingExportWrapper:
    def __new__(cls, module, frozen_rope):
        import torch

        class _Wrapped(torch.nn.Module):
            def __init__(self, wrapped_module, rope) -> None:
                super().__init__()
                self.module = wrapped_module
                sin, cos = rope
                self.register_buffer("rope_sin", sin)
                self.register_buffer("rope_cos", cos)

            def forward(self, pixel_values):
                embedding, _ = _unpack_model_outputs(
                    _forward_features_with_frozen_rope(self.module, pixel_values, (self.rope_sin, self.rope_cos))
                )
                return embedding

        return _Wrapped(module, frozen_rope)


class _FrozenRopePatchTokenExportWrapper:
    def __new__(cls, module, frozen_rope):
        import torch

        class _Wrapped(torch.nn.Module):
            def __init__(self, wrapped_module, rope) -> None:
                super().__init__()
                self.module = wrapped_module
                sin, cos = rope
                self.register_buffer("rope_sin", sin)
                self.register_buffer("rope_cos", cos)

            def forward(self, pixel_values):
                _, patch_tokens = _unpack_model_outputs(
                    _forward_features_with_frozen_rope(self.module, pixel_values, (self.rope_sin, self.rope_cos))
                )
                return patch_tokens

        return _Wrapped(module, frozen_rope)


def _build_default_session(config: ModelConfig) -> InferenceSession:
    if config.backend == "rknn":
        embedding_path = config.rknn_embedding_path or _resolve_default_rknn_path(config, "embedding.rknn")
        patch_path = config.rknn_patch_tokens_path
        if patch_path is None:
            candidate_patch = _resolve_default_rknn_path(config, "patch_tokens.rknn", required=False)
            patch_path = candidate_patch
        return RKNNRuntimeSession(embedding_model_path=embedding_path, patch_model_path=patch_path)
    if config.provider == "statistics":
        return StatisticsFeatureSession()
    if config.provider == "huggingface":
        return _load_huggingface_session(config)
    if config.provider == "torchhub":
        return _load_torchhub_session(config)
    raise ValueError(f"Unsupported provider: {config.provider}")


def _load_huggingface_session(config: ModelConfig) -> TorchModuleSession:
    from transformers import AutoModel

    model = AutoModel.from_pretrained(config.model_name)
    return TorchModuleSession(module=model, device=config.device)


def _load_torchhub_session(config: ModelConfig) -> TorchModuleSession:
    import torch

    _ensure_torch_amp_compat(torch)
    repo_dir = _resolve_repo_dir(config)
    weights_path = _resolve_weights_path(config)
    hub_name = _resolve_torchhub_entry(config.model_name)
    module = _load_dinov3_backbone(repo_dir, hub_name, str(weights_path), torch)
    return TorchModuleSession(module=module, device=config.device)


def _ensure_torch_amp_compat(torch_module) -> None:
    amp_module = getattr(torch_module, "amp", None)
    cuda_amp_module = getattr(getattr(torch_module, "cuda", None), "amp", None)
    if amp_module is None or cuda_amp_module is None:
        return
    if not hasattr(amp_module, "custom_fwd") and hasattr(cuda_amp_module, "custom_fwd"):
        legacy_custom_fwd = cuda_amp_module.custom_fwd

        def custom_fwd(func=None, *, device_type=None, cast_inputs=None):
            _ = device_type
            return legacy_custom_fwd(func, cast_inputs=cast_inputs)

        amp_module.custom_fwd = custom_fwd
    if not hasattr(amp_module, "custom_bwd") and hasattr(cuda_amp_module, "custom_bwd"):
        legacy_custom_bwd = cuda_amp_module.custom_bwd

        def custom_bwd(func=None, *, device_type=None):
            _ = device_type
            if func is None:
                return lambda wrapped: legacy_custom_bwd(wrapped)
            return legacy_custom_bwd(func)

        amp_module.custom_bwd = custom_bwd


def _load_dinov3_backbone(repo_dir: Path, hub_name: str, weights_path: str, torch_module):
    repo_dir = repo_dir.resolve()
    repo_path = str(repo_dir)
    added_path = False
    if repo_path not in sys.path:
        sys.path.insert(0, repo_path)
        added_path = True
    try:
        backbones = importlib.import_module("dinov3.hub.backbones")
        factory = getattr(backbones, hub_name)
        return factory(weights=weights_path)
    except Exception:
        hub_load = getattr(getattr(torch_module, "hub", None), "load", None)
        if hub_load is None:
            raise
        return hub_load(
            str(repo_dir),
            hub_name,
            source="local",
            weights=weights_path,
        )
    finally:
        if added_path and repo_path in sys.path:
            sys.path.remove(repo_path)


def _resolve_repo_dir(config: ModelConfig) -> Path:
    if config.repo_dir is not None:
        return config.repo_dir
    candidates = [
        _project_root() / "vendor" / "dinov3",
        _project_root() / "third_party" / "dinov3",
        _repository_root() / "vendor" / "dinov3",
        _repository_root() / "third_party" / "dinov3",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError("DINOv3 repo dir not found; pass --repo-dir or vendor the official repo locally")


def _resolve_weights_path(config: ModelConfig) -> Path:
    hub_name = _resolve_torchhub_entry(config.model_name)
    weights_dir = config.weights_dir or _find_weights_dir()
    pattern = f"{hub_name}_pretrain_lvd1689m-*.pth"
    matches = sorted(weights_dir.glob(pattern))
    if not matches:
        raise FileNotFoundError(f"No weights matching {pattern} under {weights_dir}")
    return matches[0]


def _find_weights_dir() -> Path:
    candidates = [
        _project_root() / "models",
        _repository_root() / "models",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError("Model weights directory not found; pass --weights-dir explicitly")


def _resolve_torchhub_entry(model_name: str) -> str:
    normalized = model_name.replace("-", "_").lower()
    if normalized in {"dinov3_vits16", "dinov3_vits16plus"}:
        return normalized
    if "vits16plus" in normalized:
        return "dinov3_vits16plus"
    if "vits16" in normalized:
        return "dinov3_vits16"
    raise ValueError(f"Unable to infer torchhub entry from model_name: {model_name}")


def _project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _repository_root() -> Path:
    return Path(__file__).resolve().parents[4]


def _resolve_default_rknn_path(config: ModelConfig, filename: str, *, required: bool = True) -> Path | None:
    candidates = [
        (config.weights_dir / filename) if config.weights_dir is not None else None,
        _project_root() / "data" / "object-classifier" / "export" / filename,
        _repository_root() / "data" / "object-classifier" / "export" / filename,
    ]
    for candidate in candidates:
        if candidate is not None and candidate.exists():
            return candidate
    if required:
        raise FileNotFoundError(f"RKNN artifact not found: {filename}")
    return None


def _default_rknn_runtime_factory():
    try:
        from rknnlite.api import RKNNLite

        return RKNNLite
    except ModuleNotFoundError:
        from rknn.api import RKNN

        return RKNN


def _run_feature_forward(module, pixel_values):
    if hasattr(module, "forward_features"):
        return module.forward_features(pixel_values)
    return module(pixel_values)


def _build_export_wrappers(module, sample):
    if _supports_frozen_rope_export(module):
        frozen_rope = _precompute_frozen_rope(module, sample)
        return (
            _FrozenRopeEmbeddingExportWrapper(module, frozen_rope).eval(),
            _FrozenRopePatchTokenExportWrapper(module, frozen_rope).eval(),
        )
    return _EmbeddingExportWrapper(module).eval(), _PatchTokenExportWrapper(module).eval()


def _supports_frozen_rope_export(module) -> bool:
    return (
        getattr(module, "rope_embed", None) is not None
        and hasattr(module, "prepare_tokens_with_masks")
        and hasattr(module, "blocks")
        and hasattr(module, "norm")
        and hasattr(module, "n_storage_tokens")
    )


def _precompute_frozen_rope(module, sample):
    import torch

    module.eval()
    with torch.no_grad():
        _, (height, width) = module.prepare_tokens_with_masks(sample)
        return module.rope_embed(H=height, W=width)


def _forward_features_with_frozen_rope(module, pixel_values, rope):
    x, _ = module.prepare_tokens_with_masks(pixel_values)
    for block in module.blocks:
        x = block(x, rope)

    if getattr(module, "untie_cls_and_patch_norms", False) or getattr(module, "untie_global_and_local_cls_norm", False):
        if getattr(module, "untie_cls_and_patch_norms", False):
            x_norm_cls_reg = module.cls_norm(x[:, : module.n_storage_tokens + 1])
        else:
            x_norm_cls_reg = module.norm(x[:, : module.n_storage_tokens + 1])
        x_norm_patch = module.norm(x[:, module.n_storage_tokens + 1 :])
    else:
        x_norm = module.norm(x)
        x_norm_cls_reg = x_norm[:, : module.n_storage_tokens + 1]
        x_norm_patch = x_norm[:, module.n_storage_tokens + 1 :]

    return {
        "x_norm_clstoken": x_norm_cls_reg[:, 0],
        "x_storage_tokens": x_norm_cls_reg[:, 1:],
        "x_norm_patchtokens": x_norm_patch,
        "x_prenorm": x,
        "masks": None,
    }


def _validate_onnx_export(
    module,
    input_size: tuple[int, int],
    embedding_path: Path,
    patch_tokens_path: Path,
    *,
    enabled: bool,
) -> dict[str, Any]:
    if not enabled:
        return {
            "validation_status": "skipped",
            "validated_batches": [],
            "embedding_metrics": None,
            "patch_tokens_metrics": None,
            "notes": ["validation_skipped:user_request"],
        }

    unsupported_ops = sorted(
        set(_find_unsupported_onnx_ops(embedding_path) + _find_unsupported_onnx_ops(patch_tokens_path))
    )
    if unsupported_ops:
        return {
            "validation_status": "failed",
            "validated_batches": [],
            "embedding_metrics": None,
            "patch_tokens_metrics": None,
            "notes": [f"unsupported_control_flow:{op}" for op in unsupported_ops],
        }

    try:
        import onnxruntime as ort
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "ONNX validation requires onnxruntime, which is not installed. "
            "Install it with: pip install 'object-classifier[export]'"
        ) from exc
    import torch

    module.eval()
    embedding_wrapper = _EmbeddingExportWrapper(module).eval()
    patch_wrapper = _PatchTokenExportWrapper(module).eval()
    embedding_session = ort.InferenceSession(str(embedding_path), providers=["CPUExecutionProvider"])
    patch_tokens_session = ort.InferenceSession(str(patch_tokens_path), providers=["CPUExecutionProvider"])
    batch_sizes = [1]
    embedding_errors: list[tuple[float, float]] = []
    patch_token_errors: list[tuple[float, float]] = []

    with torch.no_grad():
        for batch_size in batch_sizes:
            sample = torch.randn(batch_size, 3, input_size[0], input_size[1], dtype=torch.float32)
            torch_embedding = embedding_wrapper(sample).detach().cpu().numpy()
            torch_patch_tokens = patch_wrapper(sample).detach().cpu().numpy()
            ort_inputs = {"pixel_values": sample.cpu().numpy()}
            onnx_embedding = embedding_session.run(None, ort_inputs)[0]
            onnx_patch_tokens = patch_tokens_session.run(None, ort_inputs)[0]
            embedding_errors.append(_compute_error_metrics(torch_embedding, onnx_embedding))
            patch_token_errors.append(_compute_error_metrics(torch_patch_tokens, onnx_patch_tokens))

    embedding_metrics = _summarize_error_metrics(embedding_errors)
    patch_tokens_metrics = _summarize_error_metrics(patch_token_errors)
    validation_failed = any(
        metric["max_abs_err"] > 1e-4 or metric["mean_abs_err"] > 1e-5
        for metric in (embedding_metrics, patch_tokens_metrics)
    )
    status = "failed" if validation_failed else "passed"
    return {
        "validation_status": status,
        "validated_batches": batch_sizes,
        "embedding_metrics": embedding_metrics,
        "patch_tokens_metrics": patch_tokens_metrics,
        "notes": [f"validation_{status}"],
    }


def _compute_error_metrics(expected: np.ndarray, actual: np.ndarray) -> tuple[float, float]:
    difference = np.abs(expected - actual)
    return float(difference.max(initial=0.0)), float(difference.mean())


def _summarize_error_metrics(metrics: list[tuple[float, float]]) -> dict[str, float]:
    if not metrics:
        return {"max_abs_err": 0.0, "mean_abs_err": 0.0}
    return {
        "max_abs_err": max(metric[0] for metric in metrics),
        "mean_abs_err": max(metric[1] for metric in metrics),
    }


def _find_unsupported_onnx_ops(path: Path) -> list[str]:
    try:
        import onnx
    except ModuleNotFoundError:
        return []

    model = onnx.load(str(path))
    return sorted({node.op_type for node in model.graph.node if node.op_type == "If"})
