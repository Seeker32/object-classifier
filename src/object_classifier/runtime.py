from __future__ import annotations

from pathlib import Path

from .config import FeatureCacheConfig, PipelineConfig, StorageConfig, default_config
from .features import create_backend
from .pipeline import ObjectClassifierPipeline


def build_pipeline(
    *,
    storage_root: str | Path = "data/object-classifier",
    cache_dir: str | Path | None = None,
    backend: str = "pytorch",
    provider: str = "statistics",
    model_id: str | None = None,
    device: str = "cpu",
    repo_dir: str | Path | None = None,
    weights_dir: str | Path | None = None,
    rknn_target: str = "rk3588",
) -> ObjectClassifierPipeline:
    config = build_config(
        storage_root=storage_root,
        cache_dir=cache_dir,
        backend=backend,
        provider=provider,
        model_id=model_id,
        device=device,
        repo_dir=repo_dir,
        weights_dir=weights_dir,
        rknn_target=rknn_target,
    )
    active_backend = create_backend(backend, config.model)
    return ObjectClassifierPipeline(config=config, backend=active_backend)


def build_config(
    *,
    storage_root: str | Path = "data/object-classifier",
    cache_dir: str | Path | None = None,
    backend: str = "pytorch",
    provider: str = "statistics",
    model_id: str | None = None,
    device: str = "cpu",
    repo_dir: str | Path | None = None,
    weights_dir: str | Path | None = None,
    rknn_target: str = "rk3588",
) -> PipelineConfig:
    base = default_config()
    storage_root_path = Path(storage_root)
    cache_root = Path(cache_dir) if cache_dir else storage_root_path / "cache"
    return PipelineConfig(
        model=base.model.__class__(
            backend=backend,
            provider=provider,
            input_size=base.model.input_size,
            roi_box=base.model.roi_box,
            model_name=model_id or base.model.model_name,
            embedding_dim=base.model.embedding_dim,
            device=device,
            repo_dir=Path(repo_dir) if repo_dir else None,
            weights_dir=Path(weights_dir) if weights_dir else None,
            rknn_target=rknn_target,
        ),
        quality=base.quality,
        decision=base.decision,
        storage=StorageConfig(root=storage_root_path),
        cache=FeatureCacheConfig(enabled=True, cache_dir=cache_root),
        topk=base.topk,
    )
