from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


@dataclass(frozen=True)
class ROIBox:
    left: int
    top: int
    right: int
    bottom: int

    @property
    def width(self) -> int:
        return self.right - self.left

    @property
    def height(self) -> int:
        return self.bottom - self.top


@dataclass(frozen=True)
class QualityThresholds:
    min_width: int = 32
    min_height: int = 32
    blur_soft_limit: float = 6.0
    blur_hard_limit: float = 3.0
    dark_ratio_limit: float = 0.65
    bright_ratio_limit: float = 0.65


@dataclass(frozen=True)
class DecisionThresholds:
    absolute_score: float = 0.78
    margin_score: float = 0.05


@dataclass(frozen=True)
class StorageConfig:
    root: Path = Path("data/object-classifier")
    metadata_dir: str = "metadata"
    feature_dir: str = "features"
    patch_token_dir: str = "patch_tokens"
    faiss_index_file: str = "index.faiss"
    faiss_mapping_file: str = "index_mapping.json"

    @property
    def metadata_root(self) -> Path:
        return self.root / self.metadata_dir

    @property
    def feature_root(self) -> Path:
        return self.root / self.feature_dir

    @property
    def patch_token_root(self) -> Path:
        return self.root / self.patch_token_dir


@dataclass(frozen=True)
class FeatureCacheConfig:
    enabled: bool = True
    cache_dir: Path = Path("data/object-classifier/cache")


@dataclass(frozen=True)
class ModelConfig:
    backend: str = "pytorch"
    provider: str = "statistics"
    input_size: tuple[int, int] = (224, 224)
    roi_box: ROIBox = field(default_factory=lambda: ROIBox(0, 0, 224, 224))
    model_name: str = "facebook/dinov3-vits16-pretrain-lvd1689m"
    embedding_dim: int = 384
    device: str = "cpu"
    repo_dir: Path | None = None
    weights_dir: Path | None = None


@dataclass(frozen=True)
class PipelineConfig:
    model: ModelConfig = field(default_factory=ModelConfig)
    quality: QualityThresholds = field(default_factory=QualityThresholds)
    decision: DecisionThresholds = field(default_factory=DecisionThresholds)
    storage: StorageConfig = field(default_factory=StorageConfig)
    cache: FeatureCacheConfig = field(default_factory=FeatureCacheConfig)
    topk: int = 20


def default_config() -> PipelineConfig:
    return PipelineConfig()
