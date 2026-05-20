from __future__ import annotations

import numpy as np

from object_classifier.config import StorageConfig
from object_classifier.repository import LocalRepository
from object_classifier.schemas import FeatureBundle, QualityResult


def test_repository_creates_sku_and_persists_sample_and_features(tmp_path) -> None:
    repository = LocalRepository(StorageConfig(root=tmp_path))

    sku = repository.create_sku("Widget")
    sample = repository.add_sample(
        sku_id=sku.sku_id,
        image_path="/tmp/widget.png",
        roi_points=((1, 2), (1, 40), (30, 40), (30, 2)),
        quality=QualityResult(
            status="pass",
            score=0.9,
            blur_score=12.0,
            dark_ratio=0.1,
            bright_ratio=0.1,
            reasons=[],
        ),
    )
    record = repository.save_feature_bundle(
        sample=sample,
        bundle=FeatureBundle(
            global_embedding=np.array([1.0, 0.0, 0.0], dtype=np.float32),
            patch_tokens=np.array([[1.0, 0.0, 0.0]], dtype=np.float32),
            backend="pytorch",
        ),
        feature_version="v1",
    )

    assert sku.sku_id.startswith("sku-")
    assert repository.get_sku(sku.sku_id) == sku
    assert repository.get_sample(sample.sample_id) == sample
    assert sample.roi_points == ((1, 2), (1, 40), (30, 40), (30, 2))
    assert repository.get_feature_record(sample.sample_id) == record

    restored = repository.load_feature_bundle(record)
    assert restored.backend == "pytorch"
    assert restored.global_embedding.tolist() == [1.0, 0.0, 0.0]
