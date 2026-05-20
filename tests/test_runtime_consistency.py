from __future__ import annotations

from pathlib import Path

import numpy as np

from object_classifier.config import ModelConfig, ROIPolygon
from object_classifier.features import RKNNRuntimeSession, create_backend
from object_classifier.schemas import NormalizedROI


class FakeRKNNLite:
    def __init__(self) -> None:
        self.loaded: str | None = None

    def load_rknn(self, path: str) -> int:
        self.loaded = path
        return 0

    def init_runtime(self) -> int:
        return 0

    def inference(self, inputs: list[np.ndarray]) -> list[np.ndarray]:
        batch = inputs[0]
        embedding = batch.mean(axis=(2, 3))
        patch_tokens = batch.reshape(batch.shape[0], batch.shape[1], -1).transpose(0, 2, 1)
        return [embedding.astype(np.float32), patch_tokens.astype(np.float32)]

    def release(self) -> None:
        return None


def test_rknn_runtime_session_satisfies_backend_contract(tmp_path) -> None:
    artifact = tmp_path / "embedding.rknn"
    artifact.write_bytes(b"fake")
    session = RKNNRuntimeSession(
        embedding_model_path=artifact,
        patch_model_path=artifact,
        runtime_factory=FakeRKNNLite,
    )
    roi = NormalizedROI(
        image=np.ones((16, 16, 3), dtype=np.uint8) * 127,
        source_path=None,
        roi_points=((0, 0), (0, 15), (15, 15), (15, 0)),
        relative_points=((0, 0), (0, 15), (15, 15), (15, 0)),
        crop_box=(0, 0, 16, 16),
        original_size=(16, 16),
    )

    backend = create_backend(
        "rknn",
        ModelConfig(
            backend="rknn",
            input_size=(16, 16),
            roi_box=ROIPolygon(((0, 0), (0, 15), (15, 15), (15, 0))),
            embedding_dim=3,
            rknn_embedding_path=artifact,
            rknn_patch_tokens_path=artifact,
        ),
        session=session,
    )

    bundle = backend.extract(roi)

    assert bundle.global_embedding.shape == (3,)
    assert bundle.patch_tokens.ndim == 2
    assert bundle.backend == "rknn"


def test_rknn_runtime_session_can_fallback_to_embedding_only(tmp_path) -> None:
    artifact = tmp_path / "embedding.rknn"
    artifact.write_bytes(b"fake")
    session = RKNNRuntimeSession(
        embedding_model_path=artifact,
        patch_model_path=None,
        runtime_factory=FakeRKNNLite,
    )

    embedding, patch_tokens = session.infer(np.ones((1, 3, 16, 16), dtype=np.float32))

    assert embedding.shape == (3,)
    assert patch_tokens.shape[1] == 3
