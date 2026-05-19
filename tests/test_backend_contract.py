from __future__ import annotations

import sys
import types

import numpy as np

from object_classifier.config import ModelConfig, ROIBox
from object_classifier.features import PyTorchFeatureBackend, RKNNFeatureBackend
from object_classifier.schemas import FeatureBundle, NormalizedROI


class DummySession:
    def infer(self, batch: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        _ = batch
        return (
            np.array([3.0, 4.0, 0.0], dtype=np.float32),
            np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32),
        )


def test_backends_share_output_contract() -> None:
    roi = NormalizedROI(
        image=np.ones((16, 16, 3), dtype=np.uint8) * 127,
        source_path=None,
        roi_box=(0, 0, 16, 16),
        original_size=(16, 16),
    )
    config = ModelConfig(input_size=(16, 16), roi_box=ROIBox(0, 0, 16, 16), embedding_dim=3)

    pytorch = PyTorchFeatureBackend(config=config, session=DummySession())
    rknn = RKNNFeatureBackend(config=config, session=DummySession())

    pytorch_bundle = pytorch.extract(roi)
    rknn_bundle = rknn.extract(roi)

    assert isinstance(pytorch_bundle, FeatureBundle)
    assert isinstance(rknn_bundle, FeatureBundle)
    assert pytorch_bundle.global_embedding.shape == rknn_bundle.global_embedding.shape == (3,)
    assert pytorch_bundle.patch_tokens.shape == rknn_bundle.patch_tokens.shape == (2, 2)
    assert np.isclose(np.linalg.norm(pytorch_bundle.global_embedding), 1.0)
    assert np.isclose(np.linalg.norm(rknn_bundle.global_embedding), 1.0)


def test_torchhub_provider_uses_local_repo_and_weights(monkeypatch, tmp_path) -> None:
    repo_dir = tmp_path / "dinov3"
    repo_dir.mkdir()
    weights_dir = tmp_path / "models"
    weights_dir.mkdir()
    weight_path = weights_dir / "dinov3_vits16_pretrain_lvd1689m-08c60483.pth"
    weight_path.write_bytes(b"weights")
    loaded: dict[str, object] = {}

    class FakeModule:
        def eval(self):
            return self

    def fake_load(repo_or_dir, model, source, weights):
        loaded["repo_or_dir"] = repo_or_dir
        loaded["model"] = model
        loaded["source"] = source
        loaded["weights"] = weights
        return FakeModule()

    fake_torch = types.SimpleNamespace(hub=types.SimpleNamespace(load=fake_load))
    monkeypatch.setitem(sys.modules, "torch", fake_torch)

    from object_classifier.features import create_backend

    backend = create_backend(
        "pytorch",
        ModelConfig(
            provider="torchhub",
            model_name="dinov3_vits16",
            input_size=(16, 16),
            roi_box=ROIBox(0, 0, 16, 16),
            embedding_dim=3,
            repo_dir=repo_dir,
            weights_dir=weights_dir,
        ),
    )

    assert isinstance(backend, PyTorchFeatureBackend)
    assert loaded == {
        "repo_or_dir": str(repo_dir),
        "model": "dinov3_vits16",
        "source": "local",
        "weights": str(weight_path),
    }
