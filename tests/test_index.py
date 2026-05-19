from __future__ import annotations

import json
import sys
import types

import numpy as np

from object_classifier.index import SampleIndex
from object_classifier.schemas import FeatureRecord


class FakeIndexFlatIP:
    def __init__(self, dimension: int) -> None:
        self.dimension = dimension
        self.vectors = np.empty((0, dimension), dtype=np.float32)

    def add(self, vectors: np.ndarray) -> None:
        self.vectors = np.vstack([self.vectors, vectors.astype(np.float32)])

    def search(self, query: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
        scores = query @ self.vectors.T
        order = np.argsort(scores, axis=1)[:, ::-1][:, :k]
        ranked_scores = np.take_along_axis(scores, order, axis=1)
        return ranked_scores.astype(np.float32), order.astype(np.int64)


def test_sample_index_prefers_faiss_when_module_available(monkeypatch, tmp_path) -> None:
    written: dict[str, object] = {}

    def write_index(index, path: str) -> None:
        written[path] = index

    def read_index(path: str):
        return written[path]

    fake_faiss = types.SimpleNamespace(
        IndexFlatIP=FakeIndexFlatIP,
        write_index=write_index,
        read_index=read_index,
    )
    monkeypatch.setitem(sys.modules, "faiss", fake_faiss)

    index = SampleIndex(index_path=tmp_path / "index.faiss", mapping_path=tmp_path / "mapping.json")
    records = [
        FeatureRecord("sample-1", "sku-a", "p0", "g1.npy", "p1.npy", "pytorch"),
        FeatureRecord("sample-2", "sku-b", "p0", "g2.npy", "p2.npy", "pytorch"),
    ]
    vectors = [
        np.array([1.0, 0.0], dtype=np.float32),
        np.array([0.0, 1.0], dtype=np.float32),
    ]

    index.rebuild(records, vectors)
    index.persist()

    assert index.backend_name == "faiss"
    assert index.index_path.exists()
    assert json.loads(index.mapping_path.read_text(encoding="utf-8"))[0]["sample_id"] == "sample-1"

    restored = SampleIndex(index_path=tmp_path / "index.faiss", mapping_path=tmp_path / "mapping.json")
    restored.load()
    candidates = restored.search_topk(np.array([0.9, 0.1], dtype=np.float32), k=2)

    assert [candidate.sample_id for candidate in candidates] == ["sample-1", "sample-2"]
