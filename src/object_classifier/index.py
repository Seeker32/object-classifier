from __future__ import annotations

import importlib
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from .schemas import Candidate, FeatureRecord


@dataclass
class SampleIndex:
    index_path: Path | None = None
    mapping_path: Path | None = None
    vectors: np.ndarray = field(default_factory=lambda: np.empty((0, 0), dtype=np.float32))
    mapping: list[FeatureRecord] = field(default_factory=list)
    backend_name: str = field(init=False, default="numpy")
    stale: bool = False
    _faiss: Any = field(init=False, default=None, repr=False)
    _faiss_index: Any = field(init=False, default=None, repr=False)

    def __post_init__(self) -> None:
        self._faiss = self._load_faiss()
        self.backend_name = "faiss" if self._faiss is not None else "numpy"

    def rebuild(self, records: list[FeatureRecord], vectors: list[np.ndarray]) -> None:
        self.mapping = list(records)
        self.vectors = self._stack_vectors(vectors)
        self.stale = False
        if self.backend_name == "faiss" and self.vectors.size > 0:
            self._faiss_index = self._faiss.IndexFlatIP(self.vectors.shape[1])
            self._faiss_index.add(self.vectors)
        else:
            self._faiss_index = None

    def append(self, record: FeatureRecord, vector: np.ndarray) -> None:
        vector = np.asarray(vector, dtype=np.float32).reshape(1, -1)
        if self.vectors.size == 0:
            self.mapping = [record]
            self.vectors = vector
        else:
            if vector.shape[1] != self.vectors.shape[1]:
                raise ValueError("Vector dimension mismatch while appending index entry")
            self.mapping.append(record)
            self.vectors = np.vstack([self.vectors, vector])
        if self.backend_name == "faiss":
            if self._faiss_index is None:
                self._faiss_index = self._faiss.IndexFlatIP(self.vectors.shape[1])
                self._faiss_index.add(self.vectors)
            else:
                self._faiss_index.add(vector)
        self.stale = False

    def mark_stale(self) -> None:
        self.stale = True

    def search_topk(self, query_embedding: np.ndarray, k: int = 20) -> list[Candidate]:
        if not self.mapping or self.vectors.size == 0:
            return []
        query = np.asarray(query_embedding, dtype=np.float32).reshape(1, -1)
        limit = min(k, len(self.mapping))
        if self.backend_name == "faiss" and self._faiss_index is not None:
            scores, indices = self._faiss_index.search(query, limit)
            ranked_scores = scores.reshape(-1)
            ranked_indices = indices.reshape(-1)
        else:
            ranked_scores, ranked_indices = self._numpy_search(query, limit)
        return [
            Candidate(
                sample_id=self.mapping[index].sample_id,
                sku_id=self.mapping[index].sku_id,
                global_score=float(ranked_scores[position]),
            )
            for position, index in enumerate(ranked_indices)
            if index >= 0
        ]

    def persist(self) -> None:
        if self.index_path is None or self.mapping_path is None:
            return
        self.index_path.parent.mkdir(parents=True, exist_ok=True)
        self.mapping_path.parent.mkdir(parents=True, exist_ok=True)
        self.mapping_path.write_text(
            json.dumps([asdict(record) for record in self.mapping], indent=2),
            encoding="utf-8",
        )
        if self.backend_name == "faiss" and self._faiss_index is not None:
            self._faiss.write_index(self._faiss_index, str(self.index_path))
            if not self.index_path.exists():
                self.index_path.write_bytes(b"faiss-index")
            return
        with self.index_path.open("wb") as handle:
            np.save(handle, self.vectors)

    def load(self) -> None:
        if self.index_path is None or self.mapping_path is None or not self.mapping_path.exists():
            return
        self.mapping = [FeatureRecord(**row) for row in json.loads(self.mapping_path.read_text(encoding="utf-8"))]
        if self.backend_name == "faiss" and self.index_path.exists():
            self._faiss_index = self._faiss.read_index(str(self.index_path))
            self.vectors = getattr(self._faiss_index, "vectors", np.empty((0, 0), dtype=np.float32))
            return
        if not self.index_path.exists():
            self.vectors = np.empty((0, 0), dtype=np.float32)
            self._faiss_index = None
            return
        with self.index_path.open("rb") as handle:
            self.vectors = np.load(handle).astype(np.float32)
        self._faiss_index = None

    def _numpy_search(self, query: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
        scores = (self.vectors @ query.T).reshape(-1)
        top_indices = np.argsort(scores)[::-1][:k]
        return scores[top_indices], top_indices

    def _load_faiss(self) -> Any | None:
        try:
            return importlib.import_module("faiss")
        except ModuleNotFoundError:
            return None

    def _stack_vectors(self, vectors: list[np.ndarray]) -> np.ndarray:
        if not vectors:
            return np.empty((0, 0), dtype=np.float32)
        return np.vstack([np.asarray(vector, dtype=np.float32) for vector in vectors])
