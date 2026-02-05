from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import hnswlib
import numpy as np


@dataclass
class IndexArtifactPaths:
    embeddings_dir: Path
    index_dir: Path


class IndexService:
    """
    Loads:
      - embeddings memmap (float32, shape [N, D])
      - id_map.json (row_id -> image metadata)
      - HNSW index
    Performs:
      - ANN query (Top-K')
      - exact cosine rerank (Top-K)
    """

    def __init__(
        self,
        embeddings_f32_path: Path,
        id_map_path: Path,
        index_bin_path: Path,
        index_meta_path: Path,
        ef_search: int,
    ) -> None:
        self.embeddings_f32_path = embeddings_f32_path
        self.id_map_path = id_map_path
        self.index_bin_path = index_bin_path
        self.index_meta_path = index_meta_path

        with open(self.index_meta_path, "r", encoding="utf-8") as f:
            meta = json.load(f)

        self.dim = int(meta["dim"])
        self.count = int(meta["count"])
        self.space = str(meta["space"])

        # Load id_map
        with open(self.id_map_path, "r", encoding="utf-8") as f:
            self.id_map = json.load(f)
        if len(self.id_map) != self.count:
            raise ValueError(
                f"id_map size {len(self.id_map)} != index count {self.count}. "
                "Embedding/index artifacts are inconsistent."
            )

        # Load embeddings (memmap)
        self.embeddings = np.memmap(
            str(self.embeddings_f32_path),
            dtype=np.float32,
            mode="r",
            shape=(self.count, self.dim),
        )

        # Load HNSW
        self.index = hnswlib.Index(space=self.space, dim=self.dim)
        self.index.load_index(str(self.index_bin_path), max_elements=self.count)
        self.index.set_ef(int(ef_search))

    def ann_query(self, query_vec: np.ndarray, top_k_candidates: int) -> Tuple[np.ndarray, np.ndarray]:
        # Cap k at the number of items in the index
        k = min(int(top_k_candidates), self.count)
        labels, distances = self.index.knn_query(query_vec, k=k)
        return labels[0], distances[0]

    def rerank_cosine(
        self,
        query_vec: np.ndarray,
        candidate_labels: np.ndarray,
        top_k: int,
    ) -> List[Dict[str, Any]]:
        # embeddings are L2-normalized; query_vec should also be normalized.
        cand_vecs = self.embeddings[candidate_labels]  # (K', D)
        scores = cand_vecs @ query_vec.astype(np.float32)  # cosine similarity

        order = np.argsort(-scores)[: int(top_k)]
        results = []
        for idx in order:
            row_id = int(candidate_labels[idx])
            item = self.id_map[row_id]
            results.append(
                {
                    "image_id": item["image_id"],
                    "uri": item["uri"],
                    "score": float(scores[idx]),
                    "row_id": row_id,
                }
            )
        return results
