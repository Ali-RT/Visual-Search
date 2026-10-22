from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    FieldCondition,
    Filter,
    MatchValue,
    PointStruct,
    VectorParams,
)


class QdrantIndexService:
    """
    Drop-in replacement for IndexService backed by Qdrant.

    Stores vectors + metadata (use_case, label, uri, width, height) as
    payload.  Supports filtered search by any payload field.
    """

    def __init__(
        self,
        host: str = "localhost",
        port: int = 6333,
        collection_name: str = "images",
        vector_dim: int = 512,
        on_disk: bool = False,
    ) -> None:
        self.client = QdrantClient(host=host, port=port)
        self.collection_name = collection_name
        self.vector_dim = vector_dim

        if not self.client.collection_exists(collection_name):
            self.client.create_collection(
                collection_name=collection_name,
                vectors_config=VectorParams(
                    size=vector_dim,
                    distance=Distance.COSINE,
                    on_disk=on_disk,
                ),
            )

    @property
    def count(self) -> int:
        info = self.client.get_collection(self.collection_name)
        return info.points_count

    def upsert_batch(
        self,
        ids: List[int],
        vectors: List[np.ndarray],
        payloads: List[Dict[str, Any]],
        batch_size: int = 100,
    ) -> int:
        """Upsert vectors with payload in batches. Returns count upserted."""
        total = 0
        for start in range(0, len(ids), batch_size):
            end = min(start + batch_size, len(ids))
            points = [
                PointStruct(
                    id=point_id,
                    vector=vec.tolist() if isinstance(vec, np.ndarray) else vec,
                    payload=pay,
                )
                for point_id, vec, pay in zip(
                    ids[start:end],
                    vectors[start:end],
                    payloads[start:end],
                )
            ]
            self.client.upsert(collection_name=self.collection_name, points=points)
            total += len(points)
        return total

    def upsert_one(
        self,
        point_id: int,
        vector: np.ndarray,
        payload: Dict[str, Any],
    ) -> None:
        self.client.upsert(
            collection_name=self.collection_name,
            points=[
                PointStruct(
                    id=point_id,
                    vector=vector.tolist() if isinstance(vector, np.ndarray) else vector,
                    payload=payload,
                )
            ],
        )

    def search(
        self,
        query_vec: np.ndarray,
        top_k: int = 50,
        use_case_filter: Optional[str] = None,
        label_filter: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """
        Vector search with optional metadata filtering.
        Returns list of dicts with image_id, uri, score, use_case, label, etc.
        """
        conditions = []
        if use_case_filter:
            conditions.append(
                FieldCondition(key="use_case", match=MatchValue(value=use_case_filter))
            )
        if label_filter:
            conditions.append(
                FieldCondition(key="label", match=MatchValue(value=label_filter))
            )

        query_filter = Filter(must=conditions) if conditions else None

        hits = self.client.query_points(
            collection_name=self.collection_name,
            query=query_vec.tolist() if isinstance(query_vec, np.ndarray) else query_vec,
            query_filter=query_filter,
            limit=top_k,
            with_payload=True,
        ).points

        results = []
        for i, hit in enumerate(hits):
            payload = hit.payload or {}
            results.append(
                {
                    "image_id": payload.get("image_id", str(hit.id)),
                    "uri": payload.get("uri", ""),
                    "score": float(hit.score),
                    "position": i + 1,
                    "use_case": payload.get("use_case", ""),
                    "label": payload.get("label", ""),
                    "width": payload.get("width"),
                    "height": payload.get("height"),
                    "gt_tags": payload.get("gt_tags", []),
                    "source": payload.get("source", ""),
                }
            )
        return results

    def search_by_id(
        self,
        image_id: str,
        top_k: int = 50,
        use_case_filter: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """Find similar images using a stored point's vector. Excludes the query point itself."""
        scroll_result, _ = self.client.scroll(
            collection_name=self.collection_name,
            scroll_filter=Filter(
                must=[FieldCondition(key="image_id", match=MatchValue(value=image_id))]
            ),
            limit=1,
            with_vectors=True,
            with_payload=True,
        )
        if not scroll_result:
            return []

        point = scroll_result[0]
        query_vec = point.vector
        results = self.search(
            query_vec=np.array(query_vec, dtype=np.float32),
            top_k=top_k + 1,
            use_case_filter=use_case_filter,
        )
        return [r for r in results if r["image_id"] != image_id][:top_k]

    def delete_by_use_case(self, use_case: str) -> None:
        self.client.delete(
            collection_name=self.collection_name,
            points_selector=Filter(
                must=[FieldCondition(key="use_case", match=MatchValue(value=use_case))]
            ),
        )

    def get_collection_info(self) -> Dict[str, Any]:
        info = self.client.get_collection(self.collection_name)
        return {
            "collection_name": self.collection_name,
            "points_count": info.points_count,
            "status": str(info.status),
        }
