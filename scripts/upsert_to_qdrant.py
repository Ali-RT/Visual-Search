"""
Ingest images, compute embeddings, and upsert to Qdrant.

Reads images from SQLite metadata DB, optionally enriches with
ground truth tags from label JSON files (via parse_labels), and
upserts vectors + payload to Qdrant.

Usage:
    python scripts/upsert_to_qdrant.py --config configs/adas_usecase.yaml
    python scripts/upsert_to_qdrant.py --config configs/adas_usecase.yaml --labels data/new_images
    python scripts/upsert_to_qdrant.py --config configs/adas_usecase.yaml --use_cases crosswalk stopline
"""
from __future__ import annotations

import argparse
import sqlite3
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
from tqdm import tqdm

from app.config import load_config
from app.embedding_service import EmbeddingService
from app.qdrant_index_service import QdrantIndexService
from app.utils import safe_open_image


def fetch_images(db_path: Path, use_cases: Optional[List[str]] = None) -> List[Dict[str, Any]]:
    conn = sqlite3.connect(str(db_path))
    cur = conn.cursor()

    cur.execute("PRAGMA table_info(images);")
    cols = {row[1] for row in cur.fetchall()}
    has_use_case = "use_case" in cols and "label" in cols

    if has_use_case:
        query = "SELECT image_id, uri, width, height, use_case, label FROM images"
        params = []
        if use_cases:
            placeholders = ",".join("?" for _ in use_cases)
            query += f" WHERE use_case IN ({placeholders})"
            params = list(use_cases)
        query += " ORDER BY image_id;"
        cur.execute(query, params)
    else:
        cur.execute("SELECT image_id, uri, width, height FROM images ORDER BY image_id;")

    rows = cur.fetchall()
    conn.close()

    items = []
    for r in rows:
        item = {"image_id": r[0], "uri": r[1], "width": int(r[2]), "height": int(r[3])}
        if has_use_case:
            item["use_case"] = r[4] or ""
            item["label"] = r[5] or ""
        items.append(item)
    return items


def main() -> None:
    parser = argparse.ArgumentParser(description="Embed images and upsert to Qdrant")
    parser.add_argument("--config", required=True)
    parser.add_argument("--use_cases", nargs="*", default=None, help="Only process specific use cases")
    parser.add_argument("--labels", default=None, help="Path to label directory (e.g. new_images) for ground truth enrichment")
    parser.add_argument("--batch_size", type=int, default=100, help="Qdrant upsert batch size")
    args = parser.parse_args()

    cfg = load_config(args.config)

    qdrant_cfg = cfg.qdrant
    if qdrant_cfg is None:
        raise RuntimeError("No 'qdrant' section in config. Add it to use Qdrant.")

    items = fetch_images(cfg.data.sqlite_path, args.use_cases)
    if not items:
        raise RuntimeError("No images found in SQLite. Run ingest_usecase_images.py first.")

    print(f"Found {len(items)} images to embed and upsert")

    # Load ground truth lookup if label dir provided
    gt_lookup = {}
    if args.labels:
        from scripts.parse_labels import build_gt_lookup
        gt_lookup = build_gt_lookup(args.labels)
        print(f"Loaded {len(gt_lookup)} ground truth labels from {args.labels}")

    embedder = EmbeddingService(
        backbone=cfg.model.backbone,
        pretrained=cfg.model.pretrained,
        device=cfg.model.device,
    )

    sample_img = safe_open_image(Path(items[0]["uri"]))
    sample_out = embedder.embed_pil(sample_img)
    dim = sample_out.dim

    qdrant = QdrantIndexService(
        host=qdrant_cfg.host,
        port=qdrant_cfg.port,
        collection_name=qdrant_cfg.collection_name,
        vector_dim=dim,
    )

    print(f"Qdrant collection '{qdrant_cfg.collection_name}': {qdrant.count} existing points")
    print(f"Embedding dim: {dim}")

    batch_ids = []
    batch_vectors = []
    batch_payloads = []
    upserted = 0
    failed = 0
    gt_matched = 0

    for i, item in enumerate(tqdm(items, desc="Embedding + upserting")):
        try:
            img = safe_open_image(Path(item["uri"]))
            out = embedder.embed_pil(img)
            vec = out.embedding
            vec = vec / (np.linalg.norm(vec) + 1e-12)

            payload = {
                "image_id": item["image_id"],
                "uri": item["uri"],
                "width": item["width"],
                "height": item["height"],
                "use_case": item.get("use_case", ""),
                "label": item.get("label", ""),
            }

            # Enrich with ground truth from labels
            img_stem = Path(item["uri"]).stem
            gt = gt_lookup.get(img_stem)
            if gt:
                gt_matched += 1
                payload["gt_tags"] = gt.gt_tags
                payload["object_counts"] = gt.object_counts
                payload["horizontal_types"] = gt.horizontal_types
                payload["has_crosswalk"] = gt.has_crosswalk
                payload["has_stopline"] = gt.has_stopline
                payload["has_lane_marking"] = gt.has_lane_marking
                payload["has_cones"] = gt.has_cones
                payload["cone_count"] = gt.cone_count
                payload["source"] = "labeled"
            else:
                payload["gt_tags"] = []
                payload["source"] = "unlabeled"

            batch_ids.append(i)
            batch_vectors.append(vec)
            batch_payloads.append(payload)

            if len(batch_ids) >= args.batch_size:
                qdrant.upsert_batch(batch_ids, batch_vectors, batch_payloads)
                upserted += len(batch_ids)
                batch_ids, batch_vectors, batch_payloads = [], [], []

        except Exception as e:
            failed += 1
            if failed <= 5:
                print(f"  FAIL: {item['uri']} -> {e}")

    # Flush remaining
    if batch_ids:
        qdrant.upsert_batch(batch_ids, batch_vectors, batch_payloads)
        upserted += len(batch_ids)

    info = qdrant.get_collection_info()
    print(f"\nDone.")
    print(f"  Upserted:     {upserted}")
    print(f"  GT matched:   {gt_matched}")
    print(f"  Unlabeled:    {upserted - gt_matched}")
    print(f"  Failed:       {failed}")
    print(f"  Total points: {info['points_count']}")
    print(f"  Status:       {info['status']}")


if __name__ == "__main__":
    main()
