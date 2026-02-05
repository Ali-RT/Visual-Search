#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import math
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple
import inspect
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import numpy as np

from app.config import load_config
from app.embedding_service import EmbeddingService
from app.index_service import IndexService
from app.utils import safe_open_image, sha256_file


def dcg_at_k(rels: List[int], k: int) -> float:
    s = 0.0
    for i, rel in enumerate(rels[:k], start=1):
        s += (2**rel - 1) / math.log2(i + 1)
    return s


def ndcg_at_k(retrieved_rels: List[int], ideal_rels: List[int], k: int) -> float:
    dcg = dcg_at_k(retrieved_rels, k)
    idcg = dcg_at_k(sorted(ideal_rels, reverse=True), k)
    return 0.0 if idcg <= 0 else (dcg / idcg)


def average_precision_at_k(relevant_set: set[str], retrieved: List[str], k: int) -> float:
    hits = 0
    ap = 0.0
    for i, x in enumerate(retrieved[:k], start=1):
        if x in relevant_set:
            hits += 1
            ap += hits / i
    return ap / max(1, len(relevant_set))


def recall_at_k(relevant_set: set[str], retrieved: List[str], k: int) -> float:
    hits = sum(1 for x in retrieved[:k] if x in relevant_set)
    return hits / max(1, len(relevant_set))


def load_manifest(manifest_csv: Path) -> List[Tuple[str, int]]:
    rows: List[Tuple[str, int]] = []
    with manifest_csv.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        if not reader.fieldnames or "rel_path" not in reader.fieldnames or "label" not in reader.fieldnames:
            raise ValueError("manifest must have columns: rel_path,label")
        for r in reader:
            rows.append((r["rel_path"], int(r["label"])))
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, help="Gallery config used to build index (e.g., configs/sop_gallery.yaml)")
    ap.add_argument("--query_root", type=Path, default=Path("data/sop_poc/query"))
    ap.add_argument("--query_manifest", type=Path, default=Path("data/sop_poc/manifests/query.csv"))
    ap.add_argument("--gallery_manifest", type=Path, default=Path("data/sop_poc/manifests/gallery.csv"))
    ap.add_argument("--k", type=int, default=50)
    ap.add_argument("--k_candidates", type=int, default=200)
    ap.add_argument("--max_queries", type=int, default=500)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    cfg = load_config(args.config)

    # Build gallery index artifacts (built from cfg)
    embeddings_dir = (
        cfg.artifacts.root_dir / "embeddings" / cfg.model.model_version / cfg.preprocess.preprocess_version
    )
    index_dir = (
        cfg.artifacts.root_dir
        / "index"
        / cfg.model.model_version
        / cfg.preprocess.preprocess_version
        / cfg.index.index_version
    )

    # --- robust IndexService init (works whether or not dedup exists) ---
    sig = inspect.signature(IndexService.__init__)
    kwargs = {
        "embeddings_f32_path": str(embeddings_dir / "embeddings.f32"),
        "id_map_path": str(embeddings_dir / "id_map.json"),
        "index_bin_path": str(index_dir / "hnsw.bin"),
        "index_meta_path": str(index_dir / "index_meta.json"),
    }

    # ef_search arg name may vary by your IndexService version
    if "ef_search" in sig.parameters:
        kwargs["ef_search"] = cfg.index.ef_search
    elif "ef" in sig.parameters:
        kwargs["ef"] = cfg.index.ef_search

    # Optional dedup args only if your IndexService supports them
    if "dedup_enabled" in sig.parameters:
        kwargs["dedup_enabled"] = True
    if "dedup_similarity_threshold" in sig.parameters:
        kwargs["dedup_similarity_threshold"] = 0.995

    index_service = IndexService(**kwargs)


    embedder = EmbeddingService(
        backbone=cfg.model.backbone,
        pretrained=cfg.model.pretrained,
        device=cfg.model.device,
    )

    # Build gallery relevance map: label -> set(image_id)
    # We map rel_path -> image_id using gallery index id_map (uri stored there).
    uri_to_image_id: Dict[str, str] = {item["uri"]: item["image_id"] for item in index_service.id_map}

    gallery_rows = load_manifest(args.gallery_manifest)
    label_to_gallery_ids: Dict[int, set[str]] = defaultdict(set)
    for rel_path, label in gallery_rows:
        uri = str(Path(cfg.data.image_dir) / Path(rel_path))
        iid = uri_to_image_id.get(uri)
        if iid:
            label_to_gallery_ids[label].add(iid)

    query_rows = load_manifest(args.query_manifest)

    rng = np.random.default_rng(args.seed)
    rng.shuffle(query_rows)
    query_rows = query_rows[: min(len(query_rows), int(args.max_queries))]

    recalls, maps, ndcgs = [], [], []
    used = 0

    for rel_path, label in query_rows:
        q_path = args.query_root / Path(rel_path)
        if not q_path.exists():
            continue

        # Query image_id is sha256(file) in your ingestion; compute to exclude self if present in gallery.
        q_image_id = sha256_file(q_path)

        relevant = set(label_to_gallery_ids.get(label, set()))
        if q_image_id in relevant:
            relevant.remove(q_image_id)
        if not relevant:
            continue

        img = safe_open_image(q_path)
        q_vec = embedder.embed_pil(img).embedding
        q_vec = q_vec / (np.linalg.norm(q_vec) + 1e-12)

        cand_labels, _ = index_service.ann_query(q_vec, top_k_candidates=int(args.k_candidates))
        reranked = index_service.rerank_cosine(q_vec, cand_labels, top_k=int(args.k))

        retrieved_ids = [r["image_id"] for r in reranked if r["image_id"] != q_image_id]

        recalls.append(recall_at_k(relevant, retrieved_ids, k=int(args.k)))
        maps.append(average_precision_at_k(relevant, retrieved_ids, k=int(args.k)))

        retrieved_rels = [1 if rid in relevant else 0 for rid in retrieved_ids[: int(args.k)]]
        ideal_rels = [1] * len(relevant)
        ndcgs.append(ndcg_at_k(retrieved_rels, ideal_rels, k=int(args.k)))

        used += 1

    if used == 0:
        raise RuntimeError("No valid queries evaluated. Check query_root layout and manifests.")

    print(f"Queries evaluated: {used}")
    print(f"Recall@{args.k}: {float(np.mean(recalls)):.4f}")
    print(f"mAP@{args.k}:   {float(np.mean(maps)):.4f}")
    print(f"nDCG@{args.k}:  {float(np.mean(ndcgs)):.4f}")


if __name__ == "__main__":
    main()
