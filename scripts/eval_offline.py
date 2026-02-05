from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

from app.config import load_config
from app.index_service import IndexService


def load_labels_csv(path: Path) -> Dict[str, str]:
    out: Dict[str, str] = {}
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        if not reader.fieldnames or "image_id" not in reader.fieldnames or "label" not in reader.fieldnames:
            raise ValueError("labels_csv must contain columns: image_id,label")
        for row in reader:
            out[row["image_id"]] = row["label"]
    return out


def load_judgments_csv(path: Path) -> Dict[str, Dict[str, int]]:
    """
    Returns judgments[query_id][candidate_id] = relevance (0..5).
    """
    out: Dict[str, Dict[str, int]] = {}
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        needed = {"query_image_id", "candidate_image_id", "relevance"}
        if not reader.fieldnames or not needed.issubset(set(reader.fieldnames)):
            raise ValueError("judgments_csv must contain columns: query_image_id,candidate_image_id,relevance")
        for row in reader:
            q = row["query_image_id"]
            c = row["candidate_image_id"]
            rel = int(row["relevance"])
            out.setdefault(q, {})[c] = rel
    return out


def recall_at_k(relevant_set: Set[str], retrieved: List[str], k: int) -> float:
    topk = retrieved[:k]
    hits = sum(1 for x in topk if x in relevant_set)
    return hits / max(1, len(relevant_set))


def average_precision_at_k(relevant_set: Set[str], retrieved: List[str], k: int) -> float:
    hits = 0
    ap = 0.0
    for i, x in enumerate(retrieved[:k], start=1):
        if x in relevant_set:
            hits += 1
            ap += hits / i
    return ap / max(1, len(relevant_set))


def dcg_at_k(rels: List[int], k: int) -> float:
    """
    DCG with exponential gain: (2^rel - 1)/log2(i+1)
    """
    s = 0.0
    for i, rel in enumerate(rels[:k], start=1):
        gain = (2**rel - 1)
        denom = math.log2(i + 1)
        s += gain / denom
    return s


def ndcg_at_k(retrieved_rels: List[int], ideal_rels: List[int], k: int) -> float:
    dcg = dcg_at_k(retrieved_rels, k)
    idcg = dcg_at_k(sorted(ideal_rels, reverse=True), k)
    if idcg <= 0:
        return 0.0
    return dcg / idcg


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--k", type=int, default=50)
    parser.add_argument("--k_candidates", type=int, default=200)
    parser.add_argument("--num_queries", type=int, default=200)

    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--labels_csv", help="CSV with columns image_id,label (binary relevance)")
    group.add_argument("--judgments_csv", help="CSV with columns query_image_id,candidate_image_id,relevance (0..5)")

    args = parser.parse_args()
    cfg = load_config(args.config)

    embeddings_dir = (
        cfg.artifacts.root_dir
        / "embeddings"
        / cfg.model.model_version
        / cfg.preprocess.preprocess_version
    )
    index_dir = (
        cfg.artifacts.root_dir
        / "index"
        / cfg.model.model_version
        / cfg.preprocess.preprocess_version
        / cfg.index.index_version
    )

    index_service = IndexService(
        embeddings_f32_path=embeddings_dir / "embeddings.f32",
        id_map_path=embeddings_dir / "id_map.json",
        index_bin_path=index_dir / "hnsw.bin",
        index_meta_path=index_dir / "index_meta.json",
        ef_search=cfg.index.ef_search,
    )

    image_id_to_row = {item["image_id"]: item["row_id"] for item in index_service.id_map}

    rng = np.random.default_rng(0)

    recalls: List[float] = []
    maps: List[float] = []
    ndcgs: List[float] = []

    if args.labels_csv:
        labels = load_labels_csv(Path(args.labels_csv))
        valid_ids = [iid for iid in labels.keys() if iid in image_id_to_row]
        rng.shuffle(valid_ids)
        queries = valid_ids[: min(len(valid_ids), int(args.num_queries))]

        for qid in queries:
            q_label = labels[qid]
            relevant = {iid for iid, lab in labels.items() if lab == q_label and iid != qid and iid in image_id_to_row}
            if not relevant:
                continue

            q_row = image_id_to_row[qid]
            q_vec = np.array(index_service.embeddings[q_row], dtype=np.float32)

            cand_labels, _ = index_service.ann_query(q_vec, top_k_candidates=int(args.k_candidates))
            reranked = index_service.rerank_cosine(q_vec, cand_labels, top_k=int(args.k))

            retrieved_ids = [r["image_id"] for r in reranked if r["image_id"] != qid]

            recalls.append(recall_at_k(relevant, retrieved_ids, k=int(args.k)))
            maps.append(average_precision_at_k(relevant, retrieved_ids, k=int(args.k)))

            # Binary nDCG: rel=1 for relevant else 0; ideal has |relevant| ones
            retrieved_rels = [1 if rid in relevant else 0 for rid in retrieved_ids[: int(args.k)]]
            ideal_rels = [1] * len(relevant)
            ndcgs.append(ndcg_at_k(retrieved_rels, ideal_rels, k=int(args.k)))

    else:
        judgments = load_judgments_csv(Path(args.judgments_csv))
        valid_queries = [qid for qid in judgments.keys() if qid in image_id_to_row]
        rng.shuffle(valid_queries)
        queries = valid_queries[: min(len(valid_queries), int(args.num_queries))]

        for qid in queries:
            q_row = image_id_to_row[qid]
            q_vec = np.array(index_service.embeddings[q_row], dtype=np.float32)

            # Relevant set for Recall/mAP: rel > 0
            rel_map = judgments[qid]
            relevant = {cid for cid, rel in rel_map.items() if rel > 0 and cid in image_id_to_row and cid != qid}
            if not relevant:
                continue

            cand_labels, _ = index_service.ann_query(q_vec, top_k_candidates=int(args.k_candidates))
            reranked = index_service.rerank_cosine(q_vec, cand_labels, top_k=int(args.k))

            retrieved_ids = [r["image_id"] for r in reranked if r["image_id"] != qid]

            recalls.append(recall_at_k(relevant, retrieved_ids, k=int(args.k)))
            maps.append(average_precision_at_k(relevant, retrieved_ids, k=int(args.k)))

            # Graded nDCG:
            retrieved_rels = [int(rel_map.get(rid, 0)) for rid in retrieved_ids[: int(args.k)]]
            ideal_rels = [int(v) for v in rel_map.values() if v > 0]
            ndcgs.append(ndcg_at_k(retrieved_rels, ideal_rels, k=int(args.k)))

    if not recalls:
        raise RuntimeError("No valid queries evaluated (check labels/judgments coverage).")

    print(f"Queries evaluated: {len(recalls)}")
    print(f"Recall@{args.k}: {float(np.mean(recalls)):.4f}")
    print(f"mAP@{args.k}:   {float(np.mean(maps)):.4f}")
    print(f"nDCG@{args.k}:  {float(np.mean(ndcgs)):.4f}")


if __name__ == "__main__":
    main()
