from __future__ import annotations

import argparse
import json
import sqlite3
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
from tqdm import tqdm

from app.config import load_config
from app.embedding_service import EmbeddingService
from app.utils import safe_open_image, now_unix


def fetch_images(db_path: Path) -> List[Dict[str, Any]]:
    conn = sqlite3.connect(str(db_path))
    cur = conn.cursor()
    cur.execute("SELECT image_id, uri, width, height FROM images ORDER BY image_id;")
    rows = cur.fetchall()
    conn.close()

    items = []
    for r in rows:
        items.append({"image_id": r[0], "uri": r[1], "width": int(r[2]), "height": int(r[3])})
    return items


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--batch_size", type=int, default=64)
    args = parser.parse_args()

    cfg = load_config(args.config)
    items = fetch_images(cfg.data.sqlite_path)
    if not items:
        raise RuntimeError("No images found in SQLite. Run ingest_images.py first.")

    # Resolve artifact dirs
    embeddings_dir = (
        cfg.artifacts.root_dir
        / "embeddings"
        / cfg.model.model_version
        / cfg.preprocess.preprocess_version
    )
    embeddings_dir.mkdir(parents=True, exist_ok=True)

    id_map_path = embeddings_dir / "id_map.json"
    emb_path = embeddings_dir / "embeddings.f32"
    meta_path = embeddings_dir / "embedding_meta.json"

    embedder = EmbeddingService(
        backbone=cfg.model.backbone,
        pretrained=cfg.model.pretrained,
        device=cfg.model.device,
    )

    # Determine embedding dim with one sample
    sample_img = safe_open_image(Path(items[0]["uri"]))
    sample_out = embedder.embed_pil(sample_img)
    dim = sample_out.dim
    n = len(items)

    # Create memmap for (N,D)
    emb = np.memmap(str(emb_path), dtype=np.float32, mode="w+", shape=(n, dim))

    id_map = []
    for i, item in enumerate(items):
        id_map.append(
            {
                "row_id": i,
                "image_id": item["image_id"],
                "uri": item["uri"],
                "width": item["width"],
                "height": item["height"],
            }
        )

    # Embed sequentially (simple and reliable for POC)
    for i in tqdm(range(n), desc="Embedding images"):
        p = Path(items[i]["uri"])
        try:
            img = safe_open_image(p)
            out = embedder.embed_pil(img)
            vec = out.embedding
            # Safety: normalize again
            vec = vec / (np.linalg.norm(vec) + 1e-12)
            emb[i, :] = vec.astype(np.float32)
        except Exception:
            # If an image fails, set a zero vector (and it will be poor retrieval).
            emb[i, :] = np.zeros((dim,), dtype=np.float32)

    emb.flush()

    with open(id_map_path, "w", encoding="utf-8") as f:
        json.dump(id_map, f)

    meta = {
        "model_version": cfg.model.model_version,
        "preprocess_version": cfg.preprocess.preprocess_version,
        "count": n,
        "dim": dim,
        "created_at": now_unix(),
        "dtype": "float32",
        "format": "memmap_f32",
    }
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    print(f"Saved embeddings to: {emb_path}")
    print(f"Saved id_map to: {id_map_path}")
    print(f"Saved meta to: {meta_path}")


if __name__ == "__main__":
    main()
