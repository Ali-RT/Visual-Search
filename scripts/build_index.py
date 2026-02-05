from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import hnswlib
import numpy as np

from app.config import load_config
from app.utils import now_unix


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()

    cfg = load_config(args.config)

    embeddings_dir = (
        cfg.artifacts.root_dir
        / "embeddings"
        / cfg.model.model_version
        / cfg.preprocess.preprocess_version
    )
    emb_path = embeddings_dir / "embeddings.f32"
    meta_path = embeddings_dir / "embedding_meta.json"
    id_map_path = embeddings_dir / "id_map.json"

    if not meta_path.exists() or not emb_path.exists() or not id_map_path.exists():
        raise FileNotFoundError("Missing embedding artifacts. Run embed_images.py first.")

    with open(meta_path, "r", encoding="utf-8") as f:
        meta = json.load(f)

    n = int(meta["count"])
    dim = int(meta["dim"])

    emb = np.memmap(str(emb_path), dtype=np.float32, mode="r", shape=(n, dim))

    index_dir = (
        cfg.artifacts.root_dir
        / "index"
        / cfg.model.model_version
        / cfg.preprocess.preprocess_version
        / cfg.index.index_version
    )
    index_dir.mkdir(parents=True, exist_ok=True)

    index_bin = index_dir / "hnsw.bin"
    index_meta = index_dir / "index_meta.json"

    index = hnswlib.Index(space=cfg.index.space, dim=dim)
    index.init_index(
        max_elements=n,
        ef_construction=int(cfg.index.ef_construction),
        M=int(cfg.index.M),
    )
    labels = np.arange(n, dtype=np.int64)
    index.add_items(emb, labels)
    index.set_ef(int(cfg.index.ef_search))
    index.save_index(str(index_bin))

    out_meta = {
        "model_version": cfg.model.model_version,
        "preprocess_version": cfg.preprocess.preprocess_version,
        "index_version": cfg.index.index_version,
        "space": cfg.index.space,
        "count": n,
        "dim": dim,
        "ef_construction": int(cfg.index.ef_construction),
        "M": int(cfg.index.M),
        "ef_search": int(cfg.index.ef_search),
        "created_at": now_unix(),
    }
    with open(index_meta, "w", encoding="utf-8") as f:
        json.dump(out_meta, f, indent=2)

    print(f"Saved HNSW index to: {index_bin}")
    print(f"Saved index meta to: {index_meta}")


if __name__ == "__main__":
    main()
