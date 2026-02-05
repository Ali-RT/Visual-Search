from __future__ import annotations

import json
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api import router
from app.config import load_config
from app.embedding_service import EmbeddingService
from app.index_service import IndexService
from app.interaction_store import InteractionStore


CONFIG_PATH = "configs/sop_gallery.yaml"

app = FastAPI(title="Visual Search POC")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

services = {}


def _resolve_artifact_paths(cfg) -> dict:
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

    return {
        "embeddings_dir": embeddings_dir,
        "index_dir": index_dir,
        "embeddings_f32": embeddings_dir / "embeddings.f32",
        "id_map": embeddings_dir / "id_map.json",
        "embedding_meta": embeddings_dir / "embedding_meta.json",
        "index_bin": index_dir / "hnsw.bin",
        "index_meta": index_dir / "index_meta.json",
    }


@app.on_event("startup")
def startup_event() -> None:
    cfg = load_config(CONFIG_PATH)
    paths = _resolve_artifact_paths(cfg)

    # Load embedder
    embedder = EmbeddingService(
        backbone=cfg.model.backbone,
        pretrained=cfg.model.pretrained,
        device=cfg.model.device,
    )

    # Load index service
    index_service = IndexService(
        embeddings_f32_path=paths["embeddings_f32"],
        id_map_path=paths["id_map"],
        index_bin_path=paths["index_bin"],
        index_meta_path=paths["index_meta"],
        ef_search=cfg.index.ef_search,
    )

    # Load interaction store for CTR logging
    interaction_store = InteractionStore(cfg.data.sqlite_path)

    services.clear()
    services.update(
        {
            "embedder": embedder,
            "index": index_service,
            "interaction_store": interaction_store,
            "model_version": cfg.model.model_version,
            "preprocess_version": cfg.preprocess.preprocess_version,
            "index_version": cfg.index.index_version,
        }
    )


app.include_router(router)
