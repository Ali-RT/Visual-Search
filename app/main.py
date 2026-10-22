from __future__ import annotations

import json
import os
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from app.api import router
from app.config import load_config
from app.embedding_service import EmbeddingService
from app.index_service import IndexService
from app.interaction_store import InteractionStore
from app.tagging_service import TaggingService


CONFIG_PATH = os.environ.get("CONFIG_PATH", "configs/adas_usecase.yaml")

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

    embedder = EmbeddingService(
        backbone=cfg.model.backbone,
        pretrained=cfg.model.pretrained,
        device=cfg.model.device,
    )

    # Choose vector store backend
    if cfg.vector_store == "qdrant" and cfg.qdrant is not None:
        from app.qdrant_index_service import QdrantIndexService

        index_service = QdrantIndexService(
            host=cfg.qdrant.host,
            port=cfg.qdrant.port,
            collection_name=cfg.qdrant.collection_name,
        )
    else:
        paths = _resolve_artifact_paths(cfg)
        index_service = IndexService(
            embeddings_f32_path=paths["embeddings_f32"],
            id_map_path=paths["id_map"],
            index_bin_path=paths["index_bin"],
            index_meta_path=paths["index_meta"],
            ef_search=cfg.index.ef_search,
        )

    interaction_store = InteractionStore(cfg.data.sqlite_path)

    tagger = None
    if cfg.tagging and cfg.tagging.enabled:
        tagger = TaggingService(
            backbone=cfg.tagging.backbone or cfg.model.backbone,
            pretrained=cfg.tagging.pretrained or cfg.model.pretrained,
            device=cfg.model.device,
            tag_groups=cfg.tagging.tag_groups,
            tag_labels=cfg.tagging.tag_labels,
        )

    detector = None
    try:
        from app.detection_service import DetectionService
        detector = DetectionService(device=cfg.model.device)
    except Exception:
        pass

    segmenter = None
    try:
        from app.segmentation_service import SegmentationService
        segment_model_id = "nvidia/segformer-b2-finetuned-cityscapes-1024-1024"
        if cfg.segmentation and cfg.segmentation.model_id:
            segment_model_id = cfg.segmentation.model_id
        segmenter = SegmentationService(model_id=segment_model_id, device=cfg.model.device)
    except Exception:
        pass

    sam2_segmenter = None
    if cfg.sam2 and cfg.sam2.enabled:
        try:
            from app.sam2_service import SAM2SegmentationService
            sam2_segmenter = SAM2SegmentationService(
                model_id=cfg.sam2.model_id,
                device=cfg.model.device,
            )
        except Exception:
            sam2_segmenter = None

    quality_svc = None
    try:
        from app.quality_service import QualityService
        quality_svc = QualityService(clip_tagger=tagger)
    except Exception:
        pass

    services.clear()
    services.update(
        {
            "embedder": embedder,
            "index": index_service,
            "interaction_store": interaction_store,
            "tagger": tagger,
            "detector": detector,
            "segmenter": segmenter,
            "sam2_segmenter": sam2_segmenter,
            "quality": quality_svc,
            "image_dir": str(cfg.data.image_dir),
            "vector_store": cfg.vector_store,
            "model_version": cfg.model.model_version,
            "preprocess_version": cfg.preprocess.preprocess_version,
            "index_version": cfg.index.index_version,
        }
    )


app.include_router(router)
app.mount("/static", StaticFiles(directory="."), name="static")
