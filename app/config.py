from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional

import yaml


@dataclass(frozen=True)
class DataConfig:
    image_dir: Path
    sqlite_path: Path
    allowed_ext: List[str]


@dataclass(frozen=True)
class ArtifactsConfig:
    root_dir: Path


@dataclass(frozen=True)
class ModelConfig:
    model_version: str
    backbone: str
    pretrained: str
    device: str  # "cpu" | "cuda" | "auto"


@dataclass(frozen=True)
class PreprocessConfig:
    preprocess_version: str


@dataclass(frozen=True)
class IndexConfig:
    index_version: str
    space: str
    ef_construction: int
    M: int
    ef_search: int
    top_k: int
    top_k_candidates: int


@dataclass(frozen=True)
class TaggingConfig:
    enabled: bool
    tag_groups: Optional[Dict[str, List[str]]]
    tag_labels: Optional[Dict[str, List[str]]]
    backbone: Optional[str] = None
    pretrained: Optional[str] = None


@dataclass(frozen=True)
class SegmentationConfig:
    model_id: str


@dataclass(frozen=True)
class Sam2Config:
    enabled: bool
    model_id: str


@dataclass(frozen=True)
class QdrantConfig:
    host: str
    port: int
    collection_name: str


@dataclass(frozen=True)
class AppConfig:
    data: DataConfig
    artifacts: ArtifactsConfig
    model: ModelConfig
    preprocess: PreprocessConfig
    index: IndexConfig
    vector_store: str = "hnswlib"  # "hnswlib" | "qdrant"
    qdrant: Optional[QdrantConfig] = None
    tagging: Optional[TaggingConfig] = None
    segmentation: Optional[SegmentationConfig] = None
    sam2: Optional[Sam2Config] = None


def load_config(path: str) -> AppConfig:
    with open(path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)

    data = raw["data"]
    artifacts = raw["artifacts"]
    model = raw["model"]
    preprocess = raw["preprocess"]
    index = raw["index"]

    tagging_cfg = None
    tagging = raw.get("tagging")
    if tagging and tagging.get("enabled", False):
        tagging_cfg = TaggingConfig(
            enabled=True,
            tag_groups=tagging.get("tag_groups"),
            tag_labels=tagging.get("tag_labels"),
            backbone=tagging.get("backbone"),
            pretrained=tagging.get("pretrained"),
        )

    qdrant_cfg = None
    qdrant_raw = raw.get("qdrant")
    if qdrant_raw:
        qdrant_cfg = QdrantConfig(
            host=str(qdrant_raw.get("host", "localhost")),
            port=int(qdrant_raw.get("port", 6333)),
            collection_name=str(qdrant_raw.get("collection_name", "images")),
        )

    segmentation_cfg = None
    segmentation_raw = raw.get("segmentation")
    if segmentation_raw:
        segmentation_cfg = SegmentationConfig(
            model_id=str(segmentation_raw.get("model_id", "nvidia/segformer-b2-finetuned-cityscapes-1024-1024")),
        )

    sam2_cfg = None
    sam2_raw = raw.get("sam2")
    if sam2_raw:
        sam2_cfg = Sam2Config(
            enabled=bool(sam2_raw.get("enabled", False)),
            model_id=str(sam2_raw.get("model_id", "facebook/sam2-hiera-small")),
        )

    return AppConfig(
        data=DataConfig(
            image_dir=Path(data["image_dir"]),
            sqlite_path=Path(data["sqlite_path"]),
            allowed_ext=list(data["allowed_ext"]),
        ),
        artifacts=ArtifactsConfig(
            root_dir=Path(artifacts["root_dir"]),
        ),
        model=ModelConfig(
            model_version=str(model["model_version"]),
            backbone=str(model["backbone"]),
            pretrained=str(model["pretrained"]),
            device=str(model["device"]),
        ),
        preprocess=PreprocessConfig(
            preprocess_version=str(preprocess["preprocess_version"]),
        ),
        index=IndexConfig(
            index_version=str(index["index_version"]),
            space=str(index["space"]),
            ef_construction=int(index["ef_construction"]),
            M=int(index["M"]),
            ef_search=int(index["ef_search"]),
            top_k=int(index["top_k"]),
            top_k_candidates=int(index["top_k_candidates"]),
        ),
        vector_store=str(raw.get("vector_store", "hnswlib")),
        qdrant=qdrant_cfg,
        tagging=tagging_cfg,
        segmentation=segmentation_cfg,
        sam2=sam2_cfg,
    )
