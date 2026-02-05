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
class AppConfig:
    data: DataConfig
    artifacts: ArtifactsConfig
    model: ModelConfig
    preprocess: PreprocessConfig
    index: IndexConfig


def load_config(path: str) -> AppConfig:
    with open(path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)

    data = raw["data"]
    artifacts = raw["artifacts"]
    model = raw["model"]
    preprocess = raw["preprocess"]
    index = raw["index"]

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
    )
