from __future__ import annotations

from typing import List, Optional

from pydantic import BaseModel, Field


class CropBox(BaseModel):
    x: int = Field(ge=0)
    y: int = Field(ge=0)
    w: int = Field(gt=0)
    h: int = Field(gt=0)


class SearchResponseItem(BaseModel):
    image_id: str
    uri: str
    score: float
    position: int


class SearchResponse(BaseModel):
    search_id: str
    model_version: str
    preprocess_version: str
    index_version: str
    top_k: int
    results: List[SearchResponseItem]


class ClickEvent(BaseModel):
    search_id: str
    query_sha256: str
    image_id: str
    position: int
