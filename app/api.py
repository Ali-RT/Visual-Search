from __future__ import annotations

import hashlib
import io
import uuid
from typing import Optional

import numpy as np
from fastapi import APIRouter, File, HTTPException, UploadFile
from PIL import Image

from app.schemas import ClickEvent, CropBox, SearchResponse, SearchResponseItem


router = APIRouter()


def _apply_crop(img: Image.Image, crop: Optional[CropBox]) -> Image.Image:
    if crop is None:
        return img
    x0 = crop.x
    y0 = crop.y
    x1 = crop.x + crop.w
    y1 = crop.y + crop.h
    x1 = min(x1, img.width)
    y1 = min(y1, img.height)
    x0 = min(x0, img.width - 1)
    y0 = min(y0, img.height - 1)
    if x1 <= x0 or y1 <= y0:
        return img
    return img.crop((x0, y0, x1, y1))


@router.post("/search", response_model=SearchResponse)
async def search(
    file: UploadFile = File(...),
    top_k: int = 50,
    top_k_candidates: int = 200,
    crop_x: Optional[int] = None,
    crop_y: Optional[int] = None,
    crop_w: Optional[int] = None,
    crop_h: Optional[int] = None,
) -> SearchResponse:
    try:
        content = await file.read()
        query_sha256 = hashlib.sha256(content).hexdigest()
        img = Image.open(io.BytesIO(content)).convert("RGB")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to read image: {e}")

    crop = None
    if None not in (crop_x, crop_y, crop_w, crop_h):
        crop = CropBox(x=int(crop_x), y=int(crop_y), w=int(crop_w), h=int(crop_h))

    img = _apply_crop(img, crop)

    # Lazy import to avoid circular refs
    from app.main import services
    from app.interaction_store import ImpressionRow, now_unix

    emb_out = services["embedder"].embed_pil(img)
    q = emb_out.embedding
    q = q / (np.linalg.norm(q) + 1e-12)

    cand_labels, _ = services["index"].ann_query(q, top_k_candidates=int(top_k_candidates))
    reranked = services["index"].rerank_cosine(q, cand_labels, top_k=int(top_k))

    search_id = str(uuid.uuid4())
    ts = now_unix()

    # Log impressions (assumes UI displays returned results)
    impression_rows = [
        ImpressionRow(
            search_id=search_id,
            query_sha256=query_sha256,
            image_id=r["image_id"],
            position=i + 1,
            timestamp=ts,
        )
        for i, r in enumerate(reranked)
    ]
    services["interaction_store"].log_impressions(impression_rows)

    return SearchResponse(
        search_id=search_id,
        model_version=services["model_version"],
        preprocess_version=services["preprocess_version"],
        index_version=services["index_version"],
        top_k=int(top_k),
        results=[
            SearchResponseItem(
                image_id=r["image_id"],
                uri=r["uri"],
                score=r["score"],
                position=i + 1,
            )
            for i, r in enumerate(reranked)
        ],
    )


@router.post("/click")
async def click(event: ClickEvent) -> dict:
    from app.main import services

    services["interaction_store"].log_click(
        search_id=event.search_id,
        query_sha256=event.query_sha256,
        image_id=event.image_id,
        position=event.position,
    )
    return {"status": "ok"}