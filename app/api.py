from __future__ import annotations

import hashlib
import io
import os
from contextlib import nullcontext
import shutil
import subprocess
import tempfile
import time
import uuid
from pathlib import Path
from typing import Optional

import numpy as np
from fastapi import APIRouter, File, HTTPException, UploadFile
from PIL import Image

from app.schemas import ClickEvent, CropBox, DetectionItem, DetectionResponse, EvalResultItem, EvalSearchRequest, EvalSearchResponse, HybridTagItem, HybridTagResponse, MetricAtK, QualityItem, QualityMetricsResponse, QualityWorstResponse, SearchResponse, SearchResponseItem, SegmentInfoItem, SegmentationResponse, SimilarSearchRequest, TagResponse, TextSearchRequest, VideoSegmentationResponse


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


def _default_trace_path() -> Optional[Path]:
    traces_dir = Path("traces")
    if not traces_dir.exists():
        return None
    candidates = sorted(traces_dir.glob("*.mf4"))
    return candidates[0] if candidates else None


def _infer_trace_path_for_video(video_filename: str) -> Optional[Path]:
    traces_dir = Path("traces")
    if not traces_dir.exists():
        return None
    candidates = sorted(traces_dir.glob("*.mf4"))
    if not candidates:
        return None

    base = Path(video_filename).name
    normalized = base
    for ext in [".mp4", ".webm", ".mov", ".mkv", ".avi"]:
        if normalized.endswith(ext):
            normalized = normalized[: -len(ext)]
    if normalized.endswith(".avi"):
        normalized = normalized[:-4]

    best: Optional[Path] = None
    best_score = -1
    for c in candidates:
        stem = c.stem
        score = 0
        max_len = min(len(stem), len(normalized))
        for i in range(max_len):
            if stem[i] != normalized[i]:
                break
            score += 1
        if score > best_score:
            best_score = score
            best = c

    if best is not None and best_score >= 12:
        return best
    return _default_trace_path()


def _build_ultrasonic_prompt_points_by_frame(trace_path: Path, frame_count: int) -> list[list[tuple[float, float]]]:
    """
    Convert MF4 ultrasonic visibility channels into per-frame normalized prompt points.
    This uses MAP visibility bins as a practical proxy for obstacle location.
    """
    if frame_count <= 0:
        return []

    from asammdf import MDF

    mdf = MDF(str(trace_path))
    point_channels: list[tuple[int, str]] = []
    straight_channels: list[tuple[int, str]] = []
    for i in range(1, 21):
        pn = f"MAP_ObjVisPoint_{i:02d}"
        sn = f"MAP_ObjVisStraight_{i:02d}"
        if pn in mdf.channels_db:
            point_channels.append((i, pn))
        if sn in mdf.channels_db:
            straight_channels.append((i, sn))

    if not point_channels:
        return [[] for _ in range(frame_count)]

    point_data: list[tuple[int, np.ndarray]] = []
    straight_data: list[tuple[int, np.ndarray]] = []
    min_len = None

    for idx, name in point_channels:
        vals = np.asarray(mdf.get(name).samples, dtype=np.float32)
        point_data.append((idx, vals))
        min_len = len(vals) if min_len is None else min(min_len, len(vals))
    for idx, name in straight_channels:
        vals = np.asarray(mdf.get(name).samples, dtype=np.float32)
        straight_data.append((idx, vals))
        min_len = len(vals) if min_len is None else min(min_len, len(vals))

    if not min_len or min_len <= 0:
        return [[] for _ in range(frame_count)]

    def _channel_scalar(name: str, default: float) -> float:
        if name not in mdf.channels_db:
            return default
        vals = np.asarray(mdf.get(name).samples, dtype=np.float32)
        if vals.size == 0:
            return default
        return float(vals[0])

    # Prefer rear-sector geometry channels when available.
    ooc_start_deg = _channel_scalar("OOC_start_angle", 270.0)
    ooc_sweep_deg = _channel_scalar("OOC_sweep_angle", 180.0)

    trace_points: list[list[tuple[float, float]]] = []
    n_bins = max(1, len(point_data))
    n_straight = max(1, len(straight_data))
    for t in range(min_len):
        active_point_bins = [idx for idx, vals in point_data if vals[t] > 0.0]
        active_straight_bins = [idx for idx, vals in straight_data if vals[t] > 0.0]

        pts: list[tuple[float, float]] = []
        for b in active_point_bins[:4]:
            # Angle across rear sweep; bins map left/right around the rear arc.
            frac = (b - 0.5) / n_bins
            theta = np.deg2rad(ooc_start_deg + frac * ooc_sweep_deg)

            # Convert straight-bin activity to a rough obstacle distance in meters.
            if active_straight_bins:
                s = float(np.mean(active_straight_bins))
                s_frac = (s - 0.5) / n_straight
                dist_m = float(np.clip(0.5 + 3.0 * s_frac, 0.4, 4.5))
            else:
                dist_m = 1.8

            # Rear-sector pseudo-geometry in vehicle coordinates.
            lateral = float(np.sin(theta))
            rearward = float(np.cos(theta))
            rearward = max(0.05, rearward)
            lateral_m = dist_m * 0.95 * lateral
            rearward_m = dist_m * rearward

            # Project to image with a simple perspective model.
            x_norm = 0.5 + (lateral_m / (rearward_m + 0.9)) * 0.52
            y_norm = 0.94 - (rearward_m / (rearward_m + 1.2)) * 0.78
            x_norm = float(np.clip(x_norm, 0.02, 0.98))
            y_norm = float(np.clip(y_norm, 0.08, 0.97))
            pts.append((x_norm, y_norm))
        trace_points.append(pts)

    # Sample/align trace sequence to video frames.
    per_frame: list[list[tuple[float, float]]] = []
    for fidx in range(frame_count):
        tidx = int(round((fidx / max(1, frame_count - 1)) * (len(trace_points) - 1)))
        per_frame.append(trace_points[tidx] if trace_points else [])

    # Forward-fill empty frames to reduce flicker.
    last_non_empty: list[tuple[float, float]] = []
    for i in range(frame_count):
        if per_frame[i]:
            last_non_empty = per_frame[i]
        elif last_non_empty:
            per_frame[i] = list(last_non_empty)

    return per_frame


@router.post("/search", response_model=SearchResponse)
async def search(
    file: UploadFile = File(...),
    top_k: int = 50,
    top_k_candidates: int = 200,
    use_case: Optional[str] = None,
    label: Optional[str] = None,
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

    from app.main import services
    from app.interaction_store import ImpressionRow, now_unix

    emb_out = services["embedder"].embed_pil(img)
    q = emb_out.embedding
    q = q / (np.linalg.norm(q) + 1e-12)

    # Route to the right backend
    if services.get("vector_store") == "qdrant":
        reranked = services["index"].search(
            query_vec=q,
            top_k=int(top_k),
            use_case_filter=use_case,
            label_filter=label,
        )
    else:
        cand_labels, _ = services["index"].ann_query(q, top_k_candidates=int(top_k_candidates))
        reranked = services["index"].rerank_cosine(q, cand_labels, top_k=int(top_k))

    search_id = str(uuid.uuid4())
    ts = now_unix()

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
                use_case=r.get("use_case"),
                label=r.get("label"),
            )
            for i, r in enumerate(reranked)
        ],
    )


@router.post("/search/text", response_model=SearchResponse)
async def search_text(req: TextSearchRequest) -> SearchResponse:
    from app.main import services
    from app.interaction_store import ImpressionRow, now_unix

    emb_out = services["embedder"].embed_text(req.query)
    q = emb_out.embedding
    q = q / (np.linalg.norm(q) + 1e-12)

    query_sha256 = hashlib.sha256(req.query.encode()).hexdigest()

    if services.get("vector_store") == "qdrant":
        reranked = services["index"].search(
            query_vec=q,
            top_k=req.top_k,
            use_case_filter=req.use_case,
        )
    else:
        cand_labels, _ = services["index"].ann_query(q, top_k_candidates=200)
        reranked = services["index"].rerank_cosine(q, cand_labels, top_k=req.top_k)

    search_id = str(uuid.uuid4())
    ts = now_unix()

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
        top_k=req.top_k,
        results=[
            SearchResponseItem(
                image_id=r["image_id"],
                uri=r["uri"],
                score=r["score"],
                position=i + 1,
                use_case=r.get("use_case"),
                label=r.get("label"),
            )
            for i, r in enumerate(reranked)
        ],
    )


@router.post("/search/eval", response_model=EvalSearchResponse)
async def eval_text_search(req: EvalSearchRequest) -> EvalSearchResponse:
    """
    Evaluate text-to-image search using label-derived ground truth.

    Only images with gt_tags in their payload are used for relevance
    judgement.  Images without gt_tags are treated as unjudged.
    """
    import math

    from qdrant_client.models import FieldCondition, Filter, MatchValue

    from app.main import services

    if services.get("vector_store") != "qdrant":
        raise HTTPException(status_code=501, detail="Eval requires Qdrant backend")

    emb_out = services["embedder"].embed_text(req.query)
    q = emb_out.embedding
    q = q / (np.linalg.norm(q) + 1e-12)

    max_k = max(req.k_values) if req.k_values else req.top_k

    reranked = services["index"].search(query_vec=q, top_k=max_k)

    # Count total relevant images in DB (those with the gt_tag)
    try:
        count_result = services["index"].client.count(
            collection_name=services["index"].collection_name,
            count_filter=Filter(must=[
                FieldCondition(key="gt_tags", match=MatchValue(value=req.relevant_tag)),
            ]),
        )
        total_relevant = count_result.count
    except Exception:
        total_relevant = 0

    # For each result, fetch gt_tags from payload to determine relevance
    result_items: list[EvalResultItem] = []
    relevance: list[int] = []

    for i, r in enumerate(reranked):
        gt_tags = r.get("gt_tags", [])
        source = r.get("source", "")

        if isinstance(gt_tags, list):
            is_rel = req.relevant_tag in gt_tags
        else:
            is_rel = False

        relevance.append(1 if is_rel else 0)
        result_items.append(EvalResultItem(
            image_id=r["image_id"],
            uri=r["uri"],
            score=r["score"],
            position=i + 1,
            use_case=r.get("use_case"),
            gt_tags=gt_tags if isinstance(gt_tags, list) else [],
            is_relevant=is_rel,
            source=source,
        ))

    metrics: list[MetricAtK] = []
    for k in sorted(req.k_values):
        k_actual = min(k, len(relevance))
        rel_at_k = relevance[:k_actual]
        relevant_count = sum(rel_at_k)

        precision = relevant_count / k_actual if k_actual > 0 else 0.0
        recall = relevant_count / total_relevant if total_relevant > 0 else 0.0

        # nDCG@k
        dcg = sum(rel / math.log2(idx + 2) for idx, rel in enumerate(rel_at_k))
        ideal_rels = sorted(rel_at_k, reverse=True)
        idcg = sum(rel / math.log2(idx + 2) for idx, rel in enumerate(ideal_rels))
        ndcg = dcg / idcg if idcg > 0 else 0.0

        metrics.append(MetricAtK(
            k=k, precision=round(precision, 4), recall=round(recall, 4),
            ndcg=round(ndcg, 4), relevant_count=relevant_count, total=k_actual,
        ))

    return EvalSearchResponse(
        query=req.query,
        relevant_tag=req.relevant_tag,
        total_relevant_in_db=total_relevant,
        results_count=len(reranked),
        metrics=metrics,
        results=result_items,
    )


@router.post("/search/similar", response_model=SearchResponse)
async def search_similar(req: SimilarSearchRequest) -> SearchResponse:
    from app.main import services

    if services.get("vector_store") != "qdrant":
        raise HTTPException(status_code=501, detail="Similar search requires Qdrant backend")

    reranked = services["index"].search_by_id(
        image_id=req.image_id,
        top_k=req.top_k,
        use_case_filter=req.use_case,
    )

    search_id = str(uuid.uuid4())
    return SearchResponse(
        search_id=search_id,
        model_version=services["model_version"],
        preprocess_version=services["preprocess_version"],
        index_version=services["index_version"],
        top_k=req.top_k,
        results=[
            SearchResponseItem(
                image_id=r["image_id"],
                uri=r["uri"],
                score=r["score"],
                position=i + 1,
                use_case=r.get("use_case"),
                label=r.get("label"),
            )
            for i, r in enumerate(reranked)
        ],
    )


@router.post("/search/hybrid", response_model=SearchResponse)
async def search_hybrid(
    file: UploadFile = File(...),
    text: str = "",
    image_weight: float = 0.7,
    top_k: int = 50,
    use_case: Optional[str] = None,
    crop_x: Optional[int] = None,
    crop_y: Optional[int] = None,
    crop_w: Optional[int] = None,
    crop_h: Optional[int] = None,
) -> SearchResponse:
    from app.main import services
    from app.interaction_store import ImpressionRow, now_unix

    try:
        content = await file.read()
        img = Image.open(io.BytesIO(content)).convert("RGB")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to read image: {e}")

    crop = None
    if None not in (crop_x, crop_y, crop_w, crop_h):
        crop = CropBox(x=int(crop_x), y=int(crop_y), w=int(crop_w), h=int(crop_h))
    img = _apply_crop(img, crop)

    img_emb = services["embedder"].embed_pil(img).embedding
    img_emb = img_emb / (np.linalg.norm(img_emb) + 1e-12)

    if text.strip():
        txt_emb = services["embedder"].embed_text(text.strip()).embedding
        txt_emb = txt_emb / (np.linalg.norm(txt_emb) + 1e-12)
        w = max(0.0, min(1.0, image_weight))
        q = w * img_emb + (1.0 - w) * txt_emb
        q = q / (np.linalg.norm(q) + 1e-12)
    else:
        q = img_emb

    query_sha256 = hashlib.sha256(content + text.encode()).hexdigest()

    if services.get("vector_store") == "qdrant":
        reranked = services["index"].search(
            query_vec=q, top_k=int(top_k), use_case_filter=use_case,
        )
    else:
        cand_labels, _ = services["index"].ann_query(q, top_k_candidates=200)
        reranked = services["index"].rerank_cosine(q, cand_labels, top_k=int(top_k))

    search_id = str(uuid.uuid4())
    ts = now_unix()
    impression_rows = [
        ImpressionRow(
            search_id=search_id, query_sha256=query_sha256,
            image_id=r["image_id"], position=i + 1, timestamp=ts,
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
                image_id=r["image_id"], uri=r["uri"], score=r["score"],
                position=i + 1, use_case=r.get("use_case"), label=r.get("label"),
            )
            for i, r in enumerate(reranked)
        ],
    )


@router.post("/tag", response_model=TagResponse)
async def tag_image(
    file: UploadFile = File(...),
) -> TagResponse:
    try:
        content = await file.read()
        img = Image.open(io.BytesIO(content)).convert("RGB")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to read image: {e}")

    from app.main import services

    tagger = services.get("tagger")
    if tagger is None:
        raise HTTPException(status_code=503, detail="Tagging service not loaded")

    tag_output = tagger.tag_pil(img)
    return TagResponse(
        tags=tag_output.to_dict(),
        flat_labels=tag_output.flat_labels(),
    )


UNCERTAINTY_THRESHOLD = 0.55

VEHICLE_LABELS = {"vehicle", "car", "truck", "bus", "motorcycle", "bicycle"}
PERSON_LABELS = {"pedestrian", "person", "rider"}

SEGMENTATION_OBJECTS = {
    "car", "truck", "bus", "train", "motorcycle", "bicycle",
    "person", "rider", "traffic light", "traffic sign", "pole",
}


def _derive_detection_tags(detections: list) -> list[HybridTagItem]:
    """Derive object-presence and count tags from Grounding DINO detections."""
    tags: list[HybridTagItem] = []
    label_counts: dict[str, int] = {}
    for d in detections:
        label_counts[d.label] = label_counts.get(d.label, 0) + 1

    vehicle_count = sum(c for l, c in label_counts.items() if l in VEHICLE_LABELS)
    person_count = sum(c for l, c in label_counts.items() if l in PERSON_LABELS)

    if vehicle_count > 0:
        if vehicle_count >= 8:
            density = "heavy"
        elif vehicle_count >= 3:
            density = "moderate"
        else:
            density = "low"
        tags.append(HybridTagItem(
            group="vehicle_density", label=density, confidence=0.95,
            source="detection", details={"vehicle_count": vehicle_count, "breakdown": {l: c for l, c in label_counts.items() if l in VEHICLE_LABELS}},
        ))
    else:
        tags.append(HybridTagItem(
            group="vehicle_density", label="none", confidence=0.90,
            source="detection", details={"vehicle_count": 0},
        ))

    tags.append(HybridTagItem(
        group="pedestrian", label="present" if person_count > 0 else "absent",
        confidence=0.95 if person_count > 0 else 0.85,
        source="detection", details={"count": person_count},
    ))

    for label in ["crosswalk", "traffic sign", "traffic light", "lane marking"]:
        count = label_counts.get(label, 0)
        tag_name = label.replace(" ", "_")
        tags.append(HybridTagItem(
            group=tag_name, label="present" if count > 0 else "absent",
            confidence=0.90 if count > 0 else 0.70,
            source="detection", uncertain=count == 0,
            details={"count": count},
        ))

    return tags


def _derive_segmentation_tags(segments: list) -> list[HybridTagItem]:
    """Derive layout/coverage tags from SegFormer segmentation."""
    tags: list[HybridTagItem] = []
    seg_map = {s.label: s.percentage for s in segments}

    road_pct = seg_map.get("road", 0)
    sidewalk_pct = seg_map.get("sidewalk", 0)
    veg_pct = seg_map.get("vegetation", 0) + seg_map.get("terrain", 0)
    sky_pct = seg_map.get("sky", 0)
    building_pct = seg_map.get("building", 0) + seg_map.get("wall", 0) + seg_map.get("fence", 0)

    tags.append(HybridTagItem(
        group="road_coverage", label=f"{road_pct:.0f}%", confidence=0.95,
        source="segmentation", details={"road": round(road_pct, 1), "sidewalk": round(sidewalk_pct, 1)},
    ))

    tags.append(HybridTagItem(
        group="vegetation_coverage", label=f"{veg_pct:.0f}%", confidence=0.95,
        source="segmentation", details={"vegetation_terrain": round(veg_pct, 1)},
    ))

    tags.append(HybridTagItem(
        group="sky_visibility", label=f"{sky_pct:.0f}%", confidence=0.95,
        source="segmentation", details={"sky": round(sky_pct, 1)},
    ))

    tags.append(HybridTagItem(
        group="built_environment", label=f"{building_pct:.0f}%", confidence=0.95,
        source="segmentation", details={"building_wall_fence": round(building_pct, 1)},
    ))

    obj_labels = [s for s in segments if s.label in SEGMENTATION_OBJECTS and s.percentage >= 0.3]
    if obj_labels:
        tags.append(HybridTagItem(
            group="detected_segments",
            label=", ".join(f"{s.label} ({s.percentage:.1f}%)" for s in obj_labels),
            confidence=0.90, source="segmentation",
            details={s.label: round(s.percentage, 1) for s in obj_labels},
        ))

    return tags


@router.post("/tag/hybrid", response_model=HybridTagResponse)
async def tag_image_hybrid(
    file: UploadFile = File(...),
) -> HybridTagResponse:
    from app.main import services

    try:
        content = await file.read()
        img = Image.open(io.BytesIO(content)).convert("RGB")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to read image: {e}")

    scene_tags: list[HybridTagItem] = []
    object_tags: list[HybridTagItem] = []
    layout_tags: list[HybridTagItem] = []

    # --- Stage 1: CLIP scene tags (time_of_day, scene_type) ---
    tagger = services.get("tagger")
    if tagger is not None:
        tag_output = tagger.tag_pil(img)
        for t in tag_output.tags:
            if t.group in ("time_of_day", "scene_type"):
                scene_tags.append(HybridTagItem(
                    group=t.group, label=t.predicted_label,
                    confidence=round(t.confidence, 4), source="clip",
                    uncertain=t.confidence < UNCERTAINTY_THRESHOLD,
                    details={"scores": {k: round(v, 4) for k, v in t.all_scores.items()}},
                ))

    # --- Stage 2: Detection-derived tags ---
    detector = services.get("detector")
    if detector is not None:
        detections = detector.detect(img, threshold=0.25)
        det_items = [DetectionItem(label=d.label, score=d.score, box=d.box) for d in detections]
        object_tags = _derive_detection_tags(det_items)

    # --- Stage 3: Segmentation-derived tags ---
    segmenter = services.get("segmenter")
    if segmenter is not None:
        seg_result = segmenter.segment(img)
        seg_items = [SegmentInfoItem(
            class_id=s.class_id, label=s.label,
            pixel_count=s.pixel_count, percentage=s.percentage,
            color=list(s.color),
        ) for s in seg_result.segments]
        layout_tags = _derive_segmentation_tags(seg_items)

    summary: dict = {}
    for t in scene_tags + object_tags + layout_tags:
        summary[t.group] = t.label

    return HybridTagResponse(
        scene_tags=scene_tags,
        object_tags=object_tags,
        layout_tags=layout_tags,
        summary=summary,
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


@router.post("/detect", response_model=DetectionResponse)
async def detect_objects(
    file: UploadFile = File(...),
    prompt: Optional[str] = None,
    threshold: float = 0.3,
) -> DetectionResponse:
    from app.main import services

    detector = services.get("detector")
    if detector is None:
        raise HTTPException(status_code=503, detail="Detection service not loaded")

    try:
        content = await file.read()
        img = Image.open(io.BytesIO(content)).convert("RGB")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to read image: {e}")

    detections = detector.detect(img, text_prompt=prompt, threshold=threshold)
    return DetectionResponse(
        detections=[DetectionItem(label=d.label, score=d.score, box=d.box) for d in detections],
        width=img.width,
        height=img.height,
    )


@router.post("/segment", response_model=SegmentationResponse)
async def segment_image(
    file: UploadFile = File(...),
) -> SegmentationResponse:
    from app.main import services

    segmenter = services.get("segmenter")
    if segmenter is None:
        raise HTTPException(status_code=503, detail="Segmentation service not loaded")

    try:
        content = await file.read()
        img = Image.open(io.BytesIO(content)).convert("RGB")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to read image: {e}")

    result = segmenter.segment(img)
    return SegmentationResponse(
        segments=[
            SegmentInfoItem(
                class_id=s.class_id, label=s.label,
                pixel_count=s.pixel_count, percentage=s.percentage,
                color=list(s.color),
            )
            for s in result.segments
        ],
        width=result.width,
        height=result.height,
        colored_mask_b64=result.colored_mask_b64,
    )


@router.post("/segment/video", response_model=VideoSegmentationResponse)
async def segment_video(
    file: UploadFile = File(...),
    alpha: float = 0.45,
    sample_every_n: int = 1,
    max_frames: int = 300,
    mode: str = "segformer",
    sam2_profile: str = "easy",
    click_x: Optional[float] = None,
    click_y: Optional[float] = None,
    click_frame_idx: Optional[int] = None,
    prompt_points: Optional[str] = None,
    dino_prompt: Optional[str] = None,
    dino_threshold: float = 0.3,
    dino_max_prompts: int = 4,
    grounded_ground_prompt: str = "curb. curb stone. concrete curb. sidewalk curb. road edge. raised curb. curb edge. sidewalk edge. street ground. road. asphalt. floor.",
    grounded_other_prompt: str = "parking curb stop. wheel stop. parking block. curb stop. parking stop. wheel chock. wooden beam. wooden plank. timber. wooden barrier. wooden block. barrier. block. step. raised edge.",
    grounded_model_id: str = "facebook/sam2.1-hiera-small",
    grounded_max_prompts: int = 3,
    use_ultrasonic_prompt: bool = False,
    ultrasonic_trace_path: Optional[str] = None,
) -> VideoSegmentationResponse:
    from app.main import services

    upload_name = Path(file.filename or "uploaded_video.mp4").name
    mode = mode.strip().lower()
    if mode not in {"segformer", "sam2", "sam2_track", "dino_sam2_track", "grounded_sam2"}:
        raise HTTPException(status_code=400, detail="mode must be one of: segformer, sam2, sam2_track, dino_sam2_track, grounded_sam2")
    sam2_profile = sam2_profile.strip().lower()
    if sam2_profile not in {"easy", "balanced", "quality"}:
        raise HTTPException(status_code=400, detail="sam2_profile must be one of: easy, balanced, quality")
    if (click_x is None) != (click_y is None):
        raise HTTPException(status_code=400, detail="click_x and click_y must be provided together")
    click_point: tuple[float, float] | None = None
    if click_x is not None and click_y is not None:
        if not (0.0 <= click_x <= 1.0 and 0.0 <= click_y <= 1.0):
            raise HTTPException(status_code=400, detail="click_x and click_y must be in [0, 1]")
        click_point = (float(click_x), float(click_y))
    parsed_prompt_points: list[tuple[float, float]] = []
    if prompt_points:
        try:
            for part in prompt_points.split(";"):
                p = part.strip()
                if not p:
                    continue
                xs, ys = p.split(",", 1)
                px = float(xs.strip())
                py = float(ys.strip())
                if not (0.0 <= px <= 1.0 and 0.0 <= py <= 1.0):
                    raise ValueError("out of range")
                parsed_prompt_points.append((px, py))
        except Exception:
            raise HTTPException(
                status_code=400,
                detail="prompt_points must be ';'-separated 'x,y' pairs in [0,1], e.g. '0.4,0.3;0.42,0.31'",
            )
    if click_point is not None:
        parsed_prompt_points.append(click_point)

    segmenter = services.get("segmenter")
    sam2_segmenter = services.get("sam2_segmenter")
    detector = services.get("detector")
    if mode == "segformer" and segmenter is None:
        raise HTTPException(status_code=503, detail="SegFormer service not loaded")
    if mode in {"sam2", "sam2_track", "dino_sam2_track"} and sam2_segmenter is None:
        raise HTTPException(status_code=503, detail="SAM2 service not loaded")
    if mode in {"dino_sam2_track", "grounded_sam2"} and detector is None:
        raise HTTPException(status_code=503, detail="Detection service not loaded")
    if mode == "grounded_sam2" and sam2_segmenter is None:
        raise HTTPException(status_code=503, detail="SAM2 service not loaded")
    if mode in {"sam2_track", "dino_sam2_track"} and not parsed_prompt_points:
        if not use_ultrasonic_prompt:
            if mode == "sam2_track":
                raise HTTPException(status_code=400, detail="sam2_track mode requires click_x/click_y or prompt_points (or enable use_ultrasonic_prompt)")
    if click_frame_idx is not None and click_frame_idx < 0:
        raise HTTPException(status_code=400, detail="click_frame_idx must be >= 0")
    if not (0.0 <= dino_threshold <= 1.0):
        raise HTTPException(status_code=400, detail="dino_threshold must be in [0, 1]")
    if dino_max_prompts < 1:
        raise HTTPException(status_code=400, detail="dino_max_prompts must be >= 1")
    if grounded_max_prompts < 1:
        raise HTTPException(status_code=400, detail="grounded_max_prompts must be >= 1")
    if mode == "dino_sam2_track" and use_ultrasonic_prompt:
        raise HTTPException(status_code=400, detail="dino_sam2_track cannot be combined with ultrasonic prompts")
    if mode == "grounded_sam2" and use_ultrasonic_prompt:
        raise HTTPException(status_code=400, detail="grounded_sam2 cannot be combined with ultrasonic prompts")
    trace_path: Optional[Path] = None
    if use_ultrasonic_prompt:
        if ultrasonic_trace_path:
            trace_path = Path(ultrasonic_trace_path)
        else:
            trace_path = _infer_trace_path_for_video(upload_name)
        if trace_path is None or not trace_path.exists():
            raise HTTPException(status_code=400, detail="Ultrasonic prompt enabled but MF4 trace file not found")

    if sample_every_n < 1:
        raise HTTPException(status_code=400, detail="sample_every_n must be >= 1")
    if max_frames < 1:
        raise HTTPException(status_code=400, detail="max_frames must be >= 1")
    alpha = max(0.0, min(1.0, alpha))

    suffix = Path(upload_name).suffix.lower() or ".mp4"
    if suffix not in {".mp4", ".mov", ".avi", ".mkv"}:
        raise HTTPException(status_code=400, detail="Unsupported video format")

    import cv2

    input_path: Optional[str] = None
    cap = None
    writer = None
    t_total_start = time.perf_counter()
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp.write(await file.read())
            input_path = tmp.name

        cap = cv2.VideoCapture(input_path)
        if not cap.isOpened():
            raise HTTPException(status_code=400, detail="Failed to open uploaded video")

        fps = float(cap.get(cv2.CAP_PROP_FPS) or 15.0)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        if width <= 0 or height <= 0:
            raise HTTPException(status_code=400, detail="Invalid video dimensions")

        out_dir = Path("artifacts/video_segments")
        out_dir.mkdir(parents=True, exist_ok=True)
        out_stem = uuid.uuid4().hex
        out_name = f"{out_stem}.mp4"
        out_path = out_dir / out_name
        raw_out_path = out_dir / f"{out_stem}.raw.mp4"
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(str(raw_out_path), fourcc, fps, (width, height))
        if not writer.isOpened():
            raise HTTPException(status_code=500, detail="Failed to initialize output video writer")

        frame_idx = 0
        processed_frames = 0
        segmented_frames = 0
        per_frame_time_ms: list[float] = []
        tracking_failures = 0
        final_tracked_point: tuple[float, float] | None = None
        ultrasonic_prompt_points_by_frame: list[list[tuple[float, float]]] | None = None
        detection_guided_used = (mode == "dino_sam2_track")
        dino_prompt_used = dino_prompt or ""
        dino_detections_used = 0
        dino_detection_labels: list[str] = []
        t_segmentation_total_start = time.perf_counter()

        if mode == "grounded_sam2":
            import torch
            from sam2.build_sam import build_sam2_video_predictor_hf

            frames_bgr: list[np.ndarray] = []
            while True:
                ok, frame_bgr = cap.read()
                if not ok:
                    break
                frames_bgr.append(frame_bgr)
                if len(frames_bgr) >= max_frames:
                    break
            processed_frames = len(frames_bgr)
            if processed_frames == 0:
                raise HTTPException(status_code=400, detail="No frames found in input video")

            anchor_idx = int(click_frame_idx or 0)
            anchor_idx = max(0, min(processed_frames - 1, anchor_idx))

            def _pick_boxes(dets, max_n: int, pick_mode: str) -> list:
                if not dets:
                    return []
                if pick_mode == "largest":
                    ordered = sorted(
                        dets,
                        key=lambda d: max(0.0, d.box[2] - d.box[0]) * max(0.0, d.box[3] - d.box[1]),
                        reverse=True,
                    )
                else:
                    ordered = list(dets)
                return ordered[: max(1, int(max_n))]

            anchor_rgb = cv2.cvtColor(frames_bgr[anchor_idx], cv2.COLOR_BGR2RGB)
            anchor_img = Image.fromarray(anchor_rgb)
            ground_dets = detector.detect(anchor_img, text_prompt=grounded_ground_prompt, threshold=float(dino_threshold))
            other_dets = detector.detect(anchor_img, text_prompt=grounded_other_prompt, threshold=float(dino_threshold))

            ground_seed_dets = _pick_boxes(ground_dets, grounded_max_prompts, "largest")
            other_seed_dets = _pick_boxes(other_dets, grounded_max_prompts, "top")
            ground_boxes = [(float(d.box[0]), float(d.box[1]), float(d.box[2]), float(d.box[3])) for d in ground_seed_dets]
            other_boxes = [(float(d.box[0]), float(d.box[1]), float(d.box[2]), float(d.box[3])) for d in other_seed_dets]
            if not ground_boxes:
                ground_boxes = [(0.0, float(0.70 * height), float(width - 1), float(height - 1))]
            if not other_boxes:
                other_boxes = [(float(0.30 * width), float(0.25 * height), float(0.72 * width), float(0.92 * height))]

            dino_detection_labels = [
                *[f"ground:{d.label} ({d.score:.2f})" for d in ground_seed_dets],
                *[f"other:{d.label} ({d.score:.2f})" for d in other_seed_dets],
            ]
            dino_detections_used = len(ground_boxes) + len(other_boxes)
            dino_prompt_used = f"ground={grounded_ground_prompt} | other={grounded_other_prompt}"
            seed_boxes_for_debug: list[tuple[str, tuple[float, float, float, float], str]] = []
            for d in ground_seed_dets:
                b = (float(d.box[0]), float(d.box[1]), float(d.box[2]), float(d.box[3]))
                seed_boxes_for_debug.append(("ground", b, f"{d.label} {d.score:.2f}"))
            for d in other_seed_dets:
                b = (float(d.box[0]), float(d.box[1]), float(d.box[2]), float(d.box[3]))
                seed_boxes_for_debug.append(("other", b, f"{d.label} {d.score:.2f}"))

            tmp_frames_dir = Path(tempfile.mkdtemp(prefix="grounded_sam2_frames_"))
            try:
                for i, fbgr in enumerate(frames_bgr):
                    cv2.imwrite(str(tmp_frames_dir / f"{i:06d}.jpg"), fbgr, [int(cv2.IMWRITE_JPEG_QUALITY), 95])

                device = "cuda" if torch.cuda.is_available() else "cpu"
                predictor = build_sam2_video_predictor_hf(model_id=grounded_model_id, device=device)
                autocast_ctx = torch.autocast("cuda", dtype=torch.bfloat16) if torch.cuda.is_available() else nullcontext()
                with torch.inference_mode(), autocast_ctx:
                    state = predictor.init_state(video_path=str(tmp_frames_dir))
                    predictor.reset_state(state)
                    class_by_obj_id: dict[int, str] = {}
                    next_obj_id = 1
                    for gb in ground_boxes:
                        predictor.add_new_points_or_box(
                            inference_state=state,
                            frame_idx=anchor_idx,
                            obj_id=next_obj_id,
                            box=np.array([[gb[0], gb[1], gb[2], gb[3]]], dtype=np.float32),
                        )
                        class_by_obj_id[next_obj_id] = "ground"
                        next_obj_id += 1
                    for ob in other_boxes:
                        predictor.add_new_points_or_box(
                            inference_state=state,
                            frame_idx=anchor_idx,
                            obj_id=next_obj_id,
                            box=np.array([[ob[0], ob[1], ob[2], ob[3]]], dtype=np.float32),
                        )
                        class_by_obj_id[next_obj_id] = "other"
                        next_obj_id += 1
                    masks_by_frame: dict[int, dict[int, np.ndarray]] = {}
                    for out_frame_idx, out_obj_ids, out_mask_logits in predictor.propagate_in_video(state):
                        fdict: dict[int, np.ndarray] = {}
                        for i, oid in enumerate(out_obj_ids):
                            m = (out_mask_logits[i] > 0.0).detach().cpu().numpy()
                            if m.ndim == 3:
                                m = m[0]
                            fdict[int(oid)] = m.astype(bool)
                        masks_by_frame[int(out_frame_idx)] = fdict
            finally:
                shutil.rmtree(tmp_frames_dir, ignore_errors=True)

            for i, frame_bgr in enumerate(frames_bgr):
                t_frame_start = time.perf_counter()
                fmask = masks_by_frame.get(i, {})
                ground_union = np.zeros((height, width), dtype=np.uint8)
                other_union = np.zeros((height, width), dtype=np.uint8)
                for oid, m in fmask.items():
                    cls = class_by_obj_id.get(int(oid), "other")
                    if cls == "ground":
                        ground_union = cv2.bitwise_or(ground_union, m.astype(np.uint8) * 255)
                    else:
                        other_union = cv2.bitwise_or(other_union, m.astype(np.uint8) * 255)

                # Filled overlays are much easier to inspect than thin contours.
                if np.any(ground_union):
                    og = frame_bgr.copy()
                    og[ground_union > 0] = (0, 200, 0)
                    frame_bgr = cv2.addWeighted(og, 0.30, frame_bgr, 0.70, 0.0)
                    gc, _ = cv2.findContours(ground_union, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    cv2.drawContours(frame_bgr, gc, -1, (0, 255, 0), 2, cv2.LINE_AA)
                if np.any(other_union):
                    oo = frame_bgr.copy()
                    oo[other_union > 0] = (0, 80, 255)
                    frame_bgr = cv2.addWeighted(oo, 0.28, frame_bgr, 0.72, 0.0)
                    oc, _ = cv2.findContours(other_union, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    cv2.drawContours(frame_bgr, oc, -1, (0, 120, 255), 2, cv2.LINE_AA)

                if i == anchor_idx:
                    for cls, b, txt in seed_boxes_for_debug:
                        x1, y1, x2, y2 = [int(v) for v in b]
                        color = (0, 255, 0) if cls == "ground" else (0, 120, 255)
                        cv2.rectangle(frame_bgr, (x1, y1), (x2, y2), color, 2, cv2.LINE_AA)
                        cv2.putText(frame_bgr, txt, (x1, max(18, y1 - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 3, cv2.LINE_AA)
                        cv2.putText(frame_bgr, txt, (x1, max(18, y1 - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA)
                cv2.putText(frame_bgr, "Grounded-SAM2 | green=ground, red=other", (12, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 0), 4, cv2.LINE_AA)
                cv2.putText(frame_bgr, "Grounded-SAM2 | green=ground, red=other", (12, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2, cv2.LINE_AA)
                writer.write(frame_bgr)
                per_frame_time_ms.append(round((time.perf_counter() - t_frame_start) * 1000.0, 3))
                segmented_frames += 1
            first_other = other_boxes[0]
            final_tracked_point = (float((first_other[0] + first_other[2]) * 0.5 / max(1.0, width - 1)), float((first_other[1] + first_other[3]) * 0.5 / max(1.0, height - 1)))
            detection_guided_used = True
            tracking_failures = max(0, processed_frames - segmented_frames)
        elif mode in {"sam2_track", "dino_sam2_track"}:
            frames_bgr: list[np.ndarray] = []
            while True:
                ok, frame_bgr = cap.read()
                if not ok:
                    break
                frames_bgr.append(frame_bgr)
                if len(frames_bgr) >= max_frames:
                    break

            processed_frames = len(frames_bgr)
            if processed_frames == 0:
                raise HTTPException(status_code=400, detail="No frames found in input video")

            anchor_idx = int(click_frame_idx or 0)
            anchor_idx = max(0, min(processed_frames - 1, anchor_idx))
            if use_ultrasonic_prompt and trace_path is not None:
                ultrasonic_prompt_points_by_frame = _build_ultrasonic_prompt_points_by_frame(trace_path, processed_frames)
            if mode == "dino_sam2_track" and ultrasonic_prompt_points_by_frame is None:
                anchor_rgb = cv2.cvtColor(frames_bgr[anchor_idx], cv2.COLOR_BGR2RGB)
                anchor_img = Image.fromarray(anchor_rgb)
                detections = detector.detect(
                    anchor_img,
                    text_prompt=dino_prompt,
                    threshold=float(dino_threshold),
                )
                for d in detections[: int(dino_max_prompts)]:
                    x1, y1, x2, y2 = d.box
                    cx = max(0.0, min(float(width - 1), (x1 + x2) * 0.5))
                    cy = max(0.0, min(float(height - 1), (y1 + y2) * 0.5))
                    dino_detection_labels.append(f"{d.label} ({d.score:.2f})")
                    parsed_prompt_points.append((
                        cx / max(1.0, (width - 1)),
                        cy / max(1.0, (height - 1)),
                    ))
                # Keep manual click prompts and DINO prompts together, bounded for runtime stability.
                if len(parsed_prompt_points) > 12:
                    parsed_prompt_points = parsed_prompt_points[:12]
                dino_detections_used = len(dino_detection_labels)
                if dino_detections_used == 0:
                    raise HTTPException(status_code=400, detail="Grounding DINO found no detections on prompt frame")
            tracked_points_px: list[list[tuple[float, float] | None]] = []
            for pp in parsed_prompt_points:
                seq: list[tuple[float, float] | None] = [None] * processed_frames
                seq[anchor_idx] = (pp[0] * (width - 1), pp[1] * (height - 1))
                tracked_points_px.append(seq)
            grays = [cv2.cvtColor(f, cv2.COLOR_BGR2GRAY) for f in frames_bgr]

            if ultrasonic_prompt_points_by_frame is None:
                for seq in tracked_points_px:
                    # Track forward from anchor frame.
                    for i in range(anchor_idx + 1, processed_frames):
                        prev_pt = seq[i - 1]
                        if prev_pt is None:
                            tracking_failures += 1
                            continue
                        p0 = np.array([[prev_pt]], dtype=np.float32)
                        p1, st, _ = cv2.calcOpticalFlowPyrLK(
                            grays[i - 1],
                            grays[i],
                            p0,
                            None,
                            winSize=(21, 21),
                            maxLevel=3,
                            criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 20, 0.03),
                        )
                        if p1 is not None and st is not None and int(st[0][0]) == 1:
                            x = float(np.clip(p1[0][0][0], 0.0, width - 1.0))
                            y = float(np.clip(p1[0][0][1], 0.0, height - 1.0))
                            seq[i] = (x, y)
                        else:
                            tracking_failures += 1

                    # Track backward from anchor frame.
                    for i in range(anchor_idx - 1, -1, -1):
                        next_pt = seq[i + 1]
                        if next_pt is None:
                            tracking_failures += 1
                            continue
                        p0 = np.array([[next_pt]], dtype=np.float32)
                        p1, st, _ = cv2.calcOpticalFlowPyrLK(
                            grays[i + 1],
                            grays[i],
                            p0,
                            None,
                            winSize=(21, 21),
                            maxLevel=3,
                            criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 20, 0.03),
                        )
                        if p1 is not None and st is not None and int(st[0][0]) == 1:
                            x = float(np.clip(p1[0][0][0], 0.0, width - 1.0))
                            y = float(np.clip(p1[0][0][1], 0.0, height - 1.0))
                            seq[i] = (x, y)
                        else:
                            tracking_failures += 1

            for i, frame_bgr in enumerate(frames_bgr):
                if i % sample_every_n == 0:
                    t_frame_start = time.perf_counter()
                    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
                    current_prompt_points: list[tuple[float, float]] = []
                    if ultrasonic_prompt_points_by_frame is not None:
                        current_prompt_points = ultrasonic_prompt_points_by_frame[i]
                        if not current_prompt_points:
                            tracking_failures += 1
                    else:
                        for seq in tracked_points_px:
                            pt = seq[i] or seq[anchor_idx]
                            current_prompt_points.append((
                                float(pt[0] / max(1.0, (width - 1))),
                                float(pt[1] / max(1.0, (height - 1))),
                            ))
                    if current_prompt_points:
                        final_tracked_point = current_prompt_points[0]
                    overlay_rgb, _ = sam2_segmenter.segment_frame_rgb(
                        frame_rgb,
                        alpha=alpha,
                        profile=sam2_profile,
                        click_point=current_prompt_points[0] if current_prompt_points else None,
                        click_points=current_prompt_points,
                    )
                    frame_bgr = cv2.cvtColor(overlay_rgb, cv2.COLOR_RGB2BGR)
                    per_frame_time_ms.append(round((time.perf_counter() - t_frame_start) * 1000.0, 3))
                    segmented_frames += 1
                writer.write(frame_bgr)
        else:
            if mode == "sam2" and use_ultrasonic_prompt and trace_path is not None:
                ultrasonic_prompt_points_by_frame = _build_ultrasonic_prompt_points_by_frame(trace_path, max_frames)
            while True:
                ok, frame_bgr = cap.read()
                if not ok:
                    break
                if frame_idx % sample_every_n == 0:
                    t_frame_start = time.perf_counter()
                    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
                    if mode == "segformer":
                        overlay_rgb, _ = segmenter.segment_frame_rgb(frame_rgb, alpha=alpha)
                    else:
                        frame_prompt_points = parsed_prompt_points
                        if ultrasonic_prompt_points_by_frame is not None and frame_idx < len(ultrasonic_prompt_points_by_frame):
                            frame_prompt_points = ultrasonic_prompt_points_by_frame[frame_idx]
                        overlay_rgb, _ = sam2_segmenter.segment_frame_rgb(
                            frame_rgb,
                            alpha=alpha,
                            profile=sam2_profile,
                            click_point=frame_prompt_points[0] if frame_prompt_points else None,
                            click_points=frame_prompt_points,
                        )
                    frame_bgr = cv2.cvtColor(overlay_rgb, cv2.COLOR_RGB2BGR)
                    per_frame_time_ms.append(round((time.perf_counter() - t_frame_start) * 1000.0, 3))
                    segmented_frames += 1
                writer.write(frame_bgr)
                processed_frames += 1
                frame_idx += 1
                if processed_frames >= max_frames:
                    break
        segmentation_time_ms = (time.perf_counter() - t_segmentation_total_start) * 1000.0

        # Finalize the temporary mp4 before ffmpeg reads it.
        writer.release()
        writer = None

        t_encoding_start = time.perf_counter()
        ffmpeg_cmd = [
            "ffmpeg", "-y",
            "-i", str(raw_out_path),
            "-c:v", "libx264",
            "-pix_fmt", "yuv420p",
            "-movflags", "+faststart",
            str(out_path),
        ]
        ffmpeg_proc = subprocess.run(
            ffmpeg_cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=False,
        )
        if ffmpeg_proc.returncode != 0:
            raise HTTPException(status_code=500, detail="Failed to encode output video for browser playback")
        encoding_time_ms = (time.perf_counter() - t_encoding_start) * 1000.0
        total_time_ms = (time.perf_counter() - t_total_start) * 1000.0

        return VideoSegmentationResponse(
            mode_used=mode,
            sam2_profile_used=sam2_profile if mode in {"sam2", "sam2_track", "dino_sam2_track", "grounded_sam2"} else "n/a",
            click_prompt_used=bool(parsed_prompt_points and mode in {"sam2", "sam2_track", "dino_sam2_track", "grounded_sam2"}),
            click_point=(
                [parsed_prompt_points[0][0], parsed_prompt_points[0][1]]
                if (parsed_prompt_points and mode in {"sam2", "sam2_track", "dino_sam2_track", "grounded_sam2"})
                else None
            ),
            tracking_mode_used=(mode in {"sam2_track", "dino_sam2_track", "grounded_sam2"}),
            tracking_failures=tracking_failures if mode in {"sam2_track", "dino_sam2_track", "grounded_sam2"} else 0,
            final_tracked_point=(
                [final_tracked_point[0], final_tracked_point[1]]
                if (mode in {"sam2_track", "dino_sam2_track", "grounded_sam2"} and final_tracked_point is not None)
                else None
            ),
            detection_guided_used=detection_guided_used,
            dino_prompt_used=dino_prompt_used if detection_guided_used else None,
            dino_detections_used=dino_detections_used if detection_guided_used else 0,
            dino_detection_labels=dino_detection_labels if detection_guided_used else [],
            ultrasonic_prompt_used=use_ultrasonic_prompt and mode in {"sam2", "sam2_track", "dino_sam2_track", "grounded_sam2"},
            ultrasonic_trace_source=str(trace_path) if (use_ultrasonic_prompt and trace_path is not None) else None,
            uri=str(out_path).replace("\\", "/"),
            width=width,
            height=height,
            fps=round(fps, 2),
            total_frames=total_frames,
            processed_frames=processed_frames,
            segmented_frames=segmented_frames,
            total_time_ms=round(total_time_ms, 2),
            segmentation_time_ms=round(segmentation_time_ms, 2),
            encoding_time_ms=round(encoding_time_ms, 2),
            avg_frame_time_ms=round(sum(per_frame_time_ms) / len(per_frame_time_ms), 3) if per_frame_time_ms else 0.0,
            min_frame_time_ms=min(per_frame_time_ms) if per_frame_time_ms else 0.0,
            max_frame_time_ms=max(per_frame_time_ms) if per_frame_time_ms else 0.0,
            per_frame_time_ms=per_frame_time_ms,
        )
    finally:
        if cap is not None:
            cap.release()
        if writer is not None:
            writer.release()
        if input_path and os.path.exists(input_path):
            os.remove(input_path)
        # Remove temporary raw output if present.
        if "raw_out_path" in locals() and raw_out_path.exists():
            raw_out_path.unlink(missing_ok=True)


def _metrics_to_response(m) -> QualityMetricsResponse:
    return QualityMetricsResponse(
        blur_score=m.blur_score, blur_label=m.blur_label,
        exposure_score=m.exposure_score, exposure_label=m.exposure_label,
        contrast_score=m.contrast_score, contrast_label=m.contrast_label,
        clip_quality=m.clip_quality, clip_label=m.clip_label,
        overall_score=m.overall_score, overall_label=m.overall_label,
        issues=m.issues,
    )


@router.post("/quality", response_model=QualityMetricsResponse)
async def assess_quality(
    file: UploadFile = File(...),
) -> QualityMetricsResponse:
    from app.main import services

    quality_svc = services.get("quality")
    if quality_svc is None:
        raise HTTPException(status_code=503, detail="Quality service not loaded")

    try:
        content = await file.read()
        img = Image.open(io.BytesIO(content)).convert("RGB")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to read image: {e}")

    metrics = quality_svc.assess(img)
    return _metrics_to_response(metrics)


@router.get("/quality/worst", response_model=QualityWorstResponse)
async def get_worst_quality(
    n: int = 20,
) -> QualityWorstResponse:
    from app.main import services

    quality_svc = services.get("quality")
    if quality_svc is None:
        raise HTTPException(status_code=503, detail="Quality service not loaded")

    image_dir = services.get("image_dir", "data/use_cases")
    results = quality_svc.scan_directory(image_dir, limit=n)

    return QualityWorstResponse(
        total_scanned=len(list(__import__("pathlib").Path(image_dir).rglob("*.png"))) + len(list(__import__("pathlib").Path(image_dir).rglob("*.jpg"))),
        returned=len(results),
        items=[
            QualityItem(
                uri=r["uri"],
                filename=r["filename"],
                width=r["width"],
                height=r["height"],
                metrics=_metrics_to_response(r["metrics"]),
            )
            for r in results
        ],
    )