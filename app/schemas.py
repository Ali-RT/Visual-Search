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
    use_case: Optional[str] = None
    label: Optional[str] = None


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


class TextSearchRequest(BaseModel):
    query: str
    top_k: int = 50
    use_case: Optional[str] = None


class SimilarSearchRequest(BaseModel):
    image_id: str
    top_k: int = 50
    use_case: Optional[str] = None


class SegmentInfoItem(BaseModel):
    class_id: int
    label: str
    pixel_count: int
    percentage: float
    color: List[int]


class SegmentationResponse(BaseModel):
    segments: List[SegmentInfoItem]
    width: int
    height: int
    colored_mask_b64: str


class VideoSegmentationResponse(BaseModel):
    mode_used: str
    sam2_profile_used: str
    click_prompt_used: bool
    click_point: Optional[List[float]] = None
    tracking_mode_used: bool
    tracking_failures: int
    final_tracked_point: Optional[List[float]] = None
    detection_guided_used: bool
    dino_prompt_used: Optional[str] = None
    dino_detections_used: int
    dino_detection_labels: List[str] = []
    ultrasonic_prompt_used: bool
    ultrasonic_trace_source: Optional[str] = None
    uri: str
    width: int
    height: int
    fps: float
    total_frames: int
    processed_frames: int
    segmented_frames: int
    total_time_ms: float
    segmentation_time_ms: float
    encoding_time_ms: float
    avg_frame_time_ms: float
    min_frame_time_ms: float
    max_frame_time_ms: float
    per_frame_time_ms: List[float]


class DetectionItem(BaseModel):
    label: str
    score: float
    box: List[float]  # [x1, y1, x2, y2]


class DetectionResponse(BaseModel):
    detections: List[DetectionItem]
    width: int
    height: int


class TagScores(BaseModel):
    label: str
    confidence: float
    scores: dict


class TagResponse(BaseModel):
    tags: dict
    flat_labels: dict


class QualityMetricsResponse(BaseModel):
    blur_score: float
    blur_label: str
    exposure_score: float
    exposure_label: str
    contrast_score: float
    contrast_label: str
    clip_quality: float
    clip_label: str
    overall_score: float
    overall_label: str
    issues: List[str]


class QualityItem(BaseModel):
    uri: str
    filename: str
    width: int
    height: int
    metrics: QualityMetricsResponse


class QualityWorstResponse(BaseModel):
    total_scanned: int
    returned: int
    items: List[QualityItem]


class EvalSearchRequest(BaseModel):
    query: str
    relevant_tag: str  # gt_tag to check, e.g. "crosswalk", "stopline", "cones"
    top_k: int = 50
    k_values: List[int] = [5, 10, 20, 50]


class MetricAtK(BaseModel):
    k: int
    precision: float
    recall: float
    ndcg: float
    relevant_count: int
    total: int


class EvalResultItem(BaseModel):
    image_id: str
    uri: str
    score: float
    position: int
    use_case: Optional[str] = None
    gt_tags: List[str] = []
    is_relevant: bool = False
    source: Optional[str] = None


class EvalSearchResponse(BaseModel):
    query: str
    relevant_tag: str
    total_relevant_in_db: int
    results_count: int
    metrics: List[MetricAtK]
    results: List[EvalResultItem]


class HybridTagItem(BaseModel):
    group: str
    label: str
    confidence: float
    source: str  # "clip", "detection", "segmentation"
    uncertain: bool = False
    details: Optional[dict] = None


class HybridTagResponse(BaseModel):
    scene_tags: List[HybridTagItem]
    object_tags: List[HybridTagItem]
    layout_tags: List[HybridTagItem]
    summary: dict
