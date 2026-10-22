export interface SearchResultItem {
  image_id: string;
  uri: string;
  score: number;
  position: number;
  use_case?: string;
  label?: string;
}

export interface SearchResponse {
  search_id: string;
  model_version: string;
  preprocess_version: string;
  index_version: string;
  top_k: number;
  results: SearchResultItem[];
}

export interface SegmentInfo {
  class_id: number;
  label: string;
  pixel_count: number;
  percentage: number;
  color: [number, number, number];
}

export interface SegmentationResponse {
  segments: SegmentInfo[];
  width: number;
  height: number;
  colored_mask_b64: string;
}

export interface DetectionItem {
  label: string;
  score: number;
  box: [number, number, number, number]; // [x1, y1, x2, y2]
}

export interface DetectionResponse {
  detections: DetectionItem[];
  width: number;
  height: number;
}

export interface TagScores {
  label: string;
  confidence: number;
  scores: Record<string, number>;
}

export interface TagResponse {
  tags: Record<string, TagScores>;
  flat_labels: Record<string, string>;
}

export interface QualityMetrics {
  blur_score: number;
  blur_label: string;
  exposure_score: number;
  exposure_label: string;
  contrast_score: number;
  contrast_label: string;
  clip_quality: number;
  clip_label: string;
  overall_score: number;
  overall_label: string;
  issues: string[];
}

export interface QualityItem {
  uri: string;
  filename: string;
  width: number;
  height: number;
  metrics: QualityMetrics;
}

export interface QualityWorstResponse {
  total_scanned: number;
  returned: number;
  items: QualityItem[];
}

export interface MetricAtK {
  k: number;
  precision: number;
  recall: number;
  ndcg: number;
  relevant_count: number;
  total: number;
}

export interface EvalResultItem {
  image_id: string;
  uri: string;
  score: number;
  position: number;
  use_case?: string;
  gt_tags: string[];
  is_relevant: boolean;
  source?: string;
}

export interface EvalSearchResponse {
  query: string;
  relevant_tag: string;
  total_relevant_in_db: number;
  results_count: number;
  metrics: MetricAtK[];
  results: EvalResultItem[];
}

export interface HybridTagItem {
  group: string;
  label: string;
  confidence: number;
  source: 'clip' | 'detection' | 'segmentation';
  uncertain: boolean;
  details?: Record<string, unknown>;
}

export interface HybridTagResponse {
  scene_tags: HybridTagItem[];
  object_tags: HybridTagItem[];
  layout_tags: HybridTagItem[];
  summary: Record<string, string>;
}
