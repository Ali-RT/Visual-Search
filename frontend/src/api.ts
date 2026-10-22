import type {
  SearchResponse,
  TagResponse,
  HybridTagResponse,
  DetectionResponse,
  SegmentationResponse,
  QualityWorstResponse,
  EvalSearchResponse,
} from './types';

const BASE = '/api';

export interface CropParams {
  x: number;
  y: number;
  w: number;
  h: number;
}

export async function searchImages(
  file: File,
  topK = 50,
  topKCandidates = 200,
  useCase?: string,
  crop?: CropParams | null,
): Promise<SearchResponse> {
  const form = new FormData();
  form.append('file', file);

  const params = new URLSearchParams({
    top_k: String(topK),
    top_k_candidates: String(topKCandidates),
  });
  if (useCase) params.set('use_case', useCase);
  if (crop) {
    params.set('crop_x', String(crop.x));
    params.set('crop_y', String(crop.y));
    params.set('crop_w', String(crop.w));
    params.set('crop_h', String(crop.h));
  }

  const res = await fetch(`${BASE}/search?${params}`, { method: 'POST', body: form });
  if (!res.ok) throw new Error(`Search failed: ${res.status}`);
  return res.json();
}

export async function searchByText(
  query: string,
  topK = 50,
  useCase?: string,
): Promise<SearchResponse> {
  const body: Record<string, unknown> = { query, top_k: topK };
  if (useCase) body.use_case = useCase;

  const res = await fetch(`${BASE}/search/text`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(body),
  });
  if (!res.ok) throw new Error(`Text search failed: ${res.status}`);
  return res.json();
}

export async function searchHybrid(
  file: File,
  text: string,
  imageWeight = 0.7,
  topK = 50,
  useCase?: string,
  crop?: CropParams | null,
): Promise<SearchResponse> {
  const form = new FormData();
  form.append('file', file);

  const params = new URLSearchParams({
    text,
    image_weight: String(imageWeight),
    top_k: String(topK),
  });
  if (useCase) params.set('use_case', useCase);
  if (crop) {
    params.set('crop_x', String(crop.x));
    params.set('crop_y', String(crop.y));
    params.set('crop_w', String(crop.w));
    params.set('crop_h', String(crop.h));
  }

  const res = await fetch(`${BASE}/search/hybrid?${params}`, { method: 'POST', body: form });
  if (!res.ok) throw new Error(`Hybrid search failed: ${res.status}`);
  return res.json();
}

export async function searchSimilar(
  imageId: string,
  topK = 50,
  useCase?: string,
): Promise<SearchResponse> {
  const body: Record<string, unknown> = { image_id: imageId, top_k: topK };
  if (useCase) body.use_case = useCase;

  const res = await fetch(`${BASE}/search/similar`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(body),
  });
  if (!res.ok) throw new Error(`Similar search failed: ${res.status}`);
  return res.json();
}

export async function tagImage(file: File): Promise<TagResponse> {
  const form = new FormData();
  form.append('file', file);

  const res = await fetch(`${BASE}/tag`, { method: 'POST', body: form });
  if (!res.ok) throw new Error(`Tagging failed: ${res.status}`);
  return res.json();
}

export async function tagImageHybrid(file: File): Promise<HybridTagResponse> {
  const form = new FormData();
  form.append('file', file);

  const res = await fetch(`${BASE}/tag/hybrid`, { method: 'POST', body: form });
  if (!res.ok) throw new Error(`Hybrid tagging failed: ${res.status}`);
  return res.json();
}

export async function recordClick(
  searchId: string,
  querySha256: string,
  imageId: string,
  position: number,
): Promise<void> {
  await fetch(`${BASE}/click`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      search_id: searchId,
      query_sha256: querySha256,
      image_id: imageId,
      position,
    }),
  });
}

export async function detectObjects(
  imageUrl: string,
  prompt?: string,
  threshold = 0.3,
): Promise<DetectionResponse> {
  const blob = await fetch(imageUrl).then((r) => r.blob());
  const form = new FormData();
  form.append('file', blob, 'image.png');

  const params = new URLSearchParams({ threshold: String(threshold) });
  if (prompt) params.set('prompt', prompt);

  const res = await fetch(`${BASE}/detect?${params}`, { method: 'POST', body: form });
  if (!res.ok) throw new Error(`Detection failed: ${res.status}`);
  return res.json();
}

export async function segmentImage(imageUrl: string): Promise<SegmentationResponse> {
  const blob = await fetch(imageUrl).then((r) => r.blob());
  const form = new FormData();
  form.append('file', blob, 'image.png');

  const res = await fetch(`${BASE}/segment`, { method: 'POST', body: form });
  if (!res.ok) throw new Error(`Segmentation failed: ${res.status}`);
  return res.json();
}

export async function evalTextSearch(
  query: string,
  relevantTag: string,
  topK = 200,
  kValues = [5, 10, 20, 50, 100, 200],
): Promise<EvalSearchResponse> {
  const res = await fetch(`${BASE}/search/eval`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      query,
      relevant_tag: relevantTag,
      top_k: topK,
      k_values: kValues,
    }),
  });
  if (!res.ok) throw new Error(`Eval failed: ${res.status}`);
  return res.json();
}

export async function getWorstQuality(n = 20): Promise<QualityWorstResponse> {
  const res = await fetch(`${BASE}/quality/worst?n=${n}`);
  if (!res.ok) throw new Error(`Quality scan failed: ${res.status}`);
  return res.json();
}
