# Visual Search — Commands & Processes

Complete reference for setup, data pipeline, serving, all features, and development.

---

## Prerequisites

| Tool | Install |
|------|---------|
| Python 3.10+ | `brew install python@3.11` |
| uv | `brew install uv` |
| Docker | [Docker Desktop](https://www.docker.com/products/docker-desktop/) |
| Node.js 18+ | `brew install node` |

---

## 1. Initial Setup

### Install Python dependencies

```bash
uv sync
```

### Install frontend dependencies

```bash
cd frontend && npm install
```

### Start Qdrant (Docker)

```bash
docker run -d --name qdrant -p 6333:6333 -p 6334:6334 qdrant/qdrant:latest
```

Verify:

```bash
curl http://localhost:6333/healthz
# → healthz check passed
```

---

## 2. Data Pipeline — ADAS Use Cases (Qdrant)

### Step 1: Organize images

Place images in the use-case directory structure:

```
data/use_cases/
├── crosswalk/
│   ├── image_001.png
│   └── image_002.png
├── stopline/
│   └── ...
└── cones/
    └── ...
```

### Step 2: Ingest into SQLite

Registers images with `use_case` and `label` metadata.

```bash
uv run python scripts/ingest_usecase_images.py --config configs/adas_usecase.yaml
```

Options:

```bash
# Ingest only specific use cases
uv run python scripts/ingest_usecase_images.py --config configs/adas_usecase.yaml --use_cases crosswalk stopline
```

### Step 3: Embed + upsert to Qdrant

Computes OpenCLIP embeddings and upserts vectors + metadata to Qdrant.

```bash
uv run python scripts/upsert_to_qdrant.py --config configs/adas_usecase.yaml
```

Options:

```bash
# Only embed specific use cases
uv run python scripts/upsert_to_qdrant.py --config configs/adas_usecase.yaml --use_cases crosswalk

# Custom batch size
uv run python scripts/upsert_to_qdrant.py --config configs/adas_usecase.yaml --batch_size 50
```

### Step 4: Tag images (CLIP zero-shot)

Classifies images by time_of_day, scene_type, crosswalk, pedestrian, vehicle_density.

```bash
uv run python scripts/tag_images.py --config configs/adas_usecase.yaml
```

### Step 3b: Embed + upsert with ground truth labels

If you have label JSON files alongside images (e.g. in `data/new_images/`), pass `--labels` to enrich Qdrant payloads with ground truth tags for evaluation.

```bash
uv run python scripts/upsert_to_qdrant.py --config configs/adas_usecase.yaml --labels data/new_images
```

### Full pipeline (copy/paste)

```bash
docker run -d --name qdrant -p 6333:6333 qdrant/qdrant:latest
uv sync
uv run python scripts/ingest_usecase_images.py --config configs/adas_usecase.yaml
uv run python scripts/upsert_to_qdrant.py --config configs/adas_usecase.yaml --labels data/new_images
uv run python scripts/tag_images.py --config configs/adas_usecase.yaml
```

---

## 4. Serving

### Start backend API

```bash
# Default
uv run uvicorn app.main:app --host 0.0.0.0 --port 8000

# With auto-reload (development)
uv run uvicorn app.main:app --reload --port 8000
```

### Start React frontend

```bash
cd frontend && npm run dev
# → http://localhost:3000
```

### Docker Compose (Qdrant + API together)

```bash
docker compose up -d
# Qdrant → :6333
# API    → :8000
```

---

## 5. Features & API Reference

### 5.1 Visual Search (Image-to-Image)

Upload an image and find visually similar images in the dataset.

**Endpoint:** `POST /search`

```bash
# Basic search
curl -X POST http://localhost:8000/search -F "file=@query.png"

# With parameters
curl -X POST "http://localhost:8000/search?top_k=20&use_case=crosswalk" -F "file=@query.png"

# With crop (Region of Interest)
curl -X POST "http://localhost:8000/search?crop_x=100&crop_y=50&crop_w=300&crop_h=200" -F "file=@query.png"
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `file` | File | required | Query image |
| `top_k` | int | 50 | Number of results |
| `top_k_candidates` | int | 200 | ANN candidates |
| `use_case` | string | — | Filter by use case |
| `crop_x/y/w/h` | int | — | Crop region in pixels |

**Frontend:** Visual Search tab — drag-and-drop image, optional crop selection, text refinement.

### 5.2 Text-to-Image Search

Find images matching a natural language description.

**Endpoint:** `POST /search/text`

```bash
curl -X POST http://localhost:8000/search/text \
  -H "Content-Type: application/json" \
  -d '{"query": "a road with a crosswalk at night", "top_k": 20}'
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `query` | string | required | Text description |
| `top_k` | int | 50 | Number of results |
| `use_case` | string | — | Filter by use case |

**Frontend:** Text Search tab — textarea with example query chips.

### 5.3 Hybrid Search (Image + Text)

Combine image similarity with text refinement using a weighted blend.

**Endpoint:** `POST /search/hybrid`

```bash
curl -X POST "http://localhost:8000/search/hybrid?text=at%20night&image_weight=0.7&top_k=20" \
  -F "file=@query.png"
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `file` | File | required | Query image |
| `text` | string | "" | Text refinement |
| `image_weight` | float | 0.7 | Image vs text weight (0.0–1.0) |
| `top_k` | int | 50 | Number of results |
| `use_case` | string | — | Filter by use case |
| `crop_x/y/w/h` | int | — | Crop region |

**Frontend:** Visual Search tab — add text in the refinement field, adjust Image/Text slider.

### 5.4 Reverse Search (Find Similar)

Use a result image's stored embedding as a new query without re-uploading.

**Endpoint:** `POST /search/similar`

```bash
curl -X POST http://localhost:8000/search/similar \
  -H "Content-Type: application/json" \
  -d '{"image_id": "01411e0137b...", "top_k": 20}'
```

**Frontend:** Hover any result → click the search icon, or click a result → "Find Similar" button.

### 5.5 Image Tagging — CLIP Zero-Shot

Classify images using SigLIP2 zero-shot classification.

**Endpoint:** `POST /tag`

```bash
curl -X POST http://localhost:8000/tag -F "file=@photo.png"
```

**Tag groups:** time_of_day (day/night/dawn_dusk), scene_type (urban/rural/highway/suburban), crosswalk (present/absent), pedestrian (present/absent), vehicle_density (low/moderate/heavy).

**Model:** `ViT-SO400M-14-SigLIP2-378` (webli) — 1.1B parameter model, independent from the search embedding model.

### 5.6 Hybrid Tagging (CLIP + Detection + Segmentation)

Three-stage analysis that fuses CLIP scene understanding, Grounding DINO object detection, and SegFormer scene layout into a unified tag set.

**Endpoint:** `POST /tag/hybrid`

```bash
curl -X POST http://localhost:8000/tag/hybrid -F "file=@photo.png"
```

**Response structure:**
- `scene_tags` — CLIP-derived: time_of_day, scene_type (with confidence scores)
- `object_tags` — Detection-derived: vehicle_density (with count), pedestrian, crosswalk, traffic_sign, traffic_light, lane_marking (with confidence + count)
- `layout_tags` — Segmentation-derived: road_coverage, vegetation_coverage, sky_visibility, built_environment (with percentages)
- `summary` — flat label map

**Confidence-aware:** Tags below 55% are flagged as `uncertain`. Detection-derived tags have 90-95% confidence since they're based on actual object detections.

**Frontend:** Image Tagging tab → Hybrid/CLIP Only toggle. Hybrid mode shows color-coded cards by source (purple=CLIP, amber=Detection, teal=Segmentation).

### 5.7 Object Detection (Grounding DINO)

Open-vocabulary object detection using Grounding DINO.

**Endpoint:** `POST /detect`

```bash
# Default ADAS prompt
curl -X POST http://localhost:8000/detect -F "file=@photo.png"

# Custom prompt
curl -X POST "http://localhost:8000/detect?prompt=car.%20bicycle.%20stop%20sign.&threshold=0.25" \
  -F "file=@photo.png"
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `file` | File | required | Image to analyze |
| `prompt` | string | ADAS default | Detection classes (period-separated) |
| `threshold` | float | 0.3 | Detection confidence threshold |

**Default prompt:** `vehicle. pedestrian. traffic sign. traffic light. crosswalk. lane marking. bicycle. motorcycle. bus. truck.`

**Model:** `IDEA-Research/grounding-dino-tiny`

**Frontend:** Click a result → Segment/Detect buttons in the detail panel. Bounding boxes with labels and confidence scores overlay the image.

### 5.8 Semantic Segmentation (SegFormer)

Per-pixel scene segmentation trained on Cityscapes (19 driving scene classes).

**Endpoint:** `POST /segment`

```bash
curl -X POST http://localhost:8000/segment -F "file=@photo.png"
```

**Response:** Color-coded mask overlay (base64 JPEG), per-class pixel percentages.

**Cityscapes classes:** road, sidewalk, building, wall, fence, pole, traffic light, traffic sign, vegetation, terrain, sky, person, rider, car, truck, bus, train, motorcycle, bicycle.

**Model:** `nvidia/segformer-b2-finetuned-cityscapes-1024-1024`

### 5.9 Video Segmentation (Short Clips)

Segment a short video clip frame-by-frame and export a browser-compatible MP4.

**Endpoint:** `POST /segment/video`

```bash
curl -X POST "http://localhost:8000/segment/video?alpha=0.45&sample_every_n=1&max_frames=120" \
  -F "file=@data/use_cases/woodscape_clips_10s/woodscape_fv_clip_001_000000_000356.mp4"
```

**Response highlights:**
- Output video URI: `uri`
- Throughput stats: `processed_frames`, `segmented_frames`
- Timing stats: `total_time_ms`, `segmentation_time_ms`, `encoding_time_ms`
- Per-frame timing: `avg_frame_time_ms`, `min_frame_time_ms`, `max_frame_time_ms`, `per_frame_time_ms`

**Frontend:** Video Segment tab — upload clip, run segmentation, preview/download output, inspect timing metrics.

#### Grounded-SAM2 mode (`/segment/video`)

Use Grounding DINO on a seed frame to initialize SAM2 tracking.

```bash
curl -X POST "http://localhost:8000/segment/video?mode=grounded_sam2&click_frame_idx=152&dino_threshold=0.25&grounded_ground_prompt=street%20ground.%20road.%20asphalt.%20floor.&grounded_other_prompt=car.%20curb.%20wall.%20obstacle.%20pedestrian.&grounded_model_id=facebook/sam2.1-hiera-small" \
  -F "file=@traces/20231107_122835_PrkDemo_Mach-E_EPC6900_vehicle_standard_backward_centered_1866.avi.mp4"
```

| Parameter | Type | Description |
|-----------|------|-------------|
| `mode` | string | `grounded_sam2` |
| `click_frame_idx` | int | Seed frame index for DINO prompts |
| `dino_threshold` | float | DINO confidence threshold |
| `grounded_ground_prompt` | string | Ground class text prompt |
| `grounded_other_prompt` | string | Other/obstacle class text prompt |
| `grounded_model_id` | string | SAM2 HF model ID (e.g., `facebook/sam2.1-hiera-small`) |

### 5.10 Segmentation-Guided Detection (Two-Stage Pipeline)

Runs segmentation first, then uses detected object classes to build a targeted detection prompt.

**How it works:**
1. SegFormer identifies scene classes (road 35%, car 8%, person 2%, etc.)
2. Object classes (car, person, truck, bus, bicycle, traffic light, etc.) are extracted; background classes (road, sky, vegetation) are ignored
3. Grounding DINO runs with a prompt built from only the segmented object classes (e.g., `car. person.`)

**Frontend:** Click "1. Segment" → then "2. Detect (guided)". Segmentation chips with ⬆ markers show which classes guide detection. A green info bar confirms the guided prompt.

### 5.11 Side-by-Side Comparison

Select multiple result images and compare them in a full-screen panel.

**How to use:**
1. Hover a result → click the checkbox (top-left corner) to add to compare (up to 6)
2. Click the green "Compare (N)" button above results
3. Full-screen comparison with metadata, segment, and detect buttons per card
4. Collapse/expand or remove individual cards

**Frontend:** Available on both Visual Search and Text Search pages.

### 5.12 Image Quality Assessment

Scans all images using four quality metrics and returns the N lowest quality images.

**Endpoint (single image):** `POST /quality`

```bash
curl -X POST http://localhost:8000/quality -F "file=@photo.png"
```

**Endpoint (scan all, return worst N):** `GET /quality/worst`

```bash
# Return 20 worst quality images
curl "http://localhost:8000/quality/worst?n=20"
```

**Quality metrics:**

| Metric | Method | What it detects |
|--------|--------|----------------|
| Blur | Laplacian variance | Motion blur, defocus, camera shake |
| Exposure | Histogram analysis | Overexposed (sun glare), underexposed (too dark) |
| Contrast | RMS contrast | Fog, haze, dirty lens |
| CLIP-IQA | SigLIP2 zero-shot | Overall visual quality (sharp vs blurry/noisy) |

**Composite score:** 0–100 (blur 35%, CLIP-IQA 25%, exposure 25%, contrast 15%). Labels: good (70+), fair (45–70), poor (<45).

**Frontend:** Quality tab → set count → "Find Low Quality Images". Ranked list with per-metric bars, scores, issue descriptions, and click-to-detail view.

### 5.13 Crop / Region of Interest (ROI)

Draw a rectangle on the query image to search using only that region.

**Frontend:** Upload an image on Visual Search tab → click and drag to draw a crop rectangle. Dark overlay dims the excluded area. Search sends crop coordinates to the backend which crops before embedding.

---

## 6. Qdrant Management

### Check health

```bash
curl http://localhost:6333/healthz
```

### List collections

```bash
curl http://localhost:6333/collections | python -m json.tool
```

### Collection info (point count, status)

```bash
curl http://localhost:6333/collections/adas_images | python -m json.tool
```

### Delete a collection (start fresh)

```bash
curl -X DELETE http://localhost:6333/collections/adas_images
```

### Stop / restart Qdrant

```bash
docker stop qdrant && docker rm qdrant   # stop
docker start qdrant                       # restart (data persists with volume)
```

---

## 7. Adding New Images (Incremental)

### Add new use case

```bash
# 1. Create directory with images
mkdir -p data/use_cases/new_usecase
cp /path/to/images/* data/use_cases/new_usecase/

# 2. Ingest only the new use case
uv run python scripts/ingest_usecase_images.py --config configs/adas_usecase.yaml --use_cases new_usecase

# 3. Embed + upsert only the new use case
uv run python scripts/upsert_to_qdrant.py --config configs/adas_usecase.yaml --use_cases new_usecase

# 4. Tag the new images
uv run python scripts/tag_images.py --config configs/adas_usecase.yaml
```

No rebuild needed. No restart needed. New images are searchable immediately.

## 7b. WoodScape Video Sequence Workflow

Generate many short clips from frame folders (10 seconds max at 12 FPS):

```bash
uv run python scripts/generate_video_sequences.py \
  --input_dir data/use_cases/woodscape_sequence_fv_frames \
  --output_dir data/use_cases/woodscape_clips_10s \
  --camera_filter FV \
  --max_videos 999 \
  --min_frames 20 \
  --max_frames_per_video 120 \
  --max_gap 5 \
  --fps 12
```

Segment one generated clip offline:

```bash
uv run python scripts/segment_video.py \
  --input data/use_cases/woodscape_clips_10s/woodscape_fv_clip_001_000000_000356.mp4 \
  --output artifacts/video_segments/woodscape_fv_clip_001_000000_000356_segmented.mp4 \
  --max_frames 120
```

## 7c. Organize Video-Segmentation Project Assets

Move traces and generated outputs into a meaningful project layout:

```bash
PYTHONPATH=. uv run python scripts/organize_video_segmentation_project.py \
  --repo . \
  --action move \
  --create-legacy-symlinks
```

Target layout:

```
projects/video-segmentation/
  inputs/traces/
  inputs/metadata/
  outputs/video-segments/
  outputs/grounded-sam2/
  outputs/uss-sam2/
```

### Remove a use case from Qdrant

```python
from qdrant_client import QdrantClient
from qdrant_client.models import Filter, FieldCondition, MatchValue
client = QdrantClient(host="localhost", port=6333)
client.delete(collection_name="adas_images",
    points_selector=Filter(must=[FieldCondition(key="use_case", match=MatchValue(value="old_usecase"))]))
```

---

## 8. Record User Interactions

### Record a click

```bash
curl -X POST http://localhost:8000/click \
  -H "Content-Type: application/json" \
  -d '{"search_id":"uuid","query_sha256":"hex","image_id":"hex","position":1}'
```

### Compute CTR

```bash
uv run python scripts/compute_ctr.py --config configs/adas_usecase.yaml --k 50 --since_hours 24
```

---

## 9. Development

### Type check frontend

```bash
cd frontend && npx tsc --noEmit
```

### Build frontend for production

```bash
cd frontend && npm run build
# Output in frontend/dist/
```

### Check SQLite database

```bash
# List tables
sqlite3 artifacts/metadata/adas_usecase.sqlite ".tables"

# Image counts per use case
sqlite3 artifacts/metadata/adas_usecase.sqlite \
  "SELECT use_case, COUNT(*) FROM images GROUP BY use_case;"

# Tag distribution
sqlite3 artifacts/metadata/adas_usecase.sqlite \
  "SELECT flat_labels FROM image_tags LIMIT 5;"

# Interaction logs
sqlite3 artifacts/metadata/adas_usecase.sqlite \
  "SELECT event_type, COUNT(*) FROM interactions GROUP BY event_type;"
```

---

## 10. Architecture

### Backend Services

| Service | File | Model | Purpose |
|---------|------|-------|---------|
| EmbeddingService | `app/embedding_service.py` | OpenCLIP ViT-B-32 | Image & text embedding (512-dim) |
| TaggingService | `app/tagging_service.py` | SigLIP2 ViT-SO400M | Zero-shot scene classification |
| DetectionService | `app/detection_service.py` | Grounding DINO Tiny | Open-vocabulary object detection |
| SegmentationService | `app/segmentation_service.py` | SegFormer-B2 Cityscapes | Per-pixel semantic segmentation |
| QualityService | `app/quality_service.py` | Laplacian + CLIP-IQA | Image quality assessment |
| QdrantIndexService | `app/qdrant_index_service.py` | — | Vector search via Qdrant |
| InteractionStore | `app/interaction_store.py` | — | Click/impression logging to SQLite |

### Frontend Pages

| Page | File | Features |
|------|------|----------|
| Visual Search | `pages/SearchPage.tsx` | Image upload, crop ROI, hybrid search, compare |
| Text Search | `pages/TextSearchPage.tsx` | Text query, example chips, compare |
| Image Tagging | `pages/TagPage.tsx` | Hybrid/CLIP toggle, 3-stage analysis |
| Video Segment | `pages/VideoSegmentPage.tsx` | Upload short clips, run `/segment/video`, view timing stats |
| Quality | `pages/QualityPage.tsx` | Worst-N quality scan, 4-metric breakdown |
| Evaluate | `pages/EvaluatePage.tsx` | Precision@K, Recall@K, nDCG evaluation |

### Frontend Components

| Component | File | Purpose |
|-----------|------|---------|
| ImageDropzone | `components/ImageDropzone.tsx` | Drag-and-drop image upload |
| CropSelector | `components/CropSelector.tsx` | Draw ROI rectangle on image |
| ResultsGrid | `components/ResultsGrid.tsx` | Search results with compare checkboxes |
| CompareView | `components/CompareView.tsx` | Full-screen side-by-side comparison |
| DetectionOverlay | `components/DetectionOverlay.tsx` | Segment + detect two-stage overlay |
| HybridTagResults | `components/HybridTagResults.tsx` | 3-source tag visualization |
| TagResults | `components/TagResults.tsx` | CLIP-only tag bars |
| Navbar | `components/Navbar.tsx` | Tab navigation |

### API Endpoints

| Method | Path | Purpose |
|--------|------|---------|
| POST | `/search` | Image-to-image search |
| POST | `/search/text` | Text-to-image search |
| POST | `/search/hybrid` | Combined image + text search |
| POST | `/search/similar` | Reverse search by image ID |
| POST | `/tag` | CLIP zero-shot tagging |
| POST | `/tag/hybrid` | Three-stage hybrid tagging |
| POST | `/detect` | Object detection (Grounding DINO) |
| POST | `/segment` | Semantic segmentation (SegFormer) |
| POST | `/segment/video` | Video segmentation + timing metrics |
| POST | `/quality` | Single image quality assessment |
| GET | `/quality/worst?n=20` | Scan all images, return N worst |
| POST | `/search/eval` | Text search evaluation (P@K, R@K, nDCG) |
| POST | `/click` | Record user click |

### Tech Stack

| Layer | Technology |
|-------|-----------|
| Embeddings | OpenCLIP ViT-B-32 (laion2b) |
| Tagging | SigLIP2 ViT-SO400M-14 (webli) |
| Object Detection | Grounding DINO Tiny |
| Segmentation | SegFormer-B2 Cityscapes |
| Vector Store | Qdrant |
| Backend | FastAPI + Uvicorn |
| Frontend | React 18 + TypeScript + Vite + Tailwind CSS |
| Database | SQLite (metadata + interactions) |
| Dependency Mgmt | uv (Python), npm (frontend) |
| Containerization | Docker + Docker Compose |

---

## 11. Configuration Reference

### Key config fields (`configs/adas_usecase.yaml`)

| Field | Purpose |
|-------|---------|
| `data.image_dir` | Root directory for images |
| `data.sqlite_path` | SQLite database path |
| `model.backbone` | OpenCLIP model (e.g. `ViT-B-32`) |
| `model.device` | `auto`, `cpu`, or `cuda` |
| `vector_store` | `hnswlib` or `qdrant` |
| `qdrant.host` | Qdrant server host |
| `qdrant.port` | Qdrant server port |
| `qdrant.collection_name` | Qdrant collection name |
| `index.top_k` | Default results to return |
| `tagging.enabled` | Enable CLIP zero-shot tagging |
| `tagging.backbone` | Tagging model (independent from search) |
| `tagging.pretrained` | Tagging model weights |

### Environment variables

| Variable | Purpose | Default |
|----------|---------|---------|
| `CONFIG_PATH` | Path to YAML config | `configs/adas_usecase.yaml` |

---

## 12. Process Map

```
┌──────────────────────────────────────────────────────────────────┐
│                        DATA PIPELINE                              │
│                                                                   │
│  Images on disk    ──→  ingest_usecase_images.py                 │
│       │                       │                                   │
│       │                       ▼                                   │
│       │               SQLite (metadata)                           │
│       │                       │                                   │
│       ▼                       ▼                                   │
│  upsert_to_qdrant.py  (embed + store in Qdrant)                 │
│       │                                                           │
│       ▼                                                           │
│  tag_images.py  (CLIP zero-shot → SQLite image_tags)             │
└──────────────────────────────────────────────────────────────────┘

┌──────────────────────────────────────────────────────────────────┐
│                        SERVING                                    │
│                                                                   │
│  React Frontend (:3000) ──→ FastAPI (:8000) ──→ Qdrant (:6333)  │
│       │                         │                                 │
│       │  POST /search           │  vector search + payload filter │
│       │  POST /search/text      │                                 │
│       │  POST /search/hybrid    ▼                                 │
│       │  POST /search/similar   OpenCLIP (embed query)           │
│       │  POST /tag/hybrid       SigLIP2 (scene tags)             │
│       │  POST /detect           Grounding DINO (objects)          │
│       │  POST /segment          SegFormer (segmentation)          │
│       │  GET  /quality/worst    Laplacian+CLIP (quality)          │
│       │  POST /click            SQLite (interactions)             │
└──────────────────────────────────────────────────────────────────┘

┌──────────────────────────────────────────────────────────────────┐
│                   MODELS LOADED AT STARTUP                        │
│                                                                   │
│  ┌─────────────────┐  ┌──────────────────┐  ┌────────────────┐  │
│  │ OpenCLIP ViT-B32│  │ SigLIP2 SO400M   │  │ Grounding DINO │  │
│  │ (512-dim embed) │  │ (scene tagging)  │  │ (detection)    │  │
│  └─────────────────┘  └──────────────────┘  └────────────────┘  │
│  ┌─────────────────┐  ┌──────────────────┐                      │
│  │ SegFormer-B2    │  │ Quality Service  │                      │
│  │ (segmentation)  │  │ (Laplacian+CLIP) │                      │
│  └─────────────────┘  └──────────────────┘                      │
└──────────────────────────────────────────────────────────────────┘
```
