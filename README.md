![Alt Text](./assets/visual_search.png)

# Visual Search System — ADAS Image Retrieval

A visual search system for ADAS (Advanced Driver Assistance Systems) images that supports image-to-image search, text-to-image search, hybrid search, object detection, semantic segmentation, image tagging, quality assessment, and short video segmentation.

## Features

| Feature | Description |
|---------|-------------|
| **Visual Search** | Upload an image to find visually similar images (with optional crop/ROI) |
| **Text Search** | Find images matching natural language descriptions |
| **Hybrid Search** | Combine image + text queries with adjustable weighting |
| **Reverse Search** | Click any result to "Find Similar" using its stored embedding |
| **Image Tagging** | Zero-shot scene classification (CLIP) or hybrid 3-stage analysis |
| **Object Detection** | Open-vocabulary detection with Grounding DINO |
| **Segmentation** | Per-pixel semantic segmentation with SegFormer |
| **Segmentation-Guided Detection** | Two-stage: segment first, then detect only found classes |
| **Video Segmentation** | Segment short video clips frame-by-frame and export browser-ready MP4 |
| **Grounded-SAM2** | Grounding DINO seeded prompts + SAM2 tracking for ground/other video masks |
| **Side-by-Side Comparison** | Select and compare multiple results |
| **Quality Assessment** | Find low-quality images (blur, exposure, contrast, CLIP-IQA) |
| **Evaluation** | Precision@K, Recall@K, nDCG using label-derived ground truth |

## Tech Stack

| Layer | Technology |
|-------|-----------|
| Embeddings | OpenCLIP ViT-B-32 (laion2b, 512-dim) |
| Tagging | SigLIP2 ViT-SO400M-14 |
| Object Detection | Grounding DINO Tiny |
| Segmentation | SegFormer-B2 Cityscapes |
| Vector Store | Qdrant |
| Backend | FastAPI + Uvicorn |
| Frontend | React 18 + TypeScript + Vite + Tailwind CSS |
| Database | SQLite (metadata + interactions) |
| Dependencies | uv (Python), npm (frontend) |

---

## Quick Start

```bash
# 1. Install dependencies
uv sync
cd frontend && npm install && cd ..

# 2. Start Qdrant
docker run -d --name qdrant -p 6333:6333 qdrant/qdrant:latest

# 3. Ingest images + embed + upsert to Qdrant
uv run python scripts/ingest_usecase_images.py --config configs/adas_usecase.yaml
uv run python scripts/upsert_to_qdrant.py --config configs/adas_usecase.yaml --labels data/new_images

# 4. Start backend
uv run uvicorn app.main:app --host 0.0.0.0 --port 8000

# 5. Start frontend
cd frontend && npm run dev
# → http://localhost:3000
```

---

## Repository Layout

```
visual-search/
  app/
    main.py                     # FastAPI app, startup, service wiring
    api.py                      # All API endpoints
    config.py                   # YAML config loader
    embedding_service.py        # OpenCLIP image & text embeddings
    qdrant_index_service.py     # Qdrant vector search
    tagging_service.py          # SigLIP2 zero-shot tagging
    detection_service.py        # Grounding DINO object detection
    segmentation_service.py     # SegFormer semantic segmentation
    quality_service.py          # Image quality assessment
    interaction_store.py        # Click/impression logging
    schemas.py                  # Pydantic request/response models
    utils.py                    # Shared utilities
  scripts/
    ingest_usecase_images.py    # Register images in SQLite
    upsert_to_qdrant.py         # Embed + upsert to Qdrant
    parse_labels.py             # Parse ADAS label JSONs for ground truth
    tag_images.py               # Batch CLIP zero-shot tagging
    segment_video.py            # Segment one short video to MP4
    grounded_sam2_video.py      # Grounded-SAM2 offline video pipeline
    organize_video_segmentation_project.py # Organize traces/outputs into project layout
    generate_video_sequences.py # Build sequence clips from image frames
  projects/
    video-segmentation/
      inputs/
        traces/                 # Organized trace/video inputs (e.g. sequence_1866, sequence_1986)
        metadata/               # Trace metadata (xlsx/json)
      outputs/
        video-segments/         # Segmentation output videos
        grounded-sam2/          # Grounded-SAM2 run folders
        uss-sam2/               # USS-triggered SAM2 run folders
  configs/
    adas_usecase.yaml           # Main config (Qdrant)
  data/
    use_cases/                  # Flattened images served by backend
      crosswalk/                #   212 images
      stopline/                 #   172 images
      cones/                    #    22 images
    new_images/                 # Source images with label JSONs
      crosswalk/                #   (nested: <subfolder>/image+label)
      stopline/
      cones/
  artifacts/
    metadata/                   # SQLite database
  frontend/
    src/
      pages/                    # SearchPage, TextSearchPage, TagPage,
                                # VideoSegmentPage, QualityPage, EvaluatePage
      components/               # ImageDropzone, CropSelector, ResultsGrid,
                                # CompareView, DetectionOverlay, etc.
      api.ts                    # API client
      types.ts                  # TypeScript types
  docker-compose.yml            # Qdrant + search-api
  pyproject.toml                # Python dependencies (uv)
```

---

## API Endpoints

| Method | Path | Purpose |
|--------|------|---------|
| POST | `/search` | Image-to-image search |
| POST | `/search/text` | Text-to-image search |
| POST | `/search/hybrid` | Combined image + text search |
| POST | `/search/similar` | Reverse search by image ID |
| POST | `/search/eval` | Evaluation (P@K, R@K, nDCG) |
| POST | `/tag` | CLIP zero-shot tagging |
| POST | `/tag/hybrid` | 3-stage hybrid tagging |
| POST | `/detect` | Object detection |
| POST | `/segment` | Semantic segmentation |
| POST | `/segment/video` | Video segmentation + timing metrics |
| POST | `/quality` | Single image quality |
| GET | `/quality/worst?n=20` | Worst N quality images |
| POST | `/click` | Record user interaction |

---

## Data Pipeline

```
data/use_cases/<use_case>/*.png     (flat images served by backend)
        │
        ▼
ingest_usecase_images.py ──→ SQLite (metadata)
        │
        ▼
upsert_to_qdrant.py ──→ Qdrant (embeddings + payload)
   ├── OpenCLIP embed each image
   └── Enrich with gt_tags from data/new_images/ labels (optional)
```

### Adding new images

```bash
# 1. Add images to data/use_cases/<use_case>/
mkdir -p data/use_cases/new_usecase
cp /path/to/images/* data/use_cases/new_usecase/

# 2. Ingest + embed (incremental, no restart needed)
uv run python scripts/ingest_usecase_images.py --config configs/adas_usecase.yaml --use_cases new_usecase
uv run python scripts/upsert_to_qdrant.py --config configs/adas_usecase.yaml --use_cases new_usecase
```

### Video workflow (WoodScape-style)

```bash
# 1) Generate 10s clips from extracted frame folders
uv run python scripts/generate_video_sequences.py \
  --input_dir data/use_cases/woodscape_sequence_fv_frames \
  --output_dir data/use_cases/generated_sequences_fv \
  --camera_filter FV \
  --max_videos 5 \
  --min_frames 20 \
  --max_frames_per_video 120 \
  --max_gap 5 \
  --fps 12

# 2) Segment one clip offline
uv run python scripts/segment_video.py \
  --input data/use_cases/generated_sequences_fv/FV_000000_000027.mp4 \
  --output artifacts/video_segments/FV_000000_000027_segmented.mp4 \
  --max_frames 120
```

`POST /segment/video` returns timing fields for profiling:
`total_time_ms`, `segmentation_time_ms`, `encoding_time_ms`,
`avg_frame_time_ms`, `min_frame_time_ms`, `max_frame_time_ms`, and `per_frame_time_ms`.

### Organize traces and outputs

```bash
PYTHONPATH=. uv run python scripts/organize_video_segmentation_project.py \
  --repo . \
  --action move \
  --create-legacy-symlinks
```

---

## References

- [OpenCLIP](https://github.com/mlfoundations/open_clip)
- [Qdrant](https://qdrant.tech/documentation/)
- [Grounding DINO](https://github.com/IDEA-Research/GroundingDINO)
- [SegFormer](https://huggingface.co/nvidia/segformer-b2-finetuned-cityscapes-1024-1024)
- [SigLIP2](https://huggingface.co/google/siglip2-so400m-patch14-384)

See [COMMANDS.md](COMMANDS.md) for full command reference and feature documentation.
