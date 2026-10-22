# Visual Search System — Demo Presentation Guide

A complete guide for setting up, running, and presenting the Visual Search demo for ADAS images.

---

## Part 1: Setup (Before the Demo)

### Prerequisites

| Tool | Install | Verify |
|------|---------|--------|
| Python 3.10+ | `brew install python@3.11` | `python3 --version` |
| uv (Python package manager) | `brew install uv` | `uv --version` |
| Docker Desktop | [docker.com/products/docker-desktop](https://www.docker.com/products/docker-desktop/) | `docker --version` |
| Node.js 18+ | `brew install node` | `node --version` |

### Step-by-step setup

```bash
# Clone the repo
git clone <repo-url> visual-search
cd visual-search

# 1. Install Python dependencies (fast — uv resolves in seconds)
uv sync

# 2. Install frontend dependencies
cd frontend && npm install && cd ..

# 3. Start Qdrant vector database (Docker container)
docker run -d --name qdrant \
  -p 6333:6333 -p 6334:6334 \
  -v qdrant_data:/qdrant/storage \
  qdrant/qdrant:latest

# Verify Qdrant is healthy
curl http://localhost:6333/healthz
# Expected: "healthz check passed"

# 4. Ingest images into SQLite metadata database
#    Scans data/use_cases/<use_case>/*.png and registers each image
uv run python scripts/ingest_usecase_images.py --config configs/adas_usecase.yaml

# 5. Compute embeddings and upsert to Qdrant
#    --labels flag enriches each image with ground truth tags from label JSON files
uv run python scripts/upsert_to_qdrant.py \
  --config configs/adas_usecase.yaml \
  --labels data/new_images

# 6. Start the backend API server
uv run uvicorn app.main:app --host 0.0.0.0 --port 8000

# 7. In a separate terminal, start the React frontend
cd frontend && npm run dev
# Frontend available at http://localhost:3000
```

### Verify everything is running

| Service | URL | Expected |
|---------|-----|----------|
| Qdrant | http://localhost:6333/healthz | "healthz check passed" |
| Backend API | http://localhost:8000/docs | Swagger UI loads |
| Frontend | http://localhost:3000 | Visual Search app loads |

### Dataset overview

| Use Case | Images | Description |
|----------|--------|-------------|
| crosswalk | 212 | Dashcam images containing pedestrian crosswalks / zebra crossings |
| stopline | 172 | Dashcam images containing stop lines at intersections |
| cones | 22 | Dashcam images containing traffic cones, barrels, or barriers |
| **Total** | **406** | Labeled ADAS images from real driving data |

Of these 406 images, **313 have ground truth labels** parsed from annotation JSON files. These labels contain detailed information about what objects appear in each image (crosswalk, stopline, lane_marking, road_edge, cones) and are used for quantitative evaluation.

---

## Part 2: Architecture Overview (Present First)

### What this system does

This is an **end-to-end visual search and analysis system** for ADAS (Advanced Driver Assistance Systems) dashcam images. It lets engineers:

- **Search** for similar images using an image query, a text query, or both
- **Tag** images automatically using AI scene understanding
- **Detect** objects and **segment** scenes in images
- **Assess** image quality to find problematic data
- **Evaluate** retrieval performance with ground truth metrics

### How it works — the embedding pipeline

```
Raw Image → OpenCLIP ViT-B-32 → 512-dimensional vector (embedding)
                                        │
                                        ▼
                                  Qdrant Vector DB
                                 (stores embedding + metadata)
```

**Key idea**: Every image is converted into a 512-dimensional floating-point vector using OpenCLIP. Images that look similar end up with vectors that are close together in this high-dimensional space. To find similar images, we just find the nearest vectors.

**Why OpenCLIP ViT-B-32?**
- Trained on 2 billion image-text pairs (LAION-2B)
- Understands both images AND text in the same embedding space
- This means we can search with an image, with text, or with both simultaneously
- ViT-B-32 is a good balance between quality and speed (512-dim, fast inference)

### Models loaded at startup

| Model | Size | Purpose |
|-------|------|---------|
| **OpenCLIP ViT-B-32** | ~400 MB | Image and text embeddings for search |
| **SigLIP2 ViT-SO400M** | ~1.6 GB | Zero-shot scene classification (tagging) |
| **Grounding DINO Tiny** | ~340 MB | Open-vocabulary object detection |
| **SegFormer-B2 Cityscapes** | ~85 MB | Semantic segmentation (per-pixel labeling) |

### System architecture

```
┌─────────────────────────────────────────────────────────────────┐
│  React Frontend (localhost:3000)                                │
│  ┌──────────┐ ┌──────────┐ ┌────────┐ ┌─────────┐ ┌────────┐  │
│  │  Visual  │ │   Text   │ │ Image  │ │ Quality │ │Evaluate│  │
│  │  Search  │ │  Search  │ │Tagging │ │  Check  │ │ Metrics│  │
│  └────┬─────┘ └────┬─────┘ └───┬────┘ └────┬────┘ └───┬────┘  │
└───────┼────────────┼───────────┼───────────┼──────────┼────────┘
        │            │           │           │          │
        ▼            ▼           ▼           ▼          ▼
┌─────────────────────────────────────────────────────────────────┐
│  FastAPI Backend (localhost:8000)                                │
│  ┌───────────┐  ┌───────────┐  ┌───────────┐  ┌─────────────┐  │
│  │ OpenCLIP  │  │  SigLIP2  │  │ Grounding │  │  SegFormer   │  │
│  │ (search)  │  │ (tagging) │  │   DINO    │  │ (segment)    │  │
│  └─────┬─────┘  └───────────┘  └───────────┘  └─────────────┘  │
└────────┼────────────────────────────────────────────────────────┘
         │
         ▼
┌─────────────────┐
│  Qdrant (6333)  │
│  406 vectors    │
│  + metadata     │
│  + ground truth │
└─────────────────┘
```

---

## Part 3: Feature Demo Walkthrough

### Feature 1: Visual Search (Image-to-Image)

**Tab**: Visual Search

**What it does**: Upload a dashcam image and the system finds the most visually similar images from the dataset.

**How to demo**:
1. Drag and drop any image from `data/use_cases/crosswalk/` into the dropzone
2. Click **Search** — results appear ranked by visual similarity
3. Each result shows a similarity score (0–100%) and use_case label

**How it works behind the scenes**:
1. The uploaded image is sent to the backend
2. OpenCLIP converts it to a 512-dim embedding vector
3. Qdrant performs an approximate nearest neighbor (ANN) search across all 406 stored vectors
4. Results are ranked by **cosine similarity** — how close two vectors point in the same direction
5. Score of 100% = identical direction, 0% = completely unrelated

**Key talking points**:
- "We're using cosine similarity in a 512-dimensional space. Two images of the same intersection from slightly different angles might score 85–95% similarity."
- "The search is fast because Qdrant uses an HNSW graph index — logarithmic time complexity, not linear."
- "No training on our specific data was needed — OpenCLIP's pretrained knowledge of 2 billion images generalizes well to dashcam imagery."

### Feature 2: Region of Interest (Crop) Search

**Tab**: Visual Search

**What it does**: Draw a rectangle on the uploaded image to search using only that region.

**How to demo**:
1. Upload an image that has multiple objects (e.g., a crosswalk + vehicles)
2. Click and drag on the image to draw a crop rectangle over just the crosswalk
3. The dark overlay dims the excluded area
4. Click **Search** — results now focus on images similar to the cropped region

**Why this matters**:
- "In ADAS, an engineer might want to search for 'images with THIS specific type of crosswalk pattern' — not the whole scene."
- "The crop is applied before embedding, so the model only sees the region of interest."

### Feature 3: Text-to-Image Search

**Tab**: Text Search

**What it does**: Type a natural language description and find matching images.

**How to demo**:
1. Switch to the **Text Search** tab
2. Type: `a road with zebra crossing at night`
3. Click Search — results show nighttime crosswalk images
4. Try other queries: `traffic cones on highway`, `intersection with stop line`
5. You can also filter by use_case dropdown (crosswalk, stopline, cones)

**How it works**:
- OpenCLIP was trained on image-text pairs, so both images and text live in the **same** 512-dim vector space
- The text query is encoded with OpenCLIP's text encoder into a vector
- That vector is compared against all image vectors in Qdrant
- Images whose visual content matches the text description rank highest

**Key talking points**:
- "This is zero-shot — no fine-tuning needed. The model learned the relationship between visual concepts and language during pretraining."
- "This enables non-visual queries: you can search for 'rainy weather' or 'nighttime driving' without having metadata tags — the model understands these concepts."

### Feature 4: Hybrid Search (Image + Text Combined)

**Tab**: Visual Search (with text refinement)

**What it does**: Combine image similarity with a text description using adjustable weighting.

**How to demo**:
1. Upload a daytime crosswalk image
2. In the "Text refinement" field, type: `at night`
3. Open **Settings** and adjust the Image/Text weight slider
4. Click Search — results blend visual similarity with the text concept

**The math**:
```
final_embedding = image_weight × image_embedding + (1 - image_weight) × text_embedding
```

**Key talking points**:
- "With weight 0.7 (default), the search is primarily visual but nudged by the text concept."
- "Set weight to 0.0 for pure text search, 1.0 for pure image search, or anywhere in between."
- "Use case: 'Find images that look like this intersection but at night' — something neither modality alone can express."

### Feature 5: Reverse Search (Find Similar)

**Tab**: Any search results page

**What it does**: Click any result to use it as a new search query — without re-uploading.

**How to demo**:
1. Perform any search
2. Hover over a result thumbnail — a search icon appears
3. Click it → the system uses that image's **stored embedding** as a new query
4. Results update instantly (no re-embedding needed)

**Why this matters**:
- "This enables exploratory workflows: start with a rough query, then refine by jumping through similar images."
- "Because we reuse the stored embedding from Qdrant, this is faster than uploading a new image."

### Feature 6: Image Tagging (Zero-Shot Classification)

**Tab**: Image Tagging

**What it does**: Automatically classify images by scene attributes using AI.

**How to demo**:
1. Switch to **Image Tagging**
2. Upload a dashcam image
3. Results show confidence bars for each tag group:
   - **Time of day**: day / night / dawn_dusk
   - **Scene type**: urban / rural / highway / suburban
   - **Crosswalk**: present / absent
   - **Pedestrian**: present / absent
   - **Vehicle density**: low / moderate / heavy

**CLIP-only mode** uses SigLIP2 (1.1B parameters) to compare the image against carefully crafted text descriptions for each category.

**Hybrid mode** (toggle at top) runs a 3-stage pipeline:
1. **CLIP scene tags** (purple) — SigLIP2 zero-shot classification
2. **Detection object tags** (amber) — Grounding DINO detects actual objects (vehicles, pedestrians, crosswalks) and derives tags from counts
3. **Segmentation layout tags** (teal) — SegFormer measures road coverage, vegetation, sky visibility, built environment percentages

**Key talking points**:
- "CLIP tagging is zero-shot — we never trained it on dashcam images. We just describe what to look for in natural language."
- "Hybrid tagging fuses three AI models. Detection-derived tags have 90–95% confidence because they're based on actually finding objects in the image."
- "This could automate metadata labeling for large ADAS datasets — replacing manual annotation."

### Feature 7: Object Detection (Grounding DINO)

**Tab**: Click a search result → Detect button

**What it does**: Detect objects in an image with bounding boxes and labels.

**How to demo**:
1. Perform a search, click a result to open the detail view
2. Click the **Detect** button
3. Bounding boxes appear over detected objects: vehicles, pedestrians, crosswalks, etc.
4. Each box shows the class label and confidence score

**Model**: Grounding DINO Tiny — an **open-vocabulary** detector. Unlike YOLO which only detects fixed classes, Grounding DINO accepts any text prompt describing what to find.

**Default ADAS prompt**: `vehicle. pedestrian. traffic sign. traffic light. crosswalk. lane marking. bicycle. motorcycle. bus. truck.`

### Feature 8: Semantic Segmentation (SegFormer)

**Tab**: Click a search result → Segment button

**What it does**: Labels every pixel in the image with its semantic class.

**How to demo**:
1. Click the **Segment** button on a result
2. A color-coded overlay appears showing 19 Cityscapes classes: road (purple), sidewalk (pink), building (gray), vegetation (green), sky (blue), car (dark blue), person (red), etc.
3. Below the overlay, percentage bars show the composition: "road: 42%, vegetation: 18%, sky: 15%..."

**Model**: SegFormer-B2 fine-tuned on Cityscapes — specifically designed for driving scene understanding.

### Feature 9: Segmentation-Guided Detection (Two-Stage Pipeline)

**Tab**: Click a result → Segment first, then Detect (guided)

**What it does**: Uses segmentation to automatically build a smarter detection prompt.

**How to demo**:
1. Click **Segment** on a result
2. Then click **Detect (guided)**
3. The system sees segmentation found, e.g., "car: 8%, person: 2%"
4. It automatically builds a targeted prompt: `car. person.` instead of the generic 10-class prompt
5. Green info bar shows: "Guided by segmentation: car. person."

**Why this matters**:
- "Instead of blindly searching for 10 object classes, we first understand what's in the scene, then look only for those things."
- "This reduces false positives and improves detection confidence."

### Feature 9b: Video Segmentation (Short Clips + Timing)

**Tab**: Video Segment

**What it does**: Runs SegFormer over short clips frame-by-frame and exports a browser-compatible segmented MP4.

**How to demo**:
1. Open **Video Segment** tab.
2. Upload one clip from `data/use_cases/woodscape_clips_10s/`.
3. Set `max_frames` (e.g., 120) and click **Run Segmentation**.
4. Show output preview and download button.
5. Point out timing metrics: total time, segmentation time, encoding time, and per-frame min/avg/max.

**Key talking points**:
- "This extends image segmentation to near-field parking-style video clips."
- "The backend reports per-frame timing, so we can profile latency and throughput."
- "Output is encoded as H.264 (`yuv420p`) for browser playback compatibility."

### Feature 10: Side-by-Side Comparison

**Tab**: Any search results page

**What it does**: Select multiple results and compare them in a full-screen panel.

**How to demo**:
1. Perform a search
2. Hover results → click the checkbox (top-left) on 2–3 results
3. Click the green **Compare (N)** button that appears
4. Full-screen panel shows images side by side with:
   - Metadata (use_case, similarity score)
   - Individual Segment and Detect buttons per card

### Feature 11: Image Quality Assessment

**Tab**: Quality

**What it does**: Scans all images in the dataset and ranks them by quality, surfacing the worst ones.

**How to demo**:
1. Switch to **Quality** tab
2. Set count (e.g., 20)
3. Click **Find Low Quality Images**
4. Results show the worst images, ranked by composite score, with per-metric breakdown

**Four quality metrics**:

| Metric | Method | What it catches |
|--------|--------|----------------|
| **Blur** | Laplacian variance | Motion blur, defocus, camera shake — low variance = blurry |
| **Exposure** | Histogram analysis | Overexposed (sun glare) or underexposed (too dark) |
| **Contrast** | RMS contrast | Low contrast from fog, haze, or dirty lens |
| **CLIP-IQA** | SigLIP2 zero-shot | Overall perceptual quality — "sharp clear photo" vs "blurry noisy photo" |

**Composite score**: 0–100, weighted: blur 35%, CLIP-IQA 25%, exposure 25%, contrast 15%.

**Labels**: Good (70+), Fair (45–70), Poor (<45).

**Key talking points**:
- "In ADAS data collection, millions of images are captured. Quality assessment lets you automatically filter out unusable frames."
- "The Laplacian blur detector computes the variance of edge responses. A perfectly sharp image has high edge variance; a motion-blurred image has low variance."
- "CLIP-IQA is novel — we ask SigLIP2 'Is this a sharp, clear, high-quality photo?' vs 'Is this a blurry, noisy, low-quality photo?' and use the confidence as a quality score."

---

## Part 4: Evaluation (The Metrics)

### Feature 12: Text-to-Image Search Evaluation

**Tab**: Evaluate

**What it does**: Measures how well text search retrieves the right images, using ground truth labels parsed from annotation files.

**How to demo**:
1. Switch to **Evaluate** tab
2. Click a preset: **Crosswalk**
   - This runs the query `"crosswalk on road"` against the dataset
   - Ground truth: every image whose label JSON contains a crosswalk annotation
3. Metrics table appears with Precision@K, Recall@K, nDCG@K at K = 5, 10, 20, 50, 100, 200
4. Below: scrollable results list. Each result is color-coded:
   - **Green (Relevant)**: has the target ground truth tag
   - **Orange (Labeled but not relevant)**: has labels but not this specific tag
   - **Gray (No ground truth)**: unlabeled image, treated as unjudged
5. Click **Run All Presets** to batch-evaluate: crosswalk, stopline, cones, lane_marking, road_edge
6. **Batch Summary** table compares all tags side by side

### Understanding the Metrics

#### Precision@K — "Of the top K results, how many are actually relevant?"

```
Precision@K = (number of relevant results in top K) / K
```

**Example**: You search for "crosswalk" and look at the top 10 results. If 8 out of 10 actually contain crosswalks:
- Precision@10 = 8/10 = **80%**

**Interpretation**:
- Precision@10 = 100% means every single result in the top 10 is correct
- Precision@10 = 50% means half the results are wrong
- **Higher is better**. It answers: "How much can I trust the top results?"

**Why it matters for ADAS**: "If an engineer searches for 'crosswalk images' to review annotation quality, they don't want to waste time scrolling through irrelevant images."

#### Recall@K — "Of ALL relevant images in the database, how many did we find in the top K?"

```
Recall@K = (number of relevant results in top K) / (total relevant images in database)
```

**Example**: There are 244 crosswalk images in the database. Your top 50 results contain 44 crosswalks:
- Recall@50 = 44/244 = **18%**

**Interpretation**:
- Recall@50 = 100% means you found every single relevant image in just 50 results
- Recall@50 = 18% means you found 18% of all relevant images
- Recall naturally increases as K increases (at K = total database, recall = 100%)
- **Higher is better**. It answers: "How complete is my retrieval? Am I missing relevant images?"

**Why it matters for ADAS**: "If you're building a training dataset for crosswalk detection, you need high recall — missing images means missing edge cases."

#### nDCG@K — "Are the relevant results ranked at the TOP of the list?"

**nDCG** (normalized Discounted Cumulative Gain) measures ranking quality. It rewards placing relevant results earlier.

```
DCG@K = Σ (relevance_i / log2(i + 1))    for i = 1 to K

nDCG@K = DCG@K / ideal_DCG@K
```

The key insight: **a relevant result at position 1 is worth more than one at position 50**. The `log2(i+1)` denominator discounts later positions.

**Example**:
- Search A returns: [relevant, relevant, relevant, irrelevant, relevant] → nDCG is high (good stuff at the top)
- Search B returns: [irrelevant, irrelevant, irrelevant, relevant, relevant] → nDCG is lower (relevant stuff buried)

Both have the same Precision@5 (60%), but Search A has better nDCG because relevant results appear first.

**Interpretation**:
- nDCG@K = 1.0 (100%) means perfect ranking — all relevant items appear before all irrelevant ones
- nDCG@K = 0.0 means no relevant items found
- **Higher is better**. It answers: "Is the ranking order correct? Are the best results truly at the top?"

**Why it matters for ADAS**: "When showing 200 results, it's critical that the most relevant images appear in the first few rows, not buried on page 5."

### How to read the evaluation results

**Our benchmark results** (406 images, 313 labeled):

| Tag | P@10 | R@50 | nDCG@50 | Relevant in DB | Interpretation |
|-----|------|------|---------|----------------|----------------|
| **crosswalk** | 100% | 18% | 99% | 244 | Excellent precision — every top-10 result is correct. nDCG near-perfect. |
| **road_edge** | 80% | 13.8% | 94.7% | 282 | Strong precision and ranking quality. |
| **lane_marking** | 50% | 7.6% | 72.2% | 262 | Moderate — CLIP conflates lane markings with other road features. |
| **stopline** | 30% | 9.3% | 63.3% | 140 | Harder — stop lines are visually subtle, thin lines on road. |
| **cones** | 0% | 9.1% | 26% | 22 | Poor — cones are small objects; CLIP text-to-image struggles. |

**Key insight to explain**: "The model works well for visually distinctive features (crosswalks = clear zebra patterns, road edges = obvious curbs) but struggles with small or subtle objects (cones, stop lines). This is a known limitation of whole-image classification — the model sees the entire scene, not individual small objects."

---

## Part 5: Demo Script (Recommended Flow)

### Suggested 15-minute demo order

1. **Architecture slide** (2 min) — explain embedding pipeline, vector search concept
2. **Visual Search** (2 min) — drag-and-drop image, show results, explain similarity scores
3. **Crop Search** (1 min) — draw region on image, show how results change
4. **Text Search** (2 min) — type "crosswalk at night", show results, explain shared embedding space
5. **Hybrid Search** (1 min) — combine image + text, adjust slider
6. **Reverse Search** (30 sec) — click a result → Find Similar
7. **Image Tagging** (2 min) — show CLIP-only, then toggle Hybrid for 3-stage analysis
8. **Detection + Segmentation** (1.5 min) — click result, segment, then guided detection
9. **Video Segmentation (optional)** (1 min) — run one short clip and show timing stats
10. **Quality Assessment** (1 min) — find worst 10 images, show blur/exposure issues
11. **Evaluation** (2 min) — run all presets, explain P@K / R@K / nDCG, show batch summary

### Key messages to emphasize

- **Zero-shot**: No training on ADAS data was needed. All models use pretrained knowledge.
- **Multi-modal**: The same embedding space supports image search, text search, and hybrid queries.
- **Production-ready architecture**: Qdrant vector database, FastAPI, React frontend — all standard production tools.
- **Quantitative evaluation**: We measure real retrieval metrics using label-derived ground truth, not just qualitative "looks good."
- **Extensible**: Adding new images is incremental — no rebuild needed. New use cases just need a folder of images.

---

## Part 6: Troubleshooting During Demo

| Issue | Fix |
|-------|-----|
| Frontend shows blank page | Check `npm run dev` is running in the frontend directory |
| Search returns no results | Verify Qdrant has data: `curl http://localhost:6333/collections/adas_images` |
| Backend won't start | Check port 8000 is free: `lsof -i :8000` and kill any old process |
| Qdrant not reachable | `docker ps` to check container is running, `docker start qdrant` if stopped |
| Models loading slowly | First startup downloads models (~2.5 GB total). Subsequent starts use cache. |
| Images not displaying | Ensure `data/use_cases/` contains the images and backend is serving them |

### Quick restart everything

```bash
# Kill all processes
lsof -i :8000 -t | xargs kill 2>/dev/null
lsof -i :3000 -t | xargs kill 2>/dev/null

# Restart Qdrant
docker start qdrant

# Start backend
cd /path/to/visual-search
uv run uvicorn app.main:app --host 0.0.0.0 --port 8000 &

# Start frontend
cd frontend && npm run dev &

# Wait for backend to load models (30–60 seconds)
# Then open http://localhost:3000
```
