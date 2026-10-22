# Visual Search POC — Cloud Deployment + Training Options

This document complements the main POC README by covering:

1) **How to deploy the POC to Azure** (demo-first and production-shaped stacks)  
2) **Model training / fine-tuning options** to improve retrieval quality over time

---

## 1) Azure cloud deployment

### 1.1 What the local POC contains
Your current POC typically has these components:

- **Artifacts (index + embeddings)**
  - `embeddings.f32`
  - `id_map.json`
  - `hnsw.bin`
  - `index_meta.json`
- **Services**
  - Batch indexing: ingest → embed → build index
  - Query serving: FastAPI search API
  - Demo UI: Streamlit app
  - Offline evaluation: SOP metrics scripts
  - Online instrumentation: impressions/clicks → CTR

The Azure deployment maps these to managed components while keeping the architecture recognizable.

---

### 1.2 Option A — Fastest Azure POC stack (recommended for demos)

This option keeps your current **file-based ANN index** (HNSW) and embeds/reranks in your API container.

#### Components
**Compute**
- **Azure Container Apps (ACA)**  
  - `search-api` (FastAPI)  
  - optional `ui` (Streamlit)  

**Storage**
- **Azure Blob Storage**
  - store index and embedding artifacts under a versioned prefix:
    - `artifacts/embeddings/...`
    - `artifacts/index/...`
  - optionally store gallery images if you want cloud-only demos

**Online metrics**
- **Azure Table Storage** (simplest) or **Cosmos DB** (more flexible)
  - store impression and click events

**Scheduled metrics jobs**
- **Container Apps Jobs** or **Azure Functions**
  - compute CTR@K periodically and write summaries to storage

**Observability**
- **Application Insights** + Log Analytics  
  - API latency, error rate, custom logs

**Secrets**
- **Azure Key Vault** for connection strings/SAS tokens

**CI/CD**
- GitHub Actions → build/push to **Azure Container Registry (ACR)** → deploy to ACA

#### Serving behavior
- On container startup:
  - download artifacts from Blob into local container filesystem (or mount Azure Files)
  - load HNSW index into memory
- On query:
  - compute embedding (CPU is typically fine for POC)
  - ANN retrieve top K′
  - exact cosine rerank
  - log impressions/clicks to Table/Cosmos

#### Why this is best for a demo
- Minimal changes to your codebase
- Predictable runtime behavior
- Easy to explain as “the same system, but in the cloud”

---

### 1.3 Option B — Production-shaped Azure stack (managed vector search)

This option replaces file-based ANN with a managed vector search service.

#### Components
**Vector index**
- **Azure AI Search (Vector Search)**
  - stores vectors + metadata
  - supports filters and hybrid retrieval

**Embedding compute**
- **Azure ML Online Endpoint** (GPU if needed)  
  - online embeddings at scale  
- **Azure ML Batch Endpoints / Jobs**
  - batch embedding updates for new images

**Serving**
- FastAPI on **Azure Container Apps** or **AKS**
  - query → embed → Azure AI Search vector query → optional rerank

**Pipelines**
- **Azure Data Factory** or **Azure ML Pipelines**
  - ingest images → embed → upsert vectors

**Event logging + analytics**
- **Event Hubs** for streaming events
- **Cosmos DB** or **Azure Data Explorer** for analytics
- Power BI dashboards (optional)

#### Why this is better for production
- You no longer manage ANN binary artifacts
- Index updates are cleaner (upserts)
- Built-in scaling and HA

---

### 1.4 Recommended Azure POC blueprint (Option A)

#### Minimal resource list
- Resource Group
- Storage Account (Blob + Table)
- Container Registry (ACR)
- Container Apps Environment
- Container App: `search-api`
- (Optional) Container App: `ui`
- Application Insights
- Key Vault

#### Environment variables (example)
Use env vars to avoid hardcoding paths and to support local/cloud parity:

- `CONFIG_PATH=configs/sop_demo.yaml`
- `ARTIFACT_SOURCE=local|azure_blob`
- `BLOB_ACCOUNT=<storage_account_name>`
- `BLOB_CONTAINER=<container_name>`
- `BLOB_PREFIX=artifacts/`
- `EVENT_STORE=sqlite|table|cosmos`
- `EVENT_DB_PATH=artifacts/metadata/sop_demo.sqlite` (if sqlite)
- `TOP_K_DEFAULT=50`
- `TOP_K_CANDIDATES_DEFAULT=200`

#### Artifact packaging and upload
Use versioned folders to avoid mixing runs:
- `artifacts/<dataset_id>/<model>/<preprocess>/<index_version>/...`

Upload artifacts:
```bash
az storage blob upload-batch   --account-name <acct>   --destination <container>   --source artifacts   --destination-path artifacts
```

#### Startup artifact download pattern
At API startup:
1) Determine artifact root (`local` vs `azure_blob`)
2) Ensure the following files exist locally:
   - `embeddings.f32`, `id_map.json`, `hnsw.bin`, `index_meta.json`
3) Load index into memory
4) Serve

---

### 1.5 Common Azure POC issues and fixes

#### Issue: `FileNotFoundError ... index_meta.json`
Root cause:
- API is using a config whose `index_version` does not exist in artifacts

Fix:
- Start API with the same config you used to build the index artifacts
- Ensure `index.index_version` matches the built artifact folder name

#### Issue: `ValueError: id_map size != index count`
Root cause:
- Mixed artifacts from different runs (id_map/embeddings vs index binary)

Fix:
- Namespace artifacts by dataset_id and/or index_version
- Or delete and rebuild all matching folders:
  - embeddings folder + index folder must be rebuilt together

---

## 2) Model training / fine-tuning options

Training is optional for the POC. In production, training can provide significant relevance improvements, especially for specific domains (fashion, furniture, SKU-level matching).

### 2.1 Option 0 — No training (pretrained embeddings baseline)
Use a strong pretrained model (e.g., OpenCLIP/CLIP/DINOv2) and focus on:
- ANN quality/latency tuning
- reranking and dedup policies
- UI/UX behavior
- offline/online evaluation loops

**Pros**
- Minimal engineering
- Strong generalization
- Great for POC/demo

**Cons**
- Not tuned to your domain’s definition of “similar”

---

### 2.2 Option 1 — Supervised metric learning (best “first training” step)
Fine-tune the embedding model using labeled datasets:
- SOP: `class_id` (binary relevance by class)
- DeepFashion In-shop: `item_id` (instance retrieval)

#### Training objectives (common)
- **Triplet loss**: (anchor, positive, negative)
- **InfoNCE / contrastive loss** (batch negatives)
- **Proxy-based losses** (ProxyNCA, ArcFace-like)

#### Data requirements
- Class/item labels, enough images per class/item

**Pros**
- Usually improves retrieval quality on the dataset domain
- Straightforward to implement

**Cons**
- Risk of overfitting with small data
- Gains may not transfer to real catalog unless domain matches

**Recommended for POC if you want a “trained model” story**
- Fine-tune on SOP or DeepFashion for a measurable nDCG/mAP lift
- Re-run offline metrics using the same evaluation harness

---

### 2.3 Option 2 — Self-supervised contrastive training (SimCLR/MoCo)
Train without labels using augmentations:
- Positive pair: two augmentations of the same image
- Negatives: other images in batch/memory bank

**Pros**
- No manual labels required
- Scales with data volume

**Cons**
- Requires more training compute and careful augmentations
- Often not necessary if CLIP baseline is strong unless you have massive domain data

When to use:
- Very large unlabeled corpus
- Domain shift is strong (CLIP baseline is insufficient)

---

### 2.4 Option 3 — Click-based training (production-realistic)
Use user interactions:
- query image → clicked results as positives
- shown but not clicked results as negatives (noisy)

#### Key requirement: bias handling
Clicks are biased by rank and UI:
- position bias
- presentation bias
- popularity bias

Mitigations:
- randomized buckets / interleaving experiments
- counterfactual correction (advanced)
- use clicks primarily for **hard negative mining** and **reranker training**

**Pros**
- Optimizes for real user preference
- Strong long-term ROI

**Cons**
- Requires traffic and careful data hygiene

---

### 2.5 Option 4 — Two-stage ranking (embedding retrieval + learned reranker)
Keep ANN retrieval stage, but train a reranker for top K′.

#### Reranker inputs
- cosine similarity
- embedding element-wise features (|q − x|, q·x)
- shallow visual features (optional)
- domain metadata (optional in production)

**Pros**
- Often yields the biggest nDCG lift for the top results
- Doesn’t require changing ANN infrastructure

**Cons**
- Requires pair labels (human judgments or click-derived)

Recommended when:
- You care strongly about ordering in top-20/top-50
- You can obtain even a small labeled set

---

### 2.6 Option 5 — Use metadata (production, not required in POC)
Even if the POC assumes “pixels-only,” production systems often leverage:
- category/tags/brand
- text descriptions (image-text contrastive learning)
- “same SKU” relationships

Approaches:
- multi-task (classification + metric learning)
- CLIP-style image-text alignment
- hard negatives within-category

---

## 3) Recommended training progression (practical)

### Phase 1 — Baseline (already done)
- Pretrained embeddings
- ANN + rerank
- Offline eval (Recall/mAP/nDCG)
- Online logging (CTR)

### Phase 2 — “POC training story”
- Supervised metric learning on SOP or DeepFashion
- Evaluate with the same offline harness
- Swap weights in serving to show lift

### Phase 3 — Production alignment
- Add click logging at scale
- Use click-derived pairs for:
  - hard negative mining
  - reranker training
- Add a small human-judgment set for unbiased offline nDCG

---

## 4) Training vs serving impacts (what changes in the stack)

### If you do supervised fine-tuning (Option 1)
- Add **training compute** (Azure ML job/cluster or local GPU)
- Add **model registry** (Azure ML model registry or blob versioning)
- Serving remains the same; only the embedding model weights change

### If you do click-based reranking (Option 4 + Option 3)
- Add event pipeline:
  - Event Hubs (optional), Cosmos/ADX
- Add training pipeline for reranker model
- Add A/B test framework for online validation

---

## 5) Minimal “Azure + Training” POC plan

1) Deploy Option A to Azure Container Apps
2) Upload artifacts to Blob and load on container startup
3) Log impressions/clicks to Table Storage
4) Compute CTR@K via a scheduled job
5) Optional: run supervised fine-tuning in Azure ML and redeploy with updated weights
6) Re-run offline metrics and show deltas

---

## 6) Glossary (quick)

- **Embedding**: fixed-length vector representation of an image
- **ANN**: approximate nearest neighbor search to retrieve candidates quickly
- **HNSW**: graph-based ANN method (good quality/latency tradeoff)
- **K**: final returned results count
- **K′ (top_k_candidates)**: candidate pool size retrieved by ANN before reranking
- **nDCG**: ranking metric that discounts lower-ranked results; best with graded relevance
- **CTR**: click-through rate; clicks divided by impressions
