# Auto-labeler pipeline overview

Logic maps for the **image** notebook, **video** notebook, and **YOLO-seg training** notebook. Render Mermaid in GitHub, VS Code / Cursor (Markdown preview), or any Mermaid-compatible viewer.

Related notebooks:

- `qwen_sam3_image_pipeline.ipynb`
- `qwen_sam3_pipeline_video.ipynb`
- `train_yolo_seg_from_image_pipeline.ipynb`
- `mine_qwen_taxonomy.ipynb` (optional: mine Qwen prompts from existing runs to grow `YOLO_CLASS_NAMES`)

---

## Expanding the closed set (Option A)

To add **new YOLO classes** instead of relying on open vocabulary alone:

1. Run **`mine_qwen_taxonomy.ipynb`** on an `ARTIFACTS_ROOT` that already contains runs with **`qwen_output.json`** (same tree as image/video outputs). It prints prompt frequencies, flags strings that do not map to your current `YOLO_CLASS_NAMES` / `YOLO_CLASS_ALIASES`, and can write **`mine_taxonomy_report.json`** for review.
2. Edit **Step 3** in `qwen_sam3_image_pipeline.ipynb` (and the video notebook’s matching class list / aliases) so **`nc`**, names, and aliases cover the taxonomy you want; merge near-duplicates with aliases rather than inflating `nc` blindly.
3. **Re-run** the image (and/or video) labeling pipeline so **`labels/segments/*.txt`**, **`classes.txt`**, and **`metadata.json`** are regenerated with the new class indices. Old segment files stay on the old index scheme until you re-export.
4. Rebuild the training dataset and **retrain** with `train_yolo_seg_from_image_pipeline.ipynb` (or point training at a fresh `DATASET_ROOT`).

---

## End-to-end relationship

```text
                    ┌─────────────────┐
  images/videos ──► │  Qwen (prompts) │   slow: vision–language + text decode
                    └────────┬────────┘
                             │
                    ┌────────▼────────┐
                    │ SAM3 (masks)   │   faster: vision + mask heads
                    └────────┬────────┘
                             │
              ┌──────────────┴──────────────┐
              ▼                             ▼
   Image / video notebooks          train_yolo_seg_from_image_pipeline.ipynb
   write labels/segments +           reads artifact folders + metadata.json
   metadata.json                    → Ultralytics dataset + `yolo segment train`
```

---

## High-level diagrams

Two separate flows: **labeling** (Qwen + SAM3 in the image notebook) vs **training** (dataset build + Ultralytics only; no VL/segmentation models).

### Image labeling pipeline

```mermaid
flowchart LR
  subgraph in["Input"]
    I[Images or USS-PF jobs]
  end

  subgraph models["Per image"]
    Q[Qwen2.5-VL<br/>prompts / regions]
    S[SAM3<br/>instance masks]
  end

  subgraph out["Artifacts"]
    L[YOLO boxes + segments<br/>masks, overlays]
    J[qwen_output.json<br/>metadata, logs]
  end

  I --> Q --> S --> L
  S --> J
```

### YOLO training pipeline (separate)

```mermaid
flowchart LR
  subgraph src["Source"]
    A[ARTIFACTS_ROOT<br/>labeled run folders]
  end

  subgraph build["Dataset build"]
    D[Discover segment *.txt<br/>+ metadata image paths]
    V[Train / val split]
    Y[data.yaml + Ultralytics<br/>folder layout]
  end

  subgraph tr["Training"]
    U[Ultralytics<br/>yolo segment train]
    W[weights / best.pt]
  end

  A --> D --> V --> Y --> U --> W
```

*The detailed step-by-step Mermaid charts for each notebook are in the sections below.*

---

## 1. Image pipeline (`qwen_sam3_image_pipeline.ipynb`)

```mermaid
flowchart TD
  subgraph cfg["Step 3 — Config"]
    A1[INPUT_PATH: flat folder, single file, or USS_PF root]
    A2[OUTPUT_ROOT, YOLO_CLASS_NAMES, TOP_N, checkpoint flags, ...]
  end

  subgraph prep["Steps 4–8 — Utilities + models + export"]
    B1[collect_images OR collect_uss_pf_front_rear_jobs]
    B2[Per job: image_path + output_root e.g. OUTPUT_ROOT / seq_id / stem]
    B3[Load Qwen2.5-VL + SAM3]
    B4[process_image: read RGB → Qwen → SAM3 → export]
  end

  subgraph run["Step 9 — Run"]
    C1[Load / merge pipeline_state.json resume]
    C2[For each pending job]
    C3[process_image]
  end

  subgraph out["Per image run folder"]
    D1[qwen_output.json]
    D2[labels/boxes, labels/segments, labels/masks]
    D3[annotated PNGs]
    D4[processing_log.json]
  end

  subgraph post["Steps 10–11 + optional 10b"]
    E1[Validate segment labels + optional label summary JSON]
    E2[summary_report.json + timing table]
  end

  cfg --> prep
  prep --> run
  run --> C3
  C3 --> out
  out --> post
```

**Summary:** one **VL pass (Qwen)** for prompts, one **segmentation pass (SAM3)** per image, then **YOLO-style exports**; optional **JSON state** for resume (`pipeline_state.json`).

---

## 2. Video pipeline (`qwen_sam3_pipeline_video.ipynb`)

```mermaid
flowchart TD
  subgraph cfg["Step 3 — Config"]
    V1[INPUT_PATH: video file or folder]
    V2[OUTPUT_ROOT, TOP_N, MAX_FRAMES, BBOX format, YOLO class lists, ...]
  end

  subgraph prep["Utilities + models + export helpers"]
    W1[collect_videos]
    W2[make_video_output_dir: videos, labels/boxes, labels/segments, labels/masks, images/frames]
    W3[Load Qwen + SAM3]
  end

  subgraph pv["process_video"]
    P1[Read frames up to MAX_FRAMES]
    P2[Qwen on video path → prompts once]
    P3[SAM3 over frames → outputs_per_frame]
    P4[For each frame index in outputs]
    P5[images/frames/000xxx.jpg]
    P6[labels/boxes/000xxx.txt + labels/segments/000xxx.txt]
    P7[labels/masks + overlay MP4s]
    P8[metadata.json + processing_log.json]
  end

  subgraph run9["Step 9 — Run loop"]
    R1[For each video path]
    R2[process_video]
  end

  cfg --> prep
  prep --> run9
  run9 --> R2
  R2 --> P1 --> P2 --> P3 --> P4
  P4 --> P5
  P4 --> P6
  P4 --> P7
  P7 --> P8
```

**Summary:** **Qwen once per clip** (prompts), **SAM3 per frame**, then **aligned stems** `000123.jpg` ↔ `000123.txt` for YOLO-seg-style training exports.

---

## 3. YOLO-seg training (`train_yolo_seg_from_image_pipeline.ipynb`)

```mermaid
flowchart TD
  subgraph cfg["Config"]
    T1[ARTIFACTS_ROOT e.g. outputs/uss_pf or outputs/images]
    T2[DATASET_ROOT, DEVICE auto/cuda/mps/cpu, DRY_RUN, train hyperparams]
  end

  subgraph scan["Discover pairs"]
    S1[Walk flat or nested run dirs with labels/segments]
    S2[Segment *.txt + labels/metadata.json → field image]
    S3[Skip empty labels, missing images, inconsistent class lists]
  end

  subgraph split["Train / val split"]
    L1[sequence groups or random per image]
  end

  subgraph mat["Materialize Ultralytics layout"]
    M1[Symlink or copy into images/train, images/val, labels/train, labels/val]
    M2[Same stem for each image + label pair]
    M3[Write data.yaml: path, train, val, nc, names]
  end

  subgraph train["Train optional"]
    U1[resolve_training_device]
    U2[YOLO(weights).train(data, epochs, imgsz, batch, device, project, name)]
    U3[weights/best.pt under project/name]
  end

  cfg --> scan --> split --> mat
  mat --> train
```

**Summary:** **no Qwen/SAM3** — only **scanning artifacts**, **pairing** label files with `metadata.json["image"]`, building **`data.yaml`**, then **Ultralytics** `yolo segment train`.

---

## File reference

| Artifact | Typical location |
|----------|------------------|
| Image run (flat) | `OUTPUT_ROOT/<image_stem>/` |
| Image run (USS-PF) | `OUTPUT_ROOT/<seq_id>/<image_stem>/` |
| Video run | `OUTPUT_ROOT/<video_stem>/` |
| Segment labels | `.../labels/segments/<stem>.txt` |
| Source image pointer | `.../labels/metadata.json` → `"image"` |
| Training dataset | `DATASET_ROOT/images/{train,val}/`, `labels/{train,val}/`, `data.yaml` |
| Training runs | `PROJECT/NAME/weights/best.pt` |

---

*Generated for the visual-search `auto-labeler` workflows.*
