"""
SAM3 + Qwen2.5-VL video pipeline (synced from qwen_sam3_pipeline_video.ipynb).
For multi-process runs use: run_sam3_video_batch.py
"""
from __future__ import annotations

import datetime
import gc
import json
import os
import re
import time
import traceback
from collections import Counter
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image

# ── File collection ────────────────────────────────────────────────────────
def _video_name_excluded(path: Path, ignore_substrings: tuple[str, ...]) -> bool:
    if not ignore_substrings:
        return False
    name = f"{path.stem} {path.name}".lower()
    return any(s.lower() in name for s in ignore_substrings)


def collect_videos(
    input_path: str,
    extensions: set,
    ignore_name_substrings: tuple[str, ...] | None = None,
) -> list[Path]:
    """Return a list of video paths from a file or directory.

    ignore_name_substrings: if None, uses global VIDEO_IGNORE_NAME_SUBSTRINGS, or ("topview",)
    when unset. Pass () to skip name filtering. Match is case-insensitive on stem + filename.
    """
    if ignore_name_substrings is None:
        raw = globals().get("VIDEO_IGNORE_NAME_SUBSTRINGS")
        if raw is None:
            ignore_name_substrings = ("topview",)
        else:
            ignore_name_substrings = tuple(raw)
    p = Path(input_path)
    if p.is_file():
        if p.suffix.lower() not in extensions:
            raise ValueError(f"File {p} is not a recognised video format: {extensions}")
        if _video_name_excluded(p, ignore_name_substrings):
            raise ValueError(
                f"File {p.name} is excluded by VIDEO_IGNORE_NAME_SUBSTRINGS={ignore_name_substrings!r}"
            )
        return [p]
    elif p.is_dir():
        videos = sorted(
            f
            for f in p.rglob("*")
            if f.suffix.lower() in extensions
            and not _video_name_excluded(f, ignore_name_substrings)
        )
        if not videos:
            raise FileNotFoundError(
                f"No video files found in {p} (after extensions {extensions} and "
                f"name filters {ignore_name_substrings!r})"
            )
        return videos
    else:
        raise FileNotFoundError(f"INPUT_PATH not found: {p}")

# ── Output directory layout ────────────────────────────────────────────────
def make_video_output_dir(output_root: Path, video_path: Path) -> dict:
    """Create and return all sub-directory paths for one video."""
    slug = video_path.stem.replace(" ", "_")
    base  = output_root / slug
    dirs  = {
        "base":           base,
        "videos":         base / "videos",
        "images_frames":   base / "images" / "frames",
        "images_overlays": base / "images" / "overlays",
        "label_boxes":    base / "labels" / "boxes",
        "label_segments": base / "labels" / "segments",
        "label_masks":    base / "labels" / "masks",
        "label_meta":     base / "labels",
    }
    for d in dirs.values():
        d.mkdir(parents=True, exist_ok=True)
    return dirs

# ── JSON helpers ──────────────────────────────────────────────────────────
def extract_json_block(text: str) -> dict:
    text = text.strip()
    try:
        return json.loads(text)
    except Exception:
        pass
    m = re.search(r"```json\s*(\{.*?\})\s*```", text, flags=re.S)
    if m:
        try: return json.loads(m.group(1))
        except Exception: pass
    m = re.search(r"(\{.*\})", text, flags=re.S)
    if m:
        try: return json.loads(m.group(1))
        except Exception: pass
    raise ValueError(f"Could not parse JSON from:\n{text[:300]}")

# ── Timing context manager ────────────────────────────────────────────────
class Timer:
    def __init__(self, label=""):
        self.label   = label
        self.elapsed = 0.0
    def __enter__(self):
        self._start = time.perf_counter()
        return self
    def __exit__(self, *_):
        self.elapsed = time.perf_counter() - self._start
        print(f"  ⏱  {self.label}: {self.elapsed:.2f}s")

# ── Video reader ──────────────────────────────────────────────────────────
def read_video_opencv(path: str, max_frames=None) -> tuple[list, dict]:
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open: {path}")
    fps        = cap.get(cv2.CAP_PROP_FPS) or 15.0
    total_est  = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frames, i  = [], 0
    while True:
        ok, frame = cap.read()
        if not ok: break
        frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        i += 1
        if max_frames and i >= max_frames: break
    cap.release()
    return frames, {"fps": float(fps), "total_num_frames": len(frames),
                    "estimated_total": total_est}

def _prime_psutil_cpu_meter():
    """Return a psutil.Process with CPU % primed (first cpu_percent is baseline)."""
    try:
        import psutil

        p = psutil.Process(os.getpid())
        p.cpu_percent(interval=None)
        return p
    except Exception:
        return None


def snapshot_process_resources(cpu_proc=None) -> dict:
    """
    JSON-serializable CPU / memory snapshot for the current process.
    If cpu_proc is the return value of _prime_psutil_cpu_meter(), includes cpu_percent
    since that call (wall interval covering the video when primed at video start).
    """
    out: dict = {
        "pid": os.getpid(),
        "logical_cpu_count": os.cpu_count(),
        "torch_intraop_threads": int(torch.get_num_threads()),
        "torch_interop_threads": int(torch.get_num_interop_threads()),
    }
    if torch.cuda.is_available():
        out["torch_cuda_allocated_gib"] = round(
            float(torch.cuda.memory_allocated()) / (1024.0**3), 3
        )
        out["torch_cuda_reserved_gib"] = round(
            float(torch.cuda.memory_reserved()) / (1024.0**3), 3
        )
    if cpu_proc is not None:
        try:
            with cpu_proc.oneshot():
                mi = cpu_proc.memory_info()
                out["rss_gib"] = round(mi.rss / (1024.0**3), 3)
                out["vms_gib"] = round(mi.vms / (1024.0**3), 3)
                out["threads"] = int(cpu_proc.num_threads() or 0)
                try:
                    out["open_files_count"] = len(cpu_proc.open_files())
                except Exception:
                    out["open_files_count"] = None
                try:
                    out["num_fds"] = int(cpu_proc.num_fds())
                except Exception:
                    out["num_fds"] = None
            out["cpu_percent_since_video_start"] = round(float(cpu_proc.cpu_percent()), 2)
        except Exception as e:
            out["psutil_error"] = str(e)
    else:
        out["psutil"] = "unavailable_or_failed_import"
    return out


# ── Colour palette ────────────────────────────────────────────────────────
def color_from_text(text: str) -> tuple:
    rng = np.random.default_rng(abs(hash(text)) % (2**32))
    return tuple(int(x) for x in rng.integers(60, 230, size=3))

print('✅ Utilities loaded')

def load_models() -> None:
    """Load Qwen + SAM3. Sets module-level qwen_*, sam3_*, device, dtypes."""
    global qwen_model, qwen_processor, sam3_model, sam3_processor
    global device, qwen_dtype, sam3_dtype

    import transformers
    from transformers import (
        Qwen2_5_VLForConditionalGeneration,
        AutoProcessor,
        Sam3VideoModel,
        Sam3VideoProcessor,
    )

    device     = "cuda" if torch.cuda.is_available() else "cpu"
    qwen_dtype = torch.bfloat16 if device == "cuda" else torch.float32
    sam3_dtype = torch.bfloat16 if device == "cuda" else torch.float32

    print("transformers version:", transformers.__version__)
    print(f"Device : {device}")
    if device == "cpu":
        # inter_op must be set before intra_op (see torch.set_num_interop_threads)
        _ti = globals().get("CPU_TORCH_THREADS")
        _te = globals().get("CPU_INTEROP_THREADS")
        if _te is not None:
            torch.set_num_interop_threads(int(_te))
        if _ti is not None:
            torch.set_num_threads(int(_ti))
        print(
            f"CPU thread pools: intra_op={torch.get_num_threads()}  inter_op={torch.get_num_interop_threads()}"
        )
    if device == "cuda":
        print(f"GPU    : {torch.cuda.get_device_name(0)}")
        print(f"VRAM   : {torch.cuda.get_device_properties(0).total_memory/1e9:.1f} GB")

    # Qwen
    print("\nLoading Qwen...")
    with Timer("Qwen load"):
        qwen_model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            QWEN_MODEL_ID,
            torch_dtype=qwen_dtype,
            device_map="auto",
        )
        qwen_processor = AutoProcessor.from_pretrained(QWEN_MODEL_ID)

    # SAM3 Video
    print("\nLoading SAM3 Video...")
    with Timer("SAM3 load"):
        sam3_model = Sam3VideoModel.from_pretrained(
            SAM3_MODEL_ID
        ).to(device=device, dtype=sam3_dtype)
        sam3_processor = Sam3VideoProcessor.from_pretrained(SAM3_MODEL_ID)

    # Workaround: run_detection does get_text_features(..., return_dict=True).pooler_output.
    # Some builds return a raw tensor; instance-only monkeypatches are easy to miss—patch the
    # detector class, idempotently, and keep a ref to the true original on the class.
    from types import SimpleNamespace

    _GET_TEXT_UNPATCH = "_orig_get_text_features_sam3"


    def _normalize_sam3_text_out(out):
        if torch.is_tensor(out) or isinstance(out, torch.Tensor):
            return SimpleNamespace(pooler_output=out)
        if hasattr(out, "pooler_output") and getattr(out, "pooler_output", None) is not None:
            return out
        if hasattr(out, "text_embeds") and getattr(out, "text_embeds", None) is not None:
            return SimpleNamespace(pooler_output=out.text_embeds)
        raise TypeError(f"Unexpected get_text_features return type: {type(out)}")


    if hasattr(sam3_model, "detector_model") and hasattr(sam3_model.detector_model, "get_text_features"):
        # Older notebook versions patched only the instance; that shadows the class—remove it
        if "get_text_features" in sam3_model.detector_model.__dict__:
            delattr(sam3_model.detector_model, "get_text_features")
        _dm_cls = type(sam3_model.detector_model)
        if not hasattr(_dm_cls, _GET_TEXT_UNPATCH):
            setattr(_dm_cls, _GET_TEXT_UNPATCH, _dm_cls.get_text_features)
        _unpatched = getattr(_dm_cls, _GET_TEXT_UNPATCH)

        def _get_text_features_fixed(self, *args, **kwargs):
            return _normalize_sam3_text_out(_unpatched(self, *args, **kwargs))

        _dm_cls.get_text_features = _get_text_features_fixed
        print("✅ Applied SAM3 text-feature compatibility patch (class-level, idempotent)")

    print("\n✅ Models ready")


def apply_config(cfg: dict) -> None:
    """Set module-level options (same names as the notebook Step 3 globals)."""
    g = globals()
    for k, v in cfg.items():
        g[k] = v

try:
    from qwen_vl_utils import process_vision_info
    from qwen_vl_utils.vision_process import smart_nframes
except ModuleNotFoundError as e:  # pragma: no cover
    raise ModuleNotFoundError(
        "Missing package for Qwen2.5-VL. Install: pip install qwen-vl-utils"
    ) from e


def _probe_video_fps(path: str) -> float:
    """OpenCV probe when torchvision.io.read_video omits info['video_fps'] (common for some .avi)."""
    p = str(path)
    if p.startswith("file://"):
        p = p[7:]
    cap = cv2.VideoCapture(p)
    if not cap.isOpened():
        return 25.0
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    cap.release()
    return fps if fps > 0 else 25.0


def _patch_torchvision_read_video_for_qwen_vl() -> None:
    """qwen_vl_utils assumes info['video_fps'] exists; torchvision may omit it → KeyError."""
    try:
        from torchvision import io as _tv_io
    except Exception:
        return
    if getattr(_tv_io, "_qwen_vl_video_fps_patch", False):
        return
    _raw = _tv_io.read_video

    def _read_video_patched(path, *a, **kw):
        video, audio, info = _raw(path, *a, **kw)
        if isinstance(info, dict) and (info.get("video_fps") in (None, 0, 0.0) or "video_fps" not in info):
            info = {**info, "video_fps": _probe_video_fps(str(path))}
        return video, audio, info

    _read_video_patched._qwen_vl_unpatched_read_video = _raw
    _tv_io.read_video = _read_video_patched
    _tv_io._qwen_vl_video_fps_patch = True


_patch_torchvision_read_video_for_qwen_vl()

def _qwen_pil_frames_from_rgb(
    frames: list, video_fps: float, target_fps: float = 1.0
) -> tuple[list[Image.Image], float]:
    """Pick frames for Qwen using qwen-vl-utils smart_nframes + linspace (same as file decode path)."""
    n_tot = len(frames)
    vf = float(video_fps) if video_fps and float(video_fps) > 0 else 25.0
    ele_cfg = {"fps": float(target_fps)}
    if n_tot == 0:
        raise ValueError("no frames for Qwen video input")
    try:
        n_sample = int(smart_nframes(ele_cfg, total_frames=n_tot, video_fps=vf))
    except ValueError:
        n_sample = 2 if n_tot >= 2 else 1
    n_sample = max(1, min(n_sample, n_tot))
    pos = np.linspace(0, n_tot - 1, num=n_sample, dtype=np.float64)
    idx = np.clip(np.round(pos).astype(np.int64), 0, n_tot - 1).tolist()
    pil_list = [
        Image.fromarray(np.ascontiguousarray(frames[i], dtype=np.uint8), mode="RGB")
        for i in idx
    ]
    sample_fps = len(pil_list) / max(n_tot, 1e-6) * vf
    return pil_list, float(sample_fps)


# ── Qwen: extract dominant object prompts from a video ────────────────────
def run_qwen_video(frames_rgb: list, video_fps: float, top_n: int) -> tuple[dict, list[str]]:
    """Use OpenCV-decoded frames as a frame list so Qwen never relies on torchvision/decord on file."""
    v_fps = float(video_fps) if video_fps and float(video_fps) > 0 else 25.0
    pil_frames, sample_fps = _qwen_pil_frames_from_rgb(frames_rgb, v_fps, target_fps=1.0)
    messages = [{
        "role": "user",
        "content": [
            {
                "type": "video",
                "video": pil_frames,
                "fps": 1.0,
                "video_fps": v_fps,
                "sample_fps": sample_fps,
            },
            {"type": "text", "text": (
                f"Identify the top {top_n} dominant physical objects seen in this video. "
                "Prefer concrete visible objects, not events or abstract concepts. "
                "Use short noun phrases suitable as segmentation prompts. "
                "Return STRICT JSON only with this schema:\n"
                "{\n"
                '  "dominant_objects": [\n'
                '    {"name": "short noun phrase", "reason": "short reason", "frequency": "high|medium|low"}\n'
                "  ]\n"
                "}\n"
                f"Return at most {top_n} objects."
            )},
        ],
    }]

    text = qwen_processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = qwen_processor(
        text=[text], images=image_inputs, videos=video_inputs,
        padding=True, return_tensors="pt"
    ).to(qwen_model.device)

    gen_ids = qwen_model.generate(**inputs, max_new_tokens=256, do_sample=False)
    trimmed = [o[len(i):] for i, o in zip(inputs.input_ids, gen_ids)]
    raw_text = qwen_processor.batch_decode(trimmed, skip_special_tokens=True,
                                           clean_up_tokenization_spaces=False)[0]

    qwen_json = extract_json_block(raw_text)
    prompts   = []
    for item in qwen_json.get("dominant_objects", [])[:top_n]:
        name = str(item.get("name", "")).strip().lower()
        if name and name not in prompts:
            prompts.append(name)

    return qwen_json, prompts

# ── SAM3: segment all frames with given prompts ───────────────────────────
def run_sam3(video_frames: list, prompts: list) -> dict:
    """Returns outputs_per_frame: {frame_idx: processed_result}"""
    assert len(prompts) > 0, "No prompts to segment"

    sam_session = sam3_processor.init_video_session(
        video=video_frames,
        inference_device=device,
        processing_device="cpu",
        video_storage_device="cpu",
        dtype=sam3_dtype,
    )

    sam_session = sam3_processor.add_text_prompt(
        inference_session=sam_session,
        text=prompts,   # list of prompts is supported
    )

    outputs_per_frame = {}
    for model_outputs in sam3_model.propagate_in_video_iterator(
        inference_session=sam_session
    ):
        processed = sam3_processor.postprocess_outputs(sam_session, model_outputs)
        outputs_per_frame[model_outputs.frame_idx] = processed

    return outputs_per_frame


# ── Qwen: one image (qwen_sam3_image_pipeline parity) ────────────────────
def run_qwen_image(image_path: str, top_n: int) -> tuple[dict, list[str]]:
    messages = [{
        "role": "user",
        "content": [
            {"type": "image", "image": f"file://{image_path}"},
            {"type": "text", "text": (
                f"Identify the top {top_n} dominant physical objects seen in this image. "
                "Prefer concrete visible objects, not events or abstract concepts. "
                "Use short noun phrases suitable as segmentation prompts. "
                "Return STRICT JSON only with this schema:\n"
                "{\n"
                '  "dominant_objects": [\n'
                '    {"name": "short noun phrase", "reason": "short reason", "frequency": "high|medium|low"}\n'
                "  ]\n"
                "}\n"
                f"Return at most {top_n} objects."
            )},
        ],
    }]

    text = qwen_processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = qwen_processor(
        text=[text], images=image_inputs, videos=video_inputs,
        padding=True, return_tensors="pt"
    ).to(qwen_model.device)

    gen_ids = qwen_model.generate(**inputs, max_new_tokens=256, do_sample=False)
    trimmed = [o[len(i):] for i, o in zip(inputs.input_ids, gen_ids)]
    raw_text = qwen_processor.batch_decode(trimmed, skip_special_tokens=True,
                                           clean_up_tokenization_spaces=False)[0]

    qwen_json = extract_json_block(raw_text)
    prompts   = []
    for item in qwen_json.get("dominant_objects", [])[:top_n]:
        name = str(item.get("name", "")).strip().lower()
        if name and name not in prompts:
            prompts.append(name)
    return qwen_json, prompts


# ── SAM3: single image via 1-frame session (image pipeline parity) ─
def run_sam3_image(image_frame, prompts: list) -> dict:
    assert len(prompts) > 0, "No prompts to segment"
    sam_session = sam3_processor.init_video_session(
        video=[image_frame],
        inference_device=device,
        processing_device="cpu",
        video_storage_device="cpu",
        dtype=sam3_dtype,
    )
    sam_session = sam3_processor.add_text_prompt(
        inference_session=sam_session,
        text=prompts,
    )
    result = None
    for model_outputs in sam3_model.propagate_in_video_iterator(
        inference_session=sam_session
    ):
        result = sam3_processor.postprocess_outputs(sam_session, model_outputs)
    return result


print('✅ Inference functions ready')
# ── Tensor → numpy helper ─────────────────────────────────────────────────
def to_np(x):
    return x.cpu().numpy() if hasattr(x, "cpu") else np.asarray(x)

# ── Visualisation helpers ─────────────────────────────────────────────────
def overlay_mask(image_bgr, mask_bool, color, alpha=0.40):
    out = image_bgr.copy()
    out[mask_bool] = (
        out[mask_bool] * (1 - alpha) + np.array(color, dtype=np.float32) * alpha
    ).astype(np.uint8)
    return out

def draw_box_label(img, box_xyxy, label, color):
    x1, y1, x2, y2 = map(int, box_xyxy)
    cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
    cv2.putText(img, label, (x1, max(20, y1-8)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2, cv2.LINE_AA)

# ── Bounding-box label writer ─────────────────────────────────────────────
def resolve_yolo_class_id(prompt: str) -> int | None:
    """Map Qwen/SAM3 prompt → index in YOLO_CLASS_NAMES (None if unmapped)."""
    raw = prompt.strip()
    lk = raw.lower()
    mapped = None
    for ak, av in YOLO_CLASS_ALIASES.items():
        if ak.lower() == lk:
            mapped = av
            break
    candidate = (mapped if mapped is not None else raw).strip()
    cl = candidate.lower()
    for i, name in enumerate(YOLO_CLASS_NAMES):
        if name.lower() == cl:
            return i
    return None


def save_bbox_labels(frame_idx: int, result: dict, prompts: list,
                     label_dir: Path, img_hw: tuple, fmt: str = "yolo"):
    """
    One .txt per frame. YOLO class id = YOLO_CLASS_NAMES index (not enumerate(prompts)).
    Default 5 cols (class + cx cy w h); optional confidence column.
    """
    H, W = img_hw
    masks = to_np(result["masks"])
    boxes = to_np(result["boxes"])
    scores = to_np(result["scores"])
    obj_ids = to_np(result["object_ids"])
    p2obj = result["prompt_to_obj_ids"]

    obj_idx = {int(oid): i for i, oid in enumerate(obj_ids)}
    lines = []
    meta = []

    for prompt in prompts:
        yolo_cid = resolve_yolo_class_id(prompt)
        if yolo_cid is None:
            if YOLO_SKIP_UNMAPPED_PROMPTS:
                continue
            yolo_cid = prompts.index(prompt)

        raw_ids = to_np(p2obj.get(prompt, []))
        for obj_id in [int(x) for x in raw_ids]:
            if obj_id not in obj_idx:
                continue
            i = obj_idx[obj_id]
            x1, y1, x2, y2 = boxes[i]
            score = float(scores[i])

            if fmt == "yolo":
                cx = ((x1 + x2) / 2) / W
                cy = ((y1 + y2) / 2) / H
                bw = (x2 - x1) / W
                bh = (y2 - y1) / H
                row = f"{yolo_cid} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}"
                if YOLO_LABEL_APPEND_CONFIDENCE:
                    row += f" {score:.4f}"
                lines.append(row)
            else:
                row = f"{yolo_cid} {x1:.1f} {y1:.1f} {x2-x1:.1f} {y2-y1:.1f}"
                if YOLO_LABEL_APPEND_CONFIDENCE:
                    row += f" {score:.4f}"
                lines.append(row)

            meta.append({
                "frame": frame_idx,
                "yolo_class_id": yolo_cid,
                "yolo_class_name": (
                    YOLO_CLASS_NAMES[yolo_cid]
                    if 0 <= yolo_cid < len(YOLO_CLASS_NAMES)
                    else None
                ),
                "prompt": prompt,
                "sam3_obj_id": obj_id,
                "score": score,
                "box_xyxy": [float(x1), float(y1), float(x2), float(y2)],
            })

    txt_path = label_dir / f"{frame_idx:06d}.txt"
    txt_path.write_text("\n".join(lines))
    return meta


def write_empty_frame_labels(frame_idx: int, label_boxes: Path, label_segments: Path) -> None:
    """YOLO-style empty label files when SAM3 has no output for this frame index."""
    stem = f"{frame_idx:06d}"
    (label_boxes / f"{stem}.txt").write_text("")
    (label_segments / f"{stem}.txt").write_text("")


def mask_to_normalized_polygon(mask_hw: np.ndarray, W: int, H: int) -> list[float] | None:
    m = np.asarray(mask_hw)
    if m.dtype != np.uint8:
        m = (m > 0.5).astype(np.uint8)
    else:
        m = (m > 127).astype(np.uint8)
    m = (m * 255).astype(np.uint8)
    if float(m.sum()) < YOLO_SEG_MIN_MASK_AREA:
        return None
    contours, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    cnt = max(contours, key=cv2.contourArea)
    if cv2.contourArea(cnt) < YOLO_SEG_MIN_MASK_AREA:
        return None
    peri = cv2.arcLength(cnt, True)
    eps = max(YOLO_SEG_POLY_EPS_RATIO * peri, 1e-6)
    cnt = cv2.approxPolyDP(cnt, eps, True)
    if len(cnt) < 3:
        return None
    pts = cnt.reshape(-1, 2).astype(np.float64)
    pts[:, 0] = np.clip(pts[:, 0] / float(W), 0.0, 1.0)
    pts[:, 1] = np.clip(pts[:, 1] / float(H), 0.0, 1.0)
    flat = pts.flatten()
    if flat.size < 6:
        return None
    return flat.tolist()


def save_yolo_segment_labels(
    frame_stem: str, result: dict, prompts: list, segment_dir: Path, img_hw: tuple
):
    """One polygon label file per frame: class x1 y1 x2 y2 ... (normalized)."""
    H, W = img_hw
    masks = to_np(result["masks"])
    boxes = to_np(result["boxes"])
    scores = to_np(result["scores"])
    obj_ids = to_np(result["object_ids"])
    p2obj = result["prompt_to_obj_ids"]
    obj_idx = {int(oid): i for i, oid in enumerate(obj_ids)}
    lines = []
    meta_seg = []

    for prompt in prompts:
        yolo_cid = resolve_yolo_class_id(prompt)
        if yolo_cid is None:
            if YOLO_SKIP_UNMAPPED_PROMPTS:
                continue
            yolo_cid = prompts.index(prompt)

        raw_ids = to_np(p2obj.get(prompt, []))
        for obj_id in [int(x) for x in raw_ids]:
            if obj_id not in obj_idx:
                continue
            i = obj_idx[obj_id]
            poly = mask_to_normalized_polygon(masks[i], W, H)
            if poly is None:
                continue
            row = f"{yolo_cid} " + " ".join(f"{p:.6f}" for p in poly)
            lines.append(row)
            x1, y1, x2, y2 = boxes[i]
            meta_seg.append({
                "frame_stem": frame_stem,
                "yolo_class_id": yolo_cid,
                "prompt": prompt,
                "sam3_obj_id": obj_id,
                "score": float(scores[i]),
                "polygon_points": len(poly) // 2,
                "box_xyxy": [float(x1), float(y1), float(x2), float(y2)],
            })

    out = segment_dir / f"{frame_stem}.txt"
    out.write_text("\n".join(lines))
    return meta_seg

# ── Binary mask writer ────────────────────────────────────────────────────
def save_mask_pngs(frame_idx: int, result: dict, prompts: list, mask_dir: Path):
    """Saves one binary PNG per (prompt, obj_id) per frame."""
    masks   = to_np(result["masks"])
    obj_ids = to_np(result["object_ids"])
    p2obj   = result["prompt_to_obj_ids"]
    obj_idx = {int(oid): i for i, oid in enumerate(obj_ids)}

    paths = []
    for prompt in prompts:
        raw_ids = to_np(p2obj.get(prompt, []))
        slug    = prompt.replace(" ", "_")
        for obj_id in [int(x) for x in raw_ids]:
            if obj_id not in obj_idx:
                continue
            i    = obj_idx[obj_id]
            m    = (masks[i].astype(np.uint8)) * 255
            fpath = mask_dir / f"{frame_idx:06d}_{slug}_obj{obj_id}.png"
            cv2.imwrite(str(fpath), m)
            paths.append(str(fpath))
    return paths


def build_overlay_bgr(
    frame_rgb: np.ndarray,
    result: dict | None,
    prompts: list[str],
) -> np.ndarray:
    """BGR image: same layout as preview MP4 frames (masks + boxes + legend). Plain frame if result is None."""
    vis = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
    if result is None or not prompts:
        return vis
    masks = to_np(result["masks"])
    boxes = to_np(result["boxes"])
    scores = to_np(result["scores"])
    obj_ids = to_np(result["object_ids"])
    p2obj = result["prompt_to_obj_ids"]
    oidx = {int(oid): i for i, oid in enumerate(obj_ids)}
    colors = {p: color_from_text(p) for p in prompts}
    for prompt in prompts:
        color = colors[prompt]
        for obj_id in [int(x) for x in to_np(p2obj.get(prompt, []))]:
            if obj_id not in oidx:
                continue
            i = oidx[obj_id]
            vis = overlay_mask(vis, masks[i].astype(bool), color, 0.35)
            draw_box_label(
                vis, boxes[i], f"{prompt}|id={obj_id}|{float(scores[i]):.2f}", color
            )
    for y_off, prompt in enumerate(prompts, start=1):
        cv2.putText(
            vis,
            prompt,
            (10, 28 * y_off),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            colors[prompt],
            2,
            cv2.LINE_AA,
        )
    return vis


# ── Combined visualisation video ──────────────────────────────────────────
def render_combined_video(video_frames, outputs_per_frame, prompts, fps, out_path):
    h, w     = video_frames[0].shape[:2]
    writer   = cv2.VideoWriter(str(out_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    colors   = {p: color_from_text(p) for p in prompts}

    for fidx in sorted(outputs_per_frame):
        vis    = cv2.cvtColor(video_frames[fidx], cv2.COLOR_RGB2BGR)
        result = outputs_per_frame[fidx]
        masks  = to_np(result["masks"])
        boxes  = to_np(result["boxes"])
        scores = to_np(result["scores"])
        obj_ids = to_np(result["object_ids"])
        p2obj  = result["prompt_to_obj_ids"]
        oidx   = {int(oid): i for i, oid in enumerate(obj_ids)}

        for prompt in prompts:
            color = colors[prompt]
            for obj_id in [int(x) for x in to_np(p2obj.get(prompt, []))]:
                if obj_id not in oidx: continue
                i = oidx[obj_id]
                vis = overlay_mask(vis, masks[i].astype(bool), color, 0.35)
                draw_box_label(vis, boxes[i],
                               f"{prompt}|id={obj_id}|{float(scores[i]):.2f}", color)

        for y_off, prompt in enumerate(prompts, start=1):   # legend
            cv2.putText(vis, prompt, (10, 28*y_off),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, colors[prompt], 2, cv2.LINE_AA)
        writer.write(vis)
    writer.release()

# ── Per-prompt visualisation videos ──────────────────────────────────────
def render_per_prompt_videos(video_frames, outputs_per_frame, prompts, fps, video_dir):
    h, w   = video_frames[0].shape[:2]
    colors = {p: color_from_text(p) for p in prompts}
    writers = {
        p: cv2.VideoWriter(
            str(video_dir / f"sam3_{p.replace(' ','_')}.mp4"),
            cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h)
        ) for p in prompts
    }

    for fidx in sorted(outputs_per_frame):
        base   = cv2.cvtColor(video_frames[fidx], cv2.COLOR_RGB2BGR)
        result = outputs_per_frame[fidx]
        masks  = to_np(result["masks"])
        boxes  = to_np(result["boxes"])
        scores = to_np(result["scores"])
        obj_ids = to_np(result["object_ids"])
        p2obj  = result["prompt_to_obj_ids"]
        oidx   = {int(oid): i for i, oid in enumerate(obj_ids)}

        for prompt in prompts:
            vis   = base.copy()
            color = colors[prompt]
            for obj_id in [int(x) for x in to_np(p2obj.get(prompt, []))]:
                if obj_id not in oidx: continue
                i = oidx[obj_id]
                vis = overlay_mask(vis, masks[i].astype(bool), color, 0.40)
                draw_box_label(vis, boxes[i],
                               f"{prompt}|id={obj_id}|{float(scores[i]):.2f}", color)
            cv2.putText(vis, f"Prompt: {prompt}", (10, 32),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2, cv2.LINE_AA)
            writers[prompt].write(vis)

    for w in writers.values():
        w.release()

print('✅ Export functions ready')
def process_video(
    video_path: Path, output_root: Path,
    top_n: int, max_frames, fps_override,
    bbox_fmt: str, pipeline_mode: str | None = None,
) -> dict:
    """
    Full pipeline for one video.
    pipeline_mode: None -> use global VIDEO_PIPELINE_MODE
      - "video_session" — one Qwen on the video file, SAM3 video session (default).
      - "per_frame_image" — write frame JPEGs, then each frame: Qwen(single image)+SAM3(1 frame).
      - "qwen_per_frame_then_agg_sam3" — Qwen on each frame, vote top QWEN_AGGREGATE_TOP_K, one SAM3 video.
    """
    mode = (pipeline_mode if pipeline_mode is not None
            else globals().get("VIDEO_PIPELINE_MODE", "video_session"))
    v_stride = int(globals().get("VIDEO_EXPORT_FRAME_STRIDE", 1) or 1)
    p_every = int(globals().get("PER_FRAME_PROGRESS_EVERY", 0) or 0)
    t_cap = int(globals().get("PER_FRAME_TIMING_MAX_ROWS", 0) or 0)
    ynames = YOLO_CLASS_NAMES
    yalias = YOLO_CLASS_ALIASES

    log = {
        "video_path":     str(video_path),
        "video_stem":     video_path.stem,
        "started_at":     datetime.datetime.now().isoformat(),
        "status":         "started",
        "error":          None,
        "pipeline_mode":  mode,
        "frames_total":   0,
        "frames_processed": 0,
        "prompts":        [],
        "t_read_s":       0.0,
        "t_write_frames_s": 0.0,
        "t_qwen_s":       0.0,
        "t_sam3_s":       0.0,
        "t_export_s":     0.0,
        "t_overlay_s":    0.0,
        "t_total_s":      0.0,
        "t_mean_wall_s_per_frame": None,
        "per_frame_timings": None,
        "seconds_per_frame_mean": None,
        "output_dir":     "",
        "resources_after": None,
    }
    wall_start = time.perf_counter()
    per_frame_prompts = None
    outputs_per_frame = None
    extract_first = bool(globals().get("EXTRACT_FRAMES_TO_DISK_FIRST", True))
    cpu_meter_proc = _prime_psutil_cpu_meter()

    try:
        dirs = make_video_output_dir(output_root, video_path)
        log["output_dir"] = str(dirs["base"])

        print(f"\n{'='*60}")
        print(f"VIDEO : {video_path.name}  (mode={mode})")
        print(f"{'='*60}")
        with Timer("Read video") as t:
            frames, meta = read_video_opencv(str(video_path), max_frames)
        log["t_read_s"] = t.elapsed
        log["frames_total"] = len(frames)
        n_frames = len(frames)
        fps = fps_override if fps_override else meta["fps"]
        print(f"  Frames: {n_frames}, FPS: {fps:.2f}")

        if n_frames == 0:
            raise RuntimeError("No frames decoded from video")
        if mode not in ("video_session", "per_frame_image", "qwen_per_frame_then_agg_sam3"):
            raise ValueError(f"Unknown pipeline_mode: {mode!r}")

        _cls_txt = "\n".join(ynames)
        (dirs["label_boxes"] / "classes.txt").write_text(_cls_txt)
        (dirs["label_segments"] / "classes.txt").write_text(_cls_txt)

        if mode == "per_frame_image":
            if extract_first:
                with Timer("Write all frames to images/frames") as tw:
                    for fidx, fr in enumerate(frames):
                        bgr = cv2.cvtColor(fr, cv2.COLOR_RGB2BGR)
                        cv2.imwrite(str(dirs["images_frames"] / f"{fidx:06d}.jpg"), bgr)
                log["t_write_frames_s"] = tw.elapsed
            else:
                log["t_write_frames_s"] = 0.0

            outputs_per_frame = {}
            per_frame_prompts = {}
            qwen_rows = []
            t_qwen_acc = 0.0
            t_sam_acc = 0.0
            per_frame_timings = []
            all_prompts_union = set()
            n_proc = 0

            for fidx in range(n_frames):
                if v_stride > 1 and (fidx % v_stride) != 0:
                    continue
                stem = f"{fidx:06d}"
                jpg_path = dirs["images_frames"] / f"{stem}.jpg"
                if not extract_first:
                    bgr = cv2.cvtColor(frames[fidx], cv2.COLOR_RGB2BGR)
                    cv2.imwrite(str(jpg_path), bgr)

                t0 = time.perf_counter()
                qj, f_prompts = run_qwen_image(str(jpg_path.resolve()), top_n)
                tq = time.perf_counter() - t0
                t_qwen_acc += tq

                if not f_prompts:
                    print(f"  \u26a0\ufe0f  frame {fidx}: no Qwen prompts; using placeholder 'object'")
                    f_prompts = ["object"]
                per_frame_prompts[fidx] = list(f_prompts)
                all_prompts_union.update(f_prompts)

                t0 = time.perf_counter()
                result = run_sam3_image(frames[fidx], f_prompts)
                ts = time.perf_counter() - t0
                t_sam_acc += ts

                outputs_per_frame[fidx] = result
                n_proc += 1
                row = {
                    "frame": fidx,
                    "t_qwen_s": round(tq, 4),
                    "t_sam3_s": round(ts, 4),
                    "t_frame_s": round(tq + ts, 4),
                    "prompts": list(f_prompts),
                }
                qwen_rows.append({**row, "qwen_raw": qj})
                if t_cap == 0 or len(per_frame_timings) < t_cap:
                    per_frame_timings.append(dict(row))
                if p_every and n_proc % p_every == 0:
                    print(
                        f"  ... per_frame_image: {n_proc} frame(s)  @ {fidx}  "
                        f"qwen={tq:.2f}s  sam3={ts:.2f}s  p={f_prompts[:3]}"
                    )

            log["frames_processed"] = n_proc
            log["t_qwen_s"] = t_qwen_acc
            log["t_sam3_s"] = t_sam_acc
            log["per_frame_timings"] = per_frame_timings
            if n_proc:
                log["seconds_per_frame_mean"] = (t_qwen_acc + t_sam_acc) / n_proc
            log["prompts"] = sorted(all_prompts_union)

            (dirs["base"] / "qwen_outputs_per_frame.json").write_text(
                json.dumps(qwen_rows, indent=2)
            )
            (dirs["base"] / "qwen_output.json").write_text(
                json.dumps(
                    {
                        "pipeline_mode": "per_frame_image",
                        "detail": "qwen_outputs_per_frame.json (full per-frame Qwen+timing+prompts)",
                    },
                    indent=2,
                )
            )
            m = log.get("seconds_per_frame_mean") or 0.0
            print(f"  per_frame_image: {n_proc} frame(s); mean (Qwen+SAM3) / frame = {m:.2f}s")

        elif mode == "qwen_per_frame_then_agg_sam3":
            q_ptop = int(globals().get("QWEN_PER_FRAME_TOP_N", top_n))
            agg_k = int(globals().get("QWEN_AGGREGATE_TOP_K", 4))

            if extract_first:
                with Timer("Write all frames to images/frames") as tw:
                    for fidx, fr in enumerate(frames):
                        bgr = cv2.cvtColor(fr, cv2.COLOR_RGB2BGR)
                        cv2.imwrite(str(dirs["images_frames"] / f"{fidx:06d}.jpg"), bgr)
                log["t_write_frames_s"] = tw.elapsed
            else:
                log["t_write_frames_s"] = 0.0

            vote = Counter()
            qwen_rows = []
            t_qwen_acc = 0.0
            per_frame_timings = []

            for fidx in range(n_frames):
                stem = f"{fidx:06d}"
                jpg_path = dirs["images_frames"] / f"{stem}.jpg"
                if not extract_first:
                    bgr = cv2.cvtColor(frames[fidx], cv2.COLOR_RGB2BGR)
                    cv2.imwrite(str(jpg_path), bgr)

                t0 = time.perf_counter()
                qj, f_prompts = run_qwen_image(str(jpg_path.resolve()), q_ptop)
                tq = time.perf_counter() - t0
                t_qwen_acc += tq
                for p in f_prompts:
                    vote[p] += 1
                qwen_rows.append(
                    {
                        "frame": fidx,
                        "t_qwen_s": round(tq, 4),
                        "prompts": list(f_prompts),
                        "qwen_raw": qj,
                    }
                )
                if t_cap == 0 or len(per_frame_timings) < t_cap:
                    per_frame_timings.append(
                        {"frame": fidx, "t_qwen_s": round(tq, 4), "prompts": list(f_prompts)}
                    )
                if p_every and (fidx + 1) % p_every == 0:
                    print(
                        f"  ... qwen_per_frame_then_agg: {fidx+1}/{n_frames}  "
                        f"qwen={tq:.2f}s  p={f_prompts[:3]}"
                    )

            log["t_qwen_s"] = t_qwen_acc
            log["per_frame_timings"] = per_frame_timings
            if n_frames:
                log["seconds_per_frame_mean"] = t_qwen_acc / n_frames
            log["qwen_prompt_vote_counts"] = dict(vote)
            log["qwen_aggregate_top_k"] = agg_k

            ranked = sorted(vote.items(), key=lambda x: (-x[1], x[0]))
            final_prompts = [name for name, _c in ranked[:agg_k]]
            if not final_prompts:
                print("  \u26a0\ufe0f  no Qwen prompts across frames; using placeholder 'object'")
                final_prompts = ["object"]

            log["prompts"] = final_prompts
            log["frames_processed"] = n_frames
            log["qwen_ranked_prompts"] = [{"name": n, "frames": c} for n, c in ranked]

            (dirs["base"] / "qwen_outputs_per_frame.json").write_text(
                json.dumps(qwen_rows, indent=2)
            )
            (dirs["base"] / "qwen_vote_summary.json").write_text(
                json.dumps(
                    {
                        "aggregate_top_k": agg_k,
                        "per_frame_top_n": q_ptop,
                        "vote_counts": log["qwen_prompt_vote_counts"],
                        "ranked": log["qwen_ranked_prompts"],
                        "final_prompts_for_sam3": final_prompts,
                    },
                    indent=2,
                )
            )
            (dirs["base"] / "qwen_output.json").write_text(
                json.dumps(
                    {
                        "pipeline_mode": "qwen_per_frame_then_agg_sam3",
                        "final_prompts": final_prompts,
                        "vote_summary_file": "qwen_vote_summary.json",
                    },
                    indent=2,
                )
            )
            print(f"  Aggregated prompts for SAM3 ({len(final_prompts)}): {final_prompts}")
            print(f"  vote leaders: {ranked[:8]}")

            with Timer("SAM3 inference") as t:
                outputs_per_frame = run_sam3(frames, final_prompts)
            log["t_sam3_s"] = t.elapsed
            print(f"  Segmented frames: {len(outputs_per_frame)}")

        else:
            with Timer("Qwen inference") as t:
                qwen_json, prompts = run_qwen_video(frames, fps, top_n)

            log["t_qwen_s"] = t.elapsed
            log["prompts"] = list(prompts)
            log["frames_processed"] = n_frames
            print(f"  Prompts: {prompts}")

            (dirs["base"] / "qwen_output.json").write_text(
                json.dumps({"qwen_raw": qwen_json, "prompts": list(prompts)}, indent=2)
            )

            with Timer("SAM3 inference") as t:
                outputs_per_frame = run_sam3(frames, prompts)
            log["t_sam3_s"] = t.elapsed
            print(f"  Segmented frames: {len(outputs_per_frame)}")

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        if mode == "per_frame_image" and per_frame_prompts is not None:
            pr_union = sorted({p for v in per_frame_prompts.values() for p in v})
        else:
            pr_union = list(log["prompts"])

        skip_jpg_rewrite = extract_first and mode in ("per_frame_image", "qwen_per_frame_then_agg_sam3")
        export_preview_mp4s = bool(globals().get("VIDEO_EXPORT_PREVIEW_MP4S", False))
        export_overlay_images = bool(globals().get("VIDEO_EXPORT_OVERLAY_IMAGES", True))

        with Timer("Export") as t:
            H, W = frames[0].shape[:2]
            all_meta = []
            all_seg_meta = []
            # One row per input frame: JPEG + labels (empty .txt if SAM3 did not return that frame).
            for fidx in range(n_frames):
                if v_stride > 1 and (fidx % v_stride) != 0:
                    continue
                stem = f"{fidx:06d}"
                if not skip_jpg_rewrite:
                    bgr = cv2.cvtColor(frames[fidx], cv2.COLOR_RGB2BGR)
                    cv2.imwrite(str(dirs["images_frames"] / f"{stem}.jpg"), bgr)
                f_prompts = per_frame_prompts[fidx] if per_frame_prompts is not None else list(
                    log["prompts"]
                )
                result = outputs_per_frame.get(fidx)
                if result is None:
                    write_empty_frame_labels(fidx, dirs["label_boxes"], dirs["label_segments"])
                    continue

                frame_meta = save_bbox_labels(
                    fidx, result, f_prompts, dirs["label_boxes"], (H, W), bbox_fmt
                )
                seg_m = save_yolo_segment_labels(
                    stem, result, f_prompts, dirs["label_segments"], (H, W)
                )
                save_mask_pngs(fidx, result, f_prompts, dirs["label_masks"])
                all_meta.extend(frame_meta)
                all_seg_meta.extend([{**m, "frame": fidx} for m in seg_m])

            tmpl_det = (
                "# Ultralytics detection — set path / train / val after merging runs\n"
                f"path: .\ntrain: images/train\nval: images/val\n"
                f"nc: {len(ynames)}\nnames:\n"
            )
            for i, n in enumerate(ynames):
                tmpl_det += f"  {i}: {n}\n"
            (dirs["label_boxes"] / "yolo_dataset_template.yaml").write_text(tmpl_det)

            tmpl_seg = (
                "# Ultralytics YOLO segmentation — yolo segment train data=...\n"
                "# train/val = image folders; labels = same stem .txt with polygon rows.\n"
                f"path: .\ntrain: images/train\nval: images/val\n"
                f"nc: {len(ynames)}\nnames:\n"
            )
            for i, n in enumerate(ynames):
                tmpl_seg += f"  {i}: {n}\n"
            (dirs["label_segments"] / "yolo_seg_dataset_template.yaml").write_text(tmpl_seg)

            # image-pipeline parity: metadata['image'] = file path to RGB; per-frame: images/frames/{stem:06d}.*
            _exts = (".jpg", ".jpeg", ".png", ".bmp", ".webp")
            _ref_image: str | None = None
            _ref_segment_label: str | None = None
            for _fi in range(n_frames):
                if v_stride > 1 and (_fi % v_stride) != 0:
                    continue
                for _ext in _exts:
                    _p = dirs["base"] / "images" / "frames" / f"{_fi:06d}{_ext}"
                    if _p.is_file():
                        _ref_image = str(_p.resolve())
                        _ref_segment_label = str(
                            (dirs["label_segments"] / f"{_fi:06d}.txt").resolve()
                        )
                        break
                if _ref_image:
                    break

            meta_dict = {
                "image": _ref_image,
                "source_video": str(Path(video_path).resolve()),
                "video": str(video_path),
                "video_pipeline_mode": mode,
                "qwen_prompt_vote_counts": log.get("qwen_prompt_vote_counts"),
                "qwen_aggregate_top_k": log.get("qwen_aggregate_top_k"),
                "qwen_prompts_this_video": list(log["prompts"]),
                "qwen_prompts_this_image": list(log["prompts"]),
                "per_frame_prompts": per_frame_prompts,
                "segment_label_file": _ref_segment_label,
                "label_export_mode": "fixed_yolo",
                "bbox_format": bbox_fmt,
                "fps": fps,
                "yolo_nc": len(ynames),
                "yolo_class_names": list(ynames),
                "yolo_class_aliases": dict(yalias),
                "yolo_skip_unmapped": YOLO_SKIP_UNMAPPED_PROMPTS,
                "label_append_confidence": YOLO_LABEL_APPEND_CONFIDENCE,
                "video_export_frame_stride": v_stride,
                "video_export_preview_mp4s": export_preview_mp4s,
                "video_export_overlay_images": export_overlay_images,
                "segment_label_glob": "labels/segments/{frame:06d}.txt",
                "frame_image_glob": "images/frames/{frame:06d}.jpg",
                "frame_overlay_glob": "images/overlays/{frame:06d}.jpg",
                "id_note": (
                    "image = first exported frame path (qwen_sam3_image_pipeline-compatible); "
                    "per-frame: labels/segments/{06d}.txt matches images/frames/{06d}.jpg "
                    "(empty label file if no SAM3 detection on that frame); "
                    "images/overlays/{06d}.jpg = QA viz (boxes+masks) when enabled; "
                    "sam3_obj_id = per-frame instance; yolo_class_id = training class index"
                ),
                "frames": all_meta,
                "objects_segment": all_seg_meta,
            }
            (dirs["label_meta"] / "metadata.json").write_text(json.dumps(meta_dict, indent=2))

            if export_preview_mp4s:
                render_combined_video(
                    frames, outputs_per_frame, pr_union, fps,
                    dirs["videos"] / f"combined_top{len(log['prompts'])}_sam3.mp4"
                )
                render_per_prompt_videos(
                    frames, outputs_per_frame, pr_union, fps, dirs["videos"]
                )

        log["t_export_s"] = t.elapsed

        if export_overlay_images:
            with Timer("Overlay images") as tov:
                for fidx in range(n_frames):
                    if v_stride > 1 and (fidx % v_stride) != 0:
                        continue
                    stem = f"{fidx:06d}"
                    if per_frame_prompts is not None:
                        f_prompts = per_frame_prompts.get(
                            fidx, list(log["prompts"])
                        )
                    else:
                        f_prompts = list(log["prompts"])
                    result = outputs_per_frame.get(fidx)
                    vis_bgr = build_overlay_bgr(frames[fidx], result, f_prompts)
                    cv2.imwrite(
                        str(dirs["images_overlays"] / f"{stem}.jpg"), vis_bgr
                    )
            log["t_overlay_s"] = tov.elapsed
        else:
            log["t_overlay_s"] = 0.0

        log["status"] = "done"

    except Exception as e:
        log["status"] = "error"
        log["error"] = traceback.format_exc()
        print(f"  \u274c ERROR: {e}")

    log["t_total_s"] = time.perf_counter() - wall_start
    _ft = int(log.get("frames_total") or 0)
    if _ft > 0:
        log["t_mean_wall_s_per_frame"] = round(log["t_total_s"] / _ft, 4)
    else:
        log["t_mean_wall_s_per_frame"] = None
    log["finished_at"] = datetime.datetime.now().isoformat()
    log["resources_after"] = snapshot_process_resources(cpu_meter_proc)

    if log["output_dir"]:
        (Path(log["output_dir"]) / "processing_log.json").write_text(
            json.dumps(log, indent=2, default=str)
        )

    ra = log["resources_after"] or {}
    res_bits: list[str] = []
    if ra.get("rss_gib") is not None:
        res_bits.append(f"rss={ra['rss_gib']}GiB")
    if ra.get("vms_gib") is not None:
        res_bits.append(f"vms={ra['vms_gib']}GiB")
    if ra.get("threads") is not None:
        res_bits.append(f"threads={ra['threads']}")
    if ra.get("cpu_percent_since_video_start") is not None:
        res_bits.append(f"cpu%={ra['cpu_percent_since_video_start']}")
    res_bits.append(f"torch_threads={ra.get('torch_intraop_threads')}")
    if ra.get("torch_cuda_allocated_gib") is not None:
        res_bits.append(f"cuda_alloc={ra['torch_cuda_allocated_gib']}GiB")
    res_line = ("  |  " + " ".join(res_bits)) if res_bits else ""

    status_icon = "\u2705" if log.get("status") == "done" else "\u26a0\ufe0f"
    _mpf = log.get("t_mean_wall_s_per_frame")
    _mpf_s = (
        f"  wall/frame≈{_mpf:.3f}s  (N={int(log.get('frames_total') or 0)})"
        if _mpf is not None
        else ""
    )
    print(
        f"  {status_icon} Finished in {log['t_total_s']:.1f}s  "
        f"(read={log['t_read_s']:.1f}s  write_frames={log['t_write_frames_s']:.1f}s  "
        f"qwen={log['t_qwen_s']:.1f}s  sam3={log['t_sam3_s']:.1f}s  "
        f"export={log['t_export_s']:.1f}s  overlay={log.get('t_overlay_s', 0.0):.1f}s)"
        f"{_mpf_s}"
        f"{res_line}"
    )
    return log


print('✅ Pipeline function ready')
