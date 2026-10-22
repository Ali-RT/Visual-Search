from __future__ import annotations

import json
import hashlib
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Optional

import cv2
import numpy as np
from PIL import Image


@dataclass
class QualityMetrics:
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


def assess_blur(gray: np.ndarray) -> tuple[float, str]:
    lap_var = cv2.Laplacian(gray, cv2.CV_64F).var()
    if lap_var >= 200:
        return float(lap_var), "sharp"
    elif lap_var >= 50:
        return float(lap_var), "moderate"
    else:
        return float(lap_var), "blurry"


def assess_exposure(gray: np.ndarray) -> tuple[float, str]:
    hist = cv2.calcHist([gray], [0], None, [256], [0, 256]).flatten()
    total = hist.sum()
    mean_val = np.average(np.arange(256), weights=hist)

    dark_pct = hist[:20].sum() / total
    bright_pct = hist[235:].sum() / total

    mean_score = 1.0 - abs(mean_val - 128) / 128
    clip_penalty = max(0, 1.0 - dark_pct * 3 - bright_pct * 3)
    score = float(np.clip(mean_score * 0.6 + clip_penalty * 0.4, 0, 1))

    if bright_pct > 0.3:
        return score, "overexposed"
    elif dark_pct > 0.3:
        return score, "underexposed"
    elif dark_pct > 0.15 or bright_pct > 0.15:
        return score, "clipped"
    else:
        return score, "good"


def assess_contrast(gray: np.ndarray) -> tuple[float, str]:
    rms = float(gray.std())
    if rms >= 40:
        return rms, "good"
    elif rms >= 20:
        return rms, "low"
    else:
        return rms, "very_low"


def _build_quality_metrics(
    blur_score: float, blur_label: str,
    exp_score: float, exp_label: str,
    contrast_score: float, contrast_label: str,
    clip_score: float, clip_label: str,
) -> QualityMetrics:
    issues: List[str] = []
    if blur_label == "blurry":
        issues.append("Image is blurry (possible motion blur or defocus)")
    elif blur_label == "moderate":
        issues.append("Image has moderate blur")
    if exp_label == "overexposed":
        issues.append("Image is overexposed (sun glare or bright sky)")
    elif exp_label == "underexposed":
        issues.append("Image is underexposed (too dark)")
    elif exp_label == "clipped":
        issues.append("Image has clipped highlights or shadows")
    if contrast_label == "very_low":
        issues.append("Very low contrast (fog, haze, or dirty lens)")
    elif contrast_label == "low":
        issues.append("Low contrast")
    if clip_label == "poor":
        issues.append("Overall visual quality is poor (CLIP assessment)")

    blur_norm = min(blur_score / 300, 1.0)
    composite = (blur_norm * 0.35 + exp_score * 0.25 + min(contrast_score / 60, 1.0) * 0.15 + clip_score * 0.25) * 100

    if composite >= 70:
        overall_label = "good"
    elif composite >= 45:
        overall_label = "fair"
    else:
        overall_label = "poor"

    return QualityMetrics(
        blur_score=round(blur_score, 1), blur_label=blur_label,
        exposure_score=round(exp_score, 3), exposure_label=exp_label,
        contrast_score=round(contrast_score, 1), contrast_label=contrast_label,
        clip_quality=round(clip_score, 4), clip_label=clip_label,
        overall_score=round(composite, 1), overall_label=overall_label,
        issues=issues,
    )


CACHE_PATH = Path("artifacts/quality_cache.json")


class QualityService:
    def __init__(self, clip_tagger=None) -> None:
        self.clip_tagger = clip_tagger
        self._clip_features = None
        self._cache: Dict[str, dict] = {}
        self._init_clip_prompts()
        self._load_cache()

    def _init_clip_prompts(self) -> None:
        if self.clip_tagger is None:
            return
        import torch
        quality_prompts = [
            "a high quality sharp clear dashcam photo with good visibility",
            "a low quality blurry noisy dashcam photo with poor visibility",
        ]
        tokens = self.clip_tagger.tokenizer(quality_prompts).to(self.clip_tagger.device)
        with torch.inference_mode():
            feats = self.clip_tagger.model.encode_text(tokens)
            feats = feats / feats.norm(dim=-1, keepdim=True).clamp(min=1e-12)
        self._clip_features = feats

    def _load_cache(self) -> None:
        if CACHE_PATH.exists():
            try:
                self._cache = json.loads(CACHE_PATH.read_text())
            except Exception:
                self._cache = {}

    def _save_cache(self) -> None:
        CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
        CACHE_PATH.write_text(json.dumps(self._cache, indent=2))

    @staticmethod
    def _file_key(fpath: Path) -> str:
        stat = fpath.stat()
        raw = f"{fpath}:{stat.st_size}:{stat.st_mtime_ns}"
        return hashlib.md5(raw.encode()).hexdigest()

    def _clip_iqa_batch(self, images: List[Image.Image], batch_size: int = 16) -> List[tuple[float, str]]:
        if self.clip_tagger is None or self._clip_features is None:
            return [(0.5, "unknown")] * len(images)

        import torch
        results: List[tuple[float, str]] = []
        for start in range(0, len(images), batch_size):
            batch = images[start:start + batch_size]
            tensors = torch.stack([self.clip_tagger.preprocess(img) for img in batch]).to(self.clip_tagger.device)
            with torch.inference_mode():
                img_feats = self.clip_tagger.model.encode_image(tensors)
                img_feats = img_feats / img_feats.norm(dim=-1, keepdim=True).clamp(min=1e-12)
                sims = img_feats @ self._clip_features.T
                probs = sims.softmax(dim=-1).cpu().numpy()

            for p in probs:
                q = float(p[0])
                if q >= 0.65:
                    results.append((q, "good"))
                elif q >= 0.45:
                    results.append((q, "fair"))
                else:
                    results.append((q, "poor"))
        return results

    def _clip_iqa(self, image: Image.Image) -> tuple[float, str]:
        return self._clip_iqa_batch([image])[0]

    def assess(self, image: Image.Image) -> QualityMetrics:
        arr = np.array(image)
        gray = cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY) if arr.ndim == 3 else arr

        blur_score, blur_label = assess_blur(gray)
        exp_score, exp_label = assess_exposure(gray)
        contrast_score, contrast_label = assess_contrast(gray)
        clip_score, clip_label = self._clip_iqa(image)

        return _build_quality_metrics(
            blur_score, blur_label, exp_score, exp_label,
            contrast_score, contrast_label, clip_score, clip_label,
        )

    def scan_directory(
        self,
        image_dir: str | Path,
        extensions: tuple[str, ...] = (".png", ".jpg", ".jpeg", ".webp"),
        limit: Optional[int] = None,
    ) -> List[dict]:
        image_dir = Path(image_dir)
        files = sorted(f for f in image_dir.rglob("*") if f.suffix.lower() in extensions)

        # Phase 1: fast classical metrics + check cache for CLIP scores
        uncached_indices: List[int] = []
        uncached_images: List[Image.Image] = []
        file_data: List[dict] = []

        for fpath in files:
            try:
                key = self._file_key(fpath)
                img = Image.open(fpath).convert("RGB")
                arr = np.array(img)
                gray = cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY) if arr.ndim == 3 else arr

                blur_score, blur_label = assess_blur(gray)
                exp_score, exp_label = assess_exposure(gray)
                contrast_score, contrast_label = assess_contrast(gray)

                entry = {
                    "path": str(fpath), "uri": str(fpath),
                    "filename": fpath.name, "width": img.width, "height": img.height,
                    "key": key,
                    "blur": (blur_score, blur_label),
                    "exposure": (exp_score, exp_label),
                    "contrast": (contrast_score, contrast_label),
                    "clip": None,
                }

                if key in self._cache:
                    entry["clip"] = (self._cache[key]["clip_quality"], self._cache[key]["clip_label"])
                else:
                    uncached_indices.append(len(file_data))
                    uncached_images.append(img)

                file_data.append(entry)
            except Exception:
                continue

        # Phase 2: batch CLIP-IQA only for uncached images
        if uncached_images:
            clip_results = self._clip_iqa_batch(uncached_images, batch_size=16)
            for idx, clip_result in zip(uncached_indices, clip_results):
                file_data[idx]["clip"] = clip_result

        # Phase 3: build final metrics + update cache
        results: List[dict] = []
        new_cache_entries = 0
        for entry in file_data:
            clip_score, clip_label = entry["clip"] or (0.5, "unknown")
            blur_score, blur_label = entry["blur"]
            exp_score, exp_label = entry["exposure"]
            contrast_score, contrast_label = entry["contrast"]

            metrics = _build_quality_metrics(
                blur_score, blur_label, exp_score, exp_label,
                contrast_score, contrast_label, clip_score, clip_label,
            )

            key = entry["key"]
            if key not in self._cache:
                self._cache[key] = {"clip_quality": clip_score, "clip_label": clip_label}
                new_cache_entries += 1

            results.append({
                "path": entry["path"], "uri": entry["uri"],
                "filename": entry["filename"],
                "width": entry["width"], "height": entry["height"],
                "metrics": metrics,
            })

        if new_cache_entries > 0:
            self._save_cache()

        results.sort(key=lambda r: r["metrics"].overall_score)
        if limit:
            results = results[:limit]
        return results
