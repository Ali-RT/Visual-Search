from __future__ import annotations

from typing import Any, Iterable, List, Literal

import numpy as np
import torch
from PIL import Image
from transformers import pipeline


class SAM2SegmentationService:
    """Prompt-free SAM2 mask generation with RGB overlay output."""

    def __init__(self, model_id: str = "facebook/sam2-hiera-small", device: str = "auto") -> None:
        self.model_id = model_id
        self.device = self._resolve_device(device)
        self.generator = pipeline("mask-generation", model=model_id, device=self.device)

    @staticmethod
    def _resolve_device(device: str) -> int:
        if device == "auto":
            return 0 if torch.cuda.is_available() else -1
        if device.startswith("cuda"):
            return 0
        return -1

    def segment_frame_rgb(
        self,
        frame_rgb: np.ndarray,
        alpha: float = 0.45,
        profile: Literal["easy", "balanced", "quality"] = "balanced",
        click_point: tuple[float, float] | None = None,
        click_points: list[tuple[float, float]] | None = None,
    ) -> tuple[np.ndarray, int]:
        image = Image.fromarray(frame_rgb.astype(np.uint8), mode="RGB")
        output = self.generator(image, **self._profile_kwargs(profile))

        masks = self._extract_masks(output, frame_rgb.shape[:2])
        if not masks:
            return frame_rgb, 0

        h, w = frame_rgb.shape[:2]
        prompt_points = list(click_points or [])
        if click_point is not None:
            prompt_points.append(click_point)

        if prompt_points:
            def _mask_rank(mask: np.ndarray) -> tuple[int, int]:
                hits = 0
                for px, py in prompt_points:
                    cx = int(np.clip(px, 0.0, 1.0) * (w - 1))
                    cy = int(np.clip(py, 0.0, 1.0) * (h - 1))
                    if bool(mask[cy, cx]):
                        hits += 1
                return hits, int(mask.sum())

            masks = sorted(masks, key=_mask_rank, reverse=True)
        else:
            masks = sorted(masks, key=lambda m: int(m.sum()), reverse=True)
        overlay = frame_rgb.astype(np.float32).copy()
        alpha = float(max(0.0, min(1.0, alpha)))
        palette = self._palette()

        min_area = max(16, int(0.0004 * h * w))
        kept = 0
        for i, mask in enumerate(masks):
            area = int(mask.sum())
            if area < min_area:
                continue
            if prompt_points:
                hits = 0
                for px, py in prompt_points:
                    cx = int(np.clip(px, 0.0, 1.0) * (w - 1))
                    cy = int(np.clip(py, 0.0, 1.0) * (h - 1))
                    if bool(mask[cy, cx]):
                        hits += 1
                # Keep only masks that include at least one selected point.
                if hits == 0:
                    continue
            color = np.array(palette[i % len(palette)], dtype=np.float32)
            overlay[mask] = (1.0 - alpha) * overlay[mask] + alpha * color
            kept += 1

        return overlay.clip(0, 255).astype(np.uint8), kept

    @staticmethod
    def _profile_kwargs(profile: str) -> dict[str, float | int]:
        # CPU-friendly defaults for interactive video processing.
        if profile == "easy":
            return {
                "points_per_crop": 8,
                "points_per_batch": 16,
            }
        if profile == "quality":
            return {
                "points_per_crop": 24,
                "points_per_batch": 32,
                "pred_iou_thresh": 0.86,
                "stability_score_thresh": 0.93,
            }
        return {
            "points_per_crop": 12,
            "points_per_batch": 24,
        }

    def _extract_masks(self, output: Any, shape: tuple[int, int]) -> List[np.ndarray]:
        h, w = shape
        candidates: list[Any] = []

        if isinstance(output, dict):
            if "masks" in output:
                candidates = list(output["masks"])
            else:
                candidates = [output]
        elif isinstance(output, list):
            candidates = output
        else:
            candidates = [output]

        masks: list[np.ndarray] = []
        for item in candidates:
            mask_candidates = self._iter_mask_candidates(item)
            for raw_mask in mask_candidates:
                mask = self._to_bool_mask(raw_mask, h, w)
                if mask is not None:
                    masks.append(mask)
        return masks

    def _iter_mask_candidates(self, item: Any) -> Iterable[Any]:
        if isinstance(item, dict):
            if "masks" in item and isinstance(item["masks"], list):
                for m in item["masks"]:
                    yield m
            if "mask" in item:
                yield item["mask"]
            if "segmentation" in item:
                yield item["segmentation"]
        else:
            yield item

    @staticmethod
    def _to_bool_mask(raw: Any, h: int, w: int) -> np.ndarray | None:
        arr: np.ndarray
        if isinstance(raw, Image.Image):
            arr = np.array(raw)
        elif torch.is_tensor(raw):
            arr = raw.detach().cpu().numpy()
        else:
            arr = np.array(raw)

        if arr.ndim > 2:
            arr = np.squeeze(arr)
        if arr.ndim != 2:
            return None
        if arr.shape != (h, w):
            return None
        return arr.astype(np.float32) > 0.5

    @staticmethod
    def _palette() -> list[tuple[int, int, int]]:
        return [
            (255, 99, 71),   # tomato
            (30, 144, 255),  # dodger blue
            (50, 205, 50),   # lime green
            (255, 215, 0),   # gold
            (186, 85, 211),  # medium orchid
            (0, 206, 209),   # dark turquoise
            (255, 140, 0),   # dark orange
            (220, 20, 60),   # crimson
        ]
