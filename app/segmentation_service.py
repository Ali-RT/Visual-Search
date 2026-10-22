from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from transformers import SegformerForSemanticSegmentation, SegformerImageProcessor

CITYSCAPES_PALETTE: Dict[int, Tuple[int, int, int]] = {
    0: (128, 64, 128),    # road
    1: (244, 35, 232),    # sidewalk
    2: (70, 70, 70),      # building
    3: (102, 102, 156),   # wall
    4: (190, 153, 153),   # fence
    5: (153, 153, 153),   # pole
    6: (250, 170, 30),    # traffic light
    7: (220, 220, 0),     # traffic sign
    8: (107, 142, 35),    # vegetation
    9: (152, 251, 152),   # terrain
    10: (70, 130, 180),   # sky
    11: (220, 20, 60),    # person
    12: (255, 0, 0),      # rider
    13: (0, 0, 142),      # car
    14: (0, 0, 70),       # truck
    15: (0, 60, 100),     # bus
    16: (0, 80, 100),     # train
    17: (0, 0, 230),      # motorcycle
    18: (119, 11, 32),    # bicycle
}


@dataclass
class SegmentInfo:
    class_id: int
    label: str
    pixel_count: int
    percentage: float
    color: Tuple[int, int, int]


@dataclass
class SegmentationResult:
    mask: np.ndarray  # (H, W) int array of class ids
    segments: List[SegmentInfo]
    width: int
    height: int
    colored_mask_b64: str  # base64-encoded PNG of the colored mask


class SegmentationService:
    def __init__(
        self,
        model_id: str = "nvidia/segformer-b2-finetuned-cityscapes-1024-1024",
        device: str = "auto",
    ) -> None:
        if device == "auto":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        self.processor = SegformerImageProcessor.from_pretrained(model_id)
        self.model = SegformerForSemanticSegmentation.from_pretrained(model_id)
        self.model.eval().to(self.device)
        self.id2label: Dict[int, str] = self.model.config.id2label

    @torch.inference_mode()
    def infer_mask(self, image: Image.Image) -> np.ndarray:
        inputs = self.processor(images=image, return_tensors="pt").to(self.device)
        outputs = self.model(**inputs)

        logits = outputs.logits
        upsampled = F.interpolate(
            logits, size=(image.height, image.width), mode="bilinear", align_corners=False,
        )
        return upsampled.argmax(dim=1).squeeze(0).cpu().numpy().astype(np.uint8)

    def summarize_mask(self, mask: np.ndarray) -> List[SegmentInfo]:
        total_pixels = mask.size
        unique, counts = np.unique(mask, return_counts=True)
        segments: List[SegmentInfo] = []
        for cid, cnt in sorted(zip(unique, counts), key=lambda x: -x[1]):
            cid_int = int(cid)
            segments.append(SegmentInfo(
                class_id=cid_int,
                label=self.id2label.get(cid_int, f"class_{cid_int}"),
                pixel_count=int(cnt),
                percentage=round(float(cnt) / total_pixels * 100, 2),
                color=CITYSCAPES_PALETTE.get(cid_int, (128, 128, 128)),
            ))
        return segments

    def blend_mask(self, image: Image.Image, mask: np.ndarray, alpha: float = 0.45) -> Image.Image:
        colored = self._colorize_mask(mask)
        return Image.blend(image.convert("RGBA"), colored.convert("RGBA"), alpha=alpha).convert("RGB")

    @torch.inference_mode()
    def segment_frame_rgb(
        self, frame_rgb: np.ndarray, alpha: float = 0.45,
    ) -> tuple[np.ndarray, List[SegmentInfo]]:
        image = Image.fromarray(frame_rgb.astype(np.uint8), mode="RGB")
        mask = self.infer_mask(image)
        segments = self.summarize_mask(mask)
        blended = self.blend_mask(image, mask, alpha=alpha)
        return np.array(blended, dtype=np.uint8), segments

    @torch.inference_mode()
    def segment(self, image: Image.Image) -> SegmentationResult:
        mask = self.infer_mask(image)
        segments = self.summarize_mask(mask)
        blended = self.blend_mask(image, mask, alpha=0.45)

        import io, base64
        buf = io.BytesIO()
        blended.convert("RGB").save(buf, format="JPEG", quality=85)
        mask_b64 = base64.b64encode(buf.getvalue()).decode()

        return SegmentationResult(
            mask=mask,
            segments=segments,
            width=image.width,
            height=image.height,
            colored_mask_b64=mask_b64,
        )

    @staticmethod
    def _colorize_mask(mask: np.ndarray) -> Image.Image:
        h, w = mask.shape
        rgb = np.zeros((h, w, 3), dtype=np.uint8)
        for cid, color in CITYSCAPES_PALETTE.items():
            rgb[mask == cid] = color
        return Image.fromarray(rgb)
