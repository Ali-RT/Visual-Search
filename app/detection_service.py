from __future__ import annotations

from dataclasses import dataclass
from typing import List

import torch
from PIL import Image
from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor


@dataclass
class Detection:
    label: str
    score: float
    box: List[float]  # [x1, y1, x2, y2] in pixel coordinates


ADAS_PROMPTS = "vehicle. pedestrian. traffic sign. traffic light. crosswalk. lane marking. bicycle. motorcycle. bus. truck."


class DetectionService:
    def __init__(
        self,
        model_id: str = "IDEA-Research/grounding-dino-tiny",
        device: str = "auto",
    ) -> None:
        if device == "auto":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        self.processor = AutoProcessor.from_pretrained(model_id)
        self.model = AutoModelForZeroShotObjectDetection.from_pretrained(model_id)
        self.model.eval().to(self.device)

    @torch.inference_mode()
    def detect(
        self,
        image: Image.Image,
        text_prompt: str | None = None,
        threshold: float = 0.3,
        text_threshold: float = 0.25,
    ) -> List[Detection]:
        prompt = text_prompt or ADAS_PROMPTS
        inputs = self.processor(images=image, text=prompt, return_tensors="pt").to(self.device)

        outputs = self.model(**inputs)
        results = self.processor.post_process_grounded_object_detection(
            outputs,
            inputs.input_ids,
            threshold=threshold,
            text_threshold=text_threshold,
            target_sizes=[image.size[::-1]],
        )

        r = results[0]
        detections: List[Detection] = []
        for box, score, label in zip(r["boxes"], r["scores"], r["text_labels"]):
            detections.append(Detection(
                label=str(label),
                score=round(float(score), 4),
                box=[round(float(c), 1) for c in box.tolist()],
            ))

        detections.sort(key=lambda d: d.score, reverse=True)
        return detections
