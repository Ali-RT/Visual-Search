from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import open_clip
import numpy as np
import torch
from PIL import Image


DEFAULT_TAG_GROUPS: Dict[str, List[str]] = {
    "time_of_day": [
        "a dashcam photo of a road during daytime, the sky is bright and blue, shadows are visible",
        "a dashcam photo of a road at nighttime, headlights illuminate the dark road, street lights are on",
        "a dashcam photo of a road at dawn or dusk, the sky has orange and purple colors, low sun angle",
    ],
    "scene_type": [
        "a dashcam photo driving through an urban city with tall buildings, intersections, and dense traffic",
        "a dashcam photo driving on a rural country road surrounded by open fields, trees, and nature",
        "a dashcam photo driving on a multi-lane highway or motorway with highway signs and fast traffic",
        "a dashcam photo driving through a quiet suburban residential area with houses and parked cars",
    ],
    "crosswalk": [
        "a dashcam photo where white painted crosswalk lines or a zebra crossing is visible on the road surface",
        "a dashcam photo of a road with no crosswalk markings or pedestrian crossing visible",
    ],
    "pedestrian": [
        "a dashcam photo where one or more pedestrians, cyclists, or people are visible near the road",
        "a dashcam photo of a road with no people, pedestrians, or cyclists visible",
    ],
    "vehicle_density": [
        "a dashcam photo of an empty or nearly empty road with very few or no other vehicles",
        "a dashcam photo of a road with several vehicles, moderate amount of traffic",
        "a dashcam photo of a road packed with vehicles, heavy bumper-to-bumper traffic congestion",
    ],
}

TAG_LABELS: Dict[str, List[str]] = {
    "time_of_day": ["day", "night", "dawn_dusk"],
    "scene_type": ["urban", "rural", "highway", "suburban"],
    "crosswalk": ["present", "absent"],
    "pedestrian": ["present", "absent"],
    "vehicle_density": ["low", "moderate", "heavy"],
}


@dataclass
class TagResult:
    group: str
    predicted_label: str
    confidence: float
    all_scores: Dict[str, float]


@dataclass
class ImageTagOutput:
    tags: List[TagResult]

    def to_dict(self) -> Dict[str, Any]:
        return {
            t.group: {
                "label": t.predicted_label,
                "confidence": round(t.confidence, 4),
                "scores": {k: round(v, 4) for k, v in t.all_scores.items()},
            }
            for t in self.tags
        }

    def flat_labels(self) -> Dict[str, str]:
        return {t.group: t.predicted_label for t in self.tags}


class TaggingService:
    """
    Zero-shot image classification using OpenCLIP.

    Encodes predefined text prompts per tag group at init time, then at
    inference only the image needs encoding. Classification is a cosine
    similarity argmax between the image embedding and the cached text
    embeddings.
    """

    def __init__(
        self,
        backbone: str = "ViT-B-32",
        pretrained: str = "laion2b_s34b_b79k",
        device: str = "auto",
        tag_groups: Optional[Dict[str, List[str]]] = None,
        tag_labels: Optional[Dict[str, List[str]]] = None,
    ) -> None:
        if device == "auto":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        model, _, preprocess = open_clip.create_model_and_transforms(
            model_name=backbone,
            pretrained=pretrained,
        )
        model.eval()
        self.model = model.to(self.device)
        self.preprocess = preprocess
        self.tokenizer = open_clip.get_tokenizer(backbone)
        self._ensure_tokenizer_compatibility()

        self.tag_groups = tag_groups or DEFAULT_TAG_GROUPS
        self.tag_labels = tag_labels or TAG_LABELS

        self._text_features: Dict[str, torch.Tensor] = {}
        self._encode_all_prompts()

    def _ensure_tokenizer_compatibility(self) -> None:
        """
        Ensure OpenCLIP's HF tokenizer wrapper works with newer tokenizer APIs.

        Some tokenizers (e.g. GemmaTokenizer) no longer expose
        `batch_encode_plus`; OpenCLIP still calls it internally. We alias
        `batch_encode_plus` to `__call__` when needed.
        """
        hf_tokenizer = getattr(self.tokenizer, "tokenizer", None)
        if hf_tokenizer is None:
            return
        if hasattr(hf_tokenizer, "batch_encode_plus"):
            return

        def _batch_encode_plus(texts, **kwargs):
            return hf_tokenizer(texts, **kwargs)

        setattr(hf_tokenizer, "batch_encode_plus", _batch_encode_plus)

    @torch.inference_mode()
    def _encode_all_prompts(self) -> None:
        for group, prompts in self.tag_groups.items():
            tokens = self.tokenizer(prompts).to(self.device)
            text_feats = self.model.encode_text(tokens)
            text_feats = text_feats / text_feats.norm(dim=-1, keepdim=True).clamp(min=1e-12)
            self._text_features[group] = text_feats

    @torch.inference_mode()
    def tag_pil(self, image: Image.Image) -> ImageTagOutput:
        x = self.preprocess(image).unsqueeze(0).to(self.device)
        img_feat = self.model.encode_image(x)
        img_feat = img_feat / img_feat.norm(dim=-1, keepdim=True).clamp(min=1e-12)

        results: List[TagResult] = []
        for group, text_feats in self._text_features.items():
            sims = (img_feat @ text_feats.T).squeeze(0)
            probs = sims.softmax(dim=0).cpu().numpy().astype(float)

            labels = self.tag_labels[group]
            best_idx = int(np.argmax(probs))

            all_scores = {labels[i]: float(probs[i]) for i in range(len(labels))}
            results.append(
                TagResult(
                    group=group,
                    predicted_label=labels[best_idx],
                    confidence=float(probs[best_idx]),
                    all_scores=all_scores,
                )
            )

        return ImageTagOutput(tags=results)

    @torch.inference_mode()
    def tag_pil_batch(
        self, images: List[Image.Image], batch_size: int = 32
    ) -> List[ImageTagOutput]:
        outputs: List[ImageTagOutput] = []
        for start in range(0, len(images), batch_size):
            batch_imgs = images[start : start + batch_size]
            tensors = torch.stack([self.preprocess(img) for img in batch_imgs]).to(self.device)
            img_feats = self.model.encode_image(tensors)
            img_feats = img_feats / img_feats.norm(dim=-1, keepdim=True).clamp(min=1e-12)

            for j in range(img_feats.size(0)):
                feat = img_feats[j : j + 1]
                results: List[TagResult] = []
                for group, text_feats in self._text_features.items():
                    sims = (feat @ text_feats.T).squeeze(0)
                    probs = sims.softmax(dim=0).cpu().numpy().astype(float)
                    labels = self.tag_labels[group]
                    best_idx = int(np.argmax(probs))
                    all_scores = {labels[i]: float(probs[i]) for i in range(len(labels))}
                    results.append(
                        TagResult(
                            group=group,
                            predicted_label=labels[best_idx],
                            confidence=float(probs[best_idx]),
                            all_scores=all_scores,
                        )
                    )
                outputs.append(ImageTagOutput(tags=results))

        return outputs
