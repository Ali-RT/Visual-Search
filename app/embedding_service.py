from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import torch
from PIL import Image

import open_clip


@dataclass
class EmbeddingOutput:
    embedding: np.ndarray  # float32, shape (D,)
    dim: int


class EmbeddingService:
    """
    Loads a pretrained OpenCLIP model and produces L2-normalized embeddings.
    """

    def __init__(self, backbone: str, pretrained: str, device: str = "auto") -> None:
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

    @torch.inference_mode()
    def embed_pil(self, image: Image.Image) -> EmbeddingOutput:
        """
        Returns a float32 vector (D,) normalized to unit length.
        """
        x = self.preprocess(image).unsqueeze(0).to(self.device)  # (1,3,H,W)
        feats = self.model.encode_image(x)  # (1,D)
        feats = feats / feats.norm(dim=-1, keepdim=True).clamp(min=1e-12)
        vec = feats[0].detach().cpu().float().numpy().astype(np.float32)
        return EmbeddingOutput(embedding=vec, dim=int(vec.shape[0]))
