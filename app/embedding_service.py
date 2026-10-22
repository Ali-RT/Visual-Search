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
    Loads a pretrained OpenCLIP model and produces L2-normalized embeddings
    for both images and text (shared CLIP embedding space).
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
        self.tokenizer = open_clip.get_tokenizer(backbone)

    @torch.inference_mode()
    def embed_pil(self, image: Image.Image) -> EmbeddingOutput:
        x = self.preprocess(image).unsqueeze(0).to(self.device)
        feats = self.model.encode_image(x)
        feats = feats / feats.norm(dim=-1, keepdim=True).clamp(min=1e-12)
        vec = feats[0].detach().cpu().float().numpy().astype(np.float32)
        return EmbeddingOutput(embedding=vec, dim=int(vec.shape[0]))

    @torch.inference_mode()
    def embed_text(self, text: str) -> EmbeddingOutput:
        tokens = self.tokenizer([text]).to(self.device)
        feats = self.model.encode_text(tokens)
        feats = feats / feats.norm(dim=-1, keepdim=True).clamp(min=1e-12)
        vec = feats[0].detach().cpu().float().numpy().astype(np.float32)
        return EmbeddingOutput(embedding=vec, dim=int(vec.shape[0]))
