"""
Minimal helpers to satisfy optional flash_attn.bert_padding imports.
"""

from __future__ import annotations

import torch


def index_first_axis(x: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:
    return x[indices]


def pad_input(*args, **kwargs):
    raise RuntimeError("pad_input from flash_attn is unavailable in this environment (stub module loaded).")


def unpad_input(*args, **kwargs):
    raise RuntimeError("unpad_input from flash_attn is unavailable in this environment (stub module loaded).")

