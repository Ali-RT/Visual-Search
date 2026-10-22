"""
CPU-safe fallback stub for environments without real flash_attn.

This module exists only to satisfy model dynamic imports on systems where
flash-attn wheels are unavailable (e.g., macOS / CPU-only).
"""

from __future__ import annotations


def flash_attn_func(*args, **kwargs):
    raise RuntimeError("flash_attn is unavailable in this environment (stub module loaded).")


def flash_attn_varlen_func(*args, **kwargs):
    raise RuntimeError("flash_attn is unavailable in this environment (stub module loaded).")

