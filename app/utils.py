from __future__ import annotations

import hashlib
import os
import time
from pathlib import Path
from typing import Iterable, List, Optional, Tuple

from PIL import Image


def now_unix() -> int:
    return int(time.time())


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(chunk_size)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def is_image_file(path: Path, allowed_ext: List[str]) -> bool:
    return path.is_file() and path.suffix.lower() in set([e.lower() for e in allowed_ext])


def safe_open_image(path: Path) -> Image.Image:
    # Always convert to RGB for consistent embeddings.
    img = Image.open(path)
    return img.convert("RGB")
