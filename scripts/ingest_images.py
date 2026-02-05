from __future__ import annotations

import argparse
import sqlite3
import sys
from pathlib import Path
from typing import List, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.config import load_config
from app.utils import is_image_file, now_unix, safe_open_image, sha256_file


def init_db(db_path: Path) -> None:
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(db_path))
    cur = conn.cursor()
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS images (
          image_id TEXT PRIMARY KEY,
          uri TEXT NOT NULL,
          width INTEGER NOT NULL,
          height INTEGER NOT NULL,
          sha256 TEXT NOT NULL,
          created_at INTEGER NOT NULL
        );
        """
    )
    conn.commit()
    conn.close()


def upsert_image(
    conn: sqlite3.Connection,
    image_id: str,
    uri: str,
    width: int,
    height: int,
    sha256: str,
    created_at: int,
) -> None:
    cur = conn.cursor()
    cur.execute(
        """
        INSERT INTO images(image_id, uri, width, height, sha256, created_at)
        VALUES(?,?,?,?,?,?)
        ON CONFLICT(image_id) DO UPDATE SET
          uri=excluded.uri,
          width=excluded.width,
          height=excluded.height,
          sha256=excluded.sha256,
          created_at=excluded.created_at;
        """,
        (image_id, uri, width, height, sha256, created_at),
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()

    cfg = load_config(args.config)
    image_dir = cfg.data.image_dir
    db_path = cfg.data.sqlite_path
    allowed_ext = cfg.data.allowed_ext

    if not image_dir.exists():
        raise FileNotFoundError(f"Image directory not found: {image_dir}")

    init_db(db_path)

    conn = sqlite3.connect(str(db_path))
    conn.execute("PRAGMA journal_mode=WAL;")

    count = 0
    for p in image_dir.rglob("*"):
        if not is_image_file(p, allowed_ext):
            continue

        try:
            h = sha256_file(p)
            # Stable ID: use full sha, or shorten if desired. Keep full for safety.
            image_id = h
            img = safe_open_image(p)
            width, height = img.width, img.height
            created_at = now_unix()
            upsert_image(conn, image_id, str(p), width, height, h, created_at)
            count += 1
            if count % 500 == 0:
                conn.commit()
        except Exception:
            # Skip problematic images in POC ingestion
            continue

    conn.commit()
    conn.close()

    print(f"Ingested/updated {count} images into {db_path}")


if __name__ == "__main__":
    main()
