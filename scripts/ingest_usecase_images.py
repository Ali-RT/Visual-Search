"""
Ingest images organized by use-case into the SQLite metadata DB.

Expected directory layout:
    data/use_cases/
    ├── blockage/
    │   ├── <label_folder>/
    │   │   ├── image1.png
    │   │   └── image2.jpg
    │   └── <label_folder>/
    │       └── ...
    ├── traffic_light/
    │   └── ...
    └── crosswalk/
        └── ...

The use_case is derived from the first-level directory name,
the label is derived from the second-level directory name.

Usage:
    python scripts/ingest_usecase_images.py --config configs/adas_usecase.yaml
    python scripts/ingest_usecase_images.py --config configs/adas_usecase.yaml --use_cases blockage crosswalk
"""
from __future__ import annotations

import argparse
import sqlite3
import sys
from pathlib import Path
from typing import List, Optional

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from tqdm import tqdm

from app.config import load_config
from app.utils import is_image_file, now_unix, safe_open_image, sha256_file


def init_db(db_path: Path) -> None:
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(db_path))
    conn.execute("""
        CREATE TABLE IF NOT EXISTS images (
            image_id   TEXT PRIMARY KEY,
            uri        TEXT NOT NULL,
            width      INTEGER NOT NULL,
            height     INTEGER NOT NULL,
            sha256     TEXT NOT NULL,
            use_case   TEXT NOT NULL DEFAULT '',
            label      TEXT NOT NULL DEFAULT '',
            created_at INTEGER NOT NULL
        );
    """)
    conn.execute("CREATE INDEX IF NOT EXISTS idx_images_use_case ON images(use_case);")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_images_label ON images(label);")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_images_use_case_label ON images(use_case, label);")
    conn.commit()
    conn.close()


def upsert_image(
    conn: sqlite3.Connection,
    image_id: str,
    uri: str,
    width: int,
    height: int,
    sha256: str,
    use_case: str,
    label: str,
    created_at: int,
) -> None:
    conn.execute(
        """
        INSERT INTO images(image_id, uri, width, height, sha256, use_case, label, created_at)
        VALUES(?,?,?,?,?,?,?,?)
        ON CONFLICT(image_id) DO UPDATE SET
            uri=excluded.uri,
            width=excluded.width,
            height=excluded.height,
            sha256=excluded.sha256,
            use_case=excluded.use_case,
            label=excluded.label,
            created_at=excluded.created_at;
        """,
        (image_id, uri, width, height, sha256, use_case, label, created_at),
    )


def discover_use_cases(image_dir: Path) -> List[str]:
    """Return sorted list of use_case directory names."""
    return sorted(
        d.name for d in image_dir.iterdir() if d.is_dir() and not d.name.startswith(".")
    )


def collect_images(
    image_dir: Path,
    use_cases: Optional[List[str]],
    allowed_ext: List[str],
) -> List[dict]:
    """
    Walk the use_case directory tree and collect image paths with metadata.

    Supports two layouts:
      Flat:   data/use_cases/<use_case>/<image_file>
      Nested: data/use_cases/<use_case>/<label>/<image_file>

    Returns list of dicts with keys: path, use_case, label.
    """
    all_use_cases = discover_use_cases(image_dir)
    if use_cases:
        selected = [uc for uc in all_use_cases if uc in set(use_cases)]
    else:
        selected = all_use_cases

    items = []
    for uc in selected:
        uc_dir = image_dir / uc

        # Images directly under the use_case dir (flat layout)
        for p in sorted(uc_dir.iterdir()):
            if p.is_file() and is_image_file(p, allowed_ext):
                items.append({"path": p, "use_case": uc, "label": uc})

        # Images inside label subdirectories (nested layout)
        for label_dir in sorted(uc_dir.iterdir()):
            if not label_dir.is_dir() or label_dir.name.startswith("."):
                continue
            for p in sorted(label_dir.rglob("*")):
                if is_image_file(p, allowed_ext):
                    items.append({"path": p, "use_case": uc, "label": label_dir.name})

    return items


def main() -> None:
    parser = argparse.ArgumentParser(description="Ingest use-case organized images into SQLite")
    parser.add_argument("--config", required=True, help="Path to YAML config")
    parser.add_argument(
        "--use_cases",
        nargs="*",
        default=None,
        help="Only ingest specific use cases (space-separated). Omit to ingest all.",
    )
    args = parser.parse_args()

    cfg = load_config(args.config)
    image_dir = cfg.data.image_dir
    db_path = cfg.data.sqlite_path

    if not image_dir.exists():
        raise FileNotFoundError(f"Image directory not found: {image_dir}")

    available = discover_use_cases(image_dir)
    print(f"Available use cases in {image_dir}: {available}")

    items = collect_images(image_dir, args.use_cases, cfg.data.allowed_ext)
    print(f"Found {len(items)} images to ingest")

    if not items:
        print("No images found. Check directory structure:")
        print(f"  Expected: {image_dir}/<use_case>/<label>/<image_file>")
        return

    init_db(db_path)

    conn = sqlite3.connect(str(db_path))
    conn.execute("PRAGMA journal_mode=WAL;")

    count = 0
    failed = 0
    ts = now_unix()

    for item in tqdm(items, desc="Ingesting"):
        p = item["path"]
        try:
            h = sha256_file(p)
            img = safe_open_image(p)
            upsert_image(
                conn,
                image_id=h,
                uri=str(p),
                width=img.width,
                height=img.height,
                sha256=h,
                use_case=item["use_case"],
                label=item["label"],
                created_at=ts,
            )
            count += 1
            if count % 500 == 0:
                conn.commit()
        except Exception as e:
            failed += 1
            if failed <= 5:
                print(f"  FAIL: {p} -> {e}")

    conn.commit()
    conn.close()

    print(f"\nIngested {count} images into {db_path} ({failed} failed)")

    # Print summary
    conn = sqlite3.connect(str(db_path))
    cur = conn.cursor()
    cur.execute("""
        SELECT use_case, COUNT(*) as cnt
        FROM images
        WHERE use_case != ''
        GROUP BY use_case
        ORDER BY use_case;
    """)
    print("\n--- Images per use_case ---")
    for row in cur.fetchall():
        print(f"  {row[0]:25s} {row[1]:>6d}")

    cur.execute("""
        SELECT use_case, label, COUNT(*) as cnt
        FROM images
        WHERE use_case != ''
        GROUP BY use_case, label
        ORDER BY use_case, label;
    """)
    print("\n--- Images per use_case / label ---")
    for row in cur.fetchall():
        print(f"  {row[0]:25s} / {row[1]:30s} {row[2]:>6d}")

    conn.close()


if __name__ == "__main__":
    main()
