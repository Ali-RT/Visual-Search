"""
Batch-tag all images in the metadata SQLite and write results to a tags table.

Usage:
    python scripts/tag_images.py --config configs/adas_usecase.yaml
"""
from __future__ import annotations

import argparse
import json
import sqlite3
import sys
from pathlib import Path
from typing import Any, Dict, List

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from tqdm import tqdm

from app.config import load_config
from app.tagging_service import TaggingService
from app.utils import safe_open_image, now_unix


def fetch_images(db_path: Path) -> List[Dict[str, Any]]:
    conn = sqlite3.connect(str(db_path))
    cur = conn.cursor()
    cur.execute("SELECT image_id, uri FROM images ORDER BY image_id;")
    rows = cur.fetchall()
    conn.close()
    return [{"image_id": r[0], "uri": r[1]} for r in rows]


def ensure_tags_table(db_path: Path) -> None:
    conn = sqlite3.connect(str(db_path))
    conn.execute("""
        CREATE TABLE IF NOT EXISTS image_tags (
            image_id   TEXT PRIMARY KEY,
            tags_json  TEXT NOT NULL,
            flat_labels TEXT NOT NULL,
            created_at INTEGER NOT NULL
        );
    """)
    conn.commit()
    conn.close()


def upsert_tags(
    db_path: Path,
    image_id: str,
    tags_json: str,
    flat_labels: str,
    created_at: int,
) -> None:
    conn = sqlite3.connect(str(db_path))
    conn.execute(
        """
        INSERT INTO image_tags (image_id, tags_json, flat_labels, created_at)
        VALUES (?, ?, ?, ?)
        ON CONFLICT(image_id) DO UPDATE SET
            tags_json = excluded.tags_json,
            flat_labels = excluded.flat_labels,
            created_at = excluded.created_at;
        """,
        (image_id, tags_json, flat_labels, created_at),
    )
    conn.commit()
    conn.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Batch-tag images with CLIP zero-shot classification")
    parser.add_argument("--config", required=True, help="Path to YAML config file")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--limit", type=int, default=None, help="Tag only first N images (for testing)")
    args = parser.parse_args()

    cfg = load_config(args.config)
    items = fetch_images(cfg.data.sqlite_path)
    if not items:
        raise RuntimeError("No images found in SQLite. Run ingest_images.py first.")

    if args.limit:
        items = items[: args.limit]

    tag_groups = None
    tag_labels = None
    if cfg.tagging:
        tag_groups = cfg.tagging.tag_groups
        tag_labels = cfg.tagging.tag_labels

    tagger = TaggingService(
        backbone=cfg.model.backbone,
        pretrained=cfg.model.pretrained,
        device=cfg.model.device,
        tag_groups=tag_groups,
        tag_labels=tag_labels,
    )

    ensure_tags_table(cfg.data.sqlite_path)

    print(f"Tagging {len(items)} images...")
    ts = now_unix()
    failed = 0

    for item in tqdm(items, desc="Tagging"):
        try:
            img = safe_open_image(Path(item["uri"]))
            output = tagger.tag_pil(img)
            upsert_tags(
                cfg.data.sqlite_path,
                image_id=item["image_id"],
                tags_json=json.dumps(output.to_dict()),
                flat_labels=json.dumps(output.flat_labels()),
                created_at=ts,
            )
        except Exception as e:
            failed += 1
            if failed <= 5:
                print(f"  FAIL {item['image_id']}: {e}")

    print(f"Done. Tagged {len(items) - failed}/{len(items)} images. ({failed} failed)")

    # Print a distribution summary
    conn = sqlite3.connect(str(cfg.data.sqlite_path))
    cur = conn.cursor()
    cur.execute("SELECT flat_labels FROM image_tags;")
    rows = cur.fetchall()
    conn.close()

    from collections import Counter
    group_counters: Dict[str, Counter] = {}
    for (flat_json,) in rows:
        labels = json.loads(flat_json)
        for group, label in labels.items():
            if group not in group_counters:
                group_counters[group] = Counter()
            group_counters[group][label] += 1

    print("\n--- Tag Distribution ---")
    for group, counter in sorted(group_counters.items()):
        print(f"\n{group}:")
        for label, count in counter.most_common():
            print(f"  {label:15s} {count:>6d}")


if __name__ == "__main__":
    main()
