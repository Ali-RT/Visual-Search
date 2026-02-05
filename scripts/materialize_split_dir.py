#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import os
import shutil
from pathlib import Path


def link_or_copy(src: Path, dst: Path, mode: str) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        return
    if mode == "symlink":
        os.symlink(src.resolve(), dst)
    elif mode == "copy":
        shutil.copy2(src, dst)
    else:
        raise ValueError(f"mode must be symlink|copy, got {mode}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--images_root", type=Path, required=True, help="Root containing rel_path files (e.g., data/sop_poc/images)")
    ap.add_argument("--manifest_csv", type=Path, required=True, help="CSV with column rel_path")
    ap.add_argument("--out_dir", type=Path, required=True, help="Output dir to materialize (e.g., data/sop_poc/gallery)")
    ap.add_argument("--mode", choices=["symlink", "copy"], default="symlink")
    args = ap.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)

    missing = 0
    total = 0
    with args.manifest_csv.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        if not reader.fieldnames or "rel_path" not in reader.fieldnames:
            raise ValueError("manifest_csv must have a rel_path column")
        for row in reader:
            rel = Path(row["rel_path"])
            src = args.images_root / rel
            dst = args.out_dir / rel
            total += 1
            if not src.exists():
                missing += 1
                continue
            link_or_copy(src, dst, args.mode)

    print(f"Materialized {total - missing}/{total} files into {args.out_dir}")
    if missing:
        print(f"[WARN] Missing {missing} files (check rel_path vs images_root layout)")


if __name__ == "__main__":
    main()
