#!/usr/bin/env python3
"""
Prepare a small SOP subset for a visual search POC.

Inputs (inside SOP root):
  - Ebay_train.txt
  - Ebay_test.txt
  - images/ (or equivalent, referenced by the txt files)

Outputs (inside out_dir):
  - images/                  (symlinks or copies)
  - manifests/train.csv      (rel_path,label)
  - manifests/query.csv      (rel_path,label)
  - manifests/gallery.csv    (rel_path,label)

Notes:
- For SOP evaluation, we typically use:
  query   = test split images
  gallery = test split images (same pool; evaluator must ignore self-match)
"""

from __future__ import annotations

import argparse
import csv
import os
import random
import shutil
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Tuple


@dataclass(frozen=True)
class SopRow:
    split: str          # "train" or "test"
    rel_path: str       # path relative to SOP root (as referenced in txt)
    class_id: int
    super_id: int | None


def _parse_sop_list(list_path: Path, split: str) -> List[SopRow]:
    """
    SOP list files commonly look like:
      <num_images>
      <image_id> <class_id> <super_class_id> <path>
    but some variants omit super_class_id.

    We parse as:
      path      = last token
      class_id  = token[1]
      super_id  = token[2] if present
    """
    rows: List[SopRow] = []
    with list_path.open("r", encoding="utf-8") as f:
        lines = [ln.strip() for ln in f if ln.strip()]

    # First line often is a count
    if lines and lines[0].split()[0].isdigit() and len(lines[0].split()) == 1:
        lines = lines[1:]

    for ln in lines:
        toks = ln.split()
        if len(toks) < 3:
            continue
        rel_path = toks[-1]
        try:
            class_id = int(toks[1])
        except ValueError:
            continue
        super_id = None
        if len(toks) >= 4:
            # common: image_id class_id super_id path
            try:
                super_id = int(toks[2])
            except ValueError:
                super_id = None

        rows.append(SopRow(split=split, rel_path=rel_path, class_id=class_id, super_id=super_id))
    return rows


def _safe_link_or_copy(src: Path, dst: Path, mode: str) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        return
    if mode == "symlink":
        os.symlink(src.resolve(), dst)
    elif mode == "copy":
        shutil.copy2(src, dst)
    else:
        raise ValueError(f"Unsupported mode: {mode}")


def _write_manifest(path: Path, rows: Iterable[SopRow]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["rel_path", "label"])
        for r in rows:
            w.writerow([r.rel_path, r.class_id])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sop_root", type=Path, required=True, help="Path to extracted Stanford_Online_Products directory")
    ap.add_argument("--out_dir", type=Path, required=True, help="Output dataset package directory")
    ap.add_argument("--mode", choices=["symlink", "copy"], default="symlink", help="How to materialize images/")
    ap.add_argument("--seed", type=int, default=13)

    # Subsampling controls
    ap.add_argument("--max_superclasses", type=int, default=5, help="If super_id is available, cap number of superclasses")
    ap.add_argument("--max_classes_per_super", type=int, default=120)
    ap.add_argument("--min_images_per_class", type=int, default=3)
    ap.add_argument("--max_images_per_class", type=int, default=10)

    args = ap.parse_args()
    rng = random.Random(args.seed)

    train_list = args.sop_root / "Ebay_train.txt"
    test_list = args.sop_root / "Ebay_test.txt"
    if not train_list.exists() or not test_list.exists():
        raise FileNotFoundError("Expected Ebay_train.txt and Ebay_test.txt in sop_root")

    train_rows = _parse_sop_list(train_list, split="train")
    test_rows = _parse_sop_list(test_list, split="test")
    all_rows = train_rows + test_rows

    # Group by (super_id, class_id) if super_id present; else by class_id only
    by_super: Dict[int | None, Dict[int, List[SopRow]]] = defaultdict(lambda: defaultdict(list))
    for r in all_rows:
        by_super[r.super_id][r.class_id].append(r)

    # Choose superclasses (if available)
    super_ids = list(by_super.keys())
    # Prefer non-None if present
    if any(s is not None for s in super_ids):
        super_ids = [s for s in super_ids if s is not None]
    super_ids.sort(key=lambda x: (x is None, x))

    if super_ids and args.max_superclasses and len(super_ids) > args.max_superclasses:
        super_ids = rng.sample(super_ids, k=args.max_superclasses)

    chosen: List[SopRow] = []
    for s in super_ids if super_ids else [None]:
        classes = list(by_super[s].keys())
        # Filter by minimum images per class
        classes = [cid for cid in classes if len(by_super[s][cid]) >= args.min_images_per_class]
        if not classes:
            continue

        if args.max_classes_per_super and len(classes) > args.max_classes_per_super:
            classes = rng.sample(classes, k=args.max_classes_per_super)

        for cid in classes:
            rows = by_super[s][cid]
            # Cap images per class (sample deterministically)
            if args.max_images_per_class and len(rows) > args.max_images_per_class:
                rows = rng.sample(rows, k=args.max_images_per_class)
            chosen.extend(rows)

    # Materialize images into out_dir/images/
    out_images = args.out_dir / "images"
    missing = 0
    for r in chosen:
        src = args.sop_root / r.rel_path
        dst = out_images / r.rel_path
        if not src.exists():
            missing += 1
            continue
        _safe_link_or_copy(src, dst, mode=args.mode)

    if missing:
        print(f"[WARN] Missing {missing} files referenced by SOP lists (paths may differ in your dump).")

    # Create manifests:
    # - train.csv uses chosen train split rows
    # - query.csv/gallery.csv use chosen test split rows (classic SOP retrieval evaluation)
    chosen_train = [r for r in chosen if r.split == "train"]
    chosen_test = [r for r in chosen if r.split == "test"]

    manifests = args.out_dir / "manifests"
    _write_manifest(manifests / "train.csv", chosen_train)
    _write_manifest(manifests / "query.csv", chosen_test)
    _write_manifest(manifests / "gallery.csv", chosen_test)

    print("Done.")
    print(f"Images:     {out_images}")
    print(f"Manifests:  {manifests}")
    print(f"Train rows: {len(chosen_train)} | Test(query/gallery) rows: {len(chosen_test)}")


if __name__ == "__main__":
    main()
