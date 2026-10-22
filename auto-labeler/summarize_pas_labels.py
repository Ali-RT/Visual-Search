#!/usr/bin/env python3
"""
Walk auto-labeler/data for PAS label CSVs and summarize Which object × class counts.

By default uses one file per label stem (e.g. 20190311_LB_XO2617_012_165618) at the
highest V* version found, so nested duplicate folders do not double-count.
"""
from __future__ import annotations

import argparse
import csv
from collections import Counter
from dataclasses import dataclass
from pathlib import Path


def parse_version(path: Path) -> int:
    # ..._PAS_m_label_V2.csv -> 2
    token = path.stem.split("_PAS_m_label_V")[-1]
    return int(token) if token.isdigit() else -1


def stem_from_label_path(path: Path) -> str:
    return path.name.split("_PAS_m_label_")[0]


def class_key(row: dict[str, str]) -> str:
    for k in row:
        if k and k.startswith("class ("):
            return row[k].strip()
    raise KeyError("class column not found")


def load_rows(csv_path: Path) -> list[dict[str, str]]:
    with csv_path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        return [
            {(k.strip() if k else k): (v.strip() if isinstance(v, str) else v) for k, v in raw.items()}
            for raw in reader
        ]


def pick_latest_per_stem(paths: list[Path]) -> dict[str, Path]:
    """For each label stem, keep the path with the largest V number."""
    best: dict[str, tuple[int, Path]] = {}
    for p in paths:
        stem = stem_from_label_path(p)
        ver = parse_version(p)
        cur = best.get(stem)
        if cur is None or ver > cur[0] or (ver == cur[0] and str(p) < str(cur[1])):
            best[stem] = (ver, p)
    return {s: t[1] for s, t in best.items()}


@dataclass(frozen=True)
class PasLabelSummary:
    """Aggregated counts from PAS label CSVs under a data root."""

    data_root: Path
    every_file: bool
    all_csv_paths: list[Path]
    chosen_csv_paths: list[Path]
    pair_counts: Counter[tuple[str, str]]
    class_counts: Counter[str]
    object_counts: Counter[str]
    total_rows: int
    files_with_rows: int


def aggregate_pas_labels(data_root: Path, *, every_file: bool = False) -> PasLabelSummary:
    """
    Scan data_root for *_PAS_m_label_V*.csv and aggregate Which object × class.

    If every_file is False, only the highest V* per label stem is used.
    """
    root = data_root
    if not root.is_dir():
        raise FileNotFoundError(f"Not a directory: {root}")

    all_paths = sorted(root.glob("**/*_PAS_m_label_V*.csv"))
    if not all_paths:
        raise FileNotFoundError(f"No *_PAS_m_label_V*.csv under {root}")

    if every_file:
        chosen = all_paths
    else:
        latest = pick_latest_per_stem(all_paths)
        chosen = sorted(latest.values(), key=lambda p: str(p))

    pair_counts: Counter[tuple[str, str]] = Counter()
    class_counts: Counter[str] = Counter()
    object_counts: Counter[str] = Counter()
    files_with_rows = 0
    total_rows = 0

    for csv_path in chosen:
        rows = load_rows(csv_path)
        if not rows:
            continue
        files_with_rows += 1
        for r in rows:
            total_rows += 1
            obj = (r.get("Which object") or "").strip() or "(empty)"
            try:
                cls = class_key(r)
            except KeyError:
                cls = "(no class column)"
            pair_counts[(obj, cls)] += 1
            class_counts[cls] += 1
            object_counts[obj] += 1

    return PasLabelSummary(
        data_root=root,
        every_file=every_file,
        all_csv_paths=all_paths,
        chosen_csv_paths=chosen,
        pair_counts=pair_counts,
        class_counts=class_counts,
        object_counts=object_counts,
        total_rows=total_rows,
        files_with_rows=files_with_rows,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize PAS label CSVs under auto-labeler/data")
    parser.add_argument(
        "--data-root",
        type=Path,
        default=Path(__file__).resolve().parent / "data",
        help="Root folder to search (default: auto-labeler/data)",
    )
    parser.add_argument(
        "--every-file",
        action="store_true",
        help="Process every matching CSV (including V1+V2 and nested duplicates). Default: latest V per stem only.",
    )
    parser.add_argument(
        "--csv-out",
        type=Path,
        default=None,
        help="Optional path to write summary table as CSV (which_object, class, count)",
    )
    args = parser.parse_args()

    try:
        summary = aggregate_pas_labels(args.data_root, every_file=args.every_file)
    except (FileNotFoundError, OSError) as e:
        raise SystemExit(str(e)) from e

    if summary.every_file:
        print(f"Mode: every file ({len(summary.chosen_csv_paths)} CSVs)\n")
    else:
        print(
            f"Mode: latest V* per label stem ({len(summary.chosen_csv_paths)} CSVs, "
            f"{len(summary.all_csv_paths)} files on disk)\n"
        )

    pair_counts = summary.pair_counts
    class_counts = summary.class_counts
    object_counts = summary.object_counts

    print(f"Label files read: {summary.files_with_rows}  |  Total rows: {summary.total_rows}\n")

    print("=== Count by (Which object, class) ===")
    w_obj = max(len("Which object"), max((len(o) for o, _ in pair_counts), default=0))
    w_cls = max(len("class"), max((len(c) for _, c in pair_counts), default=0))
    header = f"{'Which object':<{w_obj}}  {'class':<{w_cls}}  count"
    print(header)
    print("-" * len(header))
    for (obj, cls), n in sorted(pair_counts.items(), key=lambda x: (-x[1], x[0][0], x[0][1])):
        print(f"{obj:<{w_obj}}  {cls:<{w_cls}}  {n}")

    print("\n=== Count by class only ===")
    for cls, n in sorted(class_counts.items(), key=lambda x: (-x[1], x[0])):
        print(f"  {cls}: {n}")

    print("\n=== Count by Which object only ===")
    for obj, n in sorted(object_counts.items(), key=lambda x: (-x[1], x[0])):
        print(f"  {obj}: {n}")

    if args.csv_out:
        args.csv_out.parent.mkdir(parents=True, exist_ok=True)
        with args.csv_out.open("w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["which_object", "class", "count"])
            for (obj, cls), n in sorted(pair_counts.items()):
                w.writerow([obj, cls, n])
        print(f"\nWrote {args.csv_out}")


if __name__ == "__main__":
    main()
