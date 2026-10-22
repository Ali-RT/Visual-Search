#!/usr/bin/env python3
"""Convert a labels.json file with one RLE mask per frame into a YOLO segmentation dataset.

Expected input schema matches the uploaded file:
- metadata.frame_width / metadata.frame_height
- frames[frame_id].has_detection
- frames[frame_id].mask_rle = {"size": [H, W], "counts": [int, int, ...]}

This script assumes each JSON file represents ONE tracked instance per frame.
Because the uploaded file prompt can contain multiple class names while each frame stores only one mask,
you must provide the target class for this file via --class-id / --class-name.

Single file (random train/val split or fixed split):

  python convert_labels_to_yolo_seg.py \\
      --dataset-layout single \\
      --labels /path/to/labels.json \\
      --video /path/to/video.mp4 \\
      --out /path/to/yolo_dataset \\
      --class-id 0 \\
      --class-name drain_cover

Pre-split folder (outputs/train/... and outputs/test/...; test maps to YOLO val/):

  python convert_labels_to_yolo_seg.py \\
      --dataset-layout train-test \\
      --root /path/to/outputs \\
      --out /path/to/yolo_dataset \\
      --class-id 0 \\
      --class-name drain_cover

Expected layout for train-test mode::

  outputs/
    train/
      seq_a/labels.json   +  *.mp4   (or frames/ with frame_*.jpg)
    test/
      seq_b/labels.json   +  *.mp4

Frames next to labels.json (same folder):

  python convert_labels_to_yolo_seg.py \\
      --labels /path/to/labels.json \\
      --frames-dir /path/to/frames \\
      --out /path/to/yolo_dataset \\
      --class-id 0 \\
      --class-name drain_cover
"""
from __future__ import annotations

import argparse
import json
import math
import random
from pathlib import Path
from typing import Dict, List, Literal, Sequence, Tuple

import cv2
import numpy as np
import yaml

Point = Tuple[float, float]
SplitMode = Literal["auto", "train", "val"]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--dataset-layout",
        choices=("single", "train-test"),
        default="single",
        help="single: one labels.json; train-test: scan --root/train and --root/test",
    )
    p.add_argument("--labels", default=None, help="Path to labels.json (single layout)")
    p.add_argument("--out", required=True, help="Output dataset root")
    p.add_argument("--video", default=None, help="Source video path. Required unless --frames-dir is used")
    p.add_argument("--frames-dir", default=None, help="Directory of already extracted frames")
    p.add_argument(
        "--frame-pattern",
        default="frame_{frame:06d}.jpg",
        help="Filename pattern used when --frames-dir is provided",
    )
    p.add_argument("--class-id", type=int, required=True, help="YOLO class id to assign to all masks in this JSON")
    p.add_argument("--class-name", required=True, help="YOLO class name for class-id")
    p.add_argument(
        "--split-mode",
        choices=("auto", "train", "val"),
        default="auto",
        help="auto: random train/val by --val-ratio; train|val: send all frames to that split",
    )
    p.add_argument("--val-ratio", type=float, default=0.2, help="Validation split ratio (split-mode=auto only)")
    p.add_argument("--seed", type=int, default=42, help="Random seed for split")
    p.add_argument("--min-area", type=int, default=20, help="Discard masks smaller than this many pixels")
    p.add_argument(
        "--approx-epsilon",
        type=float,
        default=0.002,
        help="Polygon simplification factor as fraction of contour perimeter",
    )
    p.add_argument(
        "--copy-images",
        action="store_true",
        help="Copy frames into dataset/images/... instead of symlinking them",
    )
    p.add_argument("--image-ext", default=".jpg", help="Output image extension when extracting from video")
    p.add_argument(
        "--stem-prefix",
        default="",
        help="Prefix for image/label stems (avoid collisions when merging multiple JSONs into one dataset)",
    )
    # train-test layout
    p.add_argument(
        "--root",
        default=None,
        help="Root containing train/ and test/ subfolders (dataset-layout=train-test)",
    )
    p.add_argument("--train-subdir", default="train", help="Subfolder under --root for training split")
    p.add_argument("--test-subdir", default="test", help="Subfolder under --root; written to YOLO val/")
    return p.parse_args()


def load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def decode_uncompressed_rle(size: Sequence[int], counts: Sequence[int]) -> np.ndarray:
    """Decode COCO-style uncompressed RLE counts into a binary mask.

    COCO RLE is flattened in Fortran order (column-major), starting with zeros run length.
    """
    h, w = int(size[0]), int(size[1])
    total = h * w
    flat = np.zeros(total, dtype=np.uint8)
    idx = 0
    val = 0
    for run_len in counts:
        run_len = int(run_len)
        if run_len < 0:
            raise ValueError(f"Negative RLE run length: {run_len}")
        end = idx + run_len
        if end > total:
            raise ValueError(f"RLE overruns mask size: {end} > {total}")
        if val == 1:
            flat[idx:end] = 1
        idx = end
        val = 1 - val
    if idx != total:
        raise ValueError(f"RLE underfills mask size: {idx} != {total}")
    return flat.reshape((h, w), order="F")


def mask_to_polygon(mask: np.ndarray, approx_epsilon: float) -> List[Point]:
    mask_u8 = (mask > 0).astype(np.uint8) * 255
    contours, _ = cv2.findContours(mask_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return []
    contour = max(contours, key=cv2.contourArea)
    peri = cv2.arcLength(contour, True)
    eps = approx_epsilon * peri
    approx = cv2.approxPolyDP(contour, eps, True)
    pts = approx.reshape(-1, 2)
    if len(pts) < 3:
        pts = contour.reshape(-1, 2)
    return [(float(x), float(y)) for x, y in pts]


def normalize_polygon(points: Sequence[Point], width: int, height: int) -> List[float]:
    out: List[float] = []
    for x, y in points:
        xn = min(max(x / width, 0.0), 1.0)
        yn = min(max(y / height, 0.0), 1.0)
        out.extend([xn, yn])
    return out


def ensure_dirs(root: Path) -> Dict[str, Path]:
    paths = {
        "images_train": root / "images" / "train",
        "images_val": root / "images" / "val",
        "labels_train": root / "labels" / "train",
        "labels_val": root / "labels" / "val",
    }
    for p in paths.values():
        p.mkdir(parents=True, exist_ok=True)
    return paths


def extract_frame(cap: cv2.VideoCapture, frame_index: int) -> np.ndarray:
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
    ok, frame = cap.read()
    if not ok or frame is None:
        raise RuntimeError(f"Failed to read frame {frame_index}")
    return frame


def save_or_link_image(src: Path, dst: Path, copy_images: bool) -> None:
    if dst.exists() or dst.is_symlink():
        dst.unlink()
    if copy_images:
        import shutil

        shutil.copy2(src, dst)
    else:
        dst.symlink_to(src.resolve())


def write_label(label_path: Path, class_id: int, polygon_norm: Sequence[float]) -> None:
    values = [str(class_id)] + [f"{v:.6f}" for v in polygon_norm]
    label_path.write_text(" ".join(values) + "\n", encoding="utf-8")


def write_yaml(root: Path, class_id: int, class_name: str) -> None:
    names = {int(class_id): class_name}
    payload = {
        "path": str(root.resolve()),
        "train": "images/train",
        "val": "images/val",
        "names": names,
    }
    with (root / "data.yaml").open("w", encoding="utf-8") as f:
        yaml.safe_dump(payload, f, sort_keys=False)


def find_video_or_frames_dir(labels_path: Path, meta: dict) -> tuple[Path | None, Path | None]:
    """Return (video_path, frames_dir) — at most one should be used."""
    d = labels_path.parent
    exts = ("*.mp4", "*.MP4", "*.avi", "*.AVI", "*.mov", "*.MOV", "*.mkv")
    vids: list[Path] = []
    for pat in exts:
        vids.extend(sorted(d.glob(pat)))
    if vids:
        return vids[0], None
    frames_sub = d / "frames"
    if frames_sub.is_dir() and list(frames_sub.glob("*.jpg")):
        return None, frames_sub
    vp = meta.get("video_path")
    if vp and Path(vp).is_file():
        return Path(vp), None
    return None, None


def _split_assignments(
    n: int, split_mode: SplitMode, val_ratio: float, seed: int
) -> List[Literal["train", "val"]]:
    if split_mode == "train":
        return ["train"] * n
    if split_mode == "val":
        return ["val"] * n
    random.seed(seed)
    indices = list(range(n))
    random.shuffle(indices)
    val_n = max(1, round(len(indices) * val_ratio)) if len(indices) > 1 else 0
    val_positions = set(indices[:val_n])
    out: List[Literal["train", "val"]] = []
    for pos in range(n):
        out.append("val" if pos in val_positions else "train")
    return out


def process_labels_json(
    labels_path: Path,
    out_root: Path,
    dirs: Dict[str, Path],
    *,
    video_path: Path | None,
    frames_dir: Path | None,
    frame_pattern: str,
    class_id: int,
    split_mode: SplitMode,
    val_ratio: float,
    seed: int,
    min_area: int,
    approx_epsilon: float,
    copy_images: bool,
    image_ext: str,
    stem_prefix: str,
) -> tuple[int, int]:
    data = load_json(labels_path)
    meta = data["metadata"]
    frames = data["frames"]
    width = int(meta["frame_width"])
    height = int(meta["frame_height"])

    detected: list[tuple[int, dict]] = []
    for key, rec in frames.items():
        if rec.get("has_detection"):
            idx = int(rec["frame_index"])
            detected.append((idx, rec))
    detected.sort(key=lambda x: x[0])
    if not detected:
        raise ValueError(f"No positive frames found in labels file: {labels_path}")

    splits = _split_assignments(len(detected), split_mode, val_ratio, seed)

    cap = None
    if video_path is not None:
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise RuntimeError(f"Could not open video: {video_path}")
    if frames_dir is not None and not frames_dir.exists():
        raise FileNotFoundError(frames_dir)

    kept = 0
    skipped = 0
    for (frame_index, rec), split in zip(detected, splits):
        rle = rec.get("mask_rle")
        if not rle:
            skipped += 1
            continue
        mask = decode_uncompressed_rle(rle["size"], rle["counts"])
        area = int(mask.sum())
        if area < min_area:
            skipped += 1
            continue
        polygon = mask_to_polygon(mask, approx_epsilon)
        if len(polygon) < 3:
            skipped += 1
            continue
        polygon_norm = normalize_polygon(polygon, width, height)
        if len(polygon_norm) < 6:
            skipped += 1
            continue

        split_key = f"images_{split}"
        lbl_key = f"labels_{split}"
        base = f"frame_{frame_index:06d}"
        stem = f"{stem_prefix}_{base}" if stem_prefix else base
        img_dst = dirs[split_key] / f"{stem}{image_ext}"
        lbl_dst = dirs[lbl_key] / f"{stem}.txt"

        if frames_dir is not None:
            src = frames_dir / frame_pattern.format(frame=frame_index)
            if not src.exists():
                raise FileNotFoundError(f"Frame image not found: {src}")
            save_or_link_image(src, img_dst, copy_images)
        else:
            assert cap is not None
            frame = extract_frame(cap, frame_index)
            ok = cv2.imwrite(str(img_dst), frame)
            if not ok:
                raise RuntimeError(f"Failed to write frame image: {img_dst}")

        write_label(lbl_dst, class_id, polygon_norm)
        kept += 1

    if cap is not None:
        cap.release()

    return kept, skipped


def collect_labels_json_under(root: Path) -> list[Path]:
    return sorted(root.rglob("labels.json"))


def main() -> None:
    args = parse_args()
    out_root = Path(args.out)

    if args.dataset_layout == "single":
        if not args.labels:
            raise ValueError("--labels is required when --dataset-layout single")
        labels_path = Path(args.labels)
        if not args.video and not args.frames_dir:
            raise ValueError("Provide either --video or --frames-dir")

        video_path = Path(args.video) if args.video else None
        frames_dir = Path(args.frames_dir) if args.frames_dir else None
        dirs = ensure_dirs(out_root)

        kept, skipped = process_labels_json(
            labels_path,
            out_root,
            dirs,
            video_path=video_path,
            frames_dir=frames_dir,
            frame_pattern=args.frame_pattern,
            class_id=args.class_id,
            split_mode=args.split_mode,  # type: ignore[arg-type]
            val_ratio=args.val_ratio,
            seed=args.seed,
            min_area=args.min_area,
            approx_epsilon=args.approx_epsilon,
            copy_images=args.copy_images,
            image_ext=args.image_ext,
            stem_prefix=args.stem_prefix.strip(),
        )
        write_yaml(out_root, args.class_id, args.class_name)

        print(
            json.dumps(
                {
                    "output_root": str(out_root.resolve()),
                    "kept_frames": kept,
                    "skipped_frames": skipped,
                    "train_images": len(list((out_root / "images" / "train").glob("*"))),
                    "val_images": len(list((out_root / "images" / "val").glob("*"))),
                    "data_yaml": str((out_root / "data.yaml").resolve()),
                },
                indent=2,
            )
        )
        return

    # train-test layout
    if not args.root:
        raise ValueError("--root is required when --dataset-layout train-test")
    root = Path(args.root)
    train_root = root / args.train_subdir
    test_root = root / args.test_subdir
    if not train_root.is_dir():
        raise FileNotFoundError(f"Train directory not found: {train_root}")
    if not test_root.is_dir():
        raise FileNotFoundError(f"Test directory not found: {test_root}")

    train_jsons = collect_labels_json_under(train_root)
    test_jsons = collect_labels_json_under(test_root)
    if not train_jsons and not test_jsons:
        raise ValueError(f"No labels.json under {train_root} or {test_root}")

    dirs = ensure_dirs(out_root)
    total_kept = 0
    total_skipped = 0
    errors: list[str] = []

    def run_one(
        labels_path: Path,
        yolo_split: Literal["train", "val"],
        stem_prefix: str,
    ) -> None:
        nonlocal total_kept, total_skipped
        data = load_json(labels_path)
        meta = data.get("metadata", {})
        vid, fd = find_video_or_frames_dir(labels_path, meta)
        if not vid and not fd:
            raise FileNotFoundError(
                f"No video (*.mp4, …) or frames/ next to {labels_path} and metadata.video_path not a local file"
            )
        k, s = process_labels_json(
            labels_path,
            out_root,
            dirs,
            video_path=vid,
            frames_dir=fd,
            frame_pattern=args.frame_pattern,
            class_id=args.class_id,
            split_mode=yolo_split,  # type: ignore[arg-type]
            val_ratio=args.val_ratio,
            seed=args.seed,
            min_area=args.min_area,
            approx_epsilon=args.approx_epsilon,
            copy_images=args.copy_images,
            image_ext=args.image_ext,
            stem_prefix=stem_prefix,
        )
        total_kept += k
        total_skipped += s

    def stem_from(labels_path: Path, split_root: Path, split_tag: str) -> str:
        rel = labels_path.parent.relative_to(split_root)
        parts = "_".join(rel.parts) if rel.parts else labels_path.parent.name
        return f"{split_tag}_{parts}"

    for jp in train_jsons:
        prefix = stem_from(jp, train_root, "tr")
        try:
            run_one(jp, "train", prefix)
        except Exception as e:
            errors.append(f"{jp}: {e}")

    for jp in test_jsons:
        prefix = stem_from(jp, test_root, "te")
        try:
            run_one(jp, "val", prefix)
        except Exception as e:
            errors.append(f"{jp}: {e}")

    write_yaml(out_root, args.class_id, args.class_name)

    print(
        json.dumps(
            {
                "output_root": str(out_root.resolve()),
                "kept_frames": total_kept,
                "skipped_frames": total_skipped,
                "train_labels_found": len(train_jsons),
                "test_labels_found": len(test_jsons),
                "train_images": len(list((out_root / "images" / "train").glob("*"))),
                "val_images": len(list((out_root / "images" / "val").glob("*"))),
                "data_yaml": str((out_root / "data.yaml").resolve()),
                "errors": errors,
            },
            indent=2,
        )
    )

    if errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
