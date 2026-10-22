from __future__ import annotations

import argparse
import json
import shutil
import time
from pathlib import Path

import cv2
import numpy as np
import torch
from ultralytics.data.annotator import auto_annotate


def reset_dir(path: Path) -> None:
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def extract_frames(
    video_path: str,
    out_dir: Path,
    frame_stride: int = 1,
    resize_to: tuple[int, int] | None = None,
) -> tuple[int, float, tuple[int, int] | None, int, float]:
    out_dir.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    if fps <= 0:
        fps = 30.0
    total_source_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)

    total_idx = 0
    saved_idx = 0
    frame_size: tuple[int, int] | None = None

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        if total_idx % frame_stride == 0:
            if resize_to is not None:
                frame = cv2.resize(frame, resize_to)

            h, w = frame.shape[:2]
            frame_size = (w, h)
            out_path = out_dir / f"frame_{saved_idx:06d}.jpg"
            cv2.imwrite(str(out_path), frame)
            saved_idx += 1

        total_idx += 1

    cap.release()

    if saved_idx == 0:
        raise RuntimeError("No frames were extracted. Check video path or stride.")

    duration_s = (total_source_frames / fps) if total_source_frames > 0 else (saved_idx / fps)
    return saved_idx, fps, frame_size, total_source_frames, duration_s


def parse_yolo_segmentation_label(label_path: Path, img_w: int, img_h: int) -> list[dict]:
    objects: list[dict] = []
    if not label_path.exists():
        return objects

    with label_path.open("r", encoding="utf-8") as f:
        lines = [line.strip() for line in f if line.strip()]

    for line in lines:
        parts = line.split()
        if len(parts) < 7:  # class + at least 3 points
            continue

        class_id = int(float(parts[0]))
        coords = [float(v) for v in parts[1:]]
        if len(coords) % 2 != 0:
            continue

        pts = []
        for i in range(0, len(coords), 2):
            x = int(round(coords[i] * img_w))
            y = int(round(coords[i + 1] * img_h))
            pts.append([x, y])

        pts_np = np.array(pts, dtype=np.int32)
        if len(pts_np) >= 3:
            objects.append({"class_id": class_id, "points": pts_np})

    return objects


def draw_segmentation_overlay(image: np.ndarray, objects: list[dict], alpha: float = 0.35) -> np.ndarray:
    overlay = image.copy()
    for obj in objects:
        pts = obj["points"]
        class_id = obj["class_id"]

        cv2.fillPoly(overlay, [pts], color=(0, 255, 0))
        cv2.polylines(image, [pts], isClosed=True, color=(0, 255, 0), thickness=2)

        x, y = pts[0]
        cv2.putText(
            image,
            f"class {class_id}",
            (x, max(20, y - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 0),
            2,
            cv2.LINE_AA,
        )

    return cv2.addWeighted(overlay, alpha, image, 1 - alpha, 0)


def build_preview_video(
    frames_dir: Path,
    labels_dir: Path,
    output_video_path: Path,
    fps: float,
) -> tuple[list[float], int]:
    frame_paths = sorted(frames_dir.glob("*.jpg"))
    if not frame_paths:
        raise RuntimeError("No extracted frames found for preview generation.")

    first_img = cv2.imread(str(frame_paths[0]))
    if first_img is None:
        raise RuntimeError(f"Could not read frame: {frame_paths[0]}")

    h, w = first_img.shape[:2]
    writer = cv2.VideoWriter(
        str(output_video_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (w, h),
    )
    if not writer.isOpened():
        raise RuntimeError(f"Could not open video writer: {output_video_path}")

    per_frame_ms: list[float] = []
    written = 0
    for frame_path in frame_paths:
        t0 = time.perf_counter()
        img = cv2.imread(str(frame_path))
        if img is None:
            continue
        label_path = labels_dir / f"{frame_path.stem}.txt"
        objs = parse_yolo_segmentation_label(label_path, w, h)
        vis = draw_segmentation_overlay(img, objs, alpha=0.30)
        writer.write(vis)
        written += 1
        per_frame_ms.append((time.perf_counter() - t0) * 1000.0)

    writer.release()
    return per_frame_ms, written


def _resolve_device(device: str) -> str:
    if device != "auto":
        return device
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def _find_videos(root: Path) -> list[Path]:
    patterns = ("*.mp4", "*.avi", "*.webm", "*.mov")
    out: list[Path] = []
    for pat in patterns:
        out.extend(sorted(root.rglob(pat)))
    return out


def _read_label_lines(label_path: Path) -> list[str]:
    if not label_path.exists():
        return []
    with label_path.open("r", encoding="utf-8") as f:
        return [ln.strip() for ln in f if ln.strip()]


def _write_label_lines(label_path: Path, lines: list[str]) -> None:
    label_path.parent.mkdir(parents=True, exist_ok=True)
    with label_path.open("w", encoding="utf-8") as f:
        if lines:
            f.write("\n".join(lines) + "\n")
        else:
            f.write("")


def temporal_fill_empty_labels(labels_dir: Path, frame_count: int, max_gap: int = 5) -> dict:
    """
    Lightweight temporal propagation for consistency:
    for empty frames, borrow labels from nearest non-empty frame within max_gap.
    """
    label_lines: list[list[str]] = []
    for i in range(frame_count):
        label_lines.append(_read_label_lines(labels_dir / f"frame_{i:06d}.txt"))

    initially_empty = sum(1 for lines in label_lines if not lines)
    filled = 0

    for i in range(frame_count):
        if label_lines[i]:
            continue

        prev_idx = None
        for p in range(i - 1, max(-1, i - max_gap - 1), -1):
            if label_lines[p]:
                prev_idx = p
                break

        next_idx = None
        for n in range(i + 1, min(frame_count, i + max_gap + 1)):
            if label_lines[n]:
                next_idx = n
                break

        chosen_idx = None
        if prev_idx is not None and next_idx is not None:
            chosen_idx = prev_idx if (i - prev_idx) <= (next_idx - i) else next_idx
        elif prev_idx is not None:
            chosen_idx = prev_idx
        elif next_idx is not None:
            chosen_idx = next_idx

        if chosen_idx is not None:
            label_lines[i] = list(label_lines[chosen_idx])
            filled += 1

    # Persist
    for i, lines in enumerate(label_lines):
        _write_label_lines(labels_dir / f"frame_{i:06d}.txt", lines)

    finally_empty = sum(1 for lines in label_lines if not lines)
    return {
        "temporal_fill_enabled": True,
        "temporal_max_gap": max_gap,
        "frames_initially_empty": initially_empty,
        "frames_filled": filled,
        "frames_still_empty": finally_empty,
    }


def _process_one_video(
    video_path: Path,
    out_dir: Path,
    frame_stride: int,
    resize_to: tuple[int, int] | None,
    det_model: str,
    sam_model: str,
    device: str,
    conf: float,
    iou: float,
    imgsz: int,
    max_det: int,
    classes: list[int] | None,
    keep_frames: bool,
    temporal_fill: bool,
    temporal_max_gap: int,
) -> dict:
    frames_dir = out_dir / "frames"
    labels_dir = out_dir / "labels_auto"
    preview_video_path = out_dir / "annotated_preview.mp4"
    report_path = out_dir / "report.json"

    reset_dir(out_dir)
    frames_dir.mkdir(parents=True, exist_ok=True)
    labels_dir.mkdir(parents=True, exist_ok=True)

    t_extract_0 = time.perf_counter()
    saved_count, fps, frame_size, source_total_frames, video_length_s = extract_frames(
        video_path=str(video_path),
        out_dir=frames_dir,
        frame_stride=frame_stride,
        resize_to=resize_to,
    )
    t_extract_ms = (time.perf_counter() - t_extract_0) * 1000.0

    def _run_auto_annotate() -> None:
        auto_annotate(
            data=str(frames_dir),
            det_model=det_model,
            sam_model=sam_model,
            device=device,
            conf=conf,
            iou=iou,
            imgsz=imgsz,
            max_det=max_det,
            classes=classes,
            output_dir=str(labels_dir),
        )

    t_annot_0 = time.perf_counter()
    try:
        _run_auto_annotate()
    except RuntimeError as e:
        msg = str(e)
        # Common corrupt checkpoint symptom:
        # "PytorchStreamReader failed reading zip archive: failed finding central directory"
        if "failed finding central directory" in msg.lower():
            sam_path = Path(sam_model)
            if sam_path.exists() and sam_path.is_file():
                print(f"[WARN] Corrupted SAM checkpoint detected. Deleting and retrying: {sam_path}")
                sam_path.unlink(missing_ok=True)
                _run_auto_annotate()
            else:
                raise
        else:
            raise
    t_annot_ms = (time.perf_counter() - t_annot_0) * 1000.0

    t_preview_0 = time.perf_counter()
    preview_fps = fps if frame_stride == 1 else max(1.0, fps / frame_stride)
    per_frame_render_ms, preview_written_frames = build_preview_video(
        frames_dir=frames_dir,
        labels_dir=labels_dir,
        output_video_path=preview_video_path,
        fps=preview_fps,
    )
    t_preview_ms = (time.perf_counter() - t_preview_0) * 1000.0

    temporal_stats = {
        "temporal_fill_enabled": False,
        "temporal_max_gap": temporal_max_gap,
        "frames_initially_empty": None,
        "frames_filled": 0,
        "frames_still_empty": None,
    }
    if temporal_fill:
        t_fill_0 = time.perf_counter()
        temporal_stats = temporal_fill_empty_labels(
            labels_dir=labels_dir,
            frame_count=saved_count,
            max_gap=max(1, temporal_max_gap),
        )
        t_fill_ms = (time.perf_counter() - t_fill_0) * 1000.0
    else:
        t_fill_ms = 0.0

    # Count labels quickly for diagnostics.
    total_label_lines = 0
    for lp in labels_dir.glob("*.txt"):
        with lp.open("r", encoding="utf-8") as f:
            total_label_lines += sum(1 for line in f if line.strip())

    report = {
        "video_path": str(video_path),
        "work_dir": str(out_dir),
        "source_fps": fps,
        "source_total_frames": source_total_frames if source_total_frames > 0 else None,
        "source_video_length_s": video_length_s,
        "frame_stride": frame_stride,
        "resize_to": list(resize_to) if resize_to is not None else None,
        "extracted_frames": saved_count,
        "preview_fps": preview_fps,
        "preview_written_frames": preview_written_frames,
        "labels_total_objects": total_label_lines,
        "timing_ms": {
            "extract_total_ms": t_extract_ms,
            "annotate_total_ms": t_annot_ms,
            "temporal_fill_total_ms": t_fill_ms,
            "preview_total_ms": t_preview_ms,
            "total_pipeline_ms": t_extract_ms + t_annot_ms + t_fill_ms + t_preview_ms,
            # auto_annotate does not expose per-frame timing, so this is an estimate.
            "annotate_avg_per_extracted_frame_ms": (t_annot_ms / saved_count) if saved_count > 0 else 0.0,
            "preview_per_frame_render_ms": per_frame_render_ms,
            "preview_avg_per_frame_ms": (sum(per_frame_render_ms) / len(per_frame_render_ms)) if per_frame_render_ms else 0.0,
        },
        "temporal_fill": temporal_stats,
        "paths": {
            "frames_dir": str(frames_dir),
            "labels_dir": str(labels_dir),
            "preview_video": str(preview_video_path),
            "report_json": str(report_path),
        },
    }
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    if not keep_frames:
        shutil.rmtree(frames_dir, ignore_errors=True)
        report["paths"]["frames_dir"] = None
        report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Auto-annotate video frames with YOLO + SAM.")
    parser.add_argument("--video", default="", help="Input video path")
    parser.add_argument("--video-dir", default="", help="Process all videos recursively under this directory")
    parser.add_argument("--work-dir", default="artifacts/video_auto_annotate_work", help="Output working directory")
    parser.add_argument("--frame-stride", type=int, default=1, help="Use every Nth frame")
    parser.add_argument("--resize-w", type=int, default=0, help="Resize width (0 = no resize)")
    parser.add_argument("--resize-h", type=int, default=0, help="Resize height (0 = no resize)")
    parser.add_argument("--det-model", default="yolo11x.pt", help="Detection model path/name")
    parser.add_argument("--sam-model", default="mobile_sam.pt", help="SAM model path/name")
    parser.add_argument("--device", default="auto", help="auto/cuda/cpu/mps")
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--iou", type=float, default=0.45)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--max-det", type=int, default=300)
    parser.add_argument("--classes", default="", help="Comma-separated class ids, empty=all")
    parser.add_argument(
        "--match-input-video",
        action="store_true",
        help="Force output preview to match input frame count/FPS/resolution (implies --frame-stride 1 and no resize).",
    )
    parser.add_argument(
        "--keep-frames",
        action="store_true",
        help="Keep extracted frame jpgs. Default is to delete them after preview/report.",
    )
    parser.add_argument(
        "--temporal-fill",
        action="store_true",
        help="Fill empty-frame labels from nearest labeled frames (simple temporal propagation).",
    )
    parser.add_argument(
        "--temporal-max-gap",
        type=int,
        default=5,
        help="Max frame distance for temporal fill borrowing.",
    )
    args = parser.parse_args()

    if not args.video and not args.video_dir:
        raise ValueError("Provide either --video or --video-dir")

    work_dir = Path(args.work_dir)

    resize_to = None
    if args.resize_w > 0 and args.resize_h > 0:
        resize_to = (args.resize_w, args.resize_h)

    classes = None
    if args.classes.strip():
        classes = [int(x.strip()) for x in args.classes.split(",") if x.strip()]

    device = _resolve_device(args.device)
    frame_stride = max(1, args.frame_stride)
    if args.match_input_video:
        frame_stride = 1
        resize_to = None

    videos: list[Path] = []
    if args.video:
        vp = Path(args.video)
        if not vp.exists():
            raise FileNotFoundError(f"Video not found: {vp}")
        videos = [vp]
    else:
        vd = Path(args.video_dir)
        if not vd.exists() or not vd.is_dir():
            raise FileNotFoundError(f"video-dir not found: {vd}")
        videos = _find_videos(vd)
        if not videos:
            raise RuntimeError(f"No videos found under: {vd}")

    run_reports = []
    failed_reports = []
    for idx, video_path in enumerate(videos, start=1):
        video_stem = video_path.stem
        if video_stem.endswith(".avi") and video_path.suffix.lower() == ".mp4":
            video_stem = Path(video_stem).stem
        out_dir = work_dir / video_stem

        print(f"\n[{idx}/{len(videos)}] Processing: {video_path}")
        try:
            report = _process_one_video(
                video_path=video_path,
                out_dir=out_dir,
                frame_stride=frame_stride,
                resize_to=resize_to,
                det_model=args.det_model,
                sam_model=args.sam_model,
                device=device,
                conf=args.conf,
                iou=args.iou,
                imgsz=args.imgsz,
                max_det=args.max_det,
                classes=classes,
                keep_frames=args.keep_frames,
                temporal_fill=args.temporal_fill,
                temporal_max_gap=args.temporal_max_gap,
            )
            run_reports.append(report)
            print(f"  -> Preview: {report['paths']['preview_video']}")
            print(f"  -> Report : {report['paths']['report_json']}")
            if report.get("temporal_fill", {}).get("temporal_fill_enabled"):
                tf = report["temporal_fill"]
                print(
                    "  -> Temporal fill:"
                    f" initially_empty={tf['frames_initially_empty']},"
                    f" filled={tf['frames_filled']},"
                    f" still_empty={tf['frames_still_empty']}"
                )
        except Exception as e:  # keep batch running
            err = {
                "video_path": str(video_path),
                "error_type": type(e).__name__,
                "error": str(e),
            }
            failed_reports.append(err)
            print(f"[ERROR] Failed on {video_path}: {type(e).__name__}: {e}")

    summary = {
        "videos_processed": len(run_reports),
        "videos_failed": len(failed_reports),
        "device_used": device,
        "det_model": args.det_model,
        "sam_model": args.sam_model,
        "frame_stride": frame_stride,
        "match_input_video": bool(args.match_input_video),
        "temporal_fill": bool(args.temporal_fill),
        "temporal_max_gap": int(args.temporal_max_gap),
        "reports": [r["paths"]["report_json"] for r in run_reports],
        "failures": failed_reports,
    }
    summary_path = work_dir / "summary.json"
    work_dir.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"\nDone. Summary: {summary_path}")


if __name__ == "__main__":
    main()

