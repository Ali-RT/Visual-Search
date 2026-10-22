from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import cv2
import numpy as np
import torch
from ultralytics import YOLO


def _resolve_device(device: str) -> str:
    if device != "auto":
        return device
    if torch.cuda.is_available():
        return "0"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def _find_videos(root: Path) -> list[Path]:
    out: list[Path] = []
    for pat in ("*.mp4", "*.avi", "*.webm", "*.mov"):
        out.extend(sorted(root.rglob(pat)))
    return out


def _polygon_to_yolo_line(class_id: int, polygon: np.ndarray, w: int, h: int) -> str:
    vals: list[str] = []
    for x, y in polygon:
        nx = max(0.0, min(float(w - 1), float(x))) / float(w)
        ny = max(0.0, min(float(h - 1), float(y))) / float(h)
        vals.extend([f"{nx:.6f}", f"{ny:.6f}"])
    return f"{class_id} " + " ".join(vals)


def _write_frame_labels(
    out_txt: Path,
    masks_xy: list[np.ndarray],
    classes: list[int],
    w: int,
    h: int,
) -> int:
    lines: list[str] = []
    for poly, cls_id in zip(masks_xy, classes):
        if poly is None or len(poly) < 3:
            continue
        lines.append(_polygon_to_yolo_line(int(cls_id), np.asarray(poly), w, h))
    out_txt.parent.mkdir(parents=True, exist_ok=True)
    out_txt.write_text(("\n".join(lines) + "\n") if lines else "", encoding="utf-8")
    return len(lines)


def process_video(
    model: YOLO,
    video_path: Path,
    out_dir: Path,
    conf: float,
    iou: float,
    imgsz: int,
    device: str,
    classes_filter: list[int] | None,
    compare_side_by_side: bool = False,
) -> dict:
    labels_dir = out_dir / "labels_auto"
    preview_path = out_dir / "annotated_preview.mp4"
    compare_path = out_dir / "comparison.mp4"
    classes_json = out_dir / "classes.json"
    report_json = out_dir / "report.json"
    out_dir.mkdir(parents=True, exist_ok=True)
    labels_dir.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    fps = float(cap.get(cv2.CAP_PROP_FPS) or 30.0)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    if width <= 0 or height <= 0:
        raise RuntimeError(f"Invalid video dimensions: {video_path}")

    writer = cv2.VideoWriter(
        str(preview_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height),
    )
    if not writer.isOpened():
        raise RuntimeError(f"Could not open writer: {preview_path}")

    compare_writer: cv2.VideoWriter | None = None
    if compare_side_by_side:
        compare_writer = cv2.VideoWriter(
            str(compare_path),
            cv2.VideoWriter_fourcc(*"mp4v"),
            fps,
            (width * 2, height),
        )
        if not compare_writer.isOpened():
            writer.release()
            raise RuntimeError(f"Could not open writer: {compare_path}")

    per_frame_infer_ms: list[float] = []
    per_frame_total_ms: list[float] = []
    frame_idx = 0
    total_instances = 0
    class_hist: dict[str, int] = {}
    names = model.names

    while True:
        ok, frame_bgr = cap.read()
        if not ok:
            break

        t0 = time.perf_counter()
        infer_t0 = time.perf_counter()
        results = model.predict(
            source=frame_bgr,
            conf=conf,
            iou=iou,
            imgsz=imgsz,
            device=device,
            classes=classes_filter,
            verbose=False,
        )
        infer_ms = (time.perf_counter() - infer_t0) * 1000.0

        r = results[0]
        plotted = r.plot(labels=True, boxes=True, masks=True, probs=False)
        writer.write(plotted)

        if compare_writer is not None:
            side = np.hstack([frame_bgr, plotted])
            cv2.putText(
                side,
                "input",
                (12, 28),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.85,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )
            cv2.putText(
                side,
                "prediction",
                (width + 12, 28),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.85,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )
            compare_writer.write(side)

        masks_xy: list[np.ndarray] = []
        classes: list[int] = []
        if r.masks is not None and r.boxes is not None and len(r.boxes) > 0:
            polys = r.masks.xy  # list[np.ndarray Nx2]
            cls_ids = r.boxes.cls.tolist()
            for poly, cls_id in zip(polys, cls_ids):
                masks_xy.append(np.asarray(poly, dtype=np.float32))
                classes.append(int(cls_id))
                label = str(names.get(int(cls_id), int(cls_id)))
                class_hist[label] = class_hist.get(label, 0) + 1

        written = _write_frame_labels(
            labels_dir / f"frame_{frame_idx:06d}.txt",
            masks_xy,
            classes,
            width,
            height,
        )
        total_instances += written
        frame_idx += 1
        per_frame_infer_ms.append(infer_ms)
        per_frame_total_ms.append((time.perf_counter() - t0) * 1000.0)

    cap.release()
    writer.release()
    if compare_writer is not None:
        compare_writer.release()

    classes_payload = {int(k): str(v) for k, v in dict(names).items()}
    classes_json.write_text(json.dumps(classes_payload, indent=2), encoding="utf-8")

    report = {
        "video_path": str(video_path),
        "frames_processed": frame_idx,
        "source_total_frames": total_frames,
        "fps": fps,
        "width": width,
        "height": height,
        "model": str(model.ckpt_path) if hasattr(model, "ckpt_path") else None,
        "task": getattr(model, "task", "unknown"),
        "conf": conf,
        "iou": iou,
        "imgsz": imgsz,
        "device": device,
        "classes_filter": classes_filter,
        "instances_total": total_instances,
        "class_histogram": dict(sorted(class_hist.items(), key=lambda kv: kv[1], reverse=True)),
        "timing_ms": {
            "infer_per_frame": per_frame_infer_ms,
            "total_per_frame": per_frame_total_ms,
            "infer_avg": (sum(per_frame_infer_ms) / len(per_frame_infer_ms)) if per_frame_infer_ms else 0.0,
            "total_avg": (sum(per_frame_total_ms) / len(per_frame_total_ms)) if per_frame_total_ms else 0.0,
        },
        "paths": {
            "labels_dir": str(labels_dir),
            "preview_video": str(preview_path),
            "comparison_video": str(compare_path) if compare_side_by_side else None,
            "classes_json": str(classes_json),
            "report_json": str(report_json),
        },
    }
    report_json.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Run YOLO segmentation on each frame independently.")
    parser.add_argument("--model", default="yolo11x-seg.pt", help="Segmentation checkpoint (e.g. yolo11x-seg.pt)")
    parser.add_argument("--video", default="", help="Single input video path")
    parser.add_argument("--video-dir", default="", help="Process all videos recursively in a directory")
    parser.add_argument("--output-root", default="artifacts/video_yolo_seg_each_frame", help="Output root")
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--iou", type=float, default=0.7)
    parser.add_argument("--imgsz", type=int, default=1280)
    parser.add_argument("--device", default="auto", help="auto, cpu, mps, 0, ...")
    parser.add_argument("--classes", default="", help="Comma-separated class ids (empty = all)")
    parser.add_argument(
        "--compare",
        action="store_true",
        help="Also write comparison.mp4: original frame | prediction (side by side, same FPS/resolution height)",
    )
    args = parser.parse_args()

    if not args.video and not args.video_dir:
        raise ValueError("Provide either --video or --video-dir")

    classes_filter = None
    if args.classes.strip():
        classes_filter = [int(x.strip()) for x in args.classes.split(",") if x.strip()]

    model = YOLO(args.model)
    if getattr(model, "task", None) != "segment":
        print(f"[WARN] Model task is {getattr(model, 'task', None)!r}. Use a segmentation model for masks.")

    device = _resolve_device(args.device)

    if args.video:
        videos = [Path(args.video)]
    else:
        root = Path(args.video_dir)
        if not root.is_dir():
            raise FileNotFoundError(f"video-dir not found: {root}")
        videos = _find_videos(root)
        if not videos:
            raise RuntimeError(f"No videos found under: {root}")

    out_root = Path(args.output_root)
    out_root.mkdir(parents=True, exist_ok=True)
    reports: list[dict] = []
    failures: list[dict] = []

    for idx, video in enumerate(videos, start=1):
        if not video.exists():
            failures.append({"video_path": str(video), "error": "not found"})
            continue
        stem = video.stem
        if stem.endswith(".avi") and video.suffix.lower() == ".mp4":
            stem = Path(stem).stem
        out_dir = out_root / stem
        print(f"\n[{idx}/{len(videos)}] {video}")
        try:
            report = process_video(
                model=model,
                video_path=video,
                out_dir=out_dir,
                conf=args.conf,
                iou=args.iou,
                imgsz=args.imgsz,
                device=device,
                classes_filter=classes_filter,
                compare_side_by_side=args.compare,
            )
            reports.append(report)
            print(f"  -> preview: {report['paths']['preview_video']}")
            if args.compare and report["paths"].get("comparison_video"):
                print(f"  -> compare: {report['paths']['comparison_video']}")
            print(f"  -> report : {report['paths']['report_json']}")
        except Exception as e:
            failures.append({"video_path": str(video), "error_type": type(e).__name__, "error": str(e)})
            print(f"  [ERROR] {type(e).__name__}: {e}")

    summary = {
        "videos_processed": len(reports),
        "videos_failed": len(failures),
        "model": args.model,
        "device": device,
        "conf": args.conf,
        "iou": args.iou,
        "imgsz": args.imgsz,
        "classes_filter": classes_filter,
        "reports": [r["paths"]["report_json"] for r in reports],
        "failures": failures,
    }
    summary_path = out_root / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"\nDone. Summary: {summary_path}")


if __name__ == "__main__":
    main()

