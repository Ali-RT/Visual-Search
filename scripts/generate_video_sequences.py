"""
Build multiple short video sequences from image folders.

This is useful when you have frame dumps (e.g., WoodScape-style names) and want
ready-to-use MP4 clips for demo/testing.

Examples:
    uv run python scripts/generate_video_sequences.py \
      --input_dir data/woodscape_samples/extracted/images \
      --output_dir data/use_cases/generated_sequences \
      --camera_filter FV \
      --max_videos 5 \
      --min_frames 30 \
      --max_frames_per_video 180
"""
from __future__ import annotations

import argparse
import re
import subprocess
from dataclasses import dataclass
from pathlib import Path

import cv2


FRAME_PATTERNS = [
    re.compile(r"^(?P<idx>\d+)_(?P<cam>[A-Za-z0-9]+)$", re.IGNORECASE),
    re.compile(r"^(?P<prefix>.+?)\.(?:dat_)?frame_(?P<idx>\d+)$", re.IGNORECASE),
    re.compile(r"^(?P<prefix>.+?)_frame_(?P<idx>\d+)$", re.IGNORECASE),
]


@dataclass
class FrameItem:
    path: Path
    seq_key: str
    frame_idx: int


@dataclass
class SequenceRun:
    key: str
    frames: list[FrameItem]


def parse_frame_key(path: Path) -> tuple[str, int]:
    stem = path.stem
    for pat in FRAME_PATTERNS:
        m = pat.match(stem)
        if m:
            # Handles names like 00002_FV.png from WoodScape rgb_images.
            if "cam" in m.groupdict():
                return m.group("cam"), int(m.group("idx"))
            return m.group("prefix"), int(m.group("idx"))

    # Fallback: extract trailing digits from stem.
    m2 = re.match(r"^(?P<prefix>.*?)(?P<idx>\d+)$", stem)
    if m2:
        return m2.group("prefix").rstrip("_-."), int(m2.group("idx"))

    return stem, 0


def collect_frames(input_dir: Path, recursive: bool, camera_filter: str | None) -> list[FrameItem]:
    patterns = ("*.jpg", "*.jpeg", "*.png", "*.bmp", "*.webp")
    paths: list[Path] = []
    for patt in patterns:
        if recursive:
            paths.extend(input_dir.rglob(patt))
        else:
            paths.extend(input_dir.glob(patt))

    items: list[FrameItem] = []
    for p in sorted(paths):
        key, idx = parse_frame_key(p)
        if camera_filter and not key.endswith(camera_filter):
            continue
        items.append(FrameItem(path=p, seq_key=key, frame_idx=idx))
    return items


def build_runs(items: list[FrameItem], min_frames: int, max_gap: int) -> list[SequenceRun]:
    groups: dict[str, list[FrameItem]] = {}
    for it in items:
        groups.setdefault(it.seq_key, []).append(it)

    runs: list[SequenceRun] = []
    for key, frames in groups.items():
        ordered = sorted(frames, key=lambda x: x.frame_idx)
        if not ordered:
            continue

        chunk: list[FrameItem] = [ordered[0]]
        for i in range(1, len(ordered)):
            prev = ordered[i - 1]
            cur = ordered[i]
            if cur.frame_idx - prev.frame_idx <= max_gap:
                chunk.append(cur)
            else:
                if len(chunk) >= min_frames:
                    runs.append(SequenceRun(key=key, frames=chunk))
                chunk = [cur]

        if len(chunk) >= min_frames:
            runs.append(SequenceRun(key=key, frames=chunk))

    runs.sort(key=lambda r: len(r.frames), reverse=True)
    return runs


def best_run_length(items: list[FrameItem], max_gap: int) -> int:
    groups: dict[str, list[int]] = {}
    for it in items:
        groups.setdefault(it.seq_key, []).append(it.frame_idx)

    best = 0
    for _, idxs in groups.items():
        ordered = sorted(idxs)
        if not ordered:
            continue
        cur = 1
        local_best = 1
        for i in range(1, len(ordered)):
            if ordered[i] - ordered[i - 1] <= max_gap:
                cur += 1
            else:
                local_best = max(local_best, cur)
                cur = 1
        local_best = max(local_best, cur)
        best = max(best, local_best)
    return best


def safe_name(value: str) -> str:
    return re.sub(r"[^a-zA-Z0-9._-]+", "_", value)


def write_video(frames: list[Path], out_path: Path, fps: float, resize: tuple[int, int] | None) -> tuple[int, int, int]:
    if not frames:
        raise RuntimeError("No frames to write")

    first = cv2.imread(str(frames[0]))
    if first is None:
        raise RuntimeError(f"Failed to read first frame: {frames[0]}")

    if resize is not None:
        width, height = resize
    else:
        height, width = first.shape[:2]

    raw_out = out_path.with_suffix(".raw.mp4")
    writer = cv2.VideoWriter(str(raw_out), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
    if not writer.isOpened():
        raise RuntimeError(f"Failed to open video writer for {out_path}")

    written = 0
    try:
        for frame_path in frames:
            img = cv2.imread(str(frame_path))
            if img is None:
                continue
            if img.shape[1] != width or img.shape[0] != height:
                img = cv2.resize(img, (width, height), interpolation=cv2.INTER_AREA)
            writer.write(img)
            written += 1
    finally:
        writer.release()

    ffmpeg_cmd = [
        "ffmpeg", "-y",
        "-i", str(raw_out),
        "-c:v", "libx264",
        "-pix_fmt", "yuv420p",
        "-movflags", "+faststart",
        str(out_path),
    ]
    proc = subprocess.run(ffmpeg_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, check=False)
    raw_out.unlink(missing_ok=True)
    if proc.returncode != 0:
        raise RuntimeError(f"ffmpeg failed for {out_path}")

    return written, width, height


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate short sequence videos from image frames")
    parser.add_argument("--input_dir", required=True, help="Image directory")
    parser.add_argument("--output_dir", default="data/use_cases/generated_sequences", help="Output video directory")
    parser.add_argument("--fps", type=float, default=12.0, help="Output fps")
    parser.add_argument("--max_videos", type=int, default=5, help="How many sequences to export")
    parser.add_argument("--max_frames_per_video", type=int, default=180, help="Cap frames per output video")
    parser.add_argument("--min_frames", type=int, default=30, help="Minimum frames required for a sequence")
    parser.add_argument("--max_gap", type=int, default=5, help="Max frame index gap treated as same sequence")
    parser.add_argument("--camera_filter", default=None, help="Only keep sequence keys ending with this suffix, e.g. FV")
    parser.add_argument("--recursive", action="store_true", help="Recursively scan input directory")
    parser.add_argument("--resize_width", type=int, default=0, help="Optional fixed output width")
    parser.add_argument("--resize_height", type=int, default=0, help="Optional fixed output height")
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    if not input_dir.exists():
        raise RuntimeError(f"Input directory not found: {input_dir}")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    resize = None
    if args.resize_width > 0 and args.resize_height > 0:
        resize = (args.resize_width, args.resize_height)

    items = collect_frames(
        input_dir=input_dir,
        recursive=args.recursive,
        camera_filter=args.camera_filter,
    )
    if not items:
        raise RuntimeError("No matching image frames found")

    runs = build_runs(items=items, min_frames=args.min_frames, max_gap=args.max_gap)
    if not runs:
        best_len = best_run_length(items, max_gap=args.max_gap)
        raise RuntimeError(
            "No frame sequences matched min_frames/max_gap filters. "
            f"Best available run length is {best_len} with max_gap={args.max_gap}. "
            f"Try lowering --min_frames to <= {best_len} or increasing --max_gap.",
        )

    exported = 0
    for run in runs[: args.max_videos]:
        frames = [it.path for it in run.frames[: args.max_frames_per_video]]
        start_idx = run.frames[0].frame_idx
        end_idx = run.frames[min(len(frames), len(run.frames)) - 1].frame_idx
        out_name = f"{safe_name(run.key)}_{start_idx:06d}_{end_idx:06d}.mp4"
        out_path = output_dir / out_name

        written, width, height = write_video(
            frames=frames,
            out_path=out_path,
            fps=args.fps,
            resize=resize,
        )
        exported += 1
        print(
            f"[{exported}] {out_path} | key={run.key} | frames={written} | "
            f"range={start_idx}-{end_idx} | size={width}x{height} | fps={args.fps:.2f}",
        )

    print(f"Done. Exported {exported} video(s) to {output_dir}")


if __name__ == "__main__":
    main()
