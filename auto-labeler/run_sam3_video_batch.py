#!/usr/bin/env python3
"""
Run the Qwen + SAM3 video pipeline on one or many files with N parallel workers.

Model loading and sharing
-------------------------
Each worker is a *separate OS process*. PyTorch + HF models are not safely shared for
parallel inference across processes in this script, so every worker calls load_models()
once (about N× peak RAM, N times you see "Loading Qwen/SAM3").

To load Qwen+SAM3 only once, use --workers 1: one process walks all videos in order.

To scale throughput on many clips, use --workers N>1: you trade duplicate loads and RAM
for true parallelism. A single shared in-memory model for N parallel forwards would
require a different design (e.g. one long-lived server process and clients over a queue).

Example (4 workers, folder of MP4s):
  python run_sam3_video_batch.py --input /path/to/videos --output-root /path/out --workers 4

Single load, all videos in one process:
  python run_sam3_video_batch.py --input /path/to/videos --output-root /path/out --workers 1
"""
from __future__ import annotations

import argparse
import multiprocessing as mp
import os
import sys
import threading
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any

_AUTO = Path(__file__).resolve().parent
if str(_AUTO) not in sys.path:
    sys.path.insert(0, str(_AUTO))


def _import_psutil():
    try:
        import psutil
    except ImportError:
        return None
    return psutil


def _b_to_gib(n: int | float) -> float:
    return float(n) / (1024.0**3)


def _monitor_children_loop(parent_pid: int, stop: threading.Event, interval: float) -> None:
    """Print RSS + CPU% for all descendant processes of parent (pool workers)."""
    psutil = _import_psutil()
    if psutil is None or interval <= 0:
        return
    try:
        parent = psutil.Process(parent_pid)
    except psutil.Error:
        return
    for ch in parent.children(recursive=True):
        try:
            ch.cpu_percent(interval=0.05)
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            pass
    while not stop.is_set():
        if stop.wait(interval):
            break
        lines: list[str] = []
        try:
            kids = parent.children(recursive=True)
        except psutil.Error:
            continue
        for c in sorted(kids, key=lambda p: p.pid):
            try:
                with c.oneshot():
                    rss = c.memory_info().rss
                    cpu = c.cpu_percent()
                lines.append(f"pid={c.pid} rss={_b_to_gib(rss):.2f}GiB cpu%={cpu:5.1f}")
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                continue
        if lines:
            print("  [resource monitor]  " + "  |  ".join(lines), flush=True)


def _config_keys() -> list[str]:
    return [
        "VIDEO_EXTENSIONS",
        "VIDEO_IGNORE_NAME_SUBSTRINGS",
        "OUTPUT_ROOT",
        "TOP_N",
        "MAX_FRAMES",
        "VIDEO_FPS_OVERRIDE",
        "CPU_TORCH_THREADS",
        "CPU_INTEROP_THREADS",
        "QWEN_MODEL_ID",
        "SAM3_MODEL_ID",
        "BBOX_LABEL_FORMAT",
        "YOLO_CLASS_NAMES",
        "YOLO_CLASS_ALIASES",
        "YOLO_SKIP_UNMAPPED_PROMPTS",
        "YOLO_LABEL_APPEND_CONFIDENCE",
        "YOLO_SEG_POLY_EPS_RATIO",
        "YOLO_SEG_MIN_MASK_AREA",
        "VIDEO_EXPORT_FRAME_STRIDE",
        "VIDEO_EXPORT_PREVIEW_MP4S",
        "VIDEO_EXPORT_OVERLAY_IMAGES",
        "VIDEO_PIPELINE_MODE",
        "EXTRACT_FRAMES_TO_DISK_FIRST",
        "PER_FRAME_PROGRESS_EVERY",
        "PER_FRAME_TIMING_MAX_ROWS",
        "QWEN_PER_FRAME_TOP_N",
        "QWEN_AGGREGATE_TOP_K",
    ]


def build_config_from_defaults(
    *,
    input_path: str,
    output_root: Path,
    top_n: int | None = None,
    max_frames: int | None = None,
    video_fps_override: float | None = None,
    video_pipeline_mode: str | None = None,
    qwen_model_id: str | None = None,
    sam3_model_id: str | None = None,
    cpu_torch_threads: int | None = None,
    cpu_interop_threads: int | None = None,
) -> dict:
    import sam3_video_nb_defaults as nb

    cfg = {k: getattr(nb, k) for k in _config_keys() if hasattr(nb, k)}
    cfg["INPUT_PATH"] = input_path
    cfg["OUTPUT_ROOT"] = output_root
    if top_n is not None:
        cfg["TOP_N"] = top_n
    if max_frames is not None:
        cfg["MAX_FRAMES"] = max_frames
    if video_fps_override is not None:
        cfg["VIDEO_FPS_OVERRIDE"] = video_fps_override
    if video_pipeline_mode is not None:
        cfg["VIDEO_PIPELINE_MODE"] = video_pipeline_mode
    if qwen_model_id is not None:
        cfg["QWEN_MODEL_ID"] = qwen_model_id
    if sam3_model_id is not None:
        cfg["SAM3_MODEL_ID"] = sam3_model_id
    if cpu_torch_threads is not None:
        cfg["CPU_TORCH_THREADS"] = cpu_torch_threads
    if cpu_interop_threads is not None:
        cfg["CPU_INTEROP_THREADS"] = cpu_interop_threads
    return cfg


def _chunk_paths(seq: list[str], n_workers: int) -> list[list[str]]:
    if not seq:
        return []
    n_workers = min(max(1, n_workers), len(seq))
    base, rem = divmod(len(seq), n_workers)
    out, i = [], 0
    for w in range(n_workers):
        take = base + (1 if w < rem else 0)
        out.append(seq[i : i + take])
        i += take
    return out


def _worker_entry(
    payload: tuple[int, list[str], dict],
) -> tuple[int, list[dict], dict[str, Any]]:
    wid, paths, cfg = payload
    import sam3_video_pipeline as p

    psutil = _import_psutil()
    mem: dict[str, Any] = {"pid": os.getpid(), "worker_id": int(wid)}
    proc = psutil.Process(os.getpid()) if psutil is not None else None

    def _rss() -> int | None:
        if proc is None:
            return None
        return int(proc.memory_info().rss)

    p.apply_config(cfg)
    m = _rss()
    if m is not None:
        mem["rss_gib_after_config"] = round(_b_to_gib(m), 3)

    p.load_models()
    m = _rss()
    if m is not None:
        mem["rss_gib_after_load_models"] = round(_b_to_gib(m), 3)
    peak = m or 0

    out_root = Path(cfg["OUTPUT_ROOT"])
    logs: list[dict] = []
    for v in paths:
        logs.append(
            p.process_video(
                Path(v),
                out_root,
                cfg["TOP_N"],
                cfg.get("MAX_FRAMES"),
                cfg.get("VIDEO_FPS_OVERRIDE"),
                cfg["BBOX_LABEL_FORMAT"],
                None,
            )
        )
        r = _rss()
        if r is not None:
            peak = max(peak, r)
    if proc is not None and peak:
        mem["rss_gib_peak"] = round(_b_to_gib(peak), 3)

    return int(wid), logs, mem


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument(
        "--input",
        required=True,
        help="One video file or a directory of videos (see VIDEO_EXTENSIONS in defaults).",
    )
    ap.add_argument(
        "--output-root",
        required=True,
        type=Path,
        help="Base output directory (same as notebook OUTPUT_ROOT).",
    )
    ap.add_argument(
        "--workers",
        type=int,
        default=1,
        help=(
            "Number of processes. 1 = load Qwen+SAM3 once, process every video sequentially. "
            "N>1 = N independent loads (N× RAM) and split videos across workers. "
            "You cannot share one loaded model across these processes."
        ),
    )
    ap.add_argument("--top-n", type=int, default=None)
    ap.add_argument("--max-frames", type=int, default=None)
    ap.add_argument("--fps-override", type=float, default=None)
    ap.add_argument(
        "--mode",
        default=None,
        choices=["video_session", "per_frame_image", "qwen_per_frame_then_agg_sam3"],
        help="Override notebook default VIDEO_PIPELINE_MODE if set.",
    )
    ap.add_argument("--qwen-model", default=None, dest="qwen_model_id")
    ap.add_argument("--sam3-model", default=None, dest="sam3_model_id")
    ap.add_argument("--cpu-torch-threads", type=int, default=None)
    ap.add_argument("--cpu-interop-threads", type=int, default=None)
    ap.add_argument(
        "--monitor-interval",
        type=float,
        default=0.0,
        metavar="SEC",
        help=(
            "If >0 and psutil is installed, print each child process RSS (GiB) and CPU%% every SEC "
            "while the pool runs. 0 = no live lines (per-worker summary still printed at the end if "
            "psutil is available). Example: 3.0"
        ),
    )
    args = ap.parse_args()

    cfg = build_config_from_defaults(
        input_path=args.input,
        output_root=args.output_root.resolve(),
        top_n=args.top_n,
        max_frames=args.max_frames,
        video_fps_override=args.fps_override,
        video_pipeline_mode=args.mode,
        qwen_model_id=args.qwen_model_id,
        sam3_model_id=args.sam3_model_id,
        cpu_torch_threads=args.cpu_torch_threads,
        cpu_interop_threads=args.cpu_interop_threads,
    )
    args.output_root.mkdir(parents=True, exist_ok=True)

    from sam3_video_pipeline import collect_videos, load_models, process_video, apply_config

    _ign = cfg.get("VIDEO_IGNORE_NAME_SUBSTRINGS")
    vids = [
        str(p)
        for p in collect_videos(
            cfg["INPUT_PATH"],
            cfg["VIDEO_EXTENSIONS"],
            tuple(_ign) if _ign is not None else None,
        )
    ]
    print(
        f"Found {len(vids)} video(s)  |  workers={args.workers}  |  mode={cfg.get('VIDEO_PIPELINE_MODE')!r}\n"
        f"Qwen={cfg['QWEN_MODEL_ID']}  SAM3={cfg['SAM3_MODEL_ID']}"
    )

    wall0 = time.perf_counter()
    all_logs: list[dict] = []

    live_mon = bool(_import_psutil()) and args.monitor_interval and args.monitor_interval > 0
    if not _import_psutil() and args.monitor_interval and args.monitor_interval > 0:
        print("  (install psutil for --monitor-interval: pip install psutil)", flush=True)
    one_shot_rss: dict[str, Any] = {}

    if args.workers <= 1 or len(vids) <= 1:
        apply_config(cfg)
        load_models()
        _ps = _import_psutil()
        pr = _ps.Process(os.getpid()) if _ps is not None else None
        if pr is not None:
            one_shot_rss = {
                "pid": os.getpid(),
                "mode": "single_process",
                "rss_gib_after_load_models": round(_b_to_gib(pr.memory_info().rss), 3),
            }
        peak_b = int(pr.memory_info().rss) if pr is not None else 0
        for v in vids:
            all_logs.append(
                process_video(
                    Path(v),
                    Path(cfg["OUTPUT_ROOT"]),
                    cfg["TOP_N"],
                    cfg.get("MAX_FRAMES"),
                    cfg.get("VIDEO_FPS_OVERRIDE"),
                    cfg["BBOX_LABEL_FORMAT"],
                    None,
                )
            )
            if pr is not None:
                peak_b = max(peak_b, int(pr.memory_info().rss))
        if pr is not None:
            one_shot_rss["rss_gib_peak"] = round(_b_to_gib(peak_b), 3)
    else:
        n_proc = min(args.workers, len(vids))
        print(
            f"Note: {n_proc} process(es) will each load Qwen+SAM3 (not shared). "
            f"Use --workers 1 for a single load and sequential video processing."
        )
        if live_mon:
            print(
                f"  Resource monitor: every {args.monitor_interval:.1f}s (install psutil if missing).",
                flush=True,
            )
        chunks = _chunk_paths(vids, args.workers)
        jobs: list[tuple[int, list[str], dict]] = [
            (i, ch, cfg) for i, ch in enumerate(chunks) if ch
        ]
        ctx = mp.get_context("spawn")
        stop_mon = threading.Event()
        mon_thr: threading.Thread | None = None
        if live_mon:
            mon_thr = threading.Thread(
                target=_monitor_children_loop,
                args=(os.getpid(), stop_mon, float(args.monitor_interval)),
                name="resource-monitor",
                daemon=True,
            )
            mon_thr.start()
        worker_stats: list[dict[str, Any]] = []
        try:
            with ProcessPoolExecutor(max_workers=len(jobs), mp_context=ctx) as ex:
                futs = {ex.submit(_worker_entry, job): job[0] for job in jobs}
                finished: list[tuple[int, list[dict], dict[str, Any]]] = []
                for fut in as_completed(futs):
                    finished.append(fut.result())
                for _wid, logs, wmem in sorted(finished, key=lambda x: x[0]):
                    all_logs.extend(logs)
                    worker_stats.append(wmem)
        finally:
            if mon_thr is not None:
                stop_mon.set()
                mon_thr.join(timeout=5.0)

        if worker_stats:
            _has_rss = any(
                "rss_gib_after_load_models" in s or "rss_gib_peak" in s for s in worker_stats
            )
            cap = "Per-worker stats" + (" (RSS via psutil)" if _has_rss else " (install psutil for RSS)")
            print(f"\n  {cap}:")
            for s in sorted(worker_stats, key=lambda x: int(x.get("worker_id", 0))):
                if _has_rss:
                    line = (
                        f"    worker {s.get('worker_id')}  pid {s.get('pid')}  "
                        f"after_models={s.get('rss_gib_after_load_models', '?')} GiB  "
                        f"peak={s.get('rss_gib_peak', '?')} GiB"
                    )
                else:
                    line = f"    worker {s.get('worker_id')}  pid {s.get('pid')}"
                print(line, flush=True)

    wall = time.perf_counter() - wall0
    done = [l for l in all_logs if l.get("status") == "done"]
    err = [l for l in all_logs if l.get("status") != "done"]
    if one_shot_rss:
        print(
            f"\n  Process memory (psutil)  pid={one_shot_rss.get('pid')}  "
            f"after_models={one_shot_rss.get('rss_gib_after_load_models')} GiB  "
            f"peak≈{one_shot_rss.get('rss_gib_peak')} GiB  (batch RSS peak while running). "
            f"Per-video CPU/RAM/torch threads: see each run's processing_log.json → resources_after.",
            flush=True,
        )
    print(
        f"\n{'='*60}\n"
        f"Batch wall time: {wall:.1f}s  |  done={len(done)}  |  other={len(err)}  |  total_logs={len(all_logs)}"
    )
    return 0 if not err else 1


if __name__ == "__main__":
    raise SystemExit(main())
