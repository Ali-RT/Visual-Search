#!/usr/bin/env python3
"""
Generate aligned_video_uss.csv:
  - Per-video-frame timeline (frame_idx, frame_time_s)
  - Sample selected MF4 signals at each frame time
  - Compute per-sensor min echo distance for CEP_10/11/12 from all *EchoDist channels
  - Compute rear_min_mm = min(CEP_10, CEP_11, CEP_12)
"""

from __future__ import annotations

import argparse
from pathlib import Path
import re
from typing import List, Optional, Tuple

import cv2
import numpy as np
import pandas as pd
from asammdf import MDF


def get_video_meta(video_path: str) -> Tuple[int, float, int, int]:
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = float(cap.get(cv2.CAP_PROP_FPS))
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()
    if fps <= 0:
        raise RuntimeError("Video FPS is invalid/zero.")
    return n_frames, fps, w, h


def make_timebase_seconds(sig) -> Tuple[np.ndarray, np.ndarray, float]:
    t = np.asarray(sig.timestamps, dtype=np.float64)
    y = np.asarray(sig.samples)
    if t.size == 0:
        return np.array([], dtype=np.float64), np.array([]), 0.0
    t0 = float(t[0])
    return (t - t0), y, t0


def sample_nearest(t_sig: np.ndarray, y_sig: np.ndarray, t_query: np.ndarray) -> np.ndarray:
    if t_sig.size == 0:
        return np.full_like(t_query, np.nan, dtype=np.float64)

    idx = np.searchsorted(t_sig, t_query, side="left")
    idx = np.clip(idx, 0, len(t_sig) - 1)
    idx0 = np.clip(idx - 1, 0, len(t_sig) - 1)
    choose_prev = np.abs(t_query - t_sig[idx0]) <= np.abs(t_sig[idx] - t_query)

    if y_sig.dtype.kind in ("i", "u", "f"):
        out = y_sig[idx].astype(np.float64)
        out[choose_prev] = y_sig[idx0][choose_prev].astype(np.float64)
        return out

    out = y_sig[idx].astype(object)
    out[choose_prev] = y_sig[idx0][choose_prev].astype(object)
    return out


def sample_ffill(t_sig: np.ndarray, y_sig: np.ndarray, t_query: np.ndarray) -> np.ndarray:
    if t_sig.size == 0:
        return np.full_like(t_query, np.nan, dtype=np.float64)
    idx = np.searchsorted(t_sig, t_query, side="right") - 1
    idx = np.clip(idx, 0, len(t_sig) - 1)
    if y_sig.dtype.kind in ("i", "u", "f"):
        return y_sig[idx].astype(np.float64)
    return y_sig[idx].astype(object)


def decode_bytes_array(arr: np.ndarray) -> np.ndarray:
    if arr.dtype == object:
        out = []
        for v in arr:
            if isinstance(v, (bytes, bytearray)):
                out.append(v.decode(errors="ignore"))
            else:
                out.append(v)
        return np.array(out, dtype=object)
    if arr.dtype.kind == "S":
        return np.char.decode(arr, errors="ignore")
    return arr


def list_channel_names(mdf: MDF) -> List[str]:
    return list(mdf.channels_db.keys())


def find_first_existing(mdf: MDF, candidates: List[str]) -> Optional[str]:
    names = set(list_channel_names(mdf))
    for c in candidates:
        if c in names:
            return c
    return None


def find_regex(names: List[str], pattern: str) -> List[str]:
    rx = re.compile(pattern)
    return [n for n in names if rx.search(n)]


def compute_min_echo_for_cep(
    mdf: MDF,
    names: List[str],
    cep_id: str,
    t_query: np.ndarray,
    invalid_values: Tuple[float, ...] = (0.0, 65535.0),
) -> np.ndarray:
    pattern = rf"^CEP_{cep_id}_.+EchoDist$"
    echo_chs = [n for n in names if re.match(pattern, n)]
    if not echo_chs:
        echo_chs = find_regex(names, rf"\bCEP_{cep_id}\b.*EchoDist")
    if not echo_chs:
        return np.full_like(t_query, np.nan, dtype=np.float64)

    mats = []
    for ch in echo_chs:
        sig = mdf.get(ch)
        t_sig, y_sig, _ = make_timebase_seconds(sig)
        v = sample_nearest(t_sig, y_sig, t_query)
        if not (isinstance(v, np.ndarray) and (v.dtype.kind in ("i", "u", "f", "O"))):
            continue
        v = np.asarray(pd.to_numeric(v, errors="coerce"), dtype=np.float64)
        for bad in invalid_values:
            v[v == bad] = np.nan
        mats.append(v)

    if not mats:
        return np.full_like(t_query, np.nan, dtype=np.float64)

    M = np.vstack(mats)
    with np.errstate(all="ignore"):
        out = np.nanmin(M, axis=0)
    return out


def compute_rear_min(df: pd.DataFrame) -> np.ndarray:
    arr = df[["CEP_10_min_echo_mm", "CEP_11_min_echo_mm", "CEP_12_min_echo_mm"]].to_numpy(dtype=np.float64)
    out = np.full(arr.shape[0], np.nan, dtype=np.float64)
    valid = np.isfinite(arr).any(axis=1)
    if np.any(valid):
        out[valid] = np.nanmin(arr[valid], axis=1)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", required=True)
    ap.add_argument("--mf4", required=True)
    ap.add_argument("--out", default="aligned_video_uss.csv")
    args = ap.parse_args()

    n_frames, fps, w, h = get_video_meta(args.video)
    frame_idx = np.arange(n_frames, dtype=int)
    frame_time_s = frame_idx / fps

    out = pd.DataFrame({"frame_idx": frame_idx, "frame_time_s": frame_time_s})
    out["signal_time_s"] = out["frame_time_s"]

    mdf = MDF(args.mf4)
    names = list_channel_names(mdf)

    signal_specs = {
        "VSM_VehSpeed": {
            "kind": "numeric_nearest",
            "candidates": ["VSM_VehSpeed", "VSM.VehSpeed", "VehSpeed", "VehicleSpeed"],
        },
        "VSM_Gear": {
            "kind": "discrete_ffill",
            "candidates": ["VSM_Gear", "VSM.Gear", "Gear", "GearState"],
        },
        "VHM_CanSig_ReverseGearInserted": {
            "kind": "discrete_ffill",
            "candidates": ["VHM_CanSig_ReverseGearInserted", "ReverseGearInserted", "ReverseInserted"],
        },
        "rbp_loc_Input_st.DetectRangeRearMM_u16": {
            "kind": "numeric_nearest",
            "candidates": ["rbp_loc_Input_st.DetectRangeRearMM_u16"],
        },
        "rbp_loc_Input_st.DetectRangeFrontMM_u16": {
            "kind": "numeric_nearest",
            "candidates": ["rbp_loc_Input_st.DetectRangeFrontMM_u16"],
        },
        "rbp_loc_Output_st.StopDistance_u16": {
            "kind": "numeric_nearest",
            "candidates": ["rbp_loc_Output_st.StopDistance_u16"],
        },
    }

    for out_col, spec in signal_specs.items():
        ch = find_first_existing(mdf, spec["candidates"])
        if ch is None:
            continue
        sig = mdf.get(ch)
        t_sig, y_sig, _ = make_timebase_seconds(sig)
        if spec["kind"] == "numeric_nearest":
            v = sample_nearest(t_sig, y_sig, frame_time_s)
            out[out_col] = np.asarray(pd.to_numeric(v, errors="coerce"), dtype=np.float64)
        else:
            v = sample_ffill(t_sig, y_sig, frame_time_s)
            out[out_col] = decode_bytes_array(v)

    out["CEP_10_min_echo_mm"] = compute_min_echo_for_cep(mdf, names, "10", frame_time_s)
    out["CEP_11_min_echo_mm"] = compute_min_echo_for_cep(mdf, names, "11", frame_time_s)
    out["CEP_12_min_echo_mm"] = compute_min_echo_for_cep(mdf, names, "12", frame_time_s)
    out["rear_min_mm"] = compute_rear_min(out)

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.out, index=False)
    print(f"Wrote: {args.out}")
    print(f"Video: {n_frames} frames @ {fps:.3f} FPS ({w}x{h})")


if __name__ == "__main__":
    main()
