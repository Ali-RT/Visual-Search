#!/usr/bin/env python3
"""Write mine_qwen_taxonomy.ipynb for scanning qwen_output.json under artifact roots."""
import json
from pathlib import Path

OUT = Path(__file__).resolve().parent / "mine_qwen_taxonomy.ipynb"

cells = []

def md(s):
    cells.append({"cell_type": "markdown", "metadata": {}, "source": s.splitlines(keepends=True)})

def code(s):
    cells.append(
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": s.splitlines(keepends=True),
        }
    )

md(
    """# Mine Qwen prompts for YOLO taxonomy (Option A — closed set)

Scans **`qwen_output.json`** under image-pipeline outputs (same **flat / nested** layout as the train notebook).

**Use this to:**
- See **all unique prompts** Qwen produced (with counts).
- Compare them to your current **`YOLO_CLASS_NAMES`** / aliases (coverage report).
- Decide what to **add** to `YOLO_CLASS_NAMES` and what to map via **`YOLO_CLASS_ALIASES`**.

**After you change taxonomy:** re-run `qwen_sam3_image_pipeline.ipynb` (and video if used) so `labels/segments/*.txt` and `classes.txt` match the new `nc`, then retrain.

**Run order:** config → function definitions → **one analysis cell** (scan + tables + suggested `YOLO_CLASS_NAMES`) → optional report cell.
"""
)

code(
    """from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

# Root containing per-run folders (same as train_yolo_seg_from_image_pipeline)
ARTIFACTS_ROOT = Path("/Users/ral3ply/Documents/GitHub/visual-search/auto-labeler/outputs/uss_pf")
# ARTIFACTS_ROOT = Path("/Users/ral3ply/Documents/GitHub/visual-search/auto-labeler/outputs/images")

# Paste your *current* Step-3 lists here for coverage analysis (or leave empty to skip)
CURRENT_YOLO_CLASS_NAMES: list[str] = [
    "road",
    "building",
    "car",
    "person",
    "tree",
    "sky",
]
CURRENT_YOLO_CLASS_ALIASES: dict[str, str] = {
    "automobile": "car",
    "suv": "car",
    "pedestrian": "person",
}

REPORT_PATH = ARTIFACTS_ROOT.parent / "mine_taxonomy_report.json"
"""
)

code(
    """def iter_segment_runs(artifacts_root: Path):
    if not artifacts_root.is_dir():
        raise FileNotFoundError(artifacts_root)
    for sub in sorted(artifacts_root.iterdir()):
        if not sub.is_dir() or sub.name.startswith("."):
            continue
        if (sub / "qwen_output.json").is_file():
            yield sub.name, sub
            continue
        for nest in sorted(sub.iterdir()):
            if not nest.is_dir() or nest.name.startswith("."):
                continue
            if (nest / "qwen_output.json").is_file():
                yield f"{sub.name}__{nest.name}", nest


def extract_prompts_from_qwen(obj: dict) -> list[str]:
    out: list[str] = []
    if not isinstance(obj, dict):
        return out
    for p in obj.get("prompts") or []:
        if isinstance(p, str) and p.strip():
            out.append(p.strip())
    raw = obj.get("qwen_raw")
    if isinstance(raw, dict):
        for dom in raw.get("dominant_objects") or []:
            if isinstance(dom, dict):
                n = dom.get("name")
                if isinstance(n, str) and n.strip():
                    out.append(n.strip())
    return out


def resolve_canonical(prompt: str, names: list[str], aliases: dict[str, str]) -> str | None:
    \"\"\"Same idea as pipeline: return canonical class name or None.\"\"\"
    raw = prompt.strip()
    lk = raw.lower()
    mapped = None
    for ak, av in aliases.items():
        if ak.lower() == lk:
            mapped = av
            break
    candidate = (mapped if mapped is not None else raw).strip()
    cl = candidate.lower()
    for name in names:
        if name.lower() == cl:
            return name
    return None
"""
)

code(
    """# Scan artifacts + print frequency / coverage / suggested YOLO_CLASS_NAMES (single cell)
prompt_counter: Counter[str] = Counter()
runs_with_file = 0
runs_missing = 0
per_run: list[dict] = []

for key, run_dir in iter_segment_runs(ARTIFACTS_ROOT):
    qpath = run_dir / "qwen_output.json"
    if not qpath.is_file():
        runs_missing += 1
        continue
    runs_with_file += 1
    try:
        data = json.loads(qpath.read_text())
    except Exception as e:
        per_run.append({"key": key, "error": str(e)})
        continue
    prompts = extract_prompts_from_qwen(data)
    for p in prompts:
        prompt_counter[p] += 1
    per_run.append({"key": key, "prompts": prompts})

print(f"Runs with qwen_output.json: {runs_with_file}  (dirs without file: {runs_missing})")
print(f"Total prompt occurrences: {sum(prompt_counter.values())}")
print(f"Unique prompt strings: {len(prompt_counter)}")

# Frequency table (copy into a spreadsheet if needed)
print("\\n=== Unique prompts (count desc) ===\\n")
for p, c in prompt_counter.most_common():
    print(f"{c:5d}  {p!r}")

if CURRENT_YOLO_CLASS_NAMES:
    print("\\n=== Coverage vs CURRENT_YOLO_CLASS_NAMES + aliases ===\\n")
    unmapped = []
    for p, c in prompt_counter.most_common():
        can = resolve_canonical(p, CURRENT_YOLO_CLASS_NAMES, CURRENT_YOLO_CLASS_ALIASES)
        if can is None:
            unmapped.append((p, c))
            print(f"UNMAPPED {c:5d}  {p!r}")
    if not unmapped:
        print("All observed prompts resolve to a current class (exact / alias / case-insensitive).")
    else:
        print(f"\\n→ Add {len(unmapped)} distinct strings to names or aliases, then re-export labels.")

# Suggested starter list: unique prompts sorted by frequency (edit before pasting into Step 3)
unique_by_freq = [p for p, _ in prompt_counter.most_common()]
print("\\n# --- Copy/edit into qwen_sam3_image_pipeline Step 3 ---")
print("YOLO_CLASS_NAMES = [")
for p in unique_by_freq:
    print(f'    "{p}",')
print("]")
print("# Then merge near-duplicates (e.g. trees vs tree) using YOLO_CLASS_ALIASES and trim the list.")
"""
)

code(
    """if "prompt_counter" not in globals():
    raise RuntimeError(
        "Run the analysis cell above first (it defines prompt_counter, runs_with_file, per_run)."
    )

report = {
    "artifacts_root": str(ARTIFACTS_ROOT.resolve()),
    "runs_with_qwen_json": runs_with_file,
    "runs_missing_qwen_json": runs_missing,
    "total_prompt_occurrences": int(sum(prompt_counter.values())),
    "unique_prompts": len(prompt_counter),
    "prompt_counts": dict(prompt_counter.most_common()),
    "per_run": per_run[:500],
    "note_per_run_truncated": len(per_run) > 500,
}
REPORT_PATH.write_text(json.dumps(report, indent=2))
print(f"\\nWrote {REPORT_PATH}")
"""
)

nb = {
    "nbformat": 4,
    "nbformat_minor": 5,
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.11.0"},
    },
    "cells": cells,
}

OUT.write_text(json.dumps(nb, indent=1))
print("Wrote", OUT)
