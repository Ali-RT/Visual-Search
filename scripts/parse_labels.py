"""
Parse ADAS label JSON files and extract per-image ground truth tags.

Two label formats are handled:
  1. Lane protocol (crosswalk / stopline): objects with `class` and `ddAttributes`
  2. Object detection (cones): `object_annotations` with `hierarchy_class` bounding boxes

Returns a list of dicts, each mapping an image path to its ground truth tags.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List


@dataclass
class ImageGroundTruth:
    image_path: str
    label_path: str
    use_case: str
    gt_tags: List[str] = field(default_factory=list)
    object_counts: Dict[str, int] = field(default_factory=dict)
    horizontal_types: List[str] = field(default_factory=list)
    lane_styles: List[str] = field(default_factory=list)
    frame_attributes: List[str] = field(default_factory=list)
    has_crosswalk: bool = False
    has_stopline: bool = False
    has_lane_marking: bool = False
    has_road_edge: bool = False
    has_cones: bool = False
    cone_count: int = 0


def _parse_lane_label(data: dict) -> dict:
    """Parse lane-protocol labels (crosswalk / stopline use cases)."""
    info: Dict[str, Any] = {
        "object_classes": {},
        "horizontal_types": [],
        "lane_styles": [],
        "frame_attributes": [],
        "marking_types": [],
    }

    for obj in data.get("objects", []):
        cls = obj.get("class", "unknown")
        info["object_classes"][cls] = info["object_classes"].get(cls, 0) + 1

        ddtypes = obj.get("ddtypes", [])
        ddattrs = obj.get("ddAttributes", [])

        for dt, da in zip(ddtypes, ddattrs):
            if dt == "horizontal_type":
                info["horizontal_types"].append(da)
            elif dt == "lane_style":
                info["lane_styles"].append(da)
            elif dt == "frame_attributes":
                if isinstance(da, list):
                    info["frame_attributes"].extend(da)
                else:
                    info["frame_attributes"].append(da)
            elif dt == "marking_type":
                info["marking_types"].append(da)

    return info


def _parse_cone_label(data: dict) -> dict:
    """Parse object-detection labels (cones use case)."""
    info: Dict[str, Any] = {"hierarchy_classes": {}}

    for obj in data.get("object_annotations", []) + data.get("objects", []):
        hc = obj.get("hierarchy_class", "unknown")
        if hc and hc != "unknown":
            info["hierarchy_classes"][hc] = info["hierarchy_classes"].get(hc, 0) + 1

    return info


CROSSWALK_HTYPES = {"zebra_crossing", "zebra_crossing_side"}
STOPLINE_HTYPES = {"stop/solid"}
CONE_CLASSES = {"Cone", "Barrel", "Beacon", "RedWhiteBollard", "ReflectorPost"}


def parse_image_label(image_path: str, label_path: str, use_case: str) -> ImageGroundTruth:
    """Parse a single label file and return structured ground truth."""
    gt = ImageGroundTruth(image_path=image_path, label_path=label_path, use_case=use_case)

    with open(label_path) as f:
        data = json.load(f)

    if "ProtocolID" in data:
        info = _parse_lane_label(data)
        gt.object_counts = info["object_classes"]
        gt.horizontal_types = info["horizontal_types"]
        gt.lane_styles = list(set(info["lane_styles"]))
        gt.frame_attributes = list(set(info["frame_attributes"]))

        gt.has_crosswalk = any(h in CROSSWALK_HTYPES for h in gt.horizontal_types)
        gt.has_stopline = any(h in STOPLINE_HTYPES for h in gt.horizontal_types)
        gt.has_lane_marking = gt.object_counts.get("lane_marking", 0) > 0
        gt.has_road_edge = gt.object_counts.get("road_edge", 0) > 0

    elif "object_annotations" in data or "label_info" in data:
        info = _parse_cone_label(data)
        gt.object_counts = info["hierarchy_classes"]
        cone_total = sum(
            c for cls, c in info["hierarchy_classes"].items() if cls in CONE_CLASSES
        )
        gt.has_cones = cone_total > 0
        gt.cone_count = cone_total

    # Build consolidated gt_tags list
    if gt.has_crosswalk:
        gt.gt_tags.append("crosswalk")
    if gt.has_stopline:
        gt.gt_tags.append("stopline")
    if gt.has_lane_marking:
        gt.gt_tags.append("lane_marking")
    if gt.has_road_edge:
        gt.gt_tags.append("road_edge")
    if gt.has_cones:
        gt.gt_tags.append("cones")
    for attr in gt.frame_attributes:
        gt.gt_tags.append(f"frame:{attr}")

    return gt


def parse_sampled_images(root: str | Path) -> List[ImageGroundTruth]:
    """Walk a label directory and parse all image+label pairs."""
    root = Path(root)
    results = []

    for use_case_dir in sorted(root.iterdir()):
        if not use_case_dir.is_dir():
            continue
        use_case = use_case_dir.name

        for subfolder in sorted(use_case_dir.iterdir()):
            if not subfolder.is_dir():
                continue

            images = list(subfolder.glob("*.png")) + list(subfolder.glob("*.jpg"))
            jsons = list(subfolder.glob("*.json"))

            if not images or not jsons:
                continue

            image_path = str(images[0])
            label_path = str(jsons[0])

            gt = parse_image_label(image_path, label_path, use_case)
            results.append(gt)

    return results


def build_gt_lookup(label_root: str | Path) -> Dict[str, ImageGroundTruth]:
    """
    Parse all labels under label_root and return a dict keyed by image filename
    (without extension) for fast lookup when matching to flattened images.
    """
    results = parse_sampled_images(label_root)
    lookup: Dict[str, ImageGroundTruth] = {}
    for gt in results:
        stem = Path(gt.image_path).stem
        lookup[stem] = gt
    return lookup


if __name__ == "__main__":
    import sys

    root = sys.argv[1] if len(sys.argv) > 1 else "data/new_images"
    results = parse_sampled_images(root)

    print(f"Parsed {len(results)} image-label pairs\n")

    labeled = sum(1 for r in results if r.gt_tags)
    print(f"With ground truth tags: {labeled}")
    print(f"Without ground truth tags: {len(results) - labeled}\n")

    for uc in sorted(set(r.use_case for r in results)):
        uc_items = [r for r in results if r.use_case == uc]
        tagged = sum(1 for r in uc_items if r.gt_tags)
        print(f"=== {uc} ({len(uc_items)} labeled, {tagged} with tags) ===")
        for r in uc_items[:5]:
            fname = Path(r.image_path).name
            print(f"  {fname}")
            print(f"    gt_tags: {r.gt_tags}")
            if r.horizontal_types:
                print(f"    h_types: {r.horizontal_types}")
            if r.cone_count:
                print(f"    cones: {r.cone_count}")
        if len(uc_items) > 5:
            print(f"  ... and {len(uc_items) - 5} more")
        print()
