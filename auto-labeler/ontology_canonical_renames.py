"""
Ontology merges for standardize_open_vocab_extend_yolo.ipynb.

Keys must match image-pipeline normalization: lowercased, stripped, internal
spaces collapsed (same as norm_key() in the notebook). Values are canonical
class strings (also lower case); they become YOLO `names` entries after extend.

After explicit CANONICAL_RENAMES lookup, canonicalize_label() applies a small
set of tail heuristics (vehicle / cone / sign / road / building / …) so long
Qwen phrases collapse without listing every variant.

Edit this file and re-run the standardize notebook. Keep keys in sync with
strings Qwen actually emitted (mine_qwen_taxonomy / extended_yolo_class_list).
"""

from __future__ import annotations

import re

# --- Vehicles (color / brand / plural → head noun) ---------------------------------
CAR_ALIASES: dict[str, str] = {
    "red car": "car",
    "black car": "car",
    "blue car": "car",
    "white car": "car",
    "gray car": "car",
    "grey car": "car",
    "silver car": "car",
    "yellow car": "car",
    "volkswagen car": "car",
    "skoda superb": "car",
    "mercedes-benz cars": "car",
    "cars parked": "car",
    "parked cars": "car",
    "cars": "car",
    "black suv": "car",
    "black jeep": "car",
    "jeep": "car",
    "silver van": "van",
    "white van": "van",
    "gray van": "van",
    "grey van": "van",
    "minivan": "van",
    "tow truck": "truck",
    "blue truck": "truck",
    "pickup truck": "truck",
    "trailer": "truck",
    "red bus": "bus",
    "red and white striped barrier": "barrier",
    "red and white barrier": "barrier",
    "red and white sign": "sign",
}

# --- Cones / barriers / bollards -----------------------------------------------------
TRAFFIC_ALIASES: dict[str, str] = {
    "traffic cone": "cone",
    "yellow cone": "cone",
    "orange cone": "cone",
    "yellow traffic cone": "cone",
    "yellow traffic cones": "cone",
    "yellow and black traffic cone": "cone",
    "orange traffic cone": "cone",
    "cones": "cone",
    "concrete barrier": "barrier",
    "concrete barriers": "barrier",
    "barrier wall": "barrier",
    "barrier poles": "barrier",
    "barriers": "barrier",
    "barrier": "barrier",
    "yellow bollards": "bollard",
    "yellow post": "bollard",
    "yellow posts": "bollard",
    "yellow poles": "bollard",
}

# --- Signs / poles (vertical street furniture) -------------------------------------
SIGN_POLE: dict[str, str] = {
    "street sign": "sign",
    "road sign": "sign",
    "road signs": "sign",
    "signboard": "sign",
    "overhead sign": "sign",
    "advertising sign": "sign",
    "advertising billboard": "sign",
    "white sign": "sign",
    "white signboard": "sign",
    "signpost": "pole",
    "streetlight": "pole",
    "streetlights": "pole",
    "street lamp": "pole",
    "street lamps": "pole",
    "light pole": "pole",
    "metal pole": "pole",
    "concrete pole": "pole",
    "concrete poles": "pole",
    "concrete post": "pole",
    "utility pole": "pole",
}

# --- Vegetation --------------------------------------------------------------------
VEG: dict[str, str] = {
    "tree": "vegetation",
    "trees": "vegetation",
    "tree trunks": "vegetation",
    "tree line": "vegetation",
    "tree planter": "vegetation",
    "tree planters": "vegetation",
    "trees in planters": "vegetation",
    "hedge": "vegetation",
    "bushes": "vegetation",
    "bush": "vegetation",
    "green grass": "vegetation",
    "grass field": "vegetation",
    "grass": "vegetation",
    "greenery": "vegetation",
    "green plants": "vegetation",
    "plant": "vegetation",
}

# --- Road / path / parking surface (reduce synonym explosion) ----------------------
ROADSCAPE: dict[str, str] = {
    "roadway": "road",
    "road surface": "road",
    "asphalt road": "road",
    "paved area": "road",
    "paved walkway": "sidewalk",
    "cobbled street": "road",
    "brick path": "sidewalk",
    "brick pavement": "sidewalk",
    "concrete road": "road",
    "concrete pavement": "sidewalk",
    "concrete sidewalk": "sidewalk",
    "concrete parking lot": "parking lot",
    "concrete street": "road",
    "white lines on ground": "road marking",
    "white lines on the road": "road marking",
    "white lines": "road marking",
    "white line": "road marking",
    "road markings": "road marking",
    "parking lines": "road marking",
    "parking space": "parking lot",
    "parking spaces": "parking lot",
    "parking garage": "parking lot",
    "driveway": "road",
}

# --- Building / wall / structure ----------------------------------------------------
BUILT: dict[str, str] = {
    "buildings": "building",
    "building wall": "building",
    "building facade": "building",
    "building exterior": "building",
    "building entrance": "building",
    "building roof": "building",
    "building structure": "building",
    "building structures": "building",
    "building in background": "building",
    "white building": "building",
    "building with red roof": "building",
    "rooftops": "building",
    "rooftop buildings": "building",
    "wall": "wall",
    "white wall": "wall",
    "concrete wall": "wall",
    "barrier wall": "barrier",
    "garage door": "building",
    "window": "building",
    "windows": "building",
    "glass window": "building",
    "building window": "building",
    "door": "building",
    "ceiling": "ceiling",
    "ceiling panels": "ceiling",
    "ceiling beams": "ceiling",
    "concrete ceiling": "ceiling",
    "floor": "floor",
    "concrete floor": "floor",
    "overhead structure": "overhead",
    "overhead bridge": "overhead",
    "overhead canopy": "overhead",
    "overhead shelter": "overhead",
    "overpass": "overhead",
    "canopy": "overhead",
}

# --- Ego / interior / non-instance noise -------------------------------------------
NOISE: dict[str, str] = {
    "car dashboard": "ego interior",
    "car interior": "ego interior",
    "car hood": "ego interior",
    "car windshield": "ego interior",
    "wiper blades": "ego interior",
    "car mirror": "ego interior",
    "car rearview mirror": "ego interior",
    "car door handle": "ego interior",
    "car front": "ego interior",
    "black dashboard": "ego interior",
    "sunroof": "ego interior",
    "sunlight": "scene artifact",
    "shadow": "scene artifact",
    "fog": "scene artifact",
    "darkness": "scene artifact",
    "power lines": "scene artifact",
    "wires": "scene artifact",
    "vehicle": "car",
    "vehicles": "car",
    "logo": "scene artifact",
    "text": "scene artifact",
}

# --- Misc near-duplicates seen in open-vocab lists ---------------------------------
MISC: dict[str, str] = {
    "metal structure": "structure",
    "concrete structure": "structure",
    "building structure": "building",
    "metal roof": "roof",
    "red roof": "roof",
    "metal beams": "structure",
    "metal beam": "structure",
    "metal railing": "railing",
    "metal staircase": "stairs",
    "staircase": "stairs",
    "metal pipes": "structure",
    "metal fence": "fence",
    "fence": "fence",
    "concrete fence": "fence",
    "concrete grid": "structure",
    "concrete blocks": "structure",
    "concrete planters": "planter",
    "planters": "planter",
    "planter": "planter",
    "metal carts": "cart",
    "shopping carts": "cart",
    "shopping cart": "cart",
    "garbage bin": "trash bin",
    "trash bin": "trash bin",
    "sidewalks": "sidewalk",
    "columns": "structure",
    "pole": "pole",
    "concrete": "structure",
    "ground": "road",
    "sky": "sky",
    "blue sky": "sky",
    "pedestrians crossing": "person",
    "pedestrian crossing": "road marking",
}

# Merge all sections (later dicts override earlier on duplicate keys — avoid dupes)
CANONICAL_RENAMES: dict[str, str] = {}
for _d in (
    CAR_ALIASES,
    TRAFFIC_ALIASES,
    SIGN_POLE,
    VEG,
    ROADSCAPE,
    BUILT,
    NOISE,
    MISC,
):
    CANONICAL_RENAMES.update(_d)


def norm_key(s: str) -> str:
    """Match image pipeline / notebook `_open_vocab_norm_key`."""
    return " ".join(str(s).strip().lower().split())


_ARTICLE = re.compile(r"^(a|an|the)\s+", re.I)
_COLOR_PREFIX = re.compile(
    r"^(red|blue|white|black|gray|grey|silver|yellow|orange|green|brown|beige|"
    r"gold|dark|light|bright|pale|metallic|navy|maroon|tan|cream|purple|pink|"
    r"brightly lit|dimly lit)\s+",
    re.I,
)


def _strip_article_and_colors(k: str, max_rounds: int = 8) -> str:
    """Peel leading articles / color adjectives Qwen often prefixes."""
    for _ in range(max_rounds):
        nk = norm_key(_ARTICLE.sub("", k))
        if nk != k:
            k = nk
            continue
        nk = norm_key(_COLOR_PREFIX.sub("", k))
        if nk != k:
            k = nk
            continue
        break
    return k


# Suffix / phrase heuristics: first match wins (put more specific patterns first).
_TAIL_RULES: list[tuple[re.Pattern[str], str]] = [
    (
        re.compile(
            r"\b(traffic cone|traffic cones|safety cone|safety cones|"
            r"construction cone|construction cones)\b"
        ),
        "cone",
    ),
    (
        re.compile(
            r"\b(guard rail|guardrail|guardrails|crash barrier|"
            r"jersey barrier|jersey barriers)\b"
        ),
        "barrier",
    ),
    (re.compile(r"\b(traffic lights?|stoplights?|stop lights?)\b"), "traffic light"),
    (re.compile(r"\b(billboards?)\s*$"), "sign"),
    (re.compile(r"\b(buses|bus)\s*$"), "bus"),
    (
        re.compile(
            r"\b(pickup trucks?|tow trucks?|fire trucks?|dump trucks?|"
            r"delivery trucks?|garbage trucks?|cement trucks?|"
            r"tractor trailers?|semi trucks?|trucks?|trailers?|rigs?)\s*$"
        ),
        "truck",
    ),
    (re.compile(r"\b(minivans?|delivery vans?|vans?)\s*$"), "van"),
    (re.compile(r"\b(motorcycles?|motorbikes?|scooters?)\s*$"), "motorcycle"),
    (re.compile(r"\b(bicycles?|bikes?)\s*$"), "bicycle"),
    (
        re.compile(
            r"\b(cars?|suvs?|sedans?|hatchbacks?|coupes?|wagons?|convertibles?|"
            r"roadsters?|crossovers?|subcompacts?|compacts?|automobiles?|jeeps?|"
            r"taxis?|cabs?|limousines?|ambulances?|vehicles?)\s*$"
        ),
        "car",
    ),
    (re.compile(r"\b(trains?|locomotives?|subways?|metros?)\s*$"), "train"),
    (re.compile(r"\b(cones?|pylons?)\s*$"), "cone"),
    (re.compile(r"\b(bollards?)\s*$"), "bollard"),
    (re.compile(r"\b(barriers?)\s*$"), "barrier"),
    (re.compile(r"\b(signs?)\s*$"), "sign"),
    (
        re.compile(
            r"\b(poles?|posts?|streetlights?|street lights?|"
            r"lampposts?|lamp posts?|telephone poles?|utility poles?)\s*$"
        ),
        "pole",
    ),
    (
        re.compile(
            r"\b(trees?|shrubs?|bushes?|hedges?|saplings?|branches?|"
            r"leaves?|foliage|ivy|lawns?|meadows?|greenery|plants?)\s*$"
        ),
        "vegetation",
    ),
    (
        re.compile(
            r"\b(people|persons?|pedestrians?|walkers?|joggers?|"
            r"crowds?|commuters?)\s*$"
        ),
        "person",
    ),
    (re.compile(r"\b(cyclists?|bike riders?)\s*$"), "person"),
    (
        re.compile(
            r"\b(buildings?|houses?|cottages?|warehouses?|skyscrapers?|"
            r"storefronts?|churches?|stadiums?)\s*$"
        ),
        "building",
    ),
    (re.compile(r"\b(roofs?|rooves|facades?)\s*$"), "wall"),
    (
        re.compile(
            r"\b(parking lots?|parking areas?|parking structures?|car parks?)\s*$"
        ),
        "parking lot",
    ),
    (
        re.compile(
            r"\b(crosswalks?|zebra crossings?|intersections?|"
            r"junctions?|roundabouts?)\s*$"
        ),
        "road",
    ),
    (
        re.compile(
            r"\b(roads?|streets?|highways?|freeways?|boulevards?|avenues?|"
            r"lanes?|asphalt|tarmac|pavement)\s*$"
        ),
        "road",
    ),
    (re.compile(r"\b(sidewalks?|footpaths?|walkways?|paths?)\s*$"), "sidewalk"),
    (re.compile(r"\b(curbs?|kerbs?)\s*$"), "sidewalk"),
    (re.compile(r"\b(skies|sky)\s*$"), "sky"),
    (re.compile(r"\b(benches?|public seating)\s*$"), "structure"),
    (re.compile(r"\b(stairs?|steps|staircases?|stairways?)\s*$"), "stairs"),
    (re.compile(r"\b(railings?|handrails?)\s*$"), "railing"),
    (re.compile(r"\b(fences?)\s*$"), "fence"),
]


def _heuristic_canon(k: str) -> str:
    if not k:
        return k
    for rx, head in _TAIL_RULES:
        if rx.search(k):
            return head
    return k


def canonicalize_label(raw: str, renames: dict[str, str] | None = None) -> str:
    """
    norm_key → explicit table lookup → strip article/color + table again →
    tail heuristics → optional final table lookup.
    """
    table = CANONICAL_RENAMES if renames is None else renames
    k = norm_key(raw)
    if not k:
        return k
    k = norm_key(table.get(k, k))
    stripped = _strip_article_and_colors(k)
    if stripped != k:
        k = norm_key(table.get(stripped, stripped))
    k2 = _heuristic_canon(k)
    if k2 != k:
        k = norm_key(table.get(k2, k2))
    return norm_key(k)


__all__ = ["CANONICAL_RENAMES", "norm_key", "canonicalize_label"]
