# Auto-generated from notebook config cell. Regenerate with tools if needed.
from pathlib import Path

# ──────────────────────────────────────────────
# INPUT: either a single video file OR a folder
# ──────────────────────────────────────────────
INPUT_PATH = "/Users/ral3ply/Documents/GitHub/visual-search/auto-labeler/inputs/uss_pf"
INPUT_PATH = "/Users/ral3ply/Google Drive/My Drive/auto-labeler/all/inputs/uss_pf/106041_20190311_LB_XO2617_012_165618/20190311_LB_XO2617_012_165618_Front.avi.mp4"

# Supported video extensions when INPUT_PATH is a folder (pipeline only ingests MP4)
VIDEO_EXTENSIONS = {".mp4"}
# Case-insensitive substring match on the filename (stem); skip e.g. topview when only Front/Rear matter.
# Set to () to disable filtering.
VIDEO_IGNORE_NAME_SUBSTRINGS = ("topview",)

# Output root — one sub-folder per video will be created here
OUTPUT_ROOT = Path("/Users/ral3ply/Documents/GitHub/visual-search/auto-labeler/outputs/test-video")

# Pipeline settings
TOP_N          = 3        # max dominant objects to detect
MAX_FRAMES     = None       # None = all frames; set e.g. 300 to cap
VIDEO_FPS_OVERRIDE = None   # None = use source FPS; float to override

# CPU: PyTorch thread pools (applied in Step 5 *before* loading models; ignored on CUDA)
# - These speed up matrix work *inside* each Qwen / SAM3 forward. They do **not** run several
#   forwards at once on one model (that is not thread-safe; use separate processes if you
#   need true parallelism—roughly N× model RAM for N workers).
# - Optional: set OMP_NUM_THREADS / MKL_NUM_THREADS in the shell to match, to avoid oversubscription.
CPU_TORCH_THREADS = None     # e.g. 8; None = default
CPU_INTEROP_THREADS = None  # e.g. 1–2 on CPU; None = default

# Models
QWEN_MODEL_ID  = "Qwen/Qwen2.5-VL-3B-Instruct"
SAM3_MODEL_ID  = "facebook/sam3"

# Label format for bounding boxes
# 'yolo'  → normalised cx cy w h  (one .txt per frame)
# 'coco'  → absolute x y w h      (one .txt per frame)
BBOX_LABEL_FORMAT = "yolo"

# ── YOLO detection + segmentation (Ultralytics; same idea as image pipeline) ─
# Keep YOLO_CLASS_NAMES / aliases in sync with qwen_sam3_image_pipeline Step 3.
YOLO_CLASS_NAMES = [
    "road",
    "pavement",
    "building",
    "wall",
    "car",
    "tree",
    "sky",
    "person",
    "fence",
    "pole",
    "sign",
    "barrier",
    "railing",
    "overhead",
    "cone",
    "structure",
]
YOLO_CLASS_ALIASES = {
    "automobile": "car",
    "suv": "car",
    "pedestrian": "person",
    "cars": "car",
    "cars parked": "car",
    "red car": "car",
    "black car": "car",
    "black suv": "car",
    "black SUV": "car",
    "minivan": "car",
    "white van": "car",
    "truck": "car",
    "volkswagen car": "car",
    "Volkswagen car": "car",
    "roadway": "road",
    "road surface": "road",
    "concrete road": "road",
    "concrete road surface": "road",
    "cobbled street": "road",
    "parking lot": "pavement",
    "concrete parking lot": "pavement",
    "brick pavement": "pavement",
    "concrete pavement": "pavement",
    "sidewalk": "pavement",
    "concrete floor": "pavement",
    "buildings": "building",
    "building exterior": "building",
    "building with green roof": "building",
    "rooftops": "building",
    "red roof": "building",
    "door": "building",
    "window": "building",
    "window blinds": "building",
    "concrete wall": "wall",
    "building wall": "wall",
    "trees": "tree",
    "trees in the distance": "tree",
    "hedge": "tree",
    "tree planter": "tree",
    "concrete barrier": "barrier",
    "red and white barrier": "barrier",
    "yellow poles": "pole",
    "street signs": "sign",
    "road signs": "sign",
    "signpost": "sign",
    "signboard": "sign",
    "advertising sign": "sign",
    "overhead sign": "sign",
    "license plate": "sign",
    "overhead structure": "overhead",
    "ceiling": "overhead",
    "ceiling beams": "overhead",
    "metal beams": "overhead",
    "overhead bridge": "overhead",
    "concrete overpass": "overhead",
    "carport": "overhead",
    "overhead canopy": "overhead",
    "traffic cone": "cone",
    "yellow cone": "cone",
    "metal railing": "railing",
    "metal staircase": "structure",
    "concrete pillars": "structure",
    "concrete planters": "structure",
    "concrete blocks": "structure",
    "metal pipes": "structure",
    "pipes": "structure",
    "white box": "structure",
}
YOLO_SKIP_UNMAPPED_PROMPTS = True
YOLO_LABEL_APPEND_CONFIDENCE = False  # True = 6 bbox cols (debug)
YOLO_SEG_POLY_EPS_RATIO = 0.002
YOLO_SEG_MIN_MASK_AREA = 4.0
# Per segmented frame: write images/frames/{frame:06d}.jpg + labels/segments/{frame:06d}.txt
VIDEO_EXPORT_FRAME_STRIDE = 1  # 1 = every input frame (after stride); >1 subsamples disk use
# False = no combined_top*_sam3.mp4 or per-prompt sam3_*.mp4 under videos/ (only per-frame images + labels)
VIDEO_EXPORT_PREVIEW_MP4S = False
# True = write images/overlays/{frame:06d}.jpg (boxes+masks on frame); timed separately as t_overlay_s (not in Export)
VIDEO_EXPORT_OVERLAY_IMAGES = True
# "video_session" = one Qwen on full video + SAM3 video propagate (default, faster)
# "per_frame_image" = write JPEGs, then per frame: Qwen(image)+SAM3(1 frame) (like image notebook; for timing tests)
VIDEO_PIPELINE_MODE = "video_session"
# per_frame_image: write all frames to images/frames before the per-frame Qwen+SAM3 loop
EXTRACT_FRAMES_TO_DISK_FIRST = True
PER_FRAME_PROGRESS_EVERY = 5   # 0 = no per-row prints; else every N *processed* frames
PER_FRAME_TIMING_MAX_ROWS = 500  # 0 = store all per_frame_timings; else cap
# qwen_per_frame_then_agg_sam3: Qwen on every frame, then top-K by vote, then one SAM3 video
QWEN_PER_FRAME_TOP_N = 5   # max object names to request per frame (feeds the global vote)
QWEN_AGGREGATE_TOP_K = 4  # "main" prompts: top this many by frequency across all frames

# ──────────────────────────────────────────────
