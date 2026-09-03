"""V-JEPA 2.1 feature-extraction utilities.

Trimmed vendored copy of the multi-encoder extraction module: only the
V-JEPA 2.1 load/preprocess/forward triplet and the raw clip loader used by
`3get_jepa_features.py` are kept.

Clip metadata comes from the private clip database (`db` module, not
distributed with this repo — see README); the clip MP4s are read from
FRAMES_DIR/<video_number>/<node_uid>.mp4.
"""

import os
import numpy as np
import numpy.linalg  # eager load to prevent recursion in torch._dynamo + numpy 2.x
import torch
import cv2
from pathlib import Path

# Suppress noisy H.264 decoder warnings from FFmpeg (via cv2)
os.environ.setdefault("OPENCV_FFMPEG_LOGLEVEL", "-8")

# ---------------------------------------------------------------------------
# Paths and environment
# ---------------------------------------------------------------------------
# Repo-root-relative defaults (ABMR_PROJECT_ROOT overrides the root, matching
# config.py); CLIP_FRAMES_DIR / VJEPA2_MODEL_PATH override the individual
# locations.
_REPO_ROOT = Path(
    os.environ.get("ABMR_PROJECT_ROOT", Path(__file__).resolve().parents[2])
).expanduser()

FRAMES_DIR = Path(
    os.environ.get("CLIP_FRAMES_DIR", _REPO_ROOT / "outputs" / "frames")
).expanduser()

IMAGENET_DEFAULT_MEAN = (0.485, 0.456, 0.406)
IMAGENET_DEFAULT_STD = (0.229, 0.224, 0.225)


# ---------------------------------------------------------------------------
# Clip metadata from DB
# ---------------------------------------------------------------------------

def _get_clips_for_video(video_number):
    """Fetch all clip rows for a video from DB, ordered by node_number."""
    # `db` is the private clip-database module, not distributed with this
    # repo (see README).
    from db import query_clips as _db_query_clips

    rows = _db_query_clips(
        "SELECT video_number, video_uid, node_number, node_uid, "
        "keypoints_per_frame, start_sec, end_sec "
        "FROM selected_clips WHERE video_number = %s ORDER BY node_number",
        (int(video_number),))
    if not rows:
        raise ValueError(f"No clips found for video_number={video_number}")
    return rows


# ---------------------------------------------------------------------------
# Frame decoding
# ---------------------------------------------------------------------------

def _av_decode_frames(video_path, frame_indices, fps):
    """Decode specific frames via cv2 seek-once-then-grab.

    Seeks to the first target frame, then advances with grab() (no decode)
    to skip unwanted frames. Returns {frame_idx: np.array [H,W,C] uint8} in RGB.
    """
    sorted_targets = sorted(set(frame_indices))
    if not sorted_targets:
        return {}

    result = {}
    cap = cv2.VideoCapture(video_path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, sorted_targets[0])
    pos = sorted_targets[0]
    for target in sorted_targets:
        while pos < target:
            cap.grab()
            pos += 1
        ret, frame = cap.read()
        if ret:
            result[target] = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pos += 1
    cap.release()

    # Sanity check: flag duplicate frames
    frames_list = list(result.values())
    if len(frames_list) >= 2:
        n_dup = sum(np.array_equal(frames_list[i], frames_list[i + 1]) for i in range(len(frames_list) - 1))
        if n_dup > 0:
            print(f"  WARNING: {n_dup}/{len(frames_list)} consecutive duplicate frames in {video_path}")
        else:
            print(f"  OK: {len(frames_list)} frames decoded, no duplicates")

    return result


# ---------------------------------------------------------------------------
# Raw clip loading (no resize/normalize — each encoder does its own)
# ---------------------------------------------------------------------------

def load_clips_for_video_raw(video_number):
    """
    Load raw decoded frames for every clip in a video.

    Args:
        video_number (int): Video number in selected_clips.

    Returns:
        raw_clips:  list of np.ndarray, each [T, H, W, 3] uint8 RGB.
        clip_rows:  list of DB row dicts (same order as raw_clips).
    """
    rows = _get_clips_for_video(video_number)

    clip_mp4_dir = FRAMES_DIR / str(video_number)

    raw_clips = []
    clip_rows = []
    for row in rows:
        kpts = row["keypoints_per_frame"]
        if kpts is None:
            continue
        node_uid = row.get("node_uid")
        if not node_uid:
            continue
        clip_mp4 = clip_mp4_dir / f"{node_uid}.mp4"
        if not clip_mp4.exists():
            print(f"  Warning: clip MP4 not found: {clip_mp4}")
            continue

        clip_frame_indices = sorted(int(k) for k in kpts.keys())
        cap = cv2.VideoCapture(str(clip_mp4))
        clip_fps = cap.get(cv2.CAP_PROP_FPS)
        cap.release()

        frame_idx = np.array(clip_frame_indices, dtype=int)
        decoded = _av_decode_frames(str(clip_mp4), frame_idx.tolist(), clip_fps)
        frames = [decoded[f] for f in frame_idx.tolist() if f in decoded]
        if not frames:
            continue
        raw_clips.append(np.stack(frames))  # [T, H, W, 3] uint8
        clip_rows.append(row)

    return raw_clips, clip_rows


# ===========================================================================
# V-JEPA 2.1 ViT-Giant (384px, video)
# ===========================================================================

VJEPA2_IMG_SIZE = 384
VJEPA2_MODEL_PATH = os.environ.get(
    "VJEPA2_MODEL_PATH",
    str(_REPO_ROOT / "outputs" / "vjepa2_1_vitg_384.pt"),
)

def load_vjepa2():
    model_pt, _ = torch.hub.load(
        "facebookresearch/vjepa2", "vjepa2_1_vit_giant_384", pretrained=False
    )
    ckpt = torch.load(VJEPA2_MODEL_PATH, weights_only=True, map_location="cuda")
    encoder_sd = {
        k.replace("module.", "").replace("backbone.", ""): v
        for k, v in ckpt["target_encoder"].items()
    }
    model_pt.load_state_dict(encoder_sd, strict=True)
    model_pt.cuda().eval()
    print(f"Loaded V-JEPA 2.1 ViT-Giant from {VJEPA2_MODEL_PATH}")
    return model_pt

def preprocess_vjepa2(raw_clip):
    """[T,H,W,3] uint8 -> [C,T,H,W] float32 GPU (384px, ImageNet norm)"""
    import torch.nn.functional as F

    img_size = VJEPA2_IMG_SIZE
    mean = torch.tensor(IMAGENET_DEFAULT_MEAN, device="cuda").view(1, 3, 1, 1)
    std = torch.tensor(IMAGENET_DEFAULT_STD, device="cuda").view(1, 3, 1, 1)

    # [T, H, W, C] -> [T, C, H, W] -> GPU float
    clip_t = torch.from_numpy(raw_clip).permute(0, 3, 1, 2).float().div_(255.0).cuda()

    # Resize shorter side to img_size
    h, w = clip_t.shape[2], clip_t.shape[3]
    scale = img_size / min(h, w)
    new_h, new_w = int(round(h * scale)), int(round(w * scale))
    clip_t = F.interpolate(clip_t, size=(new_h, new_w), mode="bilinear", align_corners=False)

    # Center crop
    top = (new_h - img_size) // 2
    left = (new_w - img_size) // 2
    clip_t = clip_t[:, :, top:top + img_size, left:left + img_size]

    # Normalize and permute to [C, T, H, W]
    clip_t = clip_t.sub_(mean).div_(std).permute(1, 0, 2, 3)
    return clip_t

def forward_vjepa2(model, clip_tensor):
    """Returns [num_tokens, 1408] where num_tokens = T//2 * 576"""
    with torch.inference_mode():
        features = model(clip_tensor.unsqueeze(0).cuda())  # [1, num_patches, 1408]
    return features.squeeze(0)  # [num_patches, 1408]
