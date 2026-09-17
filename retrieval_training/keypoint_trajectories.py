"""
Clip keypoints → fixed-length 126D hand trajectories.

Reads the exported clip-keypoints artifact (`KEYPOINTS_CACHE`, a list of row
dicts with per-frame 3D hand keypoints already expressed in the estimated
torso frame — see README), extracts the per-hand keypoints, and packs them
into padded [N, T, 126] trajectory tensors (21 joints × 3 coords × 2 hands,
dims [:63] = left hand, [63:] = right hand).

`build_dataset/build_trajectories.py` is the only consumer: it loads the
bundle, applies the configured DTW design transform (`dtw_designs.py`), and
writes the per-design trajectory artifact that training runs from.

Usage:
    from keypoint_trajectories import load_data_bundle_21j

    bundle = load_data_bundle_21j(num_videos=None)   # None = all videos
"""

import os
import numpy as np
import torch
from dataclasses import dataclass
from pathlib import Path

# ---------------------------------------------------------------------------
# Paths and environment
# ---------------------------------------------------------------------------
# Repo-root-relative by default (ABMR_PROJECT_ROOT overrides the root,
# matching config.py); CLIP_KEYPOINTS_CACHE overrides the file location.
# build_trajectories.py reassigns KEYPOINTS_CACHE to the run's data_dir.
_REPO_ROOT = Path(
    os.environ.get("ABMR_PROJECT_ROOT", Path(__file__).resolve().parents[1])
).expanduser()

KEYPOINTS_CACHE = Path(
    os.environ.get(
        "CLIP_KEYPOINTS_CACHE",
        _REPO_ROOT / "outputs" / "training_data" / "clip_keypoints.pt",
    )
).expanduser()


# ---------------------------------------------------------------------------
# DataBundle: clips + trajectories loaded once
# ---------------------------------------------------------------------------

@dataclass
class DataBundle:
    """Container for loaded clips + trajectories."""
    clips:          list           # keypoint-cache rows (trajectory-valid only)
    trajs:          torch.Tensor   # [N, T, 126]
    actual_lengths: torch.Tensor   # [N]
    N:              int
    T:              int


# ---------------------------------------------------------------------------
# Fetch clip rows for the first N videos
# ---------------------------------------------------------------------------

def get_clips_for_n_videos(num_videos):
    """Load the keypoints cache and keep the first `num_videos` videos.

    Args:
        num_videos: number of distinct video_numbers (ascending) to include;
                    `None` keeps every video (`[:None]` is the full slice).

    Returns:
        clips: list of row dicts with video_number, node_number, node_uid,
               keypoints_per_frame, left_hand_presence, right_hand_presence.
        video_numbers: sorted list of video numbers included
    """
    if not KEYPOINTS_CACHE.exists():
        raise FileNotFoundError(
            f"clip keypoints cache not found at {KEYPOINTS_CACHE} (see README)"
        )
    try:
        rows = torch.load(KEYPOINTS_CACHE, weights_only=False, mmap=True)
        print(f"  Loaded keypoints cache with mmap=True: {KEYPOINTS_CACHE}")
    except TypeError:
        rows = torch.load(KEYPOINTS_CACHE, weights_only=False)
        print(f"  Loaded keypoints cache without mmap support: {KEYPOINTS_CACHE}")
    video_numbers = sorted({r["video_number"] for r in rows})[:num_videos]
    vn_set = set(video_numbers)
    clips = [r for r in rows if r["video_number"] in vn_set]
    clips.sort(key=lambda r: (r["video_number"], r["node_number"]))
    return clips, video_numbers


# ---------------------------------------------------------------------------
# Extract raw 3D hand keypoints from clip rows
# ---------------------------------------------------------------------------

def extract_hand_keypoints(clips):
    """Extract raw 3D hand keypoints from clips.

    For each clip, pulls the `kpts_3d` [21, 3] array per frame per hand
    (`hands[0]["kpts_3d"]`), with no transformations applied.

    Args:
        clips: list of row dicts (from get_clips_for_n_videos).

    Returns:
        keypoints: list of dicts, one per clip. Each dict maps
            frame_key (str) -> {"L": array [21, 3] or None,
                                "R": array [21, 3] or None}
            Frames where a hand was not detected have None for that hand.
        valid_indices: list[int] — positions in `clips` that had keypoints.
    """
    keypoints = []
    valid_indices = []

    for i, clip in enumerate(clips):
        kpts = clip["keypoints_per_frame"]
        if kpts is None:
            continue

        clip_kpts = {}
        for fk in sorted((k for k in kpts.keys() if k != "_meta"), key=int):
            frame = kpts[fk]
            frame_hands = {}
            for hand_id in ("L", "R"):
                hands = frame.get(hand_id, [])
                if hands and "kpts_3d" in hands[0]:
                    frame_hands[hand_id] = np.asarray(hands[0]["kpts_3d"])
                else:
                    frame_hands[hand_id] = None
            clip_kpts[fk] = frame_hands

        keypoints.append(clip_kpts)
        valid_indices.append(i)

    if not keypoints:
        raise ValueError("No clips with keypoints found")

    return keypoints, valid_indices


def load_data_bundle_21j(num_videos):
    """Load clips + full 21-joint trajectories. Returns a DataBundle.

    Args:
        num_videos: how many videos (by ascending video_number) to include;
                    `None` for all.

    Returns:
        DataBundle with clips, trajs [N, T, 126], actual_lengths [N].
    """
    clips, _ = get_clips_for_n_videos(num_videos)
    kpts_list, kpts_valid = extract_hand_keypoints(clips)
    trajs, traj_valid, actual_lengths = keypoints_to_trajectories_21j(kpts_list)
    valid_clips = [clips[kpts_valid[i]] for i in traj_valid]
    N, T, _ = trajs.shape
    print(f"DataBundle (21j): {N} clips, T={T}, D={trajs.shape[-1]}, "
          f"from {'all' if num_videos is None else num_videos} videos")
    return DataBundle(clips=valid_clips, trajs=trajs, actual_lengths=actual_lengths, N=N, T=T)


# ---------------------------------------------------------------------------
# Convert keypoint dicts → fixed-length trajectory tensors
# ---------------------------------------------------------------------------

ALL_JOINTS = list(range(21))
HAND_TRAJ_DIM = len(ALL_JOINTS) * 3 * 2  # 126 = 21 joints × xyz × 2 hands


def keypoints_to_trajectories_21j(keypoints_list, min_frames=2):
    """Convert per-clip 3D keypoint dicts to fixed-length [M, T, 126] tensors.

    Per-hand fill rule: backfill before the first detection, forward-fill
    after the last, nearest detected frame in the interior; a hand that is
    never detected is all zeros. Clips are then padded to the max length by
    repeating their last frame.

    Args:
        keypoints_list: list of dicts, each mapping frame_key (str) ->
            {"L": ndarray [21, 3] or None, "R": ndarray [21, 3] or None}.
        min_frames: minimum keypoint frames to include a clip (default 2).

    Returns:
        trajs:          Tensor [M, T, 126] float32 (T = max frames across clips)
        valid_indices:  list[int] — positions in keypoints_list that survived
        actual_lengths: Tensor [M] int — pre-padding frame count per clip
    """
    joint_idx = ALL_JOINTS
    hand_dim = len(joint_idx) * 3  # 63
    raw_trajs = []
    valid_indices = []
    skipped = 0

    for i, clip_kpts in enumerate(keypoints_list):
        sorted_keys = sorted(clip_kpts.keys(), key=int)
        if len(sorted_keys) < min_frames:
            skipped += 1
            continue

        T_clip = len(sorted_keys)
        # Build per-hand arrays: [T_clip, hand_dim] each, with NaN for missing
        hand_arrays = {}
        for hand_id in ("L", "R"):
            arr = np.full((T_clip, hand_dim), np.nan, dtype=np.float32)
            for t, fk in enumerate(sorted_keys):
                kpts = clip_kpts[fk].get(hand_id)
                if kpts is not None:
                    arr[t] = kpts[joint_idx].flatten()
            hand_arrays[hand_id] = arr

        # Fill each hand independently
        parts = []
        for hand_id in ("L", "R"):
            arr = hand_arrays[hand_id]
            valid_mask = ~np.isnan(arr[:, 0])
            if not valid_mask.any():
                # Hand never detected — all zeros
                parts.append(np.zeros((T_clip, hand_dim), dtype=np.float32))
            else:
                first = int(np.argmax(valid_mask))
                last = T_clip - 1 - int(np.argmax(valid_mask[::-1]))
                # Backfill before first detection
                arr[:first] = arr[first]
                # Forward-fill after last detection
                arr[last + 1:] = arr[last]
                # Fill interior gaps with nearest detected frame
                for t in range(first, last + 1):
                    if np.isnan(arr[t, 0]):
                        # Find nearest valid frame
                        for offset in range(1, last - first + 1):
                            if t - offset >= first and not np.isnan(arr[t - offset, 0]):
                                arr[t] = arr[t - offset]
                                break
                            if t + offset <= last and not np.isnan(arr[t + offset, 0]):
                                arr[t] = arr[t + offset]
                                break
                parts.append(arr)

        raw_trajs.append(np.concatenate(parts, axis=1))  # [T_clip, 2*hand_dim]
        valid_indices.append(i)

    if skipped > 0:
        print(f"  Skipped {skipped} clips with < {min_frames} frames")
    if not raw_trajs:
        raise ValueError(f"No clips with >= {min_frames} keypoint frames")

    # Pad all clips to the max length by repeating last frame per hand
    actual_lengths = torch.tensor([t.shape[0] for t in raw_trajs], dtype=torch.long)
    max_len = max(t.shape[0] for t in raw_trajs)
    padded = []
    for t in raw_trajs:
        if t.shape[0] < max_len:
            pad = np.tile(t[-1:], (max_len - t.shape[0], 1))
            t = np.concatenate([t, pad], axis=0)
        padded.append(t)

    return torch.tensor(np.stack(padded), dtype=torch.float32), valid_indices, actual_lengths

