"""Extract per-token V-JEPA 2.1 features for all training clips.

Loads the V-JEPA 2.1 ViT-Giant encoder once, iterates over the clips in the
training dataset (same clips as the trajectory-building step, sourced from
clip_keypoints.pt), decodes the keypoint-sampled frames, and saves unaltered
per-token features.

Output: outputs/training_data/jepa_features/<video_number>/<node_uid>.pt
  Each .pt file: Tensor[num_tokens, 1408] in float16 (one file per clip).
  num_tokens = (T // 2) * 576, where T = number of keypoint-sampled frames.
  All clips for a video are saved atomically after the whole video is processed.

Resume-safe: skips videos whose output directory already exists (pass --force
to re-extract everything).

Parallel use: launch the script in N terminals pointing at the same --out-dir.
Each video is claimed atomically via mkdir of a `.{vn}_tmp/` directory, so two
workers won't process the same video. If a worker crashes mid-video it leaves
a stale `.{vn}_tmp/` behind that blocks future runs; pass --reclaim-stale on a
later run (when no other workers are active) to remove leftover claims. Do not
combine --force with concurrent workers.

Usage:
    python 3get_jepa_features.py                    # all training clips (default)
    python 3get_jepa_features.py -n 50              # cap to first 50 videos (smoke test)
    python 3get_jepa_features.py --force            # wipe and re-extract (single-process only)
    python 3get_jepa_features.py --reclaim-stale    # clear stale .{vn}_tmp claims, then run
"""

import argparse
import sys
import time
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
TRAINING_DIR = SCRIPT_DIR.parent  # retrieval_training/ (dtw_cknna_data.py lives here)

for _p in (str(SCRIPT_DIR), str(TRAINING_DIR)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import torch

import dtw_cknna_data

from encoder_utils import (
    FRAMES_DIR,
    load_vjepa2,
    preprocess_vjepa2,
    forward_vjepa2,
    load_clips_for_video_raw,
)
from dtw_cknna_data import get_clips_for_n_videos

DEFAULT_OUT_DIR = dtw_cknna_data.KEYPOINTS_CACHE.parent / "jepa_features"


def get_training_clips_by_video(num_videos):
    """Load training clips from cache and group by video_number.

    `num_videos` is the upper bound passed to `get_clips_for_n_videos`. Pass
    `None` (or any value ≥ total videos available) to include everything;
    `_get_clips_from_cache` slices `video_numbers[:num_videos]`, which is a
    no-op when `num_videos is None` or larger than the cached set.

    Returns dict[video_number -> list[node_uid]].
    """
    clips, _ = get_clips_for_n_videos(num_videos, use_cache=True)
    by_video = {}
    for clip in clips:
        if clip.get("keypoints_per_frame") is None:
            continue
        vn = clip["video_number"]
        by_video.setdefault(vn, []).append(clip["node_uid"])
    return by_video


def main():
    parser = argparse.ArgumentParser(
        description="Extract per-token V-JEPA 2.1 features for training clips"
    )
    parser.add_argument("-n", "--num-videos", type=int, default=-1,
                        help="Cap on number of videos to process "
                             "(-1 = all videos in the keypoints cache; default)")
    parser.add_argument("-o", "--out-dir", type=Path, default=DEFAULT_OUT_DIR,
                        help=f"Output directory (default: {DEFAULT_OUT_DIR})")
    parser.add_argument("--force", action="store_true",
                        help="Delete existing .pt files and re-extract")
    parser.add_argument("--reclaim-stale", action="store_true",
                        help="Remove leftover .{vn}_tmp claim directories at startup "
                             "(use only when no other workers are running)")
    parser.add_argument("--depth-grounded", action="store_true",
                        help="Use depth_grounded_clip_keypoints.pt as clip source")
    args = parser.parse_args()

    if args.depth_grounded:
        dtw_cknna_data.KEYPOINTS_CACHE = (
            dtw_cknna_data.KEYPOINTS_CACHE.parent / "depth_grounded_clip_keypoints.pt")

    # `-1` means "no cap" (process every video in the keypoints cache).
    # `_get_clips_from_cache` slices `[:num_videos]`, and `[:None]` is the
    # all-elements slice in Python.
    num_videos = None if args.num_videos < 0 else args.num_videos
    print(f"[get_jepa_features] num_videos="
          f"{'all' if num_videos is None else num_videos}  out_dir={args.out_dir}")

    # --- Identify training clips ---
    print(f"[get_jepa_features] Loading training clip list from cache...")
    t0 = time.time()
    clips_by_video = get_training_clips_by_video(num_videos)

    frames_dir = FRAMES_DIR
    videos_with_frames = {int(p.name) for p in frames_dir.iterdir() if p.is_dir() and p.name.isdigit()}
    before = len(clips_by_video)
    clips_by_video = {vn: uids for vn, uids in clips_by_video.items() if vn in videos_with_frames}
    print(f"[get_jepa_features] Filtered to videos with frames on disk: "
          f"{before} -> {len(clips_by_video)} videos")

    total_clips = sum(len(v) for v in clips_by_video.values())
    print(f"[get_jepa_features] {total_clips} clips across "
          f"{len(clips_by_video)} videos in {time.time() - t0:.1f}s")

    args.out_dir.mkdir(parents=True, exist_ok=True)

    if args.force:
        import shutil
        removed = 0
        for p in args.out_dir.iterdir():
            if p.is_dir():
                shutil.rmtree(p)
                removed += 1
            elif p.suffix == ".pt":
                p.unlink()
                removed += 1
        if removed:
            print(f"[get_jepa_features] --force: removed {removed} existing entries")

    if args.reclaim_stale:
        import shutil
        stale = [p for p in args.out_dir.iterdir()
                 if p.is_dir() and p.name.startswith(".") and p.name.endswith("_tmp")]
        for p in stale:
            shutil.rmtree(p)
        if stale:
            print(f"[get_jepa_features] --reclaim-stale: removed {len(stale)} stale claims")

    # --- Load model ---
    print(f"[get_jepa_features] Loading V-JEPA 2.1 ViT-Giant...")
    model = load_vjepa2()

    # --- Extract features ---
    video_numbers = sorted(clips_by_video.keys())
    clips_done = 0
    t_start = time.time()

    import shutil

    for vid_idx, vn in enumerate(video_numbers):
        video_dir = args.out_dir / str(vn)
        old_pt = args.out_dir / f"{vn}.pt"
        if video_dir.exists() or old_pt.exists():
            clips_done += len(clips_by_video[vn])
            continue

        tmp_dir = args.out_dir / f".{vn}_tmp"
        try:
            tmp_dir.mkdir(parents=True, exist_ok=False)
        except FileExistsError:
            print(f"\n[{vid_idx+1}/{len(video_numbers)}] Video {vn} "
                  f"claimed by another worker, skipping")
            clips_done += len(clips_by_video[vn])
            continue

        target_uids = set(clips_by_video[vn])

        print(f"\n[{vid_idx+1}/{len(video_numbers)}] Video {vn} "
              f"({len(target_uids)} clips)")
        try:
            raw_clips, clip_rows = load_clips_for_video_raw(vn)
        except Exception as e:
            print(f"  ERROR loading clips: {e}")
            shutil.rmtree(tmp_dir, ignore_errors=True)
            continue

        results = {}
        for i, (raw_clip, row) in enumerate(zip(raw_clips, clip_rows)):
            node_uid = row["node_uid"]
            if node_uid not in target_uids:
                continue
            print(f"  [{i+1}/{len(raw_clips)}] node {row['node_number']} "
                  f"({node_uid[:8]}...) T={raw_clip.shape[0]}", end=" ", flush=True)
            try:
                clip_tensor = preprocess_vjepa2(raw_clip)
                features = forward_vjepa2(model, clip_tensor)
                results[node_uid] = features.detach().cpu().half()
                print(f"-> {list(features.shape)}")
                del clip_tensor, features
                torch.cuda.empty_cache()
            except Exception as e:
                print(f"ERROR: {e}")

        if results:
            for node_uid, feat in results.items():
                torch.save(feat, tmp_dir / f"{node_uid}.pt")
            tmp_dir.rename(video_dir)
            print(f"  Saved {len(results)} clips to {video_dir}/")
        else:
            shutil.rmtree(tmp_dir, ignore_errors=True)
        clips_done += len(target_uids)

        if (vid_idx + 1) % 50 == 0:
            elapsed = time.time() - t_start
            rate = clips_done / elapsed if elapsed > 0 else 0
            remaining = total_clips - clips_done
            eta = remaining / rate if rate > 0 else 0
            print(f"  Progress: {clips_done}/{total_clips} clips, "
                  f"{elapsed / 60:.1f} min elapsed, ETA {eta / 60:.1f} min")

    del model
    torch.cuda.empty_cache()

    elapsed = time.time() - t_start
    n_videos = len([p for p in args.out_dir.iterdir() if p.is_dir() and not p.name.startswith(".")])
    print(f"\n[get_jepa_features] Done. {n_videos} videos, "
          f"{clips_done} clips in {elapsed / 60:.1f} min")


if __name__ == "__main__":
    main()
