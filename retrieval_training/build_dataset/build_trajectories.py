"""Build a packed, design-transformed trajectory tensor for a training run.

Mirrors the trajectory construction of the offline dataset builder
(not distributed with this repo) but writes a single packed `.pt` artifact (no
DTW matrix, no clip index — those already exist in `data_dir`). Used at
training time to compute per-batch DTW online without re-doing the
keypoints → trajectory transform on every step.

Output schema:
    {
        "trajectories": Tensor[N, T_max, D] float32,
        "lengths":      Tensor[N]            int32,
        "design":       str,
        "clip_keys":    list[(video_number, node_uid)],
        "T_max":        int,
        "D":            int,
        "num_videos":   int,
    }

Also emits `clip_index_{design}.json` next to the `.pt` (same schema as the
one previously written by the legacy offline builder) so dataset code does
not depend on it for clip-index provenance.

Row order matches the existing `clip_index_{design}.json` in `data_dir` if
one exists (kept index-compatible with `dtw_matrix_{design}.npy` during the
transition). If absent, the artifact uses `bundle.clips` order and that
ordering is the one written to the emitted clip_index.

CLI:
    python build_trajectories.py \\
        --data-dir outputs/training_data \\
        --design first_frame_midpoint_hand_length_no_z \\
        --output /path/to/trajectories_X.pt          # all videos (default)
    python build_trajectories.py ... -n 50           # cap to first 50 videos (smoke test)
"""

import argparse
import json
import sys
import time
from pathlib import Path

import torch

SCRIPT_DIR = Path(__file__).resolve().parent
CKNNA_DIR = SCRIPT_DIR.parent  # retrieval_training/ (dtw_cknna.py lives here)

if str(CKNNA_DIR) not in sys.path:
    sys.path.insert(0, str(CKNNA_DIR))


def build_trajectories(
    data_dir: Path,
    dtw_design: str,
    output_path: Path,
    num_videos: int | None = None,
    max_t: int = 42,
    max_clips: int | None = None,
    clip_subsample_seed: int = 42,
    use_depth_grounded: bool = False,
    require_jepa_features: bool = True,
) -> None:
    """Produce `${output_path}` with the design-transformed trajectories.

    Args:
        data_dir:    directory with `clip_keypoints.pt` (and optionally
                     `clip_index_{design}.json` to lock row ordering).
        dtw_design:  key into `DTW_DESIGNS` from dtw_cknna.py.
        output_path: full path including filename for the saved `.pt`.
        num_videos:  upper bound on videos to include. `None` (default) =
                     every video in the keypoints cache. `_get_clips_from_cache`
                     slices `[:num_videos]`, and `[:None]` is the all-elements
                     slice in Python.
        max_t:       hard cap on sequence length, applied before transform
                     (mirrors the legacy offline builder).
        max_clips:   if not None, seed-subsample the loaded bundle down to this
                     many clips before transform/save. Shrinks every downstream
                     artifact (trajectories tensor, DTW-neighbors cache, eval
                     N×N matrices) since they all key off this file.
        clip_subsample_seed: RNG seed for `max_clips` subsample. Stable across
                     runs of the same seed, so the trajectory artifact and the
                     holdout split see the same clip pool.
        use_depth_grounded: if True, read from depth_grounded_clip_keypoints.pt
                     instead of clip_keypoints.pt.
        require_jepa_features: restrict clips to those with an on-disk JEPA
                     feature file. Required for JEPA input mode; disable for
                     trajectory input mode, which never reads JEPA features.
    """
    import dtw_cknna_data
    cache_name = "depth_grounded_clip_keypoints.pt" if use_depth_grounded else "clip_keypoints.pt"
    dtw_cknna_data.KEYPOINTS_CACHE = data_dir / cache_name

    from dtw_cknna_data import load_data_bundle_21j
    from dtw_cknna import DTW_DESIGNS, TRAJ_TRANSFORMS

    if dtw_design not in DTW_DESIGNS:
        raise ValueError(
            f"Unknown dtw_design={dtw_design!r}; "
            f"valid keys: {sorted(DTW_DESIGNS.keys())}"
        )

    print(f"[build_trajectories] design={dtw_design}  "
          f"num_videos={'all' if num_videos is None else num_videos}  "
          f"data_dir={data_dir}  output={output_path}")

    t0 = time.time()
    bundle = load_data_bundle_21j(num_videos, use_cache=True)
    print(f"[build_trajectories] loaded bundle: N={bundle.N} T={bundle.T} "
          f"D={bundle.trajs.shape[-1]} in {time.time() - t0:.1f}s")

    # JEPA input mode can only use clips with feature files. Trajectory input
    # mode does not read JEPA at all and must retain the full trajectory pool.
    eligible_idx = list(range(bundle.N))
    n_missing = 0
    if require_jepa_features:
        features_dir = data_dir / "jepa_features"
        available = set()
        if features_dir.is_dir():
            for vdir in features_dir.iterdir():
                if not vdir.is_dir() or vdir.name.startswith("."):
                    continue
                vn = int(vdir.name)
                for pt in vdir.glob("*.pt"):
                    available.add((vn, pt.stem))

        eligible_idx = [
            i for i, c in enumerate(bundle.clips)
            if (int(c["video_number"]), str(c["node_uid"])) in available
        ]
        n_missing = bundle.N - len(eligible_idx)
        if n_missing:
            print(f"[build_trajectories] dropping {n_missing} clips without "
                  f"jepa_features ({len(eligible_idx)} of {bundle.N} remain)")
    else:
        print("[build_trajectories] trajectory input mode: retaining clips "
              "without jepa_features")

    if max_clips is not None and 0 < max_clips < len(eligible_idx):
        import numpy as _np
        rng = _np.random.default_rng(clip_subsample_seed)
        pick = sorted(rng.choice(len(eligible_idx), size=max_clips, replace=False).tolist())
        keep_idx = [eligible_idx[i] for i in pick]
        print(f"[build_trajectories] subsampled to {len(keep_idx)} clips "
              f"(seed={clip_subsample_seed})")
    elif n_missing:
        keep_idx = eligible_idx
    else:
        keep_idx = None

    if keep_idx is not None:
        keep_t = torch.as_tensor(keep_idx, dtype=torch.long)
        bundle.trajs = bundle.trajs.index_select(0, keep_t).contiguous()
        bundle.actual_lengths = bundle.actual_lengths.index_select(0, keep_t).contiguous()
        bundle.clips = [bundle.clips[i] for i in keep_idx]
        bundle.N = len(keep_idx)

    if bundle.N == 0:
        requirement = " with jepa_features" if require_jepa_features else ""
        raise RuntimeError(f"No eligible clips{requirement} found in {data_dir}")

    # Truncate trajectory length, mirroring the legacy offline builder.
    if bundle.T > max_t:
        n_truncated = int((bundle.actual_lengths > max_t).sum().item())
        print(f"[build_trajectories] truncating T={bundle.T} -> {max_t} "
              f"({n_truncated} clips affected)")
        bundle.trajs = bundle.trajs[:, :max_t, :].contiguous()
        bundle.actual_lengths = bundle.actual_lengths.clamp(max=max_t)
        bundle.T = max_t

    # Row order = bundle.clips order. The trajectory artifact's `clip_keys`
    # is the source of truth for clip_index — VideoClipDataset reads from it.
    tx_fn = TRAJ_TRANSFORMS[DTW_DESIGNS[dtw_design]["tx"]]
    trajs_tx, lens_tx = tx_fn(bundle.trajs, bundle.actual_lengths)
    trajs_tx = trajs_tx.to(torch.float32).contiguous()
    lens_tx = lens_tx.to(torch.int32).contiguous()
    N, T, D = trajs_tx.shape
    print(f"[build_trajectories] post-transform shape=[{N}, {T}, {D}] "
          f"({trajs_tx.element_size() * trajs_tx.numel() / 1e6:.1f} MB)")

    clip_keys = [(int(c["video_number"]), str(c["node_uid"])) for c in bundle.clips]

    payload = {
        "trajectories": trajs_tx,
        "lengths": lens_tx,
        "design": dtw_design,
        "clip_keys": clip_keys,
        "T_max": T,
        "D": D,
        "num_videos": num_videos,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, output_path)
    size_mb = output_path.stat().st_size / 1e6
    print(f"[build_trajectories] saved {output_path} ({size_mb:.1f} MB)")

    # Emit clip_index_{design}.json next to the artifact. Same schema as the
    # file the legacy offline builder used to write — VideoClipDataset reads this to
    # map clip_idx -> (video_number, node_uid). Sourced from `bundle.clips`
    # which has been reordered above to match the on-disk row order.
    clip_index_path = output_path.parent / f"clip_index_{dtw_design}.json"
    clip_index_out = {
        i: {
            "video_number": int(c["video_number"]),
            "node_uid": str(c["node_uid"]),
        }
        for i, c in enumerate(bundle.clips)
    }
    with open(clip_index_path, "w") as f:
        json.dump(clip_index_out, f)
    print(f"[build_trajectories] saved {clip_index_path}")
    print(f"[build_trajectories] total {time.time() - t0:.1f}s")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data-dir", type=Path,
                   default=SCRIPT_DIR.parents[1] / "outputs" / "training_data")
    p.add_argument("--design", type=str, required=True,
                   help="DTW design key (see dtw_cknna.DTW_DESIGNS)")
    p.add_argument("--output", type=Path, required=True,
                   help="Output .pt path (full path including filename)")
    p.add_argument("-n", "--num-videos", type=int, default=-1,
                   help="Cap on number of videos to include "
                        "(-1 = all videos in the keypoints cache; default)")
    p.add_argument("--max-t", type=int, default=42)
    p.add_argument("--max-clips", type=int, default=-1,
                   help="Cap total clips after bundle load (-1 = no cap). "
                        "Seeded random subsample.")
    p.add_argument("--clip-subsample-seed", type=int, default=42)
    p.add_argument("--depth-grounded", action="store_true",
                   help="Use depth_grounded_clip_keypoints.pt as clip source")
    p.add_argument("--allow-missing-jepa-features", action="store_true",
                   help="Retain clips without JEPA features (trajectory input mode)")
    args = p.parse_args()

    # `-1` means "no cap"; pass through as None so the cache slice `[:None]`
    # returns everything.
    num_videos = None if args.num_videos < 0 else args.num_videos
    max_clips = None if args.max_clips < 0 else args.max_clips
    build_trajectories(
        data_dir=args.data_dir,
        dtw_design=args.design,
        output_path=args.output,
        num_videos=num_videos,
        max_t=args.max_t,
        max_clips=max_clips,
        clip_subsample_seed=args.clip_subsample_seed,
        use_depth_grounded=args.depth_grounded,
        require_jepa_features=not args.allow_missing_jepa_features,
    )


if __name__ == "__main__":
    main()
