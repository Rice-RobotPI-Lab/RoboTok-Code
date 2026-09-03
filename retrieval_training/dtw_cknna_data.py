"""
Data loading for DTW-CKNNA experiments.

Handles local-cache / DB I/O, encoder feature loading (.pt files), hand
keypoint extraction, trajectory building, and the DataBundle container.

Each encoder saves per-video .pt files at FEATURES_DIR/<encoder>/<video_number>.pt
containing dict[node_uid -> Tensor[num_tokens, embed_dim]] in float16.

Hand keypoints are read from KEYPOINTS_CACHE by default (built via
`build_dataset/1get_clip_keypoints.py`). If the cache file is missing the
loader falls back to querying the private clip database (`db` module, not
distributed with this repo — see README).

Usage:
    from dtw_cknna_data import load_data_bundle, load_encoder_feats_into

    bundle = load_data_bundle(num_videos=100)
    load_encoder_feats_into(bundle, "vjepa2", pool="mean")
"""

import os
import numpy as np
import torch
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

# ---------------------------------------------------------------------------
# Paths and environment
# ---------------------------------------------------------------------------
# Paths are repo-root-relative by default (ABMR_PROJECT_ROOT overrides the
# root, matching config.py); ENCODER_FEATURES_DIR / CLIP_KEYPOINTS_CACHE
# override the individual locations.
_REPO_ROOT = Path(
    os.environ.get("ABMR_PROJECT_ROOT", Path(__file__).resolve().parents[1])
).expanduser()

FEATURES_DIR = Path(
    os.environ.get("ENCODER_FEATURES_DIR", _REPO_ROOT / "outputs" / "encoder_features")
).expanduser()
KEYPOINTS_CACHE = Path(
    os.environ.get(
        "CLIP_KEYPOINTS_CACHE",
        _REPO_ROOT / "outputs" / "training_data" / "clip_keypoints.pt",
    )
).expanduser()


def _lazy_db_query(sql, params=None):
    # `db` is the private clip-database module, not distributed with this
    # repo (see README). The KEYPOINTS_CACHE .pt file is the supported
    # data source for public checkouts.
    from db import query
    return query(sql, params)

def _lazy_db_query_clips(sql, params=None):
    from db import query_clips  # private module — see note above
    return query_clips(sql, params)


# ---------------------------------------------------------------------------
# DataBundle: holds loaded data for reuse across experiments
# ---------------------------------------------------------------------------

@dataclass
class DataBundle:
    """Container for loaded clips + trajectories. Load once, reuse across experiments.

    Encoder features are cached per (encoder_name, pool) key via load_encoder_feats_into().
    """
    clips:          list           # DB rows (trajectory-valid only)
    trajs:          torch.Tensor   # [N, T, 42]
    actual_lengths: torch.Tensor   # [N]
    N:              int
    T:              int
    _feats_cache:   dict = field(default_factory=dict, repr=False)


# ---------------------------------------------------------------------------
# Fetch clips for the first N videos
# ---------------------------------------------------------------------------

def get_clips_for_n_videos(num_videos, use_cache=True):
    """Fetch clip rows for the first `num_videos` videos.

    By default reads from the local .pt cache at KEYPOINTS_CACHE (built via
    `build_dataset/1get_clip_keypoints.py`). If the cache file is missing or
    `use_cache=False`, falls back to querying DB.

    Args:
        num_videos: number of distinct video_numbers (ascending) to include.
        use_cache:  if True (default) and the cache file exists, read from it.

    Returns:
        clips: list of row dicts. Cache rows contain video_number, node_number,
               node_uid, keypoints_per_frame, left_hand_presence,
               right_hand_presence. DB rows additionally contain
               video_uid, start_sec, end_sec.
        video_numbers: sorted list of video numbers included
    """
    if use_cache and KEYPOINTS_CACHE.exists():
        return _get_clips_from_cache(num_videos)
    if use_cache:
        print(f"  (cache not found at {KEYPOINTS_CACHE}, falling back to DB)")
    return _get_clips_from_db(num_videos)


def _get_clips_from_cache(num_videos):
    """Load all cached rows, then filter to the first `num_videos` video_numbers."""
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


def _get_clips_from_db(num_videos):
    """DB-backed fetch path. Used as a fallback when cache is missing."""
    vn_rows = _lazy_db_query(
        "SELECT DISTINCT video_number FROM selected_clips "
        "WHERE keypoints_per_frame IS NOT NULL ORDER BY video_number")
    video_numbers = [r["video_number"] for r in vn_rows][:num_videos]

    clips = []
    for vn in video_numbers:
        rows = _lazy_db_query_clips(
            "SELECT video_number, video_uid, node_number, node_uid, "
            "keypoints_per_frame, start_sec, end_sec "
            "FROM selected_clips WHERE video_number = %s "
            "AND keypoints_per_frame IS NOT NULL ORDER BY node_number",
            (vn,))
        clips.extend(rows)

    return clips, video_numbers


# ---------------------------------------------------------------------------
# Load encoder features from .pt files
# ---------------------------------------------------------------------------

def category_of_clip(clip):
    """Best-effort action category key for a DB clip row.

    Tries the common taxonomy fields (`category`, `action_category`,
    `label`, `super_category`); if none are present, falls back to the
    `node_uid` prefix before the first colon (the upstream annotation
    tree keeps category info there). Returns `"unknown"` when nothing
    can be inferred.
    """
    for k in ("category", "action_category", "label", "super_category"):
        if k in clip and clip[k] not in (None, ""):
            return str(clip[k])
    nu = clip.get("node_uid", "") or ""
    return nu.split(":")[0] if ":" in nu else "unknown"


def load_encoder_features(clips, encoder_name, pool="mean"):
    """Load encoder features for a list of clips.

    Args:
        clips: list of row dicts (from get_clips_for_n_videos).
        encoder_name: subdirectory under FEATURES_DIR
            (e.g. "vjepa2", "dinov3", "viclip", "raft", etc.)
        pool: key into POOLING_METHODS (e.g. "mean", "max", "cls"),
            or None to return raw per-token tensors as a list.

    Returns:
        If pool is not None:
            features: Tensor [N, D] float32.
        If pool is None:
            features: list of Tensor, each [num_tokens, D] float32
                (or [D] for already-pooled encoders like viclip).
        valid_indices: list[int] — positions in `clips` that had features.
    """
    pool_fn = POOLING_METHODS[pool] if pool is not None else None

    enc_dir = FEATURES_DIR / encoder_name
    if not enc_dir.exists():
        raise FileNotFoundError(f"No encoder directory at {enc_dir}")

    # Group clips by video_number so we load each .pt once
    video_clips = defaultdict(list)
    for i, clip in enumerate(clips):
        video_clips[clip["video_number"]].append((i, clip))

    features = []
    valid_indices = []
    # Non-finite bookkeeping. Some encoders (notably hunyuan_vae) are saved
    # in float16 by get_encoder_features.py; when the VAE runs in fp16 the
    # internal activations sometimes overflow, leaving ±inf in the .pt
    # files. We drop bad *rows* before pooling (so a single bad token
    # can't poison the clip mean) and drop the whole clip if every row is
    # non-finite.
    n_bad_rows = 0
    n_partial_clips = 0   # clips where some (but not all) rows were dropped
    n_dropped_clips = 0   # clips fully unsalvageable
    # Diagnostic: clips silently skipped because the .pt cache has no row
    # for them (either missing file or missing node_uid entry). Surfacing
    # these lets us spot-check which clips are absent from the feature
    # cache across encoders.
    missing_pt_clips = []       # list[(video_number, node_uid)]
    missing_node_clips = []     # list[(video_number, node_uid)]

    for vn in sorted(video_clips.keys()):
        pt_path = enc_dir / f"{vn}.pt"
        if not pt_path.exists():
            for _idx, _clip in video_clips[vn]:
                missing_pt_clips.append((vn, _clip["node_uid"]))
            continue

        video_feats = torch.load(pt_path, weights_only=True, map_location="cpu")

        for idx, clip in video_clips[vn]:
            node_uid = clip["node_uid"]
            if node_uid not in video_feats:
                missing_node_clips.append((vn, node_uid))
                continue

            tokens = video_feats[node_uid].float()  # [num_tokens, D] or [D]

            if tokens.dim() > 1:
                finite_rows = torch.isfinite(tokens).all(dim=1)
                n_bad = int((~finite_rows).sum().item())
                if n_bad:
                    n_bad_rows += n_bad
                    if not finite_rows.any():
                        n_dropped_clips += 1
                        continue    # unsalvageable — skip the clip entirely
                    n_partial_clips += 1
                    tokens = tokens[finite_rows].contiguous()
            else:
                if not torch.isfinite(tokens).all():
                    n_dropped_clips += 1
                    continue

            if pool_fn is not None and tokens.dim() > 1:
                tokens = pool_fn(tokens)

            features.append(tokens)
            valid_indices.append(idx)

        del video_feats

    if not features:
        raise ValueError(f"No features found for encoder '{encoder_name}' in {enc_dir}")

    if n_bad_rows or n_dropped_clips:
        print(f"  [warn] {encoder_name}: dropped {n_bad_rows} non-finite tokens "
              f"across {n_partial_clips} partial clips; "
              f"dropped {n_dropped_clips} fully unsalvageable clips "
              f"(likely fp16 overflow in stored .pt files)")

    if missing_pt_clips or missing_node_clips:
        n_missing = len(missing_pt_clips) + len(missing_node_clips)
        preview = (missing_pt_clips + missing_node_clips)[:5]
        print(f"  [warn] {encoder_name}: missing features for {n_missing} clips "
              f"({len(missing_pt_clips)} no .pt, {len(missing_node_clips)} no node_uid). "
              f"First 5: {preview}")

    if pool is not None:
        return torch.stack(features), valid_indices
    return features, valid_indices


# ---------------------------------------------------------------------------
# Feature pooling methods: (tokens: [num_tokens, D]) -> [D]
# Add new pooling functions here and register them in POOLING_METHODS.
# ---------------------------------------------------------------------------

def _pool_mean(tokens):
    """Mean over all tokens — uniform weighting across every time position."""
    return tokens.mean(dim=0)


def _pool_max(tokens):
    """Element-wise max over all tokens — simple non-linear baseline.

    Picks the strongest activation per feature dim. Not temporally
    consistent (a shift in the peak token changes the output) but widely
    used as a sanity baseline. Output dim = D.
    """
    return tokens.max(dim=0).values


def _pool_mean_std(tokens):
    """Concat mean and std over tokens — std captures temporal variation.

    High std implies the clip changes a lot across tokens (motion-heavy);
    low std implies a near-static clip. Output dim = 2D.
    """
    mean = tokens.mean(dim=0)
    if tokens.shape[0] < 2:
        std = torch.zeros_like(mean)
    else:
        std = tokens.std(dim=0)
    return torch.cat([mean, std], dim=-1)


def _pool_first_last_concat(tokens):
    """Concat the first and last token — captures clip endpoints.

    Assumes the encoder's tokens are stored time-major (time is the slowest
    axis), which is true for every encoder in this project except viclip.
    Output dim = 2D.
    """
    return torch.cat([tokens[0], tokens[-1]], dim=-1)


def _pool_temporal_diff_mean(tokens):
    """Concat mean and mean of consecutive token diffs — state + change.

    The second term is a proxy for the average temporal derivative of the
    feature sequence, so the pooled vector encodes both the clip's mean
    activation and how much it drifts. Output dim = 2D.
    """
    if tokens.shape[0] < 2:
        return torch.cat([tokens.mean(dim=0), torch.zeros_like(tokens[0])], dim=-1)
    diffs = tokens[1:] - tokens[:-1]
    return torch.cat([tokens.mean(dim=0), diffs.mean(dim=0)], dim=-1)


def _pool_weighted_end_mean(tokens):
    """Linearly-weighted mean with weights rising from 0 to 1 across tokens.

    Emphasizes tokens near the end of the clip — the "final state" often
    carries the action's outcome (object lifted, cup placed, hand withdrawn).
    Output dim = D.
    """
    n = tokens.shape[0]
    w = torch.linspace(0.0, 1.0, n, device=tokens.device, dtype=tokens.dtype)
    return (w.unsqueeze(-1) * tokens).sum(dim=0) / w.sum()


def _pool_halves_mean(tokens):
    """Mean of first half concatenated with mean of second half.

    A minimal temporal pyramid: 2 segments instead of 1. Captures the
    coarsest possible temporal ordering (early vs late). Output dim = 2D.
    """
    n = tokens.shape[0]
    mid = max(n // 2, 1)
    return torch.cat([tokens[:mid].mean(dim=0), tokens[mid:].mean(dim=0)], dim=-1)


def _pool_pyramid(tokens):
    """3-level temporal pyramid: 1 + 2 + 4 segment means, concatenated.

    Mirrors spatial pyramid pooling (He et al. 2014) adapted to time.
    Level 0: global mean (1 segment × D).
    Level 1: first/second half means (2 segments × D).
    Level 2: quarter means (4 segments × D).
    Total: 7 segments concatenated → 7D output.

    Captures temporal structure at multiple scales simultaneously. Robust
    to small temporal shifts (coarse levels absorb them) while still
    distinguishing clips with different temporal ordering (fine levels
    split them apart).
    """
    n = tokens.shape[0]
    parts = [tokens.mean(dim=0)]  # level 0

    # Level 1: 2 halves
    mid = max(n // 2, 1)
    parts.append(tokens[:mid].mean(dim=0))
    parts.append(tokens[mid:].mean(dim=0))

    # Level 2: 4 quarters (use np.array_split-style boundaries so every
    # segment is non-empty even if n < 4)
    bounds = [round(i * n / 4) for i in range(5)]
    for i in range(4):
        lo, hi = bounds[i], max(bounds[i + 1], bounds[i] + 1)
        parts.append(tokens[lo:hi].mean(dim=0))

    return torch.cat(parts, dim=-1)  # 7D output


def _pool_temporal_std(tokens):
    """Per-dim standard deviation over time — captures feature variability.

    Returns how much each feature dimension varied over the clip, ignoring
    the mean level. Useful when the *dynamics* of a feature (high vs low
    variance) carry more signal than the absolute level.
    """
    return tokens.std(dim=0)


def _pool_linear_slope(tokens):
    """Per-dim linear-regression slope over time — monotonic trend.

    Fits a scalar linear function of frame index to each feature dim.
    Captures whether a feature is rising, falling, or flat over the clip.
    Invariant to the absolute level (intercept is discarded).
    """
    n = tokens.shape[0]
    if n == 1:
        return torch.zeros(tokens.shape[1], dtype=tokens.dtype, device=tokens.device)
    t = torch.arange(n, dtype=tokens.dtype, device=tokens.device)
    t = t - t.mean()
    denom = (t * t).sum().clamp(min=1e-8)
    return (t[:, None] * tokens).sum(dim=0) / denom


def _pool_velocity_mean(tokens):
    """Mean of frame-to-frame feature velocity — average motion in feature space.

    Computes `tokens[t+1] - tokens[t]` for each consecutive pair, then
    takes the mean over time. Captures the average direction of change in
    the feature space over the clip, ignoring static levels.
    """
    if tokens.shape[0] < 2:
        return torch.zeros(tokens.shape[1], dtype=tokens.dtype, device=tokens.device)
    return (tokens[1:] - tokens[:-1]).mean(dim=0)


POOLING_METHODS = {
    "mean":               _pool_mean,
    "max":                _pool_max,
    "mean_std":           _pool_mean_std,
    "first_last_concat":  _pool_first_last_concat,
    "temporal_diff_mean": _pool_temporal_diff_mean,
    "weighted_end_mean":  _pool_weighted_end_mean,
    "halves_mean":        _pool_halves_mean,
    "pyramid":            _pool_pyramid,
    # Temporal dynamics pools
    "temporal_std":       _pool_temporal_std,
    "linear_slope":       _pool_linear_slope,
    "velocity_mean":      _pool_velocity_mean,
}


# ---------------------------------------------------------------------------
# Extract hand trajectories from keypoints_per_frame
# ---------------------------------------------------------------------------

def _extract_hand_keypoints_generic(clips, kpts_key):
    """Shared implementation for 2D and 3D hand-keypoint extraction.

    Pulls `hands[0][kpts_key]` per frame per hand; returns a list of
    per-clip dicts and the indices of clips that had any keypoints.
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
                if hands and kpts_key in hands[0]:
                    frame_hands[hand_id] = np.asarray(hands[0][kpts_key])
                else:
                    frame_hands[hand_id] = None
            clip_kpts[fk] = frame_hands

        keypoints.append(clip_kpts)
        valid_indices.append(i)

    if not keypoints:
        raise ValueError("No clips with keypoints found")

    return keypoints, valid_indices


def extract_hand_keypoints(clips):
    """Extract raw 3D hand keypoints from clips.

    For each clip, extracts the kpts_3d [21, 3] array per frame per hand,
    with no transformations applied. Returns the data as-is from DB.

    Args:
        clips: list of row dicts (from get_clips_for_n_videos).

    Returns:
        keypoints: list of dicts, one per clip. Each dict maps
            frame_key (str) -> {"L": array [21, 3] or None,
                                "R": array [21, 3] or None}
            Frames where a hand was not detected have None for that hand.
        valid_indices: list[int] — positions in `clips` that had keypoints.
    """
    return _extract_hand_keypoints_generic(clips, "kpts_3d")


def extract_hand_keypoints_2d(clips):
    """Extract raw 2D hand keypoints from clips.

    Mirrors `extract_hand_keypoints` but reads `kpts_2d` (normalized to
    [0, 1] by image dimensions). Returns [21, 2] arrays per hand per
    frame.

    Used by the WiLoR 2D-vs-3D self-consistency diagnostic: if DTW on the
    2D trajectory produces nearly the same neighbor structure as DTW on
    the 3D trajectory, WiLoR's 3D lift is self-consistent and the flat
    representation is the bottleneck; if not, 3D estimation adds noise.
    """
    return _extract_hand_keypoints_generic(clips, "kpts_2d")


# ---------------------------------------------------------------------------
# Combined loader
# ---------------------------------------------------------------------------

def load_features_and_keypoints(num_videos, encoder_name, pool="mean", use_cache=True):
    """Load encoder features and raw 3D hand keypoints for the first N videos.

    Only returns clips that have BOTH valid encoder features AND keypoints.

    Args:
        num_videos: how many videos (by ascending video_number) to include.
        encoder_name: encoder subdirectory name (e.g. "vjepa2").
        pool: key into POOLING_METHODS (e.g. "mean", "max"), or None for raw.
        use_cache: if True (default), read clips from the local .pt cache;
            otherwise query DB directly.

    Returns:
        feats:     Tensor [N, D] (pool != None) or list of Tensors (pool=None).
        keypoints: list of dicts — per-clip, frame_key -> {"L": [21,3] or None,
                   "R": [21,3] or None}.
        clip_ids:  list of (video_number, node_uid) tuples.
    """
    clips, video_numbers = get_clips_for_n_videos(num_videos, use_cache=use_cache)
    feats, feat_idx = load_encoder_features(clips, encoder_name, pool=pool)
    kpts, kpts_idx = extract_hand_keypoints(clips)

    # Intersect indices
    common = sorted(set(feat_idx) & set(kpts_idx))
    feat_pos = {idx: pos for pos, idx in enumerate(feat_idx)}
    kpts_pos = {idx: pos for pos, idx in enumerate(kpts_idx)}

    if pool is not None:
        feats_out = torch.stack([feats[feat_pos[i]] for i in common])
    else:
        feats_out = [feats[feat_pos[i]] for i in common]
    kpts_out = [kpts[kpts_pos[i]] for i in common]
    clip_ids = [(clips[i]["video_number"], clips[i]["node_uid"]) for i in common]

    return feats_out, kpts_out, clip_ids


# ---------------------------------------------------------------------------
# DataBundle loaders
# ---------------------------------------------------------------------------

def load_data_bundle(num_videos, use_cache=True):
    """Load clips + trajectories once. Returns a DataBundle for reuse.

    Args:
        num_videos: how many videos (by ascending video_number) to include.
                    Use a large value (e.g. 99999) for all videos.
        use_cache:  if True (default), read clips from the local .pt cache
                    at KEYPOINTS_CACHE; otherwise query DB directly.

    Returns:
        DataBundle with clips, trajs [N, T, 42], actual_lengths [N].
    """
    clips, _ = get_clips_for_n_videos(num_videos, use_cache=use_cache)
    kpts_list, kpts_valid = extract_hand_keypoints(clips)
    trajs, traj_valid, actual_lengths = keypoints_to_trajectories(kpts_list)
    valid_clips = [clips[kpts_valid[i]] for i in traj_valid]
    N, T, _ = trajs.shape
    print(f"DataBundle: {N} clips, T={T}, from {num_videos} videos")
    return DataBundle(clips=valid_clips, trajs=trajs, actual_lengths=actual_lengths, N=N, T=T)


def load_data_bundle_21j(num_videos, use_cache=True):
    """Load clips + full 21-joint trajectories. Returns a DataBundle.

    Same as load_data_bundle but uses all 21 joints per hand (126D)
    instead of the 7-joint subset (42D).
    """
    clips, _ = get_clips_for_n_videos(num_videos, use_cache=use_cache)
    kpts_list, kpts_valid = extract_hand_keypoints(clips)
    trajs, traj_valid, actual_lengths = keypoints_to_trajectories_21j(kpts_list)
    valid_clips = [clips[kpts_valid[i]] for i in traj_valid]
    N, T, _ = trajs.shape
    print(f"DataBundle (21j): {N} clips, T={T}, D={trajs.shape[-1]}, from {num_videos} videos")
    return DataBundle(clips=valid_clips, trajs=trajs, actual_lengths=actual_lengths, N=N, T=T)


def smooth_bundle_savgol(bundle, window_length=5, polyorder=2):
    """Return a new DataBundle with Savitzky–Golay–smoothed trajectories.

    Smooths each clip's trajectory along the time axis per dimension,
    respecting `actual_lengths` so padding frames aren't smoothed into
    the signal region. The padding region is left as a repeated-last-
    valid-frame (same as the original padding convention).

    Args:
        bundle: input DataBundle.
        window_length: Savitzky–Golay window (must be odd, >= polyorder+2).
        polyorder: polynomial order for the local fit.

    Returns:
        new DataBundle with smoothed trajectories. Other fields (clips,
        actual_lengths, N, T) are unchanged. The `_feats_cache` dict is
        **shared by reference** with the input bundle so encoder features
        don't need to be re-loaded.
    """
    from scipy.signal import savgol_filter

    trajs = bundle.trajs.numpy()  # [N, T, D]
    lengths = bundle.actual_lengths.numpy()
    smoothed = trajs.copy()

    for i in range(bundle.N):
        L = int(lengths[i])
        # SavGol needs window_length <= L. Short clips get a smaller window.
        wl = min(window_length, L if L % 2 == 1 else L - 1)
        if wl < polyorder + 2:
            # Too short to smooth meaningfully; leave as-is.
            continue
        # Smooth only the valid region; padding stays as the original
        # repeated-last-frame values.
        smoothed[i, :L] = savgol_filter(trajs[i, :L], wl, polyorder, axis=0)
        # Re-copy the smoothed final valid frame into the padding region
        # so the padding remains consistent with the smoothed boundary.
        if L < bundle.T:
            smoothed[i, L:] = smoothed[i, L - 1]

    new_trajs = torch.from_numpy(smoothed).float()
    new_bundle = DataBundle(
        clips=bundle.clips,
        trajs=new_trajs,
        actual_lengths=bundle.actual_lengths,
        N=bundle.N,
        T=bundle.T,
        _feats_cache=bundle._feats_cache,   # shared by reference
    )
    return new_bundle


def load_data_bundle_2d(num_videos, use_cache=True):
    """Load clips + 2D trajectories. Returns a DataBundle for reuse.

    Mirrors `load_data_bundle` but produces 28D trajectories from
    `kpts_2d` instead of 42D trajectories from `kpts_3d`. Used by the
    WiLoR 2D-vs-3D self-consistency diagnostic.

    The resulting bundle is interchangeable with a 3D bundle for DTW
    computation — every runner in dtw_cknna.py only cares about
    `bundle.trajs`, `bundle.actual_lengths`, `bundle.clips`, `bundle.N`,
    `bundle.T`.
    """
    clips, _ = get_clips_for_n_videos(num_videos, use_cache=use_cache)
    kpts_list, kpts_valid = extract_hand_keypoints_2d(clips)
    trajs, traj_valid, actual_lengths = keypoints_2d_to_trajectories(kpts_list)
    valid_clips = [clips[kpts_valid[i]] for i in traj_valid]
    N, T, _ = trajs.shape
    print(f"DataBundle (2D): {N} clips, T={T}, D=28, from {num_videos} videos")
    return DataBundle(clips=valid_clips, trajs=trajs, actual_lengths=actual_lengths, N=N, T=T)


def load_encoder_feats_into(bundle, encoder_name, pool="mean"):
    """Load encoder features into a DataBundle's cache.

    Not all clips in the bundle may have features for a given encoder.
    Returns the subset that does, with an index mapping.

    Args:
        bundle: DataBundle from load_data_bundle().
        encoder_name: encoder subdirectory name (e.g. "vjepa2").
        pool: key into POOLING_METHODS (default "mean").

    Returns:
        (feats, valid_indices) where feats is [M, D] and valid_indices
        maps into bundle.clips (length M ≤ N).
    """
    key = (encoder_name, pool)
    if key in bundle._feats_cache:
        return bundle._feats_cache[key]

    feats, feat_valid = load_encoder_features(bundle.clips, encoder_name, pool=pool)
    bundle._feats_cache[key] = (feats, feat_valid)
    print(f"  Loaded {encoder_name}/{pool}: {feats.shape[0]} clips, D={feats.shape[1]}")
    return feats, feat_valid


# ---------------------------------------------------------------------------
# Convert keypoint dicts → fixed-length trajectory tensors
# ---------------------------------------------------------------------------

# 7 joints per hand: wrist, MCP3, 5 fingertips
HAND_JOINT_INDICES = [0, 9, 4, 8, 12, 16, 20]
HAND_TRAJ_DIM = len(HAND_JOINT_INDICES) * 3 * 2  # 42 (3D)
HAND_TRAJ_DIM_2D = len(HAND_JOINT_INDICES) * 2 * 2  # 28 (2D)


def _keypoints_to_trajectories_generic(keypoints_list, dim_per_kpt, min_frames=2,
                                       joint_idx=None):
    """Shared fill+pad logic for 2D and 3D keypoint_list → trajectory tensor.

    The per-hand fill rule (backfill before first detection, forward-fill
    after last, nearest-neighbor in the interior) is independent of
    coordinate dimensionality — only `hand_dim` changes.

    Args:
        keypoints_list: list of dicts, each mapping frame_key (str) ->
            {"L": ndarray [21, dim_per_kpt] or None,
             "R": ndarray [21, dim_per_kpt] or None}.
        dim_per_kpt: 3 for 3D keypoints, 2 for 2D.
        min_frames: minimum keypoint frames to include a clip (default 2).
        joint_idx: which joint indices to keep (default HAND_JOINT_INDICES).
                   Pass None for the default 7-joint subset, or
                   list(range(21)) for all 21 joints.

    Returns:
        trajs:          Tensor [M, T, 2 * n_joints * dim_per_kpt] float32
                        (T = max frames across clips)
        valid_indices:  list[int] — positions in keypoints_list that survived
        actual_lengths: Tensor [M] int — pre-padding frame count per clip
    """
    if joint_idx is None:
        joint_idx = HAND_JOINT_INDICES
    n_joints = len(joint_idx)
    hand_dim = n_joints * dim_per_kpt  # 21 for 3D, 14 for 2D
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


def keypoints_to_trajectories(keypoints_list, min_frames=2):
    """Convert per-clip 3D keypoint dicts to fixed-length [M, T, 42] tensors.

    See `_keypoints_to_trajectories_generic` for the fill/pad logic. Returns
    42D trajectories: 7 joints per hand × 3 coords × 2 hands.
    """
    return _keypoints_to_trajectories_generic(keypoints_list, 3, min_frames=min_frames)


def keypoints_to_trajectories_21j(keypoints_list, min_frames=2):
    """Convert per-clip 3D keypoint dicts to fixed-length [M, T, 126] tensors.

    All 21 joints per hand × 3 coords × 2 hands = 126D.
    """
    return _keypoints_to_trajectories_generic(keypoints_list, 3, min_frames=min_frames,
                                              joint_idx=list(range(21)))


def keypoints_2d_to_trajectories(keypoints_list, min_frames=2):
    """Convert per-clip 2D keypoint dicts to fixed-length [M, T, 28] tensors.

    See `_keypoints_to_trajectories_generic` for the fill/pad logic. Returns
    28D trajectories: 7 joints per hand × 2 coords × 2 hands. Used by the
    WiLoR 2D-vs-3D self-consistency diagnostic.
    """
    return _keypoints_to_trajectories_generic(keypoints_list, 2, min_frames=min_frames)
