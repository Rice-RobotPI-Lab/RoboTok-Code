"""
DTW computation, CKNNA evaluation, design-grid utilities, and diagnostics.

Imports data loading from dtw_cknna_data.py. All GPU-accelerated DTW kernels,
cost functions, normalization strategies, HSIC, and CKNNA live here.

Usage:
    from dtw_cknna_data import load_data_bundle, load_encoder_feats_into
    from dtw_cknna import run_experiment_grid, DTW_METHODS, neighbor_overlap

    bundle = load_data_bundle(num_videos=60)
    results = run_experiment_grid(bundle, dtw_methods=["euclidean_sym2", "cosine_sym2"],
                                  encoders=["vjepa2"], return_matrices=True)
"""

import time
import numpy as np
import torch
import torch.nn.functional as F
from itertools import product

from dtw_cknna_data import (
    FEATURES_DIR, DataBundle, POOLING_METHODS,
    get_clips_for_n_videos, load_encoder_features,
    extract_hand_keypoints, load_features_and_keypoints,
    load_data_bundle, load_encoder_feats_into,
    keypoints_to_trajectories,
)


# ---------------------------------------------------------------------------
# DTW + Asymmetric Platonic CKNNA
# ---------------------------------------------------------------------------

TOPK = 10


# ============================================================================
# DTW: Euclidean cost matrix builder (for hand keypoint trajectories)
# ============================================================================

def build_euclidean_cost_matrices_gpu(vec_i, all_vecs):
    """Build [N, T_i, T_j] L2 cost matrices on GPU.

    Args:
        vec_i:    [T, D]    — frames of query trajectory i
        all_vecs: [N, T, D] — frames of all N trajectories

    Returns:
        [N, T, T] float32 — cost[n, t1, t2] = ||vec_i[t1] - all_vecs[n, t2]||_2
    """
    diff = vec_i[None, :, None, :] - all_vecs[:, None, :, :]
    return torch.sqrt((diff ** 2).sum(dim=3))


# ============================================================================
# DTW: GPU symmetric2 kernel
# ============================================================================

def make_sym2_cuda_kernel():
    """Create numba CUDA kernel for hard DTW with symmetric2 step pattern.

    Motivation: DTW finds the minimum-cost monotone alignment between two
    sequences. The "symmetric2" step pattern is the standard one (used by
    dtw-python's default). From any cell (i,j), you can step to:
      - (i+1, j+1): diagonal — both sequences advance — costs 2×D (favored)
      - (i+1, j):   vertical — sequence 1 advances, seq 2 repeats — costs 1×D
      - (i, j+1):   horizontal — seq 2 advances, seq 1 repeats — costs 1×D
    The diagonal step costs 2×D to avoid bias: without it, the diagonal path
    through a TxT matrix has T steps while vert/horiz has 2T, unfairly penalizing
    the straight-through path.

    Why GPU: Pairwise DTW is O(N² · T²). For N=6000, T=40, that's 5.76×10¹⁰
    operations. The anti-diagonal trick parallelizes across T threads per pair:
    all cells on the same anti-diagonal are independent (they only depend on
    previous anti-diagonals), so T cells compute simultaneously.

    Recurrence: R[i,j] = D[i,j] + min(R[i-1,j-1] + D[i,j], R[i-1,j], R[i,j-1])
    """
    from numba import cuda

    @cuda.jit
    def harddtw_sym2(D, bandwidth, max_i, max_j, n_passes, R):
        # Each CUDA block processes one trajectory pair (batch element)
        b = cuda.blockIdx.x
        # Each thread handles one cell on the current anti-diagonal
        tid = cuda.threadIdx.x

        # Sweep through 2T-1 anti-diagonals (p=0 is top-left corner,
        # p=2T-2 is bottom-right corner)
        for p in range(n_passes):
            # Map thread id to (i, j) on anti-diagonal p.
            # R is 1-indexed (padded with inf at row 0 and col 0).
            # On anti-diagonal p, all cells satisfy i + j = p + 2.
            i = tid + 1
            j = p - tid + 1

            # Bounds check: only process valid cells within the TxT grid
            if 1 <= i <= max_i and 1 <= j <= max_j:
                # Sakoe-Chiba bandwidth constraint (disabled when bandwidth=0)
                if not (bandwidth > 0 and abs(i - j) > bandwidth):
                    # D is 0-indexed, R is 1-indexed, so D[i-1, j-1] = R's (i,j)
                    d = D[b, i - 1, j - 1]

                    # Three candidate predecessors:
                    diag = R[b, i - 1, j - 1] + d  # diagonal: accumulated + d (total 2×d for this cell)
                    vert = R[b, i - 1, j]           # vertical: seq 1 advances alone (1×d)
                    horz = R[b, i, j - 1]           # horizontal: seq 2 advances alone (1×d)

                    # Take minimum predecessor
                    best = diag if diag < vert else vert
                    best = best if best < horz else horz

                    # Store: local cost d + best predecessor
                    # This means diagonal contributes d + (prev + d) = 2d total,
                    # while vert/horz contribute d + prev = 1d total.
                    R[b, i, j] = d + best

            # CRITICAL: all threads must finish this anti-diagonal before
            # any thread starts the next one (data dependency)
            cuda.syncthreads()

    return harddtw_sym2


# ============================================================================
# DTW: cost matrix builders
# Signature: (vec_i: [T, D], all_vecs: [N, T, D]) -> [N, T, T]
# Add new cost functions here and register them in DTW_METHODS.
# ============================================================================

def _cost_euclidean(vec_i, all_vecs):
    """L2 (Euclidean) distance."""
    diff = vec_i[None, :, None, :] - all_vecs[:, None, :, :]
    return torch.sqrt((diff ** 2).sum(dim=3))


def _cost_cosine(vec_i, all_vecs):
    """Cosine distance = 1 - cosine_similarity, in [0, 2]."""
    vi = F.normalize(vec_i, p=2, dim=-1)
    av = F.normalize(all_vecs, p=2, dim=-1)
    cos_sim = (vi[None, :, None, :] * av[:, None, :, :]).sum(dim=-1)
    return 1.0 - cos_sim


def _cost_manhattan(vec_i, all_vecs):
    """Manhattan (L1) distance."""
    diff = vec_i[None, :, None, :] - all_vecs[:, None, :, :]
    return diff.abs().sum(dim=3)


# ============================================================================
# DTW: normalization strategies
# Signature: (distances: [N], lengths: [N], query_len: scalar) -> [N]
# Add new normalization functions here and register them in DTW_NORMS.
# ============================================================================

def _norm_none(distances, _lengths, _query_len):
    """No normalization — raw DTW distance."""
    return distances


def _norm_mean_length(distances, lengths, query_len):
    """Divide by mean of query and target actual lengths."""
    return distances / ((query_len + lengths) / 2.0)


# ============================================================================
# DTW method registry: string key -> {cost_fn, kernel_fn, label}
# ============================================================================

DTW_METHODS = {
    "euclidean_sym2": {
        "cost_fn":   _cost_euclidean,
        "kernel_fn": make_sym2_cuda_kernel,
        "label":     "Euclidean L2 + symmetric2",
    },
    "cosine_sym2": {
        "cost_fn":   _cost_cosine,
        "kernel_fn": make_sym2_cuda_kernel,
        "label":     "Cosine distance + symmetric2",
    },
    "manhattan_sym2": {
        "cost_fn":   _cost_manhattan,
        "kernel_fn": make_sym2_cuda_kernel,
        "label":     "Manhattan L1 + symmetric2",
    },
}

DTW_NORMS = {
    "none":        _norm_none,
    "mean_length": _norm_mean_length,
}


# ============================================================================
# Trajectory transforms for DTW design variants
# Signature: (trajs [N, T, 42], lens [N]) -> (trajs' [N, T', D'], lens' [N])
#
# The base trajectory is 42D: 7 hand joints × 3 coords × 2 hands, with
# dims [:21] = Left hand, [21:] = Right hand (see HAND_JOINT_INDICES in
# dtw_cknna_data.py). Transforms either subset hands or take temporal
# derivatives; length is decremented appropriately so mean_length
# normalization uses the correct effective clip length.
# ============================================================================

def _tx_full(trajs, lens):
    return trajs, lens


def _tx_left_pos(trajs, lens):
    return trajs[..., :21].contiguous(), lens


def _tx_right_pos(trajs, lens):
    return trajs[..., 21:].contiguous(), lens


def _tx_both_vel(trajs, lens):
    v = trajs[:, 1:] - trajs[:, :-1]
    return v.contiguous(), (lens - 1).clamp(min=1)


def _tx_left_vel(trajs, lens):
    v = trajs[:, 1:, :21] - trajs[:, :-1, :21]
    return v.contiguous(), (lens - 1).clamp(min=1)


def _tx_right_vel(trajs, lens):
    v = trajs[:, 1:, 21:] - trajs[:, :-1, 21:]
    return v.contiguous(), (lens - 1).clamp(min=1)


def _tx_both_acc(trajs, lens):
    v = trajs[:, 1:] - trajs[:, :-1]
    a = v[:, 1:] - v[:, :-1]
    return a.contiguous(), (lens - 2).clamp(min=1)


def _tx_left_acc(trajs, lens):
    v = trajs[:, 1:, :21] - trajs[:, :-1, :21]
    a = v[:, 1:] - v[:, :-1]
    return a.contiguous(), (lens - 2).clamp(min=1)


def _tx_right_acc(trajs, lens):
    v = trajs[:, 1:, 21:] - trajs[:, :-1, 21:]
    a = v[:, 1:] - v[:, :-1]
    return a.contiguous(), (lens - 2).clamp(min=1)


# --- Motion-invariant transforms ---------------------------------------
# These make DTW comparisons invariant to various nuisance parameters:
# absolute position, clip-level scale, camera framing, motion speed.

def _tx_centered(trajs, lens):
    """Subtract per-clip temporal mean — position-invariant.

    Two clips with identical motion but different framing (e.g., hands in
    the top-left vs. bottom-right of the frame) will now have the same
    trajectory. Scale is preserved.
    """
    mean = trajs.mean(dim=1, keepdim=True)  # [N, 1, 42]
    return (trajs - mean).contiguous(), lens


def _tx_standardized(trajs, lens):
    """Per-clip z-score — position + scale invariant.

    Each clip is normalized to zero mean and unit std (per dimension).
    A big sweeping motion and a small twitch of the same *shape* now
    compare as similar. Trades off magnitude information for shape
    sensitivity.
    """
    mean = trajs.mean(dim=1, keepdim=True)
    std = trajs.std(dim=1, keepdim=True).clamp(min=1e-6)
    return ((trajs - mean) / std).contiguous(), lens


def _tx_range_norm(trajs, lens):
    """Per-clip min-max normalization to [0, 1] — shape-only.

    Even more aggressive than z-score: each dimension is rescaled to its
    own [0, 1] range. Strips all magnitude and offset info; only the
    normalized shape of the trajectory survives.
    """
    lo = trajs.amin(dim=1, keepdim=True)
    hi = trajs.amax(dim=1, keepdim=True)
    scale = (hi - lo).clamp(min=1e-6)
    return ((trajs - lo) / scale).contiguous(), lens


def _tx_wrist_centered(trajs, lens):
    """Subtract each hand's wrist from its own joints — intra-hand pose only.

    For each hand, the wrist is joint 0. Subtracting the wrist from the other
    6 joints within that hand yields a representation invariant to each hand's
    global position but still encoding finger configuration. Useful for
    comparing hand shapes across different table/scene positions.

    Dim-agnostic: accepts 28D (2 hands × 7 joints × 2 xy) or 42D (2×7×3 xyz).
    Layout: reshape to [N, T, 2, 7, C] where C = D // 14.
    """
    N, T, D = trajs.shape
    C = D // 14
    x = trajs.reshape(N, T, 2, 7, C)
    x = x - x[:, :, :, 0:1, :]  # subtract wrist from each hand's joints
    return x.reshape(N, T, D).contiguous(), lens


def _tx_both_vel_unit(trajs, lens):
    """Per-clip-normalized velocity — motion-shape invariant to speed.

    Take the frame-diff velocity, then divide each clip's entire velocity
    sequence by its own max frame-wise L2 norm. Fast and slow versions of
    the same motion now align. T→T-1.
    """
    v = trajs[:, 1:] - trajs[:, :-1]  # [N, T-1, 42]
    # Per-clip peak frame-wise magnitude
    frame_norm = v.norm(dim=-1)                       # [N, T-1]
    peak = frame_norm.amax(dim=1, keepdim=True).clamp(min=1e-6)  # [N, 1]
    v = v / peak.unsqueeze(-1)
    return v.contiguous(), (lens - 1).clamp(min=1)


# --- Depth-axis transforms ---------------------------------------------
# Flat 42D layout = 2 hands × 7 joints × 3 coords (x, y, z). z sits at
# every 3rd index within each hand block (0..20 = Left, 21..41 = Right).
# WiLoR's 3D keypoints are in *camera space* (X right, Y up, Z depth into
# scene), and the depth component comes from pred_cam_t_full — inferred
# from focal length and bbox size under a weak-perspective assumption, so
# it's far noisier than the (x, y) which come from direct 2D reprojection.
_Z_DIMS = [3 * j + 2 for j in range(7)] + [21 + 3 * j + 2 for j in range(7)]


def _tx_no_z(trajs, lens):
    """Zero out depth — compare using only image-plane (X, Y) coords.

    Why: WiLoR's per-frame depth is dominated by monocular ambiguity and
    often jitters on the order of the full hand extent frame-to-frame.
    Zeroing Z keeps only the trusted (x, y) image-plane motion. Loses any
    toward/away-from-camera hand motion but removes a large noise source.
    """
    out = trajs.clone()
    out[..., _Z_DIMS] = 0.0
    return out.contiguous(), lens


def _tx_z_standardized(trajs, lens):
    """Per-clip z-score depth only — knock Z's scale down to match X/Y.

    Why: Z's raw magnitude (meters in camera space) can be much larger than
    the X/Y channels, so Euclidean DTW cost gets dominated by noisy depth
    differences. Standardizing only the Z dims to zero-mean / unit-std per
    clip preserves *some* depth signal (relative in/out motion) but stops
    it from drowning out X/Y. X and Y are left untouched.
    """
    out = trajs.clone()
    z = out[..., _Z_DIMS]                                 # [N, T, 14]
    z_mean = z.mean(dim=1, keepdim=True)
    z_std = z.std(dim=1, keepdim=True).clamp(min=1e-6)
    out[..., _Z_DIMS] = (z - z_mean) / z_std
    return out.contiguous(), lens


# --- Wrist-only and hand-split no-z transforms -------------------------
# Wrist-only and hand-split depth-axis transforms.

_WRIST_XYZ_BOTH = [0, 1, 2, 21, 22, 23]         # L-wrist xyz + R-wrist xyz → 6D
_WRIST_XY_BOTH  = [0, 1, 21, 22]                # L-wrist xy  + R-wrist xy  → 4D
_Z_DIMS_SINGLE  = [2, 5, 8, 11, 14, 17, 20]     # z indices within a 21D hand block


def _tx_wrist_only(trajs, lens):
    """Only the wrist joint of each hand — 6D (L-xyz, R-xyz).

    Strips finger joints from the trajectory, leaving the per-hand wrist
    positions in camera space.
    """
    return trajs[..., _WRIST_XYZ_BOTH].contiguous(), lens


def _tx_wrist_only_no_z(trajs, lens):
    """Only the wrist joint of each hand, image plane only — 4D (L-xy, R-xy).

    Coarse hand-position-in-image representation.
    """
    return trajs[..., _WRIST_XY_BOTH].contiguous(), lens


def _tx_no_z_left(trajs, lens):
    """Left hand only, depth zeroed — 21D (z dims forced to 0)."""
    out = trajs[..., :21].clone()
    out[..., _Z_DIMS_SINGLE] = 0.0
    return out.contiguous(), lens


def _tx_no_z_right(trajs, lens):
    """Right hand only, depth zeroed — 21D (z dims forced to 0)."""
    out = trajs[..., 21:].clone()
    out[..., _Z_DIMS_SINGLE] = 0.0
    return out.contiguous(), lens


# --- Scale normalizations ----------------------------------------
# WiLoR 3D keypoints are not metric across videos — the same hand can
# appear at 2× different 3D magnitudes in two clips filmed from different
# cameras/distances, because pred_cam_t_full is inferred from focal length
# and bbox size under a weak-perspective model. Euclidean DTW cost scales
# linearly with that magnitude difference, which is the primary driver of
# the 25× same-video over-representation in DTW top-k neighbors at N=1138.
# Each transform below removes a specific nuisance parameter (global scale,
# per-hand scale, framing position, or absolute coordinates altogether).

def _tx_isotropic_scale(trajs, lens):
    """Divide whole clip by a single per-clip scalar — uniform scale invariance.

    Uses the per-clip std of the flattened trajectory. Single scalar per
    clip preserves relative geometry (X/Y/Z proportions, L-vs-R hand
    distance), only knocking out overall amplitude. This is the isotropic
    cousin of _tx_standardized — that one rescales each dim independently
    (destroying relative scale), this one uses one scalar.
    """
    N = trajs.shape[0]
    # Per-clip scalar: std over (time, dim)
    scale = trajs.reshape(N, -1).std(dim=1, keepdim=True).clamp(min=1e-6)  # [N, 1]
    out = trajs / scale[:, :, None]  # broadcast [N, 1, 1]
    return out.contiguous(), lens


def _tx_centered_isotropic(trajs, lens):
    """Center then divide by per-clip scalar — framing + scale invariant.

    Two-stage: subtract per-clip temporal mean (remove framing), then
    divide by per-clip std of the centered trajectory (remove amplitude).
    Preserves relative geometry, removes both primary leakage sources.
    """
    centered = trajs - trajs.mean(dim=1, keepdim=True)
    N = centered.shape[0]
    scale = centered.reshape(N, -1).std(dim=1, keepdim=True).clamp(min=1e-6)
    out = centered / scale[:, :, None]
    return out.contiguous(), lens


def _tx_bbox_diag_norm(trajs, lens):
    """Divide whole clip by its spatial bbox diagonal — another isotropic scale.

    Alternative estimator to isotropic_scale: uses the spatial extent of the
    trajectory rather than its std. More robust to outlier frames than std.
    Dim-agnostic: reshapes to [N, T, 14, C] where C = D // 14 (2 or 3).
    """
    N, T, D = trajs.shape
    C = D // 14
    x = trajs.reshape(N, T, 14, C)
    lo = x.amin(dim=(1, 2))   # [N, C] — per-clip min per axis
    hi = x.amax(dim=(1, 2))   # [N, C] — per-clip max per axis
    diag = (hi - lo).norm(dim=-1).clamp(min=1e-6)   # [N]
    out = trajs / diag[:, None, None]
    return out.contiguous(), lens


def _tx_path_length_norm(trajs, lens):
    """Velocity divided by per-clip total arc length — speed-invariant motion.

    Total path length = sum over time of per-frame L2 norm of velocity.
    Dividing velocity by this scalar produces a "unit total path" motion
    vector per clip, so slow and fast versions of the same motion align.
    Respects actual_lengths so padding doesn't contribute.
    """
    v = trajs[:, 1:] - trajs[:, :-1]   # [N, T-1, D]
    N, Tm1, D = v.shape
    # Per-frame velocity magnitude, then mask to actual length - 1
    frame_norm = v.norm(dim=-1)        # [N, T-1]
    idx = torch.arange(Tm1, device=v.device)[None, :]
    valid = idx < (lens[:, None].clamp(max=Tm1 + 1) - 1)   # [N, T-1]
    path_len = (frame_norm * valid).sum(dim=1, keepdim=True).clamp(min=1e-6)  # [N, 1]
    out = v / path_len[:, :, None]
    return out.contiguous(), (lens - 1).clamp(min=1)


def _tx_hand_length_norm(trajs, lens):
    """Divide each hand block by its own wrist-to-middle-MCP reference length.

    Per HAND_JOINT_INDICES = [0, 9, 4, 8, 12, 16, 20], joint[0] is the
    wrist and joint[1] is the middle-finger MCP (metacarpophalangeal,
    the base knuckle). The distance between them is a structurally
    stable constant per subject (it's the palm width), so dividing by
    it maps all hands to a canonical "one palm = 1 unit" scale
    regardless of camera focal length or subject distance. Directly
    targets WiLoR's weak-perspective scale confound.

    Reference length is the median over valid frames (robust to bad
    single-frame estimates).

    Dim-agnostic: accepts 28D or 42D bundles — reshape uses C = D // 14.
    """
    N, T, D = trajs.shape
    C = D // 14
    half = D // 2    # per-hand block width (14 for 28D, 21 for 42D)
    x = trajs.reshape(N, T, 2, 7, C)
    # Per-frame per-hand wrist-to-MCP distance [N, T, 2]
    wrist_to_mcp = (x[:, :, :, 1, :] - x[:, :, :, 0, :]).norm(dim=-1)
    # Full-range median — padding frames are copies of the last valid frame
    # so they don't affect the median much.
    ref_len = wrist_to_mcp.median(dim=1).values.clamp(min=1e-6)   # [N, 2]
    scale_left  = ref_len[:, 0:1, None]   # [N, 1, 1]
    scale_right = ref_len[:, 1:2, None]
    out = trajs.clone()
    out[..., :half] = trajs[..., :half] / scale_left
    out[..., half:] = trajs[..., half:] / scale_right
    return out.contiguous(), lens


def _tx_wrist_centered_hand_length(trajs, lens):
    """Wrist-centered + hand-length-normalized — strongest scale-invariance.

    Two-stage: (1) subtract each hand's wrist from its own joints
    (removes framing per hand), (2) divide each hand block by its
    wrist-to-middle-MCP reference length (removes per-subject hand
    size). Result encodes pure intra-hand finger articulation at unit
    scale. Framing, subject size, and camera distance are all gone.
    """
    # Stage 1: wrist-center each hand (dim-agnostic reshape)
    N, T, D = trajs.shape
    C = D // 14
    x = trajs.reshape(N, T, 2, 7, C)
    x = x - x[:, :, :, 0:1, :]   # subtract wrist per hand
    # Stage 2: compute per-hand wrist-to-MCP on centered x (joint[1] - joint[0]
    # is the same vector before/after wrist subtraction).
    ref_len = x[:, :, :, 1, :].norm(dim=-1).median(dim=1).values.clamp(min=1e-6)  # [N, 2]
    # Scale each hand
    x = x / ref_len[:, None, :, None, None]   # broadcast [N, 1, 2, 1, 1]
    return x.reshape(N, T, D).contiguous(), lens


def _tx_wrist_centered_scaled(trajs, lens):
    """Wrist-centered + global isotropic scale — simpler variant of wcHL.

    Same wrist-centering as wrist_centered_hand_length but uses a single
    global per-clip std as the scalar instead of per-hand palm width.
    Loses per-hand scale invariance but is less sensitive to noisy MCP
    estimates on small hands.
    """
    # Wrist-center (dim-agnostic)
    N, T, D = trajs.shape
    C = D // 14
    x = trajs.reshape(N, T, 2, 7, C) - trajs.reshape(N, T, 2, 7, C)[:, :, :, 0:1, :]
    x = x.reshape(N, T, D)
    scale = x.reshape(N, -1).std(dim=1, keepdim=True).clamp(min=1e-6)   # [N, 1]
    out = x / scale[:, :, None]
    return out.contiguous(), lens


def _tx_centroid_relative(trajs, lens):
    """Subtract per-frame centroid of all 14 joints, scale by centroid spread.

    Per frame, compute the mean position of all 14 joints, subtract it,
    then divide by the per-clip std of joint-to-centroid distances. Pure
    shape around the centroid. Removes both framing (subtract centroid)
    and scale (divide by spread). Theoretically similar to wrist_centered
    but using the centroid as the reference instead of each hand's wrist.
    """
    N, T, D = trajs.shape
    C = D // 14
    x = trajs.reshape(N, T, 14, C)
    centroid = x.mean(dim=2, keepdim=True)   # [N, T, 1, C]
    relative = x - centroid                   # [N, T, 14, C]
    # Per-clip scalar from the overall spread
    scale = relative.reshape(N, -1).std(dim=1, keepdim=True).clamp(min=1e-6)   # [N, 1]
    out = relative / scale[:, :, None, None]
    return out.reshape(N, T, D).contiguous(), lens


# --- Shape-only feature transforms --------------------------------
# These replace the raw XYZ coordinate vector with a fundamentally
# different per-frame representation that is invariant by construction
# to certain nuisance parameters.

def _tx_unit_velocity(trajs, lens):
    """Frame-wise L2-normalized velocity — motion direction only, scale-free.

    Compute v[t] = trajs[t+1] - trajs[t], then L2-normalize each frame
    independently so every frame vector has unit norm. Pure direction of
    motion per frame; amplitude is discarded. Two clips of the same
    gesture executed at different speeds or scales now look the same.
    """
    v = trajs[:, 1:] - trajs[:, :-1]       # [N, T-1, D]
    norms = v.norm(dim=-1, keepdim=True).clamp(min=1e-6)
    out = v / norms
    return out.contiguous(), (lens - 1).clamp(min=1)


def _tx_interjoint_distances(trajs, lens):
    """Per-frame 14×14 pairwise joint distance matrix, median-normalized.

    Fully invariant to rigid motion AND per-frame uniform scale by
    construction — WiLoR's weak-perspective scale confound MATHEMATICALLY
    cannot corrupt this representation. Output dim is the 91
    upper-triangular entries of the 14×14 distance matrix, divided by
    their per-frame median. The strongest theoretical candidate for
    breaking the cross-video DTW cost asymmetry.

    Output: [N, T, 91] (from either 28D or 42D input — dim-agnostic).
    """
    N, T, D = trajs.shape
    C = D // 14
    x = trajs.reshape(N, T, 14, C)
    # Pairwise distances: [N, T, 14, 14]
    diff = x[:, :, :, None, :] - x[:, :, None, :, :]
    dists = diff.norm(dim=-1)
    # Upper-triangular indices
    iu = torch.triu_indices(14, 14, offset=1, device=trajs.device)
    flat = dists[:, :, iu[0], iu[1]]   # [N, T, 91]
    # Per-frame median normalization
    median = flat.median(dim=-1, keepdim=True).values.clamp(min=1e-6)
    out = flat / median
    return out.contiguous(), lens


def _tx_turning_angle(trajs, lens):
    """Per-hand wrist velocity direction angles — pure motion direction.

    For each hand's wrist (joint 0), compute frame-wise velocity and
    convert to angle time-series. Produces 4 angles per frame: yaw and
    pitch for each of L and R wrists. Completely invariant to position
    AND scale (angles don't care). Captures "where is the hand going"
    without any magnitude information.

    Output: [N, T-1, 4] for 42D bundles (yaw + pitch per hand).
            [N, T-1, 2] for 28D bundles (yaw only — no depth axis for pitch).
    """
    N, T, D = trajs.shape
    C = D // 14
    x = trajs.reshape(N, T, 2, 7, C)
    wrist = x[:, :, :, 0, :]                                # [N, T, 2, C]
    v = wrist[:, 1:, :, :] - wrist[:, :-1, :, :]            # [N, T-1, 2, C]
    vx, vy = v[..., 0], v[..., 1]
    # Yaw: angle in image plane
    yaw = torch.atan2(vy, vx)                               # [N, T-1, 2]
    if C == 3:
        vz = v[..., 2]
        xy_norm = torch.sqrt(vx * vx + vy * vy + 1e-12)
        pitch = torch.atan2(vz, xy_norm)                    # [N, T-1, 2]
        out = torch.stack([yaw, pitch], dim=-1)             # [N, T-1, 2, 2]
        out = out.reshape(N, T - 1, 4)
    else:
        out = yaw                                           # [N, T-1, 2]
    return out.contiguous(), (lens - 1).clamp(min=1)


# --- Combined no_z stacks -----------------------------------------
# These stack a motion normalization with z-axis ablation: zeroing z on top
# of an already scale-normalized representation helps when depth is pure
# noise in the normalized space.

def _tx_centered_no_z(trajs, lens):
    """Centered + zero z dims."""
    centered, _ = _tx_centered(trajs, lens)
    out = centered.clone()
    out[..., _Z_DIMS] = 0.0
    return out.contiguous(), lens


def _tx_wrist_centered_no_z(trajs, lens):
    """Wrist-centered + zero z dims."""
    wc, _ = _tx_wrist_centered(trajs, lens)
    out = wc.clone()
    out[..., _Z_DIMS] = 0.0
    return out.contiguous(), lens


def _tx_hand_length_no_z(trajs, lens):
    """Hand-length-normalized + zero z dims."""
    hl, _ = _tx_hand_length_norm(trajs, lens)
    out = hl.clone()
    out[..., _Z_DIMS] = 0.0
    return out.contiguous(), lens


def _tx_wrist_centered_hand_length_no_z(trajs, lens):
    """Wrist-centered + hand-length-normalized + zero z — full stack."""
    wchl, _ = _tx_wrist_centered_hand_length(trajs, lens)
    out = wchl.clone()
    out[..., _Z_DIMS] = 0.0
    return out.contiguous(), lens


# --- 2D-bundle variants ----------------------------------------------------
# These mirror the corresponding 3D transforms for the 28D (2×7×2) 2D bundle,
# treated as first-class designs. They dim-infer from the input width (28 not 42).

_Z_DIMS_2D: list[int] = []   # no z-axis in 2D; placeholder for symmetry


def _extract_2d_xy(trajs):
    """Extract xy coords as [N, T, 2, 7, 2] from either 28D or 42D trajectory tensors.

    28D bundles (keypoints_2d_to_trajectories) are reshaped directly.
    42D bundles (keypoints_to_trajectories, the default) have z dropped.
    """
    N, T, D = trajs.shape
    if D == 28:
        return trajs.reshape(N, T, 2, 7, 2)
    elif D == 42:
        return trajs.reshape(N, T, 2, 7, 3)[..., :2].contiguous()
    else:
        raise ValueError(f"_extract_2d_xy: expected 28D or 42D trajectory, got {D}D")


def _tx_hand_length_norm_2d(trajs, lens):
    """Hand-length-norm in image plane — works with 28D or 42D bundles (z dropped).

    Layout after extraction: [N, T, 2, 7, 2] (hand, joint, xy).
    Wrist = joint 0, MCP = joint 1 per HAND_JOINT_INDICES.
    Reference length is the 2D wrist-to-MCP distance, median over valid frames.
    Output is always 28D (z discarded).
    """
    N, T, _ = trajs.shape
    x = _extract_2d_xy(trajs)                                           # [N, T, 2, 7, 2]
    wrist_to_mcp = (x[:, :, :, 1, :] - x[:, :, :, 0, :]).norm(dim=-1) # [N, T, 2]
    ref_len = wrist_to_mcp.median(dim=1).values.clamp(min=1e-6)         # [N, 2]
    scale_left  = ref_len[:, 0].reshape(N, 1, 1, 1)
    scale_right = ref_len[:, 1].reshape(N, 1, 1, 1)
    out = x.clone()
    out[:, :, 0] = x[:, :, 0] / scale_left
    out[:, :, 1] = x[:, :, 1] / scale_right
    return out.reshape(N, T, 28).contiguous(), lens


def _tx_wrist_centered_hand_length_2d(trajs, lens):
    """Wrist-centered + hand-length-norm in image plane — works with 28D or 42D (z dropped).

    Two-stage: subtract each hand's wrist from its joints (removes framing),
    then divide by wrist-to-MCP reference length (removes scale).
    Output is always 28D.
    """
    N, T, _ = trajs.shape
    x = _extract_2d_xy(trajs)                                           # [N, T, 2, 7, 2]
    x = x - x[:, :, :, 0:1, :]                                         # subtract wrist
    ref_len = x[:, :, :, 1, :].norm(dim=-1).median(dim=1).values.clamp(min=1e-6)  # [N, 2]
    x = x / ref_len[:, None, :, None, None]
    return x.reshape(N, T, 28).contiguous(), lens


def _tx_interjoint_distances_2d(trajs, lens):
    """14×14 pairwise 2D distances, upper-tri, median-norm — 91D output.

    Works with 28D or 42D bundles (z dropped). Rigid translation and
    per-frame uniform scale cannot corrupt pairwise distances.
    """
    N, T, _ = trajs.shape
    x = _extract_2d_xy(trajs).reshape(N, T, 14, 2)                     # [N, T, 14, 2]
    diff = x[:, :, :, None, :] - x[:, :, None, :, :]                   # [N, T, 14, 14, 2]
    dists = diff.norm(dim=-1)                                            # [N, T, 14, 14]
    iu = torch.triu_indices(14, 14, offset=1, device=trajs.device)
    flat = dists[:, :, iu[0], iu[1]]                                    # [N, T, 91]
    median = flat.median(dim=-1, keepdim=True).values.clamp(min=1e-6)
    return (flat / median).contiguous(), lens


def _tx_centroid_relative_2d(trajs, lens):
    """Centroid-relative in image plane — works with 28D or 42D bundles (z dropped).

    Subtract per-frame centroid, scale by spread. Output is always 28D.
    """
    N, T, _ = trajs.shape
    x = _extract_2d_xy(trajs).reshape(N, T, 14, 2)                     # [N, T, 14, 2]
    centroid = x.mean(dim=2, keepdim=True)
    relative = x - centroid
    scale = relative.reshape(N, -1).std(dim=1, keepdim=True).clamp(min=1e-6)
    out = relative / scale[:, :, None, None]
    return out.reshape(N, T, 28).contiguous(), lens


# --- First-frame-anchored transforms -----------------------------
# Designs that answer: "two clips tracing the same path from different starting
# positions should look similar." Frame 0 of each clip is treated as the
# origin; everything else is expressed relative to it. Unlike `wrist_centered`
# (which strips the wrist trajectory entirely per-frame), these PRESERVE the
# wrist trajectory shape while removing its absolute placement.

def _tx_first_frame_centered(trajs, lens):
    """Subtract frame 0 from every joint at every frame — per-joint anchoring.

    Each joint's trajectory is anchored to its own starting position. Two
    clips with identical per-joint trajectory shapes but different framing
    (e.g., hand starting on the left vs. right of the image) produce
    identical outputs. Preserves both wrist path shape and finger
    articulation shape, independently per joint.
    """
    # Frame 0 is always a real frame (padding replicates the last frame).
    anchor = trajs[:, 0:1, :]
    return (trajs - anchor).contiguous(), lens


def _n_joints_per_hand(D):
    """Infer joints-per-hand from trajectory dimension.

    42D → 7 joints (3D, 7-joint subset)
    28D → 7 joints (2D, 7-joint subset)
    126D → 21 joints (3D, all joints)
    """
    for n_j in (7, 21):
        C = D // (2 * n_j)
        if 2 * n_j * C == D and C in (2, 3):
            return n_j, C
    raise ValueError(f"Cannot infer joints from D={D}")


def _tx_first_frame_per_hand(trajs, lens):
    """Subtract each hand's frame-0 wrist from *all* joints in that hand.

    Per-hand anchoring: for hand H, subtract wrist_H(0) from joint_k(t) at
    every (t, k). After this, frame-0 wrist of each hand sits at the origin,
    the wrist trajectory relative to the starting position is preserved, AND
    the finger positions at every frame remain in the wrist-at-t=0 frame
    (so the finger-relative-to-wrist-at-start articulation is preserved).

    Works with 28D, 42D, or 126D input.
    """
    N, T, D = trajs.shape
    n_j, C = _n_joints_per_hand(D)
    x = trajs.reshape(N, T, 2, n_j, C)
    wrist0 = x[:, 0:1, :, 0:1, :]                  # [N, 1, 2, 1, C]
    x = x - wrist0
    return x.reshape(N, T, D).contiguous(), lens


def _tx_first_frame_midpoint(trajs, lens):
    """Subtract the frame-0 midpoint of both wrists from all joints.

    Single shared anchor: midpoint = (wrist_L(0) + wrist_R(0)) / 2, subtracted
    from every joint at every frame. Translation-invariant while preserving
    inter-hand distance and relative hand positions over time.

    Works with 28D, 42D, or 126D input.
    """
    N, T, D = trajs.shape
    n_j, C = _n_joints_per_hand(D)
    x = trajs.reshape(N, T, 2, n_j, C)
    wrist_L0 = x[:, 0:1, 0:1, 0:1, :]              # [N, 1, 1, 1, C]
    wrist_R0 = x[:, 0:1, 1:2, 0:1, :]              # [N, 1, 1, 1, C]
    midpoint0 = (wrist_L0 + wrist_R0) / 2           # [N, 1, 1, 1, C]
    x = x - midpoint0
    return x.reshape(N, T, D).contiguous(), lens


def _tx_first_frame_per_hand_no_z(trajs, lens):
    """`first_frame_per_hand` + zero z dims (3D-only)."""
    out, lens = _tx_first_frame_per_hand(trajs, lens)
    out = out.clone()
    if out.shape[-1] == 42:
        out[..., _Z_DIMS] = 0.0
    return out.contiguous(), lens


def _mcp_joint_index(n_j):
    """Middle-MCP joint index for hand-length reference.

    7-joint subset: index 1 (HAND_JOINT_INDICES[1] = joint 9 = middle MCP)
    21-joint full: index 9 (middle MCP directly)
    """
    return 1 if n_j == 7 else 9


def _per_hand_ref_length(trajs, n_j, C):
    """Median wrist-to-middle-MCP distance per hand [N, 2]. Scale reference."""
    N, T, D = trajs.shape
    x = trajs.reshape(N, T, 2, n_j, C)
    mcp_idx = _mcp_joint_index(n_j)
    ref_len = (x[:, :, :, mcp_idx, :] - x[:, :, :, 0, :]).norm(dim=-1)  # [N, T, 2]
    return ref_len.median(dim=1).values.clamp(min=1e-6)                   # [N, 2]


def _shared_ref_length(trajs, n_j, C):
    """Single shared scale factor from both hands [N, 1].

    Median of wrist-to-MCP distances across both hands and all frames.
    Using a shared scalar avoids WiLoR's noisy per-hand depth estimates
    producing different scale factors for left vs right.
    """
    N, T, D = trajs.shape
    x = trajs.reshape(N, T, 2, n_j, C)
    mcp_idx = _mcp_joint_index(n_j)
    ref_len = (x[:, :, :, mcp_idx, :] - x[:, :, :, 0, :]).norm(dim=-1)  # [N, T, 2]
    ref_len = ref_len.reshape(N, T * 2).median(dim=1).values.clamp(min=1e-6)  # [N]
    return ref_len


def _tx_first_frame_per_hand_hand_length(trajs, lens):
    """`first_frame_per_hand` + hand-length-normalized (shared scale).

    Anchored to frame-0 wrist per hand (preserves wrist path relative to
    start) and divided by shared wrist-to-MCP reference length from both
    hands (removes camera-focal / subject-size scale without amplifying
    WiLoR's per-hand depth noise). Full-articulation, translation-invariant,
    scale-invariant.

    Works with 42D or 126D input.
    """
    N, T, D = trajs.shape
    n_j, C = _n_joints_per_hand(D)
    x = trajs.reshape(N, T, 2, n_j, C)
    wrist0 = x[:, 0:1, :, 0:1, :]                  # [N, 1, 2, 1, C]
    x = x - wrist0
    ref_len = _shared_ref_length(trajs, n_j, C)     # [N]
    x = x / ref_len[:, None, None, None, None]
    return x.reshape(N, T, D).contiguous(), lens


def _tx_first_frame_midpoint_hand_length(trajs, lens):
    """`first_frame_midpoint` + hand-length-normalized (shared scale).

    Shared midpoint anchor (preserves inter-hand distance) and divided by
    shared wrist-to-MCP reference length from both hands (removes scale
    without amplifying per-hand depth noise). Translation-invariant,
    scale-invariant, inter-hand-preserving.

    Works with 42D or 126D input.
    """
    N, T, D = trajs.shape
    n_j, C = _n_joints_per_hand(D)
    x = trajs.reshape(N, T, 2, n_j, C)
    wrist_L0 = x[:, 0:1, 0:1, 0:1, :]              # [N, 1, 1, 1, C]
    wrist_R0 = x[:, 0:1, 1:2, 0:1, :]              # [N, 1, 1, 1, C]
    midpoint0 = (wrist_L0 + wrist_R0) / 2           # [N, 1, 1, 1, C]
    x = x - midpoint0
    ref_len = _shared_ref_length(trajs, n_j, C)     # [N]
    x = x / ref_len[:, None, None, None, None]
    return x.reshape(N, T, D).contiguous(), lens


Z_DOWN_WEIGHT = 0.3


def _tx_first_frame_per_hand_hand_length_zdown(trajs, lens):
    """`first_frame_per_hand_hand_length` + down-weighted z axis.

    Scales z coordinates by Z_DOWN_WEIGHT (0.3) so depth contributes ~9%
    of per-dimension variance to Euclidean DTW distance instead of 33%.
    Keeps some depth signal for clips where it's reliable without letting
    WiLoR's noisy z dominate the distance computation.

    Works with 42D or 126D input (3D only — requires C=3).
    """
    out, lens = _tx_first_frame_per_hand_hand_length(trajs, lens)
    out = out.clone()
    N, T, D = out.shape
    n_j, C = _n_joints_per_hand(D)
    if C == 3:
        x = out.reshape(N, T, 2, n_j, 3)
        x[:, :, :, :, 2] *= Z_DOWN_WEIGHT
        out = x.reshape(N, T, D).contiguous()
    return out, lens


def _drop_z(trajs_3d):
    """Drop z coordinates from [N, T, D] 3D trajectories → [N, T, D_2d]."""
    N, T, D = trajs_3d.shape
    n_j, C = _n_joints_per_hand(D)
    if C != 3:
        return trajs_3d
    x = trajs_3d.reshape(N, T, 2 * n_j, 3)
    return x[:, :, :, :2].reshape(N, T, 2 * n_j * 2).contiguous()


def _tx_first_frame_per_hand_hand_length_no_z(trajs, lens):
    """`first_frame_per_hand_hand_length` with z dropped. Works with 42D or 126D."""
    out, lens = _tx_first_frame_per_hand_hand_length(trajs, lens)
    return _drop_z(out), lens


def _tx_first_frame_midpoint_hand_length_no_z(trajs, lens):
    """`first_frame_midpoint_hand_length` with z dropped. Works with 42D or 126D."""
    out, lens = _tx_first_frame_midpoint_hand_length(trajs, lens)
    return _drop_z(out), lens


def _tx_body_full_pose_21j(trajs, lens):
    """All body-frame 21-joint hand keypoints.

    Intended for depth-grounded keypoints that have already been transformed
    into the estimated torso frame. This intentionally keeps absolute hand
    position in the shared body coordinate system.
    """
    n_j, C = _n_joints_per_hand(trajs.shape[-1])
    if n_j != 21 or C != 3:
        raise ValueError(
            f"abs_21j_coords requires 126D 3D trajectories, got D={trajs.shape[-1]}"
        )
    return trajs.contiguous(), lens


def _tx_body_wrist_path_3d(trajs, lens):
    """Left/right wrist positions in the shared body-frame XYZ space."""
    N, T, D = trajs.shape
    n_j, C = _n_joints_per_hand(D)
    if C != 3:
        raise ValueError(f"abs_wrist_coords requires 3D trajectories, got D={D}")
    x = trajs.reshape(N, T, 2, n_j, C)
    wrists = x[:, :, :, 0, :]
    return wrists.reshape(N, T, 2 * C).contiguous(), lens


def _tx_body_wrist_velocity_3d(trajs, lens):
    """Frame-to-frame left/right wrist velocity in body-frame XYZ."""
    wrists, _ = _tx_body_wrist_path_3d(trajs, lens)
    v = wrists[:, 1:] - wrists[:, :-1]
    return v.contiguous(), (lens - 1).clamp(min=1)


def _tx_body_full_pose_21j_velocity(trajs, lens):
    """Frame-to-frame velocity for all 21 hand joints in body-frame XYZ."""
    n_j, C = _n_joints_per_hand(trajs.shape[-1])
    if n_j != 21 or C != 3:
        raise ValueError(
            "body_full_pose_21j_velocity requires 126D 3D trajectories, "
            f"got D={trajs.shape[-1]}"
        )
    v = trajs[:, 1:] - trajs[:, :-1]
    return v.contiguous(), (lens - 1).clamp(min=1)


BODY_21J_HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (0, 9), (9, 10), (10, 11), (11, 12),
    (0, 13), (13, 14), (14, 15), (15, 16),
    (0, 17), (17, 18), (18, 19), (19, 20),
]


def _body_21j_angle_triples():
    neighbors = {i: [] for i in range(21)}
    for a, b in BODY_21J_HAND_CONNECTIONS:
        neighbors[a].append(b)
        neighbors[b].append(a)
    triples = []
    for center in range(21):
        ns = sorted(neighbors[center])
        for i, a in enumerate(ns):
            for b in ns[i + 1:]:
                triples.append((a, center, b))
    return triples


BODY_21J_ANGLE_TRIPLES = _body_21j_angle_triples()


def _tx_body_articulation_21j(trajs, lens):
    """Per-hand joint angles from the visualization skeleton.

    Uses the same 21-joint connections as
    retrieval_training/torso_estimation_training/estimate.py::MANO_CONNECTIONS. For each
    hand, every joint with two or more connected neighbors contributes all
    unordered neighbor-pair angles. This produces 25 angles per hand: 10 wrist
    spread angles plus 15 chain bend angles.

    Angles are invariant to rigid rotation, translation, and scale of each hand,
    so this design models articulation without body-frame hand orientation or
    body-relative hand location.
    """
    N, T, D = trajs.shape
    n_j, C = _n_joints_per_hand(D)
    if n_j != 21 or C != 3:
        raise ValueError(
            f"angles_21j requires 126D 3D trajectories, got D={D}"
        )
    x = trajs.reshape(N, T, 2, n_j, C)
    angle_feats = []
    for a, center, b in BODY_21J_ANGLE_TRIPLES:
        va = x[:, :, :, a, :] - x[:, :, :, center, :]
        vb = x[:, :, :, b, :] - x[:, :, :, center, :]
        va = F.normalize(va, p=2, dim=-1, eps=1e-8)
        vb = F.normalize(vb, p=2, dim=-1, eps=1e-8)
        cos = (va * vb).sum(dim=-1).clamp(-1.0, 1.0)
        angle_feats.append(torch.acos(cos))
    angles = torch.stack(angle_feats, dim=-1)
    return angles.reshape(N, T, 2 * len(BODY_21J_ANGLE_TRIPLES)).contiguous(), lens


def _tx_body_wrist_relative_pose_21j(trajs, lens):
    """All 21 joints expressed relative to each hand wrist per frame.

    This is the original body_articulation_21j feature: it removes absolute
    hand location while preserving 3D hand shape and orientation in body-frame
    axes. It is translation invariant per hand, but not rotation or scale
    invariant.
    """
    N, T, D = trajs.shape
    n_j, C = _n_joints_per_hand(D)
    if n_j != 21 or C != 3:
        raise ValueError(
            f"wrist_rel_21j_coords requires 126D 3D trajectories, got D={D}"
        )
    x = trajs.reshape(N, T, 2, n_j, C)
    x = x - x[:, :, :, 0:1, :]
    return x.reshape(N, T, D).contiguous(), lens


def _tx_body_interjoint_distances_21j(trajs, lens):
    """All pairwise distances among 42 hand joints per frame.

    Distances are invariant to rigid rotation/translation of the body-frame
    coordinate system while preserving hand shape and inter-hand geometry.
    Output dim is C(42, 2) = 861.
    """
    N, T, D = trajs.shape
    n_j, C = _n_joints_per_hand(D)
    if n_j != 21 or C != 3:
        raise ValueError(
            "full_interjoint_dists requires 126D 3D trajectories, "
            f"got D={D}"
        )
    x = trajs.reshape(N, T, 2 * n_j, C)
    iu = torch.triu_indices(2 * n_j, 2 * n_j, offset=1, device=trajs.device)
    dists = (x[:, :, iu[0], :] - x[:, :, iu[1], :]).norm(dim=-1)
    return dists.contiguous(), lens


PCA_INTERJOINT_DIST_PAIRS_21J = [
    # Top 40 cross-hand pairs from standardized 10k/99% PCA leave-one-out ranking.
    (11, 35),  # left:Ring2 - right:Thumb2
    (10, 35),  # left:Ring1 - right:Thumb2
    (11, 30),  # left:Ring2 - right:Pinky3
    (10, 31),  # left:Ring1 - right:Ring1
    (15, 35),  # left:Thumb3 - right:Thumb2
    (11, 31),  # left:Ring2 - right:Ring1
    (11, 34),  # left:Ring2 - right:Thumb1
    (14, 31),  # left:Thumb2 - right:Ring1
    (10, 30),  # left:Ring1 - right:Pinky3
    (14, 35),  # left:Thumb2 - right:Thumb2
    (15, 30),  # left:Thumb3 - right:Pinky3
    (15, 31),  # left:Thumb3 - right:Ring1
    (9, 31),   # left:Pinky3 - right:Ring1
    (14, 32),  # left:Thumb2 - right:Ring2
    (10, 32),  # left:Ring1 - right:Ring2
    (14, 30),  # left:Thumb2 - right:Pinky3
    (11, 36),  # left:Ring2 - right:Thumb3
    (15, 34),  # left:Thumb3 - right:Thumb1
    (10, 36),  # left:Ring1 - right:Thumb3
    (9, 35),   # left:Pinky3 - right:Thumb2
    (15, 36),  # left:Thumb3 - right:Thumb3
    (10, 34),  # left:Ring1 - right:Thumb1
    (14, 36),  # left:Thumb2 - right:Thumb3
    (9, 32),   # left:Pinky3 - right:Ring2
    (5, 31),   # left:Middle2 - right:Ring1
    (15, 32),  # left:Thumb3 - right:Ring2
    (11, 32),  # left:Ring2 - right:Ring2
    (5, 35),   # left:Middle2 - right:Thumb2
    (9, 30),   # left:Pinky3 - right:Pinky3
    (9, 36),   # left:Pinky3 - right:Thumb3
    (14, 34),  # left:Thumb2 - right:Thumb1
    (5, 30),   # left:Middle2 - right:Pinky3
    (11, 39),  # left:Ring2 - right:Middle4
    (5, 32),   # left:Middle2 - right:Ring2
    (13, 31),  # left:Thumb1 - right:Ring1
    (7, 35),   # left:Pinky1 - right:Thumb2
    (6, 35),   # left:Middle3 - right:Thumb2
    (2, 30),   # left:Index2 - right:Pinky3
    (12, 35),  # left:Ring3 - right:Thumb2
    (13, 32),  # left:Thumb1 - right:Ring2

    # Top 60 within-left-hand pairs.
    (0, 9),    # left:Wrist - left:Pinky3
    (0, 17),   # left:Wrist - left:Index4
    (0, 13),   # left:Wrist - left:Thumb1
    (1, 9),    # left:Index1 - left:Pinky3
    (0, 5),    # left:Wrist - left:Middle2
    (5, 17),   # left:Middle2 - left:Index4
    (1, 17),   # left:Index1 - left:Index4
    (1, 5),    # left:Index1 - left:Middle2
    (1, 13),   # left:Index1 - left:Thumb1
    (9, 17),   # left:Pinky3 - left:Index4
    (5, 13),   # left:Middle2 - left:Thumb1
    (5, 6),    # left:Middle2 - left:Middle3
    (0, 1),    # left:Wrist - left:Index1
    (9, 10),   # left:Pinky3 - left:Ring1
    (1, 2),    # left:Index1 - left:Index2
    (13, 14),  # left:Thumb1 - left:Thumb2
    (2, 3),    # left:Index2 - left:Index3
    (5, 9),    # left:Middle2 - left:Pinky3
    (9, 13),   # left:Pinky3 - left:Thumb1
    (10, 11),  # left:Ring1 - left:Ring2
    (14, 15),  # left:Thumb2 - left:Thumb3
    (6, 7),    # left:Middle3 - left:Pinky1
    (17, 18),  # left:Index4 - left:Middle4
    (11, 12),  # left:Ring2 - left:Ring3
    (13, 17),  # left:Thumb1 - left:Index4
    (7, 8),    # left:Pinky1 - left:Pinky2
    (1, 3),    # left:Index1 - left:Index3
    (18, 19),  # left:Middle4 - left:Ring4
    (15, 16),  # left:Thumb3 - left:Thumb4
    (0, 3),    # left:Wrist - left:Index3
    (0, 2),    # left:Wrist - left:Index2
    (3, 4),    # left:Index3 - left:Middle1
    (0, 14),   # left:Wrist - left:Thumb2
    (0, 6),    # left:Wrist - left:Middle3
    (0, 10),   # left:Wrist - left:Ring1
    (6, 8),    # left:Middle3 - left:Pinky2
    (10, 12),  # left:Ring1 - left:Ring3
    (0, 11),   # left:Wrist - left:Ring2
    (0, 15),   # left:Wrist - left:Thumb3
    (0, 7),    # left:Wrist - left:Pinky1
    (1, 14),   # left:Index1 - left:Thumb2
    (2, 10),   # left:Index2 - left:Ring1
    (2, 15),   # left:Index2 - left:Thumb3
    (0, 18),   # left:Wrist - left:Middle4
    (11, 19),  # left:Ring2 - left:Ring4
    (1, 10),   # left:Index1 - left:Ring1
    (2, 11),   # left:Index2 - left:Ring2
    (1, 15),   # left:Index1 - left:Thumb3
    (13, 15),  # left:Thumb1 - left:Thumb3
    (1, 6),    # left:Index1 - left:Middle3
    (1, 11),   # left:Index1 - left:Ring2
    (11, 13),  # left:Ring2 - left:Thumb1
    (14, 16),  # left:Thumb2 - left:Thumb4
    (2, 14),   # left:Index2 - left:Thumb2
    (5, 18),   # left:Middle2 - left:Middle4
    (1, 7),    # left:Index1 - left:Pinky1
    (2, 4),    # left:Index2 - left:Middle1
    (1, 18),   # left:Index1 - left:Middle4
    (5, 15),   # left:Middle2 - left:Thumb3
    (12, 17),  # left:Ring3 - left:Index4

    # Same within-hand local pairs mirrored onto the right hand.
    (21, 30),  # right:Wrist - right:Pinky3
    (21, 38),  # right:Wrist - right:Index4
    (21, 34),  # right:Wrist - right:Thumb1
    (22, 30),  # right:Index1 - right:Pinky3
    (21, 26),  # right:Wrist - right:Middle2
    (26, 38),  # right:Middle2 - right:Index4
    (22, 38),  # right:Index1 - right:Index4
    (22, 26),  # right:Index1 - right:Middle2
    (22, 34),  # right:Index1 - right:Thumb1
    (30, 38),  # right:Pinky3 - right:Index4
    (26, 34),  # right:Middle2 - right:Thumb1
    (26, 27),  # right:Middle2 - right:Middle3
    (21, 22),  # right:Wrist - right:Index1
    (30, 31),  # right:Pinky3 - right:Ring1
    (22, 23),  # right:Index1 - right:Index2
    (34, 35),  # right:Thumb1 - right:Thumb2
    (23, 24),  # right:Index2 - right:Index3
    (26, 30),  # right:Middle2 - right:Pinky3
    (30, 34),  # right:Pinky3 - right:Thumb1
    (31, 32),  # right:Ring1 - right:Ring2
    (35, 36),  # right:Thumb2 - right:Thumb3
    (27, 28),  # right:Middle3 - right:Pinky1
    (38, 39),  # right:Index4 - right:Middle4
    (32, 33),  # right:Ring2 - right:Ring3
    (34, 38),  # right:Thumb1 - right:Index4
    (28, 29),  # right:Pinky1 - right:Pinky2
    (22, 24),  # right:Index1 - right:Index3
    (39, 40),  # right:Middle4 - right:Ring4
    (36, 37),  # right:Thumb3 - right:Thumb4
    (21, 24),  # right:Wrist - right:Index3
    (21, 23),  # right:Wrist - right:Index2
    (24, 25),  # right:Index3 - right:Middle1
    (21, 35),  # right:Wrist - right:Thumb2
    (21, 27),  # right:Wrist - right:Middle3
    (21, 31),  # right:Wrist - right:Ring1
    (27, 29),  # right:Middle3 - right:Pinky2
    (31, 33),  # right:Ring1 - right:Ring3
    (21, 32),  # right:Wrist - right:Ring2
    (21, 36),  # right:Wrist - right:Thumb3
    (21, 28),  # right:Wrist - right:Pinky1
    (22, 35),  # right:Index1 - right:Thumb2
    (23, 31),  # right:Index2 - right:Ring1
    (23, 36),  # right:Index2 - right:Thumb3
    (21, 39),  # right:Wrist - right:Middle4
    (32, 40),  # right:Ring2 - right:Ring4
    (22, 31),  # right:Index1 - right:Ring1
    (23, 32),  # right:Index2 - right:Ring2
    (22, 36),  # right:Index1 - right:Thumb3
    (34, 36),  # right:Thumb1 - right:Thumb3
    (22, 27),  # right:Index1 - right:Middle3
    (22, 32),  # right:Index1 - right:Ring2
    (32, 34),  # right:Ring2 - right:Thumb1
    (35, 37),  # right:Thumb2 - right:Thumb4
    (23, 35),  # right:Index2 - right:Thumb2
    (26, 39),  # right:Middle2 - right:Middle4
    (22, 28),  # right:Index1 - right:Pinky1
    (23, 25),  # right:Index2 - right:Middle1
    (22, 39),  # right:Index1 - right:Middle4
    (26, 36),  # right:Middle2 - right:Thumb3
    (33, 38),  # right:Ring3 - right:Index4
]


def _tx_pca_interjoint_dists(trajs, lens):
    """Selected 21-joint distances from standardized PCA pair ranking.

    Uses 160 fixed distances: the top 40 cross-hand pairs from a standardized
    leave-one-out PCA ranking (10k clips, 99% retained variance), plus the top
    60 within-left-hand local pairs mirrored onto the right hand. This keeps
    inter-hand pose while forcing symmetric within-hand articulation coverage.
    """
    N, T, D = trajs.shape
    n_j, C = _n_joints_per_hand(D)
    if n_j != 21 or C != 3:
        raise ValueError(f"pca_interjoint_dists requires 126D 3D trajectories, got D={D}")
    x = trajs.reshape(N, T, 2 * n_j, C)
    pairs = torch.tensor(PCA_INTERJOINT_DIST_PAIRS_21J, dtype=torch.long, device=trajs.device)
    dists = (x[:, :, pairs[:, 0], :] - x[:, :, pairs[:, 1], :]).norm(dim=-1)
    return dists.contiguous(), lens


def _body_21j_canonical_wrist_path(trajs):
    """Return two-wrist paths in the frame-0 wrist-relative canonical frame."""
    N, T, D = trajs.shape
    n_j, C = _n_joints_per_hand(D)
    if n_j != 21 or C != 3:
        raise ValueError(
            f"wrist_rel_path requires 126D 3D trajectories, got D={D}"
        )
    x = trajs.reshape(N, T, 2, n_j, C)
    wrists = x[:, :, :, 0, :]  # [N, T, 2, 3]

    left0 = wrists[:, 0, 0, :]
    right0 = wrists[:, 0, 1, :]
    origin = 0.5 * (left0 + right0)
    centered = wrists - origin[:, None, None, :]

    x_axis = F.normalize(right0 - left0, p=2, dim=-1, eps=1e-8)
    up = torch.zeros_like(x_axis)
    up[:, 2] = 1.0
    y_axis = up - (up * x_axis).sum(dim=-1, keepdim=True) * x_axis
    y_norm = y_axis.norm(dim=-1, keepdim=True)
    fallback = torch.zeros_like(x_axis)
    fallback[:, 1] = 1.0
    y_axis = torch.where(y_norm > 1e-6, y_axis / y_norm.clamp_min(1e-8), fallback)
    z_axis = F.normalize(torch.cross(x_axis, y_axis, dim=-1), p=2, dim=-1, eps=1e-8)
    y_axis = F.normalize(torch.cross(z_axis, x_axis, dim=-1), p=2, dim=-1, eps=1e-8)
    basis = torch.stack([x_axis, y_axis, z_axis], dim=-1)  # columns are canonical axes

    return torch.matmul(centered.reshape(N, T * 2, C), basis).reshape(N, T, 2 * C)


def _tx_wrist_rel_path(trajs, lens):
    """Canonicalized 6D left/right wrist path.

    Subtracts the frame-0 wrist midpoint and rotates into a per-clip frame
    whose x-axis is the initial left-to-right wrist vector, matching the path
    channel used by pca_interjoint_dists_w_wrist_rel_path.
    """
    return _body_21j_canonical_wrist_path(trajs).contiguous(), lens


def _tx_pca_interjoint_dists_with_wrist_path(trajs, lens):
    """Selected interjoint distances plus canonicalized two-wrist paths.

    The 50 selected distances are already translation/rotation invariant. The
    wrist-path channel adds hand motion by subtracting the frame-0 wrist
    midpoint and rotating into a per-clip frame whose x-axis is the initial
    left-to-right wrist vector. A stable body-frame up fallback fixes the
    remaining roll, making this path invariant to starting translation and
    rotation while preserving two-hand relative motion.
    """
    dists, lens = _tx_pca_interjoint_dists(trajs, lens)
    path = _body_21j_canonical_wrist_path(trajs)
    return torch.cat([dists, path], dim=-1).contiguous(), lens


def _tx_wrist_path_first_frame(trajs, lens):
    """Wrists only, anchored to frame-0 per hand — pure hand-path, no articulation.

    Extracts the 2 wrist joints (6D for 3D bundles, 4D for 2D bundles) and
    subtracts each hand's frame-0 wrist. Result is the wrist trajectory
    shape for each hand, with starting position removed. No finger
    information at all — useful to isolate whether articulation helps.
    """
    N, T, D = trajs.shape
    C = D // 14
    x = trajs.reshape(N, T, 2, 7, C)
    w = x[:, :, :, 0, :]                           # [N, T, 2, C] — wrists only
    w = w - w[:, 0:1, :, :]                        # anchor each hand to frame 0
    return w.reshape(N, T, 2 * C).contiguous(), lens


def _tx_wrist_path_first_frame_no_z(trajs, lens):
    """Wrists only, frame-0 anchored, z zeroed (3D-only → effectively 4D)."""
    out, lens = _tx_wrist_path_first_frame(trajs, lens)
    if out.shape[-1] == 6:
        out = out.clone()
        out[..., [2, 5]] = 0.0
    return out.contiguous(), lens


def _tx_wrist_path_centered(trajs, lens):
    """Wrists only, temporal-mean centered — alternative anchoring for comparison.

    Extracts the 2 wrist joints and subtracts each hand's per-clip temporal
    mean. Same "pure hand path" feature as `wrist_path_first_frame` but with
    mean-centering instead of frame-0 anchoring. Lets us A/B the two
    translation-removal strategies.
    """
    N, T, D = trajs.shape
    C = D // 14
    x = trajs.reshape(N, T, 2, 7, C)
    w = x[:, :, :, 0, :]                           # [N, T, 2, C]
    w = w - w.mean(dim=1, keepdim=True)            # subtract per-clip mean
    return w.reshape(N, T, 2 * C).contiguous(), lens


def _wrist_path_pca_core(w, lens, C):
    """Shared PCA-canonicalization kernel for wrist paths.

    `w` has shape [N, T, 2, C] — per-clip wrist sequences for both hands.
    Returns [N, T, 2, C] with a deterministic per-clip rotation applied.

    Algorithm per clip:
      1. Stack both wrists across time into a single [2T, C] point cloud
         and mean-center.
      2. Eigendecompose the (C×C) covariance to get principal axes
         (eigh + epsilon regularizer, then reordered descending).
      3. Rotate every frame's wrist point by that matrix.
      4. Sign-fix each axis so the last-valid-frame projection summed
         across hands is non-negative — deterministic, time-aware.
      5. Enforce det = +1 (rotation, not reflection): if the product of
         sign flips and det(V) is negative, flip the smallest-variance
         axis one more time. This preserves chirality so mirror-image
         motions do not collapse to the same canonicalization.

    The same rotation is applied to both hands so their relative geometry
    (the inter-hand path structure) is preserved. Operates on CPU — the
    per-clip C×C eigendecomp is cheap for C ∈ {2, 3}.
    """
    N, T, _, _ = w.shape
    pts = w.reshape(N, T * 2, C)                         # [N, 2T, C]
    mu = pts.mean(dim=1, keepdim=True)                   # [N, 1, C]
    centered = pts - mu                                   # [N, 2T, C]

    # Per-clip covariance + epsilon regularizer. eigh returns ascending eigvals.
    cov = centered.transpose(-1, -2) @ centered          # [N, C, C]
    eps = 1e-6 * torch.eye(C, device=cov.device, dtype=cov.dtype).unsqueeze(0)
    _, eigvecs = torch.linalg.eigh(cov + eps)            # [N, C, C]
    V = eigvecs.flip(-1).contiguous()                     # descending order

    proj = centered @ V                                   # [N, 2T, C]
    proj = proj.reshape(N, T, 2, C)

    # Sign convention: last-valid-frame projection summed across hands ≥ 0.
    last_idx = (lens - 1).clamp(min=0).long()             # [N]
    ar = torch.arange(N, device=proj.device)
    ref = proj[ar, last_idx].sum(dim=1)                   # [N, C]
    signs = torch.sign(ref)
    signs[signs == 0] = 1.0
    proj = proj * signs.view(N, 1, 1, C)

    # Chirality: require det(V * diag(signs)) = +1. If < 0, flip the
    # smallest-variance axis (column index -1 after the descending flip).
    det_V = torch.det(V)                                  # [N]
    det_final = det_V * signs.prod(dim=-1)                # [N]
    flip_last = det_final < 0
    if flip_last.any():
        proj[flip_last, ..., -1] = -proj[flip_last, ..., -1]

    return proj


def _tx_wrist_path_pca(trajs, lens):
    """Wrist paths per-clip PCA-canonicalized (per-clip rigid-rotation invariant).

    Motivation: if every clip is filmed by a static camera at a different
    rigid pose relative to the world, two clips of the *same* world-frame
    motion differ by an arbitrary per-clip rotation + translation. Existing
    translation-invariant designs absorb the translation; this one absorbs
    the rotation too, while preserving the *shape* of each wrist's path and
    inter-hand geometry.

    Procedure: extract both wrist joints per clip, mean-center the stacked
    wrist point cloud, solve for its principal axes via eigendecomposition,
    and rotate every wrist frame into that basis. Sign ambiguities are
    fixed deterministically (time-aware last-frame convention, chirality
    enforced via det = +1). Throws away articulation by design — pair with
    an articulation-aware design if finger motion needs to contribute.
    """
    N, T, D = trajs.shape
    C = D // 14
    x = trajs.reshape(N, T, 2, 7, C)
    w = x[:, :, :, 0, :].contiguous()                     # [N, T, 2, C]
    proj = _wrist_path_pca_core(w, lens, C)
    return proj.reshape(N, T, 2 * C).contiguous(), lens


def _tx_wrist_path_pca_no_z(trajs, lens):
    """`wrist_path_pca` with WiLoR's noisy depth axis dropped before canonicalization.

    For 3D bundles: strip z from both wrists, run a 2D PCA canonicalization
    in the image plane, and pad the z channel back with zeros so the output
    stays in the 6D layout the rest of the pipeline expects. For 2D bundles
    this is identical to `wrist_path_pca`.
    """
    N, T, D = trajs.shape
    C = D // 14
    x = trajs.reshape(N, T, 2, 7, C)
    w = x[:, :, :, 0, :]                                  # [N, T, 2, C]
    if C == 3:
        w_xy = w[..., :2].contiguous()                    # [N, T, 2, 2]
        proj_xy = _wrist_path_pca_core(w_xy, lens, 2)
        pad = torch.zeros(N, T, 2, 1, device=w.device, dtype=w.dtype)
        proj = torch.cat([proj_xy, pad], dim=-1)          # [N, T, 2, 3]
    else:
        proj = _wrist_path_pca_core(w.contiguous(), lens, C)
    return proj.reshape(N, T, 2 * C).contiguous(), lens


# ---------------------------------------------------------------------------
# PCA-× articulation hybrids
#
# These designs apply a per-clip PCA canonicalization rotation to ALL 14
# joints (not just wrists), preserving finger articulation while removing
# per-clip rigid-rotation nuisance. Different variants mix translation /
# scale normalization and choice of PCA basis.
# ---------------------------------------------------------------------------

def _pca_basis_and_mean(pts, C):
    """Compute per-clip PCA rotation V and mean mu from point cloud [N, M, C].

    V is returned in descending-variance order (flipped from eigh's ascending),
    regularized by epsilon·I. Returns (V [N, C, C], mu [N, 1, C]).
    """
    mu = pts.mean(dim=1, keepdim=True)                  # [N, 1, C]
    centered = pts - mu
    cov = centered.transpose(-1, -2) @ centered          # [N, C, C]
    eps = 1e-6 * torch.eye(C, device=cov.device, dtype=cov.dtype).unsqueeze(0)
    _, eigvecs = torch.linalg.eigh(cov + eps)
    V = eigvecs.flip(-1).contiguous()
    return V, mu


def _apply_pca_to_joints(x, V, mu, lens, C):
    """Rotate joint tensor `x` [N, T, 2, 7, C] by V around mu.

    Sign convention: last-valid-frame wrist projection summed across hands ≥ 0
    (deterministic, time-aware). Chirality enforced via det(V · diag(signs)) = +1
    — if negative, flip the smallest-variance axis.
    """
    N, T, H, J, _ = x.shape
    centered = x - mu.view(N, 1, 1, 1, C)
    flat = centered.reshape(N, T * H * J, C)
    proj = (flat @ V).reshape(N, T, H, J, C)

    # Sign-fix from wrist positions at last valid frame (joint 0, summed over hands).
    last_idx = (lens - 1).clamp(min=0).long()
    ar = torch.arange(N, device=proj.device)
    ref = proj[ar, last_idx, :, 0, :].sum(dim=1)         # [N, C]
    signs = torch.sign(ref)
    signs[signs == 0] = 1.0
    proj = proj * signs.view(N, 1, 1, 1, C)

    # Chirality: require det(V · diag(signs)) = +1.
    det_V = torch.det(V)
    det_final = det_V * signs.prod(dim=-1)
    flip_last = det_final < 0
    if flip_last.any():
        proj[flip_last, ..., -1] = -proj[flip_last, ..., -1]

    return proj


def _tx_full_pose_pca(trajs, lens):
    """PCA canonicalization on the full 14-joint cloud, rotation applied to all joints.

    Basis: every (hand, joint, time) point contributes to the covariance, so
    the canonical axes reflect the dominant direction of the whole pose
    cloud — wrist motion and finger spread together. No translation anchor:
    the PCA mean absorbs it. Per-clip rotation-invariant with full articulation.
    """
    N, T, D = trajs.shape
    C = D // 14
    x = trajs.reshape(N, T, 2, 7, C)
    pts = x.reshape(N, T * 14, C)
    V, mu = _pca_basis_and_mean(pts, C)
    proj = _apply_pca_to_joints(x, V, mu, lens, C)
    return proj.reshape(N, T, D).contiguous(), lens


def _tx_first_frame_per_hand_pca(trajs, lens):
    """Per-hand frame-0 wrist anchor + PCA (wrist basis), rotation applied to all joints.

    Translation invariant (per-hand wrist-at-t=0 → origin), rotation
    invariant (wrist-path PCA frame), full finger articulation preserved.
    Expected to combine the top-performing wrist_path_pca with the top
    articulation-preserving first_frame_per_hand anchor.
    """
    N, T, D = trajs.shape
    C = D // 14
    x = trajs.reshape(N, T, 2, 7, C)
    wrist0 = x[:, 0:1, :, 0:1, :]                        # [N, 1, 2, 1, C]
    x = x - wrist0
    w = x[:, :, :, 0, :].reshape(N, T * 2, C)
    V, mu = _pca_basis_and_mean(w, C)
    proj = _apply_pca_to_joints(x, V, mu, lens, C)
    return proj.reshape(N, T, D).contiguous(), lens


def _tx_first_frame_per_hand_hand_length_pca(trajs, lens):
    """Anchor + hand-length scale norm + PCA (wrist basis) applied to all joints.

    All three rigid invariances (translation, scale, rotation) plus full
    articulation. The "everything combined" design.
    """
    N, T, D = trajs.shape
    C = D // 14
    x = trajs.reshape(N, T, 2, 7, C)
    orig = trajs.reshape(N, T, 2, 7, C)
    wrist0 = x[:, 0:1, :, 0:1, :]
    x = x - wrist0
    # Hand-length reference from unanchored positions (translation-invariant).
    ref_len = (orig[:, :, :, 1, :] - orig[:, :, :, 0, :]).norm(dim=-1)   # [N, T, 2]
    ref_len = ref_len.median(dim=1).values.clamp(min=1e-6)                # [N, 2]
    x = x / ref_len[:, None, :, None, None]
    w = x[:, :, :, 0, :].reshape(N, T * 2, C)
    V, mu = _pca_basis_and_mean(w, C)
    proj = _apply_pca_to_joints(x, V, mu, lens, C)
    return proj.reshape(N, T, D).contiguous(), lens


def _tx_first_frame_centered_pca(trajs, lens):
    """Per-joint frame-0 anchor + PCA (wrist basis) applied to all joints.

    Alternative anchoring to `first_frame_per_hand_pca`: every joint is
    subtracted by its own t=0 position (so finger-at-t=0 → origin), then
    the wrist-path PCA rotates the whole pose. Useful A/B comparison.
    """
    N, T, D = trajs.shape
    C = D // 14
    x = trajs.reshape(N, T, 2, 7, C)
    x = x - x[:, 0:1, :, :, :]
    w = x[:, :, :, 0, :].reshape(N, T * 2, C)
    V, mu = _pca_basis_and_mean(w, C)
    proj = _apply_pca_to_joints(x, V, mu, lens, C)
    return proj.reshape(N, T, D).contiguous(), lens


# ---------------------------------------------------------------------------
# Creative designs — invariance-gap-filling variants of the PCA hybrids
# ---------------------------------------------------------------------------

def _tx_full_pose_pca_no_z(trajs, lens):
    """`full_pose_pca` with the noisy WiLoR z axis dropped before canonicalization.

    PCA runs on xy only; z is zeroed in the output.
    """
    N, T, D = trajs.shape
    C = D // 14
    if C != 3:
        return _tx_full_pose_pca(trajs, lens)
    x = trajs.reshape(N, T, 2, 7, 3)
    pts = x[..., :2].reshape(N, T * 14, 2)
    V, mu = _pca_basis_and_mean(pts, 2)
    xy = x[..., :2].contiguous()
    proj_xy = _apply_pca_to_joints(xy, V, mu, lens, 2)
    pad = torch.zeros(N, T, 2, 7, 1, device=x.device, dtype=x.dtype)
    proj = torch.cat([proj_xy, pad], dim=-1)
    return proj.reshape(N, T, D).contiguous(), lens


def _tx_first_frame_per_hand_pca_no_z(trajs, lens):
    """`first_frame_per_hand_pca` with z dropped. Anchor + 2D PCA in xy."""
    N, T, D = trajs.shape
    C = D // 14
    if C != 3:
        return _tx_first_frame_per_hand_pca(trajs, lens)
    x = trajs.reshape(N, T, 2, 7, 3)
    wrist0 = x[:, 0:1, :, 0:1, :]
    x = x - wrist0
    w_xy = x[:, :, :, 0, :2].reshape(N, T * 2, 2)
    V, mu = _pca_basis_and_mean(w_xy, 2)
    xy = x[..., :2].contiguous()
    proj_xy = _apply_pca_to_joints(xy, V, mu, lens, 2)
    pad = torch.zeros(N, T, 2, 7, 1, device=x.device, dtype=x.dtype)
    proj = torch.cat([proj_xy, pad], dim=-1)
    return proj.reshape(N, T, D).contiguous(), lens


def _tx_first_frame_per_hand_hand_length_pca_no_z(trajs, lens):
    """`first_frame_per_hand_hand_length_pca` with z dropped."""
    N, T, D = trajs.shape
    C = D // 14
    if C != 3:
        return _tx_first_frame_per_hand_hand_length_pca(trajs, lens)
    x = trajs.reshape(N, T, 2, 7, 3)
    orig = trajs.reshape(N, T, 2, 7, 3)
    wrist0 = x[:, 0:1, :, 0:1, :]
    x = x - wrist0
    # Hand-length ref in 3D (more robust even if z is noisy it's still signal).
    ref_len = (orig[:, :, :, 1, :] - orig[:, :, :, 0, :]).norm(dim=-1)
    ref_len = ref_len.median(dim=1).values.clamp(min=1e-6)
    x = x / ref_len[:, None, :, None, None]
    w_xy = x[:, :, :, 0, :2].reshape(N, T * 2, 2)
    V, mu = _pca_basis_and_mean(w_xy, 2)
    xy = x[..., :2].contiguous()
    proj_xy = _apply_pca_to_joints(xy, V, mu, lens, 2)
    pad = torch.zeros(N, T, 2, 7, 1, device=x.device, dtype=x.dtype)
    proj = torch.cat([proj_xy, pad], dim=-1)
    return proj.reshape(N, T, D).contiguous(), lens


def _tx_first_frame_centered_pca_no_z(trajs, lens):
    """`first_frame_centered_pca` with z dropped."""
    N, T, D = trajs.shape
    C = D // 14
    if C != 3:
        return _tx_first_frame_centered_pca(trajs, lens)
    x = trajs.reshape(N, T, 2, 7, 3)
    x = x - x[:, 0:1, :, :, :]
    w_xy = x[:, :, :, 0, :2].reshape(N, T * 2, 2)
    V, mu = _pca_basis_and_mean(w_xy, 2)
    xy = x[..., :2].contiguous()
    proj_xy = _apply_pca_to_joints(xy, V, mu, lens, 2)
    pad = torch.zeros(N, T, 2, 7, 1, device=x.device, dtype=x.dtype)
    proj = torch.cat([proj_xy, pad], dim=-1)
    return proj.reshape(N, T, D).contiguous(), lens


def _tx_first_frame_per_hand_pca2d(trajs, lens):
    """Anchor per hand + 2D PCA in xy; z is preserved (anchored, not rotated).

    Middle ground between the full-3D `first_frame_per_hand_pca` and the
    z-dropped no_z variant: WiLoR's depth axis is often noisy, so
    rotating *into* it can amplify noise. This design rotates only in
    the image plane (where xy is reliable) and keeps the depth signal
    intact but expressed relative to each hand's frame-0 wrist.
    """
    N, T, D = trajs.shape
    C = D // 14
    x = trajs.reshape(N, T, 2, 7, C)
    wrist0 = x[:, 0:1, :, 0:1, :]
    x = x - wrist0
    if C == 3:
        w_xy = x[:, :, :, 0, :2].reshape(N, T * 2, 2)
        V, mu = _pca_basis_and_mean(w_xy, 2)
        xy = x[..., :2].contiguous()
        proj_xy = _apply_pca_to_joints(xy, V, mu, lens, 2)
        out = torch.cat([proj_xy, x[..., 2:3]], dim=-1)
    else:
        w = x[:, :, :, 0, :].reshape(N, T * 2, C)
        V, mu = _pca_basis_and_mean(w, C)
        out = _apply_pca_to_joints(x, V, mu, lens, C)
    return out.reshape(N, T, D).contiguous(), lens


def _tx_interjoint_distances_relative(trajs, lens):
    """Δ(interjoint distances) from frame 0 — captures articulation change.

    Same 91D pairwise-distance feature as `interjoint_distances` but each
    frame is subtracted by the clip's frame-0 distances. Inherits rigid +
    per-frame scale invariance, and additionally drops the subject's
    resting-pose baseline, so different hand sizes / camera distances
    that produce different absolute distances but identical relative
    change compare equal.
    """
    out, lens = _tx_interjoint_distances(trajs, lens)
    anchor = out[:, 0:1, :]
    return (out - anchor).contiguous(), lens


def _tx_first_frame_centered_hand_length(trajs, lens):
    """Per-joint frame-0 anchor + hand-length scale norm.

    Orthogonal cross of `first_frame_centered` and `hand_length_norm`.
    Per-joint anchoring (so each finger starts at origin) plus the
    palm-width-based scale normalization. Translation + scale invariant.
    """
    N, T, D = trajs.shape
    C = D // 14
    orig = trajs.reshape(N, T, 2, 7, C)
    x = trajs.reshape(N, T, 2, 7, C) - orig[:, 0:1, :, :, :]
    ref_len = (orig[:, :, :, 1, :] - orig[:, :, :, 0, :]).norm(dim=-1)
    ref_len = ref_len.median(dim=1).values.clamp(min=1e-6)
    x = x / ref_len[:, None, :, None, None]
    return x.reshape(N, T, D).contiguous(), lens


def _tx_wrist_centered_pca(trajs, lens):
    """`wrist_centered` (finger-relative-to-wrist) + per-clip PCA canonicalize.

    Removes each hand's global position (keeps only intra-hand pose) AND
    rotates the finger cloud into canonical axes. Captures hand
    articulation shape invariant to global hand position and global
    orientation. Throws away wrist trajectory entirely — pair with a
    wrist-path feature if whole-motion information is needed.
    """
    N, T, D = trajs.shape
    C = D // 14
    x = trajs.reshape(N, T, 2, 7, C)
    x = x - x[:, :, :, 0:1, :]
    # PCA basis from non-wrist joints (wrist row is zero after subtraction).
    pts = x[:, :, :, 1:, :].reshape(N, T * 2 * 6, C)
    V, mu = _pca_basis_and_mean(pts, C)
    proj = _apply_pca_to_joints(x, V, mu, lens, C)
    return proj.reshape(N, T, D).contiguous(), lens


def _tx_pose_hl_cat_interjoint(trajs, lens):
    """Stack scale-normalized anchored+PCA pose (42D) with interjoint distances (91D).

    Both channels are approximately unit-scaled so DTW distance weights
    are balanced across the 133D output. Captures translation + scale +
    rotation-invariant articulation AND the rotation + scale-invariant
    shape feature in one vector. The strongest "everything combined"
    candidate for cross-video retrieval.
    """
    a, _ = _tx_first_frame_per_hand_hand_length_pca(trajs, lens)
    b, _ = _tx_interjoint_distances(trajs, lens)
    return torch.cat([a, b], dim=-1).contiguous(), lens


TRAJ_TRANSFORMS = {
    "full":             _tx_full,
    "left_pos":         _tx_left_pos,
    "right_pos":        _tx_right_pos,
    "both_vel":         _tx_both_vel,
    "left_vel":         _tx_left_vel,
    "right_vel":        _tx_right_vel,
    "both_acc":         _tx_both_acc,
    "left_acc":         _tx_left_acc,
    "right_acc":        _tx_right_acc,
    "centered":         _tx_centered,
    "standardized":     _tx_standardized,
    "range_norm":       _tx_range_norm,
    "wrist_centered":   _tx_wrist_centered,
    "both_vel_unit":    _tx_both_vel_unit,
    "no_z":             _tx_no_z,
    "z_standardized":   _tx_z_standardized,
    "wrist_only":       _tx_wrist_only,
    "wrist_only_no_z":  _tx_wrist_only_no_z,
    "no_z_left":        _tx_no_z_left,
    "no_z_right":       _tx_no_z_right,
    # Per-clip/per-hand scale normalizations
    "isotropic_scale":                  _tx_isotropic_scale,
    "centered_isotropic":               _tx_centered_isotropic,
    "bbox_diag_norm":                   _tx_bbox_diag_norm,
    "path_length_norm":                 _tx_path_length_norm,
    "hand_length_norm":                 _tx_hand_length_norm,
    "wrist_centered_hand_length":       _tx_wrist_centered_hand_length,
    "wrist_centered_scaled":            _tx_wrist_centered_scaled,
    "centroid_relative":                _tx_centroid_relative,
    # Shape-only feature transforms
    "unit_velocity":                    _tx_unit_velocity,
    "interjoint_distances":             _tx_interjoint_distances,
    "turning_angle":                    _tx_turning_angle,
    # Combined no_z stacks
    "centered_no_z":                    _tx_centered_no_z,
    "wrist_centered_no_z":              _tx_wrist_centered_no_z,
    "hand_length_no_z":                 _tx_hand_length_no_z,
    "wrist_centered_hand_length_no_z":  _tx_wrist_centered_hand_length_no_z,
    # 2D-bundle variants (28D input)
    "hand_length_norm_2d":              _tx_hand_length_norm_2d,
    "wrist_centered_hand_length_2d":    _tx_wrist_centered_hand_length_2d,
    "interjoint_distances_2d":          _tx_interjoint_distances_2d,
    "centroid_relative_2d":             _tx_centroid_relative_2d,
    # First-frame-anchored transforms
    "first_frame_centered":             _tx_first_frame_centered,
    "first_frame_per_hand":             _tx_first_frame_per_hand,
    "first_frame_midpoint":             _tx_first_frame_midpoint,
    "first_frame_midpoint_hand_length":      _tx_first_frame_midpoint_hand_length,
    "first_frame_per_hand_hand_length_zdown": _tx_first_frame_per_hand_hand_length_zdown,
    "first_frame_per_hand_no_z":             _tx_first_frame_per_hand_no_z,
    "first_frame_per_hand_hand_length": _tx_first_frame_per_hand_hand_length,
    "first_frame_per_hand_hand_length_no_z": _tx_first_frame_per_hand_hand_length_no_z,
    "first_frame_midpoint_hand_length_no_z": _tx_first_frame_midpoint_hand_length_no_z,
    # Body-frame depth-grounded 21-joint designs
    "abs_21j_coords":                    _tx_body_full_pose_21j,
    "abs_wrist_coords":                  _tx_body_wrist_path_3d,
    "body_wrist_velocity_3d":            _tx_body_wrist_velocity_3d,
    "body_full_pose_21j_velocity":       _tx_body_full_pose_21j_velocity,
    "angles_21j":                        _tx_body_articulation_21j,
    "wrist_rel_21j_coords":              _tx_body_wrist_relative_pose_21j,
    "full_interjoint_dists":             _tx_body_interjoint_distances_21j,
    "wrist_rel_path":                    _tx_wrist_rel_path,
    "pca_interjoint_dists":              _tx_pca_interjoint_dists,
    "pca_interjoint_dists_w_wrist_rel_path": _tx_pca_interjoint_dists_with_wrist_path,
    "wrist_path_first_frame":           _tx_wrist_path_first_frame,
    "wrist_path_first_frame_no_z":      _tx_wrist_path_first_frame_no_z,
    "wrist_path_centered":              _tx_wrist_path_centered,
    # Per-clip PCA-canonicalized wrist paths (rigid-rotation invariant)
    "wrist_path_pca":                   _tx_wrist_path_pca,
    "wrist_path_pca_no_z":              _tx_wrist_path_pca_no_z,
    # PCA × full articulation hybrids (rotate all 14 joints)
    "full_pose_pca":                                 _tx_full_pose_pca,
    "first_frame_per_hand_pca":                      _tx_first_frame_per_hand_pca,
    "first_frame_per_hand_hand_length_pca":          _tx_first_frame_per_hand_hand_length_pca,
    "first_frame_centered_pca":                      _tx_first_frame_centered_pca,
    # Creative variants (z-dropped PCA, xy-PCA, relative interjoint, concat)
    "full_pose_pca_no_z":                            _tx_full_pose_pca_no_z,
    "first_frame_per_hand_pca_no_z":                 _tx_first_frame_per_hand_pca_no_z,
    "first_frame_per_hand_hand_length_pca_no_z":     _tx_first_frame_per_hand_hand_length_pca_no_z,
    "first_frame_centered_pca_no_z":                 _tx_first_frame_centered_pca_no_z,
    "first_frame_per_hand_pca2d":                    _tx_first_frame_per_hand_pca2d,
    "interjoint_distances_relative":                 _tx_interjoint_distances_relative,
    "first_frame_centered_hand_length":              _tx_first_frame_centered_hand_length,
    "wrist_centered_pca":                            _tx_wrist_centered_pca,
    "pose_hl_cat_interjoint":                        _tx_pose_hl_cat_interjoint,
}


# ============================================================================
# DTW design registry: (cost, norm, trajectory transform) triples
#
# A "design" is a complete DTW configuration. Each design has a unique
# (cost, transform) pair, which is the cache key for raw DTW matrices in
# run_design_grid — the 10 designs collapse to exactly 10 raw DTW
# computations regardless of encoder/pool count.
# ============================================================================

DTW_DESIGNS = {
    # Position / velocity × left / right / both hands (6)
    "both_pos":         {"cost": "euclidean_sym2", "norm": "mean_length", "tx": "full"},
    "left_pos":         {"cost": "euclidean_sym2", "norm": "mean_length", "tx": "left_pos"},
    "right_pos":        {"cost": "euclidean_sym2", "norm": "mean_length", "tx": "right_pos"},
    "both_vel":         {"cost": "euclidean_sym2", "norm": "mean_length", "tx": "both_vel"},
    "left_vel":         {"cost": "euclidean_sym2", "norm": "mean_length", "tx": "left_vel"},
    "right_vel":        {"cost": "euclidean_sym2", "norm": "mean_length", "tx": "right_vel"},

    # Depth-axis designs (WiLoR's Z is monocular-noisy; tame or drop it)
    "no_z":             {"cost": "euclidean_sym2", "norm": "mean_length", "tx": "no_z"},
    "z_standardized":   {"cost": "euclidean_sym2", "norm": "mean_length", "tx": "z_standardized"},

    # no_z ablation by hand side
    "no_z_left":        {"cost": "euclidean_sym2", "norm": "mean_length", "tx": "no_z_left"},
    "no_z_right":       {"cost": "euclidean_sym2", "norm": "mean_length", "tx": "no_z_right"},

    # Wrist-only subset + depth-dropped version
    "wrist_only":       {"cost": "euclidean_sym2", "norm": "mean_length", "tx": "wrist_only"},
    "wrist_only_no_z":  {"cost": "euclidean_sym2", "norm": "mean_length", "tx": "wrist_only_no_z"},

    # --- Additional designs kept for cost-ratio comparison
    "centered":         {"cost": "euclidean_sym2", "norm": "mean_length", "tx": "centered"},
    "standardized":     {"cost": "euclidean_sym2", "norm": "mean_length", "tx": "standardized"},
    "range_norm":       {"cost": "euclidean_sym2", "norm": "mean_length", "tx": "range_norm"},
    "wrist_centered":   {"cost": "euclidean_sym2", "norm": "mean_length", "tx": "wrist_centered"},
    "both_vel_unit":    {"cost": "euclidean_sym2", "norm": "mean_length", "tx": "both_vel_unit"},

    # --- Per-clip/per-hand scale normalizations
    "isotropic_scale":       {"cost": "euclidean_sym2", "norm": "mean_length", "tx": "isotropic_scale"},
    "centered_isotropic":    {"cost": "euclidean_sym2", "norm": "mean_length", "tx": "centered_isotropic"},
    "bbox_diag_norm":        {"cost": "euclidean_sym2", "norm": "mean_length", "tx": "bbox_diag_norm"},
    "path_length_norm":      {"cost": "euclidean_sym2", "norm": "mean_length", "tx": "path_length_norm"},
    "hand_length_norm":      {"cost": "euclidean_sym2", "norm": "mean_length", "tx": "hand_length_norm"},
    "wrist_centered_hand_length":  {"cost": "euclidean_sym2", "norm": "mean_length", "tx": "wrist_centered_hand_length"},
    "wrist_centered_scaled": {"cost": "euclidean_sym2", "norm": "mean_length", "tx": "wrist_centered_scaled"},
    "centroid_relative":     {"cost": "euclidean_sym2", "norm": "mean_length", "tx": "centroid_relative"},

    # --- Shape-only feature transforms (invariant by construction)
    "unit_velocity":          {"cost": "euclidean_sym2", "norm": "mean_length", "tx": "unit_velocity"},
    "interjoint_distances":   {"cost": "euclidean_sym2", "norm": "mean_length", "tx": "interjoint_distances"},
    "turning_angle":          {"cost": "euclidean_sym2", "norm": "mean_length", "tx": "turning_angle"},

    # --- Combined no_z stacks
    "centered_no_z":                   {"cost": "euclidean_sym2", "norm": "mean_length", "tx": "centered_no_z"},
    "wrist_centered_no_z":             {"cost": "euclidean_sym2", "norm": "mean_length", "tx": "wrist_centered_no_z"},
    "hand_length_no_z":                {"cost": "euclidean_sym2", "norm": "mean_length", "tx": "hand_length_no_z"},
    "wrist_centered_hand_length_no_z": {"cost": "euclidean_sym2", "norm": "mean_length", "tx": "wrist_centered_hand_length_no_z"},

    # --- 2D-bundle designs (use with load_data_bundle_2d())
    "hand_length_norm_2d":             {"cost": "euclidean_sym2", "norm": "mean_length", "tx": "hand_length_norm_2d"},
    "wrist_centered_hand_length_2d":   {"cost": "euclidean_sym2", "norm": "mean_length", "tx": "wrist_centered_hand_length_2d"},
    "interjoint_distances_2d":         {"cost": "euclidean_sym2", "norm": "mean_length", "tx": "interjoint_distances_2d"},
    "centroid_relative_2d":            {"cost": "euclidean_sym2", "norm": "mean_length", "tx": "centroid_relative_2d"},

    # --- First-frame-anchored designs (translation-invariant, keep path shape)
    "first_frame_centered":             {"cost": "euclidean_sym2", "norm": "mean_length", "tx": "first_frame_centered"},
    "first_frame_per_hand":             {"cost": "euclidean_sym2", "norm": "mean_length", "tx": "first_frame_per_hand"},
    "first_frame_midpoint":             {"cost": "euclidean_sym2", "norm": "mean_length", "tx": "first_frame_midpoint"},
    "first_frame_midpoint_hand_length":      {"cost": "euclidean_sym2", "norm": "mean_length", "tx": "first_frame_midpoint_hand_length"},
    "first_frame_per_hand_hand_length_zdown": {"cost": "euclidean_sym2", "norm": "mean_length", "tx": "first_frame_per_hand_hand_length_zdown"},
    "first_frame_per_hand_no_z":             {"cost": "euclidean_sym2", "norm": "mean_length", "tx": "first_frame_per_hand_no_z"},
    "first_frame_per_hand_hand_length": {"cost": "euclidean_sym2", "norm": "mean_length", "tx": "first_frame_per_hand_hand_length"},
    "first_frame_per_hand_hand_length_no_z": {"cost": "euclidean_sym2", "norm": "mean_length", "tx": "first_frame_per_hand_hand_length_no_z"},
    "first_frame_midpoint_hand_length_no_z": {"cost": "euclidean_sym2", "norm": "mean_length", "tx": "first_frame_midpoint_hand_length_no_z"},
    "abs_21j_coords":                    {"cost": "euclidean_sym2", "norm": "mean_length", "tx": "abs_21j_coords"},
    "abs_wrist_coords":                  {"cost": "euclidean_sym2", "norm": "mean_length", "tx": "abs_wrist_coords"},
    "body_wrist_velocity_3d":            {"cost": "euclidean_sym2", "norm": "mean_length", "tx": "body_wrist_velocity_3d"},
    "body_full_pose_21j_velocity":       {"cost": "euclidean_sym2", "norm": "mean_length", "tx": "body_full_pose_21j_velocity"},
    "angles_21j":                        {"cost": "euclidean_sym2", "norm": "mean_length", "tx": "angles_21j"},
    "wrist_rel_21j_coords":              {"cost": "euclidean_sym2", "norm": "mean_length", "tx": "wrist_rel_21j_coords"},
    "full_interjoint_dists":             {"cost": "euclidean_sym2", "norm": "mean_length", "tx": "full_interjoint_dists"},
    "wrist_rel_path":                    {"cost": "euclidean_sym2", "norm": "mean_length", "tx": "wrist_rel_path"},
    "pca_interjoint_dists":              {"cost": "euclidean_sym2", "norm": "mean_length", "tx": "pca_interjoint_dists"},
    "pca_interjoint_dists_w_wrist_rel_path": {"cost": "euclidean_sym2", "norm": "mean_length", "tx": "pca_interjoint_dists_w_wrist_rel_path"},
    "wrist_path_first_frame":           {"cost": "euclidean_sym2", "norm": "mean_length", "tx": "wrist_path_first_frame"},
    "wrist_path_first_frame_no_z":      {"cost": "euclidean_sym2", "norm": "mean_length", "tx": "wrist_path_first_frame_no_z"},
    "wrist_path_centered":              {"cost": "euclidean_sym2", "norm": "mean_length", "tx": "wrist_path_centered"},

    # --- PCA-canonicalized wrist paths (per-clip rigid-rotation invariant)
    "wrist_path_pca":                   {"cost": "euclidean_sym2", "norm": "mean_length", "tx": "wrist_path_pca"},
    "wrist_path_pca_no_z":              {"cost": "euclidean_sym2", "norm": "mean_length", "tx": "wrist_path_pca_no_z"},

    # --- PCA × full articulation hybrids (rotate all 14 joints, preserve finger motion)
    "full_pose_pca":                        {"cost": "euclidean_sym2", "norm": "mean_length", "tx": "full_pose_pca"},
    "first_frame_per_hand_pca":             {"cost": "euclidean_sym2", "norm": "mean_length", "tx": "first_frame_per_hand_pca"},
    "first_frame_per_hand_hand_length_pca": {"cost": "euclidean_sym2", "norm": "mean_length", "tx": "first_frame_per_hand_hand_length_pca"},
    "first_frame_centered_pca":             {"cost": "euclidean_sym2", "norm": "mean_length", "tx": "first_frame_centered_pca"},

    # --- Creative variants (z-drop PCA, xy-PCA, relative interjoint, concat)
    "full_pose_pca_no_z":                        {"cost": "euclidean_sym2", "norm": "mean_length", "tx": "full_pose_pca_no_z"},
    "first_frame_per_hand_pca_no_z":             {"cost": "euclidean_sym2", "norm": "mean_length", "tx": "first_frame_per_hand_pca_no_z"},
    "first_frame_per_hand_hand_length_pca_no_z": {"cost": "euclidean_sym2", "norm": "mean_length", "tx": "first_frame_per_hand_hand_length_pca_no_z"},
    "first_frame_centered_pca_no_z":             {"cost": "euclidean_sym2", "norm": "mean_length", "tx": "first_frame_centered_pca_no_z"},
    "first_frame_per_hand_pca2d":                {"cost": "euclidean_sym2", "norm": "mean_length", "tx": "first_frame_per_hand_pca2d"},
    "interjoint_distances_relative":             {"cost": "euclidean_sym2", "norm": "mean_length", "tx": "interjoint_distances_relative"},
    "first_frame_centered_hand_length":          {"cost": "euclidean_sym2", "norm": "mean_length", "tx": "first_frame_centered_hand_length"},
    "wrist_centered_pca":                        {"cost": "euclidean_sym2", "norm": "mean_length", "tx": "wrist_centered_pca"},
    "pose_hl_cat_interjoint":                    {"cost": "euclidean_sym2", "norm": "mean_length", "tx": "pose_hl_cat_interjoint"},
}

# Subset excluding the velocity designs.
NON_VELOCITY_DESIGNS = [k for k in DTW_DESIGNS.keys() if "_vel" not in k]

BODY_FRAME_21J_DESIGNS = [
    "abs_21j_coords",
    "abs_wrist_coords",
    "angles_21j",
    "wrist_rel_21j_coords",
    "wrist_rel_path",
    "full_interjoint_dists",
    "pca_interjoint_dists",
]

# Motion-normalization design group. Use for targeted comparisons without
# running the full design grid.
MOTION_NORM_DESIGNS = [
    # Basic motion normalizations
    "centered", "standardized", "range_norm", "wrist_centered", "both_vel_unit",
    # Per-clip/per-hand scale normalizations
    "isotropic_scale", "centered_isotropic", "bbox_diag_norm", "path_length_norm",
    "hand_length_norm", "wrist_centered_hand_length", "wrist_centered_scaled",
    "centroid_relative",
    # Shape-only feature transforms
    "unit_velocity", "interjoint_distances", "turning_angle",
    # Combined no_z stacks
    "centered_no_z", "wrist_centered_no_z", "hand_length_no_z",
    "wrist_centered_hand_length_no_z",
]

# 2D-bundle designs (require load_data_bundle_2d())
MOTION_NORM_DESIGNS_2D = [
    "hand_length_norm_2d",
    "wrist_centered_hand_length_2d",
    "interjoint_distances_2d",
    "centroid_relative_2d",
]

# Designs whose transforms touch the z-axis explicitly — incompatible with 28D
# (2D) bundles. Use `designs_compatible_with_bundle(bundle, ...)` to filter.
DESIGNS_REQUIRE_3D = frozenset([
    "no_z", "z_standardized",
    "no_z_left", "no_z_right",
    "wrist_only", "wrist_only_no_z",
    "centered_no_z", "wrist_centered_no_z",
    "hand_length_no_z", "wrist_centered_hand_length_no_z",
    "first_frame_per_hand_no_z", "wrist_path_first_frame_no_z",
    "first_frame_per_hand_hand_length_no_z",
    "wrist_path_pca_no_z",
    # Z-dropped variants
    "full_pose_pca_no_z",
    "first_frame_per_hand_pca_no_z",
    "first_frame_per_hand_hand_length_pca_no_z",
    "first_frame_centered_pca_no_z",
])


# Designs that preserve the wrist trajectory shape but are translation
# invariant ("same motion from different starting positions should match").
#
# Families:
#   A — full articulation, translation-invariant (new first-frame + legacy centered)
#   B — wrist path only, translation-invariant (articulation removed)
#   C — articulation only (wrist path removed) — for comparison
#   D — absolute / non-invariant baselines — for comparison
CENTERED_DESIGNS = [
    # A: full articulation, translation-invariant
    "first_frame_centered",
    "first_frame_per_hand",
    "first_frame_per_hand_no_z",
    "first_frame_per_hand_hand_length",
    "first_frame_per_hand_hand_length_no_z",
    "centered",
    "centered_no_z",
    "centered_isotropic",
    # B: wrist path only, translation-invariant
    "wrist_path_first_frame",
    "wrist_path_first_frame_no_z",
    "wrist_path_centered",
    # C: articulation only (no wrist path)
    "wrist_centered",
    "wrist_centered_no_z",
    "wrist_centered_hand_length",
    # D: absolute-position baselines (not translation invariant)
    "both_pos",
    "wrist_only",
    "wrist_only_no_z",
]


# Full-articulation 3D-compatible designs. All use every joint of
# every hand (7 × 2 = 14 joints), excluding wrist-only, single-hand, and
# 2D-only designs. Extended with the new PCA × articulation hybrids.
ARTICULATION_3D_DESIGNS = [
    # Position / velocity baselines (all 14 joints)
    "both_pos", "both_vel", "both_vel_unit",
    # Depth-axis variants
    "no_z", "z_standardized",
    # Whole-pose normalizations
    "centered", "standardized", "range_norm",
    "isotropic_scale", "centered_isotropic",
    "bbox_diag_norm", "path_length_norm", "hand_length_norm",
    # Wrist-relative articulation (finger positions relative to wrist)
    "wrist_centered", "wrist_centered_hand_length", "wrist_centered_scaled",
    # Centroid / shape-only
    "centroid_relative", "unit_velocity",
    "interjoint_distances", "turning_angle",
    # no-z stacks (z-dropped variants)
    "centered_no_z", "wrist_centered_no_z",
    "hand_length_no_z", "wrist_centered_hand_length_no_z",
    # First-frame anchors (full articulation)
    "first_frame_centered",
    "first_frame_per_hand", "first_frame_per_hand_no_z",
    "first_frame_per_hand_hand_length", "first_frame_per_hand_hand_length_no_z",
    # 2D-projection variants that still use all 14 joints (work on 3D bundles)
    "hand_length_norm_2d", "wrist_centered_hand_length_2d",
    "interjoint_distances_2d", "centroid_relative_2d",
    # PCA × articulation hybrids
    "full_pose_pca",
    "first_frame_per_hand_pca",
    "first_frame_per_hand_hand_length_pca",
    "first_frame_centered_pca",
]


# Curated set of articulation-focused designs, PCA hybrids, and creative
# variants.
ARTICULATION_3D_TOP_DESIGNS = [
    # Full-articulation designs
    "first_frame_centered",
    "first_frame_per_hand",
    "interjoint_distances",
    "first_frame_per_hand_hand_length",
    "both_vel",
    "first_frame_per_hand_hand_length_no_z",
    "first_frame_per_hand_no_z",
    "centered",
    "centered_no_z",
    "range_norm",
    "both_vel_unit",
    "centered_isotropic",
    # PCA × articulation hybrids
    "full_pose_pca",
    "first_frame_per_hand_pca",
    "first_frame_per_hand_hand_length_pca",
    "first_frame_centered_pca",
    # Creative variants
    "full_pose_pca_no_z",
    "first_frame_per_hand_pca_no_z",
    "first_frame_per_hand_hand_length_pca_no_z",
    "first_frame_centered_pca_no_z",
    "first_frame_per_hand_pca2d",
    "interjoint_distances_relative",
    "first_frame_centered_hand_length",
    "wrist_centered_pca",
    "pose_hl_cat_interjoint",
]


def designs_compatible_with_bundle(bundle, designs=None):
    """Filter a design list to those compatible with the bundle's trajectory dim.

    42D bundles accept everything. 28D bundles drop `DESIGNS_REQUIRE_3D`
    (transforms that index into z-axis dims). Other 2D-specific `*_2d`
    designs already work for both via `_extract_2d_xy`.
    """
    designs = list(DTW_DESIGNS.keys()) if designs is None else list(designs)
    D = bundle.trajs.shape[-1]
    if D >= 42:
        return designs
    return [d for d in designs if d not in DESIGNS_REQUIRE_3D]


# Baseline designs used for cost-ratio comparison.
BASELINE_DESIGNS = [k for k in DTW_DESIGNS.keys()
                    if k not in MOTION_NORM_DESIGNS and k not in MOTION_NORM_DESIGNS_2D]


# ============================================================================
# DTW top-k (generalized)
# ============================================================================

_KERNEL_CACHE = {}


def compute_dtw_topk(trajs_np, topk=TOPK, actual_lengths=None,
                     dtw_method="euclidean_sym2", dtw_norm="mean_length"):
    """Pairwise DTW top-k on GPU.

    Args:
        trajs_np:       [N, T, D] numpy float32 array
        topk:           number of nearest neighbors
        actual_lengths: optional Tensor [N] — pre-padding frame count per clip.
                        Required for any dtw_norm other than "none".
        dtw_method:     key into DTW_METHODS (default "euclidean_sym2")
        dtw_norm:       key into DTW_NORMS (default "mean_length")

    Returns:
        [N, topk] int64 numpy array
    """
    method = DTW_METHODS[dtw_method]
    cost_fn = method["cost_fn"]
    kernel_factory = method["kernel_fn"]
    norm_fn = DTW_NORMS[dtw_norm]

    if kernel_factory not in _KERNEL_CACHE:
        _KERNEL_CACHE[kernel_factory] = kernel_factory()
    kernel = _KERNEL_CACHE[kernel_factory]

    from numba import cuda as numba_cuda

    N, T, D = trajs_np.shape
    device = torch.device("cuda")

    all_vecs = torch.tensor(trajs_np.astype(np.float32), device=device)

    if actual_lengths is not None:
        lengths_gpu = actual_lengths.float().to(device)

    topk_indices = np.empty((N, topk), dtype=np.int64)
    n_passes = 2 * T - 1
    R_buf = torch.empty((N, T + 2, T + 2), device=device, dtype=torch.float32)

    t0 = time.time()
    with torch.no_grad():
        for i in range(N):
            D_cost = cost_fn(all_vecs[i], all_vecs)

            R_buf.fill_(float("inf"))
            R_buf[:, 0, 0] = -D_cost[:, 0, 0]

            kernel[N, T](
                numba_cuda.as_cuda_array(D_cost),
                0.0, T, T, n_passes,
                numba_cuda.as_cuda_array(R_buf),
            )

            distances = R_buf[:, -2, -2]
            distances[i] = float("inf")

            if actual_lengths is not None:
                distances = norm_fn(distances, lengths_gpu, lengths_gpu[i])

            _, topk_idx = torch.topk(distances, topk, largest=False)
            topk_indices[i] = topk_idx.cpu().numpy()

            done = i + 1
            if done % 500 == 0:
                elapsed = time.time() - t0
                rate = done / elapsed
                eta = (N - done) / rate
                print(f"    DTW [{done}/{N}]  {rate:.1f} rows/s  "
                      f"ETA {eta / 60:.1f} min", flush=True)

    print(f"    DTW done: {N} rows in {time.time() - t0:.1f}s")
    return topk_indices


def compute_dtw_topk_euclidean(trajs_np, topk=TOPK, actual_lengths=None):
    """Backward-compatible alias for compute_dtw_topk with euclidean_sym2."""
    norm = "mean_length" if actual_lengths is not None else "none"
    return compute_dtw_topk(trajs_np, topk=topk, actual_lengths=actual_lengths,
                            dtw_method="euclidean_sym2", dtw_norm=norm)


# ============================================================================
# Full pairwise DTW distance matrix
# ============================================================================

def compute_dtw_matrix(bundle, dtw_method="euclidean_sym2", dtw_norm="none",
                       device="cuda", traj_transform="full",
                       target_chunk_size=0):
    """Compute full [N, N] pairwise DTW distance matrix.

    Exploits DTW symmetry: only the upper triangle is computed on GPU,
    then mirrored to the lower triangle. This halves the number of
    kernel launches compared to the naive N×N loop.

    Args:
        bundle:         DataBundle from load_data_bundle().
        dtw_method:     key into DTW_METHODS.
        dtw_norm:       key into DTW_NORMS (default "none" for raw distances).
        device:         "cuda" or "cpu".
        traj_transform: key into TRAJ_TRANSFORMS (default "full" = identity).
                        Used to compute DTW on hand subsets (left/right) or
                        time derivatives (velocity/acceleration). The
                        transformed lengths are used for mean_length norm.
        target_chunk_size: max number of target trajectories to compare with
                        one query at a time. 0 keeps the original full-row
                        behavior.

    Returns:
        [N, N] float32 tensor. Diagonal is 0.
    """
    method = DTW_METHODS[dtw_method]
    cost_fn = method["cost_fn"]
    kernel_factory = method["kernel_fn"]
    norm_fn = DTW_NORMS[dtw_norm]
    tx_fn = TRAJ_TRANSFORMS[traj_transform]

    if kernel_factory not in _KERNEL_CACHE:
        _KERNEL_CACHE[kernel_factory] = kernel_factory()
    kernel = _KERNEL_CACHE[kernel_factory]

    import warnings
    from numba import cuda as numba_cuda
    from numba.core.errors import NumbaPerformanceWarning

    # Apply trajectory transform BEFORE moving to GPU so T' reflects the
    # post-transform length (important for velocity/acceleration variants
    # where T shrinks).
    trajs_tx, lens_tx = tx_fn(bundle.trajs, bundle.actual_lengths)
    N, T, _ = trajs_tx.shape
    all_vecs = torch.tensor(trajs_tx.numpy().astype(np.float32), device=device)
    lengths_gpu = lens_tx.float().to(device)

    dist_matrix = torch.zeros(N, N, dtype=torch.float32)
    n_passes = 2 * T - 1

    t0 = time.time()
    total_pairs = N * (N - 1) // 2
    pairs_done = 0
    with torch.no_grad(), warnings.catch_warnings():
        # Suppress low-occupancy warnings for the last few rows of the
        # symmetry loop where remaining < 32 pairs.
        warnings.simplefilter("ignore", NumbaPerformanceWarning)
        chunk_size = int(target_chunk_size) if target_chunk_size else 0
        for i in range(N - 1):
            # Only compute DTW(i, j) for j > i (upper triangle).
            remaining = N - i - 1
            step = remaining if chunk_size <= 0 else min(chunk_size, remaining)
            for start in range(0, remaining, step):
                end = min(start + step, remaining)
                abs_start = i + 1 + start
                abs_end = i + 1 + end
                targets = all_vecs[abs_start:abs_end]            # [chunk, T, D]
                D_cost = cost_fn(all_vecs[i], targets)           # [chunk, T, T]
                chunk_n = end - start

                R_buf = torch.full((chunk_n, T + 2, T + 2), float("inf"),
                                   device=device, dtype=torch.float32)
                R_buf[:, 0, 0] = -D_cost[:, 0, 0]

                kernel[chunk_n, T](
                    numba_cuda.as_cuda_array(D_cost),
                    0.0, T, T, n_passes,
                    numba_cuda.as_cuda_array(R_buf),
                )

                distances = R_buf[:, -2, -2].cpu()              # [chunk]

                if dtw_norm != "none":
                    dist_cpu = norm_fn(distances.to(device),
                                       lengths_gpu[abs_start:abs_end],
                                       lengths_gpu[i]).cpu()
                else:
                    dist_cpu = distances

                # Fill both upper and lower triangle (DTW is symmetric).
                dist_matrix[i, abs_start:abs_end] = dist_cpu
                dist_matrix[abs_start:abs_end, i] = dist_cpu

            pairs_done += remaining
            if (i + 1) % 500 == 0:
                elapsed = time.time() - t0
                frac = pairs_done / total_pairs
                eta = elapsed * (1.0 - frac) / max(frac, 1e-9)
                print(f"    DTW [{i+1}/{N}]  {frac*100:.0f}%  "
                      f"ETA {eta / 60:.1f} min", flush=True)

    print(f"    DTW matrix ({dtw_method}, tx={traj_transform}, "
          f"norm={dtw_norm}): {N}×{N} T'={T} in {time.time() - t0:.1f}s")
    return dist_matrix


# ============================================================================
# HSIC: Song et al. 2012, unbiased estimator (Eq. 5)
# ============================================================================

def hsic_unbiased(K, L):
    """Unbiased HSIC estimator (Song et al. 2012, Eq. 5).

    Motivation: HSIC (Hilbert-Schmidt Independence Criterion) measures
    statistical dependence between two kernel matrices. If K and L encode
    the same neighborhood structure, HSIC > 0. If independent, HSIC ≈ 0.
    We use it to test whether feature-space similarity (K) aligns with
    trajectory-space similarity (L).

    The "unbiased" variant (Song et al. 2012) corrects for finite-sample
    bias that makes naive HSIC positive even for independent variables.

    Args:
        K, L: [N, N] kernel/similarity matrices (diagonals will be zeroed).

    Returns:
        scalar — unbiased HSIC estimate (can be slightly negative for
                 independent inputs due to the centering correction)
    """
    m = K.shape[0]

    # Zero diagonals: self-similarity (K[i,i], L[i,i]) is always high and
    # uninformative — it would dominate HSIC if not removed.
    K_tilde = K.clone().fill_diagonal_(0)
    L_tilde = L.clone().fill_diagonal_(0)

    # term1 = Σᵢⱼ K̃ᵢⱼ · L̃ⱼᵢ  (note: L is TRANSPOSED)
    # Measures direct co-occurrence: "when i→j is strong in K, is j→i strong in L?"
    # Chunked to avoid materializing a full N×N intermediate for large N.
    chunk = 2000
    term1 = torch.tensor(0.0, device=K.device, dtype=K.dtype)
    for i in range(0, m, chunk):
        end = min(i + chunk, m)
        term1 += (K_tilde[i:end] * L_tilde.T[i:end]).sum()

    # term2 = Σ(K̃) · Σ(L̃) / ((m-1)(m-2))
    # Centering correction: the expected value of term1 if K and L were
    # independent. Subtracting it makes HSIC zero-mean under the null.
    term2 = K_tilde.sum() * L_tilde.sum() / ((m - 1) * (m - 2))

    # term3 = 2 · (col_sums_K · row_sums_L) / (m-2)
    # Hub correction: penalizes "hub" samples that are neighbors of many
    # others in both spaces. Without this, a few popular samples would
    # inflate HSIC even when K and L are independent.
    # Mathematically: 2 · 1ᵀ K̃ L̃ 1 / (m-2), computed via dot product
    # of column sums of K and row sums of L to avoid N×N matmul.
    k_col = K_tilde.sum(dim=0)   # [N] — how popular is each sample as a neighbor in K?
    l_row = L_tilde.sum(dim=1)   # [N] — how much does each sample reach out in L?
    term3 = 2.0 * (k_col @ l_row) / (m - 2)

    return (term1 + term2 - term3) / (m * (m - 3))


# ============================================================================
# Asymmetric Platonic CKNNA
# ============================================================================

def build_mask_from_topk(topk_indices, N, device):
    """Convert [N, k] top-k index array → [N, N] binary float mask.

    Motivation: The DTW phase produces top-k neighbor indices (compact [N, k]),
    but HSIC needs a full [N, N] matrix. scatter_ places 1.0 at each neighbor
    position. The resulting mask is ASYMMETRIC: mask[i,j]=1 means "j is a
    DTW neighbor of i", but j may not have i as its neighbor.
    """
    topk_idx = torch.from_numpy(topk_indices).long().to(device)
    mask = torch.zeros(N, N, dtype=torch.float32, device=device)
    mask.scatter_(1, topk_idx, 1.0)  # mask[i, topk_idx[i, :]] = 1.0
    return mask


def _build_same_video_mask(bundle, feat_valid, device):
    """Build an [M, M] binary same-video mask aligned to a feature subset.

    For a feature cache entry (feats[M, D], feat_valid[M]), each row i
    corresponds to bundle.clips[feat_valid[i]]. The returned mask has
    same_video[i, j] = 1 iff clips i and j share a video_number (diagonal
    is zeroed so self-pairs never count as same-video).

    Used by `_cknna_from_dist_matrix` when cross_video_only=True and by
    `run_per_video_split_cknna` for the same-video / cross-video split.
    """
    M = len(feat_valid)
    video_of = np.array([bundle.clips[feat_valid[i]]["video_number"] for i in range(M)])
    same = (video_of[:, None] == video_of[None, :]).astype(np.float32)
    same_t = torch.from_numpy(same).to(device)
    same_t.fill_diagonal_(0.0)
    return same_t


def compute_same_video_cost_ratio(dist_matrix, same_video_mask):
    """Mean and median DTW cost within-video vs cross-video.

    Directly diagnoses the scale confound: a "fair" DTW design has
    same-video pairs costing roughly the same as cross-video pairs, so
    `ratio ≈ 1.0`. Designs that leak framing/scale show `ratio << 1` —
    they make same-video clips abnormally cheap to align, which drives
    the same-video neighbor over-representation we see in
    `run_per_video_split_cknna`.

    Use this as a cheap pre-screen on any new DTW design: a design whose
    ratio is < 0.7 cannot possibly have low `same_video_neighbor_frac` and
    is unlikely to produce meaningful cross-masked CKNNA, regardless of
    the encoder. Rank designs by ratio before running the encoder sweep.

    Args:
        dist_matrix:     [N, N] torch.Tensor of DTW costs (symmetric, diag=0).
        same_video_mask: [N, N] torch.Tensor binary (1 = same video pair,
                         0 = cross-video), diagonal zeroed.

    Returns:
        dict with:
          mean_same:    average cost over (i, j) where same_video = 1
          mean_cross:   average cost over (i, j) where same_video = 0 and i != j
          ratio:        mean_same / mean_cross
          median_same:  median cost over same-video pairs
          median_cross: median cost over cross-video pairs
          median_ratio: median_same / median_cross
          n_same:       count of same-video pairs
          n_cross:      count of cross-video pairs
    """
    assert dist_matrix.shape == same_video_mask.shape
    N = dist_matrix.shape[0]

    dm = dist_matrix
    sv = same_video_mask.bool()
    # off-diagonal cross-video mask
    cv = (~sv) & ~torch.eye(N, dtype=torch.bool, device=dm.device)

    same_vals = dm[sv]
    cross_vals = dm[cv]

    n_same = int(same_vals.numel())
    n_cross = int(cross_vals.numel())
    if n_same == 0 or n_cross == 0:
        return {
            "mean_same": float("nan"), "mean_cross": float("nan"),
            "ratio": float("nan"),
            "median_same": float("nan"), "median_cross": float("nan"),
            "median_ratio": float("nan"),
            "n_same": n_same, "n_cross": n_cross,
        }

    mean_same = float(same_vals.mean().item())
    mean_cross = float(cross_vals.mean().item())
    median_same = float(same_vals.median().item())
    median_cross = float(cross_vals.median().item())
    return {
        "mean_same": mean_same,
        "mean_cross": mean_cross,
        "ratio": mean_same / mean_cross if mean_cross > 1e-12 else float("inf"),
        "median_same": median_same,
        "median_cross": median_cross,
        "median_ratio": median_same / median_cross if median_cross > 1e-12 else float("inf"),
        "n_same": n_same,
        "n_cross": n_cross,
    }


def compute_cosine_topk(feats_norm, topk, device):
    """L2-normalized features → cosine sim matrix + top-k binary mask.

    Motivation: For the K-side of CKNNA, we need BOTH:
      (a) K_sim — continuous cosine similarities (used as HSIC kernel values)
      (b) mask_K — binary top-k mask (used for intersection with mask_L)
    Most methods only need one or the other; Asymmetric Platonic needs both
    because it uses continuous values on the K-side but binary on the L-side.

    Returns:
        K_sim:  [N, N] cosine similarity (symmetric, continuous in [-1, 1])
        mask_K: [N, N] binary top-k mask (asymmetric: i→j ≠ j→i)
    """
    # Cosine similarity: for L2-normalized vectors, dot product = cosine sim
    K_sim = feats_norm @ feats_norm.T  # [N, N], symmetric, values in [-1, 1]

    # Find top-k most similar (excluding self: fill diagonal with -inf)
    sim_for_topk = K_sim.clone().fill_diagonal_(float("-inf"))
    _, topk_idx = torch.topk(sim_for_topk, topk, dim=1)

    # Convert to binary mask
    mask_K = torch.zeros(K_sim.shape[0], K_sim.shape[0], device=device)
    mask_K.scatter_(1, topk_idx, 1.0)
    return K_sim, mask_K


def asymmetric_platonic_cknna(K_sim, mask_K, mask_L):
    """Asymmetric Platonic CKNNA.

    Motivation: We want to measure "does the model's feature representation
    encode information about the physical trajectory?" This is a kernel
    alignment problem: does the feature-space kernel (K) align with the
    trajectory-space kernel (L)?

    Why "Asymmetric Platonic":
      - K-side uses CONTINUOUS cosine similarity — model features have
        well-calibrated cosine values, so HSIC can distinguish "very similar"
        (cos=0.95) from "barely in top-k" (cos=0.70).
      - L-side uses BINARY mask — DTW distances have no natural similarity
        scale (d=0.5 vs d=1.0 means nothing absolute), so any distance-to-
        similarity conversion (like exp(-d²)) would be arbitrary. Binary
        "neighbor or not" is cleaner.
      - INTERSECTION mask — focuses HSIC on pairs where BOTH spaces agree
        they're neighbors. Without it, non-neighbor pairs (value=0) dilute
        the signal.

    Args:
        K_sim:  [N, N] cosine similarity matrix (continuous)
        mask_K: [N, N] binary top-k mask from feature space
        mask_L: [N, N] binary top-k mask from DTW trajectory space

    Returns:
        cknna value (float, typically in [0, 0.15] for real data)
    """
    # Intersection: only pairs where BOTH spaces agree they're neighbors
    mask_inter = mask_K * mask_L  # element-wise product of two binary masks

    # Numerator: cross-space dependence
    # "Among mutually-agreed neighbor pairs, does feature similarity
    #  correlate with trajectory neighborhood membership?"
    sim_kl = hsic_unbiased(
        (mask_inter * K_sim).clone(),   # K-side: cosine values at intersection
        (mask_inter * mask_L).clone())  # L-side: binary 1s at intersection

    # Denominator: self-dependence of each space (for normalization)
    # Note: sim_kk uses mask_K (not mask_inter) — the full feature neighborhood
    sim_kk = hsic_unbiased(
        (mask_K * K_sim).clone(), (mask_K * K_sim).clone())
    # sim_ll uses mask_L — the full trajectory neighborhood
    sim_ll = hsic_unbiased(mask_L.clone(), mask_L.clone())

    # CKNNA = sim_kl / sqrt(sim_kk * sim_ll)
    # Analogous to Pearson correlation: normalize by geometric mean of self-dependences
    # Guard: if either self-HSIC is ≤ 0, the spaces are effectively independent → return 0
    if sim_kk <= 0 or sim_ll <= 0:
        return 0.0
    denom = (sim_kk * sim_ll).sqrt()
    return (sim_kl / denom).item()


# ============================================================================
# Main
# ============================================================================

def calc_cknna(num_videos, encoder_name, topk=TOPK, device="cuda",
               normalize_dtw=True, dtw_method="euclidean_sym2",
               dtw_norm=None, pool="mean"):
    """Compute CKNNA between encoder embeddings and hand keypoint trajectories.

    Args:
        num_videos:    number of videos (by ascending video_number) to load
        encoder_name:  encoder subdirectory under FEATURES_DIR,
                       or "all" to run all available encoders
        topk:          number of DTW nearest neighbors (default 10)
        device:        "cuda" or "cpu"
        normalize_dtw: if True and dtw_norm is None, use "mean_length" norm
        dtw_method:    key into DTW_METHODS (default "euclidean_sym2")
        dtw_norm:      key into DTW_NORMS; overrides normalize_dtw if given
        pool:          key into POOLING_METHODS (default "mean")

    Returns:
        If encoder_name is a single encoder: dict with cknna, mutual_knn, etc.
        If encoder_name is "all": dict mapping encoder name → result dict.
    """
    # Resolve dtw_norm from normalize_dtw if not explicitly given
    if dtw_norm is None:
        dtw_norm = "mean_length" if normalize_dtw else "none"

    bundle = load_data_bundle(num_videos)

    if encoder_name == "all":
        encoders = sorted(p.name for p in FEATURES_DIR.iterdir() if p.is_dir())
    else:
        encoders = [encoder_name]

    results_list = run_experiment_grid(
        bundle, dtw_methods=[dtw_method], dtw_norms=[dtw_norm],
        encoders=encoders, pools=[pool], topk=topk, device=device)

    if encoder_name == "all":
        results = {r["encoder"]: r for r in results_list}
        print(f"\n\n{'=' * 50}")
        print(f"Summary (dtw={dtw_method}, norm={dtw_norm}, pool={pool})")
        print(f"{'=' * 50}")
        for enc, r in sorted(results.items(), key=lambda x: x[1]["cknna"], reverse=True):
            print(f"  {enc:20s}  CKNNA={r['cknna']:.6f}  mkNN={r['mutual_knn']:.6f}  N={r['N']}")
        return results

    return results_list[0] if results_list else {
        "cknna": 0.0, "mutual_knn": 0.0, "N": 0, "T": 0, "D": 0,
        "dtw_method": dtw_method, "dtw_norm": dtw_norm, "pool": pool,
    }


# ============================================================================
# Design-grid runner
# ============================================================================

def run_experiment_grid(bundle, dtw_methods=None, dtw_norms=None,
                        encoders=None, pools=None, topk=TOPK,
                        device="cuda", return_matrices=False,
                        cross_video_only=False):
    """Run a grid of DTW method × norm × encoder × pool experiments.

    Caches raw DTW matrices per method (the expensive part) and reuses
    them across normalization variants. Encoder features are cached in
    the bundle.

    Args:
        bundle:          DataBundle from load_data_bundle().
        dtw_methods:     list of keys into DTW_METHODS (default all).
        dtw_norms:       list of keys into DTW_NORMS (default all).
        encoders:        list of encoder names (default: all on disk).
        pools:           list of keys into POOLING_METHODS (default ["mean"]).
        topk:            k for top-k neighbors (default 10).
        device:          "cuda" or "cpu".
        return_matrices: if True, include dist_matrix [N,N] in each result.

    Returns:
        List of result dicts, one per combo. Keys: dtw_method, dtw_norm,
        encoder, pool, cknna, mutual_knn, N, T, D, topk.
        If return_matrices=True, also dist_matrix.
    """
    if dtw_methods is None:
        dtw_methods = list(DTW_METHODS.keys())
    if dtw_norms is None:
        dtw_norms = list(DTW_NORMS.keys())
    if encoders is None:
        encoders = sorted(p.name for p in FEATURES_DIR.iterdir() if p.is_dir())
    if pools is None:
        pools = ["mean"]

    N = bundle.N

    # Phase 1: compute raw (unnormalized) DTW matrices — one per method
    dtw_raw_cache = {}
    for method in dtw_methods:
        label = DTW_METHODS[method]["label"]
        print(f"\nComputing DTW matrix: {label}")
        dtw_raw_cache[method] = compute_dtw_matrix(
            bundle, dtw_method=method, dtw_norm="none", device=device)

    # Phase 2: load encoder features
    for enc in encoders:
        for pool in pools:
            if (enc, pool) not in bundle._feats_cache:
                try:
                    load_encoder_feats_into(bundle, enc, pool=pool)
                except (FileNotFoundError, ValueError) as e:
                    print(f"  Skipping {enc}/{pool}: {e}")

    # Cache per-(encoder, pool) same-video masks once (each is built from the
    # encoder's own feat_valid, which can vary across encoders).
    sv_mask_cache = {}
    if cross_video_only:
        for enc_key, (_, feat_valid) in bundle._feats_cache.items():
            sv_mask_cache[enc_key] = _build_same_video_mask(
                bundle, feat_valid, device)

    # Phase 3: run CKNNA for each combo
    results = []
    for dtw_method, dtw_norm, encoder, pool in product(
            dtw_methods, dtw_norms, encoders, pools):
        if (encoder, pool) not in bundle._feats_cache:
            continue

        raw_matrix = dtw_raw_cache[dtw_method]
        lengths = bundle.actual_lengths.float()

        # Apply normalization (cheap CPU op)
        if dtw_norm == "none":
            dist_matrix = raw_matrix
        else:
            norm_fn = DTW_NORMS[dtw_norm]
            dist_matrix = torch.zeros_like(raw_matrix)
            for i in range(N):
                dist_matrix[i] = norm_fn(raw_matrix[i], lengths, lengths[i])

        row = _cknna_from_dist_matrix(
            dist_matrix, bundle, encoder, pool, topk, device,
            cross_video_only=cross_video_only,
            same_video_mask=sv_mask_cache.get((encoder, pool)))
        if row is None:
            continue
        row["dtw_method"] = dtw_method
        row["dtw_norm"] = dtw_norm
        row["T"] = bundle.T
        if return_matrices:
            row["dist_matrix"] = dist_matrix
        results.append(row)
        print(f"  {dtw_method}/{dtw_norm}/{encoder}/{pool}: "
              f"CKNNA={row['cknna']:.6f}  mkNN={row['mutual_knn']:.6f}  N={row['N']}")

    return results


# ============================================================================
# Shared helper: CKNNA from a pre-normalized distance matrix
# ============================================================================

def _cknna_from_dist_matrix(dist_matrix, bundle, encoder, pool, topk, device,
                             cross_video_only=False, same_video_mask=None):
    """Compute CKNNA given a pre-normalized [N,N] trajectory distance matrix.

    Used by both run_experiment_grid and run_design_grid. Handles the
    feature-subset case (when an encoder has features for only a subset
    of the bundle's clips).

    Args:
        cross_video_only: if True, zero out same-video pairs in both mask_L
            and mask_K post-hoc (top-k selection still happens over the full
            set). Matches the "validate / invalidate same-video leakage"
            design: by suppressing same-video intersection pairs in the
            HSIC computation, whatever CKNNA remains reflects only
            cross-video trajectory alignment.
        same_video_mask: optional pre-built [M, M] same-video mask (aligned
            to the feature subset). If None and cross_video_only=True,
            build it via _build_same_video_mask.

    Returns a result dict (without dtw_method/dtw_norm — the caller fills
    those in) or None if the encoder/pool combo isn't in the bundle cache.
    When cross_video_only=True, the result dict gains `cross_video_only`
    (bool) and `effective_L_pairs` (int) for diagnosis.
    """
    key = (encoder, pool)
    if key not in bundle._feats_cache:
        return None

    N = dist_matrix.shape[0]

    # Top-k from distance matrix → binary mask on the full N×N grid
    dist_for_topk = dist_matrix.clone()
    dist_for_topk.fill_diagonal_(float("inf"))
    _, topk_idx = torch.topk(dist_for_topk, topk, dim=1, largest=False)
    mask_L = build_mask_from_topk(topk_idx.numpy(), N, device)

    # Feature cosine similarity → K_sim, mask_K (possibly on a subset)
    feats, feat_valid = bundle._feats_cache[key]
    if len(feat_valid) < N:
        subset = sorted(set(feat_valid))
        mask_L_sub = mask_L[subset][:, subset]
        feats_sub = feats
    else:
        feats_sub = feats
        mask_L_sub = mask_L

    M = feats_sub.shape[0]
    feats_norm = F.normalize(feats_sub.float(), p=2, dim=-1).to(device)
    K_sim, mask_K = compute_cosine_topk(feats_norm, topk, device)

    mask_L_final = mask_L_sub[:M, :M].to(device) if mask_L_sub.shape[0] != M else mask_L_sub

    if cross_video_only:
        if same_video_mask is None:
            same_video_mask = _build_same_video_mask(bundle, feat_valid, device)
        cross = 1.0 - same_video_mask
        # Apply post-hoc: zero out same-video pairs in both spaces so
        # HSIC only sees cross-video neighbor agreements.
        mask_L_final = mask_L_final * cross
        mask_K = mask_K * cross

    cknna = asymmetric_platonic_cknna(K_sim, mask_K, mask_L_final)
    mknn = ((mask_K * mask_L_final).sum() / (topk * M)).item()

    result = {
        "encoder": encoder,
        "pool": pool,
        "cknna": cknna,
        "mutual_knn": mknn,
        "N": M,
        "D": feats_sub.shape[1],
        "topk": topk,
    }
    if cross_video_only:
        result["cross_video_only"] = True
        result["effective_L_pairs"] = int(mask_L_final.sum().item())
    return result


# ============================================================================
# Design grid runner — sweeps over DTW_DESIGNS × encoders × pools
# ============================================================================

def run_design_grid(bundle, designs=None, encoders=None, pools=None,
                    topk=TOPK, device="cuda", cross_video_only=False):
    """Sweep CKNNA over (DTW design × encoder × pool).

    Each "design" is a (cost, norm, trajectory transform) triple from
    DTW_DESIGNS. Raw DTW matrices depend only on (cost, transform), so
    each design is computed exactly once regardless of encoder/pool count.

    Args:
        bundle:   DataBundle from load_data_bundle().
        designs:  list of keys into DTW_DESIGNS (default: all).
        encoders: list of encoder names (default: all on disk).
        pools:    list of keys into POOLING_METHODS (default: ["mean"]).
        topk:     k for top-k neighbors (default 10).
        device:   "cuda" or "cpu".

    Returns:
        Flat list of result dicts. Each dict has:
            design, encoder, pool, cknna, mutual_knn, N, D, topk
    """
    if designs is None:
        designs = list(DTW_DESIGNS.keys())
    if encoders is None:
        encoders = sorted(p.name for p in FEATURES_DIR.iterdir() if p.is_dir())
    if pools is None:
        pools = ["mean"]

    # Phase 1: compute + normalize a distance matrix per design.
    # Note: since transforms like velocity/acceleration change T, we store
    # one [N,N] matrix per design (each already at its design's chosen norm).
    design_matrices = {}
    for d_key in designs:
        d = DTW_DESIGNS[d_key]
        print(f"\nDTW design '{d_key}': cost={d['cost']}, tx={d['tx']}, "
              f"norm={d['norm']}")
        raw = compute_dtw_matrix(
            bundle, dtw_method=d["cost"], dtw_norm="none",
            device=device, traj_transform=d["tx"])
        # Apply the design's normalization using the *transformed* lengths
        # so mean_length reflects the effective clip length.
        _, lens_tx = TRAJ_TRANSFORMS[d["tx"]](bundle.trajs, bundle.actual_lengths)
        norm_fn = DTW_NORMS[d["norm"]]
        if d["norm"] == "none":
            dist = raw
        else:
            dist = torch.zeros_like(raw)
            lens_dev = lens_tx.float().to(raw.device)
            for i in range(bundle.N):
                dist[i] = norm_fn(raw[i], lens_dev, lens_dev[i])
        design_matrices[d_key] = dist

    # Phase 2: load encoder features for each (encoder, pool)
    for enc in encoders:
        for pool in pools:
            if (enc, pool) not in bundle._feats_cache:
                try:
                    load_encoder_feats_into(bundle, enc, pool=pool)
                except (FileNotFoundError, ValueError) as e:
                    print(f"  Skipping {enc}/{pool}: {e}")

    # Cache per-(encoder, pool) same-video masks once. feat_valid can differ
    # across encoders, so the mask is keyed by (encoder, pool).
    sv_mask_cache = {}
    if cross_video_only:
        for enc_key, (_, feat_valid) in bundle._feats_cache.items():
            sv_mask_cache[enc_key] = _build_same_video_mask(
                bundle, feat_valid, device)

    # Phase 3: CKNNA for each (design, encoder, pool) combo
    mode_note = " [cross-video only]" if cross_video_only else ""
    print(f"\nRunning CKNNA{mode_note} over {len(designs)} designs × "
          f"{len(encoders)} encoders × {len(pools)} pools = "
          f"{len(designs) * len(encoders) * len(pools)} combos")
    results = []
    for d_key in designs:
        dist_matrix = design_matrices[d_key]
        for enc in encoders:
            for pool in pools:
                row = _cknna_from_dist_matrix(
                    dist_matrix, bundle, enc, pool, topk, device,
                    cross_video_only=cross_video_only,
                    same_video_mask=sv_mask_cache.get((enc, pool)))
                if row is None:
                    continue
                row["design"] = d_key
                results.append(row)
                print(f"  {d_key:16s} / {enc:14s} / {pool:19s}: "
                      f"CKNNA={row['cknna']:.6f}  "
                      f"mkNN={row['mutual_knn']:.6f}  N={row['N']}")

    return results


# ============================================================================
# Additional diagnostics built on top of run_design_grid internals
#
# These compute common diagnostic comparisons:
#   - run_shuffled_baseline     → null floor under feature-clip permutation
#   - run_topk_sweep            → CKNNA vs k for a fixed design
#   - run_inter_encoder_cknna   → encoder-vs-encoder alignment matrix
#   - run_per_video_split_cknna → same-video vs cross-video neighbor split
# ============================================================================

def _compute_design_distance_matrix(bundle, design_key, device="cuda",
                                    target_chunk_size=0):
    """Compute the normalized [N, N] DTW distance matrix for a single design.

    Factored out of run_design_grid so diagnostics can share the same
    single-design computation.
    """
    d = DTW_DESIGNS[design_key]
    raw = compute_dtw_matrix(
        bundle, dtw_method=d["cost"], dtw_norm="none",
        device=device, traj_transform=d["tx"],
        target_chunk_size=target_chunk_size)
    _, lens_tx = TRAJ_TRANSFORMS[d["tx"]](bundle.trajs, bundle.actual_lengths)
    if d["norm"] == "none":
        return raw
    norm_fn = DTW_NORMS[d["norm"]]
    dist = torch.zeros_like(raw)
    lens_dev = lens_tx.float().to(raw.device)
    for i in range(bundle.N):
        dist[i] = norm_fn(raw[i], lens_dev, lens_dev[i])
    return dist


def run_shuffled_baseline(bundle, design_key, encoder, pool="mean",
                          topk=TOPK, n_trials=10, seed=0, device="cuda",
                          cross_video_only=False):
    """Null CKNNA baseline: permute features across clips, recompute CKNNA.

    Runs the real CKNNA once, then `n_trials` random feature-clip
    permutations. Read the result as a ratio, not an absolute: at small N with
    a large top-k even random features look slightly aligned, so a real CKNNA
    of 0.30 is only meaningful relative to the measured floor.

    With cross_video_only=True, both real and shuffled CKNNA zero out
    same-video pairs from mask_L and mask_K post-hoc.

    Returns dict with: real_cknna, baseline_mean, baseline_std, ratio,
    baseline_vals (per-trial list), n_trials, N, D, topk.
    """
    dist_matrix = _compute_design_distance_matrix(bundle, design_key, device=device)
    if (encoder, pool) not in bundle._feats_cache:
        load_encoder_feats_into(bundle, encoder, pool=pool)

    sv_mask = None
    if cross_video_only:
        _, fv = bundle._feats_cache[(encoder, pool)]
        sv_mask = _build_same_video_mask(bundle, fv, device)

    real = _cknna_from_dist_matrix(
        dist_matrix, bundle, encoder, pool, topk, device,
        cross_video_only=cross_video_only, same_video_mask=sv_mask)
    if real is None:
        raise ValueError(f"No features cached for {encoder}/{pool}")

    key = (encoder, pool)
    feats, feat_valid = bundle._feats_cache[key]
    rng = np.random.default_rng(seed)
    baseline_vals = []
    for trial in range(n_trials):
        perm = rng.permutation(feats.shape[0])
        bundle._feats_cache[key] = (feats[perm].contiguous(), feat_valid)
        r = _cknna_from_dist_matrix(
            dist_matrix, bundle, encoder, pool, topk, device,
            cross_video_only=cross_video_only, same_video_mask=sv_mask)
        baseline_vals.append(float(r["cknna"]))
    bundle._feats_cache[key] = (feats, feat_valid)    # restore original mapping

    bm = float(np.mean(baseline_vals))
    bs = float(np.std(baseline_vals))
    ratio = real["cknna"] / bm if abs(bm) > 1e-9 else float("inf")
    return {
        "design": design_key,
        "encoder": encoder,
        "pool": pool,
        "real_cknna": real["cknna"],
        "baseline_mean": bm,
        "baseline_std": bs,
        "baseline_vals": baseline_vals,
        "ratio": ratio,
        "n_trials": n_trials,
        "N": real["N"],
        "D": real["D"],
        "topk": topk,
        "cross_video_only": cross_video_only,
    }


def run_topk_sweep(bundle, design_key, encoders, pool="mean",
                   ks=(5, 10, 20, 50), device="cuda", cross_video_only=False):
    """Sweep CKNNA over top-k for a fixed design, all encoders.

    Reveals whether an encoder's alignment is concentrated in the very
    nearest few neighbors (CKNNA high at k=5, drops at k=50) or spread
    across a broader shell (flat across k). The DTW distance matrix is
    computed once and reused for every k.

    With cross_video_only=True, same-video pairs are zeroed out of both
    mask_L and mask_K post-hoc — tells you whether the k=5→50 growth
    survives removing the same-video confound.

    Returns flat list of dicts: {design, encoder, pool, topk, cknna, ...}.
    """
    dist_matrix = _compute_design_distance_matrix(bundle, design_key, device=device)
    for enc in encoders:
        if (enc, pool) not in bundle._feats_cache:
            try:
                load_encoder_feats_into(bundle, enc, pool=pool)
            except (FileNotFoundError, ValueError) as e:
                print(f"  Skipping {enc}/{pool}: {e}")

    sv_mask_cache = {}
    if cross_video_only:
        for enc in encoders:
            if (enc, pool) in bundle._feats_cache:
                _, fv = bundle._feats_cache[(enc, pool)]
                sv_mask_cache[(enc, pool)] = _build_same_video_mask(bundle, fv, device)

    results = []
    for k in ks:
        for enc in encoders:
            row = _cknna_from_dist_matrix(
                dist_matrix, bundle, enc, pool, k, device,
                cross_video_only=cross_video_only,
                same_video_mask=sv_mask_cache.get((enc, pool)))
            if row is None:
                continue
            row["design"] = design_key
            results.append(row)
            print(f"  k={k:3d}  {enc:14s}: CKNNA={row['cknna']:.6f}  "
                  f"mkNN={row['mutual_knn']:.6f}")
    return results


def run_inter_encoder_cknna(bundle, encoders, pool="mean", topk=TOPK,
                             device="cuda"):
    """Cross-encoder CKNNA: each encoder's features serve as the L-side.

    For the pair (A, B), we treat B's features as the "trajectory": build
    top-k neighbors from B's cosine distances, then compute CKNNA with A
    as the K-side. Off-diagonal entries measure encoder-encoder agreement
    independently of WiLoR keypoints.

    Returns {"encoders": [...], "cknna_matrix": [E, E] np.float64,
             "N": int, "topk": int, "pool": str}.
    """
    for enc in encoders:
        if (enc, pool) not in bundle._feats_cache:
            try:
                load_encoder_feats_into(bundle, enc, pool=pool)
            except (FileNotFoundError, ValueError) as e:
                print(f"  Skipping {enc}/{pool}: {e}")

    loaded = [e for e in encoders if (e, pool) in bundle._feats_cache]

    # Intersect each encoder's valid_indices so every panel is computed
    # on exactly the same clip set (required for a fair cross-encoder
    # comparison — otherwise similarity matrices have different sizes).
    common_set = None
    for e in loaded:
        _, vi = bundle._feats_cache[(e, pool)]
        s = set(vi)
        common_set = s if common_set is None else (common_set & s)
    common = sorted(common_set or [])
    M = len(common)
    if M == 0:
        raise ValueError("No clips are present in every encoder's cache")

    feats_aligned = {}
    for e in loaded:
        f, vi = bundle._feats_cache[(e, pool)]
        pos = {idx: i for i, idx in enumerate(vi)}
        sub = f[[pos[c] for c in common]]
        feats_aligned[e] = F.normalize(sub.float(), p=2, dim=-1).to(device)

    K_all, mask_all = {}, {}
    for e in loaded:
        K, mK = compute_cosine_topk(feats_aligned[e], topk, device)
        K_all[e] = K
        mask_all[e] = mK

    n = len(loaded)
    grid = np.full((n, n), np.nan, dtype=np.float64)
    for i, e_i in enumerate(loaded):
        for j, e_j in enumerate(loaded):
            if i == j:
                grid[i, j] = 1.0
                continue
            # K-side = encoder i (continuous), L-side = encoder j (binary)
            grid[i, j] = asymmetric_platonic_cknna(
                K_all[e_i], mask_all[e_i], mask_all[e_j])

    return {
        "encoders": loaded,
        "cknna_matrix": grid,
        "N": M,
        "topk": topk,
        "pool": pool,
    }


def run_per_video_split_cknna(bundle, design_key, encoder, pool="mean",
                               topk=TOPK, device="cuda"):
    """Split CKNNA by same-video vs cross-video DTW neighbor pairs.

    Top-k DTW neighbors may include other clips from the same source
    video. We compute three CKNNA values against the same K-side:
      - full_cknna:        DTW top-k mask as-is (reference)
      - same_video_cknna:  DTW top-k mask restricted to same-video pairs
      - cross_video_cknna: DTW top-k mask restricted to cross-video pairs

    `same_video_neighbor_frac` quantifies how many raw DTW neighbors share
    a video with the query. Same-video clips share framing, scene and actor, so
    they are easy matches for both sides independently of hand motion: if
    cross_video_cknna is much lower than full_cknna, the headline number is
    driven by same-video near-duplicates rather than trajectory structure.

    Returns dict with all three CKNNAs, same_video_neighbor_frac, N,
    topk, design, encoder, pool.
    """
    dist_matrix = _compute_design_distance_matrix(bundle, design_key, device=device)
    if (encoder, pool) not in bundle._feats_cache:
        load_encoder_feats_into(bundle, encoder, pool=pool)

    key = (encoder, pool)
    feats, feat_valid = bundle._feats_cache[key]
    M = feats.shape[0]

    # Same/cross-video masks aligned to the feature subset order.
    # _build_same_video_mask zeros the diagonal; cross_video = 1 - same_video
    # therefore has 1 on the diagonal, but that's harmless because mask_L_full
    # already has 0 on the diagonal (top-k never selects self), so the product
    # below has 0 on the diagonal either way.
    same_video = _build_same_video_mask(bundle, feat_valid, device)
    cross_video = 1.0 - same_video

    # Top-k mask from the (subset-projected) distance matrix
    if len(feat_valid) < bundle.N:
        subset = sorted(set(feat_valid))
        dist_sub = dist_matrix[subset][:, subset]
    else:
        dist_sub = dist_matrix
    dist_for_topk = dist_sub.clone()
    dist_for_topk.fill_diagonal_(float("inf"))
    _, topk_idx = torch.topk(dist_for_topk, topk, dim=1, largest=False)
    mask_L_full = build_mask_from_topk(topk_idx.numpy(), M, device)

    feats_norm = F.normalize(feats.float(), p=2, dim=-1).to(device)
    K_sim, mask_K = compute_cosine_topk(feats_norm, topk, device)

    # Apply the same/cross masks to BOTH the L-side (DTW neighbors) and the
    # K-side (feature neighbors). Masking only one side leaves its
    # corresponding HSIC self-term (sim_kk or sim_ll) un-shrunk, so the
    # CKNNA denominator mixes a full-space self-dependence with a
    # masked intersection — cleaner to mask both sides symmetrically.
    # This matches `_cknna_from_dist_matrix(cross_video_only=True)` so
    # the two code paths produce the same cross_video CKNNA for a given
    # (design, encoder, pool).
    full = asymmetric_platonic_cknna(K_sim, mask_K, mask_L_full)
    same = asymmetric_platonic_cknna(K_sim, mask_K * same_video, mask_L_full * same_video)
    cross = asymmetric_platonic_cknna(K_sim, mask_K * cross_video, mask_L_full * cross_video)

    total_L = mask_L_full.sum()
    same_frac = float((mask_L_full * same_video).sum() / total_L) if total_L > 0 else 0.0

    return {
        "design": design_key,
        "encoder": encoder,
        "pool": pool,
        "full_cknna": float(full),
        "same_video_cknna": float(same),
        "cross_video_cknna": float(cross),
        "same_video_neighbor_frac": same_frac,
        "N": M,
        "topk": topk,
    }


def run_trajectory_self_cknna(bundle, designs=None, topk=TOPK, device="cuda",
                               cross_video_only=False):
    """Trajectory-self CKNNA: use the trajectory itself as the K-side.

    For each design, the L-side is the usual DTW top-k from the transformed
    trajectory, and the K-side is that same transformed trajectory
    flattened per clip and L2-normalized. This gives an upper bound on
    what any encoder could possibly score against the DTW top-k on this
    trajectory representation — if the trajectory-self score is itself
    only ~0.30, the encoders aren't leaving much on the table.

    We reuse `_cknna_from_dist_matrix` by stuffing a pseudo-entry into
    `bundle._feats_cache` keyed `("trajectory_self", design_key)`. After
    this function returns, the pseudo-entries remain in the cache
    (harmless — they just sit alongside real encoder entries).

    Args:
        bundle: DataBundle.
        designs: list of DTW_DESIGNS keys (default: all).
        topk: top-k for CKNNA.
        device: "cuda" or "cpu".
        cross_video_only: mask same-video pairs in both mask_L and mask_K.

    Returns:
        list of result dicts: {design, cknna, mutual_knn, N, D, topk,
        encoder='trajectory_self', pool=design_key, ...}.
    """
    if designs is None:
        designs = list(DTW_DESIGNS.keys())

    results = []
    for d_key in designs:
        d = DTW_DESIGNS[d_key]
        # L-side: normalized DTW distance matrix for this design
        dist_matrix = _compute_design_distance_matrix(bundle, d_key, device=device)

        # K-side: transform the trajectory, flatten the full padded [T_tx, D_tx]
        # to [T_tx * D_tx] per clip, treat as an "encoder feature".
        trajs_tx, _ = TRAJ_TRANSFORMS[d["tx"]](bundle.trajs, bundle.actual_lengths)
        # trajs_tx: [N, T_tx, D_tx]  →  flatten to [N, T_tx*D_tx]
        feats_flat = trajs_tx.reshape(trajs_tx.shape[0], -1).contiguous()
        # feat_valid spans every bundle clip
        feat_valid = list(range(bundle.N))
        pseudo_key = ("trajectory_self", d_key)
        bundle._feats_cache[pseudo_key] = (feats_flat, feat_valid)

        sv_mask = None
        if cross_video_only:
            sv_mask = _build_same_video_mask(bundle, feat_valid, device)

        row = _cknna_from_dist_matrix(
            dist_matrix, bundle, "trajectory_self", d_key, topk, device,
            cross_video_only=cross_video_only, same_video_mask=sv_mask)
        if row is None:
            continue
        row["design"] = d_key
        results.append(row)
        tag = " [cross-video]" if cross_video_only else ""
        print(f"  {d_key:16s} / trajectory_self{tag}: "
              f"CKNNA={row['cknna']:.6f}  mkNN={row['mutual_knn']:.6f}  "
              f"D={row['D']}")

    return results


# ============================================================================
# Distance-matrix comparison helpers
# ============================================================================

def neighbor_overlap(mat_a, mat_b, topk):
    """Mean fraction of shared top-k neighbors between two [N,N] distance matrices.

    Returns a float in [0, 1]. 1.0 means identical neighbor sets.
    """
    N = mat_a.shape[0]
    a_inf = mat_a.clone().fill_diagonal_(float("inf"))
    b_inf = mat_b.clone().fill_diagonal_(float("inf"))
    _, topk_a = torch.topk(a_inf, topk, dim=1, largest=False)
    _, topk_b = torch.topk(b_inf, topk, dim=1, largest=False)
    overlaps = 0.0
    for i in range(N):
        set_a = set(topk_a[i].tolist())
        set_b = set(topk_b[i].tolist())
        overlaps += len(set_a & set_b) / topk
    return overlaps / N


def rank_correlation(mat_a, mat_b):
    """Spearman rank correlation between upper triangles of two [N,N] matrices."""
    from scipy.stats import spearmanr
    N = mat_a.shape[0]
    idx = torch.triu_indices(N, N, offset=1)
    a_vals = mat_a[idx[0], idx[1]].numpy()
    b_vals = mat_b[idx[0], idx[1]].numpy()
    corr, _ = spearmanr(a_vals, b_vals)
    return float(corr)


def compare_distance_distributions(matrices_dict, percentiles=None):
    """Print percentile table for multiple named [N,N] distance matrices.

    Args:
        matrices_dict: {"label": Tensor [N,N], ...}
        percentiles: list of percentiles (default [10, 25, 50, 75, 90, 99])
    """
    if percentiles is None:
        percentiles = [10, 25, 50, 75, 90, 99]

    header = f"{'Method':30s}" + "".join(f"  p{p:<5d}" for p in percentiles)
    print(header)
    print("-" * len(header))
    for label, mat in matrices_dict.items():
        N = mat.shape[0]
        idx = torch.triu_indices(N, N, offset=1)
        vals = mat[idx[0], idx[1]]
        pcts = np.percentile(vals.numpy(), percentiles)
        row = f"{label:30s}" + "".join(f"  {v:<7.2f}" for v in pcts)
        print(row)


# ============================================================================
# Plotting
# ============================================================================

def plot_cknna_results(results, results_b=None, label_a="Raw DTW", label_b="Length-normalized DTW"):
    """Bar chart comparing CKNNA and mutual k-NN across encoders.

    Args:
        results:   dict mapping encoder_name → {"cknna": float, "mutual_knn": float, ...}
                   (as returned by calc_cknna with encoder_name="all")
        results_b: optional second results dict for side-by-side comparison.
        label_a:   legend label for `results` (default "Raw DTW").
        label_b:   legend label for `results_b` (default "Length-normalized DTW").
    """
    import matplotlib.pyplot as plt

    if results_b is None:
        # --- Single results mode (original behavior) ---
        sorted_names = sorted(results, key=lambda k: results[k]["cknna"], reverse=True)
        cknna_vals = [results[n]["cknna"] for n in sorted_names]
        mknn_vals = [results[n]["mutual_knn"] for n in sorted_names]
        dims = [results[n]["D"] for n in sorted_names]
        labels = [f"{n}\n(D={d})" for n, d in zip(sorted_names, dims)]

        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        bars1 = axes[0].bar(labels, cknna_vals, color="#4C72B0", edgecolor="white")
        axes[0].set_ylabel("CKNNA")
        axes[0].set_title("Asymmetric Platonic CKNNA")
        for bar, val in zip(bars1, cknna_vals):
            axes[0].text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.002,
                         f"{val:.4f}", ha="center", va="bottom", fontsize=9)

        bars2 = axes[1].bar(labels, mknn_vals, color="#DD8452", edgecolor="white")
        axes[1].set_ylabel("Mutual k-NN overlap")
        axes[1].set_title("Mutual k-NN")
        for bar, val in zip(bars2, mknn_vals):
            axes[1].text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.002,
                         f"{val:.4f}", ha="center", va="bottom", fontsize=9)

        N = next(iter(results.values()))["N"]
        k = next(iter(results.values())).get("k", TOPK)
        fig.suptitle(f"Encoder Feature–Trajectory Alignment  (N={N}, k={k})", fontsize=13)
        plt.tight_layout()
        plt.show()
        return

    # --- Comparison mode: two result sets side by side ---
    all_encoders = sorted(set(results) | set(results_b))
    # Sort by average CKNNA across both sets (descending)
    def avg_cknna(enc):
        vals = []
        if enc in results:
            vals.append(results[enc]["cknna"])
        if enc in results_b:
            vals.append(results_b[enc]["cknna"])
        return sum(v for v in vals if v == v) / max(len(vals), 1)  # skip nan
    sorted_names = sorted(all_encoders, key=avg_cknna, reverse=True)

    dims = []
    for n in sorted_names:
        d = results.get(n, results_b.get(n, {})).get("D", "?")
        dims.append(d)
    labels = [f"{n}\n(D={d})" for n, d in zip(sorted_names, dims)]

    import numpy as _np
    x = _np.arange(len(sorted_names))
    width = 0.35

    fig, axes = plt.subplots(1, 2, figsize=(15, 5.5))

    # CKNNA comparison
    vals_a = [results.get(n, {}).get("cknna", 0) for n in sorted_names]
    vals_b = [results_b.get(n, {}).get("cknna", 0) for n in sorted_names]
    bars_a = axes[0].bar(x - width / 2, vals_a, width, label=label_a, color="#4C72B0", edgecolor="white")
    bars_b = axes[0].bar(x + width / 2, vals_b, width, label=label_b, color="#8CB4E0", edgecolor="white")
    axes[0].set_ylabel("CKNNA")
    axes[0].set_title("Asymmetric Platonic CKNNA")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(labels)
    axes[0].legend()
    for bar, val in zip(bars_a, vals_a):
        if val == val:  # skip nan
            axes[0].text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.001,
                         f"{val:.4f}", ha="center", va="bottom", fontsize=8)
    for bar, val in zip(bars_b, vals_b):
        if val == val:
            axes[0].text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.001,
                         f"{val:.4f}", ha="center", va="bottom", fontsize=8)

    # Mutual k-NN comparison
    mknn_a = [results.get(n, {}).get("mutual_knn", 0) for n in sorted_names]
    mknn_b = [results_b.get(n, {}).get("mutual_knn", 0) for n in sorted_names]
    bars_a2 = axes[1].bar(x - width / 2, mknn_a, width, label=label_a, color="#DD8452", edgecolor="white")
    bars_b2 = axes[1].bar(x + width / 2, mknn_b, width, label=label_b, color="#F0C8A0", edgecolor="white")
    axes[1].set_ylabel("Mutual k-NN overlap")
    axes[1].set_title("Mutual k-NN")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(labels)
    axes[1].legend()
    for bar, val in zip(bars_a2, mknn_a):
        axes[1].text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.001,
                     f"{val:.4f}", ha="center", va="bottom", fontsize=8)
    for bar, val in zip(bars_b2, mknn_b):
        axes[1].text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.001,
                     f"{val:.4f}", ha="center", va="bottom", fontsize=8)

    N = next(iter(results.values()))["N"]
    k = next(iter(results.values())).get("k", TOPK)
    fig.suptitle(f"Encoder Feature–Trajectory Alignment  (N={N}, k={k})", fontsize=13)
    plt.tight_layout()
    plt.show()


def plot_design_pool_heatmap(results, figsize=None, cmap="viridis",
                             design_order=None, pool_order=None,
                             save_path=None):
    """Heatmap grid: one panel per encoder, (DTW design × pooling method).

    Args:
        results:      flat list of rows from run_design_grid. Each row has
                      keys: design, encoder, pool, cknna, mutual_knn, N, D.
        figsize:      optional (W, H) tuple. Default scales with grid size.
        cmap:         matplotlib colormap (default "viridis").
        design_order: optional list specifying row order. Defaults to
                      DTW_DESIGNS insertion order.
        pool_order:   optional list specifying column order. Defaults to
                      POOLING_METHODS insertion order.
        save_path:    optional path (str or Path). If given, the figure is
                      saved there (parent dirs created as needed) before
                      being shown.

    Also prints a summary table of the best (design, pool) per encoder.
    """
    import matplotlib.pyplot as plt
    from matplotlib.colors import Normalize

    if design_order is None:
        seen = []
        for r in results:
            if r["design"] not in seen:
                seen.append(r["design"])
        design_order = [k for k in DTW_DESIGNS.keys() if k in seen] or seen

    if pool_order is None:
        seen = []
        for r in results:
            if r["pool"] not in seen:
                seen.append(r["pool"])
        pool_order = [k for k in POOLING_METHODS.keys() if k in seen] or seen

    # Group rows by encoder
    encoders = []
    by_enc = {}
    for r in results:
        enc = r["encoder"]
        if enc not in by_enc:
            by_enc[enc] = {}
            encoders.append(enc)
        by_enc[enc][(r["design"], r["pool"])] = r

    n_enc = len(encoders)
    n_rows = len(design_order)
    n_cols = len(pool_order)

    # Build a [n_rows, n_cols] grid per encoder, collect global min/max
    grids = {}
    vmin = float("inf")
    vmax = float("-inf")
    for enc in encoders:
        grid = np.full((n_rows, n_cols), np.nan, dtype=np.float64)
        for i, d in enumerate(design_order):
            for j, p in enumerate(pool_order):
                r = by_enc[enc].get((d, p))
                if r is not None:
                    grid[i, j] = r["cknna"]
        grids[enc] = grid
        finite = grid[np.isfinite(grid)]
        if finite.size:
            vmin = min(vmin, finite.min())
            vmax = max(vmax, finite.max())

    if not np.isfinite(vmin) or not np.isfinite(vmax):
        print("No valid CKNNA values in results.")
        return
    if vmin == vmax:
        vmax = vmin + 1e-6

    # Subplot grid: 2 cols for ≤6 encoders, 3 for up to 9, etc.
    ncols_fig = 2 if n_enc <= 4 else 3
    nrows_fig = (n_enc + ncols_fig - 1) // ncols_fig

    if figsize is None:
        # Scale with design/pool count. Each panel ≈ 0.55 * n_cols wide
        # and 0.4 * n_rows tall, plus label padding.
        panel_w = max(4.0, 0.7 * n_cols + 2.5)
        panel_h = max(3.5, 0.45 * n_rows + 2.0)
        figsize = (panel_w * ncols_fig, panel_h * nrows_fig)

    fig, axes = plt.subplots(nrows_fig, ncols_fig, figsize=figsize,
                             squeeze=False)
    norm = Normalize(vmin=vmin, vmax=vmax)

    for ax_idx, enc in enumerate(encoders):
        r_i, c_i = divmod(ax_idx, ncols_fig)
        ax = axes[r_i][c_i]
        grid = grids[enc]
        im = ax.imshow(grid, cmap=cmap, norm=norm, aspect="auto")

        # Axis ticks / labels
        ax.set_xticks(range(n_cols))
        ax.set_xticklabels(pool_order, rotation=45, ha="right", fontsize=8)
        ax.set_yticks(range(n_rows))
        ax.set_yticklabels(design_order, fontsize=8)

        # Title with encoder name + best cell highlighted (skip if all-NaN)
        has_any = bool(np.any(np.isfinite(grid)))
        if has_any:
            best_idx = np.unravel_index(np.nanargmax(grid), grid.shape)
            best_val = grid[best_idx]
            best_d = design_order[best_idx[0]]
            best_p = pool_order[best_idx[1]]
            ax.set_title(
                f"{enc}\nbest: {best_d} × {best_p} = {best_val:.4f}",
                fontsize=10,
            )
        else:
            ax.set_title(f"{enc}\n(all NaN)", fontsize=10)

        # Annotate each cell with its CKNNA value
        for i in range(n_rows):
            for j in range(n_cols):
                v = grid[i, j]
                if not np.isfinite(v):
                    continue
                # Choose text color based on cell brightness
                rel = (v - vmin) / (vmax - vmin)
                txt_color = "white" if rel < 0.5 else "black"
                ax.text(j, i, f"{v:.3f}", ha="center", va="center",
                        fontsize=6, color=txt_color)

        # Outline the best cell (only if the grid has any finite values)
        if has_any:
            from matplotlib.patches import Rectangle
            ax.add_patch(Rectangle(
                (best_idx[1] - 0.5, best_idx[0] - 0.5), 1, 1,
                fill=False, edgecolor="red", linewidth=1.5,
            ))

    # Hide unused axes
    for ax_idx in range(n_enc, nrows_fig * ncols_fig):
        r_i, c_i = divmod(ax_idx, ncols_fig)
        axes[r_i][c_i].axis("off")

    # Shared colorbar
    cbar = fig.colorbar(im, ax=axes, shrink=0.8, pad=0.02, label="CKNNA")

    N_any = next(iter(results))["N"]
    fig.suptitle(
        f"DTW Design × Pooling Method CKNNA Sweep  (N={N_any} clips)",
        fontsize=14, y=0.995,
    )

    if save_path is not None:
        from pathlib import Path as _Path
        save_path = _Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved heatmap to {save_path}")

    plt.show()

    # Print per-encoder best combo summary
    print(f"\n{'Encoder':15s}  {'Best design':18s}  {'Best pool':20s}  CKNNA")
    print("-" * 70)
    for enc in encoders:
        grid = grids[enc]
        if not np.any(np.isfinite(grid)):
            print(f"  {enc:13s}  (all NaN — no finite CKNNA values)")
            continue
        best_idx = np.unravel_index(np.nanargmax(grid), grid.shape)
        best_val = grid[best_idx]
        print(f"  {enc:13s}  {design_order[best_idx[0]]:18s}  "
              f"{pool_order[best_idx[1]]:20s}  {best_val:.6f}")

    return fig


# ============================================================================
# Perturbation-based design diagnostics
#
# These helpers perturb trajectories in well-defined ways, recompute the
# per-design DTW distance matrix, and measure how much each design's neighbor
# structure is preserved or disturbed. Every helper treats the existing
# `_compute_design_distance_matrix` as a black box, so the full DTW_DESIGNS
# registry is automatically supported.
# ============================================================================


def _clone_bundle(bundle, new_trajs, new_lengths=None):
    """Return a shallow DataBundle clone with a replaced trajs tensor.

    Shares `clips` by reference (read-only in the DTW path). `_feats_cache`
    is intentionally *not* reused — perturbing trajectories does not
    perturb encoder features, so the cache would be misleading.
    """
    if new_lengths is None:
        new_lengths = bundle.actual_lengths
    return DataBundle(
        clips=bundle.clips,
        trajs=new_trajs,
        actual_lengths=new_lengths,
        N=new_trajs.shape[0],
        T=new_trajs.shape[1],
    )


def _coord_stride(trajs):
    """Return 3 for 42D (3D) bundles and 2 for 28D (2D) bundles."""
    D = trajs.shape[-1]
    if D % 42 == 0 or D == 42:
        return 3
    if D == 28:
        return 2
    # Fall back to 3 if we hit an unfamiliar layout.
    return 3 if D % 3 == 0 else 2


def _per_clip_bbox_diag(trajs, lengths):
    """Per-clip bbox diagonal over the valid (unpadded) portion of each clip.

    Returns a `[N]` float tensor. Used as the σ reference for Gaussian
    keypoint noise so a "5% noise" perturbation is meaningful at any scale.
    """
    N, T, D = trajs.shape
    stride = _coord_stride(trajs)
    n_joints = D // stride                     # 14 (2D) or 14 (3D, both halves)
    out = torch.zeros(N, dtype=trajs.dtype)
    for i in range(N):
        L = int(lengths[i].item()) if torch.is_tensor(lengths) else int(lengths[i])
        L = max(L, 1)
        pts = trajs[i, :L].reshape(L * n_joints, stride)
        lo = pts.min(dim=0).values
        hi = pts.max(dim=0).values
        out[i] = (hi - lo).norm().item() or 1.0
    return out


def _rand_rotation_matrix(dim, rng):
    """Random orthogonal matrix with det=+1 via QR of a Gaussian matrix."""
    A = rng.standard_normal((dim, dim)).astype(np.float32)
    Q, R = np.linalg.qr(A)
    # Enforce det=+1 to exclude reflections.
    d = np.sign(np.diag(R))
    d[d == 0] = 1.0
    Q = Q * d
    if np.linalg.det(Q) < 0:
        Q[:, 0] = -Q[:, 0]
    return Q


def perturb_bundle(bundle, kind, seed=0, **kwargs):
    """Return a DataBundle with perturbed trajectories.

    kind ∈ {
      "gauss_noise"       kwargs: sigma_frac (default 0.03) — iid Gaussian
                          noise, σ = sigma_frac × per-clip bbox diag, on
                          every keypoint coordinate.
      "frame_dropout"     kwargs: p (default 0.25) — mark p fraction of
                          valid frames as "dropped" and forward-fill from
                          the previous surviving frame (first frame always
                          kept). Actual lengths unchanged.
      "time_warp"         kwargs: factor (default 1.25) — linearly resample
                          the valid portion of each clip to factor × L
                          frames (capped at T), re-pad with last frame.
                          actual_lengths updated.
      "global_shift"      kwargs: scale (default 1.0) — add a per-clip
                          random translation of magnitude `scale ×
                          bbox_diag` to every joint coord uniformly.
      "global_scale"      kwargs: s (default 1.5) — multiply the whole
                          trajectory by s (one scalar per clip or a
                          single scalar shared across clips).
      "global_rotate"     — rotate each clip's joint coordinates by a random
                          orthogonal matrix (SO(3) for 3D, SO(2) for 2D).
                          Rotation is per-clip, different across clips.
      "mirror_x"          — negate x coordinates (stride 0). Preserves
                          geometry up to chirality.
      "lr_swap"           — swap the L and R halves of the 42D/28D vector.
      "per_frame_shuffle" — for each frame independently, shuffle the 7
                          joint indices within each hand. Breaks spatial
                          topology; preserves per-frame hand bounding box.
      "reverse_time"      — reverse the valid portion of each clip in time.
                          actual_lengths unchanged.
    }

    Returns a cloned DataBundle. Deterministic given `seed`.
    """
    rng = np.random.default_rng(seed)
    trajs = bundle.trajs.clone()
    lengths = bundle.actual_lengths.clone()
    N, T, D = trajs.shape
    stride = _coord_stride(trajs)
    half = D // 2
    n_joints_per_hand = half // stride

    if kind == "gauss_noise":
        sigma_frac = kwargs.get("sigma_frac", 0.03)
        diag = _per_clip_bbox_diag(trajs, lengths)
        noise = torch.from_numpy(rng.standard_normal(trajs.shape).astype(np.float32))
        trajs = trajs + noise * (sigma_frac * diag.view(N, 1, 1))
        return _clone_bundle(bundle, trajs)

    if kind == "frame_dropout":
        p = kwargs.get("p", 0.25)
        for i in range(N):
            L = int(lengths[i].item())
            if L < 2:
                continue
            keep = rng.random(L) > p
            keep[0] = True
            last_valid = 0
            for t in range(L):
                if keep[t]:
                    last_valid = t
                else:
                    trajs[i, t] = trajs[i, last_valid]
        return _clone_bundle(bundle, trajs)

    if kind == "time_warp":
        factor = kwargs.get("factor", 1.25)
        new_trajs = trajs.clone()
        new_lengths = lengths.clone()
        for i in range(N):
            L = int(lengths[i].item())
            if L < 2:
                continue
            new_L = max(2, min(T, int(round(L * factor))))
            # Linear interpolation along the valid portion.
            src = trajs[i, :L]                                      # [L, D]
            # Index fractional positions 0..L-1 spread over new_L
            idx = torch.linspace(0, L - 1, new_L)
            lo = idx.floor().clamp(max=L - 1).long()
            hi = (lo + 1).clamp(max=L - 1)
            frac = (idx - lo.float()).unsqueeze(-1)                  # [new_L, 1]
            resampled = src[lo] * (1 - frac) + src[hi] * frac        # [new_L, D]
            new_trajs[i, :new_L] = resampled
            if new_L < T:
                new_trajs[i, new_L:] = resampled[-1]
            new_lengths[i] = new_L
        return _clone_bundle(bundle, new_trajs, new_lengths)

    if kind == "global_shift":
        scale = kwargs.get("scale", 1.0)
        diag = _per_clip_bbox_diag(trajs, lengths)
        shifts = torch.from_numpy(rng.standard_normal((N, stride)).astype(np.float32))
        shifts = shifts * (scale * diag.view(N, 1))                  # [N, stride]
        # Broadcast over (T, joints_per_hand, 2 hands) for both halves.
        # trajs layout: [N, T, D]; D = 2*half, half = n_joints_per_hand*stride
        sh = shifts.view(N, 1, 1, stride).expand(N, T, 2 * n_joints_per_hand, stride)
        trajs = trajs + sh.reshape(N, T, D)
        return _clone_bundle(bundle, trajs)

    if kind == "global_scale":
        s = kwargs.get("s", 1.5)
        trajs = trajs * float(s)
        return _clone_bundle(bundle, trajs)

    if kind == "global_rotate":
        new_trajs = trajs.clone()
        for i in range(N):
            R = _rand_rotation_matrix(stride, rng)                    # [stride, stride]
            R_t = torch.from_numpy(R)
            # Apply to every joint across both hands.
            pts = trajs[i].reshape(T, 2 * n_joints_per_hand, stride)  # [T, J, stride]
            rotated = pts @ R_t.T                                     # [T, J, stride]
            new_trajs[i] = rotated.reshape(T, D)
        return _clone_bundle(bundle, new_trajs)

    if kind == "mirror_x":
        new_trajs = trajs.clone()
        pts = new_trajs.reshape(N, T, 2 * n_joints_per_hand, stride)
        pts[..., 0] = -pts[..., 0]
        return _clone_bundle(bundle, pts.reshape(N, T, D))

    if kind == "lr_swap":
        left = trajs[..., :half].clone()
        right = trajs[..., half:].clone()
        new_trajs = torch.cat([right, left], dim=-1)
        return _clone_bundle(bundle, new_trajs)

    if kind == "per_frame_shuffle":
        new_trajs = trajs.clone()
        # Work on a view: [N, T, 2 hands, n_joints_per_hand, stride]
        view = new_trajs.reshape(N, T, 2, n_joints_per_hand, stride)
        for i in range(N):
            L = int(lengths[i].item())
            for t in range(L):
                for h in range(2):
                    perm = rng.permutation(n_joints_per_hand)
                    view[i, t, h] = view[i, t, h, perm]
        return _clone_bundle(bundle, view.reshape(N, T, D))

    if kind == "reverse_time":
        new_trajs = trajs.clone()
        for i in range(N):
            L = int(lengths[i].item())
            if L < 2:
                continue
            new_trajs[i, :L] = trajs[i, :L].flip(0)
        return _clone_bundle(bundle, new_trajs)

    raise ValueError(f"Unknown perturbation kind: {kind!r}")


def design_distance_matrices(bundle, designs, device="cuda", verbose=False,
                             target_chunk_size=0):
    """Compute (and cache-in-dict) per-design [N,N] distance matrices.

    Returns `{design_key: Tensor [N,N] on CPU}`. Designs that raise are
    logged and skipped (the caller sees a missing key). No side effects
    on the bundle.
    """
    out = {}
    designs = list(designs)
    for i, d_key in enumerate(designs):
        try:
            mat = _compute_design_distance_matrix(
                bundle, d_key, device=device,
                target_chunk_size=target_chunk_size,
            ).detach().cpu()
            out[d_key] = mat
            if verbose:
                print(f"  [{i+1}/{len(designs)}] {d_key}: ok")
        except Exception as e:
            print(f"  [{i+1}/{len(designs)}] {d_key}: FAILED {type(e).__name__}: {e}")
    return out


def preservation_score(mat_ref, mat_pert, topk=TOPK):
    """How similar are two [N,N] distance matrices' neighbor structures?

    Returns `{"jaccard": float, "spearman": float}`. Higher = more
    preserved. 1.0 jaccard = identical neighbor sets; 1.0 spearman =
    identical rankings over the upper triangle.
    """
    return {
        "jaccard": neighbor_overlap(mat_ref, mat_pert, topk),
        "spearman": rank_correlation(mat_ref, mat_pert),
    }


def temporal_coherence(bundle, mat, min_same_video=3):
    """Spearman(DTW distance, |time diff|) within each source video.

    For each video with ≥ `min_same_video` clips, compute the Spearman
    correlation between (i, j) DTW distances and |start_sec_i −
    start_sec_j| over all intra-video pairs. Return the mean across
    videos. Videos without `start_sec` (cache rows) fall back to
    `node_number` as an ordinal proxy.

    A high value means adjacent clips within a video are ranked as
    nearest neighbors — a free, label-less ground-truth signal.
    """
    from collections import defaultdict
    from scipy.stats import spearmanr

    groups = defaultdict(list)
    for idx, clip in enumerate(bundle.clips):
        groups[clip["video_number"]].append(idx)

    mat_np = mat.cpu().numpy() if torch.is_tensor(mat) else np.asarray(mat)
    corrs = []
    for vn, idxs in groups.items():
        if len(idxs) < min_same_video:
            continue
        # Use start_sec if present, else fall back to node_number.
        times = []
        for j in idxs:
            clip = bundle.clips[j]
            if "start_sec" in clip and clip["start_sec"] is not None:
                times.append(float(clip["start_sec"]))
            else:
                times.append(float(clip.get("node_number", 0)))
        times = np.asarray(times)
        dtw_pairs = []
        time_pairs = []
        for a in range(len(idxs)):
            for b in range(a + 1, len(idxs)):
                dtw_pairs.append(mat_np[idxs[a], idxs[b]])
                time_pairs.append(abs(times[a] - times[b]))
        if len(set(dtw_pairs)) < 2 or len(set(time_pairs)) < 2:
            continue
        c, _ = spearmanr(dtw_pairs, time_pairs)
        if c == c:  # skip NaN
            corrs.append(float(c))
    if not corrs:
        return float("nan")
    return float(np.mean(corrs))


# ============================================================================
# Cross-video retrieval helpers
# ============================================================================

def _video_of_clip(bundle):
    """Return int64 tensor [N] of video_number for each clip in bundle order."""
    return torch.tensor(
        [int(c["video_number"]) for c in bundle.clips],
        dtype=torch.long,
    )


def cross_video_topk(bundle, mat, topk=TOPK):
    """Top-k nearest neighbors for each clip, excluding same-video pairs.

    Returns a `[N, topk]` long tensor where row i holds the indices of the
    top-k cross-video neighbors of clip i (ascending DTW distance).

    Same-video pairs (and the self-pair) are masked to +inf before topk.
    If a clip has fewer than `topk` cross-video candidates, the overflow
    slots are filled with -1.
    """
    mat_inf = mat.detach().clone().float()
    N = mat_inf.shape[0]
    video_of = _video_of_clip(bundle)
    same = (video_of[:, None] == video_of[None, :])
    mat_inf[same] = float("inf")
    mat_inf.fill_diagonal_(float("inf"))
    k_eff = min(topk, N - 1)
    vals, idx = torch.topk(mat_inf, k_eff, dim=1, largest=False)
    # Anything still at +inf = no valid cross-video neighbor (clip's video
    # exhausted the pool); flag with -1 so downstream code ignores it.
    idx = idx.clone()
    idx[vals.isinf()] = -1
    return idx


def cross_video_neighbor_fraction(bundle, mat, topk=TOPK):
    """Mean fraction of top-k neighbors that come from a *different* video.

    No same-video masking — this just measures how promiscuous the raw
    top-k is. A retrieval metric whose top-k is mostly intra-video clips
    will score low here. Good cross-video designs produce values near
    (1 - E[k_same] / k).
    """
    mat_inf = mat.detach().clone().float()
    mat_inf.fill_diagonal_(float("inf"))
    _, idx = torch.topk(mat_inf, topk, dim=1, largest=False)    # [N, topk]
    video_of = _video_of_clip(bundle)                            # [N]
    neighbor_videos = video_of[idx]                              # [N, topk]
    same = (neighbor_videos == video_of[:, None]).float()        # [N, topk]
    return float(1.0 - same.mean().item())


def cross_video_bootstrap_stability(bundle, mat, topk=TOPK, n_boot=10,
                                    subsample_frac=0.8, seed=0):
    """Bootstrap Jaccard@k stability of cross-video top-k neighbor sets.

    Repeatedly draw two 80% subsamples of clips, compute each clip's
    cross-video top-k within that subsample, then measure the Jaccard
    overlap of the two top-k sets on the intersection of the two
    bootstraps. Averaged across `n_boot` pairs and across clips.

    Stable designs return values close to 1.0 even after dropping 20% of
    clips from the pool — i.e., the top-k cross-video hits don't depend on
    which minority of clips was unlucky enough to be dropped.
    """
    rng = np.random.default_rng(seed)
    N = mat.shape[0]
    video_of = _video_of_clip(bundle).numpy()
    mat_np = mat.detach().cpu().numpy().astype(np.float32)
    # Precompute full cross-video mask once
    same_full = (video_of[:, None] == video_of[None, :])
    np.fill_diagonal(same_full, True)

    def _topk_subset(subset):
        """Top-k cross-video neighbors for each clip restricted to `subset`."""
        sub = np.sort(subset)
        sub_set = set(int(x) for x in sub)
        out = {}
        for i in sub:
            # Cross-video candidates inside the subset, excluding i itself
            cand = [j for j in sub if j != i and not same_full[i, j]]
            if len(cand) == 0:
                out[int(i)] = set()
                continue
            dists = mat_np[i, cand]
            order = np.argsort(dists)[:topk]
            out[int(i)] = set(int(cand[o]) for o in order)
        return out, sub_set

    scores = []
    for b in range(n_boot):
        keep1 = rng.choice(N, size=int(N * subsample_frac), replace=False)
        keep2 = rng.choice(N, size=int(N * subsample_frac), replace=False)
        nn1, set1 = _topk_subset(keep1)
        nn2, set2 = _topk_subset(keep2)
        inter = set1 & set2
        if not inter:
            continue
        jacs = []
        for i in inter:
            a = nn1.get(i, set())
            b2 = nn2.get(i, set())
            if not a and not b2:
                continue
            u = a | b2
            if not u:
                continue
            jacs.append(len(a & b2) / len(u))
        if jacs:
            scores.append(float(np.mean(jacs)))
    if not scores:
        return float("nan")
    return float(np.mean(scores))


def articulation_impact_jaccard(mat_full, mat_wrist, topk=TOPK,
                                bundle=None, cross_video_only=False):
    """Top-k Jaccard between a full-articulation design and a wrist-only design.

    Measures how much finger/hand articulation changes the retrieval
    ranking beyond the wrist path alone. Low Jaccard = articulation
    substantially alters neighbors (finger pose is informative). High
    Jaccard = articulation is a no-op on top of wrist paths.

    If `cross_video_only=True`, both top-k sets are restricted to
    cross-video neighbors using the bundle's video_number metadata.
    """
    def _topk(m):
        m = m.detach().clone().float()
        m.fill_diagonal_(float("inf"))
        if cross_video_only:
            assert bundle is not None
            video_of = _video_of_clip(bundle)
            m[video_of[:, None] == video_of[None, :]] = float("inf")
        _, idx = torch.topk(m, topk, dim=1, largest=False)
        return idx.cpu().numpy()

    A = _topk(mat_full)
    B = _topk(mat_wrist)
    N = A.shape[0]
    jacs = np.empty(N, dtype=np.float64)
    for i in range(N):
        a = set(A[i].tolist())
        b = set(B[i].tolist())
        u = a | b
        jacs[i] = (len(a & b) / len(u)) if u else 1.0
    return float(jacs.mean())


def hubness_skew(mat, topk=TOPK):
    """Skewness of the in-degree distribution of the top-k NN graph.

    For each clip i, count how many other clips have i in their top-k.
    A healthy metric has this distribution roughly symmetric around k;
    degenerate metrics have a small number of "hub" clips hoarding most
    in-edges (high positive skew).
    """
    mat_inf = mat.clone()
    mat_inf.fill_diagonal_(float("inf"))
    _, topk_idx = torch.topk(mat_inf, topk, dim=1, largest=False)
    N = mat.shape[0]
    counts = torch.zeros(N, dtype=torch.long)
    flat = topk_idx.reshape(-1)
    counts.scatter_add_(0, flat, torch.ones_like(flat))
    counts_np = counts.numpy().astype(np.float64)
    mu = counts_np.mean()
    sd = counts_np.std()
    if sd < 1e-12:
        return 0.0
    return float(((counts_np - mu) ** 3).mean() / (sd ** 3))


def distance_spread(mat):
    """Return p95 / p5 ratio of off-diagonal distances. 1.0 = degenerate."""
    N = mat.shape[0]
    idx = torch.triu_indices(N, N, offset=1)
    vals = mat[idx[0], idx[1]].cpu().numpy()
    p5, p95 = np.percentile(vals, [5, 95])
    if p5 <= 0:
        return float("inf")
    return float(p95 / p5)


def reverse_self_rank(mat_ref, mat_pert):
    """For each clip i, rank of i within the neighbors of i in mat_pert.

    Used for reverse-time perturbation diagnostics. Returns the mean
    1-indexed rank across clips (1 = nearest neighbor).

    Specifically: we take row i of mat_pert (distances from clip_i to
    all perturbed clips), sort ascending, and find position of j=i.
    """
    N = mat_ref.shape[0]
    # mat_pert rows: query i vs perturbed j; self is j=i.
    ranks = torch.zeros(N)
    for i in range(N):
        row = mat_pert[i].clone()
        order = torch.argsort(row)
        pos = (order == i).nonzero(as_tuple=True)[0]
        ranks[i] = float(pos.item() + 1) if pos.numel() else float(N)
    return float(ranks.mean())
