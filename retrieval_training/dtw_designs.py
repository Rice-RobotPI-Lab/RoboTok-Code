"""
DTW design registry: cost function, normalization, and trajectory transform.

A "design" is a complete DTW configuration `(cost, norm, tx)`. The trajectory
transform (`tx`) maps the exported 126D body-frame hand trajectories to the
feature space DTW is computed in; the cost and normalization are consumed by
the DTW kernels in `online_dtw.py` and `build_dataset/build_dtw_neighbors.py`,
and `build_dataset/build_trajectories.py` applies the transform once offline.

Usage:
    from dtw_designs import DTW_DESIGNS, DTW_METHODS, DTW_NORMS, TRAJ_TRANSFORMS

    d = DTW_DESIGNS["abs_21j_coords"]
    trajs_tx, lens_tx = TRAJ_TRANSFORMS[d["tx"]](trajs, lens)
    kernel = DTW_METHODS[d["cost"]]["kernel_fn"]()
"""

import torch
import torch.nn.functional as F


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
# Only the Euclidean cost is registered; the symmetric2 kernel accepts any
# [N, T, T] cost tensor, so new costs only need an entry in DTW_METHODS.
# ============================================================================

def _cost_euclidean(vec_i, all_vecs):
    """L2 (Euclidean) distance."""
    diff = vec_i[None, :, None, :] - all_vecs[:, None, :, :]
    return torch.sqrt((diff ** 2).sum(dim=3))


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
}

DTW_NORMS = {
    "none":        _norm_none,
    "mean_length": _norm_mean_length,
}


# ============================================================================
# Trajectory transforms for DTW design variants
# Signature: (trajs [N, T, D], lens [N]) -> (trajs' [N, T', D'], lens' [N])
#
# The base trajectory is 126D: 21 hand joints × 3 coords × 2 hands, with
# dims [:63] = Left hand, [63:] = Right hand, expressed in the estimated
# torso frame (see `keypoints_to_trajectories_21j` in keypoint_trajectories.py).
# A transform either keeps the coordinates as-is or derives invariant
# per-frame features from them; the second return value is the (possibly
# adjusted) clip length used by mean_length normalization.
# ============================================================================

def _n_joints_per_hand(D):
    """Infer (joints-per-hand, coords-per-joint) from trajectory dimension.

    126D → (21, 3): all joints, 3D — the layout every registered design
    expects. 42D/28D (7-joint subsets) are recognized only so the designs
    can raise a precise error on them.
    """
    for n_j in (7, 21):
        C = D // (2 * n_j)
        if 2 * n_j * C == D and C in (2, 3):
            return n_j, C
    raise ValueError(f"Cannot infer joints from D={D}")


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


TRAJ_TRANSFORMS = {
    "abs_21j_coords":       _tx_body_full_pose_21j,
    "pca_interjoint_dists": _tx_pca_interjoint_dists,
}


# ============================================================================
# DTW design registry: (cost, norm, trajectory transform) triples
#
# A "design" is a complete DTW configuration, selected by `dtw_design` in the
# training config. All designs share the Euclidean symmetric2 kernel with
# mean-length normalization and differ only in the trajectory transform.
# ============================================================================

DTW_DESIGNS = {
    # Absolute body-frame coordinates (keeps hand position in the torso frame)
    "abs_21j_coords":       {"cost": "euclidean_sym2", "norm": "mean_length", "tx": "abs_21j_coords"},
    # Selected interjoint distances (translation/rotation-invariant hand shape)
    "pca_interjoint_dists": {"cost": "euclidean_sym2", "norm": "mean_length", "tx": "pca_interjoint_dists"},
}
