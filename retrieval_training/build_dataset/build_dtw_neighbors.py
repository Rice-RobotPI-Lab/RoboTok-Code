"""Build per-clip top-K DTW neighbors for hard-positive batch mining.

Two-stage:
1. Pool trajectories to flat vectors via `stats4` (mean, std, first, last
   per time-axis). L2-normalise; FAISS IndexFlatIP self-search → top
   `candidate_k` candidates per clip.
2. Compute DTW on those candidates. Keep top-K by similarity.

The stage-2 kernel has two modes:
- `--exact` (recommended): length-aware symmetric-2 DTW on truncated
  `[:L_i] x [:L_j]` grids. No padding bias. Validated against a
  brute-force reference.
- default (backwards-compatible): `OnlineDTWComputer` on full padded T_max
  grid. Kept for compatibility with older neighbor tables. This mode carries a
  systematic positive bias proportional to `(T_max - L_i) + (T_max - L_j)` and
  has markedly lower top-K agreement with exact DTW; prefer `--exact` for new
  runs.

Output: `{output_path}` containing the neighbor table (see payload schema
below). Used at training time when `cfg.use_positive_mining=True` so each
batch can include precomputed DTW-positive clips for each anchor.

CLI is for ad-hoc / debugging use; the canonical invocation is `train.py`'s
auto-build of the cache in `run_dir`.

CLI:
    python build_dtw_neighbors.py \\
        --trajectories outputs/training_outputs/.../trajectories_X.pt \\
        --design first_frame_midpoint_hand_length_no_z \\
        --output   outputs/training_outputs/.../dtw_neighbors_X.pt \\
        --top-k 20 --candidate-k 100 --pooling stats4 \\
        --exact \\
        --sanity-check
"""

import argparse
import sys
import time
from pathlib import Path
from typing import Tuple

import numpy as np
import torch

SCRIPT_DIR = Path(__file__).resolve().parent
TRAINING_ROOT = SCRIPT_DIR.parent  # retrieval_training/ (dtw_cknna.py lives here)

if str(TRAINING_ROOT) not in sys.path:
    sys.path.insert(0, str(TRAINING_ROOT))


# ---------------------------------------------------------------------------
# Stage 1a: trajectory pooling (pure CPU, vectorised)
# ---------------------------------------------------------------------------

_POOLINGS = ("stats4", "keyframes8", "flat_full")
_KEYFRAMES8_N = 8


def _pool_trajectories(
    trajs: torch.Tensor,        # [N, T, D] fp32
    lengths: torch.Tensor,      # [N] int
    pooling: str = "stats4",
) -> torch.Tensor:              # [N, D_pool] fp32
    """Flatten each trajectory to a fixed-length vector for FAISS search.

    Supported pooling strategies:

    - **`stats4`** (D_pool = 4*D): concat([mean_t, std_t, first_frame,
      last_frame]) over the valid prefix. Compact (~336 dims at D=84) and
      cheap, but mean/std are direction-invariant — two trajectories with
      different temporal shapes can pool to the same vector.

    - **`keyframes8`** (D_pool = 8*D): 8 evenly-spaced frames sampled from
      the valid prefix, concatenated time-major. Preserves coarse temporal
      shape (start, ~quartiles, end) at compact dim (~672 at D=84), with
      FAISS cost only ~2× `stats4`.

    - **`flat_full`** (D_pool = T*D): the entire trajectory flattened
      time-major, with the padded suffix zeroed out. Preserves all temporal
      structure → much higher recall (typically ≥ 0.90 against exact DTW),
      at the cost of larger FAISS index (~T/4 × bigger memory + search time).
      Use this when `stats4` and `keyframes8` recall are still too low.

    The pooled vector's only job is to filter to ~candidate_k candidates per
    query; the exact-DTW second stage refines the ranking among those. Higher
    pooling recall = fewer true positives lost in the candidate filter.
    """
    if pooling not in _POOLINGS:
        raise ValueError(
            f"Unknown pooling={pooling!r}; valid: {sorted(_POOLINGS)}"
        )

    trajs = trajs.float()
    N, T, D = trajs.shape
    lengths_long = lengths.long()
    if (lengths_long < 1).any():
        raise ValueError("All trajectory lengths must be >= 1")
    if (lengths_long > T).any():
        raise ValueError(f"Some lengths exceed T={T}")

    # Per-clip valid mask: True for t < length[i]. Reused by both poolings.
    mask = torch.arange(T)[None, :] < lengths_long[:, None]   # [N, T] bool
    mask_f = mask.unsqueeze(-1).to(trajs.dtype)               # [N, T, 1]

    if pooling == "flat_full":
        # Zero the padded suffix so it contributes nothing to FAISS cosine
        # similarity — a trajectory's pooled vector reflects only its valid
        # frames. After L2-normalize (in _faiss_top_candidates), short clips'
        # vectors concentrate in their valid-prefix dims while long clips
        # spread across all dims, which is correct: two clips' cosine matches
        # only on the frames they both have.
        masked = trajs * mask_f                               # [N, T, D]
        return masked.reshape(N, T * D).contiguous()          # [N, T*D]

    if pooling == "keyframes8":
        # Sample _KEYFRAMES8_N evenly-spaced frames from each clip's valid
        # prefix [0, length-1]. For length=1, every keyframe is index 0. For
        # length<n_kf, some keyframes will repeat (acceptable: short clips
        # legitimately have less temporal information).
        n_kf = _KEYFRAMES8_N
        length_minus_1 = (lengths_long - 1).clamp(min=0).to(trajs.dtype)  # [N]
        if n_kf == 1:
            # Edge case: just take the first frame.
            return trajs[:, 0, :].contiguous()                # [N, D]
        # Linspace from 0 to length-1 over n_kf points, rounded to integer.
        t_steps = torch.arange(n_kf, dtype=trajs.dtype) / (n_kf - 1)  # [n_kf]
        kf_idx = (t_steps[None, :] * length_minus_1[:, None]).round().long()  # [N, n_kf]
        kf_idx = kf_idx.clamp(min=0, max=T - 1)
        # Gather: trajs[i, kf_idx[i], :] for each i.
        batch_idx = torch.arange(N)[:, None].expand(-1, n_kf)            # [N, n_kf]
        keyframes = trajs[batch_idx, kf_idx, :]                          # [N, n_kf, D]
        return keyframes.reshape(N, n_kf * D).contiguous()               # [N, n_kf*D]

    # pooling == "stats4"
    counts = mask_f.sum(dim=1).clamp(min=1.0)                 # [N, 1]
    sum_per = (trajs * mask_f).sum(dim=1)                     # [N, D]
    mean = sum_per / counts                                    # [N, D]

    sum_sq = ((trajs * trajs) * mask_f).sum(dim=1)            # [N, D]
    var = (sum_sq / counts) - mean * mean
    std = var.clamp(min=0.0).sqrt()                           # [N, D]

    first = trajs[:, 0, :]                                    # [N, D]
    last_idx = (lengths_long - 1).clamp(min=0)                # [N]
    last = trajs[torch.arange(N), last_idx, :]                # [N, D]

    return torch.cat([mean, std, first, last], dim=1)         # [N, 4D]


# ---------------------------------------------------------------------------
# Stage 1b: FAISS candidate search (pure CPU)
# ---------------------------------------------------------------------------

def _faiss_top_candidates(
    pooled: torch.Tensor,       # [N, D_pool] fp32
    candidate_k: int,
) -> torch.Tensor:              # [N, candidate_k] int64; -1 = no neighbor
    """L2-normalise pooled vectors, then FAISS IndexFlatIP top-(candidate_k+1)
    self-search. Drop the self entry per row.

    Returns clip-idx values (rows in `pooled`/`trajs`). Pads with -1 if a
    row has fewer than `candidate_k` non-self neighbors (only possible at
    pathologically small N — never in practice at N=51,880).
    """
    import faiss

    pooled_np = pooled.float().contiguous().numpy()
    norms = np.linalg.norm(pooled_np, axis=1, keepdims=True)
    pooled_np = pooled_np / np.maximum(norms, 1e-12)
    pooled_np = pooled_np.astype(np.float32)

    N, D_pool = pooled_np.shape
    index = faiss.IndexFlatIP(D_pool)
    index.add(pooled_np)

    # +1 because the self-row appears at column 0 (cos sim = 1.0 to self).
    _, neighbor = index.search(pooled_np, candidate_k + 1)

    # For each row, drop self defensively (in nominal case it's at col 0,
    # but exact duplicates can shuffle that).
    cand = torch.full((N, candidate_k), -1, dtype=torch.int64)
    for i in range(N):
        non_self = [j for j in neighbor[i].tolist() if j != i and j >= 0]
        take = min(candidate_k, len(non_self))
        if take > 0:
            cand[i, :take] = torch.as_tensor(non_self[:take], dtype=torch.int64)

    return cand


# ---------------------------------------------------------------------------
# Stage 2: length-aware (exact) DTW. Bit-equivalent to truncating both
# trajectories to (L_i, L_j) before running the existing symmetric-2 kernel.
# ---------------------------------------------------------------------------
class ExactDTWComputer:
    def __init__(self, dtw_design: str):
        from dtw_cknna import DTW_DESIGNS, DTW_METHODS, DTW_NORMS

        if dtw_design not in DTW_DESIGNS:
            raise ValueError(f"unknown design {dtw_design!r}")
        d = DTW_DESIGNS[dtw_design]
        method = DTW_METHODS[d["cost"]]
        self.cost_fn = method["cost_fn"]
        self.kernel = method["kernel_fn"]()
        self.norm_fn = DTW_NORMS[d["norm"]]
        self.norm_key = d["norm"]

    def compute_block(
        self,
        query_traj: torch.Tensor,    # [L_i, D] fp32 cuda
        target_trajs: torch.Tensor,  # [B, L_j, D] fp32 cuda (all length L_j)
        query_len_f: torch.Tensor,   # 0-d fp32 cuda
        target_lens_f: torch.Tensor, # [B] fp32 cuda (all == L_j)
    ) -> torch.Tensor:               # [B] distance (length-normalised)
        from numba import cuda as numba_cuda

        L_i = query_traj.shape[0]
        B, L_j, _ = target_trajs.shape
        device = target_trajs.device
        if L_i == 0 or L_j == 0:
            raise ValueError("zero-length trajectory")

        D_cost = self.cost_fn(query_traj, target_trajs)  # [B, L_i, L_j]
        R_buf = torch.full(
            (B, L_i + 2, L_j + 2), float("inf"),
            device=device, dtype=torch.float32,
        )
        R_buf[:, 0, 0] = -D_cost[:, 0, 0]

        n_passes = L_i + L_j - 1
        threads = max(L_i, L_j)
        self.kernel[B, threads](
            numba_cuda.as_cuda_array(D_cost),
            0.0, L_i, L_j, n_passes,
            numba_cuda.as_cuda_array(R_buf),
        )
        distances = R_buf[:, L_i, L_j]
        if self.norm_key != "none":
            distances = self.norm_fn(distances, target_lens_f, query_len_f)
        return distances


def _dtw_top_k_for_candidates_exact(
    query_clip_idx: int,
    cand_clip_idx: torch.Tensor,    # [candidate_k] int64; -1 = padding
    trajs_gpu: torch.Tensor,        # [N, T, D] fp32 on GPU
    lengths_i_gpu: torch.Tensor,    # [N] int on GPU
    lengths_f_gpu: torch.Tensor,    # [N] fp32 on GPU
    top_k: int,
    computer: "ExactDTWComputer",
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Length-aware sibling of `_dtw_top_k_for_candidates`.

    Buckets the query's candidate set by their true length L_j and dispatches
    one kernel call per bucket on the truncated `L_i x L_j` cost grid. Returns
    (top_idx, top_sim) — same contract as the padded path, with
    `top_sim = -length_normalised_distance`.
    """
    valid = cand_clip_idx[cand_clip_idx >= 0]
    if valid.numel() == 0:
        return (
            torch.empty(0, dtype=torch.int64),
            torch.empty(0, dtype=torch.float32),
        )

    device = trajs_gpu.device
    valid_gpu = valid.to(device)
    L_i = int(lengths_i_gpu[query_clip_idx].item())
    q_trunc = trajs_gpu[query_clip_idx, :L_i].contiguous()
    q_lf = lengths_f_gpu[query_clip_idx]

    cand_lens_i = lengths_i_gpu.index_select(0, valid_gpu)
    cand_lens_f = lengths_f_gpu.index_select(0, valid_gpu)

    distances = torch.empty(valid.numel(), device=device, dtype=torch.float32)

    # Bucket candidates by their length so each kernel call sees a uniform
    # L_j (the kernel grid is rectangular and needs B targets of equal length).
    unique_lens = torch.unique(cand_lens_i)
    for L_j_t in unique_lens:
        L_j = int(L_j_t.item())
        mask = cand_lens_i == L_j_t
        local_pos = mask.nonzero(as_tuple=True)[0]
        bucket_global = valid_gpu[local_pos]
        tgt = trajs_gpu.index_select(0, bucket_global)[:, :L_j, :].contiguous()
        tgt_lens_f = cand_lens_f[local_pos]
        d = computer.compute_block(q_trunc, tgt, q_lf, tgt_lens_f)
        distances.index_copy_(0, local_pos, d)

    sims = -distances
    k = min(top_k, valid.numel())
    top_sim, top_local = sims.topk(k)
    top_idx = valid[top_local.cpu()]
    return top_idx, top_sim.cpu()


# ---------------------------------------------------------------------------
# Stage 2: padded DTW refinement (GPU) — legacy path, kept for compat.
# ---------------------------------------------------------------------------

def _dtw_top_k_for_candidates(
    query_clip_idx: int,
    cand_clip_idx: torch.Tensor,    # [candidate_k] int64; -1 = padding
    trajs_gpu: torch.Tensor,        # [N, T, D] fp32 on GPU
    lengths_f_gpu: torch.Tensor,    # [N] fp32 on GPU
    top_k: int,
    computer,                       # OnlineDTWComputer
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Run exact DTW between one query and its candidates; return
    (top_idx, top_sim).

    `top_idx` are clip-idx values (rows in `trajs`); `top_sim` is similarity
    (= -DTW distance). Both returned on CPU. Padding (-1) entries in
    `cand_clip_idx` are dropped before the DTW step.
    """
    valid = cand_clip_idx[cand_clip_idx >= 0]
    if valid.numel() == 0:
        return (
            torch.empty(0, dtype=torch.int64),
            torch.empty(0, dtype=torch.float32),
        )

    valid_gpu = valid.to(trajs_gpu.device)
    cand_trajs = trajs_gpu.index_select(0, valid_gpu)
    cand_lens = lengths_f_gpu.index_select(0, valid_gpu)

    sims = computer.compute_one_query_against_targets(
        trajs_gpu[query_clip_idx],
        cand_trajs,
        lengths_f_gpu[query_clip_idx],
        cand_lens,
    )

    k = min(top_k, valid.numel())
    top_sim, top_local = sims.topk(k)
    top_idx = valid[top_local.cpu()]
    return top_idx, top_sim.cpu()


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def build_dtw_neighbors(
    trajectories_path: Path,
    dtw_design: str,
    output_path: Path,
    top_k: int = 20,
    candidate_k: int = 100,
    pooling: str = "stats4",
    chunk_size: int = 256,         # reserved for future multi-query batching
    device: str = "cuda",
    exact: bool = True,
) -> None:
    """Build the per-clip top-K DTW neighbor table and save to `output_path`.

    Loads the trajectory artifact at `trajectories_path` (typically the one
    produced by `build_trajectories.py` and sitting alongside us in `run_dir`).
    Validates `design` matches; runs stage 1 (FAISS) + stage 2 (DTW);
    writes `{neighbor_clip_idx, neighbor_dtw_sim, design, top_k, candidate_k,
    pooling, clip_keys, N, build_seconds, schema_version, dtw_mode}`.

    `exact=True` uses length-aware DTW (recommended); `exact=False` keeps the
    legacy padded path used to build older neighbor tables.
    """
    print(f"[build_dtw_neighbors] design={dtw_design}  top_k={top_k}  "
          f"candidate_k={candidate_k}  pooling={pooling}  exact={exact}")
    print(f"[build_dtw_neighbors] reading {trajectories_path}")

    payload = torch.load(trajectories_path, weights_only=False, map_location="cpu")
    build_dtw_neighbors_from_payload(
        payload=payload,
        dtw_design=dtw_design,
        output_path=output_path,
        top_k=top_k,
        candidate_k=candidate_k,
        pooling=pooling,
        chunk_size=chunk_size,
        device=device,
        exact=exact,
    )


def build_dtw_neighbors_from_payload(
    payload: dict,
    dtw_design: str,
    output_path: Path,
    top_k: int = 20,
    candidate_k: int = 100,
    pooling: str = "stats4",
    chunk_size: int = 256,         # reserved for future multi-query batching
    device: str = "cuda",
    exact: bool = True,
) -> None:
    """Build the per-clip top-K DTW neighbor table from an in-memory payload."""
    print(f"[build_dtw_neighbors] design={dtw_design}  top_k={top_k}  "
          f"candidate_k={candidate_k}  pooling={pooling}  exact={exact}")
    if payload.get("design") != dtw_design:
        raise RuntimeError(
            f"Trajectories design mismatch: file={payload.get('design')!r} "
            f"vs requested={dtw_design!r}"
        )
    trajs_cpu = payload["trajectories"]
    lengths_cpu = payload["lengths"]
    clip_keys = payload["clip_keys"]
    N, T, D = trajs_cpu.shape
    print(f"[build_dtw_neighbors] N={N}  T={T}  D={D}")

    t_start = time.time()

    # --- Stage 1a: pool trajectories ---
    print("[build_dtw_neighbors] Stage 1a: pooling trajectories...")
    t0 = time.time()
    pooled = _pool_trajectories(trajs_cpu, lengths_cpu, pooling=pooling)
    print(f"  pooled shape={tuple(pooled.shape)} in {time.time() - t0:.1f}s")

    # --- Stage 1b: FAISS top-candidate_k ---
    print(f"[build_dtw_neighbors] Stage 1b: FAISS top-{candidate_k} candidates...")
    t0 = time.time()
    cand = _faiss_top_candidates(pooled, candidate_k)
    print(f"  candidates shape={tuple(cand.shape)} in {time.time() - t0:.1f}s")

    # --- Stage 2: DTW refinement ---
    if exact:
        computer = ExactDTWComputer(dtw_design)
        dtw_mode = "exact"
    else:
        from online_dtw import OnlineDTWComputer
        computer = OnlineDTWComputer(dtw_design)
        dtw_mode = "padded"

    print(f"[build_dtw_neighbors] Stage 2: {dtw_mode} DTW on {N} × "
          f"{candidate_k} pairs...")
    t0 = time.time()
    trajs_gpu = trajs_cpu.to(device).float().contiguous()
    lengths_i_gpu = lengths_cpu.to(device, dtype=torch.int32)
    lengths_f_gpu = lengths_cpu.to(device, dtype=torch.float32)

    neighbor_clip_idx = torch.full((N, top_k), -1, dtype=torch.int32)
    neighbor_dtw_sim = torch.full((N, top_k), float("-inf"), dtype=torch.float32)

    log_every = max(1, N // 20)
    for i in range(N):
        if exact:
            top_idx, top_sim = _dtw_top_k_for_candidates_exact(
                i, cand[i], trajs_gpu, lengths_i_gpu, lengths_f_gpu,
                top_k, computer,
            )
        else:
            top_idx, top_sim = _dtw_top_k_for_candidates(
                i, cand[i], trajs_gpu, lengths_f_gpu, top_k, computer,
            )
        k = top_idx.shape[0]
        neighbor_clip_idx[i, :k] = top_idx.to(torch.int32)
        neighbor_dtw_sim[i, :k] = top_sim
        if (i + 1) % log_every == 0 or i + 1 == N:
            elapsed = time.time() - t0
            rate = (i + 1) / max(elapsed, 1e-9)
            eta = (N - i - 1) / max(rate, 1e-9)
            print(f"  [{i + 1}/{N}] elapsed={elapsed:.0f}s "
                  f"rate={rate:.0f}/s eta={eta:.0f}s")

    print(f"  {dtw_mode} DTW: {time.time() - t0:.1f}s")
    build_seconds = time.time() - t_start

    # --- Save payload ---
    out_payload = {
        "neighbor_clip_idx": neighbor_clip_idx,
        "neighbor_dtw_sim":  neighbor_dtw_sim,
        "design":            dtw_design,
        "top_k":             top_k,
        "candidate_k":       candidate_k,
        "pooling":           pooling,
        "clip_keys":         clip_keys,
        "N":                 N,
        "build_seconds":     build_seconds,
        "schema_version":    1,
        "dtw_mode":          dtw_mode,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(out_payload, output_path)
    size_mb = output_path.stat().st_size / 1e6
    print(f"[build_dtw_neighbors] saved {output_path} ({size_mb:.1f} MB) "
          f"in {build_seconds:.1f}s")


def build_train_pool_neighbors(
    trajectories_path: Path,
    train_positions,
    dtw_design: str,
    output_path: Path,
    top_k: int = 20,
    candidate_k: int = 100,
    pooling: str = "stats4",
    device: str = "cuda",
    exact: bool = True,
) -> None:
    """Build a DTW neighbor table over a train-only subset, remapped to global clip indices.

    `build_dtw_neighbors_from_payload` stores neighbors in *local row-space*
    (rows 0..N_train-1). For the train-pool CKNNA we want a table keyed by
    original clip index, with non-train rows left as `-1`. This wrapper:
      1. Slices the full trajectories payload down to `train_positions`.
      2. Calls the standard builder to a temp file.
      3. Remaps both axes (rows and column entries) from local→global
         clip-idx space via the sub-payload's `clip_keys`.
      4. Saves a sparse `[N_total, top_k]` table to `output_path`.
    """
    train_positions = sorted(int(p) for p in train_positions)
    full_payload = torch.load(trajectories_path, weights_only=False, map_location="cpu")
    train_keys = [full_payload["clip_keys"][p] for p in train_positions]
    sub_payload = {
        "trajectories": full_payload["trajectories"][train_positions],
        "lengths":      full_payload["lengths"][train_positions],
        "clip_keys":    train_keys,
        "design":       full_payload["design"],
    }

    n_total = int(full_payload["trajectories"].shape[0])
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = output_path.with_suffix(output_path.suffix + ".tmp_local")
    build_dtw_neighbors_from_payload(
        payload=sub_payload,
        dtw_design=dtw_design,
        output_path=tmp_path,
        top_k=top_k,
        candidate_k=candidate_k,
        pooling=pooling,
        device=device,
        exact=exact,
    )

    local_payload = torch.load(tmp_path, weights_only=False, map_location="cpu")
    # Important: `clip_keys` carries (video_number, node_uid) metadata, not
    # integer clip indices. The dataset establishes clip_idx == row index in
    # the trajectories tensor (see VideoClipDataset), so `train_positions`
    # IS the local→global mapping for our row slice.
    local_to_global = np.asarray(train_positions, dtype=np.int64)
    if local_to_global.shape[0] != local_payload["neighbor_clip_idx"].shape[0]:
        raise RuntimeError(
            "train-pool builder: sub-payload row count does not match train_positions length"
        )

    neigh_local = local_payload["neighbor_clip_idx"].numpy().astype(np.int64)
    sim_local = local_payload["neighbor_dtw_sim"].numpy()
    # Remap column entries: -1 stays -1; valid local index → global clip_idx.
    safe_idx = np.where(neigh_local >= 0, neigh_local, 0)
    neigh_global = np.where(
        neigh_local >= 0, local_to_global[safe_idx], -1,
    ).astype(np.int32)

    table_idx = np.full((n_total, top_k), -1, dtype=np.int32)
    table_sim = np.full((n_total, top_k), float("-inf"), dtype=np.float32)
    table_idx[local_to_global] = neigh_global
    table_sim[local_to_global] = sim_local

    out_payload = dict(local_payload)
    out_payload["neighbor_clip_idx"] = torch.from_numpy(table_idx)
    out_payload["neighbor_dtw_sim"] = torch.from_numpy(table_sim)
    out_payload["N"] = n_total
    out_payload["N_train"] = int(local_to_global.shape[0])
    out_payload["train_clip_indices"] = torch.from_numpy(local_to_global.astype(np.int64))
    torch.save(out_payload, output_path)
    tmp_path.unlink(missing_ok=True)
    size_mb = output_path.stat().st_size / 1e6
    print(f"[build_train_pool_neighbors] saved {output_path} ({size_mb:.1f} MB) "
          f"covering {local_to_global.shape[0]}/{n_total} clips")


# ---------------------------------------------------------------------------
# Sanity check (CLI flag)
# ---------------------------------------------------------------------------

def _sanity_check(
    trajectories_path: Path,
    dtw_design: str,
    neighbors_path: Path,
    sample_n: int = 1000,
    seed: int = 0,
    device: str = "cuda",
    exact: bool = False,
) -> None:
    """Compare cached top-K against brute-force DTW on a random
    sample of `sample_n` query clips. Reports recall@K (cached top-K hits
    inside the brute-force top-K) so we can confirm the candidate-filter
    stage didn't drop too many true positives.

    `exact` must match the mode used to build the cache, so the brute-force
    reference measures the same quantity that the cache stores.
    """
    print(f"[sanity-check] sampling {sample_n} clips for brute-force DTW "
          f"comparison (exact={exact})...")
    payload = torch.load(trajectories_path, weights_only=False, map_location="cpu")
    trajs_cpu = payload["trajectories"]
    lengths_cpu = payload["lengths"]
    N = trajs_cpu.shape[0]

    cache = torch.load(neighbors_path, weights_only=False, map_location="cpu")
    cached_neigh = cache["neighbor_clip_idx"]
    K = int(cache["top_k"])

    rng = np.random.default_rng(seed)
    sample = sorted(rng.choice(N, size=min(sample_n, N), replace=False).tolist())

    trajs_gpu = trajs_cpu.to(device).float().contiguous()
    lengths_i_gpu = lengths_cpu.to(device, dtype=torch.int32)
    lengths_f_gpu = lengths_cpu.to(device, dtype=torch.float32)

    if exact:
        computer = ExactDTWComputer(dtw_design)
    else:
        from online_dtw import OnlineDTWComputer
        computer = OnlineDTWComputer(dtw_design)

    # Memory-safe brute-force top-K: chunk the targets so the cost tensor
    # (shape [chunk, T, T, D]) stays small. At T=42, D=84 the per-chunk
    # tensor is `chunk * 42 * 42 * 84 * 4 bytes` ≈ chunk * 0.6 MB; chunk=4096
    # gives ~2.4 GB per call — safely fits on the L40S even with the trajs
    # tensor (730 MB) and other allocations resident.
    chunk_size = 4096

    def _sims_chunk(query_idx: int, start: int, end: int) -> torch.Tensor:
        if exact:
            L_i = int(lengths_i_gpu[query_idx].item())
            q_trunc = trajs_gpu[query_idx, :L_i].contiguous()
            q_lf = lengths_f_gpu[query_idx]
            chunk_lens_i = lengths_i_gpu[start:end]
            chunk_lens_f = lengths_f_gpu[start:end]
            distances = torch.empty(end - start, device=device, dtype=torch.float32)
            for L_j_t in torch.unique(chunk_lens_i):
                L_j = int(L_j_t.item())
                local_pos = (chunk_lens_i == L_j_t).nonzero(as_tuple=True)[0]
                tgt = trajs_gpu[start:end].index_select(0, local_pos)[:, :L_j, :].contiguous()
                tgt_lens_f = chunk_lens_f.index_select(0, local_pos)
                d = computer.compute_block(q_trunc, tgt, q_lf, tgt_lens_f)
                distances.index_copy_(0, local_pos, d)
            return (-distances).cpu()
        else:
            return computer.compute_one_query_against_targets(
                trajs_gpu[query_idx],
                trajs_gpu[start:end],
                lengths_f_gpu[query_idx],
                lengths_f_gpu[start:end],
            ).cpu()

    def _brute_force_topk(query_idx: int) -> set:
        best_sim = torch.full((K,), float("-inf"), dtype=torch.float32)
        best_idx = torch.full((K,), -1, dtype=torch.int64)
        for start in range(0, N, chunk_size):
            end = min(start + chunk_size, N)
            sims = _sims_chunk(query_idx, start, end)
            local_idx = torch.arange(start, end)
            # Drop the self-row from this chunk if it falls in the slice.
            if start <= query_idx < end:
                keep_mask = local_idx != query_idx
                sims = sims[keep_mask]
                local_idx = local_idx[keep_mask]
            cat_sim = torch.cat([best_sim, sims])
            cat_idx = torch.cat([best_idx, local_idx])
            top = cat_sim.topk(K)
            best_sim = top.values
            best_idx = cat_idx[top.indices]
        return set(best_idx.tolist())

    recalls = []
    t0 = time.time()
    for n_done, query in enumerate(sample, 1):
        true_top = _brute_force_topk(query)
        cached_top = set(int(x) for x in cached_neigh[query].tolist() if x >= 0)
        overlap = len(true_top & cached_top)
        recalls.append(overlap / K)
        if n_done % 100 == 0 or n_done == len(sample):
            elapsed = time.time() - t0
            rate = n_done / elapsed
            eta = (len(sample) - n_done) / rate
            print(f"  [{n_done}/{len(sample)}] running mean recall="
                  f"{sum(recalls)/len(recalls):.3f} "
                  f"(elapsed={elapsed:.0f}s eta={eta:.0f}s)")

    mean_recall = sum(recalls) / len(recalls)
    print(f"[sanity-check] recall@{K}: mean={mean_recall:.3f} "
          f"(min={min(recalls):.3f}, max={max(recalls):.3f}, "
          f"n={len(recalls)}) in {time.time() - t0:.0f}s")
    if mean_recall < 0.85:
        print("[sanity-check] WARNING: mean recall below 0.85 — "
              "consider raising candidate_k or trying a different pooling.")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trajectories", type=Path, required=True,
                        help="Path to trajectories_<design>.pt")
    parser.add_argument("--design", type=str, required=True,
                        help="DTW design key (must match the trajectories file)")
    parser.add_argument("--output", type=Path, required=True,
                        help="Output .pt path for the neighbor table")
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--candidate-k", type=int, default=100)
    parser.add_argument("--pooling", type=str, default="stats4")
    parser.add_argument("--exact", dest="exact", action="store_true",
                        default=True,
                        help="Use length-aware DTW (default; recommended).")
    parser.add_argument("--padded", dest="exact", action="store_false",
                        help="Use the legacy padded-grid DTW path "
                             "(systematic positive bias; kept for reproducing "
                             "older neighbor tables).")
    parser.add_argument("--sanity-check", action="store_true",
                        help="After building, sample 1000 clips and compare "
                             "cached top-K against brute-force DTW (uses "
                             "the same DTW mode as --exact)")
    args = parser.parse_args()

    build_dtw_neighbors(
        trajectories_path=args.trajectories,
        dtw_design=args.design,
        output_path=args.output,
        top_k=args.top_k,
        candidate_k=args.candidate_k,
        pooling=args.pooling,
        exact=args.exact,
    )

    if args.sanity_check:
        _sanity_check(args.trajectories, args.design, args.output,
                      exact=args.exact)


if __name__ == "__main__":
    main()
