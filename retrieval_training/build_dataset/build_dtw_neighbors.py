"""Build per-clip top-K DTW neighbors for hard-positive batch mining.

Two modes, selected by `candidate_k`:

`candidate_k > 0` — FAISS-filtered (approximate candidate set):
1. Pool trajectories to flat vectors via `stats4` (mean, std, first, last
   per time-axis). L2-normalise; FAISS IndexFlatIP self-search → top
   `candidate_k` candidates per clip.
2. Compute exact (length-aware) symmetric-2 DTW on those candidates, on the
   truncated `[:L_i] x [:L_j]` grid of each pair, via `OnlineDTWComputer`.
   Keep top-K by similarity.

`candidate_k <= 0` — exhaustive all-pairs (`dtw_mode="exact_all_pairs"`):
   No FAISS stage. Exact DTW is evaluated for every unordered pair `{i, j}`
   of the corpus exactly once (symmetric-2 DTW is symmetric), and each
   evaluation is merged into the running top-K of *both* rows, so the
   resulting table is the true DTW top-K over the whole corpus. This is the
   path `train.py` takes whenever `loss_type == "topk_set_rank"`.

Output: `{output_path}` containing the neighbor table (see payload schema
below). Used at training time when `cfg.use_positive_mining=True` so each
batch can include precomputed DTW-positive clips for each anchor.

`train.py` builds the cache in `run_dir` automatically; the CLI below builds
one standalone.

CLI:
    python build_dtw_neighbors.py \\
        --trajectories outputs/training_outputs/.../trajectories_X.pt \\
        --design abs_21j_coords \\
        --output   outputs/training_outputs/.../dtw_neighbors_X.pt \\
        --top-k 20 --candidate-k 100 --pooling stats4 \\
        --sanity-check
    # exhaustive: --candidate-k -1 (pooling is ignored)
"""

import argparse
import sys
import time
from pathlib import Path
from typing import Tuple

import numpy as np
import torch

SCRIPT_DIR = Path(__file__).resolve().parent
TRAINING_ROOT = SCRIPT_DIR.parent  # retrieval_training/ (dtw_designs.py lives here)

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
      last_frame]) over the valid prefix. Mean/std are direction-invariant,
      so two trajectories with different temporal shapes can pool to the
      same vector.

    - **`keyframes8`** (D_pool = 8*D): 8 evenly-spaced frames sampled from
      the valid prefix, concatenated time-major. Preserves coarse temporal
      shape (start, ~quartiles, end).

    - **`flat_full`** (D_pool = T*D): the entire trajectory flattened
      time-major, with the padded suffix zeroed out. Preserves all temporal
      structure at the cost of a larger FAISS index.

    The pooled vector's only job is to filter to ~candidate_k candidates per
    query; the exact-DTW second stage refines the ranking among those.
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
    row has fewer than `candidate_k` non-self neighbors.
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
# Stage 2: exact (length-aware) DTW refinement on the GPU.
# ---------------------------------------------------------------------------

def _dtw_top_k_for_candidates(
    query_clip_idx: int,
    cand_clip_idx: torch.Tensor,    # [candidate_k] int64; -1 = padding
    trajs_gpu: torch.Tensor,        # [N, T, D] fp32 on GPU
    lengths_i_gpu: torch.Tensor,    # [N] int on GPU
    lengths_f_gpu: torch.Tensor,    # [N] fp32 on GPU
    top_k: int,
    computer,                       # OnlineDTWComputer
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Exact DTW between one query and its candidates; return top-K.

    Buckets the query's candidate set by their true length L_j and dispatches
    one kernel call per bucket on the truncated `L_i x L_j` cost grid. Returns
    (top_idx, top_sim) with `top_sim = -length_normalised_distance`.
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


def build_dtw_neighbors(
    trajectories_path: Path,
    dtw_design: str,
    output_path: Path,
    top_k: int = 20,
    candidate_k: int = 100,
    pooling: str = "stats4",
    device: str = "cuda",
) -> None:
    """Build the per-clip top-K DTW neighbor table and save to `output_path`.

    Loads the trajectory artifact at `trajectories_path` (typically the one
    produced by `build_trajectories.py` and sitting alongside us in `run_dir`).
    Validates `design` matches; runs stage 1 (FAISS) + stage 2 (DTW);
    writes `{neighbor_clip_idx, neighbor_dtw_sim, design, top_k, candidate_k,
    pooling, clip_keys, N, build_seconds, schema_version, dtw_mode}`.
    """
    print(f"[build_dtw_neighbors] design={dtw_design}  top_k={top_k}  "
          f"candidate_k={candidate_k}  pooling={pooling}")
    print(f"[build_dtw_neighbors] reading {trajectories_path}")

    payload = torch.load(trajectories_path, weights_only=False, map_location="cpu")
    build_dtw_neighbors_from_payload(
        payload=payload,
        dtw_design=dtw_design,
        output_path=output_path,
        top_k=top_k,
        candidate_k=candidate_k,
        pooling=pooling,
        device=device,
    )


def build_dtw_neighbors_from_payload(
    payload: dict,
    dtw_design: str,
    output_path: Path,
    top_k: int = 20,
    candidate_k: int = 100,
    pooling: str = "stats4",
    device: str = "cuda",
) -> None:
    """Build the per-clip top-K DTW neighbor table from an in-memory payload.

    `candidate_k <= 0` selects the exhaustive all-pairs path (see
    `build_dtw_neighbors_exhaustive_from_payload`); `pooling` is then ignored.
    """
    if candidate_k <= 0:
        build_dtw_neighbors_exhaustive_from_payload(
            payload=payload,
            dtw_design=dtw_design,
            output_path=output_path,
            top_k=top_k,
            device=device,
        )
        return
    print(f"[build_dtw_neighbors] design={dtw_design}  top_k={top_k}  "
          f"candidate_k={candidate_k}  pooling={pooling}")
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

    # --- Stage 2: exact DTW refinement ---
    from online_dtw import OnlineDTWComputer
    computer = OnlineDTWComputer(dtw_design)
    dtw_mode = "exact"

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
        top_idx, top_sim = _dtw_top_k_for_candidates(
            i, cand[i], trajs_gpu, lengths_i_gpu, lengths_f_gpu,
            top_k, computer,
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


# ---------------------------------------------------------------------------
# Exhaustive all-pairs mode (no FAISS): exact DTW over the whole corpus.
# ---------------------------------------------------------------------------

def _merge_topk_row(
    best_sim: torch.Tensor,     # [N, K] running top-K sims (sorted desc per row)
    best_idx: torch.Tensor,     # [N, K] running top-K ids; -1 = empty slot
    row: int,
    sims: torch.Tensor,         # [C] new candidate sims for `row`
    idx: torch.Tensor,          # [C] new candidate ids
) -> None:
    """Merge `C` new candidates into the running top-K of one row."""
    K = best_sim.shape[1]
    cat_sim = torch.cat([best_sim[row], sims])
    cat_idx = torch.cat([best_idx[row], idx])
    top_sim, top_pos = cat_sim.topk(K)
    best_sim[row] = top_sim
    best_idx[row] = cat_idx[top_pos]


def _merge_topk_cols(
    best_sim: torch.Tensor,     # [N, K]
    best_idx: torch.Tensor,     # [N, K]
    cols: torch.Tensor,         # [C] rows to update (the targets)
    sims: torch.Tensor,         # [C] sim of each target to `src`
    src: int,                   # the query id, inserted as a candidate of every col
) -> None:
    """Insert `(src, sims[c])` into the running top-K of every `cols[c]`.

    This is the symmetric half of a pair evaluation: `sim(src, col)` is also
    `sim(col, src)`. Rows whose current K-th best already beats the new sim
    are skipped, so once the tables fill up most chunks touch few rows.
    """
    K = best_sim.shape[1]
    cur_sim = best_sim.index_select(0, cols)              # [C, K]
    need = sims > cur_sim[:, -1]                          # K-th best is last (sorted desc)
    if not bool(need.any()):
        return
    sel = need.nonzero(as_tuple=True)[0]
    cols_sel = cols.index_select(0, sel)
    cat_sim = torch.cat([cur_sim.index_select(0, sel), sims.index_select(0, sel)[:, None]], dim=1)
    cat_idx = torch.cat([
        best_idx.index_select(0, cols_sel),
        torch.full((sel.numel(), 1), src, dtype=best_idx.dtype, device=best_idx.device),
    ], dim=1)
    top_sim, top_pos = cat_sim.topk(K, dim=1)
    best_sim.index_copy_(0, cols_sel, top_sim)
    best_idx.index_copy_(0, cols_sel, cat_idx.gather(1, top_pos))


def _exhaustive_topk(
    N: int,
    sims_fn,                            # (i, start, end) -> [end-start] sims of i vs [start, end)
    top_k: int,
    device,
    chunk_size: int = 4096,
    train_mask: torch.Tensor | None = None,   # [N] bool on `device`; enables the train-pool table
    train_top_k: int | None = None,
    log_prefix: str = "[build_dtw_neighbors]",
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor | None, torch.Tensor | None]:
    """Exact top-K over all N(N-1)/2 unordered pairs, each evaluated once.

    For every query `i`, `sims_fn` is evaluated against targets `j > i` in
    chunks; each chunk is merged into row `i` (as targets) and into every
    row `j` (as the symmetric entry). Self is never a candidate. Optionally
    maintains a second table restricted to train-side rows *and* columns
    (`train_mask`), for the train-pool CKNNA cache, from the same pass.

    Returns `(best_sim, best_idx, train_sim, train_idx)`; the train pair is
    `None` when `train_mask` is None. Empty slots (when `top_k > N-1`) hold
    `-inf` / `-1`. All tensors live on `device`.
    """
    K = int(top_k)
    if K < 1:
        raise ValueError(f"top_k must be >= 1, got {top_k}")
    best_sim = torch.full((N, K), float("-inf"), dtype=torch.float32, device=device)
    best_idx = torch.full((N, K), -1, dtype=torch.int64, device=device)

    train_sim = train_idx = None
    if train_mask is not None:
        if train_top_k is None or int(train_top_k) < 1:
            raise ValueError("train_top_k must be >= 1 when train_mask is given")
        Kt = int(train_top_k)
        train_sim = torch.full((N, Kt), float("-inf"), dtype=torch.float32, device=device)
        train_idx = torch.full((N, Kt), -1, dtype=torch.int64, device=device)

    total_pairs = N * (N - 1) // 2
    pairs_done = 0
    log_every = max(1, N // 20)
    t0 = time.time()
    for i in range(N - 1):
        i_is_train = bool(train_mask[i]) if train_mask is not None else False
        for start in range(i + 1, N, chunk_size):
            end = min(start + chunk_size, N)
            sims = sims_fn(i, start, end).to(device=device, dtype=torch.float32)
            j = torch.arange(start, end, device=device, dtype=torch.int64)
            _merge_topk_row(best_sim, best_idx, i, sims, j)
            _merge_topk_cols(best_sim, best_idx, j, sims, i)
            if i_is_train:
                tm = train_mask[start:end]
                if bool(tm.any()):
                    sims_t = sims[tm]
                    j_t = j[tm]
                    _merge_topk_row(train_sim, train_idx, i, sims_t, j_t)
                    _merge_topk_cols(train_sim, train_idx, j_t, sims_t, i)
            pairs_done += end - start
        if (i + 1) % log_every == 0 or i + 2 == N:
            elapsed = time.time() - t0
            rate = pairs_done / max(elapsed, 1e-9)
            eta = (total_pairs - pairs_done) / max(rate, 1e-9)
            print(f"{log_prefix}  [{i + 1}/{N} queries | "
                  f"{pairs_done}/{total_pairs} pairs] elapsed={elapsed:.0f}s "
                  f"rate={rate:.0f} pairs/s eta={eta:.0f}s")
    return best_sim, best_idx, train_sim, train_idx


def _unpermute_table(
    sim_sorted: torch.Tensor,   # [N, K] in length-sorted row space
    idx_sorted: torch.Tensor,   # [N, K] ids in length-sorted space; -1 = empty
    perm: torch.Tensor,         # [N] sorted position -> original clip idx
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Map a table built in length-sorted space back to original clip indices."""
    N, K = sim_sorted.shape
    perm = perm.to(idx_sorted.device)
    safe = torch.where(idx_sorted >= 0, idx_sorted, torch.zeros_like(idx_sorted))
    ids_orig = torch.where(idx_sorted >= 0, perm[safe], torch.full_like(idx_sorted, -1))
    out_idx = torch.full((N, K), -1, dtype=torch.int32)
    out_sim = torch.full((N, K), float("-inf"), dtype=torch.float32)
    out_idx[perm.cpu()] = ids_orig.to(torch.int32).cpu()
    out_sim[perm.cpu()] = sim_sorted.cpu()
    return out_idx, out_sim


def build_dtw_neighbors_exhaustive_from_payload(
    payload: dict,
    dtw_design: str,
    output_path: Path,
    top_k: int = 20,
    device: str = "cuda",
    chunk_size: int = 4096,
    train_positions=None,
    train_pool_top_k: int | None = None,
    train_pool_output_path: Path | None = None,
) -> None:
    """Exact all-pairs DTW top-K over the whole corpus (no FAISS stage).

    Every unordered pair is evaluated exactly once on the GPU and merged
    into both rows' running top-K, so the output is the true symmetric-2
    DTW top-K for every clip. The corpus is processed in length-sorted
    order so each target chunk spans few distinct `L_j` buckets (fewer
    kernel launches); indices are mapped back to original clip space before
    saving.

    If `train_positions` + `train_pool_top_k` + `train_pool_output_path` are
    given, the train-pool table (train queries x train candidates, keyed by
    global clip idx, non-train rows `-1`; same schema as
    `build_train_pool_neighbors`) is produced from the *same* pass at no
    extra DTW cost.

    Payload schema matches the FAISS path with `candidate_k=-1`,
    `pooling=None`, `dtw_mode="exact_all_pairs"`.
    """
    print(f"[build_dtw_neighbors] design={dtw_design}  top_k={top_k}  "
          f"mode=exact_all_pairs (no FAISS; every pair evaluated once)")
    if payload.get("design") != dtw_design:
        raise RuntimeError(
            f"Trajectories design mismatch: file={payload.get('design')!r} "
            f"vs requested={dtw_design!r}"
        )
    trajs_cpu = payload["trajectories"]
    lengths_cpu = payload["lengths"]
    clip_keys = payload["clip_keys"]
    N, T, D = trajs_cpu.shape
    print(f"[build_dtw_neighbors] N={N}  T={T}  D={D}  "
          f"pairs={N * (N - 1) // 2}")

    want_train_pool = train_positions is not None
    if want_train_pool and (train_pool_top_k is None or train_pool_output_path is None):
        raise ValueError(
            "train_positions requires train_pool_top_k and train_pool_output_path"
        )

    t_start = time.time()

    from online_dtw import OnlineDTWComputer
    computer = OnlineDTWComputer(dtw_design)

    # Length-sorted processing order (stable so ties keep clip order).
    perm = torch.argsort(lengths_cpu.to(torch.int64), stable=True)
    trajs_gpu = trajs_cpu[perm].to(device).float().contiguous()
    lengths_f_gpu = lengths_cpu[perm].to(device, dtype=torch.float32)

    train_mask_sorted = None
    if want_train_pool:
        is_train = torch.zeros(N, dtype=torch.bool)
        is_train[torch.as_tensor(sorted(int(p) for p in train_positions), dtype=torch.int64)] = True
        train_mask_sorted = is_train[perm].to(device)

    def _sims(i: int, start: int, end: int) -> torch.Tensor:
        return computer.compute_one_query_against_targets(
            trajs_gpu[i],
            trajs_gpu[start:end],
            lengths_f_gpu[i],
            lengths_f_gpu[start:end],
        )

    print(f"[build_dtw_neighbors] exact all-pairs DTW on {N} clips "
          f"(chunk_size={chunk_size})...")
    best_sim, best_idx, train_sim, train_idx = _exhaustive_topk(
        N, _sims, top_k, device=trajs_gpu.device, chunk_size=chunk_size,
        train_mask=train_mask_sorted, train_top_k=train_pool_top_k,
    )
    dtw_seconds = time.time() - t_start
    print(f"  exact_all_pairs DTW: {dtw_seconds:.1f}s")

    neighbor_clip_idx, neighbor_dtw_sim = _unpermute_table(best_sim, best_idx, perm)
    build_seconds = time.time() - t_start

    out_payload = {
        "neighbor_clip_idx": neighbor_clip_idx,
        "neighbor_dtw_sim":  neighbor_dtw_sim,
        "design":            dtw_design,
        "top_k":             int(top_k),
        "candidate_k":       -1,
        "pooling":           None,
        "clip_keys":         clip_keys,
        "N":                 N,
        "build_seconds":     build_seconds,
        "schema_version":    1,
        "dtw_mode":          "exact_all_pairs",
    }
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(out_payload, output_path)
    size_mb = output_path.stat().st_size / 1e6
    print(f"[build_dtw_neighbors] saved {output_path} ({size_mb:.1f} MB) "
          f"in {build_seconds:.1f}s")

    if want_train_pool:
        train_clip_idx, train_dtw_sim = _unpermute_table(train_sim, train_idx, perm)
        local_to_global = np.asarray(sorted(int(p) for p in train_positions), dtype=np.int64)
        train_payload = dict(out_payload)
        train_payload["neighbor_clip_idx"] = train_clip_idx
        train_payload["neighbor_dtw_sim"] = train_dtw_sim
        train_payload["top_k"] = int(train_pool_top_k)
        train_payload["N_train"] = int(local_to_global.shape[0])
        train_payload["train_clip_indices"] = torch.from_numpy(local_to_global)
        train_pool_output_path = Path(train_pool_output_path)
        train_pool_output_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(train_payload, train_pool_output_path)
        size_mb = train_pool_output_path.stat().st_size / 1e6
        print(f"[build_train_pool_neighbors] saved {train_pool_output_path} "
              f"({size_mb:.1f} MB) covering {local_to_global.shape[0]}/{N} clips "
              f"(from the same exact all-pairs pass)")


def build_train_pool_neighbors(
    trajectories_path: Path,
    train_positions,
    dtw_design: str,
    output_path: Path,
    top_k: int = 20,
    candidate_k: int = 100,
    pooling: str = "stats4",
    device: str = "cuda",
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
) -> None:
    """Compare cached top-K against brute-force exact DTW on a random
    sample of `sample_n` query clips. Reports recall@K (cached top-K hits
    inside the brute-force top-K) of the candidate-filter stage.
    """
    print(f"[sanity-check] sampling {sample_n} clips for brute-force DTW "
          "comparison...")
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
    lengths_f_gpu = lengths_cpu.to(device, dtype=torch.float32)

    from online_dtw import OnlineDTWComputer
    computer = OnlineDTWComputer(dtw_design)

    # Brute-force top-K over every clip, chunked to bound the cost tensor.
    chunk_size = 4096

    def _sims_chunk(query_idx: int, start: int, end: int) -> torch.Tensor:
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
    print(f"[sanity-check] candidate_k={cache.get('candidate_k')} "
          f"pooling={cache.get('pooling')} dtw_mode={cache.get('dtw_mode')}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trajectories", type=Path, required=True,
                        help="Path to trajectories_<design>.pt")
    parser.add_argument("--design", type=str, required=True,
                        help="DTW design key (must match the trajectories file)")
    parser.add_argument("--output", type=Path, required=True,
                        help="Output .pt path for the neighbor table")
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--candidate-k", type=int, default=100,
                        help="FAISS candidates per clip; <= 0 = exhaustive "
                             "exact all-pairs DTW (no FAISS)")
    parser.add_argument("--pooling", type=str, default="stats4")
    parser.add_argument("--sanity-check", action="store_true",
                        help="After building, sample 1000 clips and compare "
                             "cached top-K against brute-force exact DTW")
    args = parser.parse_args()

    build_dtw_neighbors(
        trajectories_path=args.trajectories,
        dtw_design=args.design,
        output_path=args.output,
        top_k=args.top_k,
        candidate_k=args.candidate_k,
        pooling=args.pooling,
    )

    if args.sanity_check:
        _sanity_check(args.trajectories, args.design, args.output)


if __name__ == "__main__":
    main()
