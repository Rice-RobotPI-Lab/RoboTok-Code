"""Online per-batch DTW similarity matrix.

Computes a (B, B) similarity matrix from per-clip trajectories on the fly,
replacing the offline lookup into `dtw_matrix_{design}.npy`. Mirrors the
offline neighbor-building pipeline (`build_dataset/build_dtw_neighbors.py`)
so per-batch results are bit-equivalent to the corresponding submatrix of
the precomputed matrix.

Design semantics (cost / kernel / norm) come from `dtw_cknna.py` via
`DTW_DESIGNS[design]`. The trajectory transform (`tx_fn`) is *not*
re-applied here — `build_trajectories.py` saves trajectories
post-transform.

Example:
    computer = OnlineDTWComputer("first_frame_midpoint_hand_length_no_z")
    sims = computer.compute(trajs_gpu, lengths_gpu)   # [B, B] similarity (= -distance)
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch

# dtw_cknna.py lives alongside this module; make it importable regardless of
# which directory the entrypoint script was launched from.
_CKNNA_DIR = Path(__file__).resolve().parent
if str(_CKNNA_DIR) not in sys.path:
    sys.path.insert(0, str(_CKNNA_DIR))


class OnlineDTWComputer:
    """Cached DTW configuration + kernel for one design.

    Reuses the compiled numba CUDA kernel across calls so we don't pay JIT
    or kernel-creation cost per batch.
    """

    def __init__(self, dtw_design: str):
        from dtw_cknna import DTW_DESIGNS, DTW_METHODS, DTW_NORMS

        if dtw_design not in DTW_DESIGNS:
            raise ValueError(
                f"Unknown dtw_design={dtw_design!r}; "
                f"valid: {sorted(DTW_DESIGNS.keys())}"
            )
        d = DTW_DESIGNS[dtw_design]
        method = DTW_METHODS[d["cost"]]

        self.design = dtw_design
        self.cost_fn = method["cost_fn"]
        self.kernel = method["kernel_fn"]()
        self.norm_fn = DTW_NORMS[d["norm"]]
        self.norm_key = d["norm"]

    def compute(self, trajs: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        """Return (B, B) similarity matrix on the same device as `trajs`.

        Args:
            trajs:   [B, T, D] float32 — already passed through tx_fn.
            lengths: [B] int/float — actual (pre-pad) clip lengths.

        Returns:
            [B, B] float32 similarity = negated DTW distance, on `trajs.device`.
            Diagonal is exactly 0 (distance to self is 0 under symmetric2 with
            our seeding convention; the loop forces it explicitly to absorb any
            kernel-internal noise).
        """
        from numba import cuda as numba_cuda

        if trajs.device.type != "cuda":
            raise RuntimeError(
                "OnlineDTWComputer.compute requires CUDA tensors; "
                f"got device={trajs.device}"
            )
        if trajs.dtype != torch.float32:
            trajs = trajs.float()
        lengths_f = lengths.to(trajs.device, dtype=torch.float32)

        B, T, _ = trajs.shape
        device = trajs.device
        n_passes = 2 * T - 1

        dist_matrix = torch.zeros(B, B, dtype=torch.float32, device=device)

        # Iterate query-by-query so the cost tensor stays [B, T, T] (~B*T²),
        # not [B, B, T, T]. Mirrors `compute_dtw_matrix_batched` exactly.
        for i in range(B):
            D_cost = self.cost_fn(trajs[i], trajs)            # [B, T, T]

            R_buf = torch.full(
                (B, T + 2, T + 2), float("inf"),
                device=device, dtype=torch.float32,
            )
            # Seed -D[0,0] so the kernel's "always add d" recurrence makes the
            # corner cell count d[0,0] exactly once (matches dtw-python sym2).
            R_buf[:, 0, 0] = -D_cost[:, 0, 0]

            self.kernel[B, T](
                numba_cuda.as_cuda_array(D_cost),
                0.0, T, T, n_passes,
                numba_cuda.as_cuda_array(R_buf),
            )
            distances = R_buf[:, -2, -2]                      # [B]

            if self.norm_key != "none":
                distances = self.norm_fn(distances, lengths_f, lengths_f[i])

            dist_matrix[i] = distances

        # Symmetric-2 DTW is mathematically symmetric. Force the diagonal to 0
        # since trajectory-to-self distance is degenerate at d[0,0]=0 anyway.
        dist_matrix.fill_diagonal_(0.0)

        return -dist_matrix

    def compute_one_query_against_targets(
        self,
        query_traj: torch.Tensor,         # [T, D] fp32 cuda
        target_trajs: torch.Tensor,       # [B, T, D] fp32 cuda
        query_length_f,                   # 0-dim tensor or float
        target_lengths_f: torch.Tensor,   # [B] fp32 cuda
        exact: bool = False,
    ) -> torch.Tensor:                    # [B] similarity (= -distance)
        """Rectangular query→targets DTW.

        `exact=False` (default, legacy): runs the kernel on the full padded
        `T_max × T_max` grid. Padded frames contribute non-zero cost.

        `exact=True`: bucket targets by their true length L_j and dispatch
        one kernel call per bucket on a truncated `L_i × L_j` grid; the
        returned distance is read from `R[L_i, L_j]` (the corner of the real
        prefix grid). Bit-equivalent to symmetric-2 DTW on the actual
        prefixes — same kernel, same cost_fn, same seeding, same norm_fn.

        Self-exclusion is the caller's responsibility — this helper just
        evaluates the kernel; if `query_traj` happens to also appear in
        `target_trajs`, the corresponding row's similarity will be ~0
        (distance to self ≈ 0) and the caller can drop it.
        """
        from numba import cuda as numba_cuda

        if target_trajs.device.type != "cuda":
            raise RuntimeError(
                "compute_one_query_against_targets requires CUDA tensors"
            )

        B, T, _ = target_trajs.shape
        device = target_trajs.device

        if exact:
            # Length-aware path. Bucket by L_j so each kernel call sees a
            # uniform target length.
            q_len_t = (
                query_length_f
                if torch.is_tensor(query_length_f)
                else torch.as_tensor(query_length_f, device=device, dtype=torch.float32)
            )
            L_i = int(q_len_t.item())
            q_trunc = query_traj[:L_i].contiguous()
            distances = torch.empty(B, device=device, dtype=torch.float32)
            for L_j_t in torch.unique(target_lengths_f):
                L_j = int(L_j_t.item())
                local_pos = (target_lengths_f == L_j_t).nonzero(as_tuple=True)[0]
                tgt = target_trajs.index_select(0, local_pos)[:, :L_j, :].contiguous()
                tgt_lens_f = target_lengths_f.index_select(0, local_pos)
                d = self._exact_block(q_trunc, tgt, q_len_t, tgt_lens_f)
                distances.index_copy_(0, local_pos, d)
            return -distances

        # Padded path.
        n_passes = 2 * T - 1
        D_cost = self.cost_fn(query_traj, target_trajs)  # [B, T, T]
        R_buf = torch.full(
            (B, T + 2, T + 2), float("inf"),
            device=device, dtype=torch.float32,
        )
        R_buf[:, 0, 0] = -D_cost[:, 0, 0]
        self.kernel[B, T](
            numba_cuda.as_cuda_array(D_cost),
            0.0, T, T, n_passes,
            numba_cuda.as_cuda_array(R_buf),
        )
        distances = R_buf[:, -2, -2]
        if self.norm_key != "none":
            distances = self.norm_fn(distances, target_lengths_f, query_length_f)
        return -distances

    def _exact_block(
        self,
        q_trunc: torch.Tensor,            # [L_i, D] fp32 cuda
        target_trajs: torch.Tensor,       # [B, L_j, D] fp32 cuda (uniform L_j)
        query_length_f: torch.Tensor,     # 0-dim fp32 cuda
        target_lengths_f: torch.Tensor,   # [B] fp32 cuda (all == L_j)
    ) -> torch.Tensor:                    # [B] distance (length-normalised)
        """One kernel call on a truncated `L_i × L_j` grid; helper for
        `compute_one_query_against_targets(exact=True)`."""
        from numba import cuda as numba_cuda

        L_i = q_trunc.shape[0]
        B, L_j, _ = target_trajs.shape
        device = target_trajs.device

        D_cost = self.cost_fn(q_trunc, target_trajs)  # [B, L_i, L_j]
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
            distances = self.norm_fn(distances, target_lengths_f, query_length_f)
        return distances

    def compute_chunked(
        self,
        trajs: torch.Tensor,
        lengths: torch.Tensor,
        target_chunk: int = 1024,
        exact: bool = False,
    ) -> torch.Tensor:
        """Build the full (N, N) similarity matrix with bounded peak memory.

        Like `compute(...)` but iterates query rows individually and chunks
        target columns so the intermediate cost-difference tensor stays small.
        Use this for eval-set–scale matrices (N ~ 10k) where compute() would
        materialise a huge [N, T, T, D] broadcast tensor per query row.

        Result is bit-identical to `compute(...)` (same cost_fn, same kernel,
        same seeding, same norm_fn) — the per-chunk body is delegated to
        `compute_one_query_against_targets`.
        """
        if trajs.device.type != "cuda":
            raise RuntimeError("compute_chunked requires CUDA tensors")
        if trajs.dtype != torch.float32:
            trajs = trajs.float()
        lengths_f = lengths.to(trajs.device, dtype=torch.float32)

        N = trajs.shape[0]
        device = trajs.device
        sim = torch.zeros(N, N, dtype=torch.float32, device=device)

        for i in range(N):
            for j_start in range(0, N, target_chunk):
                j_end = min(j_start + target_chunk, N)
                sim[i, j_start:j_end] = self.compute_one_query_against_targets(
                    trajs[i],
                    trajs[j_start:j_end],
                    lengths_f[i],
                    lengths_f[j_start:j_end],
                    exact=exact,
                )

        sim.fill_diagonal_(0.0)
        return sim
