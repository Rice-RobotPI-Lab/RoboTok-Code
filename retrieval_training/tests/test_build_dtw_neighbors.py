"""Tests for retrieval_training/build_dataset/build_dtw_neighbors.py.

Covers the CPU-only stages (`_pool_trajectories`, `_faiss_top_candidates`).
The DTW step requires CUDA and is exercised by the integration test in
`tests/test_online_dtw.py` (gated on `torch.cuda.is_available()`).

Run from retrieval_training/: python -m pytest tests/test_build_dtw_neighbors.py -v
"""

import sys
from pathlib import Path

import pytest
import torch

THIS_DIR = Path(__file__).resolve().parent
TRAINING_DIR = THIS_DIR.parent
sys.path.insert(0, str(TRAINING_DIR))
sys.path.insert(0, str(TRAINING_DIR / "build_dataset"))

from build_dtw_neighbors import _faiss_top_candidates, _pool_trajectories

try:
    import faiss  # noqa: F401
    HAS_FAISS = True
except ImportError:
    HAS_FAISS = False

requires_faiss = pytest.mark.skipif(not HAS_FAISS, reason="faiss not installed")


# ---------------------------------------------------------------------------
# _pool_trajectories
# ---------------------------------------------------------------------------

class TestPoolTrajectories:
    def test_output_shape(self):
        N, T, D = 8, 12, 4
        trajs = torch.randn(N, T, D)
        lengths = torch.full((N,), T, dtype=torch.int32)
        out = _pool_trajectories(trajs, lengths)
        # stats4 → 4*D dims
        assert out.shape == (N, 4 * D)

    def test_unknown_pooling_raises(self):
        trajs = torch.randn(2, 4, 3)
        lengths = torch.tensor([4, 4], dtype=torch.int32)
        with pytest.raises(ValueError, match="pooling"):
            _pool_trajectories(trajs, lengths, pooling="banana")

    def test_full_length_known_values(self):
        # Trajectory: [0, 1, 2, 3] in dim 0, all zeros in dim 1.
        trajs = torch.tensor([[
            [0.0, 0.0],
            [1.0, 0.0],
            [2.0, 0.0],
            [3.0, 0.0],
        ]])
        lengths = torch.tensor([4], dtype=torch.int32)
        out = _pool_trajectories(trajs, lengths)
        # mean over t = [1.5, 0.0]
        # std (population) = [sqrt(1.25), 0.0] ≈ [1.118, 0.0]
        # first = [0, 0]
        # last = [3, 0]
        expected = torch.tensor([[
            1.5, 0.0,                # mean
            (1.25) ** 0.5, 0.0,      # std (population)
            0.0, 0.0,                # first
            3.0, 0.0,                # last
        ]])
        assert torch.allclose(out, expected, atol=1e-5)

    def test_padded_length_uses_only_valid_prefix(self):
        # Length=2 means trajs[2:] is "padding" and must be ignored by pool.
        trajs = torch.tensor([[
            [1.0, 0.0],
            [3.0, 0.0],
            [99.0, 99.0],   # padding
            [99.0, 99.0],   # padding
        ]])
        lengths = torch.tensor([2], dtype=torch.int32)
        out = _pool_trajectories(trajs, lengths)
        # Effective traj = [[1, 0], [3, 0]]
        # mean = [2.0, 0.0]
        # std = [1.0, 0.0]
        # first = [1.0, 0.0]
        # last = [3.0, 0.0]   ← MUST be index length-1, not T-1
        expected = torch.tensor([[2.0, 0.0, 1.0, 0.0, 1.0, 0.0, 3.0, 0.0]])
        assert torch.allclose(out, expected, atol=1e-5)

    def test_length_one_no_nan(self):
        # Edge case: single-frame trajectory. std should be 0, not NaN.
        trajs = torch.tensor([[
            [5.0, 7.0],
            [99.0, 99.0],
        ]])
        lengths = torch.tensor([1], dtype=torch.int32)
        out = _pool_trajectories(trajs, lengths)
        assert torch.isfinite(out).all()
        # std of a single point = 0
        assert torch.allclose(out[0, 2:4], torch.tensor([0.0, 0.0]))

    def test_invalid_lengths_raise(self):
        trajs = torch.randn(2, 4, 3)
        with pytest.raises(ValueError, match="length"):
            _pool_trajectories(trajs, torch.tensor([0, 4], dtype=torch.int32))
        with pytest.raises(ValueError, match="exceed"):
            _pool_trajectories(trajs, torch.tensor([4, 5], dtype=torch.int32))


class TestPoolTrajectoriesFlatFull:
    def test_output_shape(self):
        N, T, D = 8, 12, 4
        trajs = torch.randn(N, T, D)
        lengths = torch.full((N,), T, dtype=torch.int32)
        out = _pool_trajectories(trajs, lengths, pooling="flat_full")
        assert out.shape == (N, T * D)

    def test_full_length_equals_flatten(self):
        # When length == T (no padding), flat_full should equal the raw
        # row-major flatten of the trajectory.
        trajs = torch.tensor([[
            [1.0, 2.0],
            [3.0, 4.0],
            [5.0, 6.0],
        ]])
        lengths = torch.tensor([3], dtype=torch.int32)
        out = _pool_trajectories(trajs, lengths, pooling="flat_full")
        assert torch.equal(out, torch.tensor([[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]]))

    def test_padding_is_zeroed(self):
        # length=2 means the trailing frame should NOT contribute to the
        # pooled vector (zeroed out, not included).
        trajs = torch.tensor([[
            [1.0, 2.0],
            [3.0, 4.0],
            [99.0, 99.0],   # padding — must be zeroed in output
        ]])
        lengths = torch.tensor([2], dtype=torch.int32)
        out = _pool_trajectories(trajs, lengths, pooling="flat_full")
        # Expected: [1, 2, 3, 4, 0, 0]
        assert torch.equal(out, torch.tensor([[1.0, 2.0, 3.0, 4.0, 0.0, 0.0]]))

    def test_unknown_pooling_lists_supported(self):
        trajs = torch.randn(2, 4, 3)
        lengths = torch.tensor([4, 4], dtype=torch.int32)
        with pytest.raises(ValueError, match="flat_full|stats4"):
            _pool_trajectories(trajs, lengths, pooling="banana")

    def test_finite_values(self):
        # Edge case: length=1 trajectory in the batch — make sure the masked
        # flatten doesn't introduce NaN.
        trajs = torch.tensor([[
            [5.0, 7.0],
            [99.0, 99.0],
            [99.0, 99.0],
        ]])
        lengths = torch.tensor([1], dtype=torch.int32)
        out = _pool_trajectories(trajs, lengths, pooling="flat_full")
        assert torch.isfinite(out).all()
        # Only first frame should be non-zero
        assert torch.equal(out, torch.tensor([[5.0, 7.0, 0.0, 0.0, 0.0, 0.0]]))


class TestPoolTrajectoriesKeyframes8:
    def test_output_shape(self):
        N, T, D = 8, 12, 4
        trajs = torch.randn(N, T, D)
        lengths = torch.full((N,), T, dtype=torch.int32)
        out = _pool_trajectories(trajs, lengths, pooling="keyframes8")
        assert out.shape == (N, 8 * D)

    def test_keyframes_evenly_spaced_at_full_length(self):
        # T=8, length=8 → keyframes at indices [0, 1, 2, 3, 4, 5, 6, 7]
        # (round of [0, 1, 2, 3, 4, 5, 6, 7] = same).
        T, D = 8, 2
        trajs = torch.arange(T * D, dtype=torch.float).reshape(1, T, D)
        # trajs[0] = [[0,1],[2,3],[4,5],[6,7],[8,9],[10,11],[12,13],[14,15]]
        lengths = torch.tensor([T], dtype=torch.int32)
        out = _pool_trajectories(trajs, lengths, pooling="keyframes8")
        # Expected: full flatten in this case
        assert torch.equal(out[0], trajs[0].reshape(-1))

    def test_keyframes_at_length_15(self):
        # T=20, length=15 → indices = round([0, 2, 4, 6, 8, 10, 12, 14]) for
        # n_kf=8: t_steps = [0, 1/7, 2/7, ..., 1] × 14 = [0, 2, 4, 6, 8, 10, 12, 14]
        T, D = 20, 1
        trajs = torch.arange(T, dtype=torch.float).reshape(1, T, 1)
        lengths = torch.tensor([15], dtype=torch.int32)
        out = _pool_trajectories(trajs, lengths, pooling="keyframes8")
        # Each keyframe is just the index value (since D=1 and trajs[t,0]=t).
        expected_indices = (torch.arange(8) * 14.0 / 7.0).round().long()
        expected = trajs[0, expected_indices, 0]
        assert torch.equal(out[0], expected)

    def test_short_clip_repeats_keyframes(self):
        # length=1 → every keyframe is index 0 (the only valid frame).
        trajs = torch.tensor([[
            [3.0, 5.0],
            [99.0, 99.0],
            [99.0, 99.0],
        ]])
        lengths = torch.tensor([1], dtype=torch.int32)
        out = _pool_trajectories(trajs, lengths, pooling="keyframes8")
        # All 8 keyframes should be [3.0, 5.0]
        expected = torch.tensor([[3.0, 5.0] * 8])
        assert torch.equal(out, expected)

    def test_keyframes_only_valid_prefix(self):
        # length=4 means padding region (t=4..7) must NOT appear in output.
        T, D = 8, 1
        trajs = torch.tensor([[
            [1.0], [2.0], [3.0], [4.0],   # valid
            [99.0], [99.0], [99.0], [99.0],  # padding
        ]])
        lengths = torch.tensor([4], dtype=torch.int32)
        out = _pool_trajectories(trajs, lengths, pooling="keyframes8")
        # 8 keyframes evenly spaced over [0, 3] → indices round([0, 3/7, 6/7,
        # 9/7, 12/7, 15/7, 18/7, 21/7] × 1) = round([0, 0.43, 0.86, 1.29,
        # 1.71, 2.14, 2.57, 3.0]) = [0, 0, 1, 1, 2, 2, 3, 3]
        # Values: [1, 1, 2, 2, 3, 3, 4, 4]
        assert torch.equal(out[0], torch.tensor([1.0, 1.0, 2.0, 2.0, 3.0, 3.0, 4.0, 4.0]))
        # No 99.0 anywhere
        assert (out != 99.0).all()

    def test_finite_values(self):
        trajs = torch.randn(5, 10, 6)
        lengths = torch.tensor([1, 3, 5, 7, 10], dtype=torch.int32)
        out = _pool_trajectories(trajs, lengths, pooling="keyframes8")
        assert torch.isfinite(out).all()


# ---------------------------------------------------------------------------
# _faiss_top_candidates
# ---------------------------------------------------------------------------

@requires_faiss
class TestFaissTopCandidates:
    def test_output_shape(self):
        pooled = torch.randn(20, 16)
        cand = _faiss_top_candidates(pooled, candidate_k=5)
        assert cand.shape == (20, 5)

    def test_no_self_in_output(self):
        pooled = torch.randn(20, 16)
        cand = _faiss_top_candidates(pooled, candidate_k=5)
        for i in range(20):
            assert i not in cand[i].tolist()

    def test_nearest_is_geometrically_nearest(self):
        # Place 4 vectors such that 0 and 1 are colinear-similar, and 2 and
        # 3 are colinear-similar; so 0's top neighbor should be 1 (and vice
        # versa), 2's should be 3.
        pooled = torch.tensor([
            [1.0, 0.0],
            [0.99, 0.01],   # very close to 0
            [-1.0, 0.0],
            [-0.99, -0.01], # very close to 2
        ])
        cand = _faiss_top_candidates(pooled, candidate_k=1)
        assert cand[0, 0].item() == 1
        assert cand[1, 0].item() == 0
        assert cand[2, 0].item() == 3
        assert cand[3, 0].item() == 2

    def test_returns_int64(self):
        pooled = torch.randn(10, 8)
        cand = _faiss_top_candidates(pooled, candidate_k=3)
        assert cand.dtype == torch.int64


# ---------------------------------------------------------------------------
# Exhaustive all-pairs mode (no FAISS)
# ---------------------------------------------------------------------------

from build_dtw_neighbors import (  # noqa: E402
    _exhaustive_topk,
    _unpermute_table,
    build_dtw_neighbors_exhaustive_from_payload,
    build_dtw_neighbors_from_payload,
)


def _brute_force_topk(sim: torch.Tensor, K: int, mask: torch.Tensor | None = None):
    """Reference top-K from a full [N, N] similarity matrix (self excluded).

    With `mask`, only rows/cols where mask is True participate; other rows
    are all -1 / -inf.
    """
    N = sim.shape[0]
    s = sim.clone()
    s.fill_diagonal_(float("-inf"))
    if mask is not None:
        s[~mask, :] = float("-inf")
        s[:, ~mask] = float("-inf")
    k = min(K, N - 1)
    vals, idx = s.topk(k, dim=1)
    out_idx = torch.full((N, K), -1, dtype=torch.int64)
    out_sim = torch.full((N, K), float("-inf"))
    out_idx[:, :k] = torch.where(torch.isfinite(vals), idx, torch.full_like(idx, -1))
    out_sim[:, :k] = vals
    if mask is not None:
        out_idx[~mask] = -1
        out_sim[~mask] = float("-inf")
    return out_idx, out_sim


def _assert_same_topk(idx, sim, ref_idx, ref_sim):
    # Compare sims exactly and ids as sets per row (ties may reorder ids).
    assert torch.allclose(sim.float(), ref_sim.float(), equal_nan=True), (sim, ref_sim)
    for r in range(idx.shape[0]):
        a = idx[r].to(torch.int64)
        b = ref_idx[r].to(torch.int64)
        assert set(a[a >= 0].tolist()) == set(b[b >= 0].tolist()), (r, a, b)


class TestExhaustiveTopK:
    """Pure-CPU check of the pair loop + symmetric top-K merge."""

    def _random_sym_sim(self, N, seed=0):
        g = torch.Generator().manual_seed(seed)
        x = torch.randn(N, 5, generator=g)
        return -torch.cdist(x, x)  # symmetric, zero diagonal

    @pytest.mark.parametrize("N,K,chunk", [(7, 3, 2), (25, 5, 4), (25, 40, 7), (40, 1, 40)])
    def test_matches_brute_force(self, N, K, chunk):
        sim = self._random_sym_sim(N)
        calls = []

        def sims_fn(i, start, end):
            calls.append((i, start, end))
            return sim[i, start:end]

        best_sim, best_idx, ts, ti = _exhaustive_topk(
            N, sims_fn, K, device="cpu", chunk_size=chunk,
        )
        assert ts is None and ti is None
        ref_idx, ref_sim = _brute_force_topk(sim, K)
        _assert_same_topk(best_idx, best_sim, ref_idx, ref_sim)
        # Every unordered pair evaluated exactly once, only for j > i.
        n_pairs = sum(end - start for _, start, end in calls)
        assert n_pairs == N * (N - 1) // 2
        assert all(start > i for i, start, _ in calls)

    def test_train_pool_table_from_same_pass(self):
        N, K, Kt = 30, 4, 6
        sim = self._random_sym_sim(N, seed=1)
        g = torch.Generator().manual_seed(3)
        mask = torch.rand(N, generator=g) < 0.7
        mask[0] = True
        best_sim, best_idx, ts, ti = _exhaustive_topk(
            N, lambda i, s, e: sim[i, s:e], K, device="cpu", chunk_size=5,
            train_mask=mask, train_top_k=Kt,
        )
        ref_idx, ref_sim = _brute_force_topk(sim, K)
        _assert_same_topk(best_idx, best_sim, ref_idx, ref_sim)
        tref_idx, tref_sim = _brute_force_topk(sim, Kt, mask=mask)
        _assert_same_topk(ti, ts, tref_idx, tref_sim)
        # non-train rows untouched; train rows never cite non-train clips
        assert (ti[~mask] == -1).all()
        cited = ti[mask]
        assert mask[cited[cited >= 0]].all()

    def test_unpermute_roundtrip(self):
        perm = torch.tensor([3, 0, 2, 1])
        idx_sorted = torch.tensor([[1, 2, -1], [0, 3, -1], [3, 0, 1], [-1, -1, -1]])
        sim_sorted = torch.tensor([[0.9, 0.5, float("-inf")],
                                   [0.8, 0.1, float("-inf")],
                                   [0.7, 0.6, 0.2],
                                   [float("-inf")] * 3])
        out_idx, out_sim = _unpermute_table(sim_sorted, idx_sorted, perm)
        # sorted row r -> original row perm[r]; ids map through perm too.
        assert out_idx[3].tolist() == [0, 2, -1]
        assert out_idx[0].tolist() == [3, 1, -1]
        assert out_idx[2].tolist() == [1, 3, 0]
        assert out_idx[1].tolist() == [-1, -1, -1]
        assert torch.equal(out_sim[3], sim_sorted[0])
        assert out_idx.dtype == torch.int32


requires_cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")


@requires_cuda
class TestExhaustiveBuilderCUDA:
    def _payload(self, N=37, T=12, D=6, seed=0):
        g = torch.Generator().manual_seed(seed)
        trajs = torch.randn(N, T, D, generator=g)
        lengths = torch.randint(3, T + 1, (N,), generator=g).to(torch.int32)
        return {
            "trajectories": trajs,
            "lengths": lengths,
            "clip_keys": [(0, i) for i in range(N)],
            "design": "abs_21j_coords",
        }

    def test_matches_compute_chunked(self, tmp_path):
        from online_dtw import OnlineDTWComputer
        payload = self._payload()
        N = payload["trajectories"].shape[0]
        K, Kt = 5, 8
        train_positions = list(range(0, N, 3)) + [N - 1]
        out = tmp_path / "neigh.pt"
        out_t = tmp_path / "neigh_train.pt"
        build_dtw_neighbors_exhaustive_from_payload(
            payload, "abs_21j_coords", out, top_k=K, chunk_size=8,
            train_positions=train_positions, train_pool_top_k=Kt,
            train_pool_output_path=out_t,
        )
        computer = OnlineDTWComputer("abs_21j_coords")
        sim = computer.compute_chunked(
            payload["trajectories"].cuda(), payload["lengths"].cuda(), target_chunk=16,
        ).cpu()

        neigh = torch.load(out, weights_only=False)
        assert neigh["dtw_mode"] == "exact_all_pairs"
        assert neigh["candidate_k"] == -1 and neigh["pooling"] is None
        assert neigh["top_k"] == K
        ref_idx, ref_sim = _brute_force_topk(sim, K)
        assert torch.allclose(neigh["neighbor_dtw_sim"], ref_sim, atol=1e-5)
        for r in range(N):
            assert set(neigh["neighbor_clip_idx"][r].tolist()) == set(ref_idx[r].tolist())

        tn = torch.load(out_t, weights_only=False)
        mask = torch.zeros(N, dtype=torch.bool)
        mask[train_positions] = True
        assert tn["top_k"] == Kt and tn["N_train"] == len(train_positions)
        assert tn["train_clip_indices"].tolist() == sorted(train_positions)
        tref_idx, tref_sim = _brute_force_topk(sim, Kt, mask=mask)
        assert torch.allclose(tn["neighbor_dtw_sim"], tref_sim, atol=1e-5)
        for r in range(N):
            assert set(tn["neighbor_clip_idx"][r].tolist()) == set(tref_idx[r].tolist())

    def test_from_payload_dispatches_on_nonpositive_candidate_k(self, tmp_path):
        payload = self._payload(N=12)
        out = tmp_path / "n.pt"
        build_dtw_neighbors_from_payload(
            payload, "abs_21j_coords", out, top_k=3, candidate_k=-1, pooling="stats4",
        )
        assert torch.load(out, weights_only=False)["dtw_mode"] == "exact_all_pairs"
