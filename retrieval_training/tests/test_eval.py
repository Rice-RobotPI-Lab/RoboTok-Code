"""Tests for retrieval_training/eval.py.

Run from retrieval_training/: python -m pytest tests/test_eval.py -v
"""

import sys
from pathlib import Path

import numpy as np
import pytest
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from eval import (
    RELEVANCE_TOP_N,
    _asymmetric_platonic_cknna,
    _cosine_topk,
    _global_model_neighbors,
    _hsic_unbiased,
    _indicator_recall_at_k,
    _pool_metrics_from_encoded,
    _rectangular_centered_inner,
    _sim_topk_mask,
    _topn_indices_from_sim,
    compute_cknna,
    compute_cknna_at_k,
    compute_global_pool_cknna,
    compute_global_pool_cknna_at_k,
    compute_global_recall_at_k,
)

try:
    import faiss
    from eval import build_faiss_index
    HAS_FAISS = True
except ImportError:
    HAS_FAISS = False

requires_faiss = pytest.mark.skipif(not HAS_FAISS, reason="faiss not installed")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _random_l2_normed(n, d, seed=0):
    rng = np.random.default_rng(seed)
    x = rng.standard_normal((n, d)).astype(np.float32)
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    return x / norms


def _random_symmetric_np(n, seed=0):
    rng = np.random.default_rng(seed)
    raw = rng.standard_normal((n, n)).astype(np.float32)
    return (raw + raw.T) / 2


# ---------------------------------------------------------------------------
# build_faiss_index
# ---------------------------------------------------------------------------

@requires_faiss
class TestBuildFaissIndex:
    def test_index_size(self):
        emb = _random_l2_normed(20, 16)
        index = build_faiss_index(emb)
        assert index.ntotal == 20

    def test_dimension(self):
        emb = _random_l2_normed(20, 16)
        index = build_faiss_index(emb)
        assert index.d == 16

    def test_search_returns_self_as_top1(self):
        emb = _random_l2_normed(10, 16)
        index = build_faiss_index(emb)
        _, ids = index.search(emb, 1)
        for i in range(10):
            assert ids[i, 0] == i

    def test_flat_ip_type(self):
        emb = _random_l2_normed(10, 16)
        index = build_faiss_index(emb, "IndexFlatIP")
        assert isinstance(index, faiss.IndexFlatIP)


# ---------------------------------------------------------------------------
# _indicator_recall_at_k / _topn_indices_from_sim
# ---------------------------------------------------------------------------

def _make_neighbors(model_topk_pos: np.ndarray):
    """Build (neighbor_pos, valid_neighbor) inputs for `_indicator_recall_at_k`.

    `model_topk_pos[q]` lists the model's top-K valid neighbor positions for
    query `q`. All slots are marked valid (no self/padding to skip).
    """
    return model_topk_pos.astype(np.int64), np.ones_like(model_topk_pos, dtype=bool)


class TestTopNIndicesFromSim:
    def test_excludes_self(self):
        sim = np.eye(5, dtype=np.float32)  # diagonal = 1, off = 0
        topn = _topn_indices_from_sim(sim, 2)
        for q in range(5):
            assert q not in topn[q]

    def test_returns_top_by_similarity(self):
        sim = np.array([
            [0.0, 0.9, 0.5, 0.7],
            [0.9, 0.0, 0.3, 0.1],
            [0.5, 0.3, 0.0, 0.8],
            [0.7, 0.1, 0.8, 0.0],
        ], dtype=np.float32)
        topn = _topn_indices_from_sim(sim, 2)
        # Query 0: top-2 by sim are 1 (0.9) and 3 (0.7)
        assert set(topn[0].tolist()) == {1, 3}
        # Query 1: top-2 are 0 (0.9) and 2 (0.3)
        assert set(topn[1].tolist()) == {0, 2}

    def test_caps_at_n_minus_1(self):
        sim = np.zeros((3, 3), dtype=np.float32)
        topn = _topn_indices_from_sim(sim, 10)
        # Only 2 non-self neighbors per row
        assert topn.shape == (3, 2)


class TestIndicatorRecallAtK:
    def test_perfect_recall_when_model_matches_relevance(self):
        # 4 queries; each model top-3 equals each relevance top-3.
        topk = np.array([[1, 2, 3], [0, 2, 3], [0, 1, 3], [0, 1, 2]])
        relevance = topk.copy()
        neighbor_pos, valid = _make_neighbors(topk)
        results = _indicator_recall_at_k(neighbor_pos, valid, relevance, [1, 2, 3])
        assert results["R@1"] == 1.0
        assert results["R@2"] == 1.0
        assert results["R@3"] == 1.0

    def test_zero_recall_when_disjoint(self):
        # Model picks [4,5,6]; relevance is [1,2,3] — no overlap.
        topk = np.array([[4, 5, 6], [4, 5, 6]])
        relevance = np.array([[1, 2, 3], [1, 2, 3]])
        neighbor_pos, valid = _make_neighbors(topk)
        results = _indicator_recall_at_k(neighbor_pos, valid, relevance, [1, 2, 3])
        for v in results.values():
            assert v == 0.0

    def test_indicator_aggregation_not_count(self):
        # Query 0: 1 hit at slot 0. Query 1: 3 hits at all slots.
        # Indicator form: both contribute 1 to R@K for K covering the hit.
        topk = np.array([[1, 9, 9], [1, 2, 3]])
        relevance = np.array([[1, 2, 3], [1, 2, 3]])
        neighbor_pos, valid = _make_neighbors(topk)
        r = _indicator_recall_at_k(neighbor_pos, valid, relevance, [1, 2, 3])
        # Both queries hit at K=1 → R@1 = 1.0 (indicator, not count fraction)
        assert r["R@1"] == 1.0

    def test_monotonic_in_k(self):
        # Query has its only hit at slot 2. R@1=0, R@2=0, R@3=1.
        topk = np.array([[5, 6, 1, 7]])
        relevance = np.array([[1, 2, 3]])
        neighbor_pos, valid = _make_neighbors(topk)
        r = _indicator_recall_at_k(neighbor_pos, valid, relevance, [1, 2, 3, 4])
        assert r["R@1"] == 0.0
        assert r["R@2"] == 0.0
        assert r["R@3"] == 1.0
        assert r["R@4"] == 1.0

    def test_invalid_slots_skipped(self):
        # Slot 0 invalid (self-match), so "top-1 valid" is slot 1.
        topk = np.array([[0, 5, 6]], dtype=np.int64)
        valid = np.array([[False, True, True]])
        relevance = np.array([[5, 99, 100]])
        r = _indicator_recall_at_k(topk, valid, relevance, [1, 2])
        # Top-1 valid is position 5 → hit
        assert r["R@1"] == 1.0
        assert r["R@2"] == 1.0

    def test_prefix(self):
        topk = np.array([[1]])
        relevance = np.array([[1]])
        neighbor_pos, valid = _make_neighbors(topk)
        r = _indicator_recall_at_k(neighbor_pos, valid, relevance, [1], prefix="jepa_")
        assert "jepa_R@1" in r and "R@1" not in r


class TestRelevanceTopN:
    def test_default_is_20(self):
        assert RELEVANCE_TOP_N == 20


# ---------------------------------------------------------------------------
# _hsic_unbiased
# ---------------------------------------------------------------------------

class TestHSIC:
    def test_identical_matrices_positive(self):
        K = torch.randn(20, 20)
        K = K @ K.T
        val = _hsic_unbiased(K, K)
        assert val.item() > 0

    def test_independent_near_zero(self):
        torch.manual_seed(0)
        K = torch.randn(50, 50)
        L = torch.randn(50, 50)
        val = _hsic_unbiased(K, L)
        assert abs(val.item()) < 0.5

    def test_symmetric_inputs(self):
        K = torch.randn(10, 10)
        K = (K + K.T) / 2
        L = torch.randn(10, 10)
        L = (L + L.T) / 2
        val1 = _hsic_unbiased(K, L)
        val2 = _hsic_unbiased(L, K)
        assert abs(val1.item() - val2.item()) < 1e-5

    def test_scalar_output(self):
        K = torch.randn(10, 10)
        L = torch.randn(10, 10)
        val = _hsic_unbiased(K, L)
        assert val.shape == ()


# ---------------------------------------------------------------------------
# _cosine_topk
# ---------------------------------------------------------------------------

class TestCosineTopk:
    def test_mask_shape(self):
        feats = F.normalize(torch.randn(10, 16), dim=-1)
        K_sim, mask = _cosine_topk(feats, 3, torch.device("cpu"))
        assert K_sim.shape == (10, 10)
        assert mask.shape == (10, 10)

    def test_mask_row_sum(self):
        feats = F.normalize(torch.randn(10, 16), dim=-1)
        _, mask = _cosine_topk(feats, 3, torch.device("cpu"))
        assert (mask.sum(dim=1) == 3).all()

    def test_mask_diagonal_zero(self):
        feats = F.normalize(torch.randn(10, 16), dim=-1)
        _, mask = _cosine_topk(feats, 3, torch.device("cpu"))
        assert (mask.diagonal() == 0).all()

    def test_sim_symmetric(self):
        feats = F.normalize(torch.randn(10, 16), dim=-1)
        K_sim, _ = _cosine_topk(feats, 3, torch.device("cpu"))
        assert torch.allclose(K_sim, K_sim.T, atol=1e-5)

    def test_sim_values_in_range(self):
        feats = F.normalize(torch.randn(10, 16), dim=-1)
        K_sim, _ = _cosine_topk(feats, 3, torch.device("cpu"))
        assert K_sim.min() >= -1.0 - 1e-5
        assert K_sim.max() <= 1.0 + 1e-5


# ---------------------------------------------------------------------------
# _sim_topk_mask
# ---------------------------------------------------------------------------

class TestSimTopkMask:
    def test_mask_shape(self):
        sim = torch.randn(8, 8)
        mask = _sim_topk_mask(sim, 2)
        assert mask.shape == (8, 8)

    def test_row_sum(self):
        sim = torch.randn(8, 8)
        mask = _sim_topk_mask(sim, 2)
        assert (mask.sum(dim=1) == 2).all()

    def test_diagonal_zero(self):
        sim = torch.randn(8, 8)
        mask = _sim_topk_mask(sim, 2)
        assert (mask.diagonal() == 0).all()

    def test_selects_largest(self):
        sim = torch.tensor([
            [0.0, 0.9, 0.1, 0.5],
            [0.9, 0.0, 0.3, 0.7],
            [0.1, 0.3, 0.0, 0.6],
            [0.5, 0.7, 0.6, 0.0],
        ])
        mask = _sim_topk_mask(sim, 2)
        # Row 0: top-2 are indices 1 (0.9) and 3 (0.5)
        assert mask[0, 1] == 1.0
        assert mask[0, 3] == 1.0
        assert mask[0, 2] == 0.0


# ---------------------------------------------------------------------------
# _asymmetric_platonic_cknna
# ---------------------------------------------------------------------------

class TestAsymmetricPlatonicCKNNA:
    def test_identical_spaces(self):
        feats = F.normalize(torch.randn(20, 16), dim=-1)
        K_sim, mask_K = _cosine_topk(feats, 3, torch.device("cpu"))
        cknna = _asymmetric_platonic_cknna(K_sim, mask_K, mask_K)
        assert cknna > 0

    def test_returns_float(self):
        feats = F.normalize(torch.randn(20, 16), dim=-1)
        K_sim, mask_K = _cosine_topk(feats, 3, torch.device("cpu"))
        mask_L = _sim_topk_mask(torch.randn(20, 20), 3)
        cknna = _asymmetric_platonic_cknna(K_sim, mask_K, mask_L)
        assert isinstance(cknna, float)

    def test_zero_when_no_overlap(self):
        n = 10
        K_sim = torch.eye(n)
        mask_K = torch.zeros(n, n)
        mask_K[torch.arange(n), (torch.arange(n) + 1) % n] = 1.0
        mask_L = torch.zeros(n, n)
        mask_L[torch.arange(n), (torch.arange(n) + 5) % n] = 1.0
        cknna = _asymmetric_platonic_cknna(K_sim, mask_K, mask_L)
        assert cknna == 0.0


# ---------------------------------------------------------------------------
# compute_cknna
# ---------------------------------------------------------------------------

class TestComputeCKNNA:
    def test_output_keys(self):
        n, d_z, d_ref = 20, 16, 32
        z = _random_l2_normed(n, d_z)
        v_ref = _random_l2_normed(n, d_ref)
        dtw = _random_symmetric_np(n)
        indices = np.arange(n)
        result = compute_cknna(z, v_ref, dtw, topk=3, device=torch.device("cpu"))
        assert set(result.keys()) == {"cknna_dtw", "mutual_knn_dtw", "cknna_jepa", "mutual_knn_jepa"}

    def test_values_are_finite(self):
        n, d_z, d_ref = 20, 16, 32
        z = _random_l2_normed(n, d_z)
        v_ref = _random_l2_normed(n, d_ref)
        dtw = _random_symmetric_np(n)
        indices = np.arange(n)
        result = compute_cknna(z, v_ref, dtw, topk=3, device=torch.device("cpu"))
        for v in result.values():
            assert np.isfinite(v)

    def test_mutual_knn_between_0_and_1(self):
        n = 20
        z = _random_l2_normed(n, 16)
        v_ref = _random_l2_normed(n, 32)
        dtw = _random_symmetric_np(n)
        result = compute_cknna(z, v_ref, dtw, topk=3, device=torch.device("cpu"))
        assert 0.0 <= result["mutual_knn_dtw"] <= 1.0
        assert 0.0 <= result["mutual_knn_jepa"] <= 1.0

    def test_topk_clamped_to_n_minus_1(self):
        n = 5
        z = _random_l2_normed(n, 8)
        v_ref = _random_l2_normed(n, 8)
        dtw = _random_symmetric_np(n)
        result = compute_cknna(z, v_ref, dtw, topk=100, device=torch.device("cpu"))
        for v in result.values():
            assert np.isfinite(v)

    def test_subset_indices(self):
        # The caller subsets the DTW matrix before compute_cknna.
        n_sub = 10
        z = _random_l2_normed(n_sub, 16)
        v_ref = _random_l2_normed(n_sub, 32)
        dtw_sub = _random_symmetric_np(n_sub)
        result = compute_cknna(z, v_ref, dtw_sub, topk=3, device=torch.device("cpu"))
        for v in result.values():
            assert np.isfinite(v)

    def test_explicit_k_metrics(self):
        n = 20
        z = _random_l2_normed(n, 16)
        v_ref = _random_l2_normed(n, 32)
        dtw = _random_symmetric_np(n)
        result = compute_cknna_at_k(
            z, v_ref, dtw, [1, 5, 20], device=torch.device("cpu"),
        )
        assert set(result.keys()) == {
            "cknna_dtw@1",
            "cknna_dtw@5",
            "cknna_dtw@20",
        }
        for v in result.values():
            assert np.isfinite(v)


class TestGlobalPoolCKNNA:
    @staticmethod
    def _dense(indices, values, n_columns):
        dense = np.zeros((len(indices), n_columns), dtype=np.float64)
        for row in range(len(indices)):
            for col, value in zip(indices[row], values[row]):
                if 0 <= col < n_columns:
                    dense[row, col] += value
        return dense

    def test_sparse_centered_inner_matches_dense_reference(self):
        x_idx = np.array([[0, 3, -1], [1, 4, 2], [5, 0, -1]])
        x_val = np.array([[0.9, 0.2, 0.0], [0.7, 0.4, 0.1], [0.8, 0.3, 0.0]])
        y_idx = np.array([[3, 2, -1], [1, 5, -1], [5, 4, 0]])
        y_val = np.array([[1.0, 1.0, 0.0], [1.0, 1.0, 0.0], [1.0, 1.0, 1.0]])
        n_columns = 6

        x = self._dense(x_idx, x_val, n_columns)
        y = self._dense(y_idx, y_val, n_columns)
        x_centered = x - x.mean(1, keepdims=True) - x.mean(0, keepdims=True) + x.mean()
        y_centered = y - y.mean(1, keepdims=True) - y.mean(0, keepdims=True) + y.mean()
        expected = float((x_centered * y_centered).sum())
        actual = _rectangular_centered_inner(
            x_idx, x_val, y_idx, y_val, n_columns,
        )
        assert actual == pytest.approx(expected)

    def test_identical_binary_neighborhoods_score_one(self):
        neighbors = np.array([[1, 2], [0, 3], [3, 1]], dtype=np.int64)
        result = compute_global_pool_cknna(
            neighbors, np.ones_like(neighbors, dtype=np.float32),
            neighbors, corpus_size=4,
        )
        assert result["cknna_dtw_global_pool"] == pytest.approx(1.0)
        assert result["mutual_knn_dtw_global_pool"] == pytest.approx(1.0)

    def test_disjoint_neighborhoods_have_zero_overlap(self):
        model = np.array([[0, 1], [0, 1], [0, 1]], dtype=np.int64)
        dtw = np.array([[2, 3], [2, 3], [2, 3]], dtype=np.int64)
        result = compute_global_pool_cknna(
            model, np.ones_like(model, dtype=np.float32), dtw, corpus_size=4,
        )
        assert np.isfinite(result["cknna_dtw_global_pool"])
        assert result["mutual_knn_dtw_global_pool"] == 0.0

    def test_global_pool_explicit_k_metrics(self):
        model = np.array([[1, 2, 3], [0, 3, 4], [3, 1, 5]], dtype=np.int64)
        sims = np.ones_like(model, dtype=np.float32)
        dtw = np.array([[1, 2, 4], [0, 4, 5], [3, 5, 1]], dtype=np.int64)
        result = compute_global_pool_cknna_at_k(
            model, sims, dtw, [1, 2, 3], corpus_size=6,
        )
        assert set(result.keys()) == {
            "cknna_dtw_global_pool@1",
            "cknna_dtw_global_pool@2",
            "cknna_dtw_global_pool@3",
        }
        for v in result.values():
            assert np.isfinite(v)

    def test_global_recall_at_k_indicator_against_fixed_relevance(self):
        # relevance_n=3 → DTW positives are the full row.
        # Query 0 model=[1,2,3], DTW={1,2,3} → hits at K=1, K=2, K=3.
        # Query 1 model=[4,5,6], DTW={7,8,6} → first hit at slot 2 (clip 6).
        model = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.int64)
        dtw = np.array([[1, 2, 3], [7, 8, 6]], dtype=np.int64)
        result = compute_global_recall_at_k(
            model, dtw, [1, 2, 3], corpus_size=9, relevance_n=3,
        )
        # K=1: Q0 hit, Q1 miss → 0.5
        assert result["global_R@1"] == 0.5
        # K=2: Q0 hit, Q1 miss → 0.5
        assert result["global_R@2"] == 0.5
        # K=3: Q0 hit, Q1 hit (clip 6 in top-3) → 1.0
        assert result["global_R@3"] == 1.0

    def test_global_recall_at_k_fixed_relevance_independent_of_k(self):
        # Same model + DTW; relevance_n=1 keeps only the top DTW neighbor.
        # Query 0 dtw_top1={1}; model top-K always contains 1 → R@K=1.
        # Query 1 dtw_top1={7}; model never contains 7 → R@K=0.
        model = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.int64)
        dtw = np.array([[1, 2, 3], [7, 8, 6]], dtype=np.int64)
        result = compute_global_recall_at_k(
            model, dtw, [1, 2, 3], corpus_size=9, relevance_n=1,
        )
        assert result["global_R@1"] == 0.5
        assert result["global_R@2"] == 0.5
        assert result["global_R@3"] == 0.5

    def test_global_recall_at_k_ignores_padding(self):
        model = np.array([[1, -1], [-1, -1]], dtype=np.int64)
        dtw = np.array([[1, 2], [3, 4]], dtype=np.int64)
        result = compute_global_recall_at_k(
            model, dtw, [1, 2], corpus_size=5, relevance_n=2,
        )
        assert result["global_R@1"] == 0.5
        assert result["global_R@2"] == 0.5

    def test_global_recall_requires_matching_rows(self):
        model = np.array([[1, 2]], dtype=np.int64)
        dtw = np.array([[1, 2], [3, 4]], dtype=np.int64)
        with pytest.raises(ValueError, match="same row count"):
            compute_global_recall_at_k(model, dtw, [1], corpus_size=5)

    @requires_faiss
    def test_global_search_uses_raw_clip_ids_and_removes_self(self):
        corpus_z = np.eye(4, dtype=np.float32)
        corpus_ids = np.array([40, 10, 30, 20])
        query_z = corpus_z[[1, 3]]
        query_ids = np.array([10, 20])
        neighbor_ids, _ = _global_model_neighbors(
            query_z, query_ids, corpus_z, corpus_ids,
            topk=2, index_type="IndexFlatIP",
        )
        assert 10 not in neighbor_ids[0]
        assert 20 not in neighbor_ids[1]
        assert set(neighbor_ids.ravel()).issubset(set(corpus_ids))


# ---------------------------------------------------------------------------
# Train-pool / pool-helper metrics
# ---------------------------------------------------------------------------

class TestPoolMetricsFromEncoded:
    """Coverage for `_pool_metrics_from_encoded` (the helper shared by the
    global-pool and train-pool eval blocks)."""

    @staticmethod
    def _deterministic_corpus(n, d, seed=0):
        """Construct n clips in d dims with distinct, well-separated cosine
        similarities so FAISS top-K is unambiguous (no ties)."""
        rng = np.random.default_rng(seed)
        z = rng.standard_normal((n, d)).astype(np.float32)
        z /= np.linalg.norm(z, axis=1, keepdims=True)
        return z

    @requires_faiss
    def test_perfect_recall_when_model_matches_dtw(self):
        # Build a corpus where each query's true FAISS top-K is unambiguous.
        # We then construct dtw_rows to MATCH those FAISS neighbors → all
        # metrics should be 1.0. This decouples the test from FAISS tie-
        # breaking on degenerate geometries.
        n, d = 6, 16
        corpus_z = self._deterministic_corpus(n, d, seed=0)
        corpus_clip_idx = np.arange(n, dtype=np.int64)
        query_rows = np.array([0, 1, 2], dtype=np.int64)
        query_z = corpus_z[query_rows]
        query_clip_idx = corpus_clip_idx[query_rows]
        # Mirror the model's own retrieval as the "DTW ground truth".
        faiss_neighbors, _ = _global_model_neighbors(
            query_z, query_clip_idx, corpus_z, corpus_clip_idx,
            topk=2, index_type="IndexFlatIP",
        )
        metrics = _pool_metrics_from_encoded(
            query_z=query_z,
            query_clip_idx=query_clip_idx,
            corpus_z=corpus_z,
            corpus_clip_idx=corpus_clip_idx,
            dtw_neighbor_idx_rows=faiss_neighbors,
            k_values=[1, 2],
            cknna_k_values=[1, 2],
            cknna_topk=2,
            index_type="IndexFlatIP",
            prefix="train_pool",
            recall_prefix="train_pool_",
        )
        assert metrics["train_pool_R@1"] == pytest.approx(1.0)
        assert metrics["train_pool_R@2"] == pytest.approx(1.0)
        # mutual-kNN is pure set overlap → 1.0 when neighbor sets match.
        # cknna_topk=2 in this test → the @2 key suffix.
        assert metrics["mutual_knn_dtw_train_pool@2"] == pytest.approx(1.0)
        # CKNNA is not necessarily 1.0 here: model values are real cosine
        # sims while the DTW side is the binary indicator from
        # compute_global_pool_cknna. Identical sets but different value
        # distributions → CKNNA < 1 yet strictly positive.
        assert metrics["cknna_dtw_train_pool@2"] > 0.0
        assert "cknna_dtw_train_pool@1" in metrics

    @requires_faiss
    def test_index_space_size_larger_than_corpus(self):
        # Corpus uses non-contiguous clip indices (a subset of a larger
        # global index space) — the train-pool calling convention. Helper
        # must still produce finite metrics with correct R@K against
        # original-space indices.
        global_n = 8
        train_clip_ids = np.array([1, 3, 5, 7], dtype=np.int64)
        corpus_z = self._deterministic_corpus(4, 8, seed=1)
        query_z = corpus_z[[0, 1]]
        query_clip_idx = train_clip_ids[[0, 1]]
        # Use the model's own neighbors as the DTW ground truth → R@K = 1.
        faiss_neighbors, _ = _global_model_neighbors(
            query_z, query_clip_idx, corpus_z, train_clip_ids,
            topk=2, index_type="IndexFlatIP",
        )
        metrics = _pool_metrics_from_encoded(
            query_z=query_z,
            query_clip_idx=query_clip_idx,
            corpus_z=corpus_z,
            corpus_clip_idx=train_clip_ids,
            dtw_neighbor_idx_rows=faiss_neighbors,
            k_values=[1, 2],
            cknna_k_values=[1, 2],
            cknna_topk=2,
            index_type="IndexFlatIP",
            prefix="train_pool",
            recall_prefix="train_pool_",
            index_space_size=global_n,
        )
        assert metrics["train_pool_R@1"] == pytest.approx(1.0)
        assert np.isfinite(metrics["cknna_dtw_train_pool@2"])
        # All returned DTW neighbor ids should land in the train clip-id set.
        for nid in faiss_neighbors.ravel():
            assert int(nid) in set(train_clip_ids.tolist())


class TestBuildTrainPoolNeighbors:
    """Verify the local→global remap in `build_train_pool_neighbors`.

    The point of the wrapper is that non-train rows are left as `-1` while
    train rows hold neighbors in original clip-idx space, even though the
    underlying builder operates on local row indices.
    """

    def test_remap_roundtrip(self, tmp_path):
        pytest.importorskip("torch")
        pytest.importorskip("numba", reason="DTW kernels require numba")
        if not torch.cuda.is_available():
            pytest.skip("DTW builder requires CUDA")
        sys.path.insert(
            0, str(Path(__file__).resolve().parent.parent / "build_dataset"),
        )
        from build_dtw_neighbors import build_train_pool_neighbors

        # Synthetic full payload: 8 clips at T=42 with D=63 (matches the
        # `abs_21j_coords` design; safe even if other designs differ).
        rng = np.random.default_rng(0)
        n_total, T, D = 8, 42, 63
        trajs = torch.from_numpy(
            rng.standard_normal((n_total, T, D)).astype(np.float32)
        )
        lengths = torch.full((n_total,), T, dtype=torch.int32)
        # Use realistic (video_number, node_uid) tuples — the wrapper must
        # NOT treat clip_keys as integer clip indices.
        clip_keys = [(100 + i, f"node-{i}") for i in range(n_total)]
        traj_path = tmp_path / "trajs.pt"
        torch.save({
            "trajectories": trajs,
            "lengths": lengths,
            "clip_keys": clip_keys,
            "design": "abs_21j_coords",
        }, traj_path)

        train_positions = [1, 3, 5, 7]
        output_path = tmp_path / "train_pool_neigh.pt"
        build_train_pool_neighbors(
            trajectories_path=traj_path,
            train_positions=train_positions,
            dtw_design="abs_21j_coords",
            output_path=output_path,
            top_k=2,
            candidate_k=3,
            pooling="stats4",
        )

        payload = torch.load(output_path, weights_only=False, map_location="cpu")
        neigh_idx = payload["neighbor_clip_idx"].numpy()
        assert neigh_idx.shape == (n_total, 2)
        # Non-train rows untouched.
        for non_train in [0, 2, 4, 6]:
            assert np.all(neigh_idx[non_train] == -1)
        # Train rows reference only original-space train clip indices.
        train_set = set(train_positions)
        for train_pos in train_positions:
            for col in neigh_idx[train_pos]:
                col_int = int(col)
                assert col_int in train_set
                assert col_int != train_pos   # self excluded by builder
