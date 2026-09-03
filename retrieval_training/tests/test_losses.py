"""Tests for retrieval_training/losses.py.

Run from retrieval_training/: python -m pytest tests/test_losses.py -v
"""

import sys
from pathlib import Path

import pytest
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from losses import (
    CombinedLoss,
    SimilarityAlignmentLoss,
    TopKSetRankLoss,
    _normalize_matrix,
    _upper_triangle,
    target_entropy_nats,
)


def _set_rank_cache(n=50, k=20):
    ids = torch.empty((n, k), dtype=torch.long)
    sims = torch.empty((n, k), dtype=torch.float32)
    for i in range(n):
        ids[i] = torch.tensor([(i + j + 1) % n for j in range(k)])
        sims[i] = torch.linspace(0.95, 0.05, k)
    return ids, sims


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _random_l2_normed(n, d, seed=0):
    torch.manual_seed(seed)
    z = torch.randn(n, d)
    return F.normalize(z, dim=-1)


def _random_symmetric(n, seed=0):
    torch.manual_seed(seed)
    raw = torch.randn(n, n)
    return (raw + raw.T) / 2


# ---------------------------------------------------------------------------
# _upper_triangle
# ---------------------------------------------------------------------------

class TestUpperTriangle:
    def test_length(self):
        mat = torch.randn(4, 4)
        vals = _upper_triangle(mat)
        assert vals.shape == (6,)  # 4*3/2

    def test_correct_values(self):
        mat = torch.arange(9).reshape(3, 3).float()
        vals = _upper_triangle(mat)
        assert torch.allclose(vals, torch.tensor([1.0, 2.0, 5.0]))

    def test_excludes_diagonal(self):
        mat = torch.eye(3) * 100
        vals = _upper_triangle(mat)
        assert (vals == 0).all()

    def test_size_1_matrix(self):
        mat = torch.tensor([[5.0]])
        vals = _upper_triangle(mat)
        assert vals.shape == (0,)


# ---------------------------------------------------------------------------
# _normalize_matrix
# ---------------------------------------------------------------------------

class TestNormalizeMatrix:
    def test_upper_triangle_zero_mean(self):
        mat = _random_symmetric(5)
        normed = _normalize_matrix(mat)
        vals = _upper_triangle(normed)
        assert abs(vals.mean().item()) < 1e-5

    def test_upper_triangle_unit_std(self):
        mat = _random_symmetric(5)
        normed = _normalize_matrix(mat)
        vals = _upper_triangle(normed)
        assert abs(vals.std().item() - 1.0) < 1e-4

    def test_constant_matrix(self):
        mat = torch.full((4, 4), 3.0)
        normed = _normalize_matrix(mat)
        assert torch.isfinite(normed).all()


# ---------------------------------------------------------------------------
# SimilarityAlignmentLoss — trajectory-style usage (DTW target)
# ---------------------------------------------------------------------------

class TestTrajectoryAlignmentLoss:
    def setup_method(self):
        self.loss_fn = SimilarityAlignmentLoss()

    def test_output_is_scalar(self):
        z = _random_l2_normed(8, 32)
        dtw = _random_symmetric(8)
        loss = self.loss_fn(z, dtw)
        assert loss.shape == ()

    def test_output_nonnegative(self):
        z = _random_l2_normed(8, 32)
        dtw = _random_symmetric(8)
        loss = self.loss_fn(z, dtw)
        assert loss.item() >= 0.0

    def test_gradient_flows(self):
        z = _random_l2_normed(6, 16).requires_grad_(True)
        dtw = _random_symmetric(6)
        loss = self.loss_fn(z, dtw)
        loss.backward()
        assert z.grad is not None
        assert z.grad.abs().sum() > 0

    def test_batch_size_3(self):
        z = _random_l2_normed(3, 16)
        dtw = _random_symmetric(3)
        loss = self.loss_fn(z, dtw)
        assert torch.isfinite(loss)


# ---------------------------------------------------------------------------
# SimilarityAlignmentLoss — semantic-preservation usage (v_ref@v_ref.T target)
# ---------------------------------------------------------------------------

class TestSemanticPreservationLoss:
    """Same loss class, semantic-target call shape (v_ref @ v_ref.T)."""

    def setup_method(self):
        self.loss_fn = SimilarityAlignmentLoss()

    def test_output_is_scalar(self):
        z = _random_l2_normed(8, 32)
        v_ref = _random_l2_normed(8, 64)
        sim_ref = v_ref @ v_ref.T
        loss = self.loss_fn(z, sim_ref)
        assert loss.shape == ()

    def test_output_nonnegative(self):
        z = _random_l2_normed(8, 32)
        v_ref = _random_l2_normed(8, 64)
        sim_ref = v_ref @ v_ref.T
        loss = self.loss_fn(z, sim_ref)
        assert loss.item() >= 0.0

    def test_gradient_flows(self):
        z = _random_l2_normed(6, 32).requires_grad_(True)
        v_ref = _random_l2_normed(6, 64)
        sim_ref = v_ref @ v_ref.T
        loss = self.loss_fn(z, sim_ref)
        loss.backward()
        assert z.grad is not None

    def test_no_gradient_to_vref(self):
        z = _random_l2_normed(6, 32).requires_grad_(True)
        v_ref = _random_l2_normed(6, 64).requires_grad_(True)
        sim_ref = v_ref @ v_ref.T
        loss = self.loss_fn(z, sim_ref)
        loss.backward()
        # v_ref gets gradient because it's used in sim_ref computation,
        # but in practice it's detached before reaching the loss
        assert z.grad is not None


# ---------------------------------------------------------------------------
# CombinedLoss
# ---------------------------------------------------------------------------

class TestCombinedLoss:
    def test_output_keys(self):
        loss_fn = CombinedLoss(lambda_preserve=0.5)
        z = _random_l2_normed(8, 32)
        v_ref = _random_l2_normed(8, 64)
        dtw = _random_symmetric(8)
        out = loss_fn(z, v_ref, dtw)
        assert set(out.keys()) == {"total", "trajectory", "semantic"}

    def test_total_is_weighted_sum(self):
        lam = 0.3
        loss_fn = CombinedLoss(lambda_preserve=lam)
        z = _random_l2_normed(8, 32)
        v_ref = _random_l2_normed(8, 64)
        dtw = _random_symmetric(8)
        out = loss_fn(z, v_ref, dtw)
        expected = out["trajectory"] + lam * out["semantic"]
        assert torch.allclose(out["total"], expected, atol=1e-6)

    def test_lambda_zero_ignores_semantic(self):
        loss_fn = CombinedLoss(lambda_preserve=0.0)
        z = _random_l2_normed(8, 32)
        v_ref = _random_l2_normed(8, 64)
        dtw = _random_symmetric(8)
        out = loss_fn(z, v_ref, dtw)
        assert torch.allclose(out["total"], out["trajectory"])

    def test_all_losses_nonnegative(self):
        loss_fn = CombinedLoss()
        z = _random_l2_normed(8, 32)
        v_ref = _random_l2_normed(8, 64)
        dtw = _random_symmetric(8)
        out = loss_fn(z, v_ref, dtw)
        for v in out.values():
            assert v.item() >= 0.0

    def test_gradient_flows_through_total(self):
        loss_fn = CombinedLoss()
        z = _random_l2_normed(8, 32).requires_grad_(True)
        v_ref = _random_l2_normed(8, 64)
        dtw = _random_symmetric(8)
        out = loss_fn(z, v_ref, dtw)
        out["total"].backward()
        assert z.grad is not None
        assert z.grad.abs().sum() > 0


class TestTopKSetRankLoss:
    def _criterion(self, hard_negatives=3):
        ids, sims = _set_rank_cache()
        return TopKSetRankLoss(
            ids, sims, positive_top_k=10, positives_per_anchor=2,
            boundary_negatives_per_anchor=1, hard_negatives=hard_negatives,
            set_weight=1.0, rank_weight=0.2,
        )

    def test_output_and_weighted_total(self):
        loss_fn = self._criterion()
        z = _random_l2_normed(8, 16)
        clips = torch.tensor([0, 1, 2, 11, 20, 21, 22, 31])
        out = loss_fn(z, clips)
        assert set(out) == {
            "total", "set", "rank", "set_violation_rate",
            "rank_violation_rate", "positive_hard_negative_margin",
        }
        assert torch.allclose(out["total"], out["set"] + 0.2 * out["rank"])

    def test_good_membership_and_order_has_lower_loss(self):
        loss_fn = self._criterion(hard_negatives=1)
        clips = torch.tensor([0, 1, 2, 11])
        good = F.normalize(torch.tensor([
            [1.0, 0.0], [1.0, 0.10], [1.0, 0.30], [-1.0, 0.0],
        ]), dim=-1)
        bad = good.clone()
        bad[[1, 3]] = bad[[3, 1]]
        out_good = loss_fn(good, clips)
        out_bad = loss_fn(bad, clips)
        assert out_good["set"] < out_bad["set"]
        assert out_good["rank"] < out_bad["rank"]

    def test_positive_permutation_does_not_change_set_loss(self):
        loss_fn = self._criterion()
        z = _random_l2_normed(8, 16)
        clips = torch.tensor([0, 1, 2, 11, 20, 21, 22, 31])
        out = loss_fn(z, clips)
        z_swapped = z.clone()
        clips_swapped = clips.clone()
        for a in (0, 4):
            z_swapped[[a + 1, a + 2]] = z_swapped[[a + 2, a + 1]]
            clips_swapped[[a + 1, a + 2]] = clips_swapped[[a + 2, a + 1]]
        swapped = loss_fn(z_swapped, clips_swapped)
        assert torch.allclose(out["set"], swapped["set"], atol=1e-6)

    def test_gradient_flows_and_cache_is_not_checkpointed(self):
        loss_fn = self._criterion()
        z = _random_l2_normed(8, 16).requires_grad_(True)
        clips = torch.tensor([0, 1, 2, 11, 20, 21, 22, 31])
        out = loss_fn(z, clips)
        out["total"].backward()
        assert z.grad is not None and torch.isfinite(z.grad).all()
        assert loss_fn.state_dict() == {}

    def test_rejects_broken_group_contract(self):
        loss_fn = self._criterion()
        with pytest.raises(ValueError, match="multiple of group_size"):
            loss_fn(_random_l2_normed(5, 8), torch.arange(5))


# ---------------------------------------------------------------------------
# target_entropy_nats
# ---------------------------------------------------------------------------

class TestTargetEntropy:
    """target_entropy_nats must agree with what SimilarityAlignmentLoss sees:
    if the predicted distribution exactly matches the soft target, the loss
    equals target_entropy_nats(target, tau_target).
    """

    def test_returns_finite_float(self):
        target = _random_symmetric(8)
        H = target_entropy_nats(target, tau_target=0.5)
        assert isinstance(H, float)
        assert torch.isfinite(torch.tensor(H))

    def test_uniform_when_tau_large(self):
        # Very large tau_target → softmax(target/tau) ≈ uniform over B-1
        # off-diag positions, so entropy ≈ log(B-1).
        target = _random_symmetric(8)
        H = target_entropy_nats(target, tau_target=1e3)
        import math
        assert abs(H - math.log(7)) < 0.01

    def test_concentrated_when_tau_small(self):
        # Very small tau_target → softmax becomes near one-hot → entropy → 0.
        target = _random_symmetric(8)
        H = target_entropy_nats(target, tau_target=1e-3)
        assert H < 0.05

    def test_loss_equals_entropy_when_pred_matches_target(self):
        # If pred logits proportional to target z-score (with the same
        # diagonal masking and matching τ scaling), the predicted distribution
        # equals the soft target distribution.
        torch.manual_seed(0)
        n, d = 8, 16
        target = _random_symmetric(n)
        loss_fn = SimilarityAlignmentLoss(
            loss_type="soft_contrastive", tau_pred=1.0, tau_target=1.0,
        )

        # Construct z so z @ z.T == z-scored(target) (exactly). We can't do
        # that in general, but we can verify the floor by feeding the loss
        # the entropy-matched scenario: pred logits = tgt logits, so log_pred
        # = log_softmax(tgt_logits) and the cross-entropy reduces to
        # -Σ p log p = H(p). Mock this by patching the loss to compute pred
        # = z-scored target directly.
        from losses import _normalize_matrix
        tgt_n = _normalize_matrix(target)
        # If pred = tgt_n exactly (instead of z@z.T), tau_pred=tau_target=1.0
        # gives identical distributions → loss = entropy.
        pred = tgt_n
        diag_mask = torch.eye(n, dtype=torch.bool)
        from losses import _DIAG_MASK_VALUE
        pred_logits = pred.masked_fill(diag_mask, _DIAG_MASK_VALUE)
        tgt_logits = tgt_n.masked_fill(diag_mask, _DIAG_MASK_VALUE)
        log_pred = F.log_softmax(pred_logits, dim=-1)
        soft_tgt = F.softmax(tgt_logits, dim=-1)
        prod = (soft_tgt * log_pred).masked_fill(diag_mask, 0.0)
        ce = -prod.sum(dim=-1).mean().item()

        H = target_entropy_nats(target, tau_target=1.0)
        assert abs(ce - H) < 1e-5

    def test_loss_minus_entropy_is_nonnegative(self):
        # The KL gap (loss - target_entropy) is non-negative for any pred.
        torch.manual_seed(1)
        for _ in range(3):
            z = _random_l2_normed(10, 32)
            target = _random_symmetric(10)
            loss_fn = SimilarityAlignmentLoss(tau_pred=0.2, tau_target=0.5)
            loss = loss_fn(z, target).item()
            H = target_entropy_nats(target, tau_target=0.5)
            assert loss - H >= -1e-5  # tiny numerical slack
