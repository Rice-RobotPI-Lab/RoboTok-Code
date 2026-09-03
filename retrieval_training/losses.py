from typing import Dict

import torch
import torch.nn as nn
import torch.nn.functional as F


def _upper_triangle(mat: torch.Tensor) -> torch.Tensor:
    """Return flattened strict upper-triangle values of a square matrix."""
    idx = torch.triu_indices(mat.shape[0], mat.shape[1], offset=1, device=mat.device)
    return mat[idx[0], idx[1]]


_DEGENERATE_STD_WARN_THRESHOLD = 1e-3
# Mask value used in place of -inf for diagonal entries before softmax. Large
# enough that softmax(MASK_VALUE) ≈ 0 against logits of order ±10, but
# representable in fp16. Shared between SimilarityAlignmentLoss and
# target_entropy_nats so the two stay in sync.
_DIAG_MASK_VALUE = -1e4


def _zscore_upper_triangle(
    mat: torch.Tensor, eps: float = 1e-8,
) -> tuple[torch.Tensor, float]:
    """Z-score `mat` using stats from its strict upper-triangle.

    Returns (zscored_mat, std) — caller decides whether to warn on degenerate std.
    """
    vals = _upper_triangle(mat)
    std = vals.std()
    return (mat - vals.mean()) / (std + eps), float(std.item())


def _normalize_matrix(mat: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """Zero-mean, unit-variance normalisation over strict upper-triangle entries.

    Emits a one-line warning when the upper-triangle std is below
    `_DEGENERATE_STD_WARN_THRESHOLD` — in that regime the soft-contrastive
    target collapses to ~uniform and the loss stops carrying useful gradient.
    """
    out, std = _zscore_upper_triangle(mat, eps)
    if std < _DEGENERATE_STD_WARN_THRESHOLD:
        print(f"[losses] WARNING: target std={std:.2e} is degenerate "
              f"(< {_DEGENERATE_STD_WARN_THRESHOLD:.0e}); soft-contrastive target "
              "is near-uniform, gradient signal will be weak.")
    return out


@torch.no_grad()
def target_entropy_nats(target_sims: torch.Tensor, tau_target: float) -> float:
    """Mean per-row entropy of the soft target distribution (in nats).

    Uses the exact same z-score → temperature → diagonal-mask → softmax
    pipeline as `SimilarityAlignmentLoss`, so the value is the cross-entropy
    floor of that loss on this target. `loss − target_entropy_nats(...)`
    gives the per-batch KL gap (true model error) on the same target.

    Skips the degenerate-std warning that `_normalize_matrix` prints — the
    loss path is the canonical place to emit that, and we don't want to
    double-print when this helper is called from logging code.
    """
    tgt_n, _ = _zscore_upper_triangle(target_sims)
    B = tgt_n.shape[0]
    diag_mask = torch.eye(B, dtype=torch.bool, device=tgt_n.device)
    tgt_logits = (tgt_n / tau_target).masked_fill(diag_mask, _DIAG_MASK_VALUE)
    soft_tgt = F.softmax(tgt_logits, dim=-1)
    H = -(soft_tgt * (soft_tgt + 1e-12).log()).sum(dim=-1).mean()
    return float(H.item())


class SimilarityAlignmentLoss(nn.Module):
    """Align predicted z@z.T cosine similarities with a target similarity matrix.

    Currently supports `loss_type="soft_contrastive"`: cross-entropy between the
    row-softmax of (pred/tau_pred) and the row-softmax of (target_n/tau_target),
    where target_n is the per-batch z-scored target. The diagonal is masked out.

    `loss_type` is kept as a switch so other families can be added later, but
    no other values are valid yet.
    """

    def __init__(
        self,
        loss_type: str = "soft_contrastive",
        tau_pred: float = 0.1,
        tau_target: float = 0.5,
    ):
        super().__init__()
        if loss_type != "soft_contrastive":
            raise ValueError(
                f"Unknown loss_type={loss_type!r}; only 'soft_contrastive' is supported"
            )
        self.loss_type = loss_type
        self.tau_pred = tau_pred
        self.tau_target = tau_target

    def forward(
        self,
        z: torch.Tensor,
        target_sims: torch.Tensor,
    ) -> torch.Tensor:
        pred = z @ z.T  # cosine sim (z is L2-normed)
        # Per-batch z-score: zero-mean, unit-std over strict upper-triangle.
        # softmax is shift-invariant so the mean shift cancels; the std scaling
        # makes tau_target's effect comparable across batches with different
        # target spread.
        tgt_n = _normalize_matrix(target_sims)

        B = pred.shape[0]
        diag_mask = torch.eye(B, dtype=torch.bool, device=pred.device)
        # Use a large negative (instead of -inf) so softmax at the diagonal is
        # ~0 and log_softmax is finite — this avoids 0 * -inf = NaN in the
        # row-wise cross-entropy. -1e4 is representable in fp16; -inf is not
        # safe to multiply by 0 even when softmax returns exactly 0 elsewhere.
        pred_logits = (pred / self.tau_pred).masked_fill(diag_mask, _DIAG_MASK_VALUE)
        tgt_logits = (tgt_n / self.tau_target).masked_fill(diag_mask, _DIAG_MASK_VALUE)
        log_pred = F.log_softmax(pred_logits, dim=-1)
        soft_tgt = F.softmax(tgt_logits, dim=-1)
        # Explicitly zero the diagonal contribution to make the gradient wrt
        # diagonal logits exactly zero, regardless of any residual fp16 noise.
        prod = (soft_tgt * log_pred).masked_fill(diag_mask, 0.0)
        return -prod.sum(dim=-1).mean()


class CombinedLoss(nn.Module):
    """Symmetric loss form for trajectory (DTW) + semantic-preservation (JEPA).

    Both sub-losses use the same `loss_type` and temperatures so the two terms
    are directly comparable; only the target similarity matrix differs.

    `components` selects which terms contribute to `total` (i.e., to backward):
        "both"       — total = trajectory + lambda_preserve * semantic
        "trajectory" — total = trajectory                (semantic computed for logging only)
        "semantic"   — total = lambda_preserve * semantic (trajectory computed for logging only)

    The non-contributing term is still computed (under torch.no_grad) and
    returned in the metrics dict so TensorBoard can show it during ablations.
    """

    def __init__(
        self,
        lambda_preserve: float = 0.1,
        loss_type: str = "soft_contrastive",
        tau_pred: float = 0.1,
        tau_target: float = 0.5,
        components: str = "both",
    ):
        super().__init__()
        if components not in ("both", "trajectory", "semantic"):
            raise ValueError(
                f"Unknown components={components!r}; expected 'both', 'trajectory', or 'semantic'"
            )
        self.lambda_preserve = lambda_preserve
        self.components = components
        self.trajectory_loss = SimilarityAlignmentLoss(loss_type, tau_pred, tau_target)
        self.semantic_loss = SimilarityAlignmentLoss(loss_type, tau_pred, tau_target)

    def _maybe_no_grad(self, contributes: bool):
        """Use no_grad if this component does not contribute to total — saves
        building the autograd graph for a value used only for logging."""
        return torch.enable_grad() if contributes else torch.no_grad()

    def forward(
        self,
        z: torch.Tensor,
        v_ref: torch.Tensor,
        dtw_similarities: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        traj_contributes = self.components in ("both", "trajectory")
        sem_contributes = self.components in ("both", "semantic")

        with self._maybe_no_grad(traj_contributes):
            l_traj = self.trajectory_loss(z, dtw_similarities)

        sim_ref = v_ref @ v_ref.T
        with self._maybe_no_grad(sem_contributes):
            l_sem = self.semantic_loss(z, sim_ref)

        if self.components == "both":
            total = l_traj + self.lambda_preserve * l_sem
        elif self.components == "trajectory":
            total = l_traj
        else:  # "semantic"
            total = self.lambda_preserve * l_sem

        return {"total": total, "trajectory": l_traj, "semantic": l_sem}


class TopKSetRankLoss(nn.Module):
    """Unordered DTW top-k membership loss plus cached-DTW ordering loss.

    Batches must consist of fixed groups laid out as
    ``[anchor, positive..., boundary...]``.  The set term places each explicit
    top-k positive above the explicit boundary and the hardest other valid
    in-batch negatives.  Known top-k neighbors are never used as negatives.
    The rank term orders every pair among the explicit candidates using the
    cached DTW similarities, without regressing their magnitudes.
    """

    def __init__(
        self,
        neighbor_clip_idx: torch.Tensor,
        neighbor_dtw_sim: torch.Tensor,
        *,
        positive_top_k: int = 10,
        positives_per_anchor: int = 2,
        boundary_negatives_per_anchor: int = 1,
        hard_negatives: int = 16,
        set_weight: float = 1.0,
        rank_weight: float = 0.2,
        margin: float = 0.05,
        set_temperature: float = 0.07,
        rank_temperature: float = 0.10,
    ) -> None:
        super().__init__()
        if positives_per_anchor < 1 or boundary_negatives_per_anchor < 1:
            raise ValueError("set/rank loss requires positive and boundary candidates")
        if hard_negatives < boundary_negatives_per_anchor:
            raise ValueError("hard_negatives must include every explicit boundary")
        if positive_top_k < positives_per_anchor:
            raise ValueError("positive_top_k is smaller than positives_per_anchor")
        if set_temperature <= 0 or rank_temperature <= 0:
            raise ValueError("loss temperatures must be positive")
        if neighbor_clip_idx.shape != neighbor_dtw_sim.shape:
            raise ValueError("neighbor ID and DTW-similarity tables must have equal shape")
        if neighbor_clip_idx.ndim != 2 or neighbor_clip_idx.shape[1] < positive_top_k:
            raise ValueError("neighbor cache does not contain the requested positive top-k")

        # Cache data is run-specific supervision, not model state.  Keeping it
        # non-persistent prevents ~16 MB of duplicate data in every checkpoint.
        self.register_buffer(
            "neighbor_clip_idx", neighbor_clip_idx.to(torch.long), persistent=False,
        )
        self.register_buffer(
            "neighbor_dtw_sim", neighbor_dtw_sim.to(torch.float32), persistent=False,
        )
        self.positive_top_k = int(positive_top_k)
        self.positives_per_anchor = int(positives_per_anchor)
        self.boundary_negatives_per_anchor = int(boundary_negatives_per_anchor)
        self.hard_negatives = int(hard_negatives)
        self.set_weight = float(set_weight)
        self.rank_weight = float(rank_weight)
        self.margin = float(margin)
        self.set_temperature = float(set_temperature)
        self.rank_temperature = float(rank_temperature)
        self.group_size = 1 + self.positives_per_anchor + self.boundary_negatives_per_anchor

    def forward(
        self, z: torch.Tensor, clip_indices: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        B = z.shape[0]
        if B == 0 or B % self.group_size != 0:
            raise ValueError(
                f"batch size {B} must be a nonzero multiple of group_size={self.group_size}"
            )
        clip_indices = clip_indices.to(device=z.device, dtype=torch.long)
        if clip_indices.shape != (B,):
            raise ValueError(f"clip_indices must have shape ({B},)")

        anchor_pos = torch.arange(0, B, self.group_size, device=z.device)
        A = anchor_pos.numel()
        offsets = torch.arange(1, self.group_size, device=z.device)
        explicit_pos = anchor_pos[:, None] + offsets[None, :]
        positive_pos = explicit_pos[:, :self.positives_per_anchor]
        boundary_pos = explicit_pos[:, self.positives_per_anchor:]

        anchor_clip = clip_indices[anchor_pos]
        cached_ids = self.neighbor_clip_idx[anchor_clip]
        cached_dtw = self.neighbor_dtw_sim[anchor_clip]
        top_ids = cached_ids[:, :self.positive_top_k]

        anchor_sims = z[anchor_pos] @ z.T
        # Negatives are every in-batch clip except self and known DTW top-k.
        is_known_positive = (
            clip_indices[None, :, None] == top_ids[:, None, :]
        ).any(dim=-1)
        valid_negative = ~is_known_positive
        valid_negative.scatter_(1, anchor_pos[:, None], False)

        # The sampler guarantees explicit boundaries. Include them regardless
        # of current model score, then fill the budget with hardest remaining
        # in-batch negatives. Selection uses detached scores; gathered scores
        # retain their gradient.
        selectable = valid_negative.clone()
        selectable.scatter_(1, boundary_pos, False)
        n_extra = min(
            self.hard_negatives - self.boundary_negatives_per_anchor,
            max(B - 1 - self.boundary_negatives_per_anchor, 0),
        )
        if n_extra > 0:
            selection_scores = anchor_sims.detach().masked_fill(~selectable, float("-inf"))
            extra_scores, extra_pos = torch.topk(selection_scores, n_extra, dim=1)
            extra_valid = torch.isfinite(extra_scores)
        else:
            extra_pos = torch.empty((A, 0), dtype=torch.long, device=z.device)
            extra_valid = torch.empty((A, 0), dtype=torch.bool, device=z.device)

        negative_pos = torch.cat([boundary_pos, extra_pos], dim=1)
        negative_valid = torch.cat([
            torch.ones_like(boundary_pos, dtype=torch.bool), extra_valid,
        ], dim=1)
        negative_sims = anchor_sims.gather(1, negative_pos)
        positive_sims = anchor_sims.gather(1, positive_pos)

        set_terms = F.softplus(
            (negative_sims[:, None, :] - positive_sims[:, :, None] + self.margin)
            / self.set_temperature
        )
        set_valid = negative_valid[:, None, :].expand_as(set_terms)
        set_loss = set_terms.masked_select(set_valid).mean()

        # Recover cached DTW values for the explicit candidates, then order all
        # candidate pairs. The sampler makes every candidate a cached top-20
        # entry, so absence indicates a broken batch contract.
        explicit_clip = clip_indices[explicit_pos]
        matches = explicit_clip[:, :, None] == cached_ids[:, None, :]
        found = matches.any(dim=-1)
        if not bool(found.all()):
            raise ValueError("explicit positive/boundary candidate missing from DTW cache")
        explicit_dtw = torch.where(
            matches, cached_dtw[:, None, :], float("-inf"),
        ).max(dim=-1).values
        explicit_sims = anchor_sims.gather(1, explicit_pos)

        pair_idx = torch.triu_indices(
            explicit_pos.shape[1], explicit_pos.shape[1], offset=1, device=z.device,
        )
        left, right = pair_idx[0], pair_idx[1]
        target_delta = explicit_dtw[:, left] - explicit_dtw[:, right]
        pred_delta = explicit_sims[:, left] - explicit_sims[:, right]
        non_tie = target_delta != 0
        desired_delta = target_delta.sign() * pred_delta
        rank_terms = F.softplus(-desired_delta / self.rank_temperature)
        selected_rank_terms = rank_terms.masked_select(non_tie)
        rank_loss = (
            selected_rank_terms.mean()
            if selected_rank_terms.numel() > 0
            else explicit_sims.sum() * 0.0
        )

        total = self.set_weight * set_loss + self.rank_weight * rank_loss
        with torch.no_grad():
            set_violation = (
                negative_sims[:, None, :] >= positive_sims[:, :, None]
            ).masked_select(set_valid).float().mean()
            selected_rank_violations = (
                (desired_delta <= 0).masked_select(non_tie).float()
            )
            rank_violation = (
                selected_rank_violations.mean()
                if selected_rank_violations.numel() > 0
                else torch.zeros((), device=z.device)
            )
            hardest_negative = negative_sims.masked_fill(
                ~negative_valid, float("-inf"),
            ).max(dim=1).values
            positive_margin = (
                positive_sims - hardest_negative[:, None]
            ).mean()

        return {
            "total": total,
            "set": set_loss,
            "rank": rank_loss,
            "set_violation_rate": set_violation,
            "rank_violation_rate": rank_violation,
            "positive_hard_negative_margin": positive_margin,
        }
