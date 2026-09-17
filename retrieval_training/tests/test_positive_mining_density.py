"""End-to-end behavioral test for hard-positive mining.

Builds a synthetic dataset with known DTW-cluster ground truth, runs one
epoch of `PositiveMiningBatchSampler`, and asserts the per-anchor density
of true positives is higher than a `RandomSampler` baseline.

This catches regressions where mining emits batches of the right shape but
does not include the expected positive examples.

Pure CPU; no GPU required (the DTW precompute is short-circuited with a
hand-crafted neighbor table).

Run from retrieval_training/: python -m pytest tests/test_positive_mining_density.py -v
"""

import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Dataset, RandomSampler

THIS_DIR = Path(__file__).resolve().parent
TRAINING_DIR = THIS_DIR.parent
sys.path.insert(0, str(TRAINING_DIR))

from positive_mining_sampler import PositiveMiningBatchSampler


# ---------------------------------------------------------------------------
# Synthetic ground truth: N=400 clips in C=100 clusters of 4 each.
# Each clip's "true positives" = the other 3 clips in its cluster.
#
# The cluster_size/N ratio is intentionally small (~1%) so that random
# batches at B=40 organically contain very few cluster-mates per anchor
# (~0.29 expected). This keeps the mining-vs-random gap large and stable
# across CI runs.
# ---------------------------------------------------------------------------

N = 400
CLUSTER_SIZE = 4
N_CLUSTERS = N // CLUSTER_SIZE   # 100

assert N % CLUSTER_SIZE == 0, "test fixture: N must be divisible by CLUSTER_SIZE"


def _cluster_of(clip_idx: int) -> int:
    return clip_idx // CLUSTER_SIZE


def _true_positives_of(clip_idx: int) -> set:
    """Return the set of in-cluster neighbors (excluding self)."""
    c = _cluster_of(clip_idx)
    return {p for p in range(c * CLUSTER_SIZE, (c + 1) * CLUSTER_SIZE)
            if p != clip_idx}


def _build_synthetic_neighbor_table(top_k: int = 3) -> torch.Tensor:
    """Hand-crafted top-K = the 9 in-cluster neighbors per clip.

    Mirrors what `build_dtw_neighbors.py` would write for a perfectly-
    separated dataset. Skips the GPU step entirely since we're testing the
    sampler, not the precompute.
    """
    neigh = torch.full((N, top_k), -1, dtype=torch.int32)
    for i in range(N):
        peers = sorted(_true_positives_of(i))[:top_k]
        for k, p in enumerate(peers):
            neigh[i, k] = p
    return neigh


class _SyntheticDataset(Dataset):
    """Returns position itself; collate just stacks them."""
    def __len__(self) -> int:
        return N

    def __getitem__(self, idx: int):
        return {
            "jepa_features": torch.zeros(2, 4, dtype=torch.float16),
            "clip_index": idx,
            "trajectory": torch.zeros(4, 3),
            "traj_length": 4,
        }


def _identity_clip_idx_at(p: int) -> int:
    return p


# ---------------------------------------------------------------------------
# Density measurement
# ---------------------------------------------------------------------------

def _measure_in_batch_positive_density(batches: list, group_size: int = None,
                                        anchors_are_first_in_group: bool = False):
    """Per-anchor count of true positives in the same batch, averaged.

    For mining: anchor is at every group_size'th index (slot 0 of each group).
    For random: every clip in the batch is treated as an anchor (and we count
    its true positives in the rest of the batch).
    """
    total_anchors = 0
    total_in_batch_positives = 0

    for batch in batches:
        batch_set = set(batch)
        if anchors_are_first_in_group:
            anchors = [batch[i] for i in range(0, len(batch), group_size)]
        else:
            anchors = list(batch)
        for a in anchors:
            tps = _true_positives_of(a)
            total_anchors += 1
            total_in_batch_positives += len(tps & (batch_set - {a}))

    return total_in_batch_positives / max(total_anchors, 1)


# ---------------------------------------------------------------------------
# Test
# ---------------------------------------------------------------------------

class TestMiningDensity:
    def test_mining_beats_random_baseline(self):
        from data import collate_fn

        batch_size = 40
        positives_per_anchor = 3   # G=10 anchor groups × 4 = 40
        clip_video_number = torch.zeros(N, dtype=torch.int32)
        train_positions = list(range(N))
        position_of_clip_idx = {i: i for i in range(N)}
        neigh = _build_synthetic_neighbor_table(top_k=3)

        sampler = PositiveMiningBatchSampler(
            train_positions=train_positions,
            clip_idx_at=_identity_clip_idx_at,
            position_of_clip_idx=position_of_clip_idx,
            neighbor_clip_idx=neigh,
            clip_video_number=clip_video_number,
            batch_size=batch_size,
            positives_per_anchor=positives_per_anchor,
            exclude_same_video_positives=False,  # all clips are video 0
            pad_policy="random_train",
            seed=0,
        )

        ds = _SyntheticDataset()
        mining_loader = DataLoader(
            ds, batch_sampler=sampler, collate_fn=collate_fn,
        )
        mining_batches = [b["clip_indices"].tolist() for b in mining_loader]
        mining_density = _measure_in_batch_positive_density(
            mining_batches, group_size=sampler.group_size,
            anchors_are_first_in_group=True,
        )

        # Random baseline at the same batch size.
        random_loader = DataLoader(
            ds, batch_size=batch_size, shuffle=True, collate_fn=collate_fn,
            drop_last=True,
        )
        torch.manual_seed(0)
        random_batches = [b["clip_indices"].tolist() for b in random_loader]
        random_density = _measure_in_batch_positive_density(
            random_batches, anchors_are_first_in_group=False,
        )

        # Each anchor's 3 mined positives are guaranteed cluster-mates,
        # so the density must be at least 3.
        assert mining_density >= 3.0, (
            f"mining density too low: {mining_density:.2f} (positives are "
            f"meant to be guaranteed cluster-mates)"
        )
        # Random baseline at B=40 / N=400 / cluster=4: per anchor expects
        # roughly (40-1) * 3/399 ≈ 0.29 cluster-mates. Mining should beat
        # this by at least 5×.
        assert mining_density >= 5.0 * random_density, (
            f"mining density {mining_density:.2f} not >= 5x random "
            f"density {random_density:.2f}"
        )

    def test_anchor_seen_once_in_density_run(self):
        # Sanity: every train position appears as an anchor exactly once
        # across one epoch of batches (with drop_last=True remainder dropped).
        clip_video_number = torch.zeros(N, dtype=torch.int32)
        position_of_clip_idx = {i: i for i in range(N)}
        neigh = _build_synthetic_neighbor_table(top_k=3)

        sampler = PositiveMiningBatchSampler(
            train_positions=list(range(N)),
            clip_idx_at=_identity_clip_idx_at,
            position_of_clip_idx=position_of_clip_idx,
            neighbor_clip_idx=neigh,
            clip_video_number=clip_video_number,
            batch_size=40,
            positives_per_anchor=3,
            exclude_same_video_positives=False,
            pad_policy="random_train",
            seed=0,
        )
        anchors = []
        for b in sampler:
            for i in range(0, len(b), sampler.group_size):
                anchors.append(b[i])
        # N=400 anchors, batch=40 → G=10, len=40 batches × 10 = 400
        assert sorted(anchors) == list(range(N))
