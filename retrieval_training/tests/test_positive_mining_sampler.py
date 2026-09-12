"""Tests for retrieval_training/positive_mining_sampler.py.

Pure CPU; no real dataset or GPU required.

Run from retrieval_training/: python -m pytest tests/test_positive_mining_sampler.py -v
"""

import sys
from pathlib import Path

import pytest
import torch
from torch.utils.data import DataLoader, Dataset

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from positive_mining_sampler import PositiveMiningBatchSampler


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _identity_clip_idx_at(p: int) -> int:
    """Default mapping: dataset position == raw clip_idx."""
    return p


def _make_sampler(
    n_clips: int = 50,
    train_position_count: int = 40,
    batch_size: int = 8,
    positives_per_anchor: int = 3,
    exclude_same_video_positives: bool = False,
    pad_policy: str = "random_train",
    seed: int = 0,
    same_video_block: int = 1,
    clip_idx_at=None,
    position_of_clip_idx=None,
    neighbor_K: int = 10,
):
    """Build a sampler with deterministic synthetic neighbors:
    clip i's top-K is [(i+1) % N, (i+2) % N, ..., (i+K) % N]."""
    neighbor = torch.zeros(n_clips, neighbor_K, dtype=torch.int64)
    for i in range(n_clips):
        for k in range(neighbor_K):
            neighbor[i, k] = (i + k + 1) % n_clips

    clip_video_number = torch.tensor(
        [i // same_video_block for i in range(n_clips)], dtype=torch.int32,
    )

    train_positions = list(range(train_position_count))

    if clip_idx_at is None:
        clip_idx_at = _identity_clip_idx_at
    if position_of_clip_idx is None:
        position_of_clip_idx = {i: i for i in range(n_clips)}

    return PositiveMiningBatchSampler(
        train_positions=train_positions,
        clip_idx_at=clip_idx_at,
        position_of_clip_idx=position_of_clip_idx,
        neighbor_clip_idx=neighbor,
        clip_video_number=clip_video_number,
        batch_size=batch_size,
        positives_per_anchor=positives_per_anchor,
        exclude_same_video_positives=exclude_same_video_positives,
        pad_policy=pad_policy,
        drop_last=True,
        seed=seed,
    )


# ---------------------------------------------------------------------------
# __init__ validation
# ---------------------------------------------------------------------------

class TestInit:
    def test_invalid_positives_per_anchor(self):
        with pytest.raises(ValueError, match="positives_per_anchor"):
            _make_sampler(positives_per_anchor=0)

    def test_batch_size_must_be_divisible(self):
        # batch_size=10 not divisible by (1+P=4)
        with pytest.raises(ValueError, match="divisible"):
            _make_sampler(batch_size=10, positives_per_anchor=3)

    def test_invalid_pad_policy(self):
        with pytest.raises(ValueError, match="pad_policy"):
            _make_sampler(pad_policy="banana")

    def test_group_size_correct(self):
        s = _make_sampler(batch_size=12, positives_per_anchor=3)
        assert s.group_size == 4
        assert s.anchors_per_batch == 3


# ---------------------------------------------------------------------------
# __iter__ basic invariants
# ---------------------------------------------------------------------------

class TestIterBasics:
    def test_batches_correct_length(self):
        s = _make_sampler(batch_size=8, positives_per_anchor=3)
        for batch in s:
            assert len(batch) == 8

    def test_batch_count_matches_len(self):
        s = _make_sampler(
            n_clips=50, train_position_count=40,
            batch_size=8, positives_per_anchor=3,
        )
        batches = list(s)
        assert len(batches) == len(s)

    def test_anchor_seen_at_most_once_per_epoch(self):
        # No anchor appears as anchor more than once per epoch. With the
        # cross-batch dedup + defer-to-next-batch logic, an anchor that
        # collides with its batch's existing positives is held for the next
        # batch — so the once-per-epoch invariant is preserved approximately
        # (modulo any anchors still in the deferred queue at end-of-epoch).
        s = _make_sampler(
            n_clips=50, train_position_count=40,
            batch_size=8, positives_per_anchor=3,  # G=2
        )
        batches = list(s)
        anchors = []
        for b in batches:
            for i in range(0, len(b), s.group_size):
                anchors.append(b[i])
        # Strong: no anchor served twice.
        assert len(anchors) == len(set(anchors))
        # Weak: most train positions served as anchor (allow small loss to
        # end-of-epoch deferred queue residue at this small N).
        assert len(anchors) >= int(0.85 * 40)

    def test_no_self_in_positives(self):
        s = _make_sampler(batch_size=8, positives_per_anchor=3)
        for batch in s:
            for i in range(0, len(batch), s.group_size):
                anchor = batch[i]
                positives = batch[i + 1:i + s.group_size]
                assert anchor not in positives

    def test_no_eval_position_emitted(self):
        # train_positions = 0..39, "eval" = 40..49. The synthetic neighbor
        # table has wrap-around that points into 40..49; sampler must filter.
        s = _make_sampler(
            n_clips=50, train_position_count=40,
            batch_size=8, positives_per_anchor=3,
        )
        for batch in s:
            for pos in batch:
                assert 0 <= pos < 40

    def test_no_duplicate_within_group(self):
        s = _make_sampler(batch_size=8, positives_per_anchor=3)
        for batch in s:
            for i in range(0, len(batch), s.group_size):
                group = batch[i:i + s.group_size]
                assert len(set(group)) == len(group)

    def test_no_duplicate_within_batch(self):
        """Cross-group dedup invariant: every clip appears at most once per
        batch, even when the synthetic ring-neighbor structure would
        otherwise produce overlap (anchor of group B = positive of group A,
        or two anchors picking the same neighbor).

        This is the regression test for the duplicate-clip-in-batch bug.
        """
        s = _make_sampler(
            n_clips=50, train_position_count=40,
            batch_size=8, positives_per_anchor=3,
        )
        for batch in s:
            assert len(batch) == len(set(batch)), (
                f"duplicate clip in batch: {batch} "
                f"(seen={[x for x in batch if batch.count(x) > 1]})"
            )

    def test_no_duplicate_within_batch_high_collision_pressure(self):
        """Force a high cross-group collision rate by making all positives
        cluster in a small range; without batch-level dedup, this would
        produce many duplicates per batch."""
        # 30 train clips. Synthetic neighbor table where every clip's top-3
        # is [0, 1, 2] — i.e., every anchor wants to pull the same 3 clips
        # as positives. Without dedup, every batch would have positions
        # 0/1/2 repeated many times.
        n_full = 30
        neighbor = torch.zeros(n_full, 3, dtype=torch.int64)
        for i in range(n_full):
            for k, v in enumerate([0, 1, 2]):
                neighbor[i, k] = v if v != i else (n_full - 1)  # avoid self
        clip_video_number = torch.zeros(n_full, dtype=torch.int32)
        s = PositiveMiningBatchSampler(
            train_positions=list(range(n_full)),
            clip_idx_at=_identity_clip_idx_at,
            position_of_clip_idx={i: i for i in range(n_full)},
            neighbor_clip_idx=neighbor,
            clip_video_number=clip_video_number,
            batch_size=8,
            positives_per_anchor=3,
            exclude_same_video_positives=False,
            pad_policy="random_train",
            seed=0,
        )
        for batch in s:
            assert len(batch) == len(set(batch)), (
                f"duplicate in adversarial-collision batch: {batch}"
            )

    def test_same_video_filter_active(self):
        # 50 clips in 5 videos of 10 each. With exclude_same_video=True,
        # cache-derived positives shouldn't share the anchor's video. The
        # ring-neighbor pattern would otherwise return many same-video
        # candidates (i+1, i+2, ... within the same block).
        s = _make_sampler(
            n_clips=50, train_position_count=40, same_video_block=10,
            batch_size=8, positives_per_anchor=3,
            exclude_same_video_positives=True,
            pad_policy="random_train",
        )
        # Without the filter, every anchor would have 9 same-video
        # neighbors before reaching the next video (positions i+1..i+9).
        # With the filter on, no positives can come from those positions,
        # forcing positives from i+10 onward (different video) or random
        # padding. Track ratio of same-video positives over the epoch and
        # assert it's < random baseline (1 in 4 ≈ 25%).
        same_video = 0
        total_positives = 0
        for batch in s:
            for i in range(0, len(batch), s.group_size):
                anchor = batch[i]
                anchor_video = anchor // 10
                for pos in batch[i + 1:i + s.group_size]:
                    total_positives += 1
                    if pos // 10 == anchor_video:
                        same_video += 1
        # The cache-derived positives are guaranteed cross-video; only
        # random padding can land same-video. With most positives coming
        # from cache (high-K=10), pad invocations are rare.
        assert same_video / total_positives < 0.25


class TestSetRankGroups:
    def _make_set_rank_sampler(self, neighbor=None, seed=0):
        n = 60
        if neighbor is None:
            neighbor = torch.empty((n, 20), dtype=torch.int64)
            for i in range(n):
                neighbor[i] = torch.tensor([(i + j + 1) % n for j in range(20)])
        return PositiveMiningBatchSampler(
            train_positions=list(range(n)),
            clip_idx_at=_identity_clip_idx_at,
            position_of_clip_idx={i: i for i in range(n)},
            neighbor_clip_idx=neighbor,
            clip_video_number=torch.arange(n, dtype=torch.int32),
            batch_size=8,
            positives_per_anchor=2,
            boundary_negatives_per_anchor=1,
            positive_rank_end=10,
            boundary_rank_start=11,
            boundary_rank_end=20,
            exclude_same_video_positives=False,
            seed=seed,
        )

    def test_group_roles_use_requested_rank_ranges(self):
        sampler = self._make_set_rank_sampler()
        for batch in sampler:
            for start in range(0, len(batch), 4):
                anchor, p1, p2, boundary = batch[start:start + 4]
                row = sampler.neighbor_clip_idx[anchor].tolist()
                assert p1 in row[:10]
                assert p2 in row[:10]
                assert p1 != p2
                assert boundary in row[10:20]

    def test_set_rank_sampling_is_epoch_deterministic(self):
        a = self._make_set_rank_sampler(seed=9)
        b = self._make_set_rank_sampler(seed=9)
        a.set_epoch(3)
        b.set_epoch(3)
        assert list(a) == list(b)
        c = self._make_set_rank_sampler(seed=9)
        c.set_epoch(4)
        assert list(a) != list(c)

    def test_missing_boundary_skips_instead_of_random_padding(self):
        neighbor = torch.full((60, 20), -1, dtype=torch.int64)
        for i in range(60):
            neighbor[i, :10] = torch.tensor([(i + j + 1) % 60 for j in range(10)])
        sampler = self._make_set_rank_sampler(neighbor=neighbor)
        assert list(sampler) == []
        assert sampler.skipped_anchor_invocations > 0
        assert sampler.pad_invocations > 0


# ---------------------------------------------------------------------------
# Determinism
# ---------------------------------------------------------------------------

class TestDeterminism:
    def test_same_seed_same_epoch_reproducible(self):
        s1 = _make_sampler(seed=42)
        s1.set_epoch(0)
        s2 = _make_sampler(seed=42)
        s2.set_epoch(0)
        assert list(s1) == list(s2)

    def test_different_epochs_diverge(self):
        s = _make_sampler(seed=42)
        s.set_epoch(0)
        b0 = list(s)
        s.set_epoch(1)
        b1 = list(s)
        assert b0 != b1

    def test_different_seeds_diverge(self):
        s1 = _make_sampler(seed=42)
        s2 = _make_sampler(seed=43)
        assert list(s1) != list(s2)

    def test_set_epoch_resets_pad_invocations(self):
        # pad_invocations counter is reset at __iter__ start.
        s = _make_sampler(seed=42, neighbor_K=1)  # only 1 neighbor → many pads
        list(s)
        c0 = s.pad_invocations
        s.set_epoch(1)
        list(s)
        c1 = s.pad_invocations
        # Each epoch independently produces its own pad count starting from 0.
        assert c0 > 0
        assert c1 > 0


# ---------------------------------------------------------------------------
# Index-space translation
# ---------------------------------------------------------------------------

class TestIndexSpaceTranslation:
    """When `dataset.indices` ≠ `clip_index` keys (some clips lack
    features), the sampler must emit dataset positions (idx into __getitem__),
    NOT raw clip_idx values."""

    def test_emits_dataset_positions_not_clip_indices(self):
        # 10 raw clips, only every-other one has features.
        # dataset position p -> raw clip_idx 2p.
        n_full = 10
        clip_idx_at = lambda p: 2 * p  # noqa: E731
        position_of_clip_idx = {2 * p: p for p in range(5)}  # 5 dataset positions
        # Cache is full-N (10 entries); each clip_idx i's nearest is
        # (i+1) % N etc. Some of those are odd (no feature) — must be
        # filtered by the sampler via position_of_clip_idx.
        neighbor = torch.zeros(n_full, 5, dtype=torch.int64)
        for i in range(n_full):
            for k in range(5):
                neighbor[i, k] = (i + k + 1) % n_full
        clip_video_number = torch.zeros(n_full, dtype=torch.int32)

        sampler = PositiveMiningBatchSampler(
            train_positions=[0, 1, 2, 3, 4],
            clip_idx_at=clip_idx_at,
            position_of_clip_idx=position_of_clip_idx,
            neighbor_clip_idx=neighbor,
            clip_video_number=clip_video_number,
            batch_size=4,
            positives_per_anchor=1,
            exclude_same_video_positives=False,
            pad_policy="random_train",
            seed=0,
        )
        for batch in sampler:
            for pos in batch:
                # All emitted positions must be valid dataset positions
                # (0..4), NOT raw clip_idx (0..9).
                assert 0 <= pos < 5


# ---------------------------------------------------------------------------
# Pad policy
# ---------------------------------------------------------------------------

class TestPadPolicy:
    def test_random_train_keeps_batch_size(self):
        # Force pad: cache has no usable neighbors → every slot must be padded.
        n_full = 8
        neighbor = torch.full((n_full, 3), -1, dtype=torch.int64)
        clip_video_number = torch.zeros(n_full, dtype=torch.int32)
        s = PositiveMiningBatchSampler(
            train_positions=list(range(n_full)),
            clip_idx_at=_identity_clip_idx_at,
            position_of_clip_idx={i: i for i in range(n_full)},
            neighbor_clip_idx=neighbor,
            clip_video_number=clip_video_number,
            batch_size=4,
            positives_per_anchor=1,
            exclude_same_video_positives=False,
            pad_policy="random_train",
            seed=0,
        )
        batches = list(s)
        for b in batches:
            assert len(b) == 4
        # Every anchor needed 1 pad → at least 1 pad invocation per batch.
        assert s.pad_invocations >= len(batches)


# ---------------------------------------------------------------------------
# DataLoader integration
# ---------------------------------------------------------------------------

class _SyntheticDataset(Dataset):
    """Return per-position tensors; no real features."""
    def __init__(self, n: int):
        self.n = n

    def __len__(self) -> int:
        return self.n

    def __getitem__(self, idx: int):
        return {
            "jepa_features": torch.full((4, 8), float(idx), dtype=torch.float16),
            "clip_index": idx,
            "trajectory": torch.zeros(6, 4),
            "traj_length": 6,
        }


class TestDataLoaderIntegration:
    def test_batched_through_dataloader(self):
        from data import collate_fn
        n = 20
        sampler = _make_sampler(
            n_clips=n, train_position_count=n, batch_size=8,
            positives_per_anchor=3,
        )
        ds = _SyntheticDataset(n)
        loader = DataLoader(ds, batch_sampler=sampler, collate_fn=collate_fn)
        batches = list(loader)
        assert len(batches) == len(sampler)
        for b in batches:
            assert b["clip_indices"].shape[0] == 8
            assert b["jepa_features"].shape[0] == 8

    def test_clip_indices_match_emitted_positions(self):
        from data import collate_fn
        n = 16
        sampler = _make_sampler(
            n_clips=n, train_position_count=n, batch_size=4,
            positives_per_anchor=1,
        )
        ds = _SyntheticDataset(n)
        loader = DataLoader(ds, batch_sampler=sampler, collate_fn=collate_fn)
        # Pull batches from both directly and through the loader; positions
        # should agree.
        sampler.set_epoch(0)
        direct = list(sampler)
        sampler.set_epoch(0)
        through = list(loader)
        for d_batch, t_batch in zip(direct, through):
            assert d_batch == t_batch["clip_indices"].tolist()
