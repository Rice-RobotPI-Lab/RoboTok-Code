"""Hard-positive batch sampler for trajectory retrieval training.

Each batch is a set of `G` anchor groups. Standard groups contain an anchor and
its positives; set/rank-loss groups additionally contain explicit DTW boundary
negatives. The G anchors are drawn from `train_positions` without
replacement per epoch (every train clip serves as anchor exactly once);
their positives come from a precomputed top-K DTW-neighbor table built by
`build_dtw_neighbors.py`.

Each anchor is batched with up to `P` precomputed DTW neighbors, making those
positive pairs available to the batch-local training objective.

Index-space note: the sampler emits positions into the **base
`VideoClipDataset`** (the index space `__getitem__` accepts), NOT raw
clip indices and NOT positions into a `Subset`. The cache stores raw clip
indices (rows of `dataset.trajectories`); we translate via
`position_of_clip_idx`. When `use_positive_mining=True`, the train
DataLoader must be passed the **base** dataset (no `Subset` wrapper) so
the sampler's positions resolve correctly.
"""

from __future__ import annotations

from typing import Callable, Iterator, List, Mapping, Sequence

import torch
from torch.utils.data import Sampler


class PositiveMiningBatchSampler(Sampler[List[int]]):
    """BatchSampler that packs each batch with G anchor groups.

    Each batch has length ``batch_size = G * group_size``.
    Anchors are a seeded random permutation of `train_positions` (one anchor
    per group per epoch). Positives are drawn from the precomputed
    `neighbor_clip_idx` cache, filtered to (a) only positions in
    `train_positions` (no eval leakage) and (b) optionally same-video
    positives.

    Args:
        train_positions: dataset positions allowed as anchors / positives
            (typically the `train` half of the random_holdout_split).
        clip_idx_at: callable mapping dataset position -> raw clip_idx (=
            row in `dataset.trajectories` and key in `dataset.clip_index`).
            Use `dataset.clip_idx_at`.
        position_of_clip_idx: inverse mapping (raw clip_idx -> position).
            Build it once with
            `{dataset.clip_idx_at(p): p for p in range(len(dataset))}`.
        neighbor_clip_idx: `Tensor[N_full, K]` int — full-N table from the
            cache. Padding entries are `-1`.
        clip_video_number: `Tensor[N_full]` int — video number per raw
            clip_idx. Used for same-video positive filtering. Build with
            `[dataset.clip_index[i]["video_number"] for i in range(...)]`.
        batch_size: target batch length. MUST be divisible by
            `(1 + positives_per_anchor)`.
        positives_per_anchor: `P`; number of positives per anchor in a group.
        exclude_same_video_positives: drop neighbors whose video_number
            equals the anchor's. Recommended (avoids trivially-near
            temporally-adjacent clips dominating positives).
        pad_policy: when an anchor has < P positives surviving filtering:
            `"random_train"` (default) pads with random `train_positions`;
            `"drop_anchor"` skips the anchor entirely (changes `__len__`
            to a worst-case upper bound — not recommended).
        drop_last: if True, drop the trailing partial chunk of anchors.
        seed: base RNG seed; per-epoch RNG is `seed + self._epoch`.

    The `pad_invocations` counter is reset at each `__iter__` and reports
    the number of positive slots that had to be filled with random negatives
    during that epoch — surface this as `train/positive_pad_invocations` in
    TB to diagnose cache quality.

    **Per-batch dedup guarantee**: every clip appears at most once per batch.
    Within a group, positives never duplicate the anchor. Across groups, the
    sampler tracks a `batch_seen` set; an anchor that would collide with an
    earlier group's positive is **deferred** to the next batch (not dropped),
    so the once-per-epoch anchor invariant is preserved modulo a small lag.
    """

    def __init__(
        self,
        train_positions: Sequence[int],
        clip_idx_at: Callable[[int], int],
        position_of_clip_idx: Mapping[int, int],
        neighbor_clip_idx: torch.Tensor,
        clip_video_number: torch.Tensor,
        batch_size: int,
        positives_per_anchor: int,
        exclude_same_video_positives: bool,
        boundary_negatives_per_anchor: int = 0,
        positive_rank_end: int | None = None,
        boundary_rank_start: int = 11,
        boundary_rank_end: int = 20,
        pad_policy: str = "random_train",
        drop_last: bool = True,
        seed: int = 0,
    ) -> None:
        if positives_per_anchor < 1:
            raise ValueError(
                f"positives_per_anchor must be >= 1; got {positives_per_anchor}"
            )
        if boundary_negatives_per_anchor < 0:
            raise ValueError("boundary_negatives_per_anchor must be >= 0")
        if positive_rank_end is None:
            positive_rank_end = neighbor_clip_idx.shape[1]
        if boundary_negatives_per_anchor > 0:
            if positive_rank_end < positives_per_anchor:
                raise ValueError("positive_rank_end is smaller than positives_per_anchor")
            if boundary_rank_start <= positive_rank_end:
                raise ValueError("boundary rank range must begin after positive rank range")
            if boundary_rank_end < boundary_rank_start:
                raise ValueError("invalid boundary rank range")
            if neighbor_clip_idx.shape[1] < boundary_rank_end:
                raise ValueError("neighbor cache does not cover boundary rank range")
        group_size = 1 + positives_per_anchor + boundary_negatives_per_anchor
        if batch_size % group_size != 0:
            raise ValueError(
                f"batch_size={batch_size} must be divisible by "
                f"(1 + positives_per_anchor)={group_size}"
            )
        if pad_policy not in ("random_train", "drop_anchor"):
            raise ValueError(
                f"Unknown pad_policy={pad_policy!r}; "
                "expected 'random_train' or 'drop_anchor'"
            )

        self.train_positions: List[int] = list(train_positions)
        self.train_positions_set = frozenset(self.train_positions)
        self.clip_idx_at = clip_idx_at
        self.position_of_clip_idx = position_of_clip_idx
        self.neighbor_clip_idx = neighbor_clip_idx
        self.clip_video_number = clip_video_number
        self.batch_size = batch_size
        self.positives_per_anchor = positives_per_anchor
        self.boundary_negatives_per_anchor = boundary_negatives_per_anchor
        self.positive_rank_end = int(positive_rank_end)
        self.boundary_rank_start = int(boundary_rank_start)
        self.boundary_rank_end = int(boundary_rank_end)
        self.exclude_same_video_positives = exclude_same_video_positives
        self.pad_policy = pad_policy
        self.drop_last = drop_last
        self.seed = seed
        self._epoch = 0

        self.group_size = group_size
        self.anchors_per_batch = batch_size // group_size
        self.pad_invocations = 0
        self.skipped_anchor_invocations = 0

    def set_epoch(self, epoch: int) -> None:
        """Set the per-epoch RNG seed offset.

        Mirrors the `DistributedSampler` API. Call this from the train loop
        right before `iter(dataloader)` so each epoch reshuffles the anchor
        order deterministically (same `seed + epoch` reproduces the same
        order). Without this, `persistent_workers=True` would re-iterate
        the same epoch's order every time.
        """
        self._epoch = int(epoch)

    def __len__(self) -> int:
        n_anchors = len(self.train_positions)
        if self.drop_last:
            return n_anchors // self.anchors_per_batch
        return (n_anchors + self.anchors_per_batch - 1) // self.anchors_per_batch

    def _filtered_positives(
        self,
        anchor_position: int,
        already_in_group: set,
    ) -> List[int]:
        """Return up to `P` neighbor positions for this anchor, filtered."""
        anchor_clip_idx = self.clip_idx_at(anchor_position)
        anchor_video = int(self.clip_video_number[anchor_clip_idx].item())
        cands = self.neighbor_clip_idx[anchor_clip_idx].tolist()

        out: List[int] = []
        for ci in cands:
            ci = int(ci)
            if ci < 0:
                continue
            pos = self.position_of_clip_idx.get(ci)
            if pos is None:
                continue
            if pos not in self.train_positions_set:
                continue
            if pos == anchor_position or pos in already_in_group:
                continue
            if self.exclude_same_video_positives:
                if int(self.clip_video_number[ci].item()) == anchor_video:
                    continue
            out.append(pos)
            if len(out) == self.positives_per_anchor:
                break
        return out

    def _sample_rank_range(
        self,
        anchor_position: int,
        rank_start: int,
        rank_end: int,
        count: int,
        forbidden: set,
        gen: torch.Generator,
        *,
        filter_same_video: bool,
    ) -> List[int]:
        """Sample dataset positions from a one-indexed cached DTW rank range."""
        anchor_clip_idx = self.clip_idx_at(anchor_position)
        anchor_video = int(self.clip_video_number[anchor_clip_idx].item())
        candidates: List[int] = []
        for ci in self.neighbor_clip_idx[
            anchor_clip_idx, rank_start - 1:rank_end
        ].tolist():
            ci = int(ci)
            if ci < 0:
                continue
            pos = self.position_of_clip_idx.get(ci)
            if pos is None or pos not in self.train_positions_set:
                continue
            if pos == anchor_position or pos in forbidden or pos in candidates:
                continue
            if filter_same_video and int(self.clip_video_number[ci].item()) == anchor_video:
                continue
            candidates.append(pos)
        if len(candidates) < count:
            return candidates
        order = torch.randperm(len(candidates), generator=gen)[:count].tolist()
        return [candidates[i] for i in order]

    def _pad_with_random_train(
        self,
        n_needed: int,
        forbidden: set,
        gen: torch.Generator,
    ) -> List[int]:
        """Sample `n_needed` random positions from `train_positions` not in
        `forbidden`. Uses the per-epoch generator for determinism."""
        out: List[int] = []
        max_tries = max(64, 4 * n_needed)
        for _ in range(max_tries):
            if len(out) == n_needed:
                break
            r = int(torch.randint(
                0, len(self.train_positions), (1,), generator=gen,
            ).item())
            cand = self.train_positions[r]
            if cand in forbidden:
                continue
            forbidden.add(cand)
            out.append(cand)
        # If we still couldn't fill (extremely unlikely — train pool >> batch),
        # fall back to allow-with-replacement so the batch shape is preserved.
        while len(out) < n_needed:
            r = int(torch.randint(
                0, len(self.train_positions), (1,), generator=gen,
            ).item())
            out.append(self.train_positions[r])
        return out

    def __iter__(self) -> Iterator[List[int]]:
        self.pad_invocations = 0
        self.skipped_anchor_invocations = 0
        gen = torch.Generator()
        gen.manual_seed(self.seed + self._epoch)

        n_anchors = len(self.train_positions)
        perm = torch.randperm(n_anchors, generator=gen).tolist()
        anchor_order = [self.train_positions[i] for i in perm]

        # Anchor flow: pull fresh anchors from `anchor_order`. If an anchor
        # collides with the current batch's `batch_seen` set (because a
        # previous group already pulled it as a positive), defer it to the
        # next batch's `deferred` queue. Each new batch tries deferred first,
        # then fresh, so a deferred anchor almost always lands on its first
        # retry (collision rate per fresh batch ≈ batch_size / N_train ≪ 1
        # at the production scale).
        pool_idx = 0
        deferred: List[int] = []

        def _next_anchor():
            nonlocal pool_idx
            if deferred:
                return deferred.pop(0)
            if pool_idx < len(anchor_order):
                a = anchor_order[pool_idx]
                pool_idx += 1
                return a
            return None

        n_batches = len(self)
        for _ in range(n_batches):
            batch: List[int] = []
            batch_seen: set = set()
            new_deferred: List[int] = []

            while len(batch) < self.batch_size:
                anchor_pos = _next_anchor()
                if anchor_pos is None:
                    break  # ran out of anchors for this epoch

                # Cross-group dedup: if this anchor is already in the batch
                # (likely as a positive of an earlier group), defer it for the
                # NEXT batch. Don't requeue within the same batch — that risks
                # an infinite loop if the same conflict keeps re-appearing.
                if anchor_pos in batch_seen:
                    new_deferred.append(anchor_pos)
                    continue

                # `batch_seen` enters the forbidden set so positives can't
                # collide with already-placed clips from earlier groups.
                forbidden_for_positives = batch_seen | {anchor_pos}
                boundaries: List[int] = []

                if self.boundary_negatives_per_anchor > 0:
                    positives = self._sample_rank_range(
                        anchor_pos, 1, self.positive_rank_end,
                        self.positives_per_anchor, forbidden_for_positives, gen,
                        filter_same_video=self.exclude_same_video_positives,
                    )
                    boundary_forbidden = forbidden_for_positives | set(positives)
                    boundaries = self._sample_rank_range(
                        anchor_pos, self.boundary_rank_start, self.boundary_rank_end,
                        self.boundary_negatives_per_anchor, boundary_forbidden, gen,
                        filter_same_video=False,
                    )
                    if (
                        len(positives) < self.positives_per_anchor
                        or len(boundaries) < self.boundary_negatives_per_anchor
                    ):
                        self.pad_invocations += (
                            self.positives_per_anchor - len(positives)
                            + self.boundary_negatives_per_anchor - len(boundaries)
                        )
                        self.skipped_anchor_invocations += 1
                        continue
                elif self.pad_policy == "drop_anchor":
                    positives = self._filtered_positives(
                        anchor_pos, already_in_group=forbidden_for_positives,
                    )
                    if len(positives) < self.positives_per_anchor:
                        self.pad_invocations += (
                            self.positives_per_anchor - len(positives)
                        )
                        continue
                else:  # "random_train"
                    positives = self._filtered_positives(
                        anchor_pos, already_in_group=forbidden_for_positives,
                    )
                    n_short = self.positives_per_anchor - len(positives)
                    if n_short > 0:
                        self.pad_invocations += n_short
                        # Update the forbidden set with the chosen positives
                        # before padding so random pads also avoid them.
                        pad_forbidden = (
                            forbidden_for_positives | set(positives)
                        )
                        positives.extend(self._pad_with_random_train(
                            n_short, pad_forbidden, gen,
                        ))

                batch.append(anchor_pos)
                batch.extend(positives)
                batch.extend(boundaries)
                batch_seen.add(anchor_pos)
                batch_seen.update(positives)
                batch_seen.update(boundaries)

            # Carry the just-deferred anchors into the next batch's pool.
            deferred.extend(new_deferred)

            if len(batch) == self.batch_size:
                yield batch
