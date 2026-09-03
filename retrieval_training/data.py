from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import torch
from torch.utils.data import Dataset


# V-JEPA 2: tubelet_size=2, patch_size=16, 384x384 input → 24*24 spatial tokens per tubelet,
# laid out time-major as [tubelet0_spatial..., tubelet1_spatial..., ...].
FRAMES_PER_TUBELET = 2
TOKENS_PER_TUBELET = 24 * 24  # 576


def _truncate_to_max_frames(feats: torch.Tensor, max_clip_frames: int) -> torch.Tensor:
    """Drop tubelets evenly from start/end so feats covers <= max_clip_frames frames.

    feats has shape (num_tubelets * TOKENS_PER_TUBELET, D). Odd-extra goes to end.
    """
    max_tubelets = max_clip_frames // FRAMES_PER_TUBELET
    num_tubelets = feats.shape[0] // TOKENS_PER_TUBELET
    if num_tubelets <= max_tubelets:
        return feats
    extra = num_tubelets - max_tubelets
    drop_start = extra // 2
    start = drop_start * TOKENS_PER_TUBELET
    end = start + max_tubelets * TOKENS_PER_TUBELET
    return feats[start:end]


class VideoClipDataset(Dataset):
    """Dataset that lazily loads precomputed V-JEPA 2 features from per-clip files.

    Each item yields per-token JEPA features (loaded from disk on demand) and
    the clip's integer index for online DTW lookup at training time.

    Expected directory layout under `data_dir`:
        jepa_features/<vn>/<node_uid>.pt — Tensor[num_tokens, 1408] per clip

    `trajectories_path` is required and points to the `.pt` artifact built by
    `build_trajectories.py` — `clip_index` is derived from its `clip_keys` and
    `(trajectories, lengths)` are loaded into RAM for per-batch DTW.
    """

    def __init__(
        self,
        data_dir: str,
        dtw_design: str = "first_frame_midpoint_hand_length_no_z",
        max_clip_frames: int = -1,
        trajectories_path: str | None = None,
        load_jepa_features: bool = True,
    ):
        if trajectories_path is None:
            raise ValueError(
                "VideoClipDataset requires `trajectories_path`; build it via "
                "build_trajectories.build_trajectories() first."
            )
        self.max_clip_frames = max_clip_frames
        # When False (trajectory input mode) the model never consumes JEPA
        # features, so we skip the per-clip disk loads (and the on-disk
        # existence check below) entirely.
        self.load_jepa_features = load_jepa_features
        data_dir = Path(data_dir)
        self.features_dir = data_dir / "jepa_features"

        payload = torch.load(trajectories_path, weights_only=False, map_location="cpu")
        if payload.get("design") != dtw_design:
            raise RuntimeError(
                f"trajectories_path design={payload.get('design')!r} != "
                f"dataset dtw_design={dtw_design!r}"
            )
        keys = payload["clip_keys"]
        self.clip_index: Dict[int, Dict[str, Any]] = {
            i: {"video_number": int(vn), "node_uid": str(uid)}
            for i, (vn, uid) in enumerate(keys)
        }
        self.trajectories: torch.Tensor = (
            payload["trajectories"].to(torch.float32).contiguous()
        )
        self.traj_lengths: torch.Tensor = (
            payload["lengths"].to(torch.int32).contiguous()
        )
        print(f"VideoClipDataset: loaded trajectories shape={tuple(self.trajectories.shape)} "
              f"and clip_index (N={len(self.clip_index)}) from {trajectories_path}")

        # The trajectory artifact is built with a feature-aware filter
        # (see build_trajectories.py), so every clip in clip_index must have
        # jepa_features on disk. A per-clip stat is faster than walking
        # features_dir and trips loudly if features were deleted out from
        # under a cached trajectory artifact.
        all_indices = sorted(self.clip_index.keys())
        if self.load_jepa_features:
            missing = [i for i in all_indices if not self._clip_path(i).exists()]
            if missing:
                raise RuntimeError(
                    f"VideoClipDataset: {len(missing)} clips in trajectories "
                    f"artifact have no jepa_features on disk "
                    f"(e.g. clip_idx={missing[:3]}). Rebuild the trajectories "
                    f"artifact — its feature filter is stale."
                )
        self.indices: List[int] = all_indices
        print(f"VideoClipDataset: {len(self.indices)} clips "
              f"({'with jepa features' if self.load_jepa_features else 'trajectory-only'})")

    def _clip_path(self, clip_idx: int) -> Path:
        info = self.clip_index[clip_idx]
        return self.features_dir / str(info["video_number"]) / f"{info['node_uid']}.pt"

    # ------------------------------------------------------------------
    # Position-based accessors.
    #
    # `position` is the index into __getitem__ (i.e., 0..len(self)-1). It is
    # the same index space used by `Subset(dataset, positions)`. Internally
    # the dataset translates `position -> clip_idx` (the row in
    # `self.trajectories` / `self.clip_index`) via `self.indices`. Callers
    # should use these helpers instead of reaching into `self.indices` so the
    # translation is done in one place and stays correct under wrapping.
    # ------------------------------------------------------------------

    def clip_idx_at(self, position: int) -> int:
        """Return the raw clip_idx (row in trajectories/clip_index) at `position`."""
        return self.indices[position]

    def get_trajectories_by_position(
        self, positions: Sequence[int],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return (trajectories, lengths) for the given dataset positions.

        Output rows are ordered to match `positions`. Lookup is one indirection
        through `self.indices` (position -> clip_idx) followed by a single
        `index_select` per tensor.
        """
        clip_idx = torch.as_tensor(
            [self.indices[p] for p in positions], dtype=torch.long,
        )
        return (
            self.trajectories.index_select(0, clip_idx),
            self.traj_lengths.index_select(0, clip_idx),
        )

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        clip_idx = self.indices[idx]
        item: Dict[str, Any] = {
            "clip_index": clip_idx,
            "trajectory": self.trajectories[clip_idx],
            "traj_length": int(self.traj_lengths[clip_idx].item()),
        }
        if self.load_jepa_features:
            feats = torch.load(
                self._clip_path(clip_idx), weights_only=True, map_location="cpu",
            )
            if self.max_clip_frames > 0:
                feats = _truncate_to_max_frames(feats, self.max_clip_frames)
            # Keep stored fp16 on the worker side; cast to fp32 only after
            # .to(device) in the train/eval loops. Casting here roughly halves
            # CPU RAM consumption in the dataloader queue.
            item["jepa_features"] = feats
        return item


class JepaPooledDataset(Dataset):
    """Yields per-clip mean-pooled, L2-normalised JEPA features `v_ref` as `[D]` fp32.

    Wraps a `VideoClipDataset` and a list of `positions` into it. Pooling is
    done inside the worker so the dataloader queue holds only `[D]` vectors
    (~5.6 KB at D=1408) instead of padded `[T, D]` blocks (~6 GB at B=196,
    T=11520). Used by `train._build_eval_jepa_sim` to compute the eval JEPA
    similarity matrix without exhausting host RAM.
    """

    def __init__(self, parent: "VideoClipDataset", positions: Sequence[int]):
        self.parent = parent
        self.positions = list(positions)

    def __len__(self) -> int:
        return len(self.positions)

    def __getitem__(self, i: int) -> torch.Tensor:
        pos = self.positions[i]
        clip_idx = self.parent.indices[pos]
        feats = torch.load(
            self.parent._clip_path(clip_idx), weights_only=True, map_location="cpu",
        )
        if self.parent.max_clip_frames > 0:
            feats = _truncate_to_max_frames(feats, self.parent.max_clip_frames)
        v = feats.to(torch.float32).mean(dim=0)
        v = torch.nn.functional.normalize(v, dim=-1)
        return v


def collate_fn(batch: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
    out: Dict[str, torch.Tensor] = {
        "clip_indices": torch.tensor(
            [item["clip_index"] for item in batch], dtype=torch.long,
        ),
        "trajectories": torch.stack([item["trajectory"] for item in batch], dim=0),
        "traj_lengths": torch.tensor(
            [item["traj_length"] for item in batch], dtype=torch.int32,
        ),
    }

    # JEPA features are only present when the dataset is built with
    # load_jepa_features=True (skipped in trajectory input mode).
    if "jepa_features" in batch[0]:
        max_tokens = max(item["jepa_features"].shape[0] for item in batch)
        dim = batch[0]["jepa_features"].shape[1]
        feat_dtype = batch[0]["jepa_features"].dtype

        padded = torch.zeros(len(batch), max_tokens, dim, dtype=feat_dtype)
        mask = torch.zeros(len(batch), max_tokens, dtype=torch.bool)
        for i, item in enumerate(batch):
            t = item["jepa_features"].shape[0]
            padded[i, :t] = item["jepa_features"]
            mask[i, :t] = True
        out["jepa_features"] = padded
        out["padding_mask"] = mask

    return out


def select_model_input(
    batch: Dict[str, torch.Tensor],
    input_mode: str,
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Return `(feats, mask)` to feed the model, per the configured input mode.

    Shared by train.py, eval.py, and post_training/get_embeddings.py so the
    input-selection logic lives in exactly one place.

    - ``"jepa"``:       feats = ``jepa_features`` [B, T, encoder_dim] (fp16),
                        mask = ``padding_mask`` (bool, True = real token).
    - ``"trajectory"``: feats = ``trajectories`` [B, T_max, D] (fp32),
                        mask derived from ``traj_lengths`` (True = real frame).
                        Trajectories use boundary-repeat padding and carry no
                        boolean mask, so we build one here.

    The DTW target inputs (``trajectories`` / ``traj_lengths``) are always read
    straight from the batch regardless of mode — in trajectory mode `feats` and
    the DTW source are literally the same tensor.
    """
    if input_mode == "trajectory":
        feats = batch["trajectories"].to(device, non_blocking=True)
        traj_lens = batch["traj_lengths"].to(device, non_blocking=True)
        T = feats.shape[1]
        mask = torch.arange(T, device=device)[None, :] < traj_lens[:, None]
        return feats, mask
    if input_mode == "jepa":
        feats = batch["jepa_features"].to(device, non_blocking=True)
        mask = batch["padding_mask"].to(device, non_blocking=True)
        return feats, mask
    raise ValueError(f"Unknown input_mode={input_mode!r}; expected 'jepa' or 'trajectory'")


