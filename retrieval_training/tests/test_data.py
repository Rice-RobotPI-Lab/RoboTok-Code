"""Extensive tests for retrieval_training/data.py.

Uses synthetic fixtures (tmp_path) so tests run without real data or GPU.
Run from retrieval_training/: python -m pytest tests/test_data.py -v
"""

import sys
from pathlib import Path

import pytest
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from data import (
    TOKENS_PER_TUBELET,
    VideoClipDataset,
    _truncate_to_max_frames,
    collate_fn,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

N_CLIPS = 10
EMBED_DIM = 32
DTW_DESIGN = "first_frame_midpoint_hand_length_no_z"
TRAJ_T = 8
TRAJ_D = 6
VIDEOS = {1: ["uid_a", "uid_b", "uid_c"], 2: ["uid_d", "uid_e"], 3: ["uid_f"]}
# variable token counts per clip
TOKEN_COUNTS = {
    "uid_a": 8, "uid_b": 12, "uid_c": 6,
    "uid_d": 10, "uid_e": 4,
    "uid_f": 14,
}


def _make_clip_keys():
    keys = []
    for vn, uids in sorted(VIDEOS.items()):
        for uid in uids:
            keys.append((vn, uid))
    return keys


def _make_trajectories_path(tmp_path: Path, design: str = DTW_DESIGN) -> Path:
    """Write a synthetic trajectories.pt artifact and return its path."""
    keys = _make_clip_keys()
    n = len(keys)
    payload = {
        "trajectories": torch.randn(n, TRAJ_T, TRAJ_D, dtype=torch.float32),
        "lengths": torch.full((n,), TRAJ_T, dtype=torch.int32),
        "design": design,
        "clip_keys": keys,
        "T_max": TRAJ_T,
        "D": TRAJ_D,
        "num_videos": len(VIDEOS),
    }
    out = tmp_path / f"trajectories_{design}.pt"
    torch.save(payload, out)
    return out


@pytest.fixture
def data_dir(tmp_path):
    """Create a synthetic training_data directory plus trajectories.pt."""
    # jepa_features/<vn>/<node_uid>.pt
    feat_dir = tmp_path / "jepa_features"
    feat_dir.mkdir()
    for vn, uids in VIDEOS.items():
        vdir = feat_dir / str(vn)
        vdir.mkdir()
        for uid in uids:
            t = TOKEN_COUNTS[uid]
            torch.save(torch.randn(t, EMBED_DIM, dtype=torch.float16), vdir / f"{uid}.pt")
    _make_trajectories_path(tmp_path)
    return tmp_path


@pytest.fixture
def trajectories_path(data_dir):
    return str(data_dir / f"trajectories_{DTW_DESIGN}.pt")


@pytest.fixture
def dataset(data_dir, trajectories_path):
    return VideoClipDataset(str(data_dir), trajectories_path=trajectories_path)


# ---------------------------------------------------------------------------
# VideoClipDataset.__init__
# ---------------------------------------------------------------------------

class TestDatasetInit:
    def test_length(self, dataset):
        assert len(dataset) == sum(len(v) for v in VIDEOS.values())

    def test_indices_sorted(self, dataset):
        assert dataset.indices == sorted(dataset.indices)

    def test_clip_index_keys_are_ints(self, dataset):
        for k in dataset.clip_index:
            assert isinstance(k, int)

    def test_clip_index_values_have_required_fields(self, dataset):
        for info in dataset.clip_index.values():
            assert "video_number" in info
            assert "node_uid" in info

    def test_features_dir_exists(self, dataset, data_dir):
        assert dataset.features_dir == data_dir / "jepa_features"

    def test_missing_trajectories_path_raises(self, tmp_path):
        with pytest.raises(ValueError, match="trajectories_path"):
            VideoClipDataset(str(tmp_path))

    def test_design_mismatch_raises(self, data_dir, tmp_path):
        bad = _make_trajectories_path(tmp_path, design="some_other_design")
        with pytest.raises(RuntimeError, match="design"):
            VideoClipDataset(str(data_dir), trajectories_path=str(bad))


# ---------------------------------------------------------------------------
# VideoClipDataset.__getitem__
# ---------------------------------------------------------------------------

class TestGetItem:
    def test_returns_dict_with_required_keys(self, dataset):
        item = dataset[0]
        assert "jepa_features" in item
        assert "clip_index" in item

    def test_features_preserve_storage_dtype(self, dataset):
        # Dataset deliberately keeps the on-disk dtype (fp16) to halve dataloader
        # RAM; cast to fp32 happens after .to(device) in train/eval loops.
        item = dataset[0]
        assert item["jepa_features"].dtype == torch.float16

    def test_features_shape(self, dataset):
        item = dataset[0]
        info = dataset.clip_index[dataset.indices[0]]
        expected_tokens = TOKEN_COUNTS[info["node_uid"]]
        assert item["jepa_features"].shape == (expected_tokens, EMBED_DIM)

    def test_clip_index_matches(self, dataset):
        for idx in range(len(dataset)):
            item = dataset[idx]
            assert item["clip_index"] == dataset.indices[idx]

    def test_all_items_loadable(self, dataset):
        for idx in range(len(dataset)):
            item = dataset[idx]
            assert item["jepa_features"].shape[1] == EMBED_DIM

    def test_missing_video_dir_raises(self, data_dir, trajectories_path):
        # Trajectory artifact is built with a feature-aware filter
        # (build_trajectories.py), so any clip in clip_index without a feature
        # file on disk means the artifact is stale — VideoClipDataset must
        # raise loudly instead of silently dropping.
        import shutil
        shutil.rmtree(data_dir / "jepa_features" / "1")
        with pytest.raises(RuntimeError, match="jepa_features"):
            VideoClipDataset(str(data_dir), trajectories_path=trajectories_path)

    def test_missing_clip_file_raises(self, data_dir, trajectories_path):
        first_uid = VIDEOS[1][0]
        (data_dir / "jepa_features" / "1" / f"{first_uid}.pt").unlink()
        with pytest.raises(RuntimeError, match="jepa_features"):
            VideoClipDataset(str(data_dir), trajectories_path=trajectories_path)


# ---------------------------------------------------------------------------
# Position-based accessors
# ---------------------------------------------------------------------------

class TestGetTrajectoriesByPosition:
    def test_full_range_matches_internal_state(self, dataset):
        positions = list(range(len(dataset)))
        trajs, lens = dataset.get_trajectories_by_position(positions)
        # Must match the same rows __getitem__ would return for these positions.
        for p in positions:
            item = dataset[p]
            assert torch.equal(trajs[p], item["trajectory"])
            assert int(lens[p].item()) == item["traj_length"]

    def test_subset_preserves_order(self, dataset):
        # Out-of-order positions: output rows must follow `positions` order,
        # not the underlying clip_idx order.
        positions = [3, 0, 5, 2]
        trajs, lens = dataset.get_trajectories_by_position(positions)
        for out_row, p in enumerate(positions):
            item = dataset[p]
            assert torch.equal(trajs[out_row], item["trajectory"])
            assert int(lens[out_row].item()) == item["traj_length"]

    def test_empty_positions(self, dataset):
        trajs, lens = dataset.get_trajectories_by_position([])
        assert trajs.shape[0] == 0
        assert lens.shape[0] == 0

    def test_works_through_subset_wrapper(self, dataset):
        # Subset(...).indices is the canonical way callers obtain a list of
        # positions; verify the accessor agrees with manual lookup through
        # Subset's __getitem__.
        from torch.utils.data import Subset
        positions = [1, 4, 5]
        subset = Subset(dataset, positions)
        trajs, lens = dataset.get_trajectories_by_position(list(subset.indices))
        for out_row in range(len(subset)):
            item = subset[out_row]
            assert torch.equal(trajs[out_row], item["trajectory"])
            assert int(lens[out_row].item()) == item["traj_length"]


# ---------------------------------------------------------------------------
# collate_fn
# ---------------------------------------------------------------------------

class TestCollateFn:
    def test_output_keys(self, dataset):
        batch = [dataset[0], dataset[1]]
        out = collate_fn(batch)
        assert set(out.keys()) == {
            "jepa_features", "padding_mask", "clip_indices",
            "trajectories", "traj_lengths",
        }

    def test_padding_to_max_length(self, dataset):
        items = [dataset[i] for i in range(len(dataset))]
        out = collate_fn(items)
        max_t = max(item["jepa_features"].shape[0] for item in items)
        assert out["jepa_features"].shape == (len(items), max_t, EMBED_DIM)
        assert out["padding_mask"].shape == (len(items), max_t)

    def test_mask_true_count_matches_real_tokens(self, dataset):
        items = [dataset[i] for i in range(3)]
        out = collate_fn(items)
        for i, item in enumerate(items):
            real_t = item["jepa_features"].shape[0]
            assert out["padding_mask"][i].sum().item() == real_t

    def test_mask_false_positions_are_zero(self, dataset):
        items = [dataset[0], dataset[1]]
        out = collate_fn(items)
        for i, item in enumerate(items):
            real_t = item["jepa_features"].shape[0]
            if real_t < out["jepa_features"].shape[1]:
                pad_region = out["jepa_features"][i, real_t:]
                assert (pad_region == 0).all()

    def test_real_region_matches_original(self, dataset):
        items = [dataset[0], dataset[1]]
        out = collate_fn(items)
        for i, item in enumerate(items):
            real_t = item["jepa_features"].shape[0]
            assert torch.allclose(out["jepa_features"][i, :real_t], item["jepa_features"])

    def test_clip_indices_dtype(self, dataset):
        out = collate_fn([dataset[0]])
        assert out["clip_indices"].dtype == torch.long

    def test_single_item_batch_no_padding(self, dataset):
        item = dataset[0]
        out = collate_fn([item])
        assert out["jepa_features"].shape[1] == item["jepa_features"].shape[0]
        assert out["padding_mask"].all()

    def test_uniform_length_no_padding(self, data_dir, trajectories_path):
        # Make all clips have the same token count
        feat_dir = data_dir / "jepa_features"
        for vn, uids in VIDEOS.items():
            vdir = feat_dir / str(vn)
            for uid in uids:
                torch.save(torch.randn(5, EMBED_DIM, dtype=torch.float16), vdir / f"{uid}.pt")
        ds = VideoClipDataset(str(data_dir), trajectories_path=trajectories_path)
        items = [ds[i] for i in range(len(ds))]
        out = collate_fn(items)
        assert out["padding_mask"].all()


# ---------------------------------------------------------------------------
# DataLoader integration
# ---------------------------------------------------------------------------

class TestDataLoaderIntegration:
    def test_dataloader_iteration(self, dataset):
        loader = DataLoader(dataset, batch_size=3, shuffle=False, collate_fn=collate_fn)
        batches = list(loader)
        total = sum(b["clip_indices"].shape[0] for b in batches)
        assert total == len(dataset)

    def test_dataloader_shuffle(self, dataset):
        loader = DataLoader(dataset, batch_size=len(dataset), shuffle=True, collate_fn=collate_fn)
        batch = next(iter(loader))
        assert batch["jepa_features"].shape[0] == len(dataset)

    def test_drop_last(self, dataset):
        loader = DataLoader(
            dataset, batch_size=4, shuffle=False, collate_fn=collate_fn, drop_last=True
        )
        for batch in loader:
            assert batch["clip_indices"].shape[0] == 4


# ---------------------------------------------------------------------------
# Tubelet-boundary truncation
# ---------------------------------------------------------------------------

class TestTruncateToMaxFrames:
    """_truncate_to_max_frames drops whole tubelets evenly; odd extra biases to end."""

    @staticmethod
    def _make_feats(num_tubelets: int, dim: int = 4) -> torch.Tensor:
        # tubelet i fills rows [i*576 : (i+1)*576] with the value i (so we can
        # check which tubelets survived after slicing).
        feats = torch.zeros(num_tubelets * TOKENS_PER_TUBELET, dim)
        for i in range(num_tubelets):
            feats[i * TOKENS_PER_TUBELET : (i + 1) * TOKENS_PER_TUBELET] = i
        return feats

    def test_no_truncation_when_under_cap(self):
        feats = self._make_feats(12)
        out = _truncate_to_max_frames(feats, max_clip_frames=40)
        assert out.shape[0] == 12 * TOKENS_PER_TUBELET
        assert torch.equal(out, feats)

    def test_no_truncation_when_at_cap(self):
        feats = self._make_feats(20)
        out = _truncate_to_max_frames(feats, max_clip_frames=40)
        assert out.shape[0] == 20 * TOKENS_PER_TUBELET
        assert torch.equal(out, feats)

    def test_odd_extra_drops_from_end(self):
        # 21 tubelets, cap 40 frames = 20 tubelets → drop 1 from end.
        feats = self._make_feats(21)
        out = _truncate_to_max_frames(feats, max_clip_frames=40)
        assert out.shape[0] == 20 * TOKENS_PER_TUBELET
        # Surviving tubelet ids are 0..19
        first = out[0, 0].item()
        last = out[-1, 0].item()
        assert first == 0 and last == 19

    def test_even_extra_drops_evenly(self):
        # 30 tubelets, cap 40 = 20 tubelets → drop 10, 5 from each side.
        feats = self._make_feats(30)
        out = _truncate_to_max_frames(feats, max_clip_frames=40)
        assert out.shape[0] == 20 * TOKENS_PER_TUBELET
        first = out[0, 0].item()
        last = out[-1, 0].item()
        assert first == 5 and last == 24

    def test_dataset_applies_cap(self, tmp_path):
        # End-to-end: save synthetic clips with varying tubelet counts and check
        # that VideoClipDataset honours max_clip_frames.
        feat_dir = tmp_path / "jepa_features" / "1"
        feat_dir.mkdir(parents=True)
        feats = self._make_feats(30, dim=4).to(torch.float16)
        torch.save(feats, feat_dir / "big.pt")

        # Single-clip trajectories.pt artifact.
        traj_path = tmp_path / f"trajectories_{DTW_DESIGN}.pt"
        torch.save(
            {
                "trajectories": torch.randn(1, TRAJ_T, TRAJ_D, dtype=torch.float32),
                "lengths": torch.tensor([TRAJ_T], dtype=torch.int32),
                "design": DTW_DESIGN,
                "clip_keys": [(1, "big")],
                "T_max": TRAJ_T,
                "D": TRAJ_D,
                "num_videos": 1,
            },
            traj_path,
        )

        ds = VideoClipDataset(
            str(tmp_path), max_clip_frames=40, trajectories_path=str(traj_path),
        )
        out = ds[0]["jepa_features"]
        assert out.shape[0] == 20 * TOKENS_PER_TUBELET
        # Disabling the cap returns the full tensor.
        ds_off = VideoClipDataset(
            str(tmp_path), max_clip_frames=-1, trajectories_path=str(traj_path),
        )
        assert ds_off[0]["jepa_features"].shape[0] == 30 * TOKENS_PER_TUBELET
