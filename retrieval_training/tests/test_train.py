"""Tests for retrieval_training/train.py helper functions.

Does not test main() (requires full dataset + GPU). Tests the pure functions
that can be exercised with synthetic data on CPU.

Run from retrieval_training/: python -m pytest tests/test_train.py -v
"""

import sys
import random
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import Config
from data import VideoClipDataset, collate_fn
from model import TrajectoryRetrievalModel
import train as train_module
from train import (
    _batch_recall_top1,
    _off_diag_stats,
    _preserve_rng_state,
    _run_and_log_evaluation,
    build_optimizer_param_groups,
    initialize_query_token,
    random_holdout_split,
    save_checkpoint,
)

D = 64
R = 16
HEADS = 4


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

VIDEOS = {1: ["uid_a", "uid_b"], 2: ["uid_c"]}
TOKEN_COUNTS = {"uid_a": 8, "uid_b": 12, "uid_c": 6}
DTW_DESIGN = "first_frame_midpoint_hand_length_no_z"
TRAJ_T = 6
TRAJ_D = 4


@pytest.fixture
def data_dir(tmp_path):
    keys = []
    for vn, uids in sorted(VIDEOS.items()):
        for uid in uids:
            keys.append((vn, uid))

    feat_dir = tmp_path / "jepa_features"
    feat_dir.mkdir()
    for vn, uids in VIDEOS.items():
        vdir = feat_dir / str(vn)
        vdir.mkdir()
        for uid in uids:
            t = TOKEN_COUNTS[uid]
            torch.save(torch.randn(t, D, dtype=torch.float16), vdir / f"{uid}.pt")

    n = len(keys)
    torch.save(
        {
            "trajectories": torch.randn(n, TRAJ_T, TRAJ_D, dtype=torch.float32),
            "lengths": torch.full((n,), TRAJ_T, dtype=torch.int32),
            "design": DTW_DESIGN,
            "clip_keys": keys,
            "T_max": TRAJ_T,
            "D": TRAJ_D,
            "num_videos": len(VIDEOS),
        },
        tmp_path / f"trajectories_{DTW_DESIGN}.pt",
    )
    return tmp_path


@pytest.fixture
def dataset(data_dir):
    return VideoClipDataset(
        str(data_dir),
        trajectories_path=str(data_dir / f"trajectories_{DTW_DESIGN}.pt"),
    )


@pytest.fixture
def dataloader(dataset):
    return DataLoader(dataset, batch_size=2, shuffle=False, collate_fn=collate_fn)


@pytest.fixture
def model():
    return TrajectoryRetrievalModel(
        encoder_dim=D, retrieval_dim=R, num_heads=HEADS, num_projection_layers=2
    )


# ---------------------------------------------------------------------------
# initialize_query_token
# ---------------------------------------------------------------------------

class TestInitializeQueryToken:
    def test_query_changes(self, model, dataloader):
        original = model.head.query.clone()
        initialize_query_token(model, dataloader, torch.device("cpu"), num_batches=2)
        assert not torch.allclose(original, model.head.query)

    def test_query_shape_preserved(self, model, dataloader):
        initialize_query_token(model, dataloader, torch.device("cpu"), num_batches=1)
        assert model.head.query.shape == (1, D)

    def test_zero_batches_no_change(self, model, dataloader):
        original = model.head.query.clone()
        initialize_query_token(model, dataloader, torch.device("cpu"), num_batches=0)
        assert torch.allclose(original, model.head.query)

    def test_model_stays_eval_after(self, model, dataloader):
        model.train()
        initialize_query_token(model, dataloader, torch.device("cpu"), num_batches=1)
        assert not model.training


# ---------------------------------------------------------------------------
# save_checkpoint
# ---------------------------------------------------------------------------

def _make_train_artifacts(model):
    """Build optimizer / scheduler / scaler / cfg for save_checkpoint tests."""
    optimizer = torch.optim.AdamW(model.get_trainable_parameters(), lr=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10)
    scaler = torch.amp.GradScaler("cuda", enabled=False)
    cfg = Config()
    return optimizer, scheduler, scaler, cfg


class TestSaveCheckpoint:
    def test_creates_file(self, model, tmp_path):
        optimizer, scheduler, scaler, cfg = _make_train_artifacts(model)
        path = tmp_path / "ckpt.pt"
        save_checkpoint(
            model, optimizer, scheduler, scaler, cfg,
            epoch=3, global_step=30, best_metric=0.5, path=path,
        )
        assert path.exists()

    def test_checkpoint_contents(self, model, tmp_path):
        optimizer, scheduler, scaler, cfg = _make_train_artifacts(model)
        path = tmp_path / "ckpt.pt"
        save_checkpoint(
            model, optimizer, scheduler, scaler, cfg,
            epoch=5, global_step=50, best_metric=0.75, path=path,
        )
        ckpt = torch.load(path, weights_only=False, map_location="cpu")
        assert "head_state_dict" in ckpt
        assert "optimizer_state_dict" in ckpt
        assert "scheduler_state_dict" in ckpt
        assert "scaler_state_dict" in ckpt
        assert "config" in ckpt
        assert ckpt["epoch"] == 5
        assert ckpt["global_step"] == 50
        assert ckpt["best_metric"] == 0.75

    def test_state_dict_loadable(self, model, tmp_path):
        optimizer, scheduler, scaler, cfg = _make_train_artifacts(model)
        path = tmp_path / "ckpt.pt"
        save_checkpoint(
            model, optimizer, scheduler, scaler, cfg,
            epoch=1, global_step=1, best_metric=0.0, path=path,
        )
        ckpt = torch.load(path, weights_only=False, map_location="cpu")
        new_model = TrajectoryRetrievalModel(
            encoder_dim=D, retrieval_dim=R, num_heads=HEADS, num_projection_layers=2
        )
        new_model.head.load_state_dict(ckpt["head_state_dict"])
        for p1, p2 in zip(model.head.parameters(), new_model.head.parameters()):
            assert torch.allclose(p1, p2)

    def test_overwrite_existing(self, model, tmp_path):
        optimizer, scheduler, scaler, cfg = _make_train_artifacts(model)
        path = tmp_path / "ckpt.pt"
        save_checkpoint(
            model, optimizer, scheduler, scaler, cfg,
            epoch=1, global_step=1, best_metric=0.1, path=path,
        )
        save_checkpoint(
            model, optimizer, scheduler, scaler, cfg,
            epoch=2, global_step=2, best_metric=0.2, path=path,
        )
        ckpt = torch.load(path, weights_only=False, map_location="cpu")
        assert ckpt["epoch"] == 2
        assert ckpt["best_metric"] == 0.2


# ---------------------------------------------------------------------------
# evaluation logging / epoch-0 support
# ---------------------------------------------------------------------------

class _WriterSpy:
    def __init__(self):
        self.scalars = []

    def add_scalar(self, name, value, step):
        self.scalars.append((name, value, step))


def _call_eval_helper(monkeypatch, metrics, trajectory_mode, epoch=0):
    monkeypatch.setattr(train_module, "run_evaluation", lambda *args, **kwargs: metrics)
    writer = _WriterSpy()
    cfg = SimpleNamespace(cknna_topk=10)
    result = _run_and_log_evaluation(
        model=object(), eval_dataloader=object(), eval_dtw_matrix=object(),
        eval_jepa_sim=None, cfg=cfg,
        device=torch.device("cpu"), criterion=object(), dtw_computer=object(),
        writer=writer, epoch=epoch, trajectory_mode=trajectory_mode,
    )
    return result, writer


def test_eval_helper_logs_trajectory_metrics_at_epoch_zero(monkeypatch):
    metrics = {
        "R@5": 0.25,
        "cknna_dtw@10": 0.1,
        "cknna_dtw_global_pool@10": 0.2,
        "mutual_knn_dtw_global_pool@10": 0.3,
    }
    (returned, selection, desc), writer = _call_eval_helper(
        monkeypatch, metrics, trajectory_mode=True,
    )
    assert returned is metrics
    assert selection == pytest.approx(0.1)
    assert desc == "CKNNA DTW=0.1000"
    assert ("eval/R@5", 0.25, 0) in writer.scalars
    assert ("eval/cknna_dtw@10", 0.1, 0) in writer.scalars
    assert ("eval/selection_cknna_dtw", 0.1, 0) in writer.scalars
    assert ("eval/cknna_dtw_global_pool@10", 0.2, 0) in writer.scalars
    assert ("eval/mutual_knn_dtw_global_pool@10", 0.3, 0) in writer.scalars


def test_eval_helper_logs_balanced_jepa_selection(monkeypatch):
    metrics = {"R@5": 0.6, "jepa_R@5": 0.3}
    (_, selection, _), writer = _call_eval_helper(
        monkeypatch, metrics, trajectory_mode=False, epoch=5,
    )
    assert selection == pytest.approx(0.4)
    assert ("eval/balanced_R@5", pytest.approx(0.4), 5) in writer.scalars


def test_preserve_rng_state_restores_python_torch_and_numpy():
    random.seed(123)
    torch.manual_seed(123)
    np.random.seed(123)
    python_state = random.getstate()
    torch_state = torch.random.get_rng_state().clone()
    numpy_state = np.random.get_state()

    with _preserve_rng_state(torch.device("cpu")):
        random.random()
        torch.rand(10)
        np.random.rand(10)

    assert random.getstate() == python_state
    assert torch.equal(torch.random.get_rng_state(), torch_state)
    restored_numpy = np.random.get_state()
    assert restored_numpy[0] == numpy_state[0]
    assert np.array_equal(restored_numpy[1], numpy_state[1])
    assert restored_numpy[2:] == numpy_state[2:]


# ---------------------------------------------------------------------------
# random_holdout_split
# ---------------------------------------------------------------------------

class TestRandomHoldoutSplit:
    def test_partition_is_disjoint_and_complete(self):
        train, eval_ = random_holdout_split(100, holdout_fraction=0.1, seed=0)
        # No overlap, full coverage, sorted output.
        assert set(train).isdisjoint(eval_)
        assert sorted(train + eval_) == list(range(100))
        assert train == sorted(train)
        assert eval_ == sorted(eval_)

    def test_eval_size_matches_fraction(self):
        train, eval_ = random_holdout_split(100, holdout_fraction=0.1, seed=0)
        assert len(eval_) == 10
        assert len(train) == 90

    def test_eval_size_rounds(self):
        # 0.13 * 100 = 13; 0.155 * 100 = 15.5 → 16 (banker's rounding handled by round()).
        _, eval_ = random_holdout_split(100, holdout_fraction=0.13, seed=0)
        assert len(eval_) == 13

    def test_stable_across_calls_same_seed(self):
        a_train, a_eval = random_holdout_split(50, holdout_fraction=0.2, seed=42)
        b_train, b_eval = random_holdout_split(50, holdout_fraction=0.2, seed=42)
        assert a_train == b_train
        assert a_eval == b_eval

    def test_different_seeds_diverge(self):
        _, eval_a = random_holdout_split(200, holdout_fraction=0.1, seed=1)
        _, eval_b = random_holdout_split(200, holdout_fraction=0.1, seed=2)
        assert eval_a != eval_b

    def test_invalid_fraction_raises(self):
        with pytest.raises(ValueError, match="eval_holdout_fraction"):
            random_holdout_split(10, holdout_fraction=0.0, seed=0)
        with pytest.raises(ValueError, match="eval_holdout_fraction"):
            random_holdout_split(10, holdout_fraction=1.0, seed=0)
        with pytest.raises(ValueError, match="eval_holdout_fraction"):
            random_holdout_split(10, holdout_fraction=-0.1, seed=0)


# ---------------------------------------------------------------------------
# _off_diag_stats / _batch_recall_top1
# ---------------------------------------------------------------------------

class TestOffDiagStats:
    def test_excludes_diagonal(self):
        # Construct a matrix where diagonal is huge — if we accidentally
        # included it, the mean would be dominated by 100, not 0.
        mat = torch.zeros(5, 5)
        mat.fill_diagonal_(100.0)
        mean, std = _off_diag_stats(mat)
        assert abs(mean) < 1e-6
        assert abs(std) < 1e-6

    def test_known_values(self):
        # Off-diag entries: 1, 2, 3, 4, 5, 6 (upper) + symmetric duplicates.
        mat = torch.tensor([
            [99.0, 1.0, 2.0, 3.0],
            [1.0, 99.0, 4.0, 5.0],
            [2.0, 4.0, 99.0, 6.0],
            [3.0, 5.0, 6.0, 99.0],
        ])
        mean, _ = _off_diag_stats(mat)
        # 12 off-diag entries (each upper-triangle entry appears twice).
        # Sum = 2*(1+2+3+4+5+6) = 42, mean = 42/12 = 3.5.
        assert abs(mean - 3.5) < 1e-5


class TestBatchRecallTop1:
    def test_perfect_match(self):
        # If pred == target, the model's top-1 is always the target's top-1.
        torch.manual_seed(0)
        n = 8
        target = torch.randn(n, n)
        target = target + target.T
        recall = _batch_recall_top1(target, target, k=1)
        assert recall == 1.0

    def test_self_excluded(self):
        # Diagonal is the largest entry on each row; if we don't mask self,
        # the "top-1" would always be self and recall would be wrong.
        n = 5
        pred = torch.eye(n) * 100.0  # diagonal = 100, off-diag = 0
        target = pred.clone()
        # With self masked out, both rank everything-else identically (all 0),
        # so any top-1 is valid. Recall@1 must still be defined and finite.
        recall = _batch_recall_top1(pred, target, k=1)
        assert 0.0 <= recall <= 1.0

    def test_k_clamped_to_n_minus_1(self):
        n = 4
        pred = torch.randn(n, n)
        target = torch.randn(n, n)
        # k > n-1 must not error and should give recall = 1.0 (any neighbor
        # is in pred's top-(n-1) = all-non-self).
        recall = _batch_recall_top1(pred, target, k=100)
        assert recall == 1.0

    def test_recall_in_range(self):
        n = 16
        pred = torch.randn(n, n); pred = pred + pred.T
        target = torch.randn(n, n); target = target + target.T
        for k in [1, 3, 5]:
            r = _batch_recall_top1(pred, target, k=k)
            assert 0.0 <= r <= 1.0


class TestBuildOptimizerParamGroups:
    """`build_optimizer_param_groups`: single group on the cross-attn head,
    two groups (head + prototypes) on the DTW-NN head with the configured LR
    multiplier."""

    def test_single_group_on_cross_attn_head(self):
        model = TrajectoryRetrievalModel(encoder_dim=D, retrieval_dim=R, num_heads=HEADS)
        groups = build_optimizer_param_groups(
            model, lr=1e-4, prototype_lr_multiplier=10.0,
        )
        assert len(groups) == 1
        assert groups[0]["lr"] == 1e-4
        # All head params land in the one group.
        head_params = list(model.head.parameters())
        assert len(groups[0]["params"]) == len(head_params)

    def test_two_groups_on_dtw_nn_head(self):
        model = TrajectoryRetrievalModel(
            encoder_dim=D, retrieval_dim=R, num_heads=HEADS,
            dtw_nn_layer=True, dtw_nn_num_nodes=16, dtw_nn_prototype_length=12,
        )
        groups = build_optimizer_param_groups(
            model, lr=1e-4, prototype_lr_multiplier=10.0,
        )
        assert len(groups) == 2
        assert groups[0]["lr"] == 1e-4
        assert groups[1]["lr"] == 1e-3
        # The prototype tensor is the DTW layer's `prototypes`.
        assert len(groups[1]["params"]) == 1
        assert groups[1]["params"][0].data_ptr() == model.head.dtw.prototypes.data_ptr()
        # Disjoint union covers every head param exactly once.
        head_ids = {id(p) for p in model.head.parameters()}
        group_ids = {id(p) for g in groups for p in g["params"]}
        assert head_ids == group_ids

    def test_lr_multiplier_respected(self):
        model = TrajectoryRetrievalModel(
            encoder_dim=D, retrieval_dim=R, num_heads=HEADS,
            dtw_nn_layer=True, dtw_nn_num_nodes=16, dtw_nn_prototype_length=12,
        )
        groups = build_optimizer_param_groups(
            model, lr=2e-4, prototype_lr_multiplier=5.0,
        )
        assert groups[0]["lr"] == 2e-4
        assert groups[1]["lr"] == 1e-3  # 2e-4 * 5
