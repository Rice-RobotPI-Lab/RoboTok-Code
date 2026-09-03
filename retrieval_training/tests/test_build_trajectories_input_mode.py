"""Regression tests for input-mode-specific JEPA filtering."""

import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import torch


BUILD_DATASET_DIR = Path(__file__).resolve().parent.parent / "build_dataset"
sys.path.insert(0, str(BUILD_DATASET_DIR))

from build_trajectories import build_trajectories


def _install_fake_dtw_modules(monkeypatch, bundle):
    data_module = ModuleType("dtw_cknna_data")
    data_module.KEYPOINTS_CACHE = None
    data_module.load_data_bundle_21j = lambda *_args, **_kwargs: bundle
    monkeypatch.setitem(sys.modules, "dtw_cknna_data", data_module)

    design_module = ModuleType("dtw_cknna")
    design_module.DTW_DESIGNS = {"identity": {"tx": "identity"}}
    design_module.TRAJ_TRANSFORMS = {"identity": lambda trajs, lengths: (trajs, lengths)}
    monkeypatch.setitem(sys.modules, "dtw_cknna", design_module)


def _bundle():
    return SimpleNamespace(
        N=3,
        T=4,
        trajs=torch.arange(3 * 4 * 2, dtype=torch.float32).reshape(3, 4, 2),
        actual_lengths=torch.tensor([4, 3, 2]),
        clips=[
            {"video_number": 1, "node_uid": "a"},
            {"video_number": 1, "node_uid": "b"},
            {"video_number": 2, "node_uid": "c"},
        ],
    )


def test_trajectory_mode_retains_clips_without_jepa_features(tmp_path, monkeypatch):
    _install_fake_dtw_modules(monkeypatch, _bundle())
    output = tmp_path / "trajectories_identity.pt"

    build_trajectories(
        data_dir=tmp_path,
        dtw_design="identity",
        output_path=output,
        require_jepa_features=False,
    )

    payload = torch.load(output, weights_only=True)
    assert payload["trajectories"].shape == (3, 4, 2)
    assert payload["clip_keys"] == [(1, "a"), (1, "b"), (2, "c")]


def test_trajectory_mode_subsamples_without_jepa_features(tmp_path, monkeypatch):
    _install_fake_dtw_modules(monkeypatch, _bundle())
    output = tmp_path / "trajectories_identity.pt"

    build_trajectories(
        data_dir=tmp_path,
        dtw_design="identity",
        output_path=output,
        max_clips=2,
        clip_subsample_seed=42,
        require_jepa_features=False,
    )

    payload = torch.load(output, weights_only=True)
    assert payload["trajectories"].shape[0] == 2
    assert len(payload["clip_keys"]) == 2


def test_jepa_mode_retains_only_clips_with_features(tmp_path, monkeypatch):
    _install_fake_dtw_modules(monkeypatch, _bundle())
    feature_dir = tmp_path / "jepa_features" / "1"
    feature_dir.mkdir(parents=True)
    (feature_dir / "b.pt").touch()
    output = tmp_path / "trajectories_identity.pt"

    build_trajectories(
        data_dir=tmp_path,
        dtw_design="identity",
        output_path=output,
        require_jepa_features=True,
    )

    payload = torch.load(output, weights_only=True)
    assert payload["trajectories"].shape[0] == 1
    assert payload["clip_keys"] == [(1, "b")]


def test_jepa_mode_fails_early_when_no_features_exist(tmp_path, monkeypatch):
    import pytest

    _install_fake_dtw_modules(monkeypatch, _bundle())

    with pytest.raises(RuntimeError, match="No eligible clips with jepa_features"):
        build_trajectories(
            data_dir=tmp_path,
            dtw_design="identity",
            output_path=tmp_path / "trajectories_identity.pt",
            require_jepa_features=True,
        )
