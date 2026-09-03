import importlib
import math
import sys
from pathlib import Path

import pytest
import torch


TRAINING_DIR = Path(__file__).resolve().parents[1]  # retrieval_training/
BUILD_DATASET_DIR = TRAINING_DIR / "build_dataset"

for path in (TRAINING_DIR, BUILD_DATASET_DIR):
    path_s = str(path)
    if path_s not in sys.path:
        sys.path.insert(0, path_s)


def test_body_frame_design_registry_has_seven_designs():
    import dtw_cknna

    assert dtw_cknna.BODY_FRAME_21J_DESIGNS == [
        "abs_21j_coords",
        "abs_wrist_coords",
        "angles_21j",
        "wrist_rel_21j_coords",
        "wrist_rel_path",
        "full_interjoint_dists",
        "pca_interjoint_dists",
    ]
    for design in dtw_cknna.BODY_FRAME_21J_DESIGNS:
        tx = dtw_cknna.DTW_DESIGNS[design]["tx"]
        assert tx in dtw_cknna.TRAJ_TRANSFORMS
    assert "body_articulation_hand_norm_21j" not in dtw_cknna.BODY_FRAME_21J_DESIGNS


def test_body_frame_transforms_shapes_and_lengths():
    import dtw_cknna

    trajs = torch.arange(2 * 5 * 126, dtype=torch.float32).reshape(2, 5, 126)
    lengths = torch.tensor([5, 3], dtype=torch.long)

    expected = {
        "abs_21j_coords": ((2, 5, 126), [5, 3]),
        "abs_wrist_coords": ((2, 5, 6), [5, 3]),
        "body_wrist_velocity_3d": ((2, 4, 6), [4, 2]),
        "body_full_pose_21j_velocity": ((2, 4, 126), [4, 2]),
        "angles_21j": ((2, 5, 50), [5, 3]),
        "wrist_rel_21j_coords": ((2, 5, 126), [5, 3]),
        "full_interjoint_dists": ((2, 5, 861), [5, 3]),
        "wrist_rel_path": ((2, 5, 6), [5, 3]),
        "pca_interjoint_dists": ((2, 5, 160), [5, 3]),
        "pca_interjoint_dists_w_wrist_rel_path": ((2, 5, 166), [5, 3]),
    }

    for design, (shape, lens) in expected.items():
        tx_key = dtw_cknna.DTW_DESIGNS[design]["tx"]
        out, out_lens = dtw_cknna.TRAJ_TRANSFORMS[tx_key](trajs, lengths)
        assert tuple(out.shape) == shape
        assert out_lens.tolist() == lens


def test_body_articulation_uses_rotation_invariant_joint_angles():
    import dtw_cknna

    trajs = torch.randn(2, 5, 126)
    lengths = torch.tensor([5, 4], dtype=torch.long)
    theta = math.pi / 3.0
    R = torch.tensor(
        [
            [math.cos(theta), -math.sin(theta), 0.0],
            [math.sin(theta), math.cos(theta), 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=trajs.dtype,
    )
    transformed = trajs.reshape(2, 5, 2, 21, 3)
    transformed = (transformed @ R.T) * 3.0
    transformed = transformed + torch.tensor([10.0, -4.0, 2.0], dtype=trajs.dtype)
    transformed = transformed.reshape_as(trajs)

    out, _ = dtw_cknna.TRAJ_TRANSFORMS["angles_21j"](trajs, lengths)
    out_tx, _ = dtw_cknna.TRAJ_TRANSFORMS["angles_21j"](transformed, lengths)

    assert torch.allclose(out, out_tx, atol=1e-5)


def test_body_wrist_relative_pose_is_translation_invariant_not_rotation_invariant():
    import dtw_cknna

    trajs = torch.randn(2, 5, 126)
    lengths = torch.tensor([5, 4], dtype=torch.long)
    translated = trajs.reshape(2, 5, 2, 21, 3)
    translated = translated + torch.tensor([10.0, -4.0, 2.0], dtype=trajs.dtype)
    translated = translated.reshape_as(trajs)

    theta = math.pi / 3.0
    R = torch.tensor(
        [
            [math.cos(theta), -math.sin(theta), 0.0],
            [math.sin(theta), math.cos(theta), 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=trajs.dtype,
    )
    rotated = (trajs.reshape(2, 5, 2, 21, 3) @ R.T).reshape_as(trajs)

    tx = dtw_cknna.TRAJ_TRANSFORMS["wrist_rel_21j_coords"]
    out, _ = tx(trajs, lengths)
    out_translated, _ = tx(translated, lengths)
    out_rotated, _ = tx(rotated, lengths)

    assert torch.allclose(out, out_translated, atol=1e-5)
    assert not torch.allclose(out, out_rotated, atol=1e-5)


def test_pca_interjoint_with_wrist_path_is_start_pose_invariant():
    import dtw_cknna

    trajs = torch.randn(2, 5, 126)
    x = trajs.reshape(2, 5, 2, 21, 3)
    x[:, 0, 0, 0, :] = torch.tensor([-1.0, 0.0, 0.0])
    x[:, 0, 1, 0, :] = torch.tensor([1.0, 0.0, 0.0])
    trajs = x.reshape_as(trajs)
    lengths = torch.tensor([5, 4], dtype=torch.long)

    theta = math.pi / 5.0
    R = torch.tensor(
        [
            [math.cos(theta), -math.sin(theta), 0.0],
            [math.sin(theta), math.cos(theta), 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=trajs.dtype,
    )
    transformed = trajs.reshape(2, 5, 42, 3)
    transformed = transformed @ R.T
    transformed = transformed + torch.tensor([4.0, -3.0, 2.0], dtype=trajs.dtype)
    transformed = transformed.reshape_as(trajs)

    tx = dtw_cknna.TRAJ_TRANSFORMS["pca_interjoint_dists_w_wrist_rel_path"]
    out, _ = tx(trajs, lengths)
    out_tx, _ = tx(transformed, lengths)

    assert torch.allclose(out, out_tx, atol=1e-5)


def test_interjoint_dists_are_translation_and_rotation_invariant():
    import dtw_cknna

    trajs = torch.randn(2, 5, 126)
    lengths = torch.tensor([5, 4], dtype=torch.long)
    theta = math.pi / 4.0
    R = torch.tensor(
        [
            [math.cos(theta), -math.sin(theta), 0.0],
            [math.sin(theta), math.cos(theta), 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=trajs.dtype,
    )
    transformed = trajs.reshape(2, 5, 42, 3)
    transformed = transformed @ R.T
    transformed = transformed + torch.tensor([3.0, -2.0, 5.0], dtype=trajs.dtype)
    transformed = transformed.reshape_as(trajs)

    tx = dtw_cknna.TRAJ_TRANSFORMS["full_interjoint_dists"]
    out, _ = tx(trajs, lengths)
    out_tx, _ = tx(transformed, lengths)

    assert torch.allclose(out, out_tx, atol=1e-5)


def test_wrist_rel_path_matches_pca_interjoint_path_channels():
    import dtw_cknna

    trajs = torch.randn(2, 5, 126)
    x = trajs.reshape(2, 5, 2, 21, 3)
    x[:, 0, 0, 0, :] = torch.tensor([-1.0, 0.0, 0.0])
    x[:, 0, 1, 0, :] = torch.tensor([1.0, 0.0, 0.0])
    trajs = x.reshape_as(trajs)
    lengths = torch.tensor([5, 4], dtype=torch.long)

    path, path_lens = dtw_cknna.TRAJ_TRANSFORMS["wrist_rel_path"](trajs, lengths)
    combined, combined_lens = dtw_cknna.TRAJ_TRANSFORMS[
        "pca_interjoint_dists_w_wrist_rel_path"
    ](trajs, lengths)

    assert torch.allclose(path, combined[:, :, -6:], atol=1e-6)
    assert path_lens.tolist() == combined_lens.tolist()


def test_training_import_paths_point_to_vendored_dtw_cknna():
    import online_dtw
    import build_trajectories

    assert (Path(online_dtw._CKNNA_DIR) / "dtw_cknna.py").is_file()
    assert (Path(build_trajectories.CKNNA_DIR) / "dtw_cknna.py").is_file()


def test_online_dtw_accepts_body_frame_designs_without_cuda_call():
    # OnlineDTWComputer builds its numba CUDA kernel eagerly in __init__.
    pytest.importorskip("numba", reason="DTW kernels require numba")
    import dtw_cknna

    online_dtw = importlib.import_module("online_dtw")
    for design in dtw_cknna.BODY_FRAME_21J_DESIGNS:
        computer = online_dtw.OnlineDTWComputer(design)
        assert computer.design == design
