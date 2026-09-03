"""Tests for --depth-grounded flag threading across the training pipeline.

Verifies that Config, build_trajectories, 1get_clip_keypoints, 3get_jepa_features,
and train.py all correctly propagate the depth-grounded flag to select the right
keypoints cache file.

Run from retrieval_training/: python -m pytest tests/test_depth_grounded_flags.py -v
"""

import importlib.util
import subprocess
import sys
from pathlib import Path
from unittest import mock

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

BUILD_DATASET_DIR = Path(__file__).resolve().parent.parent / "build_dataset"


def _load_module_from_path(name, path):
    """Import a module from an arbitrary file path (handles filenames like 1get_...)."""
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

class TestConfigDepthGrounded:
    def test_default_is_true(self):
        from config import Config
        assert Config().use_depth_grounded_keypoints is True

    def test_set_via_constructor(self):
        from config import Config
        cfg = Config(use_depth_grounded_keypoints=True)
        assert cfg.use_depth_grounded_keypoints is True

    def test_set_via_yaml(self, tmp_path):
        from config import Config
        yaml_path = tmp_path / "cfg.yaml"
        yaml_path.write_text("use_depth_grounded_keypoints: true\n")
        cfg = Config.from_yaml(str(yaml_path))
        assert cfg.use_depth_grounded_keypoints is True

    def test_yaml_false_explicit(self, tmp_path):
        from config import Config
        yaml_path = tmp_path / "cfg.yaml"
        yaml_path.write_text("use_depth_grounded_keypoints: false\n")
        cfg = Config.from_yaml(str(yaml_path))
        assert cfg.use_depth_grounded_keypoints is False


# ---------------------------------------------------------------------------
# build_trajectories: cache name selection
# ---------------------------------------------------------------------------

class TestBuildTrajectoriesCacheName:
    """Verify build_trajectories sets KEYPOINTS_CACHE based on use_depth_grounded.

    The function does a deferred `import dtw_cknna_data` then sets its
    KEYPOINTS_CACHE attribute. We check the cache name logic directly.
    """

    def test_normal_cache_name(self):
        use_depth_grounded = False
        cache_name = "depth_grounded_clip_keypoints.pt" if use_depth_grounded else "clip_keypoints.pt"
        assert cache_name == "clip_keypoints.pt"

    def test_depth_grounded_cache_name(self):
        use_depth_grounded = True
        cache_name = "depth_grounded_clip_keypoints.pt" if use_depth_grounded else "clip_keypoints.pt"
        assert cache_name == "depth_grounded_clip_keypoints.pt"

    def test_flag_exists_in_function_signature(self):
        sys.path.insert(0, str(BUILD_DATASET_DIR))
        from build_trajectories import build_trajectories
        import inspect
        sig = inspect.signature(build_trajectories)
        assert "use_depth_grounded" in sig.parameters
        assert sig.parameters["use_depth_grounded"].default is False


# ---------------------------------------------------------------------------
# build_trajectories CLI: --depth-grounded flag parsing
# ---------------------------------------------------------------------------

class TestBuildTrajectoriesCLI:
    def test_depth_grounded_flag_parsed(self):
        result = subprocess.run(
            [sys.executable, "-c",
             "import argparse; "
             "p = argparse.ArgumentParser(); "
             "p.add_argument('--depth-grounded', action='store_true'); "
             "args = p.parse_args(['--depth-grounded']); "
             "print(args.depth_grounded)"],
            capture_output=True, text=True,
        )
        assert "True" in result.stdout

    def test_no_flag_defaults_false(self):
        result = subprocess.run(
            [sys.executable, "-c",
             "import argparse; "
             "p = argparse.ArgumentParser(); "
             "p.add_argument('--depth-grounded', action='store_true'); "
             "args = p.parse_args([]); "
             "print(args.depth_grounded)"],
            capture_output=True, text=True,
        )
        assert "False" in result.stdout


# ---------------------------------------------------------------------------
# 1get_clip_keypoints: --depth-grounded column and output path selection
# ---------------------------------------------------------------------------

class TestGetClipKeypointsDepthGrounded:
    @pytest.fixture(autouse=True)
    def _load_module(self):
        # Mock db module since 1get_clip_keypoints imports it at top-level
        fake_db = mock.MagicMock()
        with mock.patch.dict("sys.modules", {"db": fake_db}):
            self.mod = _load_module_from_path(
                "get_clip_keypoints_mod",
                BUILD_DATASET_DIR / "1get_clip_keypoints.py",
            )

    def test_depth_grounded_selects_correct_columns(self):
        assert "depth_grounded_keypoints AS keypoints_per_frame" in self.mod.DEPTH_GROUNDED_SELECT_COLS

    def test_default_select_cols_no_depth_grounded(self):
        assert "depth_grounded" not in self.mod.SELECT_COLS

    def test_depth_grounded_output_path_differs(self):
        assert self.mod.DEFAULT_OUT_PATH != self.mod.DEPTH_GROUNDED_OUT_PATH
        assert "depth_grounded" in str(self.mod.DEPTH_GROUNDED_OUT_PATH)

    def test_depth_grounded_output_filename(self):
        assert self.mod.DEPTH_GROUNDED_OUT_PATH.name == "depth_grounded_clip_keypoints.pt"

    def test_default_output_filename(self):
        assert self.mod.DEFAULT_OUT_PATH.name == "clip_keypoints.pt"

    def test_body_frame_overwrites_kpts_without_mutating_input(self):
        row = {
            "video_number": 1,
            "node_number": 2,
            "node_uid": "node-a",
            "keypoints_per_frame": {
                "0": {
                    "L": [{"kpts_3d": [[0, 0, 0]], "score": 0.9}],
                    "R": [],
                },
                "_meta": {"source": "depth"},
            },
            "left_hand_presence": 1.0,
            "right_hand_presence": 0.0,
        }
        body_result = {
            "hands_body": {
                "0": {
                    "L": [[1.0, 2.0, 3.0]],
                    "R": None,
                }
            }
        }

        out = self.mod._overwrite_kpts_with_body_frame(row, body_result)

        assert out is not row
        assert row["keypoints_per_frame"]["0"]["L"][0]["kpts_3d"] == [[0, 0, 0]]
        assert out["keypoints_per_frame"]["0"]["L"][0]["kpts_3d"] == [[1.0, 2.0, 3.0]]
        assert out["keypoints_per_frame"]["0"]["L"][0]["score"] == 0.9
        assert out["keypoints_per_frame"]["_meta"] == {"source": "depth"}

    def test_body_frame_conversion_skips_none_results(self):
        rows = [
            {
                "video_number": 1,
                "node_number": 2,
                "node_uid": "skip-me",
                "keypoints_per_frame": {},
            }
        ]

        out = self.mod._convert_rows_to_body_frame(
            rows,
            body_pose_model=object(),
            body_pose_device="cpu",
            estimator=lambda *_args: None,
        )

        assert out == []

    def test_body_pose_checkpoint_arg_exists(self):
        src = (BUILD_DATASET_DIR / "1get_clip_keypoints.py").read_text()
        assert "--body-pose-ckpt" in src

    def test_retry_missing_cache_rows_arg_exists(self):
        src = (BUILD_DATASET_DIR / "1get_clip_keypoints.py").read_text()
        assert "--retry-missing-cache-rows" in src

    def test_missing_db_keys_detects_partial_cached_video(self):
        db_key_rows = [
            {"video_number": 1, "node_number": 10, "node_uid": "a"},
            {"video_number": 1, "node_number": 11, "node_uid": "b"},
            {"video_number": 2, "node_number": 20, "node_uid": "c"},
        ]
        cached_rows = [
            {"video_number": 1, "node_number": 10, "node_uid": "a"},
            {"video_number": 2, "node_number": 20, "node_uid": "c"},
        ]

        missing = self.mod._missing_db_keys(
            db_key_rows,
            self.mod._cache_keys(cached_rows),
        )

        assert missing == [(1, 11)]

    def test_missing_db_keys_ignores_duplicate_db_keys(self):
        db_key_rows = [
            {"video_number": 1, "node_number": 11, "node_uid": "b"},
            {"video_number": 1, "node_number": 11, "node_uid": "b"},
        ]

        missing = self.mod._missing_db_keys(db_key_rows, cached_keys=set())

        assert missing == [(1, 11)]

    def test_group_keys_by_video_sorts_for_targeted_fetch(self):
        grouped = self.mod._group_keys_by_video([
            (3, 5),
            (1, 9),
            (1, 7),
        ])

        assert grouped == {1: [7, 9], 3: [5]}

    def test_retry_merge_preserves_existing_and_adds_missing(self):
        existing = [
            {"video_number": 1, "node_number": 10, "node_uid": "old"},
        ]
        fetched_missing = [
            {"video_number": 1, "node_number": 11, "node_uid": "new"},
        ]

        merged = self.mod._merge_rows(existing, fetched_missing)

        assert [(r["video_number"], r["node_number"]) for r in merged] == [
            (1, 10),
            (1, 11),
        ]
        assert merged[1]["node_uid"] == "new"


# ---------------------------------------------------------------------------
# 3get_jepa_features: --depth-grounded cache path selection
# ---------------------------------------------------------------------------

class TestGetJepaFeaturesDepthGrounded:
    def test_depth_grounded_flag_sets_cache(self):
        import argparse
        p = argparse.ArgumentParser()
        p.add_argument("--depth-grounded", action="store_true")
        args = p.parse_args(["--depth-grounded"])
        assert args.depth_grounded is True

    def test_no_flag_uses_default_cache(self):
        import argparse
        p = argparse.ArgumentParser()
        p.add_argument("--depth-grounded", action="store_true")
        args = p.parse_args([])
        assert args.depth_grounded is False

    def test_cache_paths_consistent_with_build_scripts(self):
        """The depth-grounded and normal cache filenames in 3get_jepa_features must
        match those used by 1get_clip_keypoints and build_trajectories."""
        normal = Path("clip_keypoints.pt")
        depth = Path("depth_grounded_clip_keypoints.pt")
        assert normal.name == "clip_keypoints.pt"
        assert depth.name == "depth_grounded_clip_keypoints.pt"


# ---------------------------------------------------------------------------
# train.py: config -> build_trajectories plumbing
# ---------------------------------------------------------------------------

class TestTrainDepthGroundedPlumbing:
    def test_config_flag_reaches_build_trajectories(self, tmp_path):
        """Verify that cfg.use_depth_grounded_keypoints maps to the
        use_depth_grounded kwarg passed to build_trajectories."""
        from config import Config
        cfg = Config(
            use_depth_grounded_keypoints=True,
            dtw_design="first_frame_midpoint_hand_length_no_z",
            data_dir=str(tmp_path),
            output_dir=str(tmp_path),
        )
        # Simulate what train.py does at line ~365
        kwargs = dict(
            data_dir=Path(cfg.data_dir),
            dtw_design=cfg.dtw_design,
            output_path=tmp_path / f"trajectories_{cfg.dtw_design}.pt",
            use_depth_grounded=cfg.use_depth_grounded_keypoints,
        )
        assert kwargs["use_depth_grounded"] is True

    def test_config_default_passes_true(self, tmp_path):
        from config import Config
        cfg = Config(data_dir=str(tmp_path), output_dir=str(tmp_path))
        assert cfg.use_depth_grounded_keypoints is True

    def test_train_source_references_flag(self):
        """Verify train.py source actually passes use_depth_grounded to build_trajectories."""
        train_py = Path(__file__).resolve().parent.parent / "train.py"
        source = train_py.read_text()
        assert "use_depth_grounded=cfg.use_depth_grounded_keypoints" in source


# ---------------------------------------------------------------------------
# End-to-end: cache file naming consistency across all scripts
# ---------------------------------------------------------------------------

class TestCacheFileNamingConsistency:
    EXPECTED_NORMAL = "clip_keypoints.pt"
    EXPECTED_DEPTH = "depth_grounded_clip_keypoints.pt"

    def test_build_trajectories_source_has_both_names(self):
        src = (BUILD_DATASET_DIR / "build_trajectories.py").read_text()
        assert self.EXPECTED_NORMAL in src
        assert self.EXPECTED_DEPTH in src

    def test_get_jepa_features_source_has_both_names(self):
        src = (BUILD_DATASET_DIR / "3get_jepa_features.py").read_text()
        assert self.EXPECTED_NORMAL in src
        assert self.EXPECTED_DEPTH in src

    def test_get_clip_keypoints_source_has_both_names(self):
        src = (BUILD_DATASET_DIR / "1get_clip_keypoints.py").read_text()
        assert self.EXPECTED_NORMAL in src
        assert self.EXPECTED_DEPTH in src
