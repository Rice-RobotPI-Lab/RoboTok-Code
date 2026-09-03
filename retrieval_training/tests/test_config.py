"""Tests for retrieval_training/config.py.

Run from retrieval_training/: python -m pytest tests/test_config.py -v
"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import Config


# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------

class TestDefaults:
    def test_encoder_dim(self):
        assert Config().encoder_dim == 1408

    def test_retrieval_dim(self):
        assert Config().retrieval_dim == 256

    def test_batch_size(self):
        assert Config().batch_size == 196

    def test_num_epochs(self):
        assert Config().num_epochs == 50

    def test_lr(self):
        assert Config().lr == 1e-4

    def test_lambda_preserve(self):
        assert Config().lambda_preserve == 1.0

    def test_recall_k_values_default(self):
        assert Config().recall_k_values == [1, 5, 10, 20]

    def test_recall_k_values_independent_instances(self):
        c1 = Config()
        c2 = Config()
        c1.recall_k_values.append(99)
        assert 99 not in c2.recall_k_values

    def test_log_frequency(self):
        assert Config().log_frequency == 50

    def test_eval_frequency(self):
        assert Config().eval_frequency == 10

    def test_mixed_precision_default_on(self):
        assert Config().mixed_precision is True

    def test_grad_clip_norm(self):
        assert Config().grad_clip_norm == 1.0


# ---------------------------------------------------------------------------
# from_yaml
# ---------------------------------------------------------------------------

class TestFromYaml:
    def test_override_single_field(self, tmp_path):
        yaml_path = tmp_path / "cfg.yaml"
        yaml_path.write_text("batch_size: 64\n")
        cfg = Config.from_yaml(str(yaml_path))
        assert cfg.batch_size == 64

    def test_override_multiple_fields(self, tmp_path):
        yaml_path = tmp_path / "cfg.yaml"
        yaml_path.write_text("lr: 0.001\nnum_epochs: 10\n")
        cfg = Config.from_yaml(str(yaml_path))
        assert cfg.lr == 0.001
        assert cfg.num_epochs == 10

    def test_unknown_fields_ignored(self, tmp_path):
        yaml_path = tmp_path / "cfg.yaml"
        yaml_path.write_text("nonexistent_field: 42\nbatch_size: 16\n")
        cfg = Config.from_yaml(str(yaml_path))
        assert cfg.batch_size == 16
        assert not hasattr(cfg, "nonexistent_field")

    def test_empty_yaml(self, tmp_path):
        yaml_path = tmp_path / "cfg.yaml"
        yaml_path.write_text("")
        cfg = Config.from_yaml(str(yaml_path))
        assert cfg.batch_size == 196  # default

    def test_non_overridden_fields_keep_defaults(self, tmp_path):
        yaml_path = tmp_path / "cfg.yaml"
        yaml_path.write_text("batch_size: 8\n")
        cfg = Config.from_yaml(str(yaml_path))
        assert cfg.encoder_dim == 1408
        assert cfg.retrieval_dim == 256

    def test_missing_file_raises(self):
        with pytest.raises(FileNotFoundError):
            Config.from_yaml("/nonexistent/path.yaml")

    def test_override_list_field(self, tmp_path):
        yaml_path = tmp_path / "cfg.yaml"
        yaml_path.write_text("recall_k_values: [1, 3, 5]\n")
        cfg = Config.from_yaml(str(yaml_path))
        assert cfg.recall_k_values == [1, 3, 5]

    def test_override_bool_field(self, tmp_path):
        yaml_path = tmp_path / "cfg.yaml"
        yaml_path.write_text("mixed_precision: false\n")
        cfg = Config.from_yaml(str(yaml_path))
        assert cfg.mixed_precision is False

    def test_override_string_field(self, tmp_path):
        yaml_path = tmp_path / "cfg.yaml"
        yaml_path.write_text("output_dir: /tmp/test_out\n")
        cfg = Config.from_yaml(str(yaml_path))
        # from_yaml resolves paths; on macOS /tmp is a symlink to /private/tmp.
        assert cfg.output_dir == str(Path("/tmp/test_out").resolve())


class TestDTWNNDefaults:
    def test_default_off(self):
        assert Config().dtw_nn_layer is False

    def test_default_num_nodes(self):
        # Default matches retrieval_dim so the MLP doesn't expand/contract.
        cfg = Config()
        assert cfg.dtw_nn_num_nodes == cfg.retrieval_dim == 256

    def test_default_prototype_length(self):
        assert Config().dtw_nn_prototype_length == 42

    def test_default_gamma(self):
        assert Config().dtw_nn_gamma == 0.1

    def test_default_lr_multiplier(self):
        assert Config().dtw_nn_prototype_lr_multiplier == 10.0

    def test_yaml_override(self, tmp_path):
        yaml_path = tmp_path / "cfg.yaml"
        yaml_path.write_text(
            "dtw_nn_layer: true\n"
            "dtw_nn_num_nodes: 128\n"
            "dtw_nn_prototype_length: 21\n"
            "dtw_nn_gamma: 0.05\n"
            "dtw_nn_prototype_lr_multiplier: 5.0\n"
        )
        cfg = Config.from_yaml(str(yaml_path))
        assert cfg.dtw_nn_layer is True
        assert cfg.dtw_nn_num_nodes == 128
        assert cfg.dtw_nn_prototype_length == 21
        assert cfg.dtw_nn_gamma == 0.05
        assert cfg.dtw_nn_prototype_lr_multiplier == 5.0
