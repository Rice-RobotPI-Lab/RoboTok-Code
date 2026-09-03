"""Tests for retrieval_training/model.py.

Run from retrieval_training/: python -m pytest tests/test_model.py -v
"""

import sys
from pathlib import Path

import pytest
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from model import CrossAttentionProjectionHead, DTWLayer, TrajectoryRetrievalModel

B = 4
T = 20
D = 64
R = 16
HEADS = 4


# ---------------------------------------------------------------------------
# CrossAttentionProjectionHead
# ---------------------------------------------------------------------------

class TestCrossAttentionProjectionHead:
    @pytest.fixture
    def head(self):
        return CrossAttentionProjectionHead(
            encoder_dim=D, retrieval_dim=R, num_heads=HEADS, num_projection_layers=2,
            num_cross_attn_layers=1,
        )

    def test_output_shape(self, head):
        x = torch.randn(B, T, D)
        mask = torch.ones(B, T, dtype=torch.bool)
        out = head(x, mask)
        assert out.shape == (B, R)

    def test_output_l2_normalized(self, head):
        x = torch.randn(B, T, D)
        mask = torch.ones(B, T, dtype=torch.bool)
        out = head(x, mask)
        norms = out.norm(dim=-1)
        assert torch.allclose(norms, torch.ones(B), atol=1e-5)

    def test_no_mask(self, head):
        x = torch.randn(B, T, D)
        out = head(x, padding_mask=None)
        assert out.shape == (B, R)

    def test_variable_length_mask(self, head):
        x = torch.randn(B, T, D)
        mask = torch.zeros(B, T, dtype=torch.bool)
        mask[0, :5] = True
        mask[1, :10] = True
        mask[2, :15] = True
        mask[3, :T] = True
        out = head(x, mask)
        assert out.shape == (B, R)
        assert torch.isfinite(out).all()

    def test_single_token(self, head):
        x = torch.randn(1, 1, D)
        mask = torch.ones(1, 1, dtype=torch.bool)
        out = head(x, mask)
        assert out.shape == (1, R)

    def test_initialize_query(self, head):
        patches = torch.randn(100, D)
        expected = patches.mean(dim=0, keepdim=True)
        head.initialize_query(patches)
        assert torch.allclose(head.query.data, expected)

    def test_query_shape(self, head):
        assert head.query.shape == (1, D)
        assert head.attention_dim == D
        assert isinstance(head.input_projection, torch.nn.Identity)

    def test_attention_dim_projection(self):
        input_dim = 126
        attention_dim = 256
        head = CrossAttentionProjectionHead(
            encoder_dim=input_dim,
            retrieval_dim=R,
            num_heads=8,
            attention_dim=attention_dim,
        )
        x = torch.randn(B, T, input_dim)
        mask = torch.ones(B, T, dtype=torch.bool)
        out = head(x, mask)
        assert out.shape == (B, R)
        assert head.query.shape == (1, attention_dim)
        assert head.cross_attns[0].embed_dim == attention_dim
        assert isinstance(head.input_projection, torch.nn.Linear)

    def test_attention_dim_projection_initialize_query(self):
        input_dim = 126
        attention_dim = 256
        head = CrossAttentionProjectionHead(
            encoder_dim=input_dim,
            retrieval_dim=R,
            num_heads=8,
            attention_dim=attention_dim,
        )
        patches = torch.randn(100, input_dim)
        head.initialize_query(patches)
        expected = head.project_input(patches.unsqueeze(0)).squeeze(0).mean(dim=0, keepdim=True)
        assert head.query.shape == (1, attention_dim)
        assert torch.allclose(head.query.data, expected)

    def test_invalid_attention_dim_raises(self):
        with pytest.raises(ValueError, match="must be divisible"):
            CrossAttentionProjectionHead(
                encoder_dim=126,
                retrieval_dim=R,
                num_heads=8,
                attention_dim=130,
            )

    def test_num_projection_layers_1(self):
        head = CrossAttentionProjectionHead(
            encoder_dim=D, retrieval_dim=R, num_heads=HEADS, num_projection_layers=1
        )
        x = torch.randn(B, T, D)
        out = head(x)
        assert out.shape == (B, R)

    def test_num_projection_layers_3(self):
        head = CrossAttentionProjectionHead(
            encoder_dim=D, retrieval_dim=R, num_heads=HEADS, num_projection_layers=3
        )
        x = torch.randn(B, T, D)
        out = head(x)
        assert out.shape == (B, R)

    def test_num_cross_attn_layers_3(self):
        head = CrossAttentionProjectionHead(
            encoder_dim=D, retrieval_dim=R, num_heads=HEADS, num_cross_attn_layers=3
        )
        x = torch.randn(B, T, D)
        out = head(x)
        assert out.shape == (B, R)

    def test_stacked_cross_attn_with_mask(self):
        head = CrossAttentionProjectionHead(
            encoder_dim=D, retrieval_dim=R, num_heads=HEADS, num_cross_attn_layers=3
        )
        x = torch.randn(B, T, D)
        mask = torch.zeros(B, T, dtype=torch.bool)
        mask[0, :5] = True
        mask[1, :T] = True
        mask[2, :10] = True
        mask[3, :3] = True
        out = head(x, mask)
        assert out.shape == (B, R)
        assert torch.isfinite(out).all()

    def test_gradients_flow(self, head):
        x = torch.randn(B, T, D)
        mask = torch.ones(B, T, dtype=torch.bool)
        out = head(x, mask)
        out.sum().backward()
        assert head.query.grad is not None
        for p in head.mlp.parameters():
            assert p.grad is not None
        for ca in head.cross_attns:
            for p in ca.parameters():
                assert p.grad is not None
        for ffn in head.ffns:
            for p in ffn.parameters():
                assert p.grad is not None

    def test_num_queries_default_is_one(self, head):
        assert head.num_queries == 1
        assert head.query.shape == (1, D)
        assert isinstance(head.query_combiner, torch.nn.Identity)

    def test_num_queries_multi_shapes(self):
        nq = 4
        head = CrossAttentionProjectionHead(
            encoder_dim=D, retrieval_dim=R, num_heads=HEADS, num_queries=nq,
        )
        assert head.query.shape == (nq, D)
        assert isinstance(head.query_combiner, torch.nn.Linear)
        assert head.query_combiner.in_features == nq * D
        assert head.query_combiner.out_features == D

    def test_num_queries_multi_forward(self):
        head = CrossAttentionProjectionHead(
            encoder_dim=D, retrieval_dim=R, num_heads=HEADS, num_queries=4,
        )
        x = torch.randn(B, T, D)
        mask = torch.ones(B, T, dtype=torch.bool)
        out = head(x, mask)
        assert out.shape == (B, R)
        norms = out.norm(dim=-1)
        assert torch.allclose(norms, torch.ones(B), atol=1e-5)

    def test_num_queries_multi_with_padding(self):
        head = CrossAttentionProjectionHead(
            encoder_dim=D, retrieval_dim=R, num_heads=HEADS, num_queries=3,
        )
        x = torch.randn(B, T, D)
        mask = torch.zeros(B, T, dtype=torch.bool)
        mask[0, :5] = True
        mask[1, :T] = True
        mask[2, :10] = True
        mask[3, :3] = True
        out = head(x, mask)
        assert out.shape == (B, R)
        assert torch.isfinite(out).all()

    def test_num_queries_multi_initialize_query(self):
        nq = 3
        head = CrossAttentionProjectionHead(
            encoder_dim=D, retrieval_dim=R, num_heads=HEADS, num_queries=nq,
        )
        torch.manual_seed(0)
        patches = torch.randn(100, D)
        head.initialize_query(patches)
        assert head.query.shape == (nq, D)
        # Rows should cluster near the data mean but not be identical (noise).
        mean = patches.mean(dim=0, keepdim=True)
        residuals = head.query.data - mean
        assert residuals.abs().max() > 0
        # ... and stay close to the mean (scaled noise, not random reinit).
        assert (head.query.data - mean).norm(dim=-1).mean() < mean.norm() + 1.0

    def test_num_queries_multi_gradients_flow(self):
        head = CrossAttentionProjectionHead(
            encoder_dim=D, retrieval_dim=R, num_heads=HEADS, num_queries=4,
        )
        x = torch.randn(B, T, D)
        out = head(x)
        out.sum().backward()
        assert head.query.grad is not None
        assert head.query.grad.shape == head.query.shape
        for p in head.query_combiner.parameters():
            assert p.grad is not None

    def test_num_queries_invalid_raises(self):
        with pytest.raises(ValueError, match="num_queries"):
            CrossAttentionProjectionHead(
                encoder_dim=D, retrieval_dim=R, num_heads=HEADS, num_queries=0,
            )

    def test_gradients_flow_stacked(self):
        head = CrossAttentionProjectionHead(
            encoder_dim=D, retrieval_dim=R, num_heads=HEADS,
            num_projection_layers=4, num_cross_attn_layers=3,
        )
        x = torch.randn(B, T, D)
        out = head(x)
        out.sum().backward()
        assert head.query.grad is not None
        for i, ca in enumerate(head.cross_attns):
            for p in ca.parameters():
                assert p.grad is not None, f"No grad in cross_attn layer {i}"
        for i, ffn in enumerate(head.ffns):
            for p in ffn.parameters():
                assert p.grad is not None, f"No grad in ffn layer {i}"


# ---------------------------------------------------------------------------
# TrajectoryRetrievalModel
# ---------------------------------------------------------------------------

class TestTrajectoryRetrievalModel:
    @pytest.fixture
    def model(self):
        return TrajectoryRetrievalModel(
            encoder_dim=D, retrieval_dim=R, num_heads=HEADS, num_projection_layers=2
        )

    def test_output_keys(self, model):
        x = torch.randn(B, T, D)
        mask = torch.ones(B, T, dtype=torch.bool)
        out = model(x, mask)
        assert set(out.keys()) == {"z", "v_ref"}

    def test_z_shape(self, model):
        x = torch.randn(B, T, D)
        out = model(x)
        assert out["z"].shape == (B, R)

    def test_v_ref_shape(self, model):
        x = torch.randn(B, T, D)
        out = model(x)
        assert out["v_ref"].shape == (B, D)

    def test_z_is_l2_normalized(self, model):
        x = torch.randn(B, T, D)
        out = model(x)
        norms = out["z"].norm(dim=-1)
        assert torch.allclose(norms, torch.ones(B), atol=1e-5)

    def test_v_ref_is_l2_normalized(self, model):
        x = torch.randn(B, T, D)
        out = model(x)
        norms = out["v_ref"].norm(dim=-1)
        assert torch.allclose(norms, torch.ones(B), atol=1e-5)

    def test_v_ref_is_detached(self, model):
        x = torch.randn(B, T, D)
        out = model(x)
        assert not out["v_ref"].requires_grad

    def test_v_ref_no_gradient(self, model):
        x = torch.randn(B, T, D, requires_grad=True)
        out = model(x)
        out["z"].sum().backward()
        # v_ref is detached so doesn't contribute to gradient
        assert out["v_ref"].grad_fn is None

    def test_with_padding_mask(self, model):
        x = torch.randn(B, T, D)
        mask = torch.zeros(B, T, dtype=torch.bool)
        mask[0, :5] = True
        mask[1, :10] = True
        mask[2, :T] = True
        mask[3, :3] = True
        out = model(x, mask)
        assert torch.isfinite(out["z"]).all()
        assert torch.isfinite(out["v_ref"]).all()

    def test_no_mask(self, model):
        x = torch.randn(B, T, D)
        out = model(x, padding_mask=None)
        assert out["z"].shape == (B, R)
        assert out["v_ref"].shape == (B, D)

    def test_attention_dim_projection_keeps_v_ref_input_dim(self):
        input_dim = 126
        attention_dim = 256
        model = TrajectoryRetrievalModel(
            encoder_dim=input_dim,
            retrieval_dim=R,
            num_heads=8,
            attention_dim=attention_dim,
        )
        x = torch.randn(B, T, input_dim)
        out = model(x)
        assert out["z"].shape == (B, R)
        assert out["v_ref"].shape == (B, input_dim)
        assert model.head.query.shape == (1, attention_dim)


# ---------------------------------------------------------------------------
# _masked_mean_pool
# ---------------------------------------------------------------------------

class TestMaskedMeanPool:
    def test_no_mask_is_simple_mean(self):
        x = torch.randn(B, T, D)
        v = TrajectoryRetrievalModel._masked_mean_pool(x, None)
        expected = F.normalize(x.mean(dim=1), dim=-1)
        assert torch.allclose(v, expected, atol=1e-5)

    def test_mask_excludes_pad_tokens(self):
        x = torch.randn(1, 10, D)
        mask = torch.zeros(1, 10, dtype=torch.bool)
        mask[0, :3] = True
        v = TrajectoryRetrievalModel._masked_mean_pool(x, mask)
        expected = F.normalize(x[0, :3].mean(dim=0, keepdim=True), dim=-1)
        assert torch.allclose(v, expected, atol=1e-5)

    def test_output_is_l2_normalized(self):
        x = torch.randn(B, T, D)
        mask = torch.ones(B, T, dtype=torch.bool)
        v = TrajectoryRetrievalModel._masked_mean_pool(x, mask)
        norms = v.norm(dim=-1)
        assert torch.allclose(norms, torch.ones(B), atol=1e-5)

    def test_output_is_detached(self):
        x = torch.randn(B, T, D, requires_grad=True)
        mask = torch.ones(B, T, dtype=torch.bool)
        v = TrajectoryRetrievalModel._masked_mean_pool(x, mask)
        assert not v.requires_grad


# ---------------------------------------------------------------------------
# get_trainable_parameters
# ---------------------------------------------------------------------------

class TestTrainableParameters:
    def test_returns_head_params(self):
        model = TrajectoryRetrievalModel(encoder_dim=D, retrieval_dim=R, num_heads=HEADS)
        trainable = list(model.get_trainable_parameters())
        head_params = list(model.head.parameters())
        assert len(trainable) == len(head_params)
        for t, h in zip(trainable, head_params):
            assert t.data_ptr() == h.data_ptr()

    def test_param_count(self):
        model = TrajectoryRetrievalModel(
            encoder_dim=D, retrieval_dim=R, num_heads=HEADS, num_projection_layers=2
        )
        total = sum(p.numel() for p in model.get_trainable_parameters())
        assert total > 0


# ---------------------------------------------------------------------------
# DTW-NN head — CPU-side checks (no pysdtw/CUDA needed)
# ---------------------------------------------------------------------------

N_DTW = 16
L_PROTO = 12


class TestDTWLayerConstruction:
    """Structural checks that don't require a CUDA-backed pysdtw forward."""

    def test_prototype_shape(self):
        layer = DTWLayer(num_nodes=N_DTW, prototype_length=L_PROTO, feature_dim=D)
        assert layer.prototypes.shape == (N_DTW, L_PROTO, D)
        assert layer.prototypes.requires_grad

    def test_batchnorm_constructed(self):
        layer = DTWLayer(num_nodes=N_DTW, prototype_length=L_PROTO, feature_dim=D)
        assert isinstance(layer.bn, torch.nn.BatchNorm1d)
        assert layer.bn.num_features == N_DTW

    def test_gamma_stored(self):
        layer = DTWLayer(
            num_nodes=N_DTW, prototype_length=L_PROTO, feature_dim=D, gamma=0.05
        )
        assert layer.gamma == 0.05

    def test_init_std_scales_prototypes(self):
        torch.manual_seed(0)
        small = DTWLayer(
            num_nodes=N_DTW, prototype_length=L_PROTO, feature_dim=D, input_std=0.01
        )
        torch.manual_seed(0)
        large = DTWLayer(
            num_nodes=N_DTW, prototype_length=L_PROTO, feature_dim=D, input_std=10.0
        )
        assert small.prototypes.std() < 1.0
        assert large.prototypes.std() > 1.0

    def test_invalid_num_nodes_raises(self):
        with pytest.raises(ValueError, match="num_nodes"):
            DTWLayer(num_nodes=0, prototype_length=L_PROTO, feature_dim=D)

    def test_invalid_prototype_length_raises(self):
        with pytest.raises(ValueError, match="prototype_length"):
            DTWLayer(num_nodes=N_DTW, prototype_length=0, feature_dim=D)

    def test_invalid_feature_dim_raises(self):
        with pytest.raises(ValueError, match="feature_dim"):
            DTWLayer(num_nodes=N_DTW, prototype_length=L_PROTO, feature_dim=0)

    def test_forward_feature_dim_mismatch_raises(self):
        layer = DTWLayer(num_nodes=N_DTW, prototype_length=L_PROTO, feature_dim=D)
        x = torch.randn(2, 8, D + 1)
        with pytest.raises(ValueError, match="feature_dim"):
            layer(x)


# ---------------------------------------------------------------------------
# DTW-NN head — integration with CrossAttentionProjectionHead /
# TrajectoryRetrievalModel — structural checks only (no forward).
# ---------------------------------------------------------------------------

class TestDTWNNHeadStructure:
    def test_head_replaces_attention_modules(self):
        head = CrossAttentionProjectionHead(
            encoder_dim=D,
            retrieval_dim=R,
            num_heads=HEADS,
            dtw_nn_layer=True,
            dtw_nn_num_nodes=N_DTW,
            dtw_nn_prototype_length=L_PROTO,
        )
        assert head.dtw_nn_layer is True
        assert hasattr(head, "dtw")
        assert isinstance(head.dtw, DTWLayer)
        # Attention modules must not exist on the DTW path.
        for attr in ("query", "query_combiner", "cross_attns", "ffns",
                     "attn_norms", "ffn_norms", "input_projection"):
            assert not hasattr(head, attr), f"unexpected {attr!r} on DTW head"
        # Sentinel attributes for downstream checks
        assert head.num_queries == 0
        assert head.attention_dim == N_DTW

    def test_mlp_input_width_equals_num_nodes(self):
        head = CrossAttentionProjectionHead(
            encoder_dim=D, retrieval_dim=R, num_heads=HEADS, dtw_nn_layer=True,
            dtw_nn_num_nodes=N_DTW, dtw_nn_prototype_length=L_PROTO,
        )
        # First linear in the MLP should consume N_DTW.
        first_linear = next(m for m in head.mlp if isinstance(m, torch.nn.Linear))
        assert first_linear.in_features == N_DTW

    def test_initialize_query_is_noop(self):
        head = CrossAttentionProjectionHead(
            encoder_dim=D, retrieval_dim=R, num_heads=HEADS, dtw_nn_layer=True,
            dtw_nn_num_nodes=N_DTW, dtw_nn_prototype_length=L_PROTO,
        )
        # Should not raise even with arbitrary input; no query exists.
        head.initialize_query(torch.randn(50, D))

    def test_project_input_is_identity_on_dtw_path(self):
        head = CrossAttentionProjectionHead(
            encoder_dim=D, retrieval_dim=R, num_heads=HEADS, dtw_nn_layer=True,
            dtw_nn_num_nodes=N_DTW, dtw_nn_prototype_length=L_PROTO,
        )
        x = torch.randn(2, 5, D)
        assert torch.equal(head.project_input(x), x)

    def test_prototype_param_split(self):
        model = TrajectoryRetrievalModel(
            encoder_dim=D, retrieval_dim=R, num_heads=HEADS,
            dtw_nn_layer=True,
            dtw_nn_num_nodes=N_DTW, dtw_nn_prototype_length=L_PROTO,
        )
        proto = list(model.get_prototype_parameters())
        non_proto = list(model.get_non_prototype_parameters())
        all_head = list(model.head.parameters())
        # Exactly one prototype tensor and it's the DTW layer's `prototypes`.
        assert len(proto) == 1
        assert proto[0].data_ptr() == model.head.dtw.prototypes.data_ptr()
        # Disjoint union covers every head param exactly once.
        ids_proto = {id(p) for p in proto}
        ids_non = {id(p) for p in non_proto}
        assert ids_proto.isdisjoint(ids_non)
        assert ids_proto | ids_non == {id(p) for p in all_head}

    def test_prototype_param_split_empty_on_cross_attn_head(self):
        model = TrajectoryRetrievalModel(
            encoder_dim=D, retrieval_dim=R, num_heads=HEADS, dtw_nn_layer=False,
        )
        assert list(model.get_prototype_parameters()) == []
        # Non-prototype iterator collapses to the full head params.
        non_proto = list(model.get_non_prototype_parameters())
        all_head = list(model.head.parameters())
        assert len(non_proto) == len(all_head)


# ---------------------------------------------------------------------------
# DTW-NN forward — requires CUDA + pysdtw
# ---------------------------------------------------------------------------

class TestDTWNNForward:
    @staticmethod
    def _setup_or_skip():
        if not torch.cuda.is_available():
            pytest.skip("DTW-NN forward needs CUDA")
        pytest.importorskip("pysdtw", reason="DTW-NN forward needs pysdtw")

    def test_forward_shape_and_normalization(self):
        self._setup_or_skip()
        model = TrajectoryRetrievalModel(
            encoder_dim=D, retrieval_dim=R, num_heads=HEADS,
            dtw_nn_layer=True,
            dtw_nn_num_nodes=N_DTW, dtw_nn_prototype_length=L_PROTO,
        ).cuda()
        x = torch.randn(B, T, D, device="cuda")
        out = model(x)
        assert out["z"].shape == (B, R)
        assert out["v_ref"].shape == (B, D)
        norms = out["z"].norm(dim=-1)
        assert torch.allclose(norms, torch.ones(B, device="cuda"), atol=1e-4)

    def test_gradient_flows_to_prototypes(self):
        self._setup_or_skip()
        model = TrajectoryRetrievalModel(
            encoder_dim=D, retrieval_dim=R, num_heads=HEADS,
            dtw_nn_layer=True,
            dtw_nn_num_nodes=N_DTW, dtw_nn_prototype_length=L_PROTO,
        ).cuda()
        x = torch.randn(B, T, D, device="cuda")
        out = model(x)
        out["z"].sum().backward()
        proto = model.head.dtw.prototypes
        assert proto.grad is not None
        assert proto.grad.shape == proto.shape
        # At least some gradient signal reached the prototypes.
        assert proto.grad.abs().sum() > 0

    def test_padding_mask_is_ignored(self):
        # The DTW-NN path doesn't consume the mask. Passing any mask should
        # not change z (modulo BN running stats, which we sidestep with eval).
        self._setup_or_skip()
        model = TrajectoryRetrievalModel(
            encoder_dim=D, retrieval_dim=R, num_heads=HEADS,
            dtw_nn_layer=True,
            dtw_nn_num_nodes=N_DTW, dtw_nn_prototype_length=L_PROTO,
        ).cuda().eval()
        x = torch.randn(B, T, D, device="cuda")
        mask = torch.zeros(B, T, dtype=torch.bool, device="cuda")
        mask[:, :5] = True
        with torch.no_grad():
            z_with_mask = model(x, mask)["z"]
            z_no_mask = model(x, None)["z"]
        assert torch.allclose(z_with_mask, z_no_mask, atol=1e-5)

    def test_no_nan_inf_under_autocast_before_bn_warmup(self):
        """Regression: at the first eval (BN running stats still (0, 1) so BN
        is approximately identity), the DTW head used to overflow fp16
        downstream under autocast. The fix runs the head in fp32 and
        length-normalises the soft-DTW output. This test exercises the worst
        case: full-size N_dtw + L matching the training default, in eval mode
        under autocast, and checks the embedding stays finite."""
        self._setup_or_skip()
        model = TrajectoryRetrievalModel(
            encoder_dim=126, retrieval_dim=256, num_heads=8,
            dtw_nn_layer=True,
            dtw_nn_num_nodes=256, dtw_nn_prototype_length=42,
            dtw_nn_input_std=0.33,
        ).cuda().eval()
        x = torch.randn(8, 42, 126, device="cuda") * 0.33
        with torch.no_grad(), torch.amp.autocast("cuda", enabled=True):
            z = model(x)["z"]
        assert torch.isfinite(z).all(), (
            f"non-finite z under autocast: {(~torch.isfinite(z)).sum().item()} "
            f"of {z.numel()} entries"
        )
        # And unit-norm.
        norms = z.norm(dim=-1)
        assert torch.allclose(norms, torch.ones(8, device="cuda"), atol=1e-3)

    def test_length_normalisation_keeps_activations_bounded(self):
        """The soft-DTW output scales with the DP path length. The DTWLayer
        divides by max(T, L) so the activations are on the per-step scale
        regardless of clip length. Check the magnitude is on the order of
        the per-step squared-L2 cost rather than T× that."""
        self._setup_or_skip()
        layer = DTWLayer(
            num_nodes=N_DTW, prototype_length=42, feature_dim=126,
            gamma=0.1, input_std=0.33,
        ).cuda()
        x = torch.randn(8, 42, 126, device="cuda") * 0.33
        with torch.no_grad():
            a = layer(x)
        # Per-step squared-L2 cost ≈ 2 * 0.33² * 126 ≈ 27. With BN initialised
        # weight=1/bias=0/running=(0,1), output ≈ -d / max(T, L). So |a| is
        # bounded by O(local_cost), well under O(T·local_cost) ≈ 1000.
        assert a.abs().max().item() < 200.0

    def test_module_enumeration_stable_across_forward(self):
        """SoftDTW is bypass-registered so `modules()` doesn't change after
        the first forward — checkpoint round-trip + model.modules() iteration
        stay deterministic regardless of when the lazy sdtw object is built."""
        self._setup_or_skip()
        model = TrajectoryRetrievalModel(
            encoder_dim=D, retrieval_dim=R, num_heads=HEADS,
            dtw_nn_layer=True,
            dtw_nn_num_nodes=N_DTW, dtw_nn_prototype_length=L_PROTO,
        ).cuda()
        before = {name for name, _ in model.named_modules()}
        with torch.no_grad():
            model(torch.randn(B, T, D, device="cuda"))
        after = {name for name, _ in model.named_modules()}
        assert before == after
