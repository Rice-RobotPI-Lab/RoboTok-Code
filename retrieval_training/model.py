import math
from typing import Dict

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint


class DTWLayer(nn.Module):
    """Learnable soft-DTW prototype layer (DTW-NN, Iwana et al.).

    Holds `num_nodes` prototypes of shape `[L, D]`. The forward pass returns,
    for every (sample, prototype) pair, the soft-DTW distance between the
    prototype and the input sequence, giving a `[B, num_nodes]` activation
    matrix. The activation `−soft_dtw` is then BatchNorm'd to produce a
    similarity-like signal centered around zero.

    The DTW layer subsumes the input projection, positional encoding and
    cross-attention pool — it is the temporal aggregator. The D dimension of
    the input is absorbed by the prototypes' feature dim, so no `Linear(D→·)`
    is needed before it.

    Backend: pysdtw.SoftDTW with `pairwise_l2_squared` (CUDA path selected
    internally by `use_cuda=True`). pysdtw's API is
    1-to-1 batch pairing (`SoftDTW(x[B,M,D], y[B,N,D]) → [B]`), so we replicate
    inputs `num_nodes`× and tile prototypes `B`× to get `[B*num_nodes]`
    parallel DPs that we reshape back to `[B, num_nodes]`. Variable-length
    masking is not supported by pysdtw — we rely on the existing boundary-repeat
    padding of trajectories (padded frames match the last real frame, so the
    DP cost against the prototype tail is small but nonzero).
    """

    def __init__(
        self,
        num_nodes: int,
        prototype_length: int,
        feature_dim: int,
        gamma: float = 0.1,
        input_std: float = 1.0,
    ):
        super().__init__()
        if num_nodes < 1:
            raise ValueError(f"num_nodes must be >= 1, got {num_nodes}")
        if prototype_length < 1:
            raise ValueError(f"prototype_length must be >= 1, got {prototype_length}")
        if feature_dim < 1:
            raise ValueError(f"feature_dim must be >= 1, got {feature_dim}")
        self.num_nodes = num_nodes
        self.prototype_length = prototype_length
        self.feature_dim = feature_dim
        self.gamma = float(gamma)
        # Gaussian init scaled to the input standard deviation. Iwana et al.
        # report little difference between Gaussian and prototype-pattern init.
        self.prototypes = nn.Parameter(
            torch.randn(num_nodes, prototype_length, feature_dim) * input_std
        )
        # BatchNorm over the per-prototype similarity activations. Note this
        # deviates from Iwana et al., who apply a ReLU to the inverted distance
        # before BN; we skip the ReLU because the inverted distance is already a
        # usable similarity and the ReLU would zero out its lower half.
        self.bn = nn.BatchNorm1d(num_nodes)
        # Lazy holder for the pysdtw.SoftDTW object. pysdtw.SoftDTW inherits
        # from nn.Module, so assigning it to `self._sdtw` would auto-register
        # it as a child module and change module traversal after the first
        # forward. Bypass with object.__setattr__ in `_get_sdtw` so SoftDTW
        # stays a regular python attribute.
        object.__setattr__(self, "_sdtw", None)

    def _get_sdtw(self):
        # Lazy-import + lazy-construct so unit tests / CPU smoke can at least
        # import the module without pysdtw/CUDA on the path.
        if self._sdtw is None:
            import pysdtw  # noqa: PLC0415 (intentional lazy import)
            sdtw = pysdtw.SoftDTW(
                use_cuda=True,
                gamma=self.gamma,
                dist_func=pysdtw.distance.pairwise_l2_squared,
            )
            # Bypass nn.Module.__setattr__ — see note in __init__.
            object.__setattr__(self, "_sdtw", sdtw)
        return self._sdtw

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Compute soft-DTW activations against all prototypes.

        Args:
            x: `[B, T, D]` input trajectories (boundary-repeat padded).

        Returns:
            `[B, num_nodes]` BatchNorm'd similarity activations.
        """
        B, T, D = x.shape
        if D != self.feature_dim:
            raise ValueError(
                f"DTWLayer feature_dim={self.feature_dim} but received x.shape[-1]={D}"
            )
        N = self.num_nodes

        # Replicate inputs N× and tile prototypes B× into the [B*N, *, D]
        # pairwise batch pysdtw expects. Both branches preserve contiguity.
        x_rep = x.unsqueeze(1).expand(B, N, T, D).reshape(B * N, T, D).contiguous()
        proto_tile = (
            self.prototypes.unsqueeze(0)
            .expand(B, N, self.prototype_length, D)
            .reshape(B * N, self.prototype_length, D)
            .contiguous()
        )

        # Soft-DTW DP is numerically sensitive — keep it in fp32 regardless of
        # any surrounding autocast context (matches the OnlineDTWComputer
        # pattern used for the training target).
        with torch.amp.autocast("cuda", enabled=False):
            sdtw = self._get_sdtw()
            d = sdtw(x_rep.float(), proto_tile.float())  # [B*N]
        d = d.view(B, N)

        # Path-length normalisation: soft-DTW with squared-L2 cost scales
        # roughly with the DP path length (~max(T, L)). Without this the
        # raw activations are O(T·local_cost) — e.g. ~1k for T=L=42 and
        # input std ~0.3, which overflows fp16 downstream before BN has
        # populated meaningful running stats. Dividing by max(T, L) keeps
        # the activation magnitude on the per-step scale (~local_cost),
        # which is safe under any autocast configuration and also makes
        # the activation scale-invariant to clip length.
        d = d / max(T, self.prototype_length)

        # Distance → similarity (so larger = more similar), then BN to
        # zero-center and scale per node.
        a = -d
        return self.bn(a)


def _sinusoidal_positional_encoding(
    seq_len: int, dim: int, device: torch.device, dtype: torch.dtype,
) -> torch.Tensor:
    """Standard sinusoidal positional encoding, shape `(seq_len, dim)`.

    Parameter-free and length-agnostic (no `max_len`). Handles odd `dim` by
    giving the cosine channels one fewer column than the sine channels.
    """
    pe = torch.zeros(seq_len, dim, device=device, dtype=torch.float32)
    position = torch.arange(seq_len, device=device, dtype=torch.float32).unsqueeze(1)
    div_term = torch.exp(
        torch.arange(0, dim, 2, device=device, dtype=torch.float32)
        * (-math.log(10000.0) / dim)
    )
    pe[:, 0::2] = torch.sin(position * div_term)
    pe[:, 1::2] = torch.cos(position * div_term[: pe[:, 1::2].shape[1]])
    return pe.to(dtype)


class CrossAttentionProjectionHead(nn.Module):
    """Learnable query token + cross-attention pooling + MLP projection.

    A single query token attends over the encoder's patch token sequence,
    producing a fixed-size retrieval embedding regardless of input length.
    """

    def __init__(
        self,
        encoder_dim: int,
        retrieval_dim: int,
        num_heads: int = 8,
        num_projection_layers: int = 2,
        num_cross_attn_layers: int = 1,
        use_grad_checkpoint: bool = False,
        use_positional_encoding: bool = False,
        attention_dim: int | None = None,
        num_queries: int = 1,
        dtw_nn_layer: bool = False,
        dtw_nn_num_nodes: int = 256,
        dtw_nn_prototype_length: int = 42,
        dtw_nn_gamma: float = 0.1,
        dtw_nn_input_std: float = 1.0,
    ):
        super().__init__()
        self.encoder_dim = encoder_dim
        self.dtw_nn_layer = dtw_nn_layer

        # ----- DTW-NN head path -----
        # The DTW layer subsumes input_projection + PE + cross-attention. Only
        # the MLP downstream is shared with the cross-attention head path.
        # MLP width = num_nodes (the new bottleneck); attention_dim is
        # vestigial and we don't construct any of the attention modules.
        if dtw_nn_layer:
            self.dtw = DTWLayer(
                num_nodes=dtw_nn_num_nodes,
                prototype_length=dtw_nn_prototype_length,
                feature_dim=encoder_dim,
                gamma=dtw_nn_gamma,
                input_std=dtw_nn_input_std,
            )
            in_dim = dtw_nn_num_nodes
            layers = []
            for _ in range(num_projection_layers - 1):
                layers += [nn.Linear(in_dim, in_dim), nn.GELU()]
            layers.append(nn.Linear(in_dim, retrieval_dim))
            self.mlp = nn.Sequential(*layers)
            # Sentinel attributes so attribute-existence checks elsewhere
            # (and tests) can distinguish the two head modes.
            self.attention_dim = dtw_nn_num_nodes
            self.num_queries = 0
            self.use_grad_checkpoint = False
            self.use_positional_encoding = False
            return

        # ----- Cross-attention head path (default) -----
        self.use_grad_checkpoint = use_grad_checkpoint
        # Add sinusoidal temporal positional encoding to the input sequence
        # before cross-attention. The cross-attention pools tokens
        # order-agnostically otherwise — fine for V-JEPA tokens (which already
        # carry spatiotemporal structure) but it would erase motion direction
        # for raw trajectory frames.
        self.use_positional_encoding = use_positional_encoding
        self.attention_dim = attention_dim or encoder_dim
        if num_queries < 1:
            raise ValueError(f"num_queries must be >= 1, got {num_queries}")
        self.num_queries = num_queries
        if self.attention_dim % num_heads != 0:
            raise ValueError(
                f"attention_dim={self.attention_dim} must be divisible by num_heads={num_heads}"
            )
        self.input_projection = (
            nn.Identity()
            if self.attention_dim == encoder_dim
            else nn.Linear(encoder_dim, self.attention_dim)
        )
        ffn_dim = 4 * self.attention_dim
        self.query = nn.Parameter(torch.randn(num_queries, self.attention_dim))
        # With >1 queries, fold the per-query outputs back into a single
        # attention_dim-sized vector before the MLP so the downstream
        # contract (B, attention_dim) -> (B, retrieval_dim) is unchanged.
        # Identity for num_queries=1 keeps the single-query path bit-exact.
        self.query_combiner = (
            nn.Identity()
            if num_queries == 1
            else nn.Linear(num_queries * self.attention_dim, self.attention_dim)
        )
        self.cross_attns = nn.ModuleList([
            nn.MultiheadAttention(embed_dim=self.attention_dim, num_heads=num_heads, batch_first=False)
            for _ in range(num_cross_attn_layers)
        ])
        self.attn_norms = nn.ModuleList([
            nn.LayerNorm(self.attention_dim)
            for _ in range(num_cross_attn_layers)
        ])
        self.ffns = nn.ModuleList([
            nn.Sequential(
                nn.Linear(self.attention_dim, ffn_dim),
                nn.GELU(),
                nn.Linear(ffn_dim, self.attention_dim),
            )
            for _ in range(num_cross_attn_layers)
        ])
        self.ffn_norms = nn.ModuleList([
            nn.LayerNorm(self.attention_dim)
            for _ in range(num_cross_attn_layers)
        ])

        layers = []
        in_dim = self.attention_dim
        for _ in range(num_projection_layers - 1):
            layers += [nn.Linear(in_dim, in_dim), nn.GELU()]
        layers.append(nn.Linear(in_dim, retrieval_dim))
        self.mlp = nn.Sequential(*layers)

    @staticmethod
    def _cross_attn_block(q, kv, key_padding_mask, cross_attn, attn_norm, ffn, ffn_norm):
        attn_out, _ = cross_attn(q, kv, kv, key_padding_mask=key_padding_mask)
        q = attn_norm(q + attn_out)
        q = ffn_norm(q + ffn(q))
        return q

    def initialize_query(self, patch_embeddings: torch.Tensor) -> None:
        """Set query rows to the grand mean of projected patch embeddings.

        With num_queries > 1 a small Gaussian perturbation is added so the
        rows don't collapse to identical attention distributions in the early
        steps. num_queries == 1 stays bit-exact (no perturbation).

        No-op when `dtw_nn_layer=True` (no learnable query exists).

        Args:
            patch_embeddings: concatenated patch tokens, shape (N_total, D).
        """
        if self.dtw_nn_layer:
            return
        with torch.no_grad():
            mean = (
                self.project_input(patch_embeddings.unsqueeze(0))
                .squeeze(0)
                .mean(dim=0, keepdim=True)
            )
            q = mean.expand(self.num_queries, -1).contiguous()
            if self.num_queries > 1:
                # Scale noise to the per-channel magnitude of the mean so the
                # perturbation is meaningful regardless of feature scale.
                noise_scale = 0.02 * mean.abs().mean().clamp(min=1e-6)
                q = q + noise_scale * torch.randn_like(q)
            self.query.copy_(q)

    def project_input(self, patch_embeddings: torch.Tensor) -> torch.Tensor:
        """Map input tokens to the attention dimension.

        Identity on the DTW-NN head path (the prototypes' feature dim absorbs
        D directly). This branch exists because `train.py:initialize_query_token`
        calls `project_input` to compute a streaming mean for query init —
        making it a no-op transform on the DTW path keeps that callsite safe
        even though `initialize_query` itself early-returns above.
        """
        if self.dtw_nn_layer:
            return patch_embeddings
        return self.input_projection(patch_embeddings)

    def forward(
        self,
        patch_embeddings: torch.Tensor,
        padding_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Args:
            patch_embeddings: (B, T_patches, D) from the frozen encoder, or
                in trajectory+dtw_nn mode the raw trajectory tensor.
            padding_mask: (B, T_patches) bool, True = real token, False = pad.
                Ignored on the DTW-NN path (pysdtw doesn't accept length masks;
                we rely on boundary-repeat padding).

        Returns:
            L2-normalised retrieval embeddings, shape (B, retrieval_dim).
        """
        if self.dtw_nn_layer:
            # DTW layer pools over T and produces [B, num_nodes]; the MLP
            # then maps to retrieval_dim. Run the entire DTW head in fp32:
            # at the first eval BN's running stats are still (0, 1) so it
            # passes activations through unscaled, and downstream MLP +
            # F.normalize would overflow fp16 for typical soft-DTW magnitudes.
            # The MLP is tiny relative to the soft-DTW DP, so fp32 cost is
            # negligible.
            with torch.amp.autocast("cuda", enabled=False):
                a = self.dtw(patch_embeddings.float())
                z = self.mlp(a)
                return F.normalize(z, dim=-1)

        patch_embeddings = self.project_input(patch_embeddings)
        B, T, D = patch_embeddings.shape

        if self.use_positional_encoding:
            pe = _sinusoidal_positional_encoding(
                T, D, patch_embeddings.device, patch_embeddings.dtype,
            )
            patch_embeddings = patch_embeddings + pe.unsqueeze(0)

        # MHA expects (seq, batch, dim) when batch_first=False
        kv = patch_embeddings.permute(1, 0, 2)  # (T, B, D)
        # Broadcast the (num_queries, D) parameter to (num_queries, B, D).
        q = self.query.unsqueeze(1).expand(-1, B, -1)

        # nn.MultiheadAttention key_padding_mask: True = ignore
        key_padding_mask = ~padding_mask if padding_mask is not None else None

        for cross_attn, attn_norm, ffn, ffn_norm in zip(
            self.cross_attns, self.attn_norms, self.ffns, self.ffn_norms,
        ):
            if self.use_grad_checkpoint and self.training:
                q = checkpoint(
                    self._cross_attn_block, q, kv, key_padding_mask,
                    cross_attn, attn_norm, ffn, ffn_norm,
                    use_reentrant=False,
                )
            else:
                q = self._cross_attn_block(
                    q, kv, key_padding_mask, cross_attn, attn_norm, ffn, ffn_norm,
                )

        # (num_queries, B, D) -> (B, num_queries * D) -> (B, D)
        # For num_queries == 1 the combiner is Identity and this is equivalent
        # to the previous q.squeeze(0).
        attn_out = q.permute(1, 0, 2).reshape(B, self.num_queries * D)
        attn_out = self.query_combiner(attn_out)

        z = self.mlp(attn_out)
        return F.normalize(z, dim=-1)


class TrajectoryRetrievalModel(nn.Module):
    """Trainable cross-attention projection head over precomputed V-JEPA 2 features."""

    def __init__(
        self,
        encoder_dim: int,
        retrieval_dim: int,
        num_heads: int = 8,
        num_projection_layers: int = 2,
        num_cross_attn_layers: int = 1,
        use_grad_checkpoint: bool = False,
        use_positional_encoding: bool = False,
        attention_dim: int | None = None,
        num_queries: int = 1,
        dtw_nn_layer: bool = False,
        dtw_nn_num_nodes: int = 256,
        dtw_nn_prototype_length: int = 42,
        dtw_nn_gamma: float = 0.1,
        dtw_nn_input_std: float = 1.0,
    ):
        super().__init__()
        self.head = CrossAttentionProjectionHead(
            encoder_dim=encoder_dim,
            retrieval_dim=retrieval_dim,
            num_heads=num_heads,
            num_projection_layers=num_projection_layers,
            num_cross_attn_layers=num_cross_attn_layers,
            use_grad_checkpoint=use_grad_checkpoint,
            use_positional_encoding=use_positional_encoding,
            attention_dim=attention_dim,
            num_queries=num_queries,
            dtw_nn_layer=dtw_nn_layer,
            dtw_nn_num_nodes=dtw_nn_num_nodes,
            dtw_nn_prototype_length=dtw_nn_prototype_length,
            dtw_nn_gamma=dtw_nn_gamma,
            dtw_nn_input_std=dtw_nn_input_std,
        )

    def forward(
        self,
        jepa_features: torch.Tensor,
        padding_mask: torch.Tensor | None = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Args:
            jepa_features: (B, T, D) precomputed per-token V-JEPA 2 features.
            padding_mask: (B, T) bool, True = real token.
        """
        z = self.head(jepa_features, padding_mask)
        v_ref = self._masked_mean_pool(jepa_features, padding_mask)

        return {"z": z, "v_ref": v_ref}

    @staticmethod
    def _masked_mean_pool(
        features: torch.Tensor, mask: torch.Tensor | None
    ) -> torch.Tensor:
        """Mean-pool over real tokens only, then L2-normalise. Detached."""
        if mask is not None:
            mask_f = mask.unsqueeze(-1).float()  # (B, T, 1)
            v = (features * mask_f).sum(dim=1) / mask_f.sum(dim=1).clamp(min=1)
        else:
            v = features.mean(dim=1)
        return F.normalize(v.detach(), dim=-1)

    def get_trainable_parameters(self):
        return self.head.parameters()

    def get_prototype_parameters(self):
        """Iterator over DTW-NN prototype parameters (empty for non-DTW head).

        Used by `train.py` to put prototypes in a separate optimizer param
        group with a configurable learning-rate multiplier. Empty
        iterator when `dtw_nn_layer=False`, so the caller's group-split logic
        is the same regardless of head mode.
        """
        if not self.head.dtw_nn_layer:
            return iter(())
        return iter([self.head.dtw.prototypes])

    def get_non_prototype_parameters(self):
        """Iterator over head params excluding DTW-NN prototypes.

        Companion to `get_prototype_parameters`. When `dtw_nn_layer=False`
        this is identical to `get_trainable_parameters`.
        """
        if not self.head.dtw_nn_layer:
            return self.head.parameters()
        proto_ids = {id(p) for p in self.get_prototype_parameters()}
        return (p for p in self.head.parameters() if id(p) not in proto_ids)
