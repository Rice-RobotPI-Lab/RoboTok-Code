from dataclasses import dataclass, field, fields
import os
from pathlib import Path
from typing import List

import yaml


@dataclass
class Config:
    # -- Model --
    # Input stream the head consumes:
    #   "jepa"       — frozen V-JEPA per-token features [B, T, encoder_dim]
    #   "trajectory" — the per-design hand trajectory itself [B, T_max, D] (DTW
    #                  distillation; no semantic term). `encoder_dim` is derived
    #                  from the trajectory feature dim D at runtime, not the YAML.
    input_mode: str = "jepa"  # "jepa" | "trajectory"
    # Add sinusoidal temporal positional encoding before cross-attention.
    # Only consulted when input_mode == "trajectory" (raw frames carry no
    # positional signal, unlike V-JEPA tokens).
    use_trajectory_positional_encoding: bool = True
    encoder_dim: int = 1408
    trajectory_model_dim: int = 256
    retrieval_dim: int = 256
    num_cross_attn_heads: int = 8
    num_cross_attn_layers: int = 3
    num_projection_layers: int = 4
    # Number of learned query tokens that pool the encoder sequence. With
    # num_queries > 1, the queries are concatenated after cross-attention and
    # projected back to attention_dim before the MLP. Each query can specialize
    # on a different temporal aspect of the clip; default 1 preserves the
    # single-query behavior bit-exact.
    num_queries: int = 1
    use_grad_checkpoint: bool = True  # trade ~25% step time for ~3x less activation mem

    # -- DTW-NN head (Iwana et al.) --
    # When True (trajectory mode only), replaces input_projection + PE +
    # cross-attention with N_dtw learnable prototypes; per-prototype soft-DTW
    # distances against the input become the [B, N_dtw] activations fed to the
    # MLP. The DTW layer is the temporal aggregator (T collapses to a scalar
    # per node), so attention_dim, num_cross_attn_*, num_queries,
    # query_init_batches, use_grad_checkpoint, and
    # use_trajectory_positional_encoding all become vestigial.
    dtw_nn_layer: bool = False
    dtw_nn_num_nodes: int = 256                # N_dtw; matches retrieval_dim so MLP doesn't expand
    dtw_nn_prototype_length: int = 42          # L; cap at 2*T=84 per asymmetric slope constraint
    dtw_nn_gamma: float = 0.1                  # soft-DTW temperature; small = close to true DTW
    dtw_nn_prototype_lr_multiplier: float = 10.0  # prototypes go in a separate optimizer group

    # -- Training --
    lr: float = 1e-4
    weight_decay: float = 1e-5
    batch_size: int = 196
    num_epochs: int = 50
    optimizer: str = "adamw"
    scheduler: str = "cosine"
    max_steps_per_epoch: int = -1  # -1 = full epoch
    grad_clip_norm: float = 1.0
    mixed_precision: bool = True
    lambda_preserve: float = 1.0
    loss_type: str = "soft_contrastive"  # "soft_contrastive" | "topk_set_rank"
    loss_tau_pred: float = 0.1    # softmax temperature on predicted similarities
    loss_tau_target: float = 0.3  # softmax temperature on target similarities
    loss_components: str = "both"  # "both" | "trajectory" | "semantic" — which contributes to backward
    # Unordered DTW-neighborhood + within-neighborhood ordering objective.
    # Used only when loss_type == "topk_set_rank".
    set_loss_top_k: int = 10
    set_loss_boundary_start: int = 11  # one-indexed inclusive DTW rank
    set_loss_boundary_end: int = 20    # one-indexed inclusive DTW rank
    boundary_negatives_per_anchor: int = 1
    set_loss_hard_negatives: int = 16  # includes the explicit boundary negative
    set_loss_weight: float = 1.0
    rank_loss_weight: float = 0.2
    distribution_loss_weight: float = 0.0
    set_loss_margin: float = 0.05
    set_loss_temperature: float = 0.07
    rank_loss_temperature: float = 0.10
    seed: int = 42

    # -- Data --
    data_dir: str = "outputs/training_data"
    use_depth_grounded_keypoints: bool = True  # use depth_grounded_clip_keypoints.pt instead of clip_keypoints.pt
    dtw_design: str = "pca_interjoint_dists"
    # DataLoader knobs that affect throughput on this pipeline. The tensors
    # moved each step are large (multi-GB at batch_size=196 + max_clip_frames=40),
    # so pinned memory + a small prefetch queue typically halves h2d_transfer
    # time and keeps the GPU fed.
    pin_memory: bool = True       # pinned host memory → DMA h2d copies
    prefetch_factor: int = 1      # batches each worker prefetches (only used when num_workers > 0)
    num_workers: int = 8
    # Keep DataLoader workers alive across epochs. True saves ~3s/epoch of
    # worker re-spawn cost, but workers accumulate copy-on-write pages from
    # touching shared Python containers (e.g. clip_index), causing host RSS
    # to grow steadily across epochs. False resets RSS at each epoch
    # boundary at the cost of one cold prefetch fill per epoch.
    persistent_workers: bool = False
    max_clip_frames: int = 40  # cap per-clip frames via tubelet-boundary truncation; -1 disables
    # Cap on total number of clips used end-to-end (build_trajectories +
    # build_dtw_neighbors + train/eval). -1 = use every clip in the keypoints
    # cache. >0 = seeded random subsample down to this many clips before any
    # downstream artifact is built; the holdout split then runs on the capped
    # pool, so eval = round(eval_holdout_fraction * cap), train = the rest.
    max_total_clips: int = -1

    # -- Evaluation --
    output_dir: str = "outputs/training_outputs"
    training_name: str = ""  # optional prefix for RUN_DIR (e.g., "bs512_lr2e4")
    faiss_index_type: str = "IndexFlatIP"
    recall_k_values: List[int] = field(default_factory=lambda: [1, 5, 10, 20])
    # K values at which CKNNA is reported, applied to every CKNNA variant
    # (eval-set square, global-pool, train-pool). Independent of recall_k_values
    # so CKNNA can probe larger neighborhoods. The DTW-neighbor caches and FAISS
    # model-neighbor searches are sized to cover max(cknna_k_values).
    cknna_k_values: List[int] = field(
        default_factory=lambda: [1, 5, 10, 20, 50, 100, 250, 500]
    )
    eval_frequency: int = 10
    checkpoint_frequency: int = 10
    # Random clip-level held-out split (`train.py:random_holdout_split`), seeded
    # by `seed` so it is stable across runs with the same seed. The split is NOT
    # grouped by source video: clips from the same video can land on both sides,
    # so a held-out clip may have same-video near-duplicates both in the training
    # set and in the `eval_global_cknna` candidate pool. Set
    # `exclude_same_video_positives=True` to keep them out of mined positives,
    # and use `dtw_cknna.run_per_video_split_cknna` to report cross-video-only
    # CKNNA alongside the full number.
    eval_holdout_fraction: float = 0.1  # fraction of clips held out for eval
    eval_max_clips: int = 10000  # -1 = use all held-out clips; >0 = subsample for faster eval
    cknna_topk: int = 10  # k for the binary mask used in mutual_knn / CKNNA
    # Also evaluate the held-out queries against a candidate pool containing
    # every clip in the run. Uses the cached approximate global DTW neighbors.
    eval_global_cknna: bool = True

    # -- Logging --
    log_frequency: int = 50

    # -- Query init --
    query_init_batches: int = 10

    # -- Hard-positive mining (opt-in) --
    # Inject precomputed top-K DTW neighbors into each batch so soft-contrastive
    # targets carry real concentrated mass (instead of being smeared across
    # mostly-random negatives). See `positive_mining_sampler.py` and
    # `build_dataset/build_dtw_neighbors.py`. The neighbor cache is rebuilt
    # per run inside `run_dir`.
    use_positive_mining: bool = True           # toggle the sampler; default preserves current runs
    positives_per_anchor: int = 3               # batch_size must be divisible by (1+P); 49 × 4 = 196
    positive_mining_top_k: int = 20             # top-K stored per anchor in the cache
    positive_mining_candidate_k: int = 1000      # FAISS top-K candidates before exact DTW refinement
    positive_mining_pooling: str = "stats4"     # trajectory-pooling key for the FAISS candidate filter
    exclude_same_video_positives: bool = False   # filter out same-video clips from positives at sample time
    positive_mining_pad_policy: str = "random_train"  # "random_train" | "drop_anchor"

    @classmethod
    def from_yaml(cls, path: str) -> "Config":
        with open(path) as f:
            overrides = yaml.safe_load(f) or {}
        valid_names = {f.name for f in fields(cls)}
        filtered = {k: v for k, v in overrides.items() if k in valid_names}
        config = cls(**filtered)

        # Snapshots live below output_dir, so their YAML location is not the
        # correct base for shared data. Slurm wrappers export the live checkout
        # root; standalone runs discover it by locating this file's checkout.
        project_root = os.environ.get("ABMR_PROJECT_ROOT")
        if project_root:
            base_dir = Path(project_root).expanduser().resolve()
        else:
            config_path = Path(path).expanduser().resolve()
            base_dir = config_path.parent
            for parent in config_path.parents:
                if (parent / "retrieval_training" / "config.py").is_file():
                    base_dir = parent
                    break

        for name in ("data_dir", "output_dir"):
            value = Path(getattr(config, name)).expanduser()
            if not value.is_absolute():
                value = base_dir / value
            setattr(config, name, str(value.resolve()))
        return config
