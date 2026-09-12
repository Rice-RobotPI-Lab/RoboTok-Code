import argparse
import random
import time
import warnings
from collections import defaultdict
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from torch.utils.tensorboard import SummaryWriter


class EpochProfiler:
    """Accumulates per-section wall-clock time within an epoch.

    Times CUDA work accurately by syncing on enter/exit. Use sparingly — each
    sync forces the GPU to flush and adds a small fixed overhead per call.
    """

    def __init__(self, sync_cuda: bool = True):
        self.sync_cuda = sync_cuda and torch.cuda.is_available()
        self.totals: dict[str, float] = defaultdict(float)
        self.counts: dict[str, int] = defaultdict(int)

    @contextmanager
    def section(self, name: str):
        if self.sync_cuda:
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        try:
            yield
        finally:
            if self.sync_cuda:
                torch.cuda.synchronize()
            self.totals[name] += time.perf_counter() - t0
            self.counts[name] += 1

    def report(self, total_wall: float, prefix: str = "  ") -> str:
        lines = [f"{prefix}timing breakdown (total wall {total_wall:.1f}s):"]
        accounted = sum(self.totals.values())
        rows = sorted(self.totals.items(), key=lambda kv: -kv[1])
        for name, t in rows:
            pct = 100.0 * t / total_wall if total_wall > 0 else 0.0
            n = self.counts[name]
            avg_ms = 1000.0 * t / n if n else 0.0
            lines.append(f"{prefix}  {name:<22} {t:7.2f}s  ({pct:5.1f}%)  "
                         f"n={n:<5d} avg={avg_ms:.1f}ms")
        other = total_wall - accounted
        if total_wall > 0:
            pct_other = 100.0 * other / total_wall
            lines.append(f"{prefix}  {'(unaccounted)':<22} {other:7.2f}s  ({pct_other:5.1f}%)")
        # Host RSS at end of epoch: parent + sum over child workers (so the
        # number tracks what slurm's cgroup OOM-killer sees).
        try:
            import psutil
            p = psutil.Process()
            rss_gb = p.memory_info().rss / 1e9
            children = p.children(recursive=True)
            child_rss_gb = sum(c.memory_info().rss for c in children) / 1e9
            lines.append(f"{prefix}  host RSS: parent={rss_gb:.2f}GB "
                         f"+ {len(children)} children={child_rss_gb:.2f}GB "
                         f"= total={rss_gb + child_rss_gb:.2f}GB")
        except Exception as e:
            lines.append(f"{prefix}  host RSS: <psutil failed: {e}>")
        return "\n".join(lines)

from dataclasses import fields

from config import Config
from data import JepaPooledDataset, VideoClipDataset, collate_fn, select_model_input
from eval import run_evaluation
from losses import CombinedLoss, TopKSetRankLoss, _upper_triangle, target_entropy_nats
from model import TrajectoryRetrievalModel
from online_dtw import OnlineDTWComputer
from positive_mining_sampler import PositiveMiningBatchSampler


def _z_collapse_stats(z: torch.Tensor) -> tuple[float, float]:
    """Per-batch embedding-collapse signals.

    Returns:
        z_std:            mean per-dim std across the batch (→ 0 means collapse)
        effective_rank:   exp(entropy of singular value distribution); ranges
                          from 1 (collapsed) up to retrieval_dim (full spread)
    """
    z32 = z.detach().float()
    z_std = z32.std(dim=0).mean().item()
    S = torch.linalg.svdvals(z32 - z32.mean(dim=0, keepdim=True))
    p = S / (S.sum() + 1e-12)
    eff_rank = torch.exp(-(p * torch.log(p + 1e-12)).sum()).item()
    return z_std, eff_rank


def _spearman_pred_vs_target(pred: torch.Tensor, dtw_sims: torch.Tensor) -> float:
    """Spearman rank correlation between predicted and DTW similarity matrices,
    over the strict upper-triangle of pairs in the batch. `pred` is z@z.T.
    """
    pred_ut = _upper_triangle(pred)
    tgt_ut = _upper_triangle(dtw_sims.float())
    pred_rank = pred_ut.argsort().argsort().float()
    tgt_rank = tgt_ut.argsort().argsort().float()
    pred_rank = (pred_rank - pred_rank.mean()) / (pred_rank.std() + 1e-12)
    tgt_rank = (tgt_rank - tgt_rank.mean()) / (tgt_rank.std() + 1e-12)
    return (pred_rank * tgt_rank).mean().item()


def _off_diag_stats(mat: torch.Tensor) -> tuple[float, float]:
    """Mean and std of off-diagonal entries of a square similarity matrix.

    Used for predicted z@z.T, DTW target sims, and v_ref@v_ref.T target sims —
    a near-zero std on either target is the signal that `_normalize_matrix`
    will see degenerate spread and the loss gradient on that target will be
    weak.
    """
    n = mat.shape[0]
    off_diag_mask = ~torch.eye(n, dtype=torch.bool, device=mat.device)
    vals = mat[off_diag_mask]
    return vals.mean().item(), vals.std().item()


def _batch_recall_top1(pred: torch.Tensor, target: torch.Tensor, k: int) -> float:
    """Fraction of rows where the target's top-1 neighbor (excluding self) is in
    pred's top-k neighbors (excluding self).

    A lightweight in-batch ranking proxy for eval Recall@K — same idea, but
    computed on the per-step batch against whichever target is at hand.
    Both inputs are [B, B] similarity matrices.
    """
    n = pred.shape[0]
    self_mask = torch.eye(n, dtype=torch.bool, device=pred.device)
    pred_m = pred.masked_fill(self_mask, float("-inf"))
    target_m = target.masked_fill(self_mask, float("-inf"))
    target_top1 = target_m.argmax(dim=1)
    _, pred_topk = pred_m.topk(min(k, n - 1), dim=1)
    hit = (pred_topk == target_top1.unsqueeze(1)).any(dim=1)
    return hit.float().mean().item()


def _build_eval_jepa_sim(
    eval_dataset,
    device: torch.device,
    num_workers: int,
) -> np.ndarray:
    """Compute the eval-set JEPA pairwise cosine similarity matrix once.

    `v_ref = mean-pool(jepa_features) → L2-normalise` is deterministic per clip
    (no model parameters), so the (N_eval, N_eval) similarity matrix can be
    cached at startup instead of being recomputed every eval call.

    Pooling is done inside a `JepaPooledDataset` worker so the dataloader queue
    holds only `[D]` vectors per clip — never the padded `[B, T, D]` block,
    which would blow up host RAM at the batch sizes used during training.
    """
    base, positions = _resolve_subset(eval_dataset)
    pooled_dataset = JepaPooledDataset(base, positions)
    pooled_loader = DataLoader(
        pooled_dataset,
        batch_size=512,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=False,
        drop_last=False,
    )
    print(f"Building eval JEPA similarity matrix...")
    t0 = time.perf_counter()
    all_v = []
    with torch.no_grad():
        for v in pooled_loader:
            all_v.append(v)
    V = torch.cat(all_v, dim=0).numpy().astype(np.float32)
    jepa_sim = V @ V.T
    elapsed = time.perf_counter() - t0
    print(f"  built [{jepa_sim.shape[0]}, {jepa_sim.shape[1]}] in {elapsed:.1f}s")
    return jepa_sim


def _resolve_subset(d):
    """Return (base_dataset, positions) where `positions` index into base_dataset.

    For a `Subset(dataset, positions)`, returns (dataset, list(positions)).
    For an unwrapped dataset, returns (dataset, [0..len-1]).
    """
    if isinstance(d, torch.utils.data.Subset):
        return d.dataset, list(d.indices)
    return d, list(range(len(d)))


def _build_eval_dtw_matrix(
    eval_dataset,
    dtw_computer: "OnlineDTWComputer",
    device: torch.device,
) -> np.ndarray:
    """Build the (N_eval, N_eval) DTW similarity submatrix once at startup.

    Indexed by eval-position (the order eval_dataloader yields clips with
    shuffle=False). Numerically equivalent to slicing the offline matrix at
    the same clip indices — see test_online_dtw.py for verification.
    """
    base, positions = _resolve_subset(eval_dataset)
    eval_trajs, eval_lens = base.get_trajectories_by_position(positions)
    eval_trajs = eval_trajs.to(device)
    eval_lens = eval_lens.to(device)
    print(f"Building eval DTW submatrix (length-aware): N_eval={len(positions)} ...")
    t0 = time.perf_counter()
    eval_sims = dtw_computer.compute_chunked(
        eval_trajs, eval_lens, target_chunk=1024, exact=True
    )
    elapsed = time.perf_counter() - t0
    eval_sims_cpu = eval_sims.cpu().numpy()
    print(f"  built [{eval_sims_cpu.shape[0]}, {eval_sims_cpu.shape[1]}] in {elapsed:.1f}s")
    return eval_sims_cpu


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def random_holdout_split(
    n: int,
    holdout_fraction: float,
    seed: int,
) -> tuple[list[int], list[int]]:
    """Random clip-level holdout split.

    A seeded random subset of `int(round(holdout_fraction * n))` positions is
    held out for eval; the complement is the training set. Both lists are
    sorted so dataloader iteration order is reproducible and independent of
    the internal shuffle order used during the split.

    Args:
        n:                total number of dataset positions.
        holdout_fraction: fraction of clips to hold out for eval.
        seed:             RNG seed (typically `cfg.seed`) so the split is
                          stable across runs of the same seed.

    Returns:
        (train_positions, eval_positions), each a sorted list[int].
    """
    if not 0.0 < holdout_fraction < 1.0:
        raise ValueError(
            f"eval_holdout_fraction must be in (0, 1); got {holdout_fraction}"
        )
    n_eval = int(round(holdout_fraction * n))
    rng = np.random.default_rng(seed)
    perm = rng.permutation(n)
    eval_positions = sorted(perm[:n_eval].tolist())
    train_positions = sorted(perm[n_eval:].tolist())
    return train_positions, eval_positions


def initialize_query_token(
    model: TrajectoryRetrievalModel,
    dataloader: DataLoader,
    device: torch.device,
    num_batches: int,
    input_mode: str = "jepa",
) -> None:
    """Initialise the query token from the running mean of real patch tokens.

    Accumulates a streaming sum + count across `num_batches` batches without
    holding all patch embeddings on GPU at once (avoids OOM on long clips).
    In trajectory mode the "patch tokens" are the per-frame trajectory features.
    """
    model.eval()
    D = model.head.query.shape[-1]
    sum_patches = torch.zeros(D, device=device, dtype=torch.float32)
    n_patches = 0
    with torch.no_grad():
        for i, batch in enumerate(dataloader):
            if i >= num_batches:
                break
            # Keep everything in the input dtype on GPU; only the tiny [D]
            # reduction is fp32. Multiplying with a same-dtype mask avoids an
            # fp32 materialisation of the full [B, T, D] tensor (which costs
            # ~10 GB at B=256, T~8640 for JEPA features).
            feats, mask = select_model_input(batch, input_mode, device)
            feats = model.head.project_input(feats)
            mask_h = mask.unsqueeze(-1).to(feats.dtype)
            # Sum reduction in fp32 to avoid fp16 overflow (V-JEPA features
            # aren't zero-centered; accumulating millions of values overflows).
            sum_patches += (feats * mask_h).sum(dim=(0, 1), dtype=torch.float32)
            n_patches += int(mask.sum().item())
            del feats, mask, mask_h
    if not torch.isfinite(sum_patches).all():
        raise RuntimeError("Non-finite values in query-init sum; check feature dtype/range.")
    if n_patches > 0:
        mean = (sum_patches / n_patches).unsqueeze(0)
        with torch.no_grad():
            model.head.query.copy_(mean.to(model.head.query.dtype))
        print(f"Query token initialised from {n_patches} patch embeddings")
    # Free the init-time pool so it doesn't carry into training.
    torch.cuda.empty_cache()


def build_optimizer_param_groups(
    model: TrajectoryRetrievalModel,
    lr: float,
    prototype_lr_multiplier: float,
) -> list[dict]:
    """Two-group split: head params at `lr`, DTW prototypes at `lr * multiplier`.

    Collapses to a single group when `model` has no prototype parameters (i.e.
    `dtw_nn_layer=False`), preserving the prior single-group behavior.
    """
    head_params = list(model.get_non_prototype_parameters())
    proto_params = list(model.get_prototype_parameters())
    groups = [{"params": head_params, "lr": lr}]
    if proto_params:
        groups.append({"params": proto_params, "lr": lr * prototype_lr_multiplier})
    return groups


def save_checkpoint(
    model: TrajectoryRetrievalModel,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler._LRScheduler,
    scaler: torch.amp.GradScaler,
    cfg: Config,
    epoch: int,
    global_step: int,
    best_metric: float,
    path: Path,
) -> None:
    torch.save(
        {
            "head_state_dict": model.head.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "scaler_state_dict": scaler.state_dict(),
            "config": cfg.__dict__,
            "epoch": epoch,
            "global_step": global_step,
            "best_metric": best_metric,
        },
        path,
    )


@contextmanager
def _preserve_rng_state(device: torch.device):
    """Prevent an instrumentation-only eval pass from changing training RNG."""
    python_state = random.getstate()
    numpy_state = np.random.get_state()
    cuda_devices = []
    if device.type == "cuda" and torch.cuda.is_available():
        cuda_devices = [
            device.index if device.index is not None else torch.cuda.current_device()
        ]
    try:
        with torch.random.fork_rng(devices=cuda_devices):
            yield
    finally:
        random.setstate(python_state)
        np.random.set_state(numpy_state)


def _run_and_log_evaluation(
    *, model, eval_dataloader, eval_dtw_matrix, eval_jepa_sim,
    cfg, device, criterion, dtw_computer,
    writer, epoch: int, trajectory_mode: bool,
    global_candidate_dataloader=None, global_dtw_neighbor_clip_idx=None,
    train_pool_query_clip_idx=None, train_pool_dtw_neighbor_clip_idx=None,
):
    """Run one evaluation and write its complete TensorBoard epoch point."""
    metrics = run_evaluation(
        model, eval_dataloader, eval_dtw_matrix,
        eval_jepa_sim,
        cfg, device,
        criterion=criterion,
        dtw_computer=dtw_computer,
        global_candidate_dataloader=global_candidate_dataloader,
        global_dtw_neighbor_clip_idx=global_dtw_neighbor_clip_idx,
        train_pool_query_clip_idx=train_pool_query_clip_idx,
        train_pool_dtw_neighbor_clip_idx=train_pool_dtw_neighbor_clip_idx,
    )
    for key, value in metrics.items():
        writer.add_scalar(f"eval/{key}", value, epoch)

    if trajectory_mode:
        selection_metric = metrics[f"cknna_dtw@{cfg.cknna_topk}"]
        writer.add_scalar("eval/selection_cknna_dtw", selection_metric, epoch)
        selection_desc = f"CKNNA DTW={selection_metric:.4f}"
    else:
        r5_dtw = metrics.get("R@5", 0.0)
        r5_jepa = metrics.get("jepa_R@5", 0.0)
        selection_metric = (
            2.0 * r5_dtw * r5_jepa / (r5_dtw + r5_jepa + 1e-8)
        )
        writer.add_scalar("eval/balanced_R@5", selection_metric, epoch)
        selection_desc = (f"balanced_R@5={selection_metric:.4f} "
                          f"(DTW R@5={r5_dtw:.4f}, JEPA R@5={r5_jepa:.4f})")
    return metrics, selection_metric, selection_desc


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config")
    parser.add_argument("--output-dir", type=str, default=None,
                        help="Override output directory from config")
    args = parser.parse_args()

    cfg = Config.from_yaml(args.config)
    set_rank_mode = cfg.loss_type == "topk_set_rank"
    if cfg.loss_type not in ("soft_contrastive", "topk_set_rank"):
        raise ValueError(f"Unknown loss_type={cfg.loss_type!r}")
    if set_rank_mode:
        if cfg.input_mode != "trajectory":
            raise ValueError("topk_set_rank currently supports trajectory input mode only")
        if not cfg.use_positive_mining:
            raise ValueError("topk_set_rank requires use_positive_mining=True")
        if cfg.distribution_loss_weight != 0.0:
            raise ValueError("topk_set_rank requires distribution_loss_weight=0")
        if cfg.set_loss_boundary_start != cfg.set_loss_top_k + 1:
            raise ValueError(
                "set_loss_boundary_start must immediately follow set_loss_top_k"
            )
        if cfg.set_loss_boundary_end < cfg.set_loss_boundary_start:
            raise ValueError("invalid set-loss boundary rank range")
        if cfg.set_loss_hard_negatives < cfg.boundary_negatives_per_anchor:
            raise ValueError("hard-negative budget must include explicit boundaries")
        group_size = 1 + cfg.positives_per_anchor + cfg.boundary_negatives_per_anchor
        if cfg.batch_size % group_size != 0:
            raise ValueError(
                f"batch_size={cfg.batch_size} must be divisible by set/rank group_size={group_size}"
            )
    if cfg.dtw_nn_layer:
        if cfg.input_mode != "trajectory":
            raise ValueError("dtw_nn_layer=True requires input_mode='trajectory'")
        if cfg.dtw_nn_num_nodes < 1:
            raise ValueError(f"dtw_nn_num_nodes must be >= 1, got {cfg.dtw_nn_num_nodes}")
        if cfg.dtw_nn_prototype_length < 1:
            raise ValueError(
                f"dtw_nn_prototype_length must be >= 1, got {cfg.dtw_nn_prototype_length}"
            )
        # Asymmetric slope constraint: prototype no longer than 2T. Trajectory T
        # is hardcoded to 42 by build_trajectories.py:max_t — independent of
        # cfg.max_clip_frames, which only affects JEPA-mode truncation.
        TRAJECTORY_T_MAX = 42  # keep in sync with build_trajectories.py default `max_t`
        max_proto = 2 * TRAJECTORY_T_MAX
        if cfg.dtw_nn_prototype_length > max_proto:
            raise ValueError(
                f"dtw_nn_prototype_length={cfg.dtw_nn_prototype_length} exceeds "
                f"the asymmetric-slope cap 2*T_max={max_proto}"
            )
        if cfg.dtw_nn_gamma <= 0:
            raise ValueError(f"dtw_nn_gamma must be > 0, got {cfg.dtw_nn_gamma}")
        if cfg.dtw_nn_prototype_lr_multiplier <= 0:
            raise ValueError(
                f"dtw_nn_prototype_lr_multiplier must be > 0, "
                f"got {cfg.dtw_nn_prototype_lr_multiplier}"
            )
    set_seed(cfg.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    run_dir = Path(args.output_dir if args.output_dir else cfg.output_dir)

    print("=" * 60)
    print("CONFIG")
    print("=" * 60)
    print(f"  config_file: {args.config!r}")
    print(f"  run_dir: {str(run_dir)!r}")
    for f in fields(cfg):
        print(f"  {f.name}: {getattr(cfg, f.name)!r}")
    print("=" * 60)
    run_dir.mkdir(parents=True, exist_ok=True)

    # -- Trajectory artifact (built once per run, design-keyed filename) --
    traj_path = run_dir / f"trajectories_{cfg.dtw_design}.pt"
    if not traj_path.exists():
        print(f"Building trajectories for design={cfg.dtw_design} -> {traj_path}")
        import sys as _sys
        _bd = str(Path(__file__).resolve().parent / "build_dataset")
        if _bd not in _sys.path:
            _sys.path.insert(0, _bd)
        from build_trajectories import build_trajectories
        build_trajectories(
            data_dir=Path(cfg.data_dir),
            dtw_design=cfg.dtw_design,
            output_path=traj_path,
            max_clips=cfg.max_total_clips if cfg.max_total_clips > 0 else None,
            clip_subsample_seed=cfg.seed,
            use_depth_grounded=cfg.use_depth_grounded_keypoints,
            require_jepa_features=cfg.input_mode == "jepa",
        )
    else:
        print(f"Reusing trajectories at {traj_path}")

    # -- DTW-neighbor cache (built once per run, mirrors trajectory artifact).
    # Stored in run_dir, NOT data_dir — recomputed every training run. Order
    # matters: `traj_path` must exist first (used as input by the builder).
    need_dtw_neighbor_cache = cfg.use_positive_mining or cfg.eval_global_cknna
    neigh_path = run_dir / f"dtw_neighbors_{cfg.dtw_design}.pt"
    if need_dtw_neighbor_cache:
        if not neigh_path.exists():
            print(f"Building DTW neighbors for design={cfg.dtw_design} -> {neigh_path}")
            import sys as _sys
            _bd = str(Path(__file__).resolve().parent / "build_dataset")
            if _bd not in _sys.path:
                _sys.path.insert(0, _bd)
            from build_dtw_neighbors import build_dtw_neighbors
            build_dtw_neighbors(
                trajectories_path=traj_path,
                dtw_design=cfg.dtw_design,
                output_path=neigh_path,
                top_k=max(
                    cfg.positive_mining_top_k if cfg.use_positive_mining else 0,
                    cfg.cknna_topk if cfg.eval_global_cknna else 0,
                    max(cfg.recall_k_values) if cfg.eval_global_cknna else 0,
                    max(cfg.cknna_k_values) if cfg.eval_global_cknna else 0,
                    cfg.set_loss_boundary_end if set_rank_mode else 0,
                ),
                candidate_k=cfg.positive_mining_candidate_k,
                pooling=cfg.positive_mining_pooling,
            )
        else:
            print(f"Reusing DTW neighbors at {neigh_path}")

    # -- Input mode --
    # "jepa": head consumes frozen V-JEPA features; "trajectory": head consumes
    # the per-design hand trajectory itself and trains on DTW similarity alone.
    trajectory_mode = cfg.input_mode == "trajectory"
    if cfg.input_mode not in ("jepa", "trajectory"):
        raise ValueError(
            f"Unknown input_mode={cfg.input_mode!r}; expected 'jepa' or 'trajectory'"
        )

    # -- Dataset --
    dataset = VideoClipDataset(
        cfg.data_dir,
        dtw_design=cfg.dtw_design,
        max_clip_frames=cfg.max_clip_frames,
        trajectories_path=str(traj_path),
        load_jepa_features=not trajectory_mode,
    )

    # -- Held-out split --
    # Random clip-level holdout, seeded by `cfg.seed` so it's stable across
    # runs of the same seed (and reproducible from the saved config). The
    # model never sees eval clips during training.
    train_positions, eval_positions = random_holdout_split(
        len(dataset), cfg.eval_holdout_fraction, cfg.seed,
    )
    print(
        f"Holdout split (seed={cfg.seed}, holdout={cfg.eval_holdout_fraction:.2f}): "
        f"train {len(train_positions)} clips | eval {len(eval_positions)} clips"
    )
    neigh = None
    if need_dtw_neighbor_cache:
        neigh = torch.load(neigh_path, weights_only=False, map_location="cpu")
        if neigh.get("design") != cfg.dtw_design:
            raise RuntimeError(
                f"DTW neighbor cache design mismatch: "
                f"file={neigh.get('design')!r} vs cfg={cfg.dtw_design!r}"
            )
        required_topk = max(
            cfg.positives_per_anchor if cfg.use_positive_mining else 0,
            cfg.cknna_topk if cfg.eval_global_cknna else 0,
            max(cfg.recall_k_values) if cfg.eval_global_cknna else 0,
            max(cfg.cknna_k_values) if cfg.eval_global_cknna else 0,
            cfg.set_loss_boundary_end if set_rank_mode else 0,
        )
        if neigh.get("top_k", 0) < required_topk:
            raise RuntimeError(
                f"Cache top_k={neigh.get('top_k')} < required {required_topk}; "
                "rebuild the DTW neighbor cache with a higher --top-k."
            )
        if neigh.get("pooling") != cfg.positive_mining_pooling:
            raise RuntimeError(
                f"DTW neighbor cache pooling mismatch: "
                f"file={neigh.get('pooling')!r} vs "
                f"cfg={cfg.positive_mining_pooling!r}"
            )
        neighbor_table = neigh.get("neighbor_clip_idx")
        if (
            not isinstance(neighbor_table, torch.Tensor)
            or neighbor_table.ndim != 2
            or neighbor_table.shape[0] != len(dataset)
        ):
            shape = getattr(neighbor_table, "shape", None)
            raise RuntimeError(
                f"DTW neighbor cache row mismatch: table shape={shape}, "
                f"dataset N={len(dataset)}"
            )
        neighbor_dtw_sim = neigh.get("neighbor_dtw_sim")
        if (
            not isinstance(neighbor_dtw_sim, torch.Tensor)
            or neighbor_dtw_sim.shape != neighbor_table.shape
        ):
            shape = getattr(neighbor_dtw_sim, "shape", None)
            raise RuntimeError(
                f"DTW neighbor similarity table mismatch: sims={shape}, "
                f"ids={tuple(neighbor_table.shape)}"
            )

    sampler = None  # only set when use_positive_mining=True; used for set_epoch
    if cfg.use_positive_mining:
        # Per-clip-idx video number, vectorised once for the same-video filter.
        clip_video_number = torch.tensor(
            [dataset.clip_index[i]["video_number"]
             for i in range(len(dataset.clip_index))],
            dtype=torch.int32,
        )
        # Inverse of `clip_idx_at`: raw clip_idx -> dataset position.
        position_of_clip_idx = {
            dataset.clip_idx_at(p): p for p in range(len(dataset))
        }

        sampler = PositiveMiningBatchSampler(
            train_positions=train_positions,
            clip_idx_at=dataset.clip_idx_at,
            position_of_clip_idx=position_of_clip_idx,
            neighbor_clip_idx=neigh["neighbor_clip_idx"],
            clip_video_number=clip_video_number,
            batch_size=cfg.batch_size,
            positives_per_anchor=cfg.positives_per_anchor,
            exclude_same_video_positives=cfg.exclude_same_video_positives,
            boundary_negatives_per_anchor=(
                cfg.boundary_negatives_per_anchor if set_rank_mode else 0
            ),
            positive_rank_end=(cfg.set_loss_top_k if set_rank_mode else None),
            boundary_rank_start=cfg.set_loss_boundary_start,
            boundary_rank_end=cfg.set_loss_boundary_end,
            pad_policy=cfg.positive_mining_pad_policy,
            drop_last=True,
            seed=cfg.seed,
        )
        # NB: pass the **base** dataset, not Subset — sampler emits base-dataset
        # positions directly. `batch_sampler` is mutually exclusive with
        # `batch_size`, `shuffle`, `sampler`, and `drop_last`, so omit those.
        dataloader = DataLoader(
            dataset,
            batch_sampler=sampler,
            num_workers=cfg.num_workers,
            collate_fn=collate_fn,
            pin_memory=cfg.pin_memory,
            prefetch_factor=cfg.prefetch_factor if cfg.num_workers > 0 else None,
            persistent_workers=cfg.persistent_workers and cfg.num_workers > 0,
        )
        print(
            f"Positive-mining sampler: G={sampler.anchors_per_batch} anchors "
            f"× group_size={sampler.group_size} clips per batch "
            f"(P={sampler.positives_per_anchor}, "
            f"boundary={sampler.boundary_negatives_per_anchor}) | "
            f"{len(sampler)} batches/epoch | "
            f"exclude_same_video={cfg.exclude_same_video_positives}, "
            f"pad_policy={cfg.positive_mining_pad_policy}"
        )
    else:
        train_dataset = Subset(dataset, train_positions)
        dataloader = DataLoader(
            train_dataset,
            batch_size=cfg.batch_size,
            shuffle=True,
            num_workers=cfg.num_workers,
            collate_fn=collate_fn,
            pin_memory=cfg.pin_memory,
            drop_last=True,
            prefetch_factor=cfg.prefetch_factor if cfg.num_workers > 0 else None,
            persistent_workers=cfg.persistent_workers and cfg.num_workers > 0,
        )

    # -- Eval dataset/dataloader --
    # Eval is over the held-out subset; if it's larger than `eval_max_clips`,
    # subsample at random (seeded so the subsample is fixed across eval calls
    # within a run). shuffle=False so the same clips are encoded in the same
    # order every epoch.
    if 0 < cfg.eval_max_clips < len(eval_positions):
        rng = np.random.default_rng(cfg.seed)
        eval_positions = sorted(rng.choice(
            eval_positions, size=cfg.eval_max_clips, replace=False,
        ).tolist())
        print(f"Eval subsample: {len(eval_positions)} clips "
              f"(capped via eval_max_clips, seed={cfg.seed})")
    eval_dataset = Subset(dataset, eval_positions)
    eval_dataloader = DataLoader(
        eval_dataset,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        collate_fn=collate_fn,
        pin_memory=cfg.pin_memory,
        drop_last=False,
        prefetch_factor=cfg.prefetch_factor if cfg.num_workers > 0 else None,
    )
    global_candidate_dataloader = None
    global_dtw_neighbor_clip_idx = None
    train_pool_query_clip_idx = None
    train_pool_dtw_neighbor_clip_idx = None
    if cfg.eval_global_cknna:
        eval_position_set = set(eval_positions)
        global_remainder_positions = [
            position for position in range(len(dataset))
            if position not in eval_position_set
        ]
        global_candidate_dataloader = DataLoader(
            Subset(dataset, global_remainder_positions),
            batch_size=cfg.batch_size,
            shuffle=False,
            num_workers=cfg.num_workers,
            collate_fn=collate_fn,
            pin_memory=cfg.pin_memory,
            drop_last=False,
            prefetch_factor=cfg.prefetch_factor if cfg.num_workers > 0 else None,
        )
        global_dtw_neighbor_clip_idx = neigh["neighbor_clip_idx"].numpy()
        print(
            f"Global-pool CKNNA enabled: {len(eval_positions)} eval queries | "
            f"{len(dataset)} total candidates | cached DTW top-{neigh['top_k']} "
            f"from candidate_k={neigh['candidate_k']}, pooling={neigh['pooling']}"
        )

        # -- Train-pool CKNNA: 10k random train queries vs the 90k train corpus.
        # Mirrors the global block but restricted to train-side data; the gap
        # between train-pool and global-pool CKNNA is the canonical
        # overfitting signature on the DTW geometry.
        train_pool_topk = max(
            cfg.cknna_topk, max(cfg.recall_k_values), max(cfg.cknna_k_values)
        )
        train_neigh_path = (
            run_dir / f"dtw_neighbors_{cfg.dtw_design}_train_pool.pt"
        )
        if not train_neigh_path.exists():
            print(
                f"Building train-pool DTW neighbors for design={cfg.dtw_design} "
                f"-> {train_neigh_path}"
            )
            from build_dtw_neighbors import build_train_pool_neighbors
            build_train_pool_neighbors(
                trajectories_path=traj_path,
                train_positions=train_positions,
                dtw_design=cfg.dtw_design,
                output_path=train_neigh_path,
                top_k=train_pool_topk,
                candidate_k=cfg.positive_mining_candidate_k,
                pooling=cfg.positive_mining_pooling,
            )
        else:
            print(f"Reusing train-pool DTW neighbors at {train_neigh_path}")
        train_neigh = torch.load(
            train_neigh_path, weights_only=False, map_location="cpu",
        )
        if train_neigh.get("design") != cfg.dtw_design:
            raise RuntimeError(
                f"Train-pool DTW neighbor cache design mismatch: "
                f"file={train_neigh.get('design')!r} vs cfg={cfg.dtw_design!r}"
            )
        if int(train_neigh.get("top_k", 0)) < train_pool_topk:
            raise RuntimeError(
                f"Train-pool DTW neighbor cache top_k={train_neigh.get('top_k')} "
                f"< required {train_pool_topk}"
            )
        train_pool_dtw_neighbor_clip_idx = train_neigh["neighbor_clip_idx"].numpy()

        train_pool_query_size = min(cfg.eval_max_clips, len(train_positions))
        train_pool_rng = np.random.default_rng(cfg.seed + 1)
        train_pool_query_clip_idx = np.asarray(
            sorted(train_pool_rng.choice(
                train_positions, size=train_pool_query_size, replace=False,
            ).tolist()),
            dtype=np.int64,
        )
        print(
            f"Train-pool CKNNA enabled: {len(train_pool_query_clip_idx)} train "
            f"queries | {len(train_positions)} train candidates | "
            f"top_k={train_pool_topk}"
        )

    # -- Model --
    # In trajectory mode the head's input dim is the trajectory feature dim D
    # (varies by dtw_design), not the YAML's JEPA-specific encoder_dim (1408).
    if trajectory_mode:
        effective_encoder_dim = int(dataset.trajectories.shape[-1])
        attention_dim = cfg.trajectory_model_dim
        print(f"Trajectory input mode: encoder_dim={effective_encoder_dim}, "
              f"attention_dim={attention_dim} "
              f"(derived from trajectory feature dim D), "
              f"positional_encoding={cfg.use_trajectory_positional_encoding}")
    else:
        effective_encoder_dim = cfg.encoder_dim
        attention_dim = None
    # DTW-NN head: derive a Gaussian-init std from the actual
    # trajectory feature range so prototypes start in the same numeric scale
    # as the inputs.
    if cfg.dtw_nn_layer:
        dtw_nn_input_std = float(dataset.trajectories.std().clamp(min=1e-6))
        print(
            f"DTW-NN head: num_nodes={cfg.dtw_nn_num_nodes}, "
            f"prototype_length={cfg.dtw_nn_prototype_length}, "
            f"gamma={cfg.dtw_nn_gamma}, "
            f"input_std={dtw_nn_input_std:.4f}, "
            f"lr_mult={cfg.dtw_nn_prototype_lr_multiplier}"
        )
        # Fields below become vestigial when the DTW head is on; the
        # CrossAttentionProjectionHead constructor ignores them on that path.
        print(
            "  [info] dtw_nn_layer=True ignores: attention_dim, num_cross_attn_*, "
            "num_queries, query_init_batches, use_grad_checkpoint, "
            "use_trajectory_positional_encoding"
        )
    else:
        dtw_nn_input_std = 1.0

    model = TrajectoryRetrievalModel(
        encoder_dim=effective_encoder_dim,
        retrieval_dim=cfg.retrieval_dim,
        num_heads=cfg.num_cross_attn_heads,
        num_projection_layers=cfg.num_projection_layers,
        num_cross_attn_layers=cfg.num_cross_attn_layers,
        use_grad_checkpoint=cfg.use_grad_checkpoint,
        use_positional_encoding=trajectory_mode and cfg.use_trajectory_positional_encoding,
        attention_dim=attention_dim,
        num_queries=cfg.num_queries,
        dtw_nn_layer=cfg.dtw_nn_layer,
        dtw_nn_num_nodes=cfg.dtw_nn_num_nodes,
        dtw_nn_prototype_length=cfg.dtw_nn_prototype_length,
        dtw_nn_gamma=cfg.dtw_nn_gamma,
        dtw_nn_input_std=dtw_nn_input_std,
    ).to(device)
    if cfg.use_grad_checkpoint:
        print(f"Gradient checkpointing enabled on {cfg.num_cross_attn_layers} cross-attn block(s)")

    # -- Query init --
    if cfg.dtw_nn_layer:
        print("Skipping query init (dtw_nn_layer=True; no learnable query token)")
    else:
        print("Initialising query token from precomputed features...")
        initialize_query_token(
            model, dataloader, device, cfg.query_init_batches, input_mode=cfg.input_mode,
        )

    # -- Optimizer & scheduler --
    # DTW-NN prototypes use a separate optimizer group with a configurable
    # learning-rate multiplier. When dtw_nn_layer=False the prototype iterator
    # is empty, so this collapses to a single param group identical to the
    # prior behaviour.
    param_groups = build_optimizer_param_groups(
        model, lr=cfg.lr, prototype_lr_multiplier=cfg.dtw_nn_prototype_lr_multiplier,
    )
    if len(param_groups) > 1:
        n_head = sum(p.numel() for p in param_groups[0]["params"])
        n_proto = sum(p.numel() for p in param_groups[1]["params"])
        print(
            f"Optimizer: 2 param groups | head lr={param_groups[0]['lr']:.2e} ({n_head} params) "
            f"| prototypes lr={param_groups[1]['lr']:.2e} ({n_proto} params)"
        )
    optimizer = torch.optim.AdamW(
        param_groups,
        lr=cfg.lr,
        weight_decay=cfg.weight_decay,
    )
    steps_per_epoch = (
        min(cfg.max_steps_per_epoch, len(dataloader))
        if cfg.max_steps_per_epoch > 0
        else len(dataloader)
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=cfg.num_epochs * steps_per_epoch
        )
    scaler = torch.amp.GradScaler("cuda", enabled=cfg.mixed_precision)

    # In trajectory mode there is no V-JEPA semantic anchor (v_ref is a pooled
    # trajectory, not a semantic embedding). The set/rank loss consumes
    # cached global neighbor IDs/similarities and omits the legacy
    # batch-DTW distribution term.
    loss_components = "trajectory" if trajectory_mode else cfg.loss_components
    if set_rank_mode:
        criterion = TopKSetRankLoss(
            neigh["neighbor_clip_idx"], neigh["neighbor_dtw_sim"],
            positive_top_k=cfg.set_loss_top_k,
            positives_per_anchor=cfg.positives_per_anchor,
            boundary_negatives_per_anchor=cfg.boundary_negatives_per_anchor,
            hard_negatives=cfg.set_loss_hard_negatives,
            set_weight=cfg.set_loss_weight,
            rank_weight=cfg.rank_loss_weight,
            margin=cfg.set_loss_margin,
            set_temperature=cfg.set_loss_temperature,
            rank_temperature=cfg.rank_loss_temperature,
        ).to(device)
        print(
            f"Loss: topk_set_rank | top_k={cfg.set_loss_top_k}, "
            f"boundary={cfg.set_loss_boundary_start}-{cfg.set_loss_boundary_end}, "
            f"hard_negatives={cfg.set_loss_hard_negatives}, "
            f"weights=(set={cfg.set_loss_weight}, rank={cfg.rank_loss_weight}, "
            f"distribution={cfg.distribution_loss_weight})"
        )
    else:
        criterion = CombinedLoss(
            lambda_preserve=cfg.lambda_preserve,
            loss_type=cfg.loss_type,
            tau_pred=cfg.loss_tau_pred,
            tau_target=cfg.loss_tau_target,
            components=loss_components,
        )
        print(f"Loss: {cfg.loss_type} | components={loss_components} "
              f"(lambda_preserve={cfg.lambda_preserve}, "
              f"tau_pred={cfg.loss_tau_pred}, tau_target={cfg.loss_tau_target})"
              + (" [trajectory mode: semantic term disabled]" if trajectory_mode else ""))
    eval_criterion = None if set_rank_mode else criterion
    dtw_computer = OnlineDTWComputer(cfg.dtw_design)
    print(f"Online DTW computer initialised for design={cfg.dtw_design}")

    # -- Build the eval-set DTW similarity matrix once (replaces the offline
    #    `dtw_matrix_{design}.npy` consumed by run_evaluation).
    eval_dtw_matrix = _build_eval_dtw_matrix(eval_dataset, dtw_computer, device)
    # The JEPA semantic eval axis is meaningless in trajectory mode (no V-JEPA
    # input), and JepaPooledDataset would load features from disk we don't use.
    if trajectory_mode:
        eval_jepa_sim = None
        print("Trajectory input mode: skipping eval JEPA similarity matrix "
              "(model selection uses CKNNA DTW alone).")
    else:
        eval_jepa_sim = _build_eval_jepa_sim(
            eval_dataset, device, cfg.num_workers,
        )
    # Headline metric: harmonic mean of DTW R@5 and JEPA R@5. Decoupled from
    # `lambda_preserve` (which controls gradient mechanics) — the selector
    # encodes the *goal* directly: a good embedding space preserves *both*
    # retrievals, and the harmonic mean penalises trading one off for the other
    # (drops sharply if either component is small).
    best_selection_metric = 0.0
    # Write TB events under <parent_output_dir>/tb_runs/<run_basename>/. This
    # consolidates events across runs so you can `tensorboard --logdir <parent>/tb_runs`
    # and see each run as a clean, named entry (e.g. "bs512_lr2e4_2026-04-29_19-00").
    tb_root = run_dir.parent / "tb_runs"
    tb_log_dir = tb_root / run_dir.name
    tb_log_dir.mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(log_dir=str(tb_log_dir))
    print(f"TensorBoard log dir: {tb_log_dir}")
    print(f"  → tensorboard --logdir {tb_root}")
    global_step = 0

    # Baseline for the exact initialized state that enters optimization: the
    # query token is data-initialized, but no parameter update has occurred.
    # This is metrics-only and deliberately cannot become best_model.pt.
    print("Epoch 0 evaluation (after query initialization, before optimizer steps)...")
    with _preserve_rng_state(device):
        _, _, epoch0_desc = _run_and_log_evaluation(
            model=model,
            eval_dataloader=eval_dataloader,
            eval_dtw_matrix=eval_dtw_matrix,
            eval_jepa_sim=eval_jepa_sim,
            cfg=cfg,
            device=device,
            criterion=eval_criterion,
            dtw_computer=dtw_computer,
            writer=writer,
            epoch=0,
            trajectory_mode=trajectory_mode,
            global_candidate_dataloader=global_candidate_dataloader,
            global_dtw_neighbor_clip_idx=global_dtw_neighbor_clip_idx,
            train_pool_query_clip_idx=train_pool_query_clip_idx,
            train_pool_dtw_neighbor_clip_idx=train_pool_dtw_neighbor_clip_idx,
        )
    writer.flush()
    print(f"Epoch 0 baseline: {epoch0_desc} (metrics only; no checkpoint)")

    # -- Training loop --
    for epoch in range(1, cfg.num_epochs + 1):
        model.train()
        epoch_loss_keys = (
            ("total", "set", "rank")
            if set_rank_mode else
            ("total", "trajectory", "semantic")
        )
        epoch_losses = {key: 0.0 for key in epoch_loss_keys}
        num_steps = 0
        prof = EpochProfiler()
        epoch_t0 = time.perf_counter()

        # Reseed the positive-mining sampler per-epoch (mirrors DistributedSampler).
        if sampler is not None:
            sampler.set_epoch(epoch)

        # Time the wait between yielded batches as "data_load"; everything else
        # inside the iteration is a labeled section.
        data_iter = iter(dataloader)
        step = 0
        while True:
            t_data = time.perf_counter()
            try:
                batch = next(data_iter)
            except StopIteration:
                break
            prof.totals["data_load"] += time.perf_counter() - t_data
            prof.counts["data_load"] += 1
            step += 1

            with prof.section("h2d_transfer"):
                # Model input per the configured mode; autocast handles precision
                # inside the model. In trajectory mode `feats` is the trajectory
                # tensor itself (same source as the DTW target below).
                feats, mask = select_model_input(batch, cfg.input_mode, device)
                if not set_rank_mode:
                    trajs = batch["trajectories"].to(device, non_blocking=True)
                    traj_lens = batch["traj_lengths"].to(device, non_blocking=True)

            if not set_rank_mode:
                dtw_t0 = time.perf_counter()
                # Online per-batch DTW similarity matrix (fp32, autocast-disabled
                # to preserve numerical equivalence with the offline matrix).
                with torch.amp.autocast("cuda", enabled=False):
                    dtw_sims = dtw_computer.compute(trajs, traj_lens)
                prof.totals["dtw_compute"] += time.perf_counter() - dtw_t0
                prof.counts["dtw_compute"] += 1

            with prof.section("forward+loss"):
                with torch.amp.autocast("cuda", enabled=cfg.mixed_precision):
                    out = model(feats, mask)
                    if set_rank_mode:
                        losses = criterion(
                            out["z"],
                            batch["clip_indices"].to(device, non_blocking=True),
                        )
                    else:
                        losses = criterion(out["z"], out["v_ref"], dtw_sims)

            with prof.section("backward+step"):
                scaler.scale(losses["total"]).backward()
                scaler.unscale_(optimizer)
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    model.get_trainable_parameters(), cfg.grad_clip_norm
                )
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
                scheduler.step()
            global_step += 1

            for k in epoch_losses:
                epoch_losses[k] += losses[k].item()
            num_steps += 1

            if cfg.max_steps_per_epoch > 0 and step >= cfg.max_steps_per_epoch:
                break

            if step % cfg.log_frequency == 0:
                lr = scheduler.get_last_lr()[0]
                pred = (out["z"] @ out["z"].T).detach().float()
                z_std, eff_rank = _z_collapse_stats(out["z"])
                pred_mean, pred_std = _off_diag_stats(pred)
                writer.add_scalar("train/loss_total", losses["total"].item(), global_step)
                writer.add_scalar("train/lr", lr, global_step)
                writer.add_scalar("train/grad_norm", grad_norm.item(), global_step)
                writer.add_scalar("train/z_std", z_std, global_step)
                writer.add_scalar("train/z_effective_rank", eff_rank, global_step)
                writer.add_scalar("train/pred_sim_mean", pred_mean, global_step)
                writer.add_scalar("train/pred_sim_std", pred_std, global_step)

                if set_rank_mode:
                    for key in (
                        "set", "rank", "set_violation_rate",
                        "rank_violation_rate", "positive_hard_negative_margin",
                    ):
                        writer.add_scalar(f"train/{key}", losses[key].item(), global_step)
                    msg = (
                        f"  epoch {epoch} step {step}/{len(dataloader)} | "
                        f"loss={losses['total'].item():.4f} "
                        f"(set={losses['set'].item():.4f} rank={losses['rank'].item():.4f}) "
                        f"viol=(set={losses['set_violation_rate'].item():.3f} "
                        f"rank={losses['rank_violation_rate'].item():.3f}) "
                        f"margin={losses['positive_hard_negative_margin'].item():+.3f} "
                        f"lr={lr:.2e} | z_std={z_std:.3f} eff_rank={eff_rank:.1f}"
                    )
                else:
                    dtw_sims_f = dtw_sims.float()
                    dtw_mean, dtw_std = _off_diag_stats(dtw_sims_f)
                    H_traj = target_entropy_nats(dtw_sims_f, cfg.loss_tau_target)
                    kl_gap_traj = losses["trajectory"].item() - H_traj
                    rho_dtw = _spearman_pred_vs_target(pred, dtw_sims_f)
                    br1_dtw = _batch_recall_top1(pred, dtw_sims_f, k=1)
                    br5_dtw = _batch_recall_top1(pred, dtw_sims_f, k=5)
                    writer.add_scalar("train/loss_trajectory", losses["trajectory"].item(), global_step)
                    writer.add_scalar("train/H_target_traj", H_traj, global_step)
                    writer.add_scalar("train/KL_gap_traj", kl_gap_traj, global_step)
                    writer.add_scalar("train/spearman_pred_vs_dtw", rho_dtw, global_step)
                    writer.add_scalar("train/dtw_target_mean", dtw_mean, global_step)
                    writer.add_scalar("train/dtw_target_std", dtw_std, global_step)
                    writer.add_scalar("train/batch_recall@1_dtw", br1_dtw, global_step)
                    writer.add_scalar("train/batch_recall@5_dtw", br5_dtw, global_step)
                    msg = (
                        f"  epoch {epoch} step {step}/{len(dataloader)} | "
                        f"loss={losses['total'].item():.4f} "
                        f"(traj={losses['trajectory'].item():.4f}) "
                        f"KL_gap_traj={kl_gap_traj:+.3f} lr={lr:.2e} | "
                        f"rho_dtw={rho_dtw:+.3f} z_std={z_std:.3f} "
                        f"eff_rank={eff_rank:.1f}"
                    )
                    if not trajectory_mode:
                        jepa_sim = (out["v_ref"] @ out["v_ref"].T).detach().float()
                        jepa_mean, jepa_std = _off_diag_stats(jepa_sim)
                        H_sem = target_entropy_nats(jepa_sim, cfg.loss_tau_target)
                        kl_gap_sem = losses["semantic"].item() - H_sem
                        rho_jepa = _spearman_pred_vs_target(pred, jepa_sim)
                        br1_jepa = _batch_recall_top1(pred, jepa_sim, k=1)
                        br5_jepa = _batch_recall_top1(pred, jepa_sim, k=5)
                        writer.add_scalar("train/loss_semantic", losses["semantic"].item(), global_step)
                        writer.add_scalar("train/H_target_sem", H_sem, global_step)
                        writer.add_scalar("train/KL_gap_sem", kl_gap_sem, global_step)
                        writer.add_scalar("train/spearman_pred_vs_jepa", rho_jepa, global_step)
                        writer.add_scalar("train/jepa_target_mean", jepa_mean, global_step)
                        writer.add_scalar("train/jepa_target_std", jepa_std, global_step)
                        writer.add_scalar("train/batch_recall@1_jepa", br1_jepa, global_step)
                        writer.add_scalar("train/batch_recall@5_jepa", br5_jepa, global_step)
                        msg = msg.replace(
                            f"eff_rank={eff_rank:.1f}",
                            f"rho_jepa={rho_jepa:+.3f} eff_rank={eff_rank:.1f}",
                        )

                # Positive-mining cache health: number of anchor slots padded
                # with random negatives after filtering eval positions and
                # same-video clips.
                if sampler is not None:
                    writer.add_scalar(
                        "train/positive_pad_invocations",
                        sampler.pad_invocations, global_step,
                    )
                    writer.add_scalar(
                        "train/skipped_anchor_invocations",
                        sampler.skipped_anchor_invocations, global_step,
                    )

                # Histograms 10x less often than scalars (TB file size)
                if step % (cfg.log_frequency * 10) == 0:
                    writer.add_histogram(
                        "train/pred_sim_hist", _upper_triangle(pred), global_step)
                    if not set_rank_mode:
                        writer.add_histogram(
                            "train/dtw_sim_hist", _upper_triangle(dtw_sims_f), global_step)
                    if not set_rank_mode and not trajectory_mode:
                        writer.add_histogram(
                            "train/jepa_sim_hist", _upper_triangle(jepa_sim), global_step)
                print(msg)

        avg = {k: v / max(num_steps, 1) for k, v in epoch_losses.items()}
        writer.add_scalar("epoch/loss_total", avg["total"], epoch)
        if set_rank_mode:
            writer.add_scalar("epoch/loss_set", avg["set"], epoch)
            writer.add_scalar("epoch/loss_rank", avg["rank"], epoch)
            msg = (
                f"Epoch {epoch}/{cfg.num_epochs} | avg_loss={avg['total']:.4f} "
                f"set={avg['set']:.4f} rank={avg['rank']:.4f}"
            )
        else:
            writer.add_scalar("epoch/loss_trajectory", avg["trajectory"], epoch)
            if not trajectory_mode:
                writer.add_scalar("epoch/loss_semantic", avg["semantic"], epoch)
            msg = (
                f"Epoch {epoch}/{cfg.num_epochs} | "
                f"avg_loss={avg['total']:.4f} traj={avg['trajectory']:.4f}"
            )
            if not trajectory_mode:
                msg += f" sem={avg['semantic']:.4f}"
        print(msg)

        # -- Evaluation --
        if epoch % cfg.eval_frequency == 0 or epoch == cfg.num_epochs:
            with prof.section("eval"):
                metrics, selection_metric, selection_desc = _run_and_log_evaluation(
                    model=model,
                    eval_dataloader=eval_dataloader,
                    eval_dtw_matrix=eval_dtw_matrix,
                    eval_jepa_sim=eval_jepa_sim,
                    cfg=cfg,
                    device=device,
                    criterion=eval_criterion,
                    dtw_computer=dtw_computer,
                    writer=writer,
                    epoch=epoch,
                    trajectory_mode=trajectory_mode,
                    global_candidate_dataloader=global_candidate_dataloader,
                    global_dtw_neighbor_clip_idx=global_dtw_neighbor_clip_idx,
                    train_pool_query_clip_idx=train_pool_query_clip_idx,
                    train_pool_dtw_neighbor_clip_idx=train_pool_dtw_neighbor_clip_idx,
                )
            if selection_metric > best_selection_metric:
                best_selection_metric = selection_metric
                with prof.section("checkpoint_save"):
                    save_checkpoint(
                        model, optimizer, scheduler, scaler, cfg,
                        epoch, global_step, best_selection_metric,
                        run_dir / "best_model.pt",
                    )
                print(f"  New best {selection_desc}, saved best_model.pt")

        if epoch % cfg.checkpoint_frequency == 0 or epoch == cfg.num_epochs:
            with prof.section("checkpoint_save"):
                save_checkpoint(
                    model, optimizer, scheduler, scaler, cfg,
                    epoch, global_step, best_selection_metric,
                    run_dir / f"checkpoint_epoch_{epoch}.pt",
                )

        epoch_wall = time.perf_counter() - epoch_t0
        print(prof.report(epoch_wall))

    writer.close()
    if trajectory_mode:
        print(f"Training complete. Best selection metric (CKNNA DTW)={best_selection_metric:.4f}")
    else:
        print(f"Training complete. Best selection metric (balanced_R@5)={best_selection_metric:.4f}")


if __name__ == "__main__":
    main()
