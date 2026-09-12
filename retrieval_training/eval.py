from __future__ import annotations

import time
from typing import Dict, NamedTuple

import numpy as np
import torch
from torch.utils.data import DataLoader

from config import Config
from data import select_model_input
from model import TrajectoryRetrievalModel


class EncodedDataset(NamedTuple):
    z: np.ndarray          # (N, retrieval_dim) L2-normed retrieval embeddings
    v_ref: np.ndarray      # (N, encoder_dim) L2-normed mean-pooled JEPA features
    clip_indices: np.ndarray  # (N,) clip index for DTW matrix lookup


def encode_dataset(
    model: TrajectoryRetrievalModel,
    dataloader: DataLoader,
    device: torch.device,
    mixed_precision: bool = True,
    input_mode: str = "jepa",
) -> EncodedDataset:
    """Run clips through the model and collect embeddings + indices.

    Encodes the full dataloader; the caller bounds N via the underlying
    `Subset` (no per-call early-exit knob — that just risked drift between
    encoded N and the precomputed eval matrices).
    """
    model.eval()
    all_z, all_vref, all_idx = [], [], []
    autocast_enabled = mixed_precision and torch.cuda.is_available()
    with torch.no_grad(), torch.amp.autocast("cuda", enabled=autocast_enabled):
        for batch in dataloader:
            feats, mask = select_model_input(batch, input_mode, device)
            out = model(feats, mask)
            all_z.append(out["z"].float().cpu().numpy())
            all_vref.append(out["v_ref"].float().cpu().numpy())
            all_idx.append(batch["clip_indices"].numpy())
    return EncodedDataset(
        np.concatenate(all_z),
        np.concatenate(all_vref),
        np.concatenate(all_idx),
    )


def encode_dataset_with_loss(
    model: TrajectoryRetrievalModel,
    dataloader: DataLoader,
    criterion,                # CombinedLoss
    dtw_computer,             # OnlineDTWComputer
    cfg: "Config",
    device: torch.device,
) -> tuple["EncodedDataset", Dict[str, float]]:
    """One-pass eval encoder + per-batch loss accumulator.

    Same forward pass as `encode_dataset`, plus the same DTW-similarity and
    loss computations the train loop runs each step — averaged over eval
    batches. Returns `(EncodedDataset, loss_metrics)` where `loss_metrics`
    contains trajectory loss/entropy/KL metrics. In JEPA input mode it also
    includes semantic loss/entropy/KL metrics; in trajectory input mode `v_ref`
    is a pooled trajectory, so semantic-labeled metrics are omitted.

    These are *directly comparable* to the corresponding train scalars
    (`train/loss_trajectory`, `train/KL_gap_traj`, etc.) — same loss class,
    same temperatures, same z-score normalisation, just averaged over the
    held-out subset instead of a single train batch. A widening
    train→eval gap is the canonical overfitting signature.
    """
    from losses import target_entropy_nats

    model.eval()
    all_z, all_vref, all_idx = [], [], []
    sum_loss_traj = 0.0
    sum_loss_sem = 0.0
    sum_H_traj = 0.0
    sum_H_sem = 0.0
    n_batches = 0
    include_semantic = cfg.input_mode != "trajectory"

    autocast_enabled = cfg.mixed_precision and torch.cuda.is_available()
    with torch.no_grad():
        for batch in dataloader:
            feats, mask = select_model_input(batch, cfg.input_mode, device)
            trajs = batch["trajectories"].to(device, non_blocking=True)
            traj_lens = batch["traj_lengths"].to(device, non_blocking=True)

            # DTW under autocast-disabled fp32 to match the train regime
            # exactly (online_dtw is bit-equivalent to the offline matrix
            # only in fp32).
            with torch.amp.autocast("cuda", enabled=False):
                dtw_sims = dtw_computer.compute(trajs, traj_lens)

            with torch.amp.autocast("cuda", enabled=autocast_enabled):
                out = model(feats, mask)
                losses = criterion(out["z"], out["v_ref"], dtw_sims)

            dtw_sims_f = dtw_sims.float()

            all_z.append(out["z"].float().cpu().numpy())
            all_vref.append(out["v_ref"].float().cpu().numpy())
            all_idx.append(batch["clip_indices"].numpy())

            sum_loss_traj += float(losses["trajectory"].item())
            sum_H_traj += target_entropy_nats(dtw_sims_f, cfg.loss_tau_target)
            if include_semantic:
                # Compute the JEPA target sim in fp32 for the entropy helper.
                sim_ref_f = (out["v_ref"] @ out["v_ref"].T).detach().float()
                sum_loss_sem += float(losses["semantic"].item())
                sum_H_sem += target_entropy_nats(sim_ref_f, cfg.loss_tau_target)
            n_batches += 1

    encoded = EncodedDataset(
        np.concatenate(all_z),
        np.concatenate(all_vref),
        np.concatenate(all_idx),
    )

    n = max(n_batches, 1)
    avg_loss_traj = sum_loss_traj / n
    avg_H_traj = sum_H_traj / n
    loss_metrics = {
        "loss_trajectory": avg_loss_traj,
        "H_target_traj":   avg_H_traj,
        "KL_gap_traj":     avg_loss_traj - avg_H_traj,
    }
    if include_semantic:
        avg_loss_sem = sum_loss_sem / n
        avg_H_sem = sum_H_sem / n
        loss_metrics.update({
            "loss_semantic": avg_loss_sem,
            "H_target_sem":  avg_H_sem,
            "KL_gap_sem":    avg_loss_sem - avg_H_sem,
        })
    return encoded, loss_metrics


def build_faiss_index(embeddings: np.ndarray, index_type: str = "IndexFlatIP"):
    import faiss

    dim = embeddings.shape[1]
    index = getattr(faiss, index_type)(dim)
    index.add(embeddings.astype(np.float32))
    return index


RELEVANCE_TOP_N = 20  # fixed positive-set size for kNN-recall (matches cached dtw_neighbors top-K)


def _retrieve_neighbors(index, query_embeddings: np.ndarray, clip_indices: np.ndarray,
                        max_k: int):
    """Run the FAISS search and return per-query neighbor info.

    Returns:
        neighbor_pos:    (n_queries, max_k+1) positions into clip_indices
        neighbor_clip:   (n_queries, max_k+1) actual clip indices (DTW matrix indices)
        valid_neighbor:  (n_queries, max_k+1) bool — True for in-bounds & non-self
    """
    n_queries = len(query_embeddings)
    # +1 because the query itself may be in the FAISS index and would otherwise
    # consume one of the top-k slots.
    _, neighbor_pos = index.search(query_embeddings.astype(np.float32), max_k + 1)
    valid_pos = (neighbor_pos >= 0) & (neighbor_pos < n_queries)
    safe_pos = np.where(valid_pos, neighbor_pos, 0)
    neighbor_clip = clip_indices[safe_pos]
    self_mask = neighbor_clip == clip_indices[:, None]
    valid_neighbor = valid_pos & ~self_mask
    return neighbor_pos, neighbor_clip, valid_neighbor


def _topn_indices_from_sim(target_sim: np.ndarray, top_n: int) -> np.ndarray:
    """Return [N, top_n] positions of each row's top-`top_n` neighbors, excluding self."""
    n = target_sim.shape[0]
    top_n = min(top_n, n - 1)
    sim = np.array(target_sim, dtype=np.float32, copy=True)
    np.fill_diagonal(sim, -np.inf)
    # argpartition for top-n then sort within for deterministic order (order
    # doesn't affect indicator R@K but keeps results stable for tests).
    part = np.argpartition(-sim, top_n - 1, axis=1)[:, :top_n]
    row_idx = np.arange(n)[:, None]
    part_sorted = part[row_idx, np.argsort(-sim[row_idx, part], axis=1)]
    return part_sorted.astype(np.int64)


def _indicator_recall_at_k(
    neighbor_pos: np.ndarray,
    valid_neighbor: np.ndarray,
    relevance_topn_idx: np.ndarray,
    k_values: list,
    prefix: str = "",
) -> Dict[str, float]:
    """Indicator-form Recall@K against a fixed top-N relevance set.

    For each query, R@K = 1 iff any of the model's top-K valid neighbors is in
    that query's relevance set, else 0; averaged over queries. Matches the
    metric-learning / image-retrieval standard (CUB, SOP, DPR).

    `neighbor_pos` / `valid_neighbor` come from `_retrieve_neighbors` and are
    eval-position-indexed. `relevance_topn_idx` is [N_queries, N_relevant]
    eval positions of each query's true top-N (by DTW or JEPA similarity).
    """
    n_queries = neighbor_pos.shape[0]
    # safe_pos: invalid slots get -1 so they can't match any relevance entry.
    safe_pos = np.where(valid_neighbor, neighbor_pos, -1)
    # is_positive[q, slot] = True iff that slot's position is in relevance set for q.
    is_positive = (
        safe_pos[:, :, None] == relevance_topn_idx[:, None, :]
    ).any(axis=2)

    # 1-indexed rank among valid neighbors so "top-K" means top-K valid
    # (skipping any invalid / self-match slots).
    rank = np.where(valid_neighbor, np.cumsum(valid_neighbor, axis=1), 0)
    results: Dict[str, float] = {}
    for k in k_values:
        in_top_k = valid_neighbor & (rank <= k)
        hit = (is_positive & in_top_k).any(axis=1)
        results[f"{prefix}R@{k}"] = float(hit.mean()) if n_queries else 0.0
    return results


def _hsic_unbiased(K: torch.Tensor, L: torch.Tensor) -> torch.Tensor:
    """Unbiased HSIC estimator (Song et al. 2012, Eq. 5)."""
    m = K.shape[0]
    K_tilde = K.clone().fill_diagonal_(0)
    L_tilde = L.clone().fill_diagonal_(0)

    chunk = 2000
    term1 = torch.tensor(0.0, device=K.device, dtype=K.dtype)
    for i in range(0, m, chunk):
        end = min(i + chunk, m)
        term1 += (K_tilde[i:end] * L_tilde.T[i:end]).sum()

    term2 = K_tilde.sum() * L_tilde.sum() / ((m - 1) * (m - 2))

    k_col = K_tilde.sum(dim=0)
    l_row = L_tilde.sum(dim=1)
    term3 = 2.0 * (k_col @ l_row) / (m - 2)

    return (term1 + term2 - term3) / (m * (m - 3))


def _cosine_topk(
    feats: torch.Tensor, topk: int, device: torch.device
) -> tuple[torch.Tensor, torch.Tensor]:
    """L2-normed features → cosine similarity matrix + top-k binary mask."""
    K_sim = feats @ feats.T
    sim_for_topk = K_sim.clone().fill_diagonal_(float("-inf"))
    _, topk_idx = torch.topk(sim_for_topk, topk, dim=1)
    mask = torch.zeros_like(K_sim)
    mask.scatter_(1, topk_idx, 1.0)
    return K_sim, mask


def _sim_topk_mask(
    sim_matrix: torch.Tensor, topk: int
) -> torch.Tensor:
    """Similarity matrix → top-k binary mask (largest = nearest)."""
    sim_for_topk = sim_matrix.clone().fill_diagonal_(float("-inf"))
    _, topk_idx = torch.topk(sim_for_topk, topk, dim=1)
    mask = torch.zeros_like(sim_matrix)
    mask.scatter_(1, topk_idx, 1.0)
    return mask


def _asymmetric_platonic_cknna(
    K_sim: torch.Tensor, mask_K: torch.Tensor, mask_L: torch.Tensor
) -> float:
    """Asymmetric Platonic CKNNA: continuous K-side, binary L-side."""
    mask_inter = mask_K * mask_L
    sim_kl = _hsic_unbiased(mask_inter * K_sim, mask_inter * mask_L)
    sim_kk = _hsic_unbiased(mask_K * K_sim, mask_K * K_sim)
    sim_ll = _hsic_unbiased(mask_L, mask_L)
    if sim_kk <= 0 or sim_ll <= 0:
        return 0.0
    return (sim_kl / (sim_kk * sim_ll).sqrt()).item()


def compute_cknna(
    z: np.ndarray,
    v_ref: np.ndarray,
    eval_dtw_matrix: np.ndarray,
    topk: int,
    device: torch.device,
    include_jepa: bool = True,
) -> Dict[str, float]:
    """Compute CKNNA between retrieval embeddings and DTW / JEPA neighborhoods.

    `eval_dtw_matrix` is the (N_eval, N_eval) DTW submatrix indexed by
    eval-position (matches z / v_ref ordering). When `include_jepa` is False
    (trajectory input mode) the JEPA-side CKNNA is skipped — `v_ref` is then a
    pooled trajectory, not a semantic embedding.
    """
    N = len(z)
    topk = min(topk, N - 1)

    z_t = torch.from_numpy(z).float().to(device)

    # Retrieval embedding similarities + top-k mask
    K_sim_z, mask_z = _cosine_topk(z_t, topk, device)

    # DTW similarity submatrix → top-k mask (L-side)
    dtw_t = torch.from_numpy(np.array(eval_dtw_matrix)).float().to(device)
    mask_dtw = _sim_topk_mask(dtw_t, topk)

    cknna_dtw = _asymmetric_platonic_cknna(K_sim_z, mask_z, mask_dtw)
    mknn_dtw = ((mask_z * mask_dtw).sum() / (topk * N)).item()

    metrics = {
        "cknna_dtw": cknna_dtw,
        "mutual_knn_dtw": mknn_dtw,
    }

    if include_jepa:
        # JEPA ref similarity → top-k mask (L-side)
        vref_t = torch.from_numpy(v_ref).float().to(device)
        _, mask_ref = _cosine_topk(vref_t, topk, device)
        metrics["cknna_jepa"] = _asymmetric_platonic_cknna(K_sim_z, mask_z, mask_ref)
        metrics["mutual_knn_jepa"] = ((mask_z * mask_ref).sum() / (topk * N)).item()

    return metrics


def compute_cknna_at_k(
    z: np.ndarray,
    v_ref: np.ndarray,
    eval_dtw_matrix: np.ndarray,
    k_values: list,
    device: torch.device,
) -> Dict[str, float]:
    """Compute eval-set DTW CKNNA metrics at explicit K values.

    The unsuffixed :func:`compute_cknna` metrics remain tied to
    ``config.cknna_topk``. These suffixed metrics line up with Recall@K.
    """
    metrics: Dict[str, float] = {}
    for k in k_values:
        one_k = compute_cknna(
            z, v_ref, eval_dtw_matrix,
            topk=int(k),
            device=device,
            include_jepa=False,
        )
        metrics[f"cknna_dtw@{k}"] = one_k["cknna_dtw"]
    return metrics


def _rectangular_centered_inner(
    x_indices: np.ndarray,
    x_values: np.ndarray,
    y_indices: np.ndarray,
    y_values: np.ndarray,
    n_columns: int,
) -> float:
    """Frobenius inner product after double-centering sparse Q x C matrices.

    Each row has a small, fixed-width list of column indices and values.  A
    negative column index denotes padding.  The implementation uses sparse
    sufficient statistics and never materializes the dense Q x C matrices.
    """
    if x_indices.shape != x_values.shape or y_indices.shape != y_values.shape:
        raise ValueError("sparse indices and values must have matching shapes")
    if x_indices.shape[0] != y_indices.shape[0]:
        raise ValueError("sparse matrices must have the same number of rows")
    n_rows = x_indices.shape[0]
    if n_rows == 0 or n_columns <= 0:
        return 0.0

    def stats(indices: np.ndarray, values: np.ndarray):
        valid = (indices >= 0) & (indices < n_columns)
        clean_values = np.where(valid, values, 0.0).astype(np.float64, copy=False)
        row_sum = clean_values.sum(axis=1)
        flat_idx = indices[valid].astype(np.int64, copy=False)
        col_sum = np.bincount(
            flat_idx, weights=clean_values[valid], minlength=n_columns,
        ).astype(np.float64, copy=False)
        return valid, clean_values, row_sum, col_sum, float(row_sum.sum())

    x_valid, x_clean, x_row, x_col, x_total = stats(x_indices, x_values)
    y_valid, y_clean, y_row, y_col, y_total = stats(y_indices, y_values)

    # Raw sparse <X,Y>: widths are tiny (normally k=10), so row-local matching
    # avoids constructing either a dense matrix or a general sparse object.
    raw_inner = 0.0
    for row in range(n_rows):
        y_by_col = {
            int(col): float(value)
            for col, value, valid in zip(y_indices[row], y_clean[row], y_valid[row])
            if valid
        }
        raw_inner += sum(
            float(value) * y_by_col.get(int(col), 0.0)
            for col, value, valid in zip(x_indices[row], x_clean[row], x_valid[row])
            if valid
        )

    return float(
        raw_inner
        - np.dot(x_row, y_row) / n_columns
        - np.dot(x_col, y_col) / n_rows
        + (x_total * y_total) / (n_rows * n_columns)
    )


def compute_global_pool_cknna(
    model_neighbor_clip_idx: np.ndarray,
    model_neighbor_similarity: np.ndarray,
    dtw_neighbor_clip_idx: np.ndarray,
    corpus_size: int,
) -> Dict[str, float]:
    """Sparse rectangular CKNNA for eval queries against a global corpus.

    The model side contains cosine similarity on its top-k support; the DTW
    side is a binary top-k mask from the cached global neighbor table.  Both
    Q x C matrices are double-centered before normalized Frobenius alignment.
    This is intentionally a separate metric from the square unbiased-HSIC
    CKNNA returned by :func:`compute_cknna`.
    """
    model_idx = np.asarray(model_neighbor_clip_idx, dtype=np.int64)
    model_sim = np.asarray(model_neighbor_similarity, dtype=np.float64)
    dtw_idx = np.asarray(dtw_neighbor_clip_idx, dtype=np.int64)
    if model_idx.shape != model_sim.shape or model_idx.shape != dtw_idx.shape:
        raise ValueError("model and DTW neighbor tables must have identical shapes")
    if model_idx.ndim != 2 or model_idx.shape[1] == 0:
        raise ValueError("neighbor tables must have shape [num_queries, topk > 0]")

    dtw_values = ((dtw_idx >= 0) & (dtw_idx < corpus_size)).astype(np.float64)
    numerator = _rectangular_centered_inner(
        model_idx, model_sim, dtw_idx, dtw_values, corpus_size,
    )
    model_norm_sq = _rectangular_centered_inner(
        model_idx, model_sim, model_idx, model_sim, corpus_size,
    )
    dtw_norm_sq = _rectangular_centered_inner(
        dtw_idx, dtw_values, dtw_idx, dtw_values, corpus_size,
    )
    denom = np.sqrt(max(model_norm_sq, 0.0) * max(dtw_norm_sq, 0.0))
    cknna = float(np.clip(numerator / denom, -1.0, 1.0)) if denom > 0 else 0.0

    overlaps = []
    for model_row, dtw_row in zip(model_idx, dtw_idx):
        model_set = {int(i) for i in model_row if 0 <= i < corpus_size}
        dtw_set = {int(i) for i in dtw_row if 0 <= i < corpus_size}
        overlaps.append(len(model_set & dtw_set) / model_idx.shape[1])

    return {
        "cknna_dtw_global_pool": cknna,
        "mutual_knn_dtw_global_pool": float(np.mean(overlaps)) if overlaps else 0.0,
    }


def compute_global_recall_at_k(
    model_neighbor_clip_idx: np.ndarray,
    dtw_neighbor_clip_idx: np.ndarray,
    k_values: list,
    corpus_size: int,
    relevance_n: int = RELEVANCE_TOP_N,
    prefix: str = "global_",
) -> Dict[str, float]:
    """Indicator-form Recall@K for eval queries against a global candidate corpus.

    For each query and requested K:
        R@K = mean_q 1[ |model_top-K(q) ∩ dtw_top-relevance_n(q)| > 0 ]

    `dtw_neighbor_clip_idx` is sliced to its first `relevance_n` columns as the
    fixed positive set; the same set is used for every K. Padding entries
    (negative or out-of-corpus) are ignored.
    """
    model_idx = np.asarray(model_neighbor_clip_idx, dtype=np.int64)
    dtw_idx = np.asarray(dtw_neighbor_clip_idx, dtype=np.int64)
    if model_idx.ndim != 2 or dtw_idx.ndim != 2:
        raise ValueError("neighbor tables must have shape [num_queries, topk]")
    if model_idx.shape[0] != dtw_idx.shape[0]:
        raise ValueError("model and DTW neighbor tables must have same row count")

    n_eff = min(int(relevance_n), dtw_idx.shape[1])
    dtw_relevance = dtw_idx[:, :n_eff]

    metrics: Dict[str, float] = {}
    for k in k_values:
        kk = min(int(k), model_idx.shape[1])
        if kk <= 0 or model_idx.shape[0] == 0:
            metrics[f"{prefix}R@{k}"] = 0.0
            continue
        hits = []
        for model_row, dtw_row in zip(model_idx[:, :kk], dtw_relevance):
            model_set = {int(i) for i in model_row if 0 <= i < corpus_size}
            dtw_set = {int(i) for i in dtw_row if 0 <= i < corpus_size}
            hits.append(bool(model_set & dtw_set))
        metrics[f"{prefix}R@{k}"] = float(np.mean(hits)) if hits else 0.0
    return metrics


def compute_global_pool_cknna_at_k(
    model_neighbor_clip_idx: np.ndarray,
    model_neighbor_similarity: np.ndarray,
    dtw_neighbor_clip_idx: np.ndarray,
    k_values: list,
    corpus_size: int,
) -> Dict[str, float]:
    """Compute global-pool DTW CKNNA at explicit K values."""
    metrics: Dict[str, float] = {}
    for k in k_values:
        kk = int(k)
        global_k = compute_global_pool_cknna(
            model_neighbor_clip_idx[:, :kk],
            model_neighbor_similarity[:, :kk],
            dtw_neighbor_clip_idx[:, :kk],
            corpus_size=corpus_size,
        )
        metrics[f"cknna_dtw_global_pool@{k}"] = global_k["cknna_dtw_global_pool"]
    return metrics


def _global_model_neighbors(
    query_z: np.ndarray,
    query_clip_indices: np.ndarray,
    corpus_z: np.ndarray,
    corpus_clip_indices: np.ndarray,
    topk: int,
    index_type: str,
) -> tuple[np.ndarray, np.ndarray]:
    """Return exact/FAISS model neighbors in raw clip-index space."""
    n_corpus = len(corpus_z)
    k = min(topk, max(n_corpus - 1, 0))
    if k < 1:
        raise ValueError("global-pool CKNNA requires at least two corpus clips")
    index = build_faiss_index(corpus_z, index_type)
    similarities, positions = index.search(
        query_z.astype(np.float32), min(k + 1, n_corpus),
    )
    out_idx = np.full((len(query_z), k), -1, dtype=np.int64)
    out_sim = np.full((len(query_z), k), -np.inf, dtype=np.float32)
    for row in range(len(query_z)):
        write = 0
        for sim, pos in zip(similarities[row], positions[row]):
            if pos < 0:
                continue
            clip_idx = int(corpus_clip_indices[pos])
            if clip_idx == int(query_clip_indices[row]):
                continue
            out_idx[row, write] = clip_idx
            out_sim[row, write] = sim
            write += 1
            if write == k:
                break
        if write != k:
            raise RuntimeError(f"query {row} produced only {write}/{k} global neighbors")
    return out_idx, out_sim


def _eval_embedding_health(z: np.ndarray, device: torch.device) -> Dict[str, float]:
    """Eval-set collapse signals on the encoded retrieval embeddings.

    Mirrors the train-side `_z_collapse_stats` / `_off_diag_stats` so a
    diverging value between train and eval is easy to spot — the model can
    look healthy on training batches but be collapsed on the held-out set.
    """
    z_t = torch.from_numpy(z).to(device).float()
    z_std = float(z_t.std(dim=0).mean().item())
    S = torch.linalg.svdvals(z_t - z_t.mean(dim=0, keepdim=True))
    p = S / (S.sum() + 1e-12)
    eff_rank = float(torch.exp(-(p * torch.log(p + 1e-12)).sum()).item())

    sim = z_t @ z_t.T
    n = sim.shape[0]
    off_mask = ~torch.eye(n, dtype=torch.bool, device=sim.device)
    off = sim[off_mask]
    return {
        "z_std": z_std,
        "z_effective_rank": eff_rank,
        "z_off_diag_mean": float(off.mean().item()),
        "z_off_diag_std": float(off.std().item()),
    }


def _pool_metrics_from_encoded(
    query_z: np.ndarray,
    query_clip_idx: np.ndarray,
    corpus_z: np.ndarray,
    corpus_clip_idx: np.ndarray,
    dtw_neighbor_idx_rows: np.ndarray,
    k_values: list,
    cknna_k_values: list,
    cknna_topk: int,
    index_type: str,
    prefix: str,
    recall_prefix: str,
    index_space_size: int | None = None,
) -> Dict[str, float]:
    """Compute (CKNNA + mutual-kNN + R@K) for a query subset against a corpus.

    `prefix` selects the metric suffix:
      - `"global_pool"` → `cknna_dtw_global_pool@K`,
        `mutual_knn_dtw_global_pool@cknna_topk`, `global_R@K`
      - `"train_pool"`  → `cknna_dtw_train_pool@K`,
        `mutual_knn_dtw_train_pool@cknna_topk`, `train_pool_R@K`
    Recall is reported at `k_values`; CKNNA at `cknna_k_values`.
    `dtw_neighbor_idx_rows` is `[Q, >=max_topk]` of original-clip-idx neighbors
    aligned to `query_clip_idx`; `-1` entries are treated as missing.
    """
    pool_topk = max(
        int(cknna_topk),
        max(int(k) for k in k_values),
        max(int(k) for k in cknna_k_values),
    )
    if dtw_neighbor_idx_rows.shape[1] < pool_topk:
        raise RuntimeError(
            f"{prefix} DTW neighbor rows have top_k={dtw_neighbor_idx_rows.shape[1]} "
            f"< required {pool_topk}"
        )
    model_neighbor_idx, model_neighbor_sim = _global_model_neighbors(
        query_z, query_clip_idx, corpus_z, corpus_clip_idx,
        pool_topk, index_type,
    )
    dtw_neighbor_idx = dtw_neighbor_idx_rows[:, :model_neighbor_idx.shape[1]]
    corpus_size = int(index_space_size) if index_space_size is not None else len(corpus_z)

    recall_metrics = compute_global_recall_at_k(
        model_neighbor_idx, dtw_neighbor_idx, k_values,
        corpus_size=corpus_size,
        prefix=recall_prefix,
    )
    cknna_metrics = compute_global_pool_cknna(
        model_neighbor_idx[:, :cknna_topk],
        model_neighbor_sim[:, :cknna_topk],
        dtw_neighbor_idx[:, :cknna_topk],
        corpus_size=corpus_size,
    )
    explicit_cknna_k_values = [
        k for k in cknna_k_values if int(k) != int(cknna_topk)
    ]
    extra_cknna = compute_global_pool_cknna_at_k(
        model_neighbor_idx, model_neighbor_sim, dtw_neighbor_idx,
        explicit_cknna_k_values, corpus_size=corpus_size,
    )
    metrics: Dict[str, float] = {}
    metrics.update(recall_metrics)
    # Rename the "@k=cknna_topk" key from the generic "cknna_dtw_global_pool"
    # produced by compute_global_pool_cknna to the prefix-specific scheme.
    base_cknna_key = "cknna_dtw_global_pool"
    base_mutual_key = "mutual_knn_dtw_global_pool"
    target_cknna_key = (
        f"cknna_dtw_{prefix}@{cknna_topk}"
    )
    target_mutual_key = f"mutual_knn_dtw_{prefix}@{cknna_topk}"
    metrics[target_cknna_key] = cknna_metrics[base_cknna_key]
    metrics[target_mutual_key] = cknna_metrics[base_mutual_key]
    for k in explicit_cknna_k_values:
        src_key = f"cknna_dtw_global_pool@{k}"
        metrics[f"cknna_dtw_{prefix}@{k}"] = extra_cknna[src_key]
    return metrics


def run_evaluation(
    model: TrajectoryRetrievalModel,
    dataloader: DataLoader,
    eval_dtw_matrix: np.ndarray,
    eval_jepa_sim: np.ndarray,
    config: Config,
    device: torch.device,
    criterion=None,        # CombinedLoss; if provided, eval/loss_* + KL_gap_* are computed
    dtw_computer=None,     # OnlineDTWComputer; required if criterion is provided
    global_candidate_dataloader: DataLoader | None = None,
    global_dtw_neighbor_clip_idx: np.ndarray | None = None,
    train_pool_query_clip_idx: np.ndarray | None = None,
    train_pool_dtw_neighbor_clip_idx: np.ndarray | None = None,
) -> Dict[str, float]:
    """Run all eval metrics against precomputed eval-set DTW + JEPA matrices.

    Both target matrices are (N_eval, N_eval), indexed by eval-position, and
    are built once at startup (they do not depend on model state).

    When `criterion` and `dtw_computer` are provided, also computes
    trajectory loss/entropy/KL metrics averaged over eval batches. In JEPA
    input mode, semantic loss/entropy/KL metrics are included too. These are
    directly comparable to the corresponding train-side scalars; any
    divergence is the canonical overfitting signature.
    """
    eval_start = time.time()
    compute_loss_metrics = criterion is not None and dtw_computer is not None
    # No semantic axis in trajectory mode: eval_jepa_sim is passed as None.
    include_jepa = eval_jepa_sim is not None

    t0 = time.time()
    if compute_loss_metrics:
        encoded, loss_metrics = encode_dataset_with_loss(
            model, dataloader, criterion, dtw_computer, config, device,
        )
        msg = (
            f"  [eval] encode_dataset (with loss): "
            f"{time.time() - t0:.1f}s ({len(encoded.z)} clips, "
            f"loss_traj={loss_metrics['loss_trajectory']:.3f}, "
            f"KL_gap_traj={loss_metrics['KL_gap_traj']:.3f}"
        )
        if "loss_semantic" in loss_metrics:
            msg += f", loss_sem={loss_metrics['loss_semantic']:.3f}"
        msg += ")"
        print(msg)
    else:
        encoded = encode_dataset(
            model, dataloader, device, mixed_precision=config.mixed_precision,
            input_mode=config.input_mode,
        )
        loss_metrics = {}
        print(f"  [eval] encode_dataset: {time.time() - t0:.1f}s ({len(encoded.z)} clips)")

    n = encoded.z.shape[0]
    if n != eval_dtw_matrix.shape[0] or (include_jepa and n != eval_jepa_sim.shape[0]):
        raise RuntimeError(
            f"size mismatch: encoded N={n}, eval_dtw_matrix N={eval_dtw_matrix.shape[0]}"
            + (f", eval_jepa_sim N={eval_jepa_sim.shape[0]}" if include_jepa else "")
        )

    t0 = time.time()
    index = build_faiss_index(encoded.z, config.faiss_index_type)
    print(f"  [eval] build_faiss_index: {time.time() - t0:.1f}s")

    # FAISS lookup is shared across DTW and JEPA recall — only the target
    # similarity matrix differs.
    max_k = max(config.recall_k_values)
    neighbor_pos, _, valid_neighbor = _retrieve_neighbors(
        index, encoded.z, encoded.clip_indices, max_k,
    )

    t0 = time.time()
    dtw_topn_idx = _topn_indices_from_sim(eval_dtw_matrix, RELEVANCE_TOP_N)
    metrics = _indicator_recall_at_k(
        neighbor_pos, valid_neighbor, dtw_topn_idx, config.recall_k_values,
    )
    print(f"  [eval] recall (DTW): {time.time() - t0:.1f}s (relevance top-{RELEVANCE_TOP_N})")

    if include_jepa:
        t0 = time.time()
        jepa_topn_idx = _topn_indices_from_sim(eval_jepa_sim, RELEVANCE_TOP_N)
        jepa_metrics = _indicator_recall_at_k(
            neighbor_pos, valid_neighbor, jepa_topn_idx,
            config.recall_k_values, prefix="jepa_",
        )
        metrics.update(jepa_metrics)
        print(f"  [eval] recall (JEPA): {time.time() - t0:.1f}s (relevance top-{RELEVANCE_TOP_N})")

    t0 = time.time()
    cknna_metrics = compute_cknna(
        encoded.z,
        encoded.v_ref,
        eval_dtw_matrix,
        topk=config.cknna_topk,
        device=device,
        include_jepa=include_jepa,
    )
    explicit_cknna_k_values = [
        k for k in config.cknna_k_values if int(k) != int(config.cknna_topk)
    ]
    cknna_metrics.update(compute_cknna_at_k(
        encoded.z,
        encoded.v_ref,
        eval_dtw_matrix,
        explicit_cknna_k_values,
        device=device,
    ))
    cknna_metrics[f"cknna_dtw@{config.cknna_topk}"] = cknna_metrics.pop("cknna_dtw")
    cknna_metrics[f"mutual_knn_dtw@{config.cknna_topk}"] = cknna_metrics.pop("mutual_knn_dtw")
    if "mutual_knn_jepa" in cknna_metrics:
        cknna_metrics[f"mutual_knn_jepa@{config.cknna_topk}"] = cknna_metrics.pop("mutual_knn_jepa")
    print(f"  [eval] compute_cknna: {time.time() - t0:.1f}s")

    metrics.update(cknna_metrics)

    if global_candidate_dataloader is not None:
        if global_dtw_neighbor_clip_idx is None:
            raise ValueError("global DTW neighbors are required for global-pool CKNNA")
        t0 = time.time()
        remainder = encode_dataset(
            model, global_candidate_dataloader, device,
            mixed_precision=config.mixed_precision,
            input_mode=config.input_mode,
        )
        corpus_z = np.concatenate([encoded.z, remainder.z], axis=0)
        corpus_clip_indices = np.concatenate(
            [encoded.clip_indices, remainder.clip_indices], axis=0,
        )
        if len(np.unique(corpus_clip_indices)) != len(corpus_clip_indices):
            raise RuntimeError("global candidate corpus contains duplicate clip indices")
        if not np.array_equal(
            np.sort(corpus_clip_indices), np.arange(len(corpus_clip_indices)),
        ):
            raise RuntimeError(
                "global-pool CKNNA requires contiguous raw clip indices [0, N)"
            )
        dtw_table = np.asarray(global_dtw_neighbor_clip_idx)
        if dtw_table.ndim != 2 or dtw_table.shape[0] <= int(encoded.clip_indices.max()):
            raise RuntimeError(
                "global DTW neighbor table does not cover every eval clip index"
            )
        global_topk = max(
            config.cknna_topk,
            max(config.recall_k_values),
            max(config.cknna_k_values),
        )
        if dtw_table.shape[1] < global_topk:
            raise RuntimeError(
                f"global DTW neighbor table top_k={dtw_table.shape[1]} "
                f"< required {global_topk}"
            )
        global_metrics = _pool_metrics_from_encoded(
            query_z=encoded.z,
            query_clip_idx=encoded.clip_indices,
            corpus_z=corpus_z,
            corpus_clip_idx=corpus_clip_indices,
            dtw_neighbor_idx_rows=dtw_table[encoded.clip_indices],
            k_values=config.recall_k_values,
            cknna_k_values=config.cknna_k_values,
            cknna_topk=config.cknna_topk,
            index_type=config.faiss_index_type,
            prefix="global_pool",
            recall_prefix="global_",
        )
        metrics.update(global_metrics)
        global_cknna_key = f"cknna_dtw_global_pool@{config.cknna_topk}"
        print(
            f"  [eval] global-pool metrics: {time.time() - t0:.1f}s "
            f"({len(encoded.z)} queries, {len(corpus_z)} candidates, "
            f"cknna_topk={config.cknna_topk}, recall_topk={global_topk}, "
            f"cknna={global_metrics[global_cknna_key]:.4f})"
        )

        if (
            train_pool_query_clip_idx is not None
            and train_pool_dtw_neighbor_clip_idx is not None
        ):
            t0 = time.time()
            train_query_clip_idx = np.asarray(train_pool_query_clip_idx, dtype=np.int64)
            # Locate query rows inside `remainder` (the train-only corpus).
            remainder_pos_by_clip = {
                int(c): pos for pos, c in enumerate(remainder.clip_indices)
            }
            try:
                query_rows = np.asarray(
                    [remainder_pos_by_clip[int(c)] for c in train_query_clip_idx],
                    dtype=np.int64,
                )
            except KeyError as exc:
                raise RuntimeError(
                    f"train-pool query clip {exc.args[0]} not found in global "
                    "candidate corpus — query subset must be a subset of train positions"
                ) from None
            query_z = remainder.z[query_rows]
            query_idx = remainder.clip_indices[query_rows]

            train_pool_dtw_table = np.asarray(train_pool_dtw_neighbor_clip_idx)
            if train_pool_dtw_table.ndim != 2:
                raise RuntimeError("train-pool DTW neighbor table must be 2-D")
            if train_pool_dtw_table.shape[1] < global_topk:
                raise RuntimeError(
                    f"train-pool DTW neighbor table top_k="
                    f"{train_pool_dtw_table.shape[1]} < required {global_topk}"
                )
            train_pool_metrics = _pool_metrics_from_encoded(
                query_z=query_z,
                query_clip_idx=query_idx,
                corpus_z=remainder.z,
                corpus_clip_idx=remainder.clip_indices,
                dtw_neighbor_idx_rows=train_pool_dtw_table[query_idx],
                k_values=config.recall_k_values,
                cknna_k_values=config.cknna_k_values,
                cknna_topk=config.cknna_topk,
                index_type=config.faiss_index_type,
                prefix="train_pool",
                recall_prefix="train_pool_",
                index_space_size=len(corpus_z),
            )
            metrics.update(train_pool_metrics)
            train_pool_cknna_key = f"cknna_dtw_train_pool@{config.cknna_topk}"
            print(
                f"  [eval] train-pool metrics: {time.time() - t0:.1f}s "
                f"({len(query_z)} queries, {len(remainder.z)} candidates, "
                f"cknna_topk={config.cknna_topk}, recall_topk={global_topk}, "
                f"cknna={train_pool_metrics[train_pool_cknna_key]:.4f})"
            )

    metrics.update(_eval_embedding_health(encoded.z, device))
    metrics.update(loss_metrics)
    print(f"  [eval] total: {time.time() - eval_start:.1f}s")
    print(f"  Eval: {' | '.join(f'{k}={v:.4f}' for k, v in sorted(metrics.items()))}")
    return metrics
