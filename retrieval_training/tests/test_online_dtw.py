"""Tests for `OnlineDTWComputer` — exact (length-aware) symmetric-2 DTW.

Validates, against the `trajectories_{design}.pt` artifact:
    * `compute` matches a CPU numpy symmetric-2 DTW on the true prefixes,
    * the result is invariant to how much padding the batch carries,
    * `compute`, `compute_chunked` and `compute_one_query_against_targets`
      all agree.

Skipped automatically if the artifact is missing or CUDA is unavailable
(so the test file is safe to leave in tests/ on CI without GPU).

Run: cd retrieval_training && python -m pytest tests/test_online_dtw.py -v -s
"""

import sys
from pathlib import Path

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

DATA_DIR = Path(__file__).resolve().parents[2] / "outputs" / "training_data"
DESIGN = "abs_21j_coords"


@pytest.fixture(scope="module")
def artifacts():
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available; online DTW kernel requires GPU")
    traj_path = DATA_DIR / f"trajectories_{DESIGN}.pt"
    if not traj_path.exists():
        pytest.skip(f"Trajectory artifact missing: {traj_path}")

    payload = torch.load(traj_path, weights_only=False, map_location="cpu")
    assert payload["design"] == DESIGN, (
        f"trajectory file design={payload['design']} != {DESIGN}"
    )
    return payload


def _batch(payload, batch_idx):
    trajs = payload["trajectories"][batch_idx].cuda().float().contiguous()
    lens = payload["lengths"][batch_idx].cuda()
    return trajs, lens


def _online_for_indices(payload, batch_idx, design=DESIGN):
    from online_dtw import OnlineDTWComputer

    trajs, lens = _batch(payload, batch_idx)
    return OnlineDTWComputer(design).compute(trajs, lens).cpu().numpy()


def _reference_sym2_dtw(a: np.ndarray, b: np.ndarray) -> float:
    """CPU symmetric-2 DTW (dtw-python convention) on two unpadded sequences.

    Diagonal steps cost 2*d, vertical/horizontal steps cost d.
    """
    a = a.astype(np.float64)
    b = b.astype(np.float64)
    D = np.sqrt(((a[:, None, :] - b[None, :, :]) ** 2).sum(-1))
    L_i, L_j = D.shape
    R = np.full((L_i, L_j), np.inf)
    R[0, 0] = D[0, 0]
    for i in range(1, L_i):
        R[i, 0] = R[i - 1, 0] + D[i, 0]
    for j in range(1, L_j):
        R[0, j] = R[0, j - 1] + D[0, j]
    for i in range(1, L_i):
        for j in range(1, L_j):
            R[i, j] = D[i, j] + min(R[i - 1, j - 1] + D[i, j], R[i - 1, j], R[i, j - 1])
    return float(R[L_i - 1, L_j - 1])


def _reference_similarity_matrix(payload, batch_idx, design=DESIGN):
    from dtw_designs import DTW_DESIGNS, DTW_NORMS

    norm_fn = DTW_NORMS[DTW_DESIGNS[design]["norm"]]
    trajs = payload["trajectories"][batch_idx].numpy()
    lens = payload["lengths"][batch_idx].numpy().astype(np.int64)
    B = len(batch_idx)
    sim = np.zeros((B, B), dtype=np.float64)
    for i in range(B):
        for j in range(B):
            if i == j:
                continue
            d = _reference_sym2_dtw(trajs[i, :lens[i]], trajs[j, :lens[j]])
            d = norm_fn(
                torch.tensor([d], dtype=torch.float64),
                torch.tensor([float(lens[j])], dtype=torch.float64),
                torch.tensor(float(lens[i]), dtype=torch.float64),
            ).item()
            sim[i, j] = -d
    return sim


@pytest.mark.parametrize("seed", [0, 1, 7])
def test_compute_matches_cpu_reference(artifacts, seed):
    """Random batch: GPU exact DTW == numpy symmetric-2 DTW on true prefixes."""
    rng = np.random.default_rng(seed)
    N = artifacts["trajectories"].shape[0]
    batch_idx = rng.choice(N, size=12, replace=False)

    online = _online_for_indices(artifacts, batch_idx)
    reference = _reference_similarity_matrix(artifacts, batch_idx)

    assert np.all(np.diag(online) == 0.0)
    diff = np.abs(online - reference)
    max_abs = float(diff.max())
    print(f"\n[seed={seed}] max|gpu - reference| = {max_abs:.3e}")
    assert max_abs < 1e-3, (
        f"GPU DTW diverged from CPU reference: max abs diff = {max_abs}."
        f"\n{_first_diffs(online, reference, 5)}"
    )


def test_padding_does_not_change_similarity(artifacts):
    """Extending every clip's boundary-repeat padding must leave the
    similarity matrix unchanged: only the true `L_i × L_j` prefix grid is
    ever visited."""
    from online_dtw import OnlineDTWComputer

    rng = np.random.default_rng(3)
    N = artifacts["trajectories"].shape[0]
    batch_idx = rng.choice(N, size=16, replace=False)
    trajs, lens = _batch(artifacts, batch_idx)

    extra = 17
    tail = trajs[:, -1:, :].expand(-1, extra, -1)
    trajs_more_pad = torch.cat([trajs, tail], dim=1).contiguous()

    computer = OnlineDTWComputer(DESIGN)
    base = computer.compute(trajs, lens).cpu().numpy()
    padded = computer.compute(trajs_more_pad, lens).cpu().numpy()

    assert trajs_more_pad.shape[1] == trajs.shape[1] + extra
    assert np.array_equal(base, padded), (
        f"padding changed the DTW matrix: max diff "
        f"{np.abs(base - padded).max():.3e}"
    )


def test_online_matrix_symmetric(artifacts):
    """Online DTW matrix should be (close to) symmetric for symmetric2."""
    rng = np.random.default_rng(42)
    N = artifacts["trajectories"].shape[0]
    batch_idx = rng.choice(N, size=16, replace=False)

    online = _online_for_indices(artifacts, batch_idx)
    asym = float(np.abs(online - online.T).max())
    print(f"\n[symmetry] max|M - M.T| = {asym:.3e}")
    assert asym < 1e-4


def test_compute_matches_compute_chunked(artifacts):
    """`compute` and `compute_chunked` produce the same matrix."""
    from online_dtw import OnlineDTWComputer

    rng = np.random.default_rng(5)
    N = artifacts["trajectories"].shape[0]
    batch_idx = rng.choice(N, size=20, replace=False)
    trajs, lens = _batch(artifacts, batch_idx)

    computer = OnlineDTWComputer(DESIGN)
    full = computer.compute(trajs, lens).cpu().numpy()
    chunked = computer.compute_chunked(trajs, lens, target_chunk=7).cpu().numpy()
    assert np.abs(full - chunked).max() < 1e-5


def _first_diffs(a, b, k=5):
    diff = np.abs(a - b)
    flat = np.argsort(-diff, axis=None)[:k]
    rows, cols = np.unravel_index(flat, diff.shape)
    return "\n".join(
        f"  ({r},{c}): gpu={a[r, c]:.6f}  reference={b[r, c]:.6f}  "
        f"diff={diff[r, c]:.3e}"
        for r, c in zip(rows, cols)
    )


# ---------------------------------------------------------------------------
# compute_one_query_against_targets — the rectangular query→targets primitive
# used by compute / compute_chunked and by build_dtw_neighbors.
# ---------------------------------------------------------------------------

def test_one_query_matches_compute_chunked_slice(artifacts):
    """For a small N=8 sub-batch, compute_one_query_against_targets(traj_i,
    all_trajs, ...) must equal compute_chunked(...)[i, :] exactly."""
    from online_dtw import OnlineDTWComputer

    rng = np.random.default_rng(7)
    N_full = artifacts["trajectories"].shape[0]
    batch_idx = rng.choice(N_full, size=8, replace=False)
    trajs, lens = _batch(artifacts, batch_idx)
    lens_f = lens.float()

    computer = OnlineDTWComputer(DESIGN)
    full = computer.compute_chunked(trajs, lens, target_chunk=8).cpu().numpy()

    for i in [0, 3, 7]:
        sims = computer.compute_one_query_against_targets(
            trajs[i], trajs, lens_f[i], lens_f,
        ).cpu().numpy()
        # compute_chunked zeros the diagonal post-hoc; compute_one_query does
        # not. Keep the comparison off-diagonal to be strict.
        off = np.delete(np.arange(8), i)
        diff = np.abs(sims[off] - full[i, off])
        assert diff.max() < 1e-5, (
            f"row {i}: max diff {diff.max():.3e} between rectangular helper "
            f"and compute_chunked"
        )


def test_one_query_self_distance_near_zero(artifacts):
    """Helper(traj_i, [traj_i]) should return similarity ≈ 0 (i.e., DTW
    distance ≈ 0)."""
    from online_dtw import OnlineDTWComputer

    trajs = artifacts["trajectories"][:1].cuda().float().contiguous()  # [1, T, D]
    lens = artifacts["lengths"][:1].cuda().float()

    computer = OnlineDTWComputer(DESIGN)
    sims = computer.compute_one_query_against_targets(
        trajs[0], trajs, lens[0], lens,
    ).cpu().numpy()
    assert abs(float(sims[0])) < 1e-3, f"self-similarity = {sims[0]}"
