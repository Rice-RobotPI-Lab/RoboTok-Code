"""End-to-end equivalence test for online per-batch DTW vs the offline matrix.

Validates that:
    OnlineDTWComputer.compute(trajs[batch_idx], lengths[batch_idx])
        == dtw_matrix[np.ix_(batch_idx, batch_idx)]

within float-precision tolerance, for the precomputed
`dtw_matrix_{design}.npy` and `trajectories_{design}.pt` artifacts.

Skipped automatically if either artifact is missing or CUDA is unavailable
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
DESIGN = "first_frame_midpoint_hand_length_no_z"


@pytest.fixture(scope="module")
def artifacts():
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available; online DTW kernel requires GPU")
    traj_path = DATA_DIR / f"trajectories_{DESIGN}.pt"
    mat_path = DATA_DIR / f"dtw_matrix_{DESIGN}.npy"
    if not traj_path.exists():
        pytest.skip(f"Trajectory artifact missing: {traj_path}")
    if not mat_path.exists():
        pytest.skip(f"Offline matrix missing: {mat_path}")

    payload = torch.load(traj_path, weights_only=False, map_location="cpu")
    assert payload["design"] == DESIGN, (
        f"trajectory file design={payload['design']} != {DESIGN}"
    )
    matrix = np.load(mat_path, mmap_mode="r")
    assert matrix.shape[0] == payload["trajectories"].shape[0], (
        f"matrix N={matrix.shape[0]} != trajectories N={payload['trajectories'].shape[0]}"
    )
    return payload, matrix


def _online_for_indices(payload, batch_idx, design=DESIGN):
    from online_dtw import OnlineDTWComputer

    trajs = payload["trajectories"][batch_idx].cuda()
    lens = payload["lengths"][batch_idx].cuda()
    computer = OnlineDTWComputer(design)
    sims = computer.compute(trajs, lens)
    return sims.cpu().numpy()


def _offline_for_indices(matrix, batch_idx):
    return np.array(matrix[np.ix_(batch_idx, batch_idx)])


@pytest.mark.parametrize("seed", [0, 1, 7])
def test_online_matches_offline_random_batch(artifacts, seed):
    """Random batch of 32 clips: online (B,B) == offline submatrix."""
    payload, matrix = artifacts
    rng = np.random.default_rng(seed)
    N = matrix.shape[0]
    batch_idx = rng.choice(N, size=32, replace=False)

    online = _online_for_indices(payload, batch_idx)
    offline = _offline_for_indices(matrix, batch_idx)

    # Diagonal: offline matrix has 0.0 on the diagonal (distance-to-self,
    # then negated). Online forces diagonal to 0.0 explicitly. Both should
    # agree exactly.
    assert np.all(np.diag(online) == 0.0)
    assert np.all(np.diag(offline) == 0.0)

    # Off-diagonal: should match within float32 precision. The kernel and
    # cost computation are deterministic, but the offline matrix was built
    # row-major with `all_vecs[i+1:]`-style slicing while we compute the full
    # row here — sub-batch slicing of the cost tensor does not change FP
    # results, so this should be bit-equal in practice. Allow a tiny atol
    # to absorb any non-determinism we missed.
    diff = np.abs(online - offline)
    max_abs = float(diff.max())
    n_exact = int((diff == 0).sum())
    print(f"\n[seed={seed}] max|online - offline| = {max_abs:.3e}, "
          f"exact-match cells = {n_exact}/{online.size}")

    assert max_abs < 1e-4, (
        f"Online vs offline DTW similarity diverged: max abs diff "
        f"= {max_abs}. Sample mismatches:\n{_first_diffs(online, offline, 5)}"
    )


def test_online_matches_offline_dense_batch(artifacts):
    """A larger batch (128) — same equivalence."""
    payload, matrix = artifacts
    rng = np.random.default_rng(123)
    N = matrix.shape[0]
    batch_idx = rng.choice(N, size=128, replace=False)

    online = _online_for_indices(payload, batch_idx)
    offline = _offline_for_indices(matrix, batch_idx)

    diff = np.abs(online - offline)
    max_abs = float(diff.max())
    print(f"\n[B=128] max|online - offline| = {max_abs:.3e}")
    assert max_abs < 1e-4


def test_online_matrix_symmetric(artifacts):
    """Online DTW matrix should be (close to) symmetric for symmetric2."""
    payload, _ = artifacts
    rng = np.random.default_rng(42)
    N = payload["trajectories"].shape[0]
    batch_idx = rng.choice(N, size=16, replace=False)

    online = _online_for_indices(payload, batch_idx)
    asym = float(np.abs(online - online.T).max())
    print(f"\n[symmetry] max|M - M.T| = {asym:.3e}")
    assert asym < 1e-4


def _first_diffs(a, b, k=5):
    diff = np.abs(a - b)
    flat = np.argsort(-diff, axis=None)[:k]
    rows, cols = np.unravel_index(flat, diff.shape)
    return "\n".join(
        f"  ({r},{c}): online={a[r, c]:.6f}  offline={b[r, c]:.6f}  "
        f"diff={diff[r, c]:.3e}"
        for r, c in zip(rows, cols)
    )


# ---------------------------------------------------------------------------
# compute_one_query_against_targets — the rectangular query→targets primitive
# extracted from compute_chunked's inner loop. Used by build_dtw_neighbors.
# ---------------------------------------------------------------------------

def test_one_query_matches_compute_chunked_slice(artifacts):
    """For a small N=8 sub-batch, compute_one_query_against_targets(traj_i,
    all_trajs, ...) must equal compute_chunked(...)[i, :] exactly. Bit-equal
    by construction — both paths use the same primitive — but we anchor it
    here so any future refactor that breaks the contract fails loudly."""
    from online_dtw import OnlineDTWComputer

    payload, _ = artifacts
    rng = np.random.default_rng(7)
    N_full = payload["trajectories"].shape[0]
    batch_idx = rng.choice(N_full, size=8, replace=False)

    trajs = payload["trajectories"][batch_idx].cuda().float().contiguous()
    lens = payload["lengths"][batch_idx].cuda()
    lens_f = lens.float()

    computer = OnlineDTWComputer(DESIGN)
    full = computer.compute_chunked(trajs, lens, target_chunk=8).cpu().numpy()

    # Pick a query row and compute the rectangular helper against ALL targets
    # (including self — caller is responsible for self-handling).
    for i in [0, 3, 7]:
        sims = computer.compute_one_query_against_targets(
            trajs[i], trajs, lens_f[i], lens_f,
        ).cpu().numpy()
        # compute_chunked zeros the diagonal post-hoc; compute_one_query does
        # not (the kernel returns a near-zero distance to self anyway). Keep
        # the comparison off-diagonal to be strict.
        off = np.delete(np.arange(8), i)
        diff = np.abs(sims[off] - full[i, off])
        assert diff.max() < 1e-5, (
            f"row {i}: max diff {diff.max():.3e} between rectangular helper "
            f"and compute_chunked"
        )


def test_one_query_self_distance_near_zero(artifacts):
    """Helper(traj_i, [traj_i]) should return similarity ≈ 0 (i.e., DTW
    distance ≈ 0) — sanity-check that self-comparison doesn't have a sign
    bug."""
    from online_dtw import OnlineDTWComputer

    payload, _ = artifacts
    trajs = payload["trajectories"][:1].cuda().float().contiguous()  # [1, T, D]
    lens = payload["lengths"][:1].cuda().float()

    computer = OnlineDTWComputer(DESIGN)
    sims = computer.compute_one_query_against_targets(
        trajs[0], trajs, lens[0], lens,
    ).cpu().numpy()
    # Single target = self → similarity should be ≈ 0 (negated distance,
    # distance to self is ~0 under symmetric2). Use tolerance because the
    # raw kernel may have tiny FP noise and norm_fn may divide by lengths.
    assert abs(float(sims[0])) < 1e-3, f"self-similarity = {sims[0]}"


def test_exact_matches_truncated_reference(artifacts):
    """`compute_one_query_against_targets(exact=True)` must equal a reference
    that truncates trajectories to (L_i, L_j) and feeds them through the
    existing padded path with T_max = max(L_i, L_j) (so padding region is
    empty). Anchors the bit-equivalence contract claimed in the docstring."""
    from online_dtw import OnlineDTWComputer

    payload, _ = artifacts
    rng = np.random.default_rng(11)
    N_full = payload["trajectories"].shape[0]
    batch_idx = rng.choice(N_full, size=8, replace=False)

    trajs = payload["trajectories"][batch_idx].cuda().float().contiguous()
    lens = payload["lengths"][batch_idx].cuda()
    lens_f = lens.float()

    computer = OnlineDTWComputer(DESIGN)

    for i in [0, 3, 7]:
        L_i = int(lens[i].item())

        sims_exact = computer.compute_one_query_against_targets(
            trajs[i], trajs, lens_f[i], lens_f, exact=True,
        ).cpu().numpy()

        # Reference: per target, truncate query to L_i and target to L_j,
        # then run the (padded) helper on a 1-target batch where the grid
        # is exactly L_i x L_j.
        sims_ref = np.empty(trajs.shape[0], dtype=np.float32)
        for j in range(trajs.shape[0]):
            L_j = int(lens[j].item())
            q_trunc = trajs[i, :L_i].contiguous()
            t_trunc = trajs[j:j + 1, :L_j, :].contiguous()
            s = computer.compute_one_query_against_targets(
                q_trunc, t_trunc, lens_f[i], lens_f[j:j + 1], exact=False,
            ).cpu().numpy()
            sims_ref[j] = s[0]

        diff = np.abs(sims_exact - sims_ref)
        assert diff.max() < 1e-4, (
            f"row {i}: max diff {diff.max():.3e} between exact path and "
            f"per-pair truncated reference"
        )


def test_compute_chunked_exact_matches_one_query(artifacts):
    """`compute_chunked(exact=True)` row i must equal `compute_one_query_
    against_targets(..., exact=True)` for traj_i vs the batch. Locks in the
    contract that `compute_chunked` just iterates the rectangular helper."""
    from online_dtw import OnlineDTWComputer

    payload, _ = artifacts
    rng = np.random.default_rng(13)
    N_full = payload["trajectories"].shape[0]
    batch_idx = rng.choice(N_full, size=8, replace=False)

    trajs = payload["trajectories"][batch_idx].cuda().float().contiguous()
    lens = payload["lengths"][batch_idx].cuda()
    lens_f = lens.float()

    computer = OnlineDTWComputer(DESIGN)
    full = computer.compute_chunked(
        trajs, lens, target_chunk=8, exact=True,
    ).cpu().numpy()

    for i in [0, 4, 7]:
        sims = computer.compute_one_query_against_targets(
            trajs[i], trajs, lens_f[i], lens_f, exact=True,
        ).cpu().numpy()
        off = np.delete(np.arange(8), i)
        diff = np.abs(sims[off] - full[i, off])
        assert diff.max() < 1e-5, (
            f"row {i}: max diff {diff.max():.3e} between exact helper and "
            f"compute_chunked(exact=True)"
        )
