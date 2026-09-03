"""End-to-end verification: build_trajectories -> VideoClipDataset -> DataLoader -> OnlineDTWComputer.

For 5 batches drawn from the real DataLoader, asserts that the online (B, B)
DTW similarity matrix exactly matches the corresponding submatrix of the
offline `dtw_matrix_{design}.npy` produced by the offline builder
(not distributed with this repo).

Run from retrieval_training/ on a GPU node:
    python tests/end_to_end_dtw_check.py
"""

import shutil
import sys
import tempfile
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

THIS = Path(__file__).resolve()
TRAIN_ROOT = THIS.parent.parent
sys.path.insert(0, str(TRAIN_ROOT))
sys.path.insert(0, str(TRAIN_ROOT / "build_dataset"))

from data import VideoClipDataset, collate_fn  # noqa: E402
from online_dtw import OnlineDTWComputer  # noqa: E402
from build_trajectories import build_trajectories  # noqa: E402

DATA_DIR = TRAIN_ROOT.parent / "outputs" / "training_data"
DESIGN = "first_frame_midpoint_hand_length_no_z"
BATCH_SIZE = 64
NUM_BATCHES = 5
SEED = 0


def main() -> int:
    if not torch.cuda.is_available():
        print("CUDA not available; aborting (online DTW kernel is GPU-only).")
        return 2

    offline_path = DATA_DIR / f"dtw_matrix_{DESIGN}.npy"
    if not offline_path.exists():
        print(f"Offline matrix missing: {offline_path}")
        return 2

    tmp_dir = Path(tempfile.mkdtemp(prefix="e2e_dtw_"))
    print(f"[setup] tmp_dir = {tmp_dir}")
    try:
        traj_path = tmp_dir / f"trajectories_{DESIGN}.pt"

        print("\n[step 1/4] build_trajectories (fresh from clip_keypoints.pt)")
        build_trajectories(
            data_dir=DATA_DIR,
            dtw_design=DESIGN,
            output_path=traj_path,
            num_videos=10000,
        )
        assert traj_path.exists(), traj_path
        emitted_index = tmp_dir / f"clip_index_{DESIGN}.json"
        assert emitted_index.exists(), emitted_index
        print(f"        wrote {traj_path.name} and {emitted_index.name}")

        # Cross-check that the freshly built `clip_keys` line up with the
        # offline `clip_index_{design}.json` row-for-row. If they don't, the
        # offline matrix submatrix indexed by `clip_indices` from the
        # DataLoader will not be comparable.
        print("\n[step 2/4] verify clip_keys row alignment with offline clip_index")
        import json
        offline_idx_path = DATA_DIR / f"clip_index_{DESIGN}.json"
        with open(offline_idx_path) as f:
            raw = json.load(f)
        offline_index = {int(k): v for k, v in raw.items()}

        payload = torch.load(traj_path, weights_only=False, map_location="cpu")
        clip_keys = payload["clip_keys"]
        n_keys = len(clip_keys)
        for i in range(n_keys):
            info = offline_index[i]
            assert clip_keys[i] == (int(info["video_number"]), str(info["node_uid"])), (
                f"row {i} mismatch: trajectories.clip_keys={clip_keys[i]} "
                f"offline_index={(info['video_number'], info['node_uid'])}"
            )
        print(f"        all {n_keys} rows aligned")

        print("\n[step 3/4] open VideoClipDataset + DataLoader")
        dataset = VideoClipDataset(
            data_dir=str(DATA_DIR),
            dtw_design=DESIGN,
            max_clip_frames=-1,
            trajectories_path=str(traj_path),
        )
        gen = torch.Generator().manual_seed(SEED)
        loader = DataLoader(
            dataset,
            batch_size=BATCH_SIZE,
            shuffle=True,
            num_workers=2,
            collate_fn=collate_fn,
            generator=gen,
        )
        print(f"        dataset N={len(dataset)}, batch_size={BATCH_SIZE}")

        print("\n[step 4/4] compute online DTW for 5 batches and diff vs offline")
        offline = np.load(offline_path, mmap_mode="r")
        computer = OnlineDTWComputer(DESIGN)

        all_pass = True
        for b, batch in enumerate(loader):
            if b >= NUM_BATCHES:
                break
            clip_indices = batch["clip_indices"].numpy()  # [B], indexes into N
            trajs = batch["trajectories"].cuda()           # [B, T_max, D]
            lens = batch["traj_lengths"].cuda()            # [B]

            online = computer.compute(trajs, lens).cpu().numpy()  # [B, B]
            offline_sub = np.array(offline[np.ix_(clip_indices, clip_indices)])

            diff = np.abs(online - offline_sub)
            max_abs = float(diff.max())
            n_exact = int((diff == 0).sum())
            B = online.shape[0]
            ok = max_abs < 1e-4
            all_pass = all_pass and ok
            tag = "PASS" if ok else "FAIL"
            print(f"  batch {b}: B={B}  max|online-offline|={max_abs:.3e}  "
                  f"exact={n_exact}/{B*B}  [{tag}]")
            if not ok:
                # Show first few worst cells.
                flat = np.argsort(-diff, axis=None)[:5]
                rows, cols = np.unravel_index(flat, diff.shape)
                for r, c in zip(rows, cols):
                    print(f"      ({r},{c}): online={online[r, c]:.6f}  "
                          f"offline={offline_sub[r, c]:.6f}  "
                          f"diff={diff[r, c]:.3e}  "
                          f"clip_idx=({clip_indices[r]}, {clip_indices[c]})")

        print()
        if all_pass:
            print("RESULT: PASS — online DTW matches offline submatrix for all 5 batches.")
            return 0
        else:
            print("RESULT: FAIL — see batch-level diffs above.")
            return 1
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)
        print(f"[cleanup] removed {tmp_dir}")


if __name__ == "__main__":
    raise SystemExit(main())
