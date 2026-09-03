"""Build depth-grounded trajectory and DTW-neighbor artifacts.

This is a training-oriented wrapper for the body-frame DTW designs defined
in ``dtw_cknna.py``. It reads ``depth_grounded_clip_keypoints.pt``,
optionally filters to clips that have V-JEPA features on disk, writes a
matching ``trajectories_<design>.pt`` per design, then writes the
positive-mining ``dtw_neighbors_<design>.pt`` cache.

Default output:
    outputs/dtw_neighbors/

Smoke test:
    python build_depth_grounded_dtw_neighbors.py \\
        --out-dir /tmp/dtw_neighbors_smoke \\
        --designs abs_wrist_coords \\
        --num-clips 100 \\
        --candidate-k 50 \\
        --overwrite
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

import torch

SCRIPT_DIR = Path(__file__).resolve().parent
TRAINING_DIR = SCRIPT_DIR.parent  # retrieval_training/ (dtw_cknna.py lives here)

for _p in (str(TRAINING_DIR), str(SCRIPT_DIR)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import dtw_cknna_data  # noqa: E402
from build_dtw_neighbors import build_dtw_neighbors_from_payload  # noqa: E402
from dtw_cknna import BODY_FRAME_21J_DESIGNS, DTW_DESIGNS, TRAJ_TRANSFORMS  # noqa: E402
from dtw_cknna_data import load_data_bundle_21j  # noqa: E402


DEFAULT_DATA_DIR = dtw_cknna_data.KEYPOINTS_CACHE.parent
DEFAULT_CACHE = DEFAULT_DATA_DIR / "depth_grounded_clip_keypoints.pt"
DEFAULT_OUT_DIR = DEFAULT_DATA_DIR.parent / "dtw_neighbors"


def _parse_designs(raw: str | None) -> list[str]:
    if raw is None or not raw.strip():
        return list(BODY_FRAME_21J_DESIGNS)
    designs = [d.strip() for d in raw.split(",") if d.strip()]
    unknown = [d for d in designs if d not in DTW_DESIGNS]
    if unknown:
        raise ValueError(
            f"Unknown DTW design(s): {unknown}; valid keys: {sorted(DTW_DESIGNS)}"
        )
    return designs


def _filter_to_featured_clips(bundle: Any, data_dir: Path) -> None:
    """Filter bundle in-place to clips with matching jepa_features files."""
    features_dir = data_dir / "jepa_features"
    available = set()
    if features_dir.is_dir():
        for vdir in features_dir.iterdir():
            if not vdir.is_dir() or vdir.name.startswith("."):
                continue
            try:
                vn = int(vdir.name)
            except ValueError:
                continue
            for pt in vdir.glob("*.pt"):
                available.add((vn, pt.stem))

    keep_idx = [
        i for i, c in enumerate(bundle.clips)
        if (int(c["video_number"]), str(c["node_uid"])) in available
    ]
    n_missing = bundle.N - len(keep_idx)
    if n_missing:
        print(
            f"[depth_neighbors] dropping {n_missing} clips without "
            f"jepa_features ({len(keep_idx)} of {bundle.N} remain)",
            flush=True,
        )

    if len(keep_idx) == bundle.N:
        return
    keep_t = torch.as_tensor(keep_idx, dtype=torch.long)
    bundle.trajs = bundle.trajs.index_select(0, keep_t).contiguous()
    bundle.actual_lengths = bundle.actual_lengths.index_select(0, keep_t).contiguous()
    bundle.clips = [bundle.clips[i] for i in keep_idx]
    bundle.N = len(keep_idx)


def _subsample_clips(bundle: Any, max_clips: int | None, seed: int) -> None:
    if max_clips is None or max_clips <= 0 or max_clips >= bundle.N:
        return
    import numpy as np

    rng = np.random.default_rng(seed)
    keep_idx = sorted(rng.choice(bundle.N, size=max_clips, replace=False).tolist())
    keep_t = torch.as_tensor(keep_idx, dtype=torch.long)
    bundle.trajs = bundle.trajs.index_select(0, keep_t).contiguous()
    bundle.actual_lengths = bundle.actual_lengths.index_select(0, keep_t).contiguous()
    bundle.clips = [bundle.clips[i] for i in keep_idx]
    bundle.N = len(keep_idx)
    print(f"[depth_neighbors] subsampled to {bundle.N} clips (seed={seed})", flush=True)


def _truncate_bundle(bundle: Any, max_t: int) -> None:
    if max_t <= 0 or bundle.T <= max_t:
        return
    n_truncated = int((bundle.actual_lengths > max_t).sum().item())
    print(
        f"[depth_neighbors] truncating T={bundle.T} -> {max_t} "
        f"({n_truncated} clips affected)",
        flush=True,
    )
    bundle.trajs = bundle.trajs[:, :max_t, :].contiguous()
    bundle.actual_lengths = bundle.actual_lengths.clamp(max=max_t)
    bundle.T = max_t


def _load_base_bundle(args: argparse.Namespace) -> Any:
    dtw_cknna_data.KEYPOINTS_CACHE = args.cache
    t0 = time.time()
    bundle = load_data_bundle_21j(None, use_cache=True)
    print(
        f"[depth_neighbors] loaded bundle: N={bundle.N} T={bundle.T} "
        f"D={bundle.trajs.shape[-1]} from {args.cache} in {time.time() - t0:.1f}s",
        flush=True,
    )
    if bundle.trajs.shape[-1] != 126:
        raise ValueError(
            f"Expected 126D 21-joint depth-grounded trajectories, "
            f"got D={bundle.trajs.shape[-1]}"
        )
    if args.skip_feature_check:
        print(
            "[depth_neighbors] skipping jepa_features availability check; "
            f"using all {bundle.N} depth-grounded clips before optional subsampling",
            flush=True,
        )
    else:
        _filter_to_featured_clips(bundle, args.data_dir)
    _subsample_clips(bundle, args.num_clips, args.seed)
    _truncate_bundle(bundle, args.max_t)
    if bundle.N <= 1:
        raise ValueError(f"Need at least 2 clips after filtering; got N={bundle.N}")
    return bundle


def _write_trajectory_artifact(
    bundle: Any,
    design: str,
    output_path: Path,
) -> tuple[dict[str, Any], dict[str, Any]]:
    tx_fn = TRAJ_TRANSFORMS[DTW_DESIGNS[design]["tx"]]
    trajs_tx, lens_tx = tx_fn(bundle.trajs, bundle.actual_lengths)
    trajs_tx = trajs_tx.to(torch.float32).contiguous()
    lens_tx = lens_tx.to(torch.int32).contiguous()
    clip_keys = [(int(c["video_number"]), str(c["node_uid"])) for c in bundle.clips]
    N, T, D = trajs_tx.shape
    payload = {
        "trajectories": trajs_tx,
        "lengths": lens_tx,
        "design": design,
        "clip_keys": clip_keys,
        "T_max": T,
        "D": D,
        "num_videos": None,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, output_path)
    size_mb = output_path.stat().st_size / 1e6
    print(
        f"[depth_neighbors] saved {output_path} "
        f"shape=[{N}, {T}, {D}] ({size_mb:.1f} MB)",
        flush=True,
    )
    return {
        "path": str(output_path),
        "N": int(N),
        "T": int(T),
        "D": int(D),
        "size_mb": size_mb,
    }, payload


def _write_clip_index(bundle: Any, design: str, out_dir: Path) -> Path:
    path = out_dir / f"clip_index_{design}.json"
    clip_index = {
        i: {
            "video_number": int(c["video_number"]),
            "node_uid": str(c["node_uid"]),
        }
        for i, c in enumerate(bundle.clips)
    }
    with open(path, "w") as f:
        json.dump(clip_index, f)
    print(f"[depth_neighbors] saved {path}", flush=True)
    return path


def _write_manifest(out_dir: Path, manifest: dict[str, Any]) -> None:
    path = out_dir / "dtw_neighbors_index.json"
    with open(path, "w") as f:
        json.dump(manifest, f, indent=2, sort_keys=True)
    print(f"[depth_neighbors] saved manifest {path}", flush=True)


def build_depth_grounded_neighbors(args: argparse.Namespace) -> None:
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but torch.cuda.is_available() is false")

    designs = _parse_designs(args.designs)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    bundle = _load_base_bundle(args)
    started_at = time.strftime("%Y-%m-%dT%H:%M:%S%z")

    manifest: dict[str, Any] = {
        "started_at": started_at,
        "cache": str(args.cache),
        "data_dir": str(args.data_dir),
        "out_dir": str(args.out_dir),
        "designs": designs,
        "top_k": int(args.top_k),
        "candidate_k": int(args.candidate_k),
        "pooling": args.pooling,
        "max_t": int(args.max_t),
        "num_clips": None if args.num_clips is None else int(args.num_clips),
        "seed": int(args.seed),
        "artifacts": {},
    }

    for i, design in enumerate(designs, 1):
        print(f"\n[depth_neighbors] [{i}/{len(designs)}] design={design}", flush=True)
        traj_path = args.out_dir / f"trajectories_{design}.pt"
        neigh_path = args.out_dir / f"dtw_neighbors_{design}.pt"
        clip_index_path = args.out_dir / f"clip_index_{design}.json"

        if not args.overwrite and traj_path.exists() and neigh_path.exists():
            print(
                f"[depth_neighbors] skipping existing artifacts for {design}; "
                "pass --overwrite to rebuild",
                flush=True,
            )
            manifest["artifacts"][design] = {
                "trajectory": str(traj_path),
                "neighbors": str(neigh_path),
                "clip_index": str(clip_index_path) if clip_index_path.exists() else None,
                "status": "skipped_existing",
            }
            _write_manifest(args.out_dir, manifest)
            continue

        traj_meta, traj_payload = _write_trajectory_artifact(bundle, design, traj_path)
        clip_index_path = _write_clip_index(bundle, design, args.out_dir)
        print(
            "[depth_neighbors] building neighbors from in-memory trajectories "
            "(skipping trajectory reload)",
            flush=True,
        )
        build_dtw_neighbors_from_payload(
            payload=traj_payload,
            dtw_design=design,
            output_path=neigh_path,
            top_k=args.top_k,
            candidate_k=args.candidate_k,
            pooling=args.pooling,
            device=args.device,
        )
        neigh_size_mb = neigh_path.stat().st_size / 1e6
        manifest["artifacts"][design] = {
            "trajectory": traj_meta,
            "neighbors": {
                "path": str(neigh_path),
                "size_mb": neigh_size_mb,
            },
            "clip_index": str(clip_index_path),
            "status": "built",
        }
        _write_manifest(args.out_dir, manifest)
        del traj_payload

    manifest["finished_at"] = time.strftime("%Y-%m-%dT%H:%M:%S%z")
    _write_manifest(args.out_dir, manifest)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--cache", type=Path, default=DEFAULT_CACHE,
                        help="Depth-grounded keypoint cache")
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--designs", type=str, default=None,
                        help="Comma-separated DTW designs; default is BODY_FRAME_21J_DESIGNS")
    parser.add_argument("--top-k", type=int, default=50)
    parser.add_argument("--candidate-k", type=int, default=1000)
    parser.add_argument("--pooling", choices=["stats4", "keyframes8", "flat_full"],
                        default="stats4")
    parser.add_argument("--max-t", type=int, default=40)
    parser.add_argument("--num-clips", type=int, default=-1,
                        help="Smoke-test cap after feature filtering; -1 uses all clips")
    parser.add_argument("--skip-feature-check", action="store_true",
                        help="Do not require matching jepa_features files; use all valid depth-grounded clips")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", choices=["cuda"], default="cuda")
    parser.add_argument("--overwrite", action="store_true",
                        help="Rebuild artifacts even if both trajectory and neighbor files exist")
    args = parser.parse_args()
    args.num_clips = None if args.num_clips < 0 else args.num_clips
    return args


def main() -> None:
    build_depth_grounded_neighbors(parse_args())


if __name__ == "__main__":
    main()
