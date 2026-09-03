"""
Visualize body frame + hand keypoints from estimate_body_frame() for the first 5 clips.

Outputs GLB scenes to retrieval_training/tests/body_pose_est_tests/{video_number}/{node_uid}/{frame}.glb

Requires `eval_data/torso_relative_clip_keypoints.pt`, which is not
distributed with the repo (see README).

Usage:
    python test_body_pose_est.py            # uses default checkpoint
    python test_body_pose_est.py --ckpt /path/to/model.pt
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import trimesh

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "build_dataset"))
from body_pose_est import (
    _make_hand_viz,
    _make_torso_frame_viz,
    estimate_body_frame,
    load_model,
)

RAW_PATH = Path(__file__).resolve().parents[2] / "eval_data" / "torso_relative_clip_keypoints.pt"
OUT_ROOT = Path(__file__).resolve().parent / "body_pose_est_tests"

def build_frame_scene(result, frame_key):
    hands = result["hands_body"][frame_key]
    parts = _make_torso_frame_viz(
        np.zeros(3, dtype=np.float64),
        np.eye(3, dtype=np.float64),
        axis_len=0.08,
        axis_r=0.002,
    )

    if hands["L"] is not None:
        parts += _make_hand_viz(hands["L"], [60, 120, 255, 255])
    if hands["R"] is not None:
        parts += _make_hand_viz(hands["R"], [255, 120, 60, 255])

    scene = trimesh.Scene()
    for i, mesh in enumerate(parts):
        scene.add_geometry(mesh, geom_name=f"frame_{frame_key}_part_{i:03d}")
    scene.metadata["frame_key"] = str(frame_key)
    return scene


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", type=str, default=None)
    parser.add_argument("--n-clips", type=int, default=5)
    args = parser.parse_args()

    clips = torch.load(RAW_PATH, map_location="cpu", weights_only=False)
    clips = clips[: args.n_clips]
    print(f"Loaded {len(clips)} clips from {RAW_PATH.name}")

    model, device = load_model(args.ckpt)
    print(f"Model loaded on {device}")

    for ci, clip in enumerate(clips):
        vn = clip.get("video_number", "unknown")
        node_number = clip.get("node_number", "unknown")
        nuid = clip.get("node_uid", f"clip_{ci}")
        result = estimate_body_frame(clip, model, device)
        if result is None:
            print(f"  [{ci+1}] vn={vn} node_number={node_number} node={nuid} — skipped (no hands)")
            continue

        clip_dir = OUT_ROOT / str(vn) / str(nuid)
        n_frames = 0
        for fk, hands in result["hands_body"].items():
            if hands["L"] is None and hands["R"] is None:
                continue
            out_path = clip_dir / f"{fk}.glb"
            out_path.parent.mkdir(parents=True, exist_ok=True)
            build_frame_scene(result, fk).export(str(out_path))
            n_frames += 1

        print(f"  [{ci+1}] vn={vn} node_number={node_number} node={nuid} — {n_frames} scenes → {clip_dir}")

    print("Done.")


if __name__ == "__main__":
    main()
