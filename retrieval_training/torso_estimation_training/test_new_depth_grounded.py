"""Display raw depth-grounded hand keypoints against an origin frame.

This script reads torso_relative_clip_keypoints.pt, selects one clip/frame, and
shows the hand keypoints in the same coordinate frame as an RGB triad located
at (0, 0, 0).
"""

from __future__ import annotations

import argparse
import random
import sys
import threading
import time
from pathlib import Path

import numpy as np
import torch
import trimesh
from trimesh.viewer.windowed import SceneViewer


DIR = Path(__file__).resolve().parent
KEYPOINTS_PATH = DIR.parents[1] / "eval_data" / "torso_relative_clip_keypoints.pt"

SPHERE_TEMPLATE = trimesh.creation.uv_sphere(radius=1.0, count=[10, 10])
CYLINDER_TEMPLATE = trimesh.creation.cylinder(radius=1.0, height=1.0)

MANO_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (0, 9), (9, 10), (10, 11), (11, 12),
    (0, 13), (13, 14), (14, 15), (15, 16),
    (0, 17), (17, 18), (18, 19), (19, 20),
]


def numeric_frame_keys(frame_dict):
    return sorted([k for k in frame_dict.keys() if str(k).isdigit()], key=lambda x: int(x))


def playable_frame_keys(frame_dict):
    return [k for k in numeric_frame_keys(frame_dict) if frame_dict[k].get("L") or frame_dict[k].get("R")]


def clip_id(clip):
    return clip.get("node_uid") or (clip.get("video_number"), clip.get("node_number"))


def rotation_between_vectors(src, dst):
    src = np.asarray(src, dtype=np.float64)
    dst = np.asarray(dst, dtype=np.float64)
    src = src / (np.linalg.norm(src) + 1e-12)
    dst = dst / (np.linalg.norm(dst) + 1e-12)

    cross = np.cross(src, dst)
    cross_norm = np.linalg.norm(cross)
    dot = float(np.clip(np.dot(src, dst), -1.0, 1.0))
    if cross_norm < 1e-12:
        if dot > 0:
            return np.eye(3)
        axis = np.array([1.0, 0.0, 0.0])
        if abs(src[0]) > 0.9:
            axis = np.array([0.0, 1.0, 0.0])
        axis = np.cross(src, axis)
        axis = axis / (np.linalg.norm(axis) + 1e-12)
        angle = np.pi
    else:
        axis = cross / cross_norm
        angle = np.arctan2(cross_norm, dot)

    x, y, z = axis
    K = np.array([[0.0, -z, y], [z, 0.0, -x], [-y, x, 0.0]])
    return np.eye(3) + np.sin(angle) * K + (1.0 - np.cos(angle)) * (K @ K)


def make_sphere(center, radius, color):
    sphere = SPHERE_TEMPLATE.copy()
    transform = np.eye(4, dtype=np.float64)
    transform[:3, :3] *= radius
    transform[:3, 3] = np.asarray(center, dtype=np.float64)
    sphere.apply_transform(transform)
    sphere.visual.vertex_colors = color
    return sphere


def make_cylinder_between(a, b, radius, color):
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    vec = b - a
    length = np.linalg.norm(vec)
    if length < 1e-10:
        return None

    transform = np.eye(4, dtype=np.float64)
    transform[:3, :3] = rotation_between_vectors([0.0, 0.0, 1.0], vec / length)
    transform[:3, :3] = transform[:3, :3] @ np.diag([radius, radius, length])
    transform[:3, 3] = (a + b) * 0.5

    cylinder = CYLINDER_TEMPLATE.copy()
    cylinder.apply_transform(transform)
    cylinder.visual.vertex_colors = color
    return cylinder


def make_origin_frame(axis_len=0.25, axis_r=0.005):
    origin = np.zeros(3, dtype=np.float64)
    parts = [make_sphere(origin, axis_r * 3.0, [255, 255, 255, 255])]
    for axis, color in [
        (np.array([axis_len, 0.0, 0.0]), [255, 0, 0, 255]),
        (np.array([0.0, axis_len, 0.0]), [0, 255, 0, 255]),
        (np.array([0.0, 0.0, axis_len]), [0, 0, 255, 255]),
    ]:
        cyl = make_cylinder_between(origin, axis, axis_r, color)
        if cyl is not None:
            parts.append(cyl)
    return parts


def make_hand(kpts, color):
    pts = np.asarray(kpts, dtype=np.float64).reshape(-1, 3)
    if pts.shape[0] < 21:
        raise ValueError(f"Expected at least 21 hand keypoints, got shape {pts.shape}")

    span = max(float(np.ptp(pts, axis=0).max()), 0.01)
    sphere_r = max(span * 0.035, 0.004)
    parts = []
    for i, p in enumerate(pts):
        parts.append(make_sphere(p, sphere_r * (1.6 if i == 0 else 1.0), color))
    for i, j in MANO_CONNECTIONS:
        cyl = make_cylinder_between(pts[i], pts[j], sphere_r * 0.35, color)
        if cyl is not None:
            parts.append(cyl)
    return parts


def choose_clip(clips, video_number=None, node_number=None, clip_index=0):
    matches = []
    for clip in clips:
        if video_number is not None and clip.get("video_number") != video_number:
            continue
        if node_number is not None and clip.get("node_number") != node_number:
            continue
        matches.append(clip)

    if not matches:
        raise ValueError(
            f"No clip matched video_number={video_number} node_number={node_number}"
        )
    if clip_index < 0 or clip_index >= len(matches):
        raise IndexError(f"clip_index={clip_index} out of range for {len(matches)} matches")
    return matches[clip_index]


def matching_clips(clips, video_number=None, node_number=None):
    matches = []
    for clip in clips:
        if video_number is not None and clip.get("video_number") != video_number:
            continue
        if node_number is not None and clip.get("node_number") != node_number:
            continue
        frame_dict = clip.get("keypoints_per_frame", {})
        if playable_frame_keys(frame_dict):
            matches.append(clip)
    return matches


def select_frame(frame_dict, frame_key=None, frame_index=0):
    frame_keys = numeric_frame_keys(frame_dict)
    if not frame_keys:
        raise ValueError("Selected clip has no numeric frame keys")
    if frame_key is None:
        if frame_index < 0 or frame_index >= len(frame_keys):
            raise IndexError(f"frame_index={frame_index} out of range for {len(frame_keys)} frames")
        frame_key = frame_keys[frame_index]
    if frame_key not in frame_dict:
        candidates = [str(frame_key)]
        if isinstance(frame_key, str) and frame_key.isdigit():
            candidates.append(int(frame_key))
        for candidate in candidates:
            if candidate in frame_dict:
                frame_key = candidate
                break
    if frame_key not in frame_dict:
        raise KeyError(f"Frame {frame_key!r} not found. Available first keys: {frame_keys[:10]}")
    return frame_key, frame_dict[frame_key], frame_keys


def hand_keypoints(frame, hand_id):
    entries = frame.get(hand_id)
    if not entries:
        return None
    if not isinstance(entries, (list, tuple)) or not entries:
        raise ValueError(f"Frame hand {hand_id!r} has unexpected value: {type(entries)}")
    first = entries[0]
    if "kpts_3d" not in first:
        raise KeyError(f"Frame hand {hand_id!r} entry has no 'kpts_3d' key")
    return np.asarray(first["kpts_3d"], dtype=np.float64).reshape(-1, 3)


def frame_origin(hands, center):
    present = [pts for pts in hands.values() if pts is not None]
    if not present:
        raise ValueError("Frame has no L/R hand keypoints")

    if center == "left-wrist":
        if hands["L"] is None:
            raise ValueError("--center left-wrist requested, but no left hand is present")
        return hands["L"][0]
    if center == "right-wrist":
        if hands["R"] is None:
            raise ValueError("--center right-wrist requested, but no right hand is present")
        return hands["R"][0]
    if center == "hands-centroid":
        return np.concatenate(present, axis=0).mean(axis=0)
    return np.zeros(3, dtype=np.float64)


def build_frame_mesh(frame, center):
    hands = {
        "L": hand_keypoints(frame, "L"),
        "R": hand_keypoints(frame, "R"),
    }
    origin = frame_origin(hands, center)

    parts = make_origin_frame()
    if hands["L"] is not None:
        parts += make_hand(hands["L"] - origin, [60, 120, 255, 255])
    if hands["R"] is not None:
        parts += make_hand(hands["R"] - origin, [255, 120, 60, 255])
    return trimesh.util.concatenate(parts), origin


def make_stdin_enter_event():
    evt = threading.Event()

    def reader():
        while True:
            line = sys.stdin.readline()
            if not line:
                return
            evt.set()

    thread = threading.Thread(target=reader, daemon=True)
    thread.start()
    return evt


def describe_clip(prefix, clip, clip_index, frame_key, frame_index, frame_count, origin):
    print(
        f"{prefix} clip_index={clip_index} clip={clip_id(clip)} "
        f"video={clip.get('video_number')} node={clip.get('node_number')} "
        f"frame={frame_key} ({frame_index + 1}/{frame_count}) "
        f"subtracted_origin={origin.tolist()}",
        flush=True,
    )


def play_random_clip_viewer(clips, start_index, center, fps, rng):
    if not clips:
        raise ValueError("No clips with numeric frame keys matched the requested filters")

    enter_event = make_stdin_enter_event()
    period_seconds = 1.0 / fps

    state = {
        "clip_index": start_index,
        "frame_index": 0,
        "last_switch": time.monotonic(),
    }

    def current_frame():
        clip = clips[state["clip_index"]]
        frame_dict = clip["keypoints_per_frame"]
        frame_keys = playable_frame_keys(frame_dict)
        state["frame_index"] %= len(frame_keys)
        frame_key = frame_keys[state["frame_index"]]
        return clip, frame_dict, frame_keys, frame_key, frame_dict[frame_key]

    def build_current_mesh(prefix):
        clip, _, frame_keys, frame_key, frame = current_frame()
        mesh, origin = build_frame_mesh(frame, center)
        describe_clip(
            prefix,
            clip,
            state["clip_index"],
            frame_key,
            state["frame_index"],
            len(frame_keys),
            origin,
        )
        return mesh

    scene = trimesh.Scene()
    scene.add_geometry(build_current_mesh("show"), geom_name="frame_mesh")
    print(
        f"Loaded {len(clips)} clips. Playing at {fps:g} fps. "
        "Press Enter in this terminal to switch to a random different clip.",
        flush=True,
    )

    def choose_random_different_clip():
        if len(clips) <= 1:
            return state["clip_index"]
        choices = [i for i in range(len(clips)) if i != state["clip_index"]]
        return rng.choice(choices)

    def callback(scene_obj):
        changed_clip = False
        if enter_event.is_set():
            enter_event.clear()
            state["clip_index"] = choose_random_different_clip()
            state["frame_index"] = 0
            state["last_switch"] = time.monotonic()
            changed_clip = True
        else:
            now = time.monotonic()
            if now - state["last_switch"] < period_seconds:
                return
            state["last_switch"] = now
            state["frame_index"] += 1

        prefix = "switch" if changed_clip else "show"
        try:
            scene_obj.geometry["frame_mesh"] = build_current_mesh(prefix)
        except Exception as exc:
            print(f"Skipping bad frame: {type(exc).__name__}: {exc}", flush=True)
            state["frame_index"] += 1

    try:
        SceneViewer(scene, callback=callback, callback_period=1.0 / 30.0, start_loop=True)
    except SystemExit:
        pass


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", type=Path, default=KEYPOINTS_PATH)
    parser.add_argument("--video-number", type=int, default=None)
    parser.add_argument("--node-number", type=int, default=None)
    parser.add_argument("--clip-index", type=int, default=0)
    parser.add_argument(
        "--center",
        choices=["none", "left-wrist", "right-wrist", "hands-centroid"],
        default="none",
        help="Subtract an origin from the keypoints before display. Default keeps raw depth-grounded coordinates.",
    )
    parser.add_argument("--fps", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args()

    if args.fps <= 0:
        parser.error("--fps must be positive")

    clips = torch.load(args.path, map_location="cpu", weights_only=False)
    clips = matching_clips(
        clips,
        video_number=args.video_number,
        node_number=args.node_number,
    )
    if args.clip_index < 0 or args.clip_index >= len(clips):
        parser.error(f"--clip-index must be in [0, {len(clips) - 1}] for the matched clips")

    print(
        f"Loaded {args.path}\n"
        "viewer axes: X=red, Y=green, Z=blue; L=blue hand, R=orange hand",
        flush=True,
    )
    play_random_clip_viewer(
        clips,
        start_index=args.clip_index,
        center=args.center,
        fps=args.fps,
        rng=random.Random(args.seed),
    )


if __name__ == "__main__":
    main()
