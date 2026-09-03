"""Visualize first clip's hand keypoints in a gravity-aligned world frame."""

import argparse
import atexit
import os
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
import numpy as np
import trimesh
import torch
from pathlib import Path
from torso_est_utils import (
    fit_smpl_mesh_from_wrist_frames,
    wrist_frame_from_keypoints,
    make_wrist_frame_viz,
)

DIR = Path(__file__).resolve().parent
KEYPOINTS_PATH = DIR.parents[1] / "eval_data" / "torso_relative_clip_keypoints.pt"
DEFAULT_PROCESS_ALL_OUTPUT_PATH = DIR / "depth_grounded_clip_keypoints_transforms.pt"
SHOW_SMPL_SEPARATELY = False
WORLD_HAND_CENTER = np.array([0.0, 1.0, 1.2])
FRAME_GRAVITY_IS_UP = True
FLIP_HAND_Y_ABOUT_WRIST = True
FLIP_HAND_Z_ABOUT_WRIST = True
SMPL_OFFSET_DISTANCE_M = 0.3
SMPL_HAND_FRAME_LOCAL_OFFSET = np.array([0.0, 0.0, 0.0])
FRAME_DURATION_SECONDS = 1.0
VIDEO_NUMBER = 3
SINGLE_HAND_CIRCLE_SAMPLES = 16
GLOBAL_SEARCH_COARSE_STRIDE = 2
GLOBAL_SEARCH_TOP_K = 8
GLOBAL_SEARCH_EARLY_ABORT_EPS = 1e-4
GLOBAL_SEARCH_DEDUP_ROUND_DECIMALS = 5

_SPHERE_TEMPLATE = trimesh.creation.uv_sphere(radius=1.0, count=[10, 10])
_CYLINDER_TEMPLATE = trimesh.creation.cylinder(radius=1.0, height=1.0)
_GROUND_PLANE_TEMPLATE = None

JOINT_NAMES = [
    "Wrist",
    "Index1", "Index2", "Index3",
    "Middle1", "Middle2", "Middle3",
    "Pinky1", "Pinky2", "Pinky3",
    "Ring1", "Ring2", "Ring3",
    "Thumb1", "Thumb2", "Thumb3",
    "Thumb4", "Index4", "Middle4", "Ring4", "Pinky4",
]

MANO_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (0, 9), (9, 10), (10, 11), (11, 12),
    (0, 13), (13, 14), (14, 15), (15, 16),
    (0, 17), (17, 18), (18, 19), (19, 20),
]


class Profiler:
    def __init__(self):
        self.inclusive_timings = {}
        self.exclusive_timings = {}
        self.counts = {}
        self._stack = []
        self._root_wall = 0.0

    def add(self, name, dt):
        self.inclusive_timings[name] = self.inclusive_timings.get(name, 0.0) + float(dt)
        self.exclusive_timings[name] = self.exclusive_timings.get(name, 0.0) + float(dt)
        self.counts[name] = self.counts.get(name, 0) + 1

    def section(self, name):
        return _ProfileSection(self, name)

    def report(self):
        total = self._root_wall if self._root_wall > 0.0 else sum(self.exclusive_timings.values())
        print("\n=== Profiling Summary ===", flush=True)
        if total <= 0.0:
            print("No timings recorded.", flush=True)
            return
        rows = sorted(self.exclusive_timings.items(), key=lambda kv: kv[1], reverse=True)
        for name, t_excl in rows:
            t_incl = self.inclusive_timings.get(name, t_excl)
            c = self.counts.get(name, 1)
            avg = t_excl / max(c, 1)
            pct = 100.0 * (t_excl / total)
            print(
                f"{name:34s} {t_excl:8.3f}s  ({pct:5.1f}%)  count={c:4d} avg={avg:.4f}s"
                f"  incl={t_incl:.3f}s",
                flush=True,
            )
        print(f"{'TOTAL (wall, non-overlap)':34s} {total:8.3f}s", flush=True)

    def _enter_section(self, name):
        self._stack.append({
            "name": name,
            "start": time.perf_counter(),
            "child_inclusive": 0.0,
        })

    def _exit_section(self):
        node = self._stack.pop()
        dt = time.perf_counter() - node["start"]
        excl = max(0.0, dt - node["child_inclusive"])
        name = node["name"]

        self.inclusive_timings[name] = self.inclusive_timings.get(name, 0.0) + dt
        self.exclusive_timings[name] = self.exclusive_timings.get(name, 0.0) + excl
        self.counts[name] = self.counts.get(name, 0) + 1

        if self._stack:
            self._stack[-1]["child_inclusive"] += dt
        else:
            self._root_wall += dt


class _ProfileSection:
    def __init__(self, profiler, name):
        self.profiler = profiler
        self.name = name

    def __enter__(self):
        self.profiler._enter_section(self.name)
        return self

    def __exit__(self, exc_type, exc, tb):
        self.profiler._exit_section()
        return False


def _depth_shade(base_color, z_min, z_max, z):
    """Darken base_color proportionally: larger z (farther from camera) → darker."""
    t = (z - z_min) / (z_max - z_min) if z_max > z_min else 0.0
    shade = 0.4 + 0.6 * t
    return [int(c * shade) for c in base_color[:3]] + [base_color[3]]


def _safe_unit(vec, fallback=None, eps=1e-8):
    arr = np.asarray(vec, dtype=np.float64)
    norm = np.linalg.norm(arr)
    if norm < eps:
        if fallback is None:
            fallback = np.array([0.0, 1.0, 0.0])
        return np.asarray(fallback, dtype=np.float64)
    return arr / norm


def _skew(vec):
    x, y, z = vec
    return np.array([[0.0, -z, y],
                     [z, 0.0, -x],
                     [-y, x, 0.0]], dtype=np.float64)


def _orthogonal_axis(vec):
    vec = _safe_unit(vec)
    basis = np.array([1.0, 0.0, 0.0])
    if abs(np.dot(vec, basis)) > 0.9:
        basis = np.array([0.0, 1.0, 0.0])
    return _safe_unit(np.cross(vec, basis))


def _axis_angle_to_matrix(axis, angle):
    axis = _safe_unit(axis)
    K = _skew(axis)
    return np.eye(3) + np.sin(angle) * K + (1.0 - np.cos(angle)) * (K @ K)


def _rotation_between_vectors(src, dst):
    src = _safe_unit(src)
    dst = _safe_unit(dst)
    cross = np.cross(src, dst)
    dot = float(np.clip(np.dot(src, dst), -1.0, 1.0))
    cross_norm = np.linalg.norm(cross)
    if cross_norm < 1e-8:
        if dot > 0.0:
            return np.eye(3)
        return _axis_angle_to_matrix(_orthogonal_axis(src), np.pi)
    axis = cross / cross_norm
    angle = np.arctan2(cross_norm, dot)
    return _axis_angle_to_matrix(axis, angle)


def _make_sphere(center, radius, color):
    s = _SPHERE_TEMPLATE.copy()
    transform = np.eye(4, dtype=np.float64)
    transform[:3, :3] *= radius
    transform[:3, 3] = np.asarray(center, dtype=np.float64)
    s.apply_transform(transform)
    s.visual.vertex_colors = color
    return s


def _make_cylinder_between(a, b, radius, color):
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    vec = b - a
    length = np.linalg.norm(vec)
    if length < 1e-10:
        return None
    dir_vec = vec / length
    R = _rotation_between_vectors(np.array([0.0, 0.0, 1.0]), dir_vec)
    mid = (a + b) * 0.5

    transform = np.eye(4, dtype=np.float64)
    transform[:3, :3] = R @ np.diag([radius, radius, length])
    transform[:3, 3] = mid

    c = _CYLINDER_TEMPLATE.copy()
    c.apply_transform(transform)
    c.visual.vertex_colors = color
    return c


def rotation_world_from_camera(gravity_cam):
    gravity_cam = _safe_unit(gravity_cam)
    world_down = np.array([0.0, 0.0, -1.0])
    dot = float(np.clip(np.dot(gravity_cam, world_down), -1.0, 1.0))
    axis = np.cross(gravity_cam, world_down)
    axis_norm = np.linalg.norm(axis)

    if axis_norm < 1e-8:
        if dot > 0.0:
            R_align = np.eye(3)
        else:
            R_align = _axis_angle_to_matrix(_orthogonal_axis(gravity_cam), np.pi)
    else:
        R_align = _axis_angle_to_matrix(axis / axis_norm, np.arccos(dot))

    return _apply_yaw_choice(R_align)


def _apply_yaw_choice(R_world_from_cam):
    cam_forward_world = R_world_from_cam @ np.array([0.0, 0.0, 1.0])
    target_forward_world = np.array([0.0, 1.0, 0.0])

    cam_forward_xy = cam_forward_world.copy()
    cam_forward_xy[2] = 0.0
    if np.linalg.norm(cam_forward_xy) < 1e-8:
        return R_world_from_cam

    cam_forward_xy = _safe_unit(cam_forward_xy)
    signed_cross_z = np.cross(cam_forward_xy, target_forward_world)[2]
    signed_dot = np.clip(np.dot(cam_forward_xy, target_forward_world), -1.0, 1.0)
    yaw = np.arctan2(signed_cross_z, signed_dot)
    return _axis_angle_to_matrix(np.array([0.0, 0.0, 1.0]), yaw) @ R_world_from_cam


def transform_camera_points_to_world(kpts_3d, R_world_from_cam, t_world_from_cam):
    pts_cam = np.asarray(kpts_3d, dtype=np.float64)
    return (R_world_from_cam @ pts_cam.T).T + t_world_from_cam


def se3_from_R_t(R, t):
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = np.asarray(R, dtype=np.float64).reshape(3, 3)
    T[:3, 3] = np.asarray(t, dtype=np.float64).reshape(3)
    return T


def flip_about_wrist(kpts_world, flip_y=False, flip_z=False):
    pts = np.asarray(kpts_world, dtype=np.float64).copy()
    if flip_y:
        wrist_y = pts[0, 1]
        pts[:, 1] = 2 * wrist_y - pts[:, 1]
    if flip_z:
        wrist_z = pts[0, 2]
        pts[:, 2] = 2 * wrist_z - pts[:, 2]
    return pts


def estimate_world_translation(hand_points_cam, R_world_from_cam):
    if not hand_points_cam:
        return np.zeros(3, dtype=np.float64)
    pts_cam = np.concatenate(hand_points_cam, axis=0)
    centroid_world_no_t = (R_world_from_cam @ pts_cam.T).T.mean(axis=0)
    return WORLD_HAND_CENTER - centroid_world_no_t


def smpl_offset_from_hand_centroids(hand_centroids_world, distance_m=0.3):
    if len(hand_centroids_world) < 2:
        return np.array([distance_m, 0.0, 0.0], dtype=np.float64)

    line_dir = hand_centroids_world[1] - hand_centroids_world[0]
    line_norm = np.linalg.norm(line_dir)
    if line_norm < 1e-8:
        return np.array([distance_m, 0.0, 0.0], dtype=np.float64)
    line_dir /= line_norm

    # Prefer an orthogonal direction in the world horizontal plane.
    world_up = np.array([0.0, 0.0, 1.0], dtype=np.float64)
    ortho_dir = np.cross(world_up, line_dir)
    ortho_norm = np.linalg.norm(ortho_dir)
    if ortho_norm < 1e-8:
        ortho_dir = np.cross(np.array([1.0, 0.0, 0.0], dtype=np.float64), line_dir)
        ortho_norm = np.linalg.norm(ortho_dir)
        if ortho_norm < 1e-8:
            return np.array([distance_m, 0.0, 0.0], dtype=np.float64)
    ortho_dir /= ortho_norm
    return ortho_dir * distance_m


def smpl_candidate_offsets(hand_centroids_world, distance_m=0.3):
    base = smpl_offset_from_hand_centroids(hand_centroids_world, distance_m)
    if len(hand_centroids_world) < 2:
        # For a single hand, search a full circle around the hand in the
        # horizontal world plane and choose the most natural fit.
        angles = np.linspace(0.0, 2.0 * np.pi, SINGLE_HAND_CIRCLE_SAMPLES,
                             endpoint=False)
        return [
            np.array([distance_m * np.cos(a), distance_m * np.sin(a), 0.0],
                     dtype=np.float64)
            for a in angles
        ]
    return [base, -base]


def make_body_rotation_toward_target(body_pos_world, target_pos_world,
                                     world_up=np.array([0.0, 0.0, 1.0])):
    up = _safe_unit(world_up, fallback=np.array([0.0, 0.0, 1.0]))
    forward = np.asarray(target_pos_world, dtype=np.float64) - np.asarray(
        body_pos_world, dtype=np.float64)
    forward = forward - up * np.dot(forward, up)
    forward = _safe_unit(forward, fallback=np.array([0.0, 1.0, 0.0]))

    # SMPL local axes: +Y up, +Z forward, +X right.
    right = np.cross(up, forward)
    right = _safe_unit(right, fallback=np.array([1.0, 0.0, 0.0]))
    forward = _safe_unit(np.cross(right, up), fallback=forward)
    return np.column_stack([right, up, forward])


def make_hand_viz(kpts_world, color):
    pts = np.asarray(kpts_world, dtype=np.float64)
    hand_span = np.ptp(pts, axis=0).max()
    sphere_r = max(hand_span * 0.06, 0.004)

    z_viz = pts[:, 1]
    z_min = float(z_viz.min())
    z_max = float(z_viz.max())

    parts = []
    for ki, p in enumerate(pts):
        s = _make_sphere(
            p,
            sphere_r * (1.5 if ki == 0 else 1.0),
            _depth_shade(color, z_min, z_max, z_viz[ki]),
        )
        parts.append(s)
    for i, j in MANO_CONNECTIONS:
        mid_z = (z_viz[i] + z_viz[j]) / 2
        cyl = _make_cylinder_between(
            pts[i], pts[j], sphere_r * 0.4,
            _depth_shade(color, z_min, z_max, mid_z),
        )
        if cyl is not None:
            parts.append(cyl)
    return parts


def make_camera_viz(R_world_from_cam, t_world_from_cam,
                    axis_len=0.12, axis_r=0.003):
    origin = np.asarray(t_world_from_cam, dtype=np.float64)
    parts = []

    for axis, color in [
            (np.array([axis_len, 0.0, 0.0]), [255, 0, 0, 255]),
            (np.array([0.0, axis_len, 0.0]), [0, 255, 0, 255]),
            (np.array([0.0, 0.0, axis_len]), [0, 0, 255, 255])]:
        tip = origin + R_world_from_cam @ axis
        cyl = _make_cylinder_between(origin, tip, axis_r, color)
        if cyl is not None:
            parts.append(cyl)

    depth = 0.25
    half_w = 0.12
    half_h = 0.08
    corners_cam = [
        np.array([-half_w, -half_h, depth]),
        np.array([half_w, -half_h, depth]),
        np.array([half_w, half_h, depth]),
        np.array([-half_w, half_h, depth]),
    ]
    corners = [origin + R_world_from_cam @ p for p in corners_cam]
    edges = [(origin, c) for c in corners]
    edges += [(corners[i], corners[(i + 1) % 4]) for i in range(4)]
    for a, b in edges:
        cyl = _make_cylinder_between(a, b, axis_r * 0.6, [255, 255, 255, 255])
        if cyl is not None:
            parts.append(cyl)

    marker = _make_sphere(origin, axis_r * 3.0, [255, 255, 255, 255])
    parts.append(marker)
    return parts


def make_ground_plane(size=2.0, color=(80, 80, 80, 90)):
    global _GROUND_PLANE_TEMPLATE
    if size == 2.0 and tuple(color) == (80, 80, 80, 90):
        if _GROUND_PLANE_TEMPLATE is None:
            _GROUND_PLANE_TEMPLATE = trimesh.creation.box(extents=[2.0, 2.0, 0.002])
            _GROUND_PLANE_TEMPLATE.apply_translation([0.0, 1.0, -0.001])
            _GROUND_PLANE_TEMPLATE.visual.vertex_colors = (80, 80, 80, 90)
        return _GROUND_PLANE_TEMPLATE.copy()

    plane = trimesh.creation.box(extents=[size, size, 0.002])
    plane.apply_translation([0.0, 1.0, -0.001])
    plane.visual.vertex_colors = color
    return plane


def make_torso_frame_viz(origin, R, axis_len=0.20, axis_r=0.005):
    axis_colors = [
        [255, 0, 0, 255],   # X
        [0, 255, 0, 255],   # Y
        [0, 0, 255, 255],   # Z
    ]
    parts = []
    for i, color in enumerate(axis_colors):
        tip = origin + R[:, i] * axis_len
        cyl = _make_cylinder_between(origin, tip, axis_r, color)
        if cyl is not None:
            parts.append(cyl)
    sphere = _make_sphere(origin, axis_r * 2.3, [255, 255, 255, 255])
    parts.append(sphere)
    return parts


def show_scene_for_seconds(mesh, seconds=1.0):
    start_t = time.monotonic()

    def _close_after_timeout(_scene):
        if time.monotonic() - start_t < seconds:
            return
        try:
            import pyglet
            for win in list(pyglet.app.windows):
                win.close()
        except Exception:
            pass

    try:
        mesh.show(viewer="gl", callback=_close_after_timeout,
                  callback_period=1.0 / 30.0)
    except SystemExit:
        return


def _frame_gravity_down_in_camera(frame):
    gravity_cam = np.asarray(frame.get("gravity", [0.0, 1.0, 0.0]),
                             dtype=np.float64)
    if FRAME_GRAVITY_IS_UP:
        gravity_cam = -gravity_cam
    return _safe_unit(gravity_cam, fallback=np.array([0.0, 1.0, 0.0]))


def _frame_hand_points_cam(frame):
    hand_points_cam = []
    for hid in ["L", "R"]:
        if frame.get(hid):
            hand_points_cam.append(
                np.asarray(frame[hid][0]["kpts_3d"], dtype=np.float64))
    return hand_points_cam


def compute_clip_camera_transform(frame_dict, frame_keys):
    gravity_vectors = []
    for k in frame_keys:
        frame = frame_dict[k]
        if "gravity" in frame:
            gravity_vectors.append(_frame_gravity_down_in_camera(frame))

    if gravity_vectors:
        gravity_clip = _safe_unit(np.mean(gravity_vectors, axis=0))
    else:
        gravity_clip = _frame_gravity_down_in_camera({})

    R_world_from_cam = rotation_world_from_camera(gravity_clip)

    t_world_from_cam = np.zeros(3, dtype=np.float64)
    for k in frame_keys:
        hand_points_cam = _frame_hand_points_cam(frame_dict[k])
        if hand_points_cam:
            t_world_from_cam = estimate_world_translation(
                hand_points_cam, R_world_from_cam)
            break

    return R_world_from_cam, t_world_from_cam, gravity_clip


def build_frame_observation(frame, R_world_from_cam, t_world_from_cam, include_visuals=True):
    parts = [make_ground_plane()] if include_visuals else []
    wrist_frames = {}
    hand_points_world = []
    hand_centroids_world = []

    for hid, color in [("L", [60, 120, 255, 255]), ("R", [255, 120, 60, 255])]:
        if not frame.get(hid):
            continue
        hand = frame[hid][0]
        pts_world = transform_camera_points_to_world(
            hand["kpts_3d"], R_world_from_cam, t_world_from_cam)
        pts_world = flip_about_wrist(
            pts_world,
            flip_y=FLIP_HAND_Y_ABOUT_WRIST,
            flip_z=FLIP_HAND_Z_ABOUT_WRIST,
        )
        hand_points_world.append(pts_world)
        hand_centroids_world.append(pts_world.mean(axis=0))
        if include_visuals:
            parts += make_hand_viz(pts_world, color)

        origin, R = wrist_frame_from_keypoints(pts_world, hand_side=hid)
        wrist_frames[hid] = (origin, R)
        if include_visuals:
            parts += make_wrist_frame_viz(origin, R)

    if include_visuals:
        parts += make_camera_viz(R_world_from_cam, t_world_from_cam)

    if hand_points_world:
        all_hand_points_world = np.concatenate(hand_points_world, axis=0)
        hand_midpoint_world = all_hand_points_world.mean(axis=0)
    else:
        hand_midpoint_world = None

    return {
        "parts": parts,
        "wrist_frames": wrist_frames,
        "hand_centroids_world": hand_centroids_world,
        "hand_midpoint_world": hand_midpoint_world,
    }


def camera_to_body_se3(R_world_from_cam, t_world_from_cam, body_pose_world):
    R_world_from_body = np.asarray(body_pose_world["rotation"], dtype=np.float64)
    t_world_from_body = np.asarray(body_pose_world["translation"], dtype=np.float64)
    R_body_from_world = R_world_from_body.T
    R_body_from_cam = R_body_from_world @ R_world_from_cam
    t_body_from_cam = R_body_from_world @ (np.asarray(t_world_from_cam, dtype=np.float64) - t_world_from_body)
    return se3_from_R_t(R_body_from_cam, t_body_from_cam)


def camera_to_wrist_frame_se3_for_frame(frame):
    out = {}
    for hid in ["L", "R"]:
        if not frame.get(hid):
            continue
        pts_cam = np.asarray(frame[hid][0]["kpts_3d"], dtype=np.float64)
        pts_cam = flip_about_wrist(
            pts_cam,
            flip_y=FLIP_HAND_Y_ABOUT_WRIST,
            flip_z=FLIP_HAND_Z_ABOUT_WRIST,
        )
        origin, R = wrist_frame_from_keypoints(pts_cam, hand_side=hid)
        out[hid] = se3_from_R_t(R, origin)
    return out


def process_all_clips_and_save(clips, output_path, profiler=None):
    all_results = []
    seen_videos = set()
    processed_clip_ids = set()
    processed_since_save = 0

    output_path_obj = Path(output_path)
    if output_path_obj.exists():
        try:
            existing = torch.load(output_path_obj, map_location="cpu", weights_only=False)
            existing_clips = existing.get("clips", [])
            if isinstance(existing_clips, list):
                all_results.extend(existing_clips)
                for item in existing_clips:
                    seen_videos.add(item.get("video_number"))
                    cid = item.get("node_uid")
                    if cid is None:
                        cid = (item.get("video_number"), item.get("node_number"))
                    processed_clip_ids.add(cid)
                print(
                    "Resuming process-all from existing output: "
                    f"{len(existing_clips)} clips, {len(seen_videos)} videos already saved.",
                    flush=True,
                )
        except Exception as exc:
            print(
                f"Failed to load existing output for resume ({output_path}): {exc}. "
                "Starting fresh.",
                flush=True,
            )

    def _save_checkpoint():
        payload = {
            "source_path": str(KEYPOINTS_PATH),
            "created_unix_time": time.time(),
            "num_clips_input": len(clips),
            "num_clips_output": len(all_results),
            "num_videos_processed": len(seen_videos),
            "clips": all_results,
        }
        torch.save(payload, output_path)
        return payload

    pending_clips = []
    for clip in clips:
        clip_id = clip.get("node_uid")
        if clip_id is None:
            clip_id = (clip.get("video_number"), clip.get("node_number"))
        if clip_id in processed_clip_ids:
            continue
        pending_clips.append(clip)

    print(
        f"Process-all pending clips: {len(pending_clips)} / {len(clips)}",
        flush=True,
    )
    if not pending_clips:
        return _save_checkpoint()

    max_workers = max(1, min((os.cpu_count() or 1), 16))
    print(f"Using {max_workers} workers for process-all.", flush=True)

    def _consume_result(res, i_progress):
        nonlocal processed_since_save
        if profiler is not None and res.get("timings"):
            for name, dt in res["timings"].items():
                profiler.add(f"process_all_{name}_each", dt)

        if not res.get("ok", False):
            return

        clip_result = res["clip_result"]
        all_results.append(clip_result)
        cid = clip_result.get("node_uid")
        if cid is None:
            cid = (clip_result.get("video_number"), clip_result.get("node_number"))
        processed_clip_ids.add(cid)
        seen_videos.add(clip_result.get("video_number"))
        processed_since_save += 1

        if i_progress % 100 == 0:
            print(f"Completed clips: {i_progress}/{len(pending_clips)}", flush=True)
        if processed_since_save >= 500:
            payload = _save_checkpoint()
            print(
                "Checkpoint save: "
                f"{payload['num_clips_output']} clips, "
                f"{payload['num_videos_processed']} videos -> {output_path}",
                flush=True,
            )
            processed_since_save = 0

    try:
        with ProcessPoolExecutor(max_workers=max_workers) as ex:
            future_map = {ex.submit(_process_single_clip_for_transforms, clip): clip for clip in pending_clips}
            for i, fut in enumerate(as_completed(future_map), start=1):
                clip = future_map[fut]
                try:
                    res = fut.result()
                except Exception as exc:
                    print(
                        f"Failed clip video={clip.get('video_number')} node={clip.get('node_number')}: {exc}",
                        flush=True,
                    )
                    continue
                _consume_result(res, i)
    except (PermissionError, OSError) as exc:
        print(
            f"Parallel process pool unavailable ({exc}). Falling back to serial processing.",
            flush=True,
        )
        for i, clip in enumerate(pending_clips, start=1):
            try:
                res = _process_single_clip_for_transforms(clip)
            except Exception as inner_exc:
                print(
                    f"Failed clip video={clip.get('video_number')} node={clip.get('node_number')}: {inner_exc}",
                    flush=True,
                )
                continue
            _consume_result(res, i)

    payload = _save_checkpoint()
    return payload


def _process_single_clip_for_transforms(clip):
    timings = {}

    frame_dict = clip["keypoints_per_frame"]
    frame_keys = sorted(
        [k for k in frame_dict.keys() if str(k).isdigit()],
        key=lambda x: int(x),
    )
    if not frame_keys:
        return {"ok": False, "timings": timings}

    t0 = time.perf_counter()
    R_world_from_cam, t_world_from_cam, gravity_clip = compute_clip_camera_transform(
        frame_dict, frame_keys
    )
    timings["clip_camera_transform"] = time.perf_counter() - t0

    t0 = time.perf_counter()
    observations = [
        build_frame_observation(
            frame_dict[k], R_world_from_cam, t_world_from_cam, include_visuals=False)
        for k in frame_keys
    ]
    timings["build_observations"] = time.perf_counter() - t0

    t0 = time.perf_counter()
    global_body_pose = select_global_body_pose(observations)
    timings["select_global_body_pose"] = time.perf_counter() - t0
    if global_body_pose is None:
        return {"ok": False, "timings": timings}

    t0 = time.perf_counter()
    fixed_camera_frame_se3 = se3_from_R_t(R_world_from_cam, t_world_from_cam)
    camera_to_body_torso_se3 = camera_to_body_se3(
        R_world_from_cam, t_world_from_cam, global_body_pose)

    camera_to_wrist_frames = {}
    for fk in frame_keys:
        wrist_map = camera_to_wrist_frame_se3_for_frame(frame_dict[fk])
        if wrist_map:
            camera_to_wrist_frames[fk] = wrist_map
    timings["pack_transforms"] = time.perf_counter() - t0

    clip_result = {
        "video_number": clip.get("video_number"),
        "node_number": clip.get("node_number"),
        "node_uid": clip.get("node_uid"),
        "fixed_camera_frame_se3": fixed_camera_frame_se3,
        "camera_to_body_torso_se3": camera_to_body_torso_se3,
        "camera_to_hand_wrist_frames_se3": camera_to_wrist_frames,
        "clip_gravity_down_camera": gravity_clip,
        "frame_keys": frame_keys,
    }
    return {"ok": True, "timings": timings, "clip_result": clip_result}


def candidate_body_poses_for_observation(obs):
    hand_midpoint_world = obs["hand_midpoint_world"]
    if hand_midpoint_world is None:
        return []

    poses = []
    for smpl_offset in smpl_candidate_offsets(
            obs["hand_centroids_world"], SMPL_OFFSET_DISTANCE_M):
        smpl_body_translation = hand_midpoint_world + smpl_offset
        smpl_body_rotation = make_body_rotation_toward_target(
            smpl_body_translation, hand_midpoint_world)
        poses.append({
            "translation": smpl_body_translation,
            "rotation": smpl_body_rotation,
        })
    return poses


def fit_smpl_for_observation(obs, body_pose, return_mesh=True):
    if not obs["wrist_frames"]:
        return None, {"fit_score": 0.0}
    return fit_smpl_mesh_from_wrist_frames(
        obs["wrist_frames"],
        np.array([0.0, 0.0, -1.0]),
        body_translation=body_pose["translation"],
        body_rotation=body_pose["rotation"],
        mesh_color=(180, 180, 200, 65),
        align_body_root=False,
        fit_arms=True,
        return_diagnostics=True,
        hand_frame_local_offset=SMPL_HAND_FRAME_LOCAL_OFFSET,
        return_mesh=return_mesh,
    )


def select_global_body_pose(observations):
    t_start = time.perf_counter()
    frame_candidates = []
    for obs in observations:
        frame_candidates.extend(candidate_body_poses_for_observation(obs))
    if not frame_candidates:
        return None

    # Deduplicate candidate poses to avoid repeated scoring.
    t0 = time.perf_counter()
    unique = {}
    for pose in frame_candidates:
        t = np.round(pose["translation"], GLOBAL_SEARCH_DEDUP_ROUND_DECIMALS)
        r = np.round(pose["rotation"].reshape(-1), GLOBAL_SEARCH_DEDUP_ROUND_DECIMALS)
        key = (tuple(t.tolist()), tuple(r.tolist()))
        if key not in unique:
            unique[key] = pose
    candidates = list(unique.values())
    t_dedup = time.perf_counter() - t0

    t0 = time.perf_counter()
    coarse_idx = list(range(0, len(observations), max(GLOBAL_SEARCH_COARSE_STRIDE, 1)))
    if coarse_idx[-1] != len(observations) - 1:
        coarse_idx.append(len(observations) - 1)
    if 0 not in coarse_idx:
        coarse_idx.insert(0, 0)
    coarse_obs = [observations[i] for i in sorted(set(coarse_idx))]

    coarse_scores = []
    for body_pose in candidates:
        total = 0.0
        for obs in coarse_obs:
            _, diag = fit_smpl_for_observation(obs, body_pose, return_mesh=False)
            total += float(diag["fit_score"])
        coarse_scores.append(total)
    t_coarse = time.perf_counter() - t0

    t0 = time.perf_counter()
    top_k = max(1, min(GLOBAL_SEARCH_TOP_K, len(candidates)))
    top_idx = np.argsort(np.asarray(coarse_scores))[:top_k]
    top_candidates = [candidates[i] for i in top_idx]

    best_pose = None
    best_total = np.inf
    eps = float(GLOBAL_SEARCH_EARLY_ABORT_EPS)
    for body_pose in top_candidates:
        total = 0.0
        for obs in observations:
            _, diag = fit_smpl_for_observation(obs, body_pose, return_mesh=False)
            total += float(diag["fit_score"])
            if total > best_total + eps:
                break
        if total < best_total:
            best_total = total
            best_pose = body_pose
    t_full = time.perf_counter() - t0

    # Optional profiling counters captured in function attributes for caller.
    select_global_body_pose.last_stats = {
        "time_total": time.perf_counter() - t_start,
        "time_dedup": t_dedup,
        "time_coarse": t_coarse,
        "time_full": t_full,
        "num_frame_candidates": len(frame_candidates),
        "num_unique_candidates": len(candidates),
        "num_coarse_frames": len(coarse_obs),
        "num_top_candidates": len(top_candidates),
        "num_full_frames": len(observations),
    }

    return best_pose


def build_frame_mesh(observation, global_body_pose):
    parts = list(observation["parts"])
    z_shift = 0.0
    if global_body_pose is not None and observation["wrist_frames"]:
        torso_origin = np.asarray(global_body_pose["translation"], dtype=np.float64)
        torso_R = np.asarray(global_body_pose["rotation"], dtype=np.float64)
        mesh, _ = fit_smpl_for_observation(observation, global_body_pose)
        if mesh is not None:
            min_z = float(mesh.vertices[:, 2].min())
            if min_z > 0.0:
                z_shift = -min_z
            parts.append(mesh)

        # Move frame to a visible torso/chest anchor rather than pelvis center.
        torso_frame_origin = (
            torso_origin
            + torso_R[:, 1] * 0.20  # up
            + torso_R[:, 2] * 0.08  # forward
        )
        parts += make_torso_frame_viz(
            torso_frame_origin,
            torso_R,
        )
    elif global_body_pose is not None:
        # Keep consistent shift behavior even on frames without wrist fits.
        mesh, _ = fit_smpl_for_observation(observation, global_body_pose)
        if mesh is not None:
            min_z = float(mesh.vertices[:, 2].min())
            if min_z > 0.0:
                z_shift = -min_z

    # Shift all frame geometry together so hand frames/camera remain attached.
    if z_shift != 0.0:
        shifted_parts = []
        for part in parts:
            p = part.copy()
            p.apply_translation([0.0, 0.0, z_shift])
            shifted_parts.append(p)
        parts = shifted_parts
    return trimesh.util.concatenate(parts)


if __name__ == "__main__":
    profiler = Profiler()
    atexit.register(profiler.report)
    parser = argparse.ArgumentParser()
    parser.add_argument("--video-number", type=int, default=None)
    parser.add_argument("--process-all", action="store_true")
    parser.add_argument("--output-path", type=str, default=str(DEFAULT_PROCESS_ALL_OUTPUT_PATH))
    args = parser.parse_args()

    with profiler.section("load_clips_torch"):
        clips = torch.load(KEYPOINTS_PATH, map_location="cpu", weights_only=False)
    if args.process_all:
        with profiler.section("process_all_total"):
            payload = process_all_clips_and_save(
                clips=clips,
                output_path=args.output_path,
                profiler=profiler,
            )
        print(
            "Saved process-all transforms: "
            f"{payload['num_clips_output']} clips -> {args.output_path}",
            flush=True,
        )
        profiler.report()
        raise SystemExit(0)

    with profiler.section("select_clip"):
        if args.video_number is None:
            clip = next(c for c in clips if c["video_number"] == VIDEO_NUMBER)
        else:
            matching = [c for c in clips if c["video_number"] == args.video_number]
            if not matching:
                raise ValueError(f"No clips found for video_number={args.video_number}")
            rng = np.random.default_rng()
            clip = matching[int(rng.integers(len(matching)))]
    frame_dict = clip["keypoints_per_frame"]
    frame_keys = sorted(
        [k for k in frame_dict.keys() if str(k).isdigit()],
        key=lambda x: int(x),
    )

    from trimesh.viewer.windowed import SceneViewer

    with profiler.section("compute_clip_camera_transform"):
        R_world_from_cam_clip, t_world_from_cam_clip, gravity_clip = (
            compute_clip_camera_transform(frame_dict, frame_keys)
        )
    print(
        "Clip gravity (camera-down): "
        f"[{gravity_clip[0]:.4f}, {gravity_clip[1]:.4f}, {gravity_clip[2]:.4f}]"
    )
    print(
        "Fixed camera translation: "
        f"[{t_world_from_cam_clip[0]:.3f}, {t_world_from_cam_clip[1]:.3f}, "
        f"{t_world_from_cam_clip[2]:.3f}]"
    )

    observations = []
    with profiler.section("build_frame_observations_total"):
        for k in frame_keys:
            t0 = time.perf_counter()
            obs = build_frame_observation(
                frame_dict[k], R_world_from_cam_clip, t_world_from_cam_clip)
            observations.append(obs)
            profiler.add("build_frame_observation_each", time.perf_counter() - t0)

    with profiler.section("select_global_body_pose"):
        global_body_pose = select_global_body_pose(observations)
    stats = getattr(select_global_body_pose, "last_stats", None)
    if stats:
        profiler.add("select_body_dedup_stage", stats["time_dedup"])
        profiler.add("select_body_coarse_stage", stats["time_coarse"])
        profiler.add("select_body_full_stage", stats["time_full"])
        print(
            "Global pose search stats: "
            f"frame_candidates={stats['num_frame_candidates']}, "
            f"unique={stats['num_unique_candidates']}, "
            f"coarse_frames={stats['num_coarse_frames']}, "
            f"top_candidates={stats['num_top_candidates']}",
            flush=True,
        )
    if global_body_pose is None:
        print("No valid hand observations for global body-pose selection.")
    else:
        t = global_body_pose["translation"]
        print(
            "Selected global body pose translation: "
            f"[{t[0]:.3f}, {t[1]:.3f}, {t[2]:.3f}]"
        )
    frame_meshes = []
    with profiler.section("build_frame_meshes_total"):
        for obs in observations:
            t0 = time.perf_counter()
            frame_meshes.append(build_frame_mesh(obs, global_body_pose))
            profiler.add("build_frame_mesh_each", time.perf_counter() - t0)
    profiler.report()
    scene = trimesh.Scene()
    with profiler.section("init_scene"):
        scene.add_geometry(frame_meshes[0], geom_name="frame_mesh")
    print(f"Showing frame {frame_keys[0]} (1/{len(frame_keys)})")

    playback_state = {
        "index": 0,
        "last_switch": time.monotonic(),
    }

    def _playback_callback(scene_obj):
        now = time.monotonic()
        if now - playback_state["last_switch"] < FRAME_DURATION_SECONDS:
            return
        playback_state["last_switch"] = now
        playback_state["index"] += 1

        if playback_state["index"] >= len(frame_meshes):
            try:
                import pyglet
                for win in list(pyglet.app.windows):
                    win.close()
            except Exception:
                pass
            return

        i = playback_state["index"]
        scene_obj.geometry["frame_mesh"] = frame_meshes[i]
        print(f"Showing frame {frame_keys[i]} ({i + 1}/{len(frame_keys)})")

    try:
        with profiler.section("viewer_playback_loop"):
            SceneViewer(
                scene,
                callback=_playback_callback,
                callback_period=1.0 / 30.0,
                start_loop=True,
            )
    except SystemExit:
        pass
    finally:
        profiler.report()
