"""
VN torso/body-frame estimation from depth-grounded hand keypoints.

Reusable module: recomputes camera transforms and wrist frames from raw
keypoints in torso_relative_clip_keypoints.pt — no dependency on
depth_grounded_clip_keypoints_transforms.pt — and runs the VN alignment
model (models/body_pose_est.pt) to predict per-clip torso frames.
"""

from __future__ import annotations

import argparse
import os
import sys
import threading
import time
from pathlib import Path

import numpy as np
import roma
import torch
import trimesh

_REPO_ROOT = Path(__file__).resolve().parents[2]
RAW_PATH = _REPO_ROOT / "eval_data" / "torso_relative_clip_keypoints.pt"
# The VN alignment / retargeting code is an external dependency (see README).
RETARGET_DIR = Path(
    os.environ.get("RETARGET_DIR", _REPO_ROOT.parent / "external_retargeting" / "retarget")
)
DEFAULT_VN_CHECKPOINT = _REPO_ROOT / "models" / "body_pose_est.pt"

FRAME_GRAVITY_IS_UP = True
FLIP_HAND_Y_ABOUT_WRIST = True
FLIP_HAND_Z_ABOUT_WRIST = True
WORLD_HAND_CENTER = np.array([0.0, 1.0, 1.2])

MANO_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (0, 9), (9, 10), (10, 11), (11, 12),
    (0, 13), (13, 14), (14, 15), (15, 16),
    (0, 17), (17, 18), (18, 19), (19, 20),
]

_SPHERE_TEMPLATE = trimesh.creation.uv_sphere(radius=1.0, count=[10, 10])
_CYLINDER_TEMPLATE = trimesh.creation.cylinder(radius=1.0, height=1.0)


# ── math helpers ─────────────────────────────────────────────────────────────

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
    return np.array([[0.0, -z, y], [z, 0.0, -x], [-y, x, 0.0]], dtype=np.float64)


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


def _rotmat_to_quat_wxyz(R: np.ndarray) -> np.ndarray:
    q_xyzw = roma.rotmat_to_unitquat(torch.from_numpy(R.astype(np.float32))).numpy()
    return q_xyzw[[3, 0, 1, 2]]


def _se3_from_R_t(R, t):
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = np.asarray(R, dtype=np.float64).reshape(3, 3)
    T[:3, 3] = np.asarray(t, dtype=np.float64).reshape(3)
    return T



# ── camera frame computation ────────────────────────────────────────────────

def _frame_gravity_down_in_camera(frame):
    gravity_cam = np.asarray(frame.get("gravity", [0.0, 1.0, 0.0]), dtype=np.float64)
    if FRAME_GRAVITY_IS_UP:
        gravity_cam = -gravity_cam
    return _safe_unit(gravity_cam, fallback=np.array([0.0, 1.0, 0.0]))


def _rotation_world_from_camera(gravity_cam):
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
    cam_forward_world = R_align @ np.array([0.0, 0.0, 1.0])
    target_forward_world = np.array([0.0, 1.0, 0.0])
    cam_forward_xy = cam_forward_world.copy()
    cam_forward_xy[2] = 0.0
    if np.linalg.norm(cam_forward_xy) < 1e-8:
        return R_align
    cam_forward_xy = _safe_unit(cam_forward_xy)
    signed_cross_z = np.cross(cam_forward_xy, target_forward_world)[2]
    signed_dot = np.clip(np.dot(cam_forward_xy, target_forward_world), -1.0, 1.0)
    yaw = np.arctan2(signed_cross_z, signed_dot)
    return _axis_angle_to_matrix(np.array([0.0, 0.0, 1.0]), yaw) @ R_align


def _compute_clip_camera_transform(frame_dict, frame_keys):
    gravity_vectors = []
    for k in frame_keys:
        frame = frame_dict[k]
        if "gravity" in frame:
            gravity_vectors.append(_frame_gravity_down_in_camera(frame))
    if gravity_vectors:
        gravity_clip = _safe_unit(np.mean(gravity_vectors, axis=0))
    else:
        gravity_clip = _frame_gravity_down_in_camera({})
    R_wc = _rotation_world_from_camera(gravity_clip)
    t_wc = np.zeros(3, dtype=np.float64)
    for k in frame_keys:
        hand_points = []
        for hid in ["L", "R"]:
            if frame_dict[k].get(hid):
                hand_points.append(np.asarray(frame_dict[k][hid][0]["kpts_3d"], dtype=np.float64))
        if hand_points:
            pts_cam = np.concatenate(hand_points, axis=0)
            centroid_world_no_t = (R_wc @ pts_cam.T).T.mean(axis=0)
            t_wc = WORLD_HAND_CENTER - centroid_world_no_t
            break
    return R_wc, t_wc, gravity_clip


# ── wrist frame from keypoints ───────────────────────────────────────────────

def _wrist_frame_from_keypoints_cam(kpts_3d_cam, hand_side):
    """Compute wrist frame directly in camera frame from camera-frame keypoints."""
    pts = np.asarray(kpts_3d_cam, dtype=np.float64)
    wrist = pts[0]
    index_mcp = pts[5]
    middle_mcp = pts[9]
    pinky_mcp = pts[17]

    x = middle_mcp - wrist
    x_norm = np.linalg.norm(x)
    if x_norm < 1e-8:
        return wrist, np.eye(3)
    x /= x_norm

    index_dir = index_mcp - wrist
    pinky_dir = pinky_mcp - wrist
    z = np.cross(index_dir, pinky_dir)
    z_norm = np.linalg.norm(z)
    if z_norm < 1e-8:
        return wrist, np.eye(3)
    z /= z_norm

    if hand_side is not None and str(hand_side).upper().startswith("L"):
        z = -z

    y = np.cross(z, x)
    return wrist, np.column_stack([x, y, z])


# ── visualization primitives ─────────────────────────────────────────────────

def _depth_shade(base_color, z_min, z_max, z):
    t = (z - z_min) / (z_max - z_min) if z_max > z_min else 0.0
    shade = 0.4 + 0.6 * t
    return [int(c * shade) for c in base_color[:3]] + [base_color[3]]


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


def _make_ground_plane(size=2.0, color=(80, 80, 80, 90)):
    plane = trimesh.creation.box(extents=[size, size, 0.002])
    plane.apply_translation([0.0, 1.0, -0.001])
    plane.visual.vertex_colors = color
    return plane


def _make_camera_viz(R_wc, t_wc, axis_len=0.12, axis_r=0.003):
    origin = np.asarray(t_wc, dtype=np.float64)
    parts = []
    for axis, color in [
        (np.array([axis_len, 0.0, 0.0]), [255, 0, 0, 255]),
        (np.array([0.0, axis_len, 0.0]), [0, 255, 0, 255]),
        (np.array([0.0, 0.0, axis_len]), [0, 0, 255, 255]),
    ]:
        tip = origin + R_wc @ axis
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
    corners = [origin + R_wc @ p for p in corners_cam]
    edges = [(origin, c) for c in corners]
    edges += [(corners[i], corners[(i + 1) % 4]) for i in range(4)]
    for a, b in edges:
        cyl = _make_cylinder_between(a, b, axis_r * 0.6, [255, 255, 255, 255])
        if cyl is not None:
            parts.append(cyl)
    marker = _make_sphere(origin, axis_r * 3.0, [255, 255, 255, 255])
    parts.append(marker)
    return parts


def _make_hand_viz(kpts_world, color):
    pts = np.asarray(kpts_world, dtype=np.float64)
    hand_span = np.ptp(pts, axis=0).max()
    sphere_r = max(hand_span * 0.06, 0.004)
    z_viz = pts[:, 1]
    z_min = float(z_viz.min())
    z_max = float(z_viz.max())
    parts = []
    for ki, p in enumerate(pts):
        s = _make_sphere(
            p, sphere_r * (1.5 if ki == 0 else 1.0),
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


def _make_wrist_frame_viz(origin, R, axis_len=0.035, axis_r=0.0015):
    axis_colors = [[255, 0, 0, 255], [0, 255, 0, 255], [0, 0, 255, 255]]
    parts = []
    for i, color in enumerate(axis_colors):
        tip = origin + R[:, i] * axis_len
        cyl = _make_cylinder_between(origin, tip, axis_r, color)
        if cyl is not None:
            parts.append(cyl)
    return parts


def _make_torso_frame_viz(origin, R, axis_len=0.20, axis_r=0.005):
    axis_colors = [[255, 0, 0, 255], [0, 255, 0, 255], [0, 0, 255, 255]]
    parts = []
    for i, color in enumerate(axis_colors):
        tip = origin + R[:, i] * axis_len
        cyl = _make_cylinder_between(origin, tip, axis_r, color)
        if cyl is not None:
            parts.append(cyl)
    sphere = _make_sphere(origin, axis_r * 2.3, [255, 255, 255, 255])
    parts.append(sphere)
    return parts


def _make_arrow_viz(start, direction, length=0.5,
                    color=(255, 220, 40, 255), shaft_r=0.008):
    start = np.asarray(start, dtype=np.float64).reshape(3)
    direction = np.asarray(direction, dtype=np.float64).reshape(3)
    n = float(np.linalg.norm(direction))
    if n < 1e-9:
        return []
    d = direction / n
    shaft_len = 0.78 * length
    head_len = length - shaft_len
    head_r = shaft_r * 2.8
    shaft_end = start + d * shaft_len
    tip = start + d * length
    shaft = trimesh.creation.cylinder(
        radius=shaft_r, segment=[start, shaft_end], sections=24)
    shaft.visual.vertex_colors = color
    cone = trimesh.creation.cone(radius=head_r, height=head_len, sections=32)
    z_axis = np.array([0.0, 0.0, 1.0])
    axis = np.cross(z_axis, d)
    a_norm = float(np.linalg.norm(axis))
    if a_norm < 1e-9:
        R = np.eye(3) if d[2] >= 0 else np.diag([1.0, -1.0, -1.0])
    else:
        axis = axis / a_norm
        ang = float(np.arccos(np.clip(float(z_axis @ d), -1.0, 1.0)))
        K = _skew(axis)
        R = np.eye(3) + np.sin(ang) * K + (1.0 - np.cos(ang)) * (K @ K)
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = R
    T[:3, 3] = tip - d * (head_len * 0.5)
    cone.apply_transform(T)
    cone.visual.vertex_colors = color
    return [shaft, cone]


def _flip_about_wrist(kpts_world, flip_y=False, flip_z=False):
    pts = np.asarray(kpts_world, dtype=np.float64).copy()
    if flip_y:
        wrist_y = pts[0, 1]
        pts[:, 1] = 2 * wrist_y - pts[:, 1]
    if flip_z:
        wrist_z = pts[0, 2]
        pts[:, 2] = 2 * wrist_z - pts[:, 2]
    return pts


# ── VN model loading ─────────────────────────────────────────────────────────

def _load_vn_model(path, device):
    retarget_dir = str(RETARGET_DIR)
    if retarget_dir not in sys.path:
        sys.path.insert(0, retarget_dir)
    from models.vn_transformer import VNAlignmentPolicy

    ckpt = torch.load(path, map_location=device, weights_only=False)
    saved = ckpt.get("args", {})
    model = VNAlignmentPolicy(
        d_model=saved.get("d_model", 128),
        num_heads=saved.get("num_heads", 4),
        num_layers=saved.get("num_layers", 4),
        dim_feedforward=saved.get("dim_feedforward", 512),
        dropout=saved.get("dropout", 0.1),
        mode=saved.get("mode", "deterministic"),
    ).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    return model


# ── build VN input from raw keypoints ────────────────────────────────────────

def _build_pose7_trajectory_from_raw(frame_dict, frame_keys, hand_id):
    """Build (T, 7) camera-frame pose trajectory and mask from raw keypoints.

    Matches estimate.camera_to_wrist_frame_se3_for_frame: the Y/Z flip is
    applied to the camera-frame keypoints BEFORE computing the wrist frame,
    so the origin and rotation columns are consistent with the saved transforms.
    """
    poses = np.zeros((len(frame_keys), 7), dtype=np.float32)
    mask = np.zeros(len(frame_keys), dtype=bool)
    for i, fk in enumerate(frame_keys):
        frame = frame_dict[fk]
        if not frame.get(hand_id):
            continue
        pts_cam = np.asarray(frame[hand_id][0]["kpts_3d"], dtype=np.float64)
        pts_cam = _flip_about_wrist(
            pts_cam,
            flip_y=FLIP_HAND_Y_ABOUT_WRIST,
            flip_z=FLIP_HAND_Z_ABOUT_WRIST,
        )
        origin, R = _wrist_frame_from_keypoints_cam(pts_cam, hand_id)
        poses[i, :3] = origin.astype(np.float32)
        poses[i, 3:] = _rotmat_to_quat_wxyz(R).astype(np.float32)
        mask[i] = True
    return poses, mask


@torch.no_grad()
def _predict_torso_with_vn(vn_model, frame_dict, frame_keys,
                           gravity_down_cam, device):
    left_p7, left_m = _build_pose7_trajectory_from_raw(frame_dict, frame_keys, "L")
    right_p7, right_m = _build_pose7_trajectory_from_raw(frame_dict, frame_keys, "R")
    if not left_m.any() and not right_m.any():
        return None

    left_cam = torch.from_numpy(left_p7).unsqueeze(0).to(device)
    right_cam = torch.from_numpy(right_p7).unsqueeze(0).to(device)
    left_mask = torch.from_numpy(left_m).unsqueeze(0).to(device)
    right_mask = torch.from_numpy(right_m).unsqueeze(0).to(device)
    g_cam = torch.from_numpy(
        np.asarray(gravity_down_cam, dtype=np.float32)
    ).reshape(1, 3).to(device)

    if vn_model.mode == "flow":
        R_pred, t_pred = vn_model.sample(left_cam, right_cam, g_cam=g_cam)
    else:
        R_pred, t_pred = vn_model(left_cam, right_cam, g_cam=g_cam,
                                  left_mask=left_mask, right_mask=right_mask)

    R_cb = R_pred[0].cpu().numpy().astype(np.float64)
    t_cb = t_pred[0].cpu().numpy().astype(np.float64)
    return _se3_from_R_t(R_cb, t_cb)


# ── public API ───────────────────────────────────────────────────────────────

def load_model(checkpoint_path=None, device=None):
    if checkpoint_path is None:
        checkpoint_path = DEFAULT_VN_CHECKPOINT
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return _load_vn_model(Path(checkpoint_path), device), device


def estimate_body_frame(clip, vn_model, device):
    """Estimate body frame and transform hand keypoints into body coordinates.

    Args:
        clip: dict from torso_relative_clip_keypoints.pt with 'keypoints_per_frame'
        vn_model: loaded VNAlignmentPolicy model
        device: torch device

    Returns:
        dict with:
            R_cb: (3, 3) rotation camera→body
            t_cb: (3,) translation camera→body
            hands_body: dict[frame_key] → {"L": (21, 3) or None, "R": (21, 3) or None}
                hand keypoints in body frame (after Y/Z flip, consistent with model input)
        None if no hands are detected in the clip.
    """
    frame_dict = clip["keypoints_per_frame"]
    frame_keys = _numeric_frame_keys(frame_dict)
    if not frame_keys:
        return None

    _, _, gravity_clip = _compute_clip_camera_transform(frame_dict, frame_keys)

    T_cb = _predict_torso_with_vn(vn_model, frame_dict, frame_keys, gravity_clip, device)
    if T_cb is None:
        return None

    R_cb = T_cb[:3, :3]
    t_cb = T_cb[:3, 3]
    R_bc = R_cb.T
    t_bc = -R_bc @ t_cb

    hands_body = {}
    for fk in frame_keys:
        frame = frame_dict[fk]
        entry = {}
        for hid in ("L", "R"):
            if not frame.get(hid):
                entry[hid] = None
                continue
            pts_cam = np.asarray(frame[hid][0]["kpts_3d"], dtype=np.float64)
            pts_cam = _flip_about_wrist(
                pts_cam,
                flip_y=FLIP_HAND_Y_ABOUT_WRIST,
                flip_z=FLIP_HAND_Z_ABOUT_WRIST,
            )
            pts_body = (R_bc @ pts_cam.T).T + t_bc
            entry[hid] = pts_body
        hands_body[fk] = entry

    return {"R_cb": R_cb, "t_cb": t_cb, "hands_body": hands_body}


# ── per-clip mesh building ───────────────────────────────────────────────────

def _numeric_frame_keys(frame_dict):
    return sorted([k for k in frame_dict.keys() if str(k).isdigit()], key=lambda x: int(x))


def _clip_id(clip):
    return clip.get("node_uid") or (clip.get("video_number"), clip.get("node_number"))


def _build_clip_mesh(raw_clip, vn_model, vn_device, render_frame_key=None):
    """
    Build a scene mesh for a single clip.

    If render_frame_key is given, only that frame is rendered (but the VN
    prediction uses the full trajectory).
    """
    frame_dict = raw_clip["keypoints_per_frame"]
    frame_keys = _numeric_frame_keys(frame_dict)
    if not frame_keys:
        return None

    R_wc, t_wc, gravity_clip = _compute_clip_camera_transform(frame_dict, frame_keys)

    # VN prediction over full clip trajectory
    vn_pred_world = None
    if vn_model is not None:
        T_cb_vn = _predict_torso_with_vn(
            vn_model, frame_dict, frame_keys, gravity_clip, vn_device)
        if T_cb_vn is not None:
            T_wc_mat = _se3_from_R_t(R_wc, t_wc)
            T_wb_vn = T_wc_mat @ T_cb_vn
            R_wb_vn = T_wb_vn[:3, :3]
            t_wb_vn = T_wb_vn[:3, 3]
            vn_pred_world = (R_wb_vn, t_wb_vn)

    # Render single frame
    if render_frame_key is not None:
        render_keys = [render_frame_key]
    else:
        render_keys = frame_keys

    meshes = []
    for fk in render_keys:
        frame = frame_dict[fk]
        parts = [_make_ground_plane()]
        parts += _make_camera_viz(R_wc, t_wc)

        # Gravity arrow
        g_world_down = R_wc @ np.asarray(gravity_clip, dtype=np.float64).reshape(3)
        parts += _make_arrow_viz(t_wc, g_world_down, length=0.5,
                                 color=(255, 220, 40, 255))

        # VN torso frame
        if vn_pred_world is not None:
            R_vn, t_vn = vn_pred_world
            vn_origin = t_vn + R_vn[:, 1] * 0.20 + R_vn[:, 2] * 0.08
            parts += _make_torso_frame_viz(vn_origin, R_vn, axis_len=0.36, axis_r=0.009)

        # Hand keypoints + wrist frames in world
        for hid, color in [("L", [60, 120, 255, 255]), ("R", [255, 120, 60, 255])]:
            if frame.get(hid):
                pts_cam = np.asarray(frame[hid][0]["kpts_3d"], dtype=np.float64)
                pts_world = (R_wc @ pts_cam.T).T + t_wc
                pts_world = _flip_about_wrist(
                    pts_world,
                    flip_y=FLIP_HAND_Y_ABOUT_WRIST,
                    flip_z=FLIP_HAND_Z_ABOUT_WRIST,
                )
                parts += _make_hand_viz(pts_world, color)

                # Wrist frame: flip in CAMERA frame (matching how the saved
                # transforms were generated), then rotate to world via R_wc.
                pts_cam_flipped = _flip_about_wrist(
                    pts_cam,
                    flip_y=FLIP_HAND_Y_ABOUT_WRIST,
                    flip_z=FLIP_HAND_Z_ABOUT_WRIST,
                )
                w_origin_cam, w_R_cam = _wrist_frame_from_keypoints_cam(
                    pts_cam_flipped, hand_side=hid)
                w_origin_world = R_wc @ w_origin_cam + t_wc
                w_R_world = R_wc @ w_R_cam
                parts += _make_wrist_frame_viz(w_origin_world, w_R_world)

        meshes.append(trimesh.util.concatenate(parts))

    return meshes, frame_keys


# ── streaming viewer ─────────────────────────────────────────────────────────

def _make_stdin_advance_event():
    evt = threading.Event()

    def _reader():
        while True:
            line = sys.stdin.readline()
            if not line:
                return
            evt.set()

    t = threading.Thread(target=_reader, daemon=True)
    t.start()
    return evt


def _play_clips_streaming(builder, n_total, title,
                          period_seconds=None, advance_event=None,
                          loop=False):
    try:
        from trimesh.viewer.windowed import SceneViewer
    except ImportError as exc:
        raise RuntimeError(
            "The interactive trimesh viewer requires pyglet/OpenGL system "
            "libraries such as GLU. Install those dependencies to use the "
            "streaming viewer; load_model() and estimate_body_frame() do not "
            "need them."
        ) from exc

    if advance_event is None and period_seconds is None:
        period_seconds = 1.0

    i = 0
    first = None
    while i < n_total:
        first = builder(i)
        if first is not None:
            break
        i += 1
    if first is None:
        raise RuntimeError("No clips produced any frames.")
    mesh, label = first
    scene = trimesh.Scene()
    scene.add_geometry(mesh, geom_name="frame_mesh")
    print(f"{title}: {label}", flush=True)
    if advance_event is not None:
        print("(press Enter in terminal to advance to the next clip)", flush=True)
    state = {"index": i, "last_switch": time.monotonic()}

    def _next_valid(start):
        j = start
        while j < n_total:
            r = builder(j)
            if r is not None:
                return j, r
            j += 1
        return None, None

    def _callback(scene_obj):
        if advance_event is not None:
            if not advance_event.is_set():
                return
            advance_event.clear()
        else:
            now = time.monotonic()
            if now - state["last_switch"] < period_seconds:
                return
        nxt, result = _next_valid(state["index"] + 1)
        if result is None:
            if loop:
                nxt, result = _next_valid(0)
            if result is None:
                try:
                    import pyglet
                    for win in list(pyglet.app.windows):
                        win.close()
                except Exception:
                    pass
                return
        state["index"] = nxt
        state["last_switch"] = time.monotonic()
        mesh, label = result
        scene_obj.geometry["frame_mesh"] = mesh
        print(f"{title}: {label}", flush=True)

    try:
        SceneViewer(scene, callback=_callback, callback_period=1.0 / 30.0,
                    start_loop=True)
    except SystemExit:
        pass


# ── main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Visualize VN torso predictions from raw keypoints only.")
    parser.add_argument("--vn-checkpoint", type=str,
                        default=str(DEFAULT_VN_CHECKPOINT),
                        help="Path to VNAlignmentPolicy checkpoint")
    parser.add_argument("--video-number", type=int, default=None,
                        help="Filter to clips from this video number")
    parser.add_argument("--node-number", type=int, default=None,
                        help="Filter to a specific node number")
    parser.add_argument("--period", type=float, default=None,
                        help="Auto-advance every N seconds (default: Enter to advance)")
    parser.add_argument("--loop", action="store_true",
                        help="Wrap around at end")
    args = parser.parse_args()

    raw_clips = torch.load(RAW_PATH, map_location="cpu", weights_only=False)
    print(f"Loaded {len(raw_clips)} clips from {RAW_PATH.name}", flush=True)

    def _match(c):
        if args.video_number is not None and c.get("video_number") != args.video_number:
            return False
        if args.node_number is not None and c.get("node_number") != args.node_number:
            return False
        return True

    clips = [c for c in raw_clips if _match(c)]
    clips.sort(key=lambda c: str(_clip_id(c)))
    if not clips:
        raise ValueError("No matching clips found.")
    print(f"Matched {len(clips)} clips", flush=True)

    vn_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    vn_model = _load_vn_model(Path(args.vn_checkpoint), vn_device)
    print(f"Loaded VN policy (mode={vn_model.mode}) on {vn_device}", flush=True)

    n_total = len(clips)

    def _build_clip_fast(i):
        if i < 0 or i >= n_total:
            return None
        raw_clip = clips[i]
        frame_dict = raw_clip["keypoints_per_frame"]
        all_keys = _numeric_frame_keys(frame_dict)
        if not all_keys:
            return None
        mid_idx = len(all_keys) // 2
        middle_fk = all_keys[mid_idx]
        total_frames = len(all_keys)
        try:
            result = _build_clip_mesh(raw_clip, vn_model, vn_device,
                                      render_frame_key=middle_fk)
        except Exception as e:
            cid = _clip_id(raw_clip)
            print(f"[skip {i+1}/{n_total} cid={cid}] {type(e).__name__}: {e}",
                  flush=True)
            return None
        if result is None or not result[0]:
            return None
        mesh = result[0][0]
        label = (f"clip {i+1}/{n_total} "
                 f"video={raw_clip.get('video_number')} "
                 f"node={raw_clip.get('node_number')} "
                 f"uid={raw_clip.get('node_uid')} "
                 f"frame={middle_fk} ({mid_idx + 1}/{total_frames})")
        return mesh, label

    if args.period is None:
        print(f"Streaming {n_total} clips, press Enter to advance "
              f"(close window to stop)", flush=True)
    else:
        print(f"Streaming {n_total} clips at {args.period:.2f}s/clip "
              f"(close window to stop)", flush=True)

    advance_event = _make_stdin_advance_event() if args.period is None else None
    _play_clips_streaming(_build_clip_fast, n_total, title="raw-vis",
                          period_seconds=args.period,
                          advance_event=advance_event,
                          loop=args.loop)


if __name__ == "__main__":
    main()
