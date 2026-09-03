"""Visualize one random frame of hand keypoints with torso-estimation transforms.

This is a small debug entrypoint for checking the exact world-frame hand
keypoint visualization path used by visualize.py.
"""

from __future__ import annotations

import argparse
import pickle
import random
import sys
import threading
import warnings
from pathlib import Path

import numpy as np
import torch
import trimesh
from trimesh.viewer.windowed import SceneViewer

import estimate


DIR = Path(__file__).resolve().parent
# This debug entrypoint queries the private clip database (`db` module, not
# distributed with this repo — see README).
DEFAULT_MANO_MODEL_DIR = DIR / "neutral"

MANO_TIP_VERTEX_IDS = {
    "thumb": 745,
    "index": 317,
    "middle": 444,
    "ring": 556,
    "pinky": 673,
}
MANO_INTERNAL_TO_WILOR = [
    0,                # wrist
    13, 14, 15, 16,  # thumb
    1, 2, 3, 17,     # index
    4, 5, 6, 18,     # middle
    10, 11, 12, 19,  # ring
    7, 8, 9, 20,     # pinky
]
JOINT_NAMES_WILOR = [
    "wrist",
    "thumb_cmc", "thumb_mcp", "thumb_ip", "thumb_tip",
    "index_mcp", "index_pip", "index_dip", "index_tip",
    "middle_mcp", "middle_pip", "middle_dip", "middle_tip",
    "ring_mcp", "ring_pip", "ring_dip", "ring_tip",
    "pinky_mcp", "pinky_pip", "pinky_dip", "pinky_tip",
]


def _numeric_frame_keys(frame_dict):
    return sorted(
        [k for k in frame_dict.keys() if str(k).isdigit()],
        key=lambda x: int(x),
    )


def _frame_has_hand_keypoints(frame):
    for hand_id in ("L", "R"):
        hands = frame.get(hand_id)
        if hands and "kpts_3d" in hands[0]:
            return True
    return False


def _playable_frame_keys(frame_dict):
    return [
        k for k in _numeric_frame_keys(frame_dict)
        if _frame_has_hand_keypoints(frame_dict[k])
    ]


def _clip_id(clip):
    return clip.get("node_uid") or (clip.get("video_number"), clip.get("node_number"))


def _db_where_for_clip_filters(video_number=None, node_number=None):
    where = ["depth_grounded_keypoints IS NOT NULL"]
    params = []
    if video_number is not None:
        where.append("video_number = %s")
        params.append(int(video_number))
    if node_number is not None:
        where.append("node_number = %s")
        params.append(int(node_number))
    return where, params


def _fetch_depth_grounded_clip_by_meta(meta):
    from db import query_clips

    where = [
        "depth_grounded_keypoints IS NOT NULL",
        "video_number = %s",
        "node_number = %s",
    ]
    params = [int(meta["video_number"]), int(meta["node_number"])]
    if meta.get("node_uid") is not None:
        where.append("node_uid = %s")
        params.append(str(meta["node_uid"]))

    print(
        "Fetching keypoints for selected clip "
        f"video={meta.get('video_number')} node={meta.get('node_number')} "
        f"uid={meta.get('node_uid')}...",
        flush=True,
    )
    rows = query_clips(
        "SELECT video_number, node_number, node_uid, "
        "depth_grounded_keypoints AS keypoints_per_frame, "
        "left_hand_presence, right_hand_presence "
        "FROM selected_clips "
        f"WHERE {' AND '.join(where)} "
        "ORDER BY video_number, node_number, node_uid",
        tuple(params),
    )
    if not rows:
        return None
    return rows[0]


def _fetch_random_depth_grounded_clip(video_number=None, node_number=None, rng=None):
    from db import query

    if rng is None:
        rng = random.Random()

    where, params = _db_where_for_clip_filters(video_number, node_number)
    where_sql = " AND ".join(where)

    print("Counting matching DB clips with depth_grounded_keypoints...", flush=True)
    count_rows = query(
        f"SELECT COUNT(1) AS clip_count FROM selected_clips WHERE {where_sql}",
        tuple(params),
    )
    clip_count = int(count_rows[0]["clip_count"]) if count_rows else 0
    print(f"DB matched {clip_count} clips before frame-level validation", flush=True)
    if clip_count <= 0:
        raise ValueError(
            "No selected_clips rows with depth_grounded_keypoints matched "
            f"video_number={video_number} node_number={node_number}"
        )

    # Select one lightweight metadata row at a time. ORDER BY keeps seeded
    # random offsets stable for a fixed table state, and the keypoint JSON blob
    # is fetched only after a specific clip has been selected.
    max_attempts = min(10, clip_count)
    tried_offsets = set()
    for attempt in range(1, max_attempts + 1):
        random_offset = rng.randrange(clip_count)
        while random_offset in tried_offsets and len(tried_offsets) < clip_count:
            random_offset = rng.randrange(clip_count)
        tried_offsets.add(random_offset)
        print(
            f"Fetching random DB clip attempt {attempt}/{max_attempts} "
            f"at offset {random_offset}/{clip_count - 1}",
            flush=True,
        )
        meta_rows = query(
            "SELECT video_number, node_number, node_uid, "
            "left_hand_presence, right_hand_presence "
            "FROM selected_clips "
            f"WHERE {where_sql} "
            "ORDER BY video_number, node_number, node_uid "
            "OFFSET %s ROWS FETCH NEXT 1 ROWS ONLY",
            tuple(params + [random_offset]),
        )
        if not meta_rows:
            print("Random metadata query returned no row; retrying...", flush=True)
            continue
        meta = meta_rows[0]
        print(
            "Selected metadata "
            f"video={meta.get('video_number')} node={meta.get('node_number')} "
            f"uid={meta.get('node_uid')} "
            f"L_presence={meta.get('left_hand_presence')} "
            f"R_presence={meta.get('right_hand_presence')}",
            flush=True,
        )

        clip = _fetch_depth_grounded_clip_by_meta(meta)
        if clip is None or not clip.get("keypoints_per_frame"):
            print("Selected DB clip had no parsed keypoints; retrying...", flush=True)
            continue
        if not _playable_frame_keys(clip["keypoints_per_frame"]):
            print("Fetched DB row had no playable hand frame; retrying...", flush=True)
            continue
        return clip

    raise ValueError(
        "Could not find a playable hand-keypoint frame after "
        f"{max_attempts} random DB fetch attempt(s)"
    )


def _choose_random_frame(frame_dict, rng):
    frame_keys = _playable_frame_keys(frame_dict)
    if not frame_keys:
        raise ValueError("Selected clip has no frame with hand keypoints")
    index = rng.randrange(len(frame_keys))
    print(
        f"Random frame index among playable frames: {index}/{len(frame_keys) - 1}",
        flush=True,
    )
    return frame_keys[index]


def _fmt_points(points):
    pts = np.asarray(points, dtype=np.float64)
    return np.array2string(pts, precision=6, suppress_small=False)


def _make_stdin_enter_event():
    evt = threading.Event()

    def _reader():
        while True:
            line = sys.stdin.readline()
            if not line:
                return
            evt.set()

    thread = threading.Thread(target=_reader, daemon=True)
    thread.start()
    return evt


def _as_numpy_array(value, dtype=np.float32):
    if hasattr(value, "toarray"):
        value = value.toarray()
    elif hasattr(value, "r"):
        value = value.r
    return np.asarray(value, dtype=dtype)


def _patch_legacy_chumpy_numpy_aliases():
    for name, value in [
        ("bool", bool),
        ("int", int),
        ("float", float),
        ("complex", complex),
        ("object", object),
        ("unicode", str),
        ("str", str),
    ]:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", FutureWarning)
            exists = hasattr(np, name)
        if not exists:
            setattr(np, name, value)


def _mano_parents(kintree_table, num_joints):
    kt = np.asarray(kintree_table)
    parents = np.full(num_joints, -1, dtype=np.int64)
    if kt.ndim == 2 and kt.shape[0] == 2:
        ids = kt[1].astype(np.int64)
        parent_ids = kt[0].astype(np.int64)
        id_to_idx = {int(jid): idx for idx, jid in enumerate(ids)}
        for idx, parent_id in enumerate(parent_ids):
            parents[idx] = id_to_idx.get(int(parent_id), -1)
    else:
        parents[:] = kt.astype(np.int64).reshape(-1)[:num_joints]
    parents[0] = -1
    return parents


def _batch_rodrigues(rot_vecs):
    batch_size = rot_vecs.shape[0]
    dtype = rot_vecs.dtype
    device = rot_vecs.device
    angle = torch.linalg.norm(rot_vecs + 1e-8, dim=1, keepdim=True)
    direction = rot_vecs / angle
    ca = torch.cos(angle).view(-1, 1, 1)
    sa = torch.sin(angle).view(-1, 1, 1)
    x, y, z = direction[:, 0], direction[:, 1], direction[:, 2]
    zeros = torch.zeros(batch_size, dtype=dtype, device=device)
    K = torch.stack([
        zeros, -z, y,
        z, zeros, -x,
        -y, x, zeros,
    ], dim=1).reshape(batch_size, 3, 3)
    eye = torch.eye(3, dtype=dtype, device=device).unsqueeze(0)
    return eye + sa * K + (1.0 - ca) * torch.bmm(K, K)


def _transform_mat(R, t):
    batch_size = R.shape[0]
    bottom = torch.zeros(batch_size, 1, 4, dtype=R.dtype, device=R.device)
    bottom[:, 0, 3] = 1.0
    return torch.cat([torch.cat([R, t.unsqueeze(-1)], dim=2), bottom], dim=1)


def _batch_rigid_transform(rot_mats, joints, parents):
    joints_h = joints.clone()
    joints_h[:, 1:] -= joints[:, parents[1:]]
    transforms = _transform_mat(rot_mats.reshape(-1, 3, 3),
                                joints_h.reshape(-1, 3)).reshape(
        rot_mats.shape[0], -1, 4, 4)
    transform_chain = [transforms[:, 0]]
    for i in range(1, parents.shape[0]):
        parent_i = int(parents[i].item())
        transform_chain.append(torch.matmul(transform_chain[parent_i],
                                            transforms[:, i]))
    A = torch.stack(transform_chain, dim=1)
    posed_joints = A[:, :, :3, 3]
    joints_homo = torch.cat(
        [joints, torch.zeros_like(joints[:, :, :1])], dim=2
    ).unsqueeze(-1)
    rel_joints = torch.matmul(A, joints_homo)
    A = A.clone()
    A[:, :, :, 3:4] -= rel_joints
    return posed_joints, A


class LocalManoLayer(torch.nn.Module):
    """Minimal MANO LBS forward pass for this fitting/debug script."""

    def __init__(self, model_path, device):
        super().__init__()
        print(f"Loading MANO model from {model_path}...", flush=True)
        _patch_legacy_chumpy_numpy_aliases()
        with open(model_path, "rb") as f:
            data = pickle.load(f, encoding="latin1")

        v_template = _as_numpy_array(data["v_template"])
        shapedirs = _as_numpy_array(data["shapedirs"])
        posedirs = _as_numpy_array(data["posedirs"])
        J_regressor = _as_numpy_array(data["J_regressor"])
        weights = _as_numpy_array(data["weights"])
        hands_mean = _as_numpy_array(data.get("hands_mean", np.zeros(45)))
        faces = _as_numpy_array(data["f"], dtype=np.int64)
        parents = _mano_parents(data["kintree_table"], J_regressor.shape[0])

        self.register_buffer("v_template", torch.tensor(
            v_template, dtype=torch.float32, device=device))
        self.register_buffer("shapedirs", torch.tensor(
            shapedirs[:, :, :10], dtype=torch.float32, device=device))
        self.register_buffer("posedirs", torch.tensor(
            posedirs, dtype=torch.float32, device=device))
        self.register_buffer("J_regressor", torch.tensor(
            J_regressor, dtype=torch.float32, device=device))
        self.register_buffer("weights", torch.tensor(
            weights, dtype=torch.float32, device=device))
        self.register_buffer("hands_mean", torch.tensor(
            hands_mean.reshape(45), dtype=torch.float32, device=device))
        self.register_buffer("parents", torch.tensor(
            parents, dtype=torch.long, device=device))
        self.faces = faces

    def forward(self, global_orient, hand_pose_delta, betas):
        batch_size = global_orient.shape[0]
        hand_pose = self.hands_mean.unsqueeze(0) + hand_pose_delta
        full_pose = torch.cat([global_orient, hand_pose], dim=1).reshape(
            batch_size, 16, 3)
        rot_mats = _batch_rodrigues(full_pose.reshape(-1, 3)).reshape(
            batch_size, 16, 3, 3)

        v_shaped = self.v_template.unsqueeze(0) + torch.einsum(
            "bl,vkl->bvk", betas, self.shapedirs)
        J = torch.einsum("jv,bvk->bjk", self.J_regressor, v_shaped)

        ident = torch.eye(3, dtype=v_shaped.dtype, device=v_shaped.device)
        pose_feature = (rot_mats[:, 1:] - ident).reshape(batch_size, -1)
        pose_offsets = torch.einsum("bp,vkp->bvk", pose_feature, self.posedirs)
        v_posed = v_shaped + pose_offsets

        J_transformed, A = _batch_rigid_transform(rot_mats, J, self.parents)
        W = self.weights.unsqueeze(0).expand(batch_size, -1, -1)
        T = torch.einsum("bvj,bjkl->bvkl", W, A)
        v_homo = torch.cat(
            [v_posed, torch.ones(batch_size, v_posed.shape[1], 1,
                                 dtype=v_posed.dtype, device=v_posed.device)],
            dim=2,
        ).unsqueeze(-1)
        vertices = torch.matmul(T, v_homo)[:, :, :3, 0]

        tip_ids = [
            MANO_TIP_VERTEX_IDS["thumb"],
            MANO_TIP_VERTEX_IDS["index"],
            MANO_TIP_VERTEX_IDS["middle"],
            MANO_TIP_VERTEX_IDS["ring"],
            MANO_TIP_VERTEX_IDS["pinky"],
        ]
        joints_internal_21 = torch.cat(
            [J_transformed, vertices[:, tip_ids]], dim=1)
        joints_wilor = joints_internal_21[:, MANO_INTERNAL_TO_WILOR]
        return vertices, joints_wilor


def _load_mano_layer(model_dir, hand_id, device):
    suffix = "RIGHT" if hand_id == "R" else "LEFT"
    model_path = Path(model_dir) / f"MANO_{suffix}.pkl"
    if not model_path.exists():
        raise FileNotFoundError(f"MANO model not found: {model_path}")
    return LocalManoLayer(model_path, device).to(device)


def _choose_fit_hands(frame, requested_hand):
    if requested_hand is not None:
        hand_id = requested_hand.upper()
        if hand_id == "BOTH":
            hand_ids = [hid for hid in ("L", "R") if frame.get(hid)]
            if not hand_ids:
                raise ValueError("Cannot fit MANO: frame has no hand keypoints")
            return hand_ids
        if hand_id not in ("L", "R"):
            raise ValueError("--fit-hand must be L, R, or both")
        if not frame.get(hand_id):
            raise ValueError(f"--fit-hand {hand_id} requested, but frame has no hand")
        return [hand_id]
    hand_ids = [hid for hid in ("L", "R") if frame.get(hid)]
    if hand_ids:
        return hand_ids
    raise ValueError("Cannot fit MANO: frame has no hand keypoints")


def _fit_mano_to_keypoints(target_kpts, hand_id, model_dir, iters, lr,
                           print_fit_loss):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    mano = _load_mano_layer(model_dir, hand_id, device)
    target = torch.tensor(
        np.asarray(target_kpts, dtype=np.float32),
        dtype=torch.float32,
        device=device,
    ).unsqueeze(0)

    global_orient = torch.zeros(1, 3, device=device, requires_grad=True)
    hand_pose_delta = torch.zeros(1, 45, device=device, requires_grad=True)
    betas = torch.zeros(1, 10, device=device, requires_grad=True)
    log_scale = torch.zeros(1, device=device, requires_grad=True)
    transl = target[:, 0, :].clone().detach().requires_grad_(True)

    params = {
        "global_orient": global_orient,
        "hand_pose_delta": hand_pose_delta,
        "betas": betas,
        "log_scale": log_scale,
        "transl": transl,
    }

    def forward_loss(active_names):
        vertices, joints = mano(global_orient, hand_pose_delta, betas)
        scale = torch.exp(log_scale).reshape(1, 1, 1)
        pred_joints = joints * scale + transl.reshape(1, 1, 3)
        pred_vertices = vertices * scale + transl.reshape(1, 1, 3)
        residual = pred_joints - target
        data_loss = torch.nn.functional.smooth_l1_loss(
            pred_joints, target, beta=0.01)
        pose_reg = (hand_pose_delta ** 2).mean()
        beta_reg = (betas ** 2).mean()
        scale_reg = (log_scale ** 2).mean()
        loss = data_loss + 1e-4 * pose_reg + 1e-4 * beta_reg + 1e-3 * scale_reg
        rmse = torch.sqrt((residual ** 2).sum(dim=2).mean())
        return loss, data_loss, pose_reg, beta_reg, scale_reg, rmse, pred_vertices, pred_joints

    if iters == 1:
        stage_counts = (1, 0, 0)
    elif iters == 2:
        stage_counts = (1, 1, 0)
    else:
        rigid_iters = max(1, iters // 4)
        pose_iters = max(1, iters // 2)
        shape_iters = max(1, iters - rigid_iters - pose_iters)
        stage_counts = (rigid_iters, pose_iters, shape_iters)
    stages = [
        ("rigid", stage_counts[0], ["global_orient", "transl", "log_scale"]),
        ("pose", stage_counts[1],
         ["global_orient", "transl", "log_scale", "hand_pose_delta"]),
        ("shape", stage_counts[2],
         ["global_orient", "transl", "log_scale", "hand_pose_delta", "betas"]),
    ]

    with torch.no_grad():
        initial = forward_loss([])[5].item()
    print(f"MANO fit [{hand_id}] initial RMSE={initial:.6f} m", flush=True)

    last = None
    for stage_name, stage_iters, active_names in stages:
        if stage_iters <= 0:
            continue
        for name, value in params.items():
            value.requires_grad_(name in active_names)
        optimizer = torch.optim.Adam([params[n] for n in active_names], lr=lr)
        print(
            f"MANO fit stage {stage_name}: {stage_iters} iters, "
            f"optimizing {', '.join(active_names)}",
            flush=True,
        )
        for i in range(stage_iters):
            optimizer.zero_grad()
            last = forward_loss(active_names)
            last[0].backward()
            optimizer.step()
            if print_fit_loss and (
                i == 0 or (i + 1) % max(1, stage_iters // 5) == 0
                or i + 1 == stage_iters
            ):
                print(
                    f"  {stage_name} iter {i + 1}/{stage_iters}: "
                    f"loss={last[0].item():.6f} "
                    f"data={last[1].item():.6f} "
                    f"rmse={last[5].item():.6f}m",
                    flush=True,
                )

    with torch.no_grad():
        loss, data_loss, pose_reg, beta_reg, scale_reg, rmse, vertices, joints = (
            forward_loss([])
        )
        residuals = torch.linalg.norm(joints - target, dim=2)[0]
        result = {
            "hand_id": hand_id,
            "vertices_cam": vertices[0].detach().cpu().numpy(),
            "joints_cam": joints[0].detach().cpu().numpy(),
            "faces": mano.faces,
            "loss": float(loss.item()),
            "data_loss": float(data_loss.item()),
            "pose_reg": float(pose_reg.item()),
            "beta_reg": float(beta_reg.item()),
            "scale_reg": float(scale_reg.item()),
            "rmse": float(rmse.item()),
            "translation": transl[0].detach().cpu().numpy(),
            "scale": float(torch.exp(log_scale)[0].item()),
            "global_orient": global_orient[0].detach().cpu().numpy(),
            "residuals": residuals.detach().cpu().numpy(),
        }

    print(
        f"MANO fit [{hand_id}] final RMSE={result['rmse']:.6f} m "
        f"loss={result['loss']:.6f} data={result['data_loss']:.6f}",
        flush=True,
    )
    print(
        f"MANO fit [{hand_id}] translation={result['translation'].round(6).tolist()} "
        f"scale={result['scale']:.6f} "
        f"global_orient={result['global_orient'].round(6).tolist()}",
        flush=True,
    )
    print("MANO per-joint residuals:", flush=True)
    for name, value in zip(JOINT_NAMES_WILOR, result["residuals"]):
        print(f"  {name:12s} {float(value):.6f} m", flush=True)
    print(
        f"MANO fitted raw camera joints:\n{_fmt_points(result['joints_cam'])}",
        flush=True,
    )
    return result


def _flip_points_about_wrist(points, wrist):
    pts = np.asarray(points, dtype=np.float64).copy()
    wrist = np.asarray(wrist, dtype=np.float64).reshape(3)
    if estimate.FLIP_HAND_Y_ABOUT_WRIST:
        pts[:, 1] = 2.0 * wrist[1] - pts[:, 1]
    if estimate.FLIP_HAND_Z_ABOUT_WRIST:
        pts[:, 2] = 2.0 * wrist[2] - pts[:, 2]
    return pts


def _add_mano_fit_viz(parts, fit_result, R_wc, t_wc):
    joints_cam = fit_result["joints_cam"]
    wrist_cam = joints_cam[0]
    joints_world = (
        R_wc @ _flip_points_about_wrist(joints_cam, wrist_cam).T
    ).T + t_wc
    vertices_world = (
        R_wc @ _flip_points_about_wrist(fit_result["vertices_cam"], wrist_cam).T
    ).T + t_wc
    print(
        f"MANO fitted transformed world joints:\n{_fmt_points(joints_world)}",
        flush=True,
    )
    parts += estimate.make_hand_viz(joints_world, [80, 255, 120, 255])
    mesh = trimesh.Trimesh(
        vertices=vertices_world,
        faces=np.asarray(fit_result["faces"], dtype=np.int64),
        process=False,
    )
    mesh.visual.vertex_colors = np.tile(
        np.array([80, 255, 120, 90], dtype=np.uint8),
        (mesh.vertices.shape[0], 1),
    )
    parts.append(mesh)


def _build_frame_mesh(raw_clip, frame_key, fit_mano=False, fit_hand=None,
                      fit_iters=500, fit_lr=0.03,
                      mano_model_dir=DEFAULT_MANO_MODEL_DIR,
                      print_fit_loss=True):
    print("Building frame mesh...", flush=True)
    frame_dict = raw_clip["keypoints_per_frame"]
    frame_keys = _numeric_frame_keys(frame_dict)
    if not frame_keys:
        raise ValueError("Selected clip has no numeric frame keys")
    print(f"Clip has {len(frame_keys)} numeric frames", flush=True)
    if frame_key not in frame_dict:
        str_key = str(frame_key)
        int_key = int(frame_key) if str_key.isdigit() else None
        if str_key in frame_dict:
            frame_key = str_key
        elif int_key in frame_dict:
            frame_key = int_key
        else:
            raise KeyError(f"Frame {frame_key!r} not found in selected clip")

    print("Computing clip camera transform...", flush=True)
    R_wc, t_wc, gravity = estimate.compute_clip_camera_transform(frame_dict, frame_keys)
    print("Computed clip camera transform", flush=True)
    frame = frame_dict[frame_key]
    print("Creating ground plane and camera visualization...", flush=True)
    parts = [estimate.make_ground_plane()]
    parts += estimate.make_camera_viz(R_wc, t_wc)
    raw_hand_keypoints = {}

    for hand_id, color in [
        ("L", [60, 120, 255, 255]),
        ("R", [255, 120, 60, 255]),
    ]:
        if not frame.get(hand_id):
            print(f"Frame {frame_key}: no {hand_id} hand", flush=True)
            continue
        print(f"Frame {frame_key}: rendering {hand_id} hand keypoints", flush=True)
        pts_cam = np.asarray(frame[hand_id][0]["kpts_3d"], dtype=np.float64)
        print(
            f"Frame {frame_key}: {hand_id} raw camera keypoints:\n"
            f"{_fmt_points(pts_cam)}",
            flush=True,
        )
        pts_cam = estimate.flip_about_wrist(
            pts_cam,
            flip_y=estimate.FLIP_HAND_Y_ABOUT_WRIST,
            flip_z=estimate.FLIP_HAND_Z_ABOUT_WRIST,
        )
        pts_world = (R_wc @ pts_cam.T).T + t_wc
        raw_hand_keypoints[hand_id] = np.asarray(
            frame[hand_id][0]["kpts_3d"], dtype=np.float64)
        print(
            f"Frame {frame_key}: {hand_id} transformed world keypoints:\n"
            f"{_fmt_points(pts_world)}",
            flush=True,
        )
        parts += estimate.make_hand_viz(pts_world, color)
        print(f"Frame {frame_key}: rendered {hand_id} hand", flush=True)

    if fit_mano:
        for hand_id in _choose_fit_hands(frame, fit_hand):
            print(f"Fitting MANO to {hand_id} hand on frame {frame_key}...",
                  flush=True)
            fit_result = _fit_mano_to_keypoints(
                raw_hand_keypoints[hand_id],
                hand_id=hand_id,
                model_dir=mano_model_dir,
                iters=fit_iters,
                lr=fit_lr,
                print_fit_loss=print_fit_loss,
            )
            _add_mano_fit_viz(parts, fit_result, R_wc, t_wc)
            print(f"Added fitted {hand_id} MANO joints and mesh to visualization",
                  flush=True)

    print(f"Concatenating {len(parts)} geometry parts...", flush=True)
    mesh = trimesh.util.concatenate(parts)
    print("Frame mesh ready", flush=True)
    return mesh, R_wc, t_wc, gravity


def _build_random_sample(args, rng, sample_index):
    print(
        f"\n=== Building sample {sample_index} ===\n"
        f"Loading one random clip from DB with video_number={args.video_number} "
        f"node_number={args.node_number}...",
        flush=True,
    )
    clip = _fetch_random_depth_grounded_clip(
        video_number=args.video_number,
        node_number=args.node_number,
        rng=rng,
    )
    frame_dict = clip["keypoints_per_frame"]
    print("Choosing frame...", flush=True)
    frame_key = args.frame_key if args.frame_key is not None else _choose_random_frame(
        frame_dict, rng)
    if args.frame_key is not None:
        print(f"Using explicit frame key: {frame_key}", flush=True)

    mesh, R_wc, t_wc, gravity = _build_frame_mesh(
        clip,
        frame_key,
        fit_mano=args.fit_mano,
        fit_hand=args.fit_hand,
        fit_iters=args.fit_iters,
        fit_lr=args.fit_lr,
        mano_model_dir=args.mano_model_dir,
        print_fit_loss=not args.no_print_fit_loss,
    )
    print(
        f"Selected clip={_clip_id(clip)} "
        f"video={clip.get('video_number')} node={clip.get('node_number')} "
        f"frame={frame_key}",
        flush=True,
    )
    print(f"gravity_down_camera={np.asarray(gravity).round(6).tolist()}", flush=True)
    print(f"t_world_from_camera={np.asarray(t_wc).round(6).tolist()}", flush=True)
    print(f"R_world_from_camera=\n{np.asarray(R_wc).round(6)}", flush=True)
    print(
        "L hand is blue, R hand is orange"
        + (", fitted MANO is green." if args.fit_mano else "."),
        flush=True,
    )
    label = (
        f"sample={sample_index} clip={_clip_id(clip)} "
        f"video={clip.get('video_number')} node={clip.get('node_number')} "
        f"frame={frame_key}"
    )
    return mesh, label


def _replace_scene_geometry(scene_obj, mesh, geom_name):
    for old_name in list(scene_obj.geometry.keys()):
        try:
            scene_obj.delete_geometry(old_name)
        except Exception:
            try:
                del scene_obj.geometry[old_name]
            except Exception:
                pass
    scene_obj.add_geometry(mesh, geom_name=geom_name)
    if hasattr(scene_obj, "graph"):
        try:
            scene_obj.graph.update(frame_from=geom_name, frame_to="world")
        except Exception:
            pass


def _show_random_samples(args, rng):
    enter_event = _make_stdin_enter_event()
    mesh, label = _build_random_sample(args, rng, sample_index=1)
    scene = trimesh.Scene()
    scene.add_geometry(mesh, geom_name="frame_mesh_1")
    print(f"Showing {label}", flush=True)
    print("Press Enter in this terminal to choose another random clip/frame.",
          flush=True)
    state = {"sample_index": 1, "building": False}

    def _callback(scene_obj):
        if not enter_event.is_set() or state["building"]:
            return
        enter_event.clear()
        state["building"] = True
        state["sample_index"] += 1
        try:
            mesh_next, label_next = _build_random_sample(
                args, rng, state["sample_index"])
            _replace_scene_geometry(
                scene_obj, mesh_next, f"frame_mesh_{state['sample_index']}")
            print(f"Showing {label_next}", flush=True)
            print("Press Enter in this terminal to choose another random clip/frame.",
                  flush=True)
        except Exception as exc:
            print(
                f"Failed to build sample {state['sample_index']}: "
                f"{type(exc).__name__}: {exc}",
                flush=True,
            )
        finally:
            state["building"] = False

    print("Opening trimesh viewer...", flush=True)
    try:
        SceneViewer(scene, callback=_callback, callback_period=1.0 / 30.0,
                    start_loop=True)
    except SystemExit:
        pass


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Choose a random clip/frame and visualize hand keypoints under the "
            "same world transform used by visualize.py."
        )
    )
    parser.add_argument("--video-number", type=int, default=None)
    parser.add_argument("--node-number", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument(
        "--frame-key",
        default=None,
        help="Optional explicit frame key. Defaults to a random playable frame.",
    )
    parser.add_argument("--fit-mano", action="store_true",
                        help="Fit a local MANO model to the selected frame hand.")
    parser.add_argument("--fit-hand", choices=["L", "R", "both"], default=None,
                        help="Hand to fit. Defaults to all hands present.")
    parser.add_argument("--fit-iters", type=int, default=500,
                        help="Total MANO optimization iterations.")
    parser.add_argument("--fit-lr", type=float, default=0.03,
                        help="Adam learning rate for MANO fitting.")
    parser.add_argument("--mano-model-dir", type=Path, default=DEFAULT_MANO_MODEL_DIR,
                        help="Directory containing MANO_LEFT.pkl and MANO_RIGHT.pkl.")
    parser.add_argument("--no-print-fit-loss", action="store_true",
                        help="Suppress per-stage MANO optimization progress.")
    args = parser.parse_args()
    if args.fit_iters <= 0:
        parser.error("--fit-iters must be positive")
    if args.fit_lr <= 0:
        parser.error("--fit-lr must be positive")

    rng = random.Random(args.seed)
    _show_random_samples(args, rng)


if __name__ == "__main__":
    main()
