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
from trimesh.viewer.windowed import SceneViewer

import estimate

# The retargeting code is an external dependency (see README).
RETARGET_DIR = Path(
    os.environ.get(
        "RETARGET_DIR",
        Path(__file__).resolve().parents[2].parent / "external_retargeting" / "retarget",
    )
)
DEFAULT_GRAVITY_DOWN_CAMERA = np.array([0.0, 1.0, 0.0], dtype=np.float64)
# This script queries the private clip database (`db` module, not distributed
# with this repo — see README). visualize_raw.py is the equivalent that runs
# from the exported eval_data artifacts instead.


def _numeric_frame_keys(frame_dict):
    return sorted([k for k in frame_dict.keys() if str(k).isdigit()], key=lambda x: int(x))


def _make_stdin_advance_event():
    """Spawn a daemon thread that sets an Event whenever the user presses Enter."""
    evt = threading.Event()

    def _reader():
        while True:
            line = sys.stdin.readline()
            if not line:  # EOF (Ctrl-D) — stop listening
                return
            evt.set()

    t = threading.Thread(target=_reader, daemon=True)
    t.start()
    return evt


def _play_clips_streaming(builder, n_total, title,
                          period_seconds=None, advance_event=None,
                          loop=False):
    """Stream clips into the SceneViewer one at a time.

    Advancement modes:
      - advance_event (threading.Event): advance whenever the event is set
        (and then cleared). Use _make_stdin_advance_event for Enter-to-advance.
      - period_seconds (float): advance every N seconds.

    builder(i) -> (mesh, label) or None to skip clip i.
    """
    if advance_event is None and period_seconds is None:
        period_seconds = 1.0  # fallback default

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


def _play_meshes(meshes, frame_keys, title, period_seconds=None, loop=False):
    """Animate a list of meshes in-place inside a SceneViewer.

    period_seconds: time between mesh swaps. Defaults to estimate.FRAME_DURATION_SECONDS.
    loop:          if True, wrap around to the start instead of closing the window.
    """
    if period_seconds is None:
        period_seconds = estimate.FRAME_DURATION_SECONDS
    scene = trimesh.Scene()
    scene.add_geometry(meshes[0], geom_name="frame_mesh")
    print(f"{title}: {frame_keys[0]} (1/{len(frame_keys)})", flush=True)
    state = {"index": 0, "last_switch": time.monotonic()}

    def _callback(scene_obj):
        now = time.monotonic()
        if now - state["last_switch"] < period_seconds:
            return
        state["last_switch"] = now
        state["index"] += 1
        if state["index"] >= len(meshes):
            if loop:
                state["index"] = 0
            else:
                try:
                    import pyglet
                    for win in list(pyglet.app.windows):
                        win.close()
                except Exception:
                    pass
                return
        i = state["index"]
        scene_obj.geometry["frame_mesh"] = meshes[i]
        print(f"{title}: {frame_keys[i]} ({i + 1}/{len(frame_keys)})", flush=True)

    try:
        SceneViewer(scene, callback=_callback, callback_period=1.0 / 30.0, start_loop=True)
    except SystemExit:
        pass


def _clip_id(clip):
    return clip.get("node_uid") or (clip.get("video_number"), clip.get("node_number"))


def _db_where_for_clip_filters(video_number=None, node_number=None,
                               include_keypoints=True):
    where = []
    if include_keypoints:
        where.append("depth_grounded_keypoints IS NOT NULL")
    params = []
    if video_number is not None:
        where.append("video_number = %s")
        params.append(int(video_number))
    if node_number is not None:
        where.append("node_number = %s")
        params.append(int(node_number))
    return where, params


def _fetch_depth_grounded_clip_metadata(video_number=None, node_number=None):
    from db import query as _db_query

    where, params = _db_where_for_clip_filters(video_number, node_number)
    sql = (
        "SELECT video_number, node_number, node_uid, "
        "left_hand_presence, right_hand_presence "
        "FROM selected_clips "
        f"WHERE {' AND '.join(where)} "
        "ORDER BY video_number, node_number"
    )
    return _db_query(sql, tuple(params))


def _fetch_depth_grounded_clips(video_number=None, node_number=None,
                                node_uid=None):
    from db import query_clips as _db_query_clips

    where, params = _db_where_for_clip_filters(video_number, node_number)
    if node_uid is not None:
        where.append("node_uid = %s")
        params.append(str(node_uid))
    sql = (
        "SELECT video_number, node_number, node_uid, "
        "depth_grounded_keypoints AS keypoints_per_frame, "
        "left_hand_presence, right_hand_presence "
        "FROM selected_clips "
        f"WHERE {' AND '.join(where)} "
        "ORDER BY video_number, node_number"
    )
    clips = _db_query_clips(sql, tuple(params))
    return [c for c in clips if c.get("keypoints_per_frame")]


def _fetch_depth_grounded_clip_by_meta(meta):
    clips = _fetch_depth_grounded_clips(
        video_number=meta.get("video_number"),
        node_number=meta.get("node_number"),
        node_uid=meta.get("node_uid"),
    )
    if not clips:
        return None
    return clips[0]


def _build_computed_meshes(raw_clip):
    frame_dict = raw_clip["keypoints_per_frame"]
    frame_keys = _numeric_frame_keys(frame_dict)
    R_wc, t_wc, gravity = estimate.compute_clip_camera_transform(frame_dict, frame_keys)
    observations = [
        estimate.build_frame_observation(frame_dict[k], R_wc, t_wc, include_visuals=True)
        for k in frame_keys
    ]
    body_pose = estimate.select_global_body_pose(observations)
    meshes = [estimate.build_frame_mesh(obs, body_pose) for obs in observations]
    return {
        "frame_keys": frame_keys,
        "R_wc": R_wc,
        "t_wc": t_wc,
        "gravity": gravity,
        "body_pose": body_pose,
        "meshes": meshes,
        "frame_dict": frame_dict,
    }


def _as_R_t(T):
    T = np.asarray(T, dtype=np.float64).reshape(4, 4)
    return T[:3, :3], T[:3, 3]


def _invert_se3(T):
    R, t = _as_R_t(T)
    R_inv = R.T
    t_inv = -R_inv @ t
    return estimate.se3_from_R_t(R_inv, t_inv)


def _rotation_error_deg(R_a, R_b):
    R = np.asarray(R_a, dtype=np.float64).T @ np.asarray(R_b, dtype=np.float64)
    tr = np.trace(R)
    c = np.clip((tr - 1.0) * 0.5, -1.0, 1.0)
    return float(np.rad2deg(np.arccos(c)))


def _fmt_vec(v):
    return "[" + ", ".join(f"{x:.4f}" for x in np.asarray(v).reshape(-1)) + "]"


def _print_se3(label, T):
    R, t = _as_R_t(T)
    print(f"{label} t={_fmt_vec(t)}", flush=True)
    print(f"{label} R rows:", flush=True)
    for row in R:
        print(f"  {_fmt_vec(row)}", flush=True)


def _load_vn_model(path, device):
    """Load a retarget VNAlignmentPolicy checkpoint. Adds retarget/ to sys.path."""
    retarget_dir = str(RETARGET_DIR)
    if retarget_dir not in sys.path:
        sys.path.insert(0, retarget_dir)
    from models.vn_transformer import VNAlignmentPolicy  # noqa: E402

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


def _make_arrow_viz(start, direction, length=0.5,
                    color=(255, 220, 40, 255), shaft_r=0.008):
    """A simple arrow at `start` pointing along `direction` (world-frame)."""
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
        K = np.array([[0.0, -axis[2], axis[1]],
                      [axis[2], 0.0, -axis[0]],
                      [-axis[1], axis[0], 0.0]], dtype=np.float64)
        R = np.eye(3) + np.sin(ang) * K + (1.0 - np.cos(ang)) * (K @ K)
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = R
    T[:3, 3] = tip - d * (head_len * 0.5)
    cone.apply_transform(T)
    cone.visual.vertex_colors = color
    return [shaft, cone]


def _smplh_tpose_mesh_at(R_world_from_body, t_world_from_body,
                         mesh_color=(180, 180, 200, 65)):
    """Build a T-pose SMPL-H neutral mesh placed at (R, t) in world coords.

    Applies the rigid transform `v_world = R @ v_body + t` to the model's
    canonical v_template vertices. No skinning / arm fitting — the mesh shows
    where the *predicted body torso frame* sits, not the actual hand-induced
    pose.
    """
    from torso_est_utils import _load_smplh_model_cached, _DEFAULT_SMPLH_MODEL_PATH
    vertices, faces, *_ = _load_smplh_model_cached(_DEFAULT_SMPLH_MODEL_PATH)
    R = np.asarray(R_world_from_body, dtype=np.float64).reshape(3, 3)
    t = np.asarray(t_world_from_body, dtype=np.float64).reshape(3)
    v_world = (R @ vertices.T).T + t
    mesh = trimesh.Trimesh(vertices=v_world, faces=faces, process=False)
    mesh.visual.vertex_colors = mesh_color
    return mesh


def _vn_prediction_case(vn_model, saved_wrist_frames, frame_keys,
                        wrist_convention, gravity_down_cam, device,
                        T_wc, R_wb_true, t_wb_true, name, color):
    T_cb_vn = _predict_torso_with_vn(
        vn_model, saved_wrist_frames, frame_keys,
        wrist_convention, gravity_down_cam, device,
    )
    if T_cb_vn is None:
        return None

    T_wb_vn = T_wc @ T_cb_vn
    R_wb_vn, t_wb_vn = _as_R_t(T_wb_vn)
    return {
        "name": name,
        "gravity_down_cam": np.asarray(gravity_down_cam, dtype=np.float64).reshape(3),
        "R_wb": R_wb_vn,
        "t_wb": t_wb_vn,
        "T_cb": T_cb_vn,
        "T_wb": T_wb_vn,
        "trans_err": (None if t_wb_true is None
                      else float(np.linalg.norm(t_wb_vn - t_wb_true))),
        "rot_err": (None if R_wb_true is None
                    else _rotation_error_deg(R_wb_true, R_wb_vn)),
        "color": color,
    }


def _rotmat_to_quat_wxyz(R: np.ndarray) -> np.ndarray:
    """3x3 rotmat -> (w, x, y, z) unit quaternion, matching smplh_backend convention."""
    q_xyzw = roma.rotmat_to_unitquat(torch.from_numpy(R.astype(np.float32))).numpy()
    return q_xyzw[[3, 0, 1, 2]]


def _wrist_frame_from_raw_hand(frame, hand_id):
    if not frame.get(hand_id):
        return None
    pts_cam = np.asarray(frame[hand_id][0]["kpts_3d"], dtype=np.float64)
    pts_cam = estimate.flip_about_wrist(
        pts_cam,
        flip_y=estimate.FLIP_HAND_Y_ABOUT_WRIST,
        flip_z=estimate.FLIP_HAND_Z_ABOUT_WRIST,
    )
    origin, R = estimate.wrist_frame_from_keypoints(pts_cam, hand_side=hand_id)
    return estimate.se3_from_R_t(R, origin)


def _raw_wrist_frames_by_frame(frame_dict, frame_keys):
    out = {}
    for fk in frame_keys:
        frame = frame_dict[fk]
        wrist_map = {}
        for hid in ["L", "R"]:
            T = _wrist_frame_from_raw_hand(frame, hid)
            if T is not None:
                wrist_map[hid] = T
        if wrist_map:
            out[str(fk)] = wrist_map
    return out


def _build_pose7_trajectory(saved_wrist_frames, frame_keys, hand_id, wrist_convention):
    """Build a [T, 7] camera-frame pose7 trajectory for one hand.

    The saved camera_to_hand_wrist_frames_se3[hid] is T_camera_from_wrist, so
    T[:3, 3] is the wrist origin in the camera frame and T[:3, :3] is
    R_camera_from_wrist — exactly the per-frame (pos, R) the VN model expects.
    Frames without a detection are filled with zeros and masked out.
    """
    poses = np.zeros((len(frame_keys), 7), dtype=np.float32)
    mask = np.zeros(len(frame_keys), dtype=bool)
    for i, fk in enumerate(frame_keys):
        wmap = saved_wrist_frames.get(str(fk), {})
        if hand_id not in wmap:
            continue
        T_cw = np.asarray(wmap[hand_id], dtype=np.float64).reshape(4, 4)
        if wrist_convention == "invert":
            T_cw = _invert_se3(T_cw)
        origin = T_cw[:3, 3].astype(np.float32)
        R = T_cw[:3, :3]
        poses[i, :3] = origin
        poses[i, 3:] = _rotmat_to_quat_wxyz(R).astype(np.float32)
        mask[i] = True
    return poses, mask


@torch.no_grad()
def _predict_torso_with_vn(vn_model, saved_wrist_frames, frame_keys,
                           wrist_convention, gravity_down_cam, device):
    """Run the VN policy once on the clip's full trajectory.

    Returns T_camera_from_body (4x4 np.ndarray) or None if both hands are
    entirely missing from the clip.
    """
    left_p7, left_m = _build_pose7_trajectory(
        saved_wrist_frames, frame_keys, "L", wrist_convention)
    right_p7, right_m = _build_pose7_trajectory(
        saved_wrist_frames, frame_keys, "R", wrist_convention)
    if not left_m.any() and not right_m.any():
        return None

    left_cam   = torch.from_numpy(left_p7 ).unsqueeze(0).to(device)   # (1, T, 7)
    right_cam  = torch.from_numpy(right_p7).unsqueeze(0).to(device)
    left_mask  = torch.from_numpy(left_m  ).unsqueeze(0).to(device)
    right_mask = torch.from_numpy(right_m ).unsqueeze(0).to(device)
    g_cam = torch.from_numpy(
        np.asarray(gravity_down_cam, dtype=np.float32)
    ).reshape(1, 3).to(device)

    if vn_model.mode == "flow":
        # VNAlignmentPolicy.sample() doesn't accept per-hand masks — flow
        # inference uses the full trajectory unmasked.
        R_pred, t_pred = vn_model.sample(left_cam, right_cam, g_cam=g_cam)
    else:
        R_pred, t_pred = vn_model(left_cam, right_cam, g_cam=g_cam,
                                  left_mask=left_mask, right_mask=right_mask)

    R_cb = R_pred[0].cpu().numpy().astype(np.float64)
    t_cb = t_pred[0].cpu().numpy().astype(np.float64)
    return estimate.se3_from_R_t(R_cb, t_cb)  # T_camera_from_body


def _compute_camera_transform_for_gravity(frame_dict, frame_keys, gravity_down_cam):
    R_wc = estimate.rotation_world_from_camera(gravity_down_cam)
    t_wc = np.zeros(3, dtype=np.float64)
    for k in frame_keys:
        hand_points = []
        frame = frame_dict[k]
        for hid in ["L", "R"]:
            if frame.get(hid):
                hand_points.append(
                    np.asarray(frame[hid][0]["kpts_3d"], dtype=np.float64))
        if hand_points:
            t_wc = estimate.estimate_world_translation(hand_points, R_wc)
            break
    return R_wc, t_wc


def _build_raw_meshes(raw_clip, vn_model=None, vn_device=None,
                      print_transforms=False, max_print_frames=3,
                      render_frame_keys=None, quiet=False,
                      use_default_gravity=False, test_gravity=False):
    """
    render_frame_keys: optional iterable of frame keys to actually build meshes
        for. The VN prediction is always computed over the *full* clip
        trajectory regardless of this filter; this only controls which frames
        get a rendered scene mesh. Default None = render every frame.
    quiet: if True, suppress per-clip VN prediction printout (caller will print
        a summary instead).
    """
    frame_dict = raw_clip["keypoints_per_frame"]
    frame_keys = _numeric_frame_keys(frame_dict)
    render_set = (set(str(k) for k in render_frame_keys)
                  if render_frame_keys is not None else None)
    R_est_wc, t_est_wc, estimated_gravity_down_cam = (
        estimate.compute_clip_camera_transform(frame_dict, frame_keys)
    )
    estimated_gravity_down_cam = np.asarray(
        estimated_gravity_down_cam, dtype=np.float64).reshape(3)
    default_gravity_down_cam = DEFAULT_GRAVITY_DOWN_CAMERA.copy()
    gravity_down_cam = (default_gravity_down_cam.copy()
                        if use_default_gravity else estimated_gravity_down_cam.copy())
    if use_default_gravity:
        R_wc, t_wc = _compute_camera_transform_for_gravity(
            frame_dict, frame_keys, default_gravity_down_cam)
    else:
        R_wc, t_wc = R_est_wc, t_est_wc
    T_wc = estimate.se3_from_R_t(R_wc, t_wc)

    observations = [
        estimate.build_frame_observation(
            frame_dict[k], R_wc, t_wc, include_visuals=False)
        for k in frame_keys
    ]
    body_pose = estimate.select_global_body_pose(observations)
    if body_pose is None:
        R_wb = None
        t_wb = None
        T_bc = None
        T_wb = None
    else:
        R_wb = np.asarray(body_pose["rotation"], dtype=np.float64).reshape(3, 3)
        t_wb = np.asarray(body_pose["translation"], dtype=np.float64).reshape(3)
        T_bc = estimate.camera_to_body_se3(R_wc, t_wc, body_pose)
        T_wb = estimate.se3_from_R_t(R_wb, t_wb)

    if print_transforms:
        print("\n=== Clip Transforms ===", flush=True)
        _print_se3("T_world_from_camera / computed_from_db_keypoints", T_wc)
        if T_bc is not None:
            _print_se3("T_body_from_camera / computed_from_db_keypoints", T_bc)
            _print_se3("T_world_from_body_computed", T_wb)
        else:
            print("No computed body pose for this clip.", flush=True)

    # For visualization and model input, derive wrist frames from the raw hand
    # keypoints in this clip. This keeps wrist-frame origins attached to
    # keypoint 0.
    wrist_convention = "direct"
    raw_wrist_frames = _raw_wrist_frames_by_frame(frame_dict, frame_keys)
    # ── one-shot VN policy prediction over the full clip ──────────────────────
    vn_prediction_cases = []
    vn_trans_err = None
    vn_rot_err = None
    if vn_model is not None:
        if test_gravity:
            gravity_cases = [
                ("estimated-gravity", estimated_gravity_down_cam, (70, 210, 255, 70)),
                ("default-gravity", default_gravity_down_cam, (255, 120, 220, 70)),
            ]
        else:
            gravity_cases = [
                ("vn", gravity_down_cam, (180, 180, 200, 65)),
            ]
        for case_name, case_gravity, case_color in gravity_cases:
            pred_case = _vn_prediction_case(
                vn_model, raw_wrist_frames, frame_keys, wrist_convention,
                case_gravity, vn_device or torch.device("cpu"),
                T_wc, R_wb, t_wb, case_name, case_color,
            )
            if pred_case is None:
                continue
            vn_prediction_cases.append(pred_case)
            if vn_trans_err is None and pred_case["trans_err"] is not None:
                vn_trans_err = pred_case["trans_err"]
                vn_rot_err = pred_case["rot_err"]
            if not quiet:
                err_str = (
                    f"trans_err: {pred_case['trans_err']:.4f} m   "
                    f"rot_err: {pred_case['rot_err']:.3f}°"
                    if pred_case["trans_err"] is not None
                    else "no computed body reference"
                )
                print(
                    f"VN prediction [{case_name}] (T_camera_from_body):\n"
                    f"  gravity down cam: {_fmt_vec(case_gravity)}\n"
                    f"  pred world torso t: {_fmt_vec(pred_case['t_wb'])}\n"
                    f"  computed world torso t: "
                    f"{_fmt_vec(t_wb) if t_wb is not None else 'None'}\n"
                    f"  {err_str}",
                    flush=True,
                )
            if print_transforms:
                _print_se3(f"T_body_from_camera_{case_name}",
                           _invert_se3(pred_case["T_cb"]))
                _print_se3(f"T_world_from_body_{case_name}", pred_case["T_wb"])

    meshes = []

    printed = 0
    rendered_frame_keys = []
    for fk in frame_keys:
        if render_set is not None and str(fk) not in render_set:
            continue
        rendered_frame_keys.append(fk)
        frame = frame_dict[fk]
        parts = [estimate.make_ground_plane()]
        parts += estimate.make_camera_viz(R_wc, t_wc)

        # Gravity arrow(s): camera-frame gravity-down vectors consumed as input,
        # drawn in world coordinates from the camera origin.
        if test_gravity:
            arrow_cases = [
                (estimated_gravity_down_cam, (70, 210, 255, 255), 0.50),
                (default_gravity_down_cam, (255, 120, 220, 255), 0.42),
            ]
        else:
            arrow_cases = [(gravity_down_cam, (255, 220, 40, 255), 0.50)]
        for arrow_gravity, arrow_color, arrow_len in arrow_cases:
            g_world_down = R_wc @ np.asarray(arrow_gravity, dtype=np.float64).reshape(3)
            parts += _make_arrow_viz(t_wc, g_world_down, length=arrow_len,
                                     color=arrow_color)

        # VN policy predictions (constant across the clip — drawn each frame).
        for pred_i, pred_case in enumerate(vn_prediction_cases):
            R_vn = pred_case["R_wb"]
            t_vn = pred_case["t_wb"]
            vn_origin = t_vn + R_vn[:, 1] * 0.20 + R_vn[:, 2] * 0.08
            if test_gravity:
                vn_origin = vn_origin + R_vn[:, 0] * (0.08 * (pred_i - 0.5))
            parts += estimate.make_torso_frame_viz(
                vn_origin, R_vn, axis_len=0.36, axis_r=0.009,
            )

        for hid, color in [("L", [60, 120, 255, 255]), ("R", [255, 120, 60, 255])]:
            if frame.get(hid):
                pts_cam = np.asarray(frame[hid][0]["kpts_3d"], dtype=np.float64)
                pts_cam = estimate.flip_about_wrist(
                    pts_cam,
                    flip_y=estimate.FLIP_HAND_Y_ABOUT_WRIST,
                    flip_z=estimate.FLIP_HAND_Z_ABOUT_WRIST,
                )
                pts_world = (R_wc @ pts_cam.T).T + t_wc
                parts += estimate.make_hand_viz(pts_world, color)

        wrist_frame_map = raw_wrist_frames.get(str(fk), {})
        wrist_frames_world = {}
        wrist_frames_cam_for_pred = {}
        for hid in ["L", "R"]:
            if hid not in wrist_frame_map:
                continue
            T_cw = np.asarray(wrist_frame_map[hid], dtype=np.float64).reshape(4, 4)
            if wrist_convention == "invert":
                T_cw = _invert_se3(T_cw)
            wrist_frames_cam_for_pred[hid] = T_cw
            T_ww = T_wc @ T_cw
            R_ww, t_ww = _as_R_t(T_ww)
            wrist_frames_world[hid] = (t_ww, R_ww)
            parts += estimate.make_wrist_frame_viz(t_ww, R_ww)

        if print_transforms and printed < max_print_frames:
            print(f"\n=== Frame {fk} ===", flush=True)
            for hid, T_ch in wrist_frames_cam_for_pred.items():
                _print_se3(f"T_hand_{hid}_from_camera raw", T_ch)

        if vn_prediction_cases:
            # When the VN policy gave a prediction, show a T-pose mesh placed
            # at the predicted body torso frame — this makes the predicted
            # body pose vs the (smaller) ground-truth torso triad visually
            # obvious.
            for pred_case in vn_prediction_cases:
                parts.append(_smplh_tpose_mesh_at(
                    pred_case["R_wb"], pred_case["t_wb"],
                    mesh_color=pred_case["color"],
                ))
        elif wrist_frames_world and body_pose is not None:
            # No VN prediction → fall back to the original heuristic SMPL fit
            # at the ground-truth torso pose for reference.
            smpl_mesh = estimate.fit_smpl_mesh_from_wrist_frames(
                wrist_frames_world,
                np.array([0.0, 0.0, -1.0]),
                body_translation=t_wb,
                body_rotation=R_wb,
                mesh_color=(180, 180, 200, 65),
                align_body_root=False,
                fit_arms=True,
                hand_frame_local_offset=estimate.SMPL_HAND_FRAME_LOCAL_OFFSET,
                return_mesh=True,
            )
            if smpl_mesh is not None:
                parts.append(smpl_mesh)

        meshes.append(trimesh.util.concatenate(parts))
        if print_transforms and printed < max_print_frames:
            printed += 1

    return {
        "frame_keys": rendered_frame_keys if render_set is not None else frame_keys,
        "all_frame_keys": frame_keys,
        "R_wc": R_wc,
        "t_wc": t_wc,
        "R_wb": R_wb,
        "t_wb": t_wb,
        "meshes": meshes,
        "wrist_convention": wrist_convention,
        "vn_trans_err": vn_trans_err,
        "vn_rot_err": vn_rot_err,
        "vn_prediction_cases": vn_prediction_cases,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--video-number", type=int, default=None,
                        help="Filter to clips from this video. Required unless "
                             "--show-many is given.")
    parser.add_argument("--node-number", type=int, default=None)
    parser.add_argument("--show-many", action="store_true",
                        help="Iterate through every matching clip, rendering one "
                             "representative frame per clip at 1 Hz. Use with "
                             "--vn-checkpoint to scan model predictions across the dataset.")
    parser.add_argument("--show-many-period", type=float, default=None,
                        help="If set, advance every N seconds. Default is "
                             "Enter-to-advance (press Enter in the terminal "
                             "to move to the next clip).")
    parser.add_argument("--show-many-loop", action="store_true",
                        help="Wrap around at the end in --show-many mode.")
    parser.add_argument("--show-many-seed", type=int, default=None,
                        help="Deterministically shuffle the --show-many clip "
                             "order with this seed. By default clips are shown "
                             "in sorted ID order.")
    parser.add_argument("--vn-checkpoint", type=str, default=None,
                        help="Path to a retarget VNAlignmentPolicy checkpoint "
                             "(e.g. ../../models/body_pose_est.pt). "
                             "If given, runs the model on the clip's wrist trajectories "
                             "and overlays the predicted torso frame for comparison.")
    parser.add_argument("--print-transforms", action="store_true")
    parser.add_argument("--max-print-frames", type=int, default=3)
    parser.add_argument("--use-default-gravity", action="store_true",
                        help="Use the default camera-frame gravity-down vector "
                             "[0, 1, 0] instead of gravity estimated from the "
                             "DB keypoints.")
    parser.add_argument("--test-gravity", action="store_true",
                        help="Run VN body estimation twice in one scene: once "
                             "with estimated clip gravity and once with the "
                             "default [0, 1, 0] gravity.")
    args = parser.parse_args()

    if not args.show_many and args.video_number is None:
        parser.error("--video-number is required unless --show-many is passed.")
    if args.test_gravity and not args.vn_checkpoint:
        parser.error("--test-gravity requires --vn-checkpoint.")

    if args.show_many:
        print("--show-many: fetching clip metadata from selected_clips ...",
              flush=True)
        clip_meta = _fetch_depth_grounded_clip_metadata(
            video_number=args.video_number,
            node_number=args.node_number,
        )
        clip_by_id = {_clip_id(c): c for c in clip_meta}
        clip_ids = list(clip_by_id.keys())
        if not clip_ids:
            raise ValueError(
                f"No selected_clips rows with depth_grounded_keypoints found for "
                f"video={args.video_number}, node={args.node_number}"
            )
        clip_ids = sorted(clip_ids, key=lambda cid: str(cid))
        if args.show_many_seed is not None:
            rng = np.random.default_rng(args.show_many_seed)
            clip_ids = [clip_ids[i] for i in rng.permutation(len(clip_ids))]
        n_total = len(clip_ids)
        loop_msg = ", looping indefinitely" if args.show_many_loop else ""
        if args.show_many_period is None:
            print(f"--show-many: streaming {n_total} clips, press Enter to "
                  f"advance{loop_msg} (close window to stop)", flush=True)
        else:
            print(f"--show-many: streaming {n_total} clips at "
                  f"{args.show_many_period:.2f}s/clip{loop_msg} "
                  f"(close window to stop)",
                  flush=True)
        if args.show_many_seed is not None:
            print(f"--show-many: shuffled clip order with seed {args.show_many_seed}",
                  flush=True)
        vn_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        vn_model = _load_vn_model(Path(args.vn_checkpoint), vn_device) if args.vn_checkpoint else None
        if vn_model is not None:
            print(
                f"Loaded VN policy from {args.vn_checkpoint} "
                f"(mode={vn_model.mode}) on {vn_device}",
                flush=True,
            )
        # Running tally of VN errors for an end-of-stream summary.
        errs = {"t": [], "r": []}
        clip_cache = {}

        # Restrict per-clip rendering to a single representative frame for
        # speed. The VN prediction still uses the full clip trajectory.
        def _build_clip_fast(i):
            if i < 0 or i >= n_total:
                return None
            cid = clip_ids[i]
            raw_clip = clip_cache.get(cid)
            if raw_clip is None:
                raw_clip = _fetch_depth_grounded_clip_by_meta(clip_by_id[cid])
                if raw_clip is None:
                    print(f"[skip {i+1}/{n_total} cid={cid}] missing DB keypoints",
                          flush=True)
                    return None
                clip_cache[cid] = raw_clip
            all_keys = _numeric_frame_keys(raw_clip["keypoints_per_frame"])
            if not all_keys:
                return None
            mid_idx = len(all_keys) // 2
            middle_fk = all_keys[mid_idx]
            total_frames = len(all_keys)
            try:
                result = _build_raw_meshes(
                    raw_clip,
                    vn_model=vn_model, vn_device=vn_device,
                    print_transforms=False, max_print_frames=0,
                    render_frame_keys=[middle_fk], quiet=True,
                    use_default_gravity=args.use_default_gravity,
                    test_gravity=args.test_gravity,
                )
            except Exception as e:
                print(f"[skip {i+1}/{n_total} cid={cid}] {type(e).__name__}: {e}",
                      flush=True)
                return None
            if not result["meshes"]:
                return None
            mesh = result["meshes"][0]
            te = result.get("vn_trans_err")
            re = result.get("vn_rot_err")
            if te is not None:
                errs["t"].append(te)
                errs["r"].append(re)
            err_str = (f"trans_err={te:.4f}m rot_err={re:.3f}°"
                       if te is not None else "no-vn-pred")
            label = (f"clip {i+1}/{n_total} "
                     f"video={raw_clip.get('video_number')} "
                     f"node={raw_clip.get('node_number')} "
                     f"uid={raw_clip.get('node_uid')} "
                     f"frame={middle_fk} ({mid_idx + 1}/{total_frames}) "
                     f"{err_str}")
            return mesh, label

        advance_event = (_make_stdin_advance_event()
                         if args.show_many_period is None else None)
        try:
            _play_clips_streaming(_build_clip_fast, n_total, title="show-many",
                                  period_seconds=args.show_many_period,
                                  advance_event=advance_event,
                                  loop=args.show_many_loop)
        finally:
            if errs["t"]:
                ts = np.array(errs["t"]); rs = np.array(errs["r"])
                print(
                    f"\nVN error summary over {len(ts)} clips shown:\n"
                    f"  trans  mean={ts.mean():.4f}m  median={np.median(ts):.4f}m  "
                    f"p90={np.percentile(ts,90):.4f}m  max={ts.max():.4f}m\n"
                    f"  rot    mean={rs.mean():.3f}°  median={np.median(rs):.3f}°  "
                    f"p90={np.percentile(rs,90):.3f}°  max={rs.max():.3f}°\n",
                    flush=True,
                )
        return

    # ── single-clip mode (default) ────────────────────────────────────────────
    clips = _fetch_depth_grounded_clips(
        video_number=args.video_number,
        node_number=args.node_number,
    )
    clip_by_id = {_clip_id(c): c for c in clips}
    clip_ids = list(clip_by_id.keys())
    if not clip_ids:
        raise ValueError(
            f"No selected_clips rows with depth_grounded_keypoints found for "
            f"video={args.video_number}, node={args.node_number}"
        )

    vn_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    vn_model = _load_vn_model(Path(args.vn_checkpoint), vn_device) if args.vn_checkpoint else None
    if vn_model is not None:
        print(f"Loaded VN policy from {args.vn_checkpoint} (mode={vn_model.mode}) on {vn_device}",
              flush=True)

    if args.node_number is None:
        rng = np.random.default_rng()
        cid = clip_ids[int(rng.integers(len(clip_ids)))]
    else:
        clip_ids = sorted(clip_ids, key=lambda cid: str(cid))
        cid = clip_ids[0]
    raw_clip = clip_by_id[cid]

    print(
        f"Selected clip video={raw_clip.get('video_number')} "
        f"node={raw_clip.get('node_number')} uid={raw_clip.get('node_uid')}",
        flush=True,
    )

    result = _build_raw_meshes(
        raw_clip,
        vn_model=vn_model,
        vn_device=vn_device,
        print_transforms=args.print_transforms,
        max_print_frames=args.max_print_frames,
        use_default_gravity=args.use_default_gravity,
        test_gravity=args.test_gravity,
    )
    _play_meshes(result["meshes"], result["frame_keys"], title="DB")


if __name__ == "__main__":
    main()
