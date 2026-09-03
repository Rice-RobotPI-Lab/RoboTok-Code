"""Utilities for wrist frame estimation from hand keypoints."""

from pathlib import Path

import numpy as np
import trimesh


# MediaPipe / WiLoR keypoint indices
_WRIST = 0
_INDEX_MCP = 5  # index finger knuckle
_MIDDLE_MCP = 9  # middle finger knuckle
_PINKY_MCP = 17  # pinky knuckle

_DEFAULT_SMPLH_MODEL_PATH = Path(__file__).resolve().parent / "neutral" / "model.npz"

# SMPL-H body joint indices. The body is kept in its neutral standing pose
# except for these arm chains.
_SMPLH_LEFT_SHOULDER = 16
_SMPLH_RIGHT_SHOULDER = 17
_SMPLH_LEFT_ELBOW = 18
_SMPLH_RIGHT_ELBOW = 19
_SMPLH_LEFT_WRIST = 20
_SMPLH_RIGHT_WRIST = 21
_SMPLH_LEFT_MIDDLE1 = 25
_SMPLH_LEFT_PINKY1 = 28
_SMPLH_RIGHT_MIDDLE1 = 40
_SMPLH_RIGHT_PINKY1 = 43

_SMPLH_ARM_CHAINS = {
    "L": (_SMPLH_LEFT_SHOULDER, _SMPLH_LEFT_ELBOW, _SMPLH_LEFT_WRIST),
    "R": (_SMPLH_RIGHT_SHOULDER, _SMPLH_RIGHT_ELBOW, _SMPLH_RIGHT_WRIST),
}
_SMPLH_MODEL_CACHE = {}


def wrist_frame_from_keypoints(kpts_3d, hand_side=None):
    """Compute a right-handed XYZ frame at the wrist.

    Joints used (MediaPipe/WiLoR ordering):
      0  — wrist (origin)
      5  — index MCP (for palm normal)
      9  — middle MCP (middle knuckle)  → defines X axis
      17 — pinky MCP (for palm normal / lateral direction)

    Frame axes (right-hand rule: Z = cross(X, Y)):
      X (red)   — wrist → middle MCP
      Z (blue)  — palm normal (primary axis)
      Y (green) — derived from Z and X to keep a right-handed frame

    Args:
        kpts_3d: array-like (21, 3), already in the visualization coordinate
                 system (i.e. after any depth-grounding or Z-flip transforms).

    Returns:
        origin (3,): wrist position
        R (3, 3): rotation matrix whose columns are [x_axis, y_axis, z_axis]
    """
    pts = np.asarray(kpts_3d, dtype=np.float64)
    wrist = pts[_WRIST]
    index_mcp = pts[_INDEX_MCP]
    middle_mcp = pts[_MIDDLE_MCP]
    pinky_mcp = pts[_PINKY_MCP]

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

    # Keep a consistent palm-normal convention across hands.
    if hand_side is not None and str(hand_side).upper().startswith("L"):
        z = -z

    y = np.cross(z, x)

    return wrist, np.column_stack([x, y, z])


def make_wrist_frame_viz(origin, R, axis_len=0.035, axis_r=0.0015):
    """Render a wrist XYZ frame as colored cylinders.

    X = red, Y = green, Z = blue  (same convention as the camera frame).

    Args:
        origin (3,): frame origin in visualization space
        R (3, 3): rotation matrix with columns [x_axis, y_axis, z_axis]
        axis_len: length of each axis cylinder in metres
        axis_r: radius of each axis cylinder in metres

    Returns:
        list of trimesh objects
    """
    axis_colors = [
        [255, 0, 0, 255],   # X — red
        [0, 255, 0, 255],   # Y — green
        [0, 0, 255, 255],   # Z — blue
    ]
    parts = []
    for i, color in enumerate(axis_colors):
        tip = origin + R[:, i] * axis_len
        cyl = trimesh.creation.cylinder(radius=axis_r, segment=[origin, tip])
        cyl.visual.vertex_colors = color
        parts.append(cyl)

    # Small sphere at origin
    sphere = trimesh.creation.uv_sphere(radius=axis_r * 2.5)
    sphere.apply_translation(origin)
    sphere.visual.vertex_colors = [255, 255, 255, 255]
    parts.append(sphere)

    return parts


def fit_smplh_tpose_arms_from_wrist_frames(
        wrist_frames,
        gravity_vec,
        model_path=_DEFAULT_SMPLH_MODEL_PATH,
        body_translation=None,
        body_rotation=None,
        mesh_color=(180, 180, 200, 255),
        gravity_arrow_color=(255, 220, 40, 255),
        align_body_root=True,
        fit_arms=True,
        return_diagnostics=False,
        hand_frame_local_offset=(0.0, 0.0, 0.0),
        return_mesh=True):
    """Pose a neutral SMPL-H mesh upright while moving only the arms.

    Args:
        wrist_frames: mapping with optional ``"L"`` and ``"R"`` entries.
            Each entry may be either ``(origin, R)`` or a dict containing
            ``{"origin": ..., "R": ...}``, where origin is ``(3,)`` and R is
            a wrist-frame rotation matrix with columns as axes in world/camera
            coordinates.
        gravity_vec: required ``(3,)`` vector pointing down in the same
            coordinate system as the wrist frames. The SMPL-H +Y axis is
            rotated to ``-gravity_vec`` so the body stands upright.
        model_path: SMPL-H ``model.npz`` path.
        body_translation: optional world translation for the model pelvis. If
            omitted, the mesh is translated so the neutral T-pose wrist
            midpoint (or single available wrist) lines up with the provided
            wrist origin(s) before arm IK.
        body_rotation: optional world rotation matrix ``(3,3)`` for the body
            root. When provided, it overrides upright/yaw estimation.
        mesh_color: RGBA vertex color applied to the returned mesh.
        gravity_arrow_color: RGBA color for the arrow drawn next to the head
            in the direction of ``gravity_vec``.
        align_body_root: if True, solve the upright body's yaw and translation
            from the available wrist frames before arm IK.
        fit_arms: if False, keep both arms in neutral T-pose and skip wrist IK.
        return_diagnostics: if True, return ``(mesh, diagnostics)`` where
            diagnostics includes a fit score used for candidate selection.
        hand_frame_local_offset: local hand-frame offset from the wrist joint
            in the hand frame coordinates (X, Y, Z). The extracted frame origin
            is treated as a hand-frame anchor, and this offset is removed to
            produce the wrist-joint IK target.
        return_mesh: if False, skip skinning/mesh creation and return only
            diagnostics when ``return_diagnostics`` is True.

    Returns:
        trimesh.Trimesh: posed SMPL-H mesh. Body, head, torso, and legs remain
        in neutral T-pose; only shoulders, elbows, and wrists receive non-
        identity joint rotations.
    """
    if gravity_vec is None:
        raise ValueError("gravity_vec is required to orient the SMPL-H mesh upright")

    frames = _normalize_wrist_frames(wrist_frames)
    (vertices, faces, weights, joints, parents,
     hand_frame_local_map, arm_chain_params) = _load_smplh_model_cached(model_path)

    root_R, root_t = _estimate_body_root_transform(
        frames,
        joints,
        gravity_vec,
        body_translation,
        body_rotation,
        align_body_root=align_body_root,
    )

    local_rots = np.tile(np.eye(3), (len(joints), 1, 1))

    reach_errors = []
    rotation_magnitudes = []
    target_wrist_world = {}
    fitted_wrist_world = {}

    if fit_arms:
        hand_frame_local_offset = np.asarray(
            hand_frame_local_offset, dtype=np.float64).reshape(3)
        for side, frame in frames.items():
            if side not in _SMPLH_ARM_CHAINS:
                continue
            shoulder, elbow, wrist = _SMPLH_ARM_CHAINS[side]
            target_frame_R = root_R.T @ frame["R"]
            target_hand_origin = root_R.T @ (frame["origin"] - root_t)
            target_wrist = target_hand_origin - target_frame_R @ hand_frame_local_offset
            hand_frame_local_R = hand_frame_local_map.get(side, np.eye(3))
            target_wrist_global_R = target_frame_R @ hand_frame_local_R.T

            shoulder_R, elbow_R, wrist_R, fit_diag = _solve_arm_chain_rotations(
                arm_chain_params[side], target_wrist, target_wrist_global_R)
            local_rots[shoulder] = shoulder_R
            local_rots[elbow] = elbow_R
            local_rots[wrist] = wrist_R
            reach_errors.append(float(fit_diag["wrist_reach_error"]))
            rotation_magnitudes.append(float(fit_diag["rotation_magnitude"]))
            target_wrist_world[side] = root_R @ target_wrist + root_t
            fitted_wrist_world[side] = root_R @ fit_diag["fitted_wrist"] + root_t

    mean_reach_error = float(np.mean(reach_errors)) if reach_errors else 0.0
    mean_rotation = float(np.mean(rotation_magnitudes)) if rotation_magnitudes else 0.0
    diagnostics = {
        "mean_wrist_reach_error": mean_reach_error,
        "mean_rotation_magnitude": mean_rotation,
        "fit_score": mean_reach_error + 0.05 * mean_rotation,
        "target_wrist_world": target_wrist_world,
        "fitted_wrist_world": fitted_wrist_world,
    }

    joint_globals = _joint_global_transforms(joints, parents, local_rots)
    posed_joints = np.asarray(
        [joint_globals[j, :3, 3] for j in range(len(joints))],
        dtype=np.float64,
    )
    posed_joints_world = (root_R @ posed_joints.T).T + root_t
    fitted_hand_frames_world = {}
    hand_frame_errors = {}
    for side, frame in frames.items():
        if side not in _SMPLH_ARM_CHAINS:
            continue
        fitted_origin, fitted_R = _smplh_hand_frame_from_joints(
            posed_joints_world, side)
        fitted_hand_frames_world[side] = {
            "origin": fitted_origin,
            "R": fitted_R,
        }
        hand_frame_errors[side] = {
            "origin_error": float(np.linalg.norm(frame["origin"] - fitted_origin)),
            "rotation_error": _rotation_angle(frame["R"].T @ fitted_R),
        }
    diagnostics["fitted_hand_frames_world"] = fitted_hand_frames_world
    diagnostics["hand_frame_errors"] = hand_frame_errors

    if not return_mesh:
        if return_diagnostics:
            return None, diagnostics
        return None

    transforms = _joint_skinning_transforms(joints, parents, local_rots)
    posed_vertices = _skin_vertices(vertices, weights, transforms)
    posed_vertices = (root_R @ posed_vertices.T).T + root_t

    mesh = trimesh.Trimesh(vertices=posed_vertices, faces=faces, process=False)
    mesh.visual.vertex_colors = mesh_color

    gravity_arrow = _make_gravity_arrow_viz(
        posed_vertices, gravity_vec, gravity_arrow_color)
    out_mesh = trimesh.util.concatenate([mesh, gravity_arrow])
    if not return_diagnostics:
        return out_mesh
    return out_mesh, diagnostics


def fit_smpl_mesh_from_wrist_frames(*args, **kwargs):
    """Alias for ``fit_smplh_tpose_arms_from_wrist_frames``."""
    return fit_smplh_tpose_arms_from_wrist_frames(*args, **kwargs)


def _load_smplh_model_cached(model_path):
    path_key = str(model_path)
    cached = _SMPLH_MODEL_CACHE.get(path_key)
    if cached is not None:
        return cached

    data = np.load(model_path, allow_pickle=True)
    vertices = np.asarray(data["v_template"], dtype=np.float64)
    faces = np.asarray(data["f"])
    weights = np.asarray(data["weights"], dtype=np.float64)
    joints = _smplh_joints(data, vertices, weights.shape[1])
    parents = _smplh_parents(data["kintree_table"], len(joints))
    hand_frame_local_map = {
        "L": _smplh_hand_frame_local_from_joints(joints, "L"),
        "R": _smplh_hand_frame_local_from_joints(joints, "R"),
    }
    arm_chain_params = {}
    for side, chain in _SMPLH_ARM_CHAINS.items():
        shoulder, elbow, wrist = chain
        shoulder_p = joints[shoulder]
        elbow_p = joints[elbow]
        wrist_p = joints[wrist]
        upper_rest = elbow_p - shoulder_p
        forearm_rest = wrist_p - elbow_p
        arm_chain_params[side] = {
            "indices": chain,
            "shoulder_p": shoulder_p,
            "elbow_p": elbow_p,
            "wrist_p": wrist_p,
            "upper_rest": upper_rest,
            "forearm_rest": forearm_rest,
            "upper_len": max(np.linalg.norm(upper_rest), 1e-8),
            "forearm_len": max(np.linalg.norm(forearm_rest), 1e-8),
        }

    cached = (vertices, faces, weights, joints, parents,
              hand_frame_local_map, arm_chain_params)
    _SMPLH_MODEL_CACHE[path_key] = cached
    return cached


def _normalize_wrist_frames(wrist_frames):
    frames = {}
    for side, frame in wrist_frames.items():
        side_key = str(side).upper()[0]
        if isinstance(frame, dict):
            origin = frame["origin"]
            R = frame["R"]
        else:
            origin, R = frame
        frames[side_key] = {
            "origin": np.asarray(origin, dtype=np.float64).reshape(3),
            "R": np.asarray(R, dtype=np.float64).reshape(3, 3),
        }
    return frames


def _safe_unit(vec, fallback=None, eps=1e-8):
    arr = np.asarray(vec, dtype=np.float64)
    norm = np.linalg.norm(arr)
    if norm < eps:
        if fallback is None:
            fallback = np.array([0.0, 1.0, 0.0])
        return np.asarray(fallback, dtype=np.float64)
    return arr / norm


def _rotation_between(src, dst):
    src = _safe_unit(src)
    dst = _safe_unit(dst)
    cross = np.cross(src, dst)
    dot = float(np.clip(np.dot(src, dst), -1.0, 1.0))
    cross_norm = np.linalg.norm(cross)
    if cross_norm < 1e-8:
        if dot > 0.0:
            return np.eye(3)
        axis = _orthogonal_axis(src)
        return _axis_angle_to_matrix(axis, np.pi)

    K = _skew(cross / cross_norm)
    angle = np.arctan2(cross_norm, dot)
    return np.eye(3) + np.sin(angle) * K + (1.0 - np.cos(angle)) * (K @ K)


def _axis_angle_to_matrix(axis, angle):
    axis = _safe_unit(axis)
    K = _skew(axis)
    return np.eye(3) + np.sin(angle) * K + (1.0 - np.cos(angle)) * (K @ K)


def _skew(vec):
    x, y, z = vec
    return np.array([[0.0, -z, y],
                     [z, 0.0, -x],
                     [-y, x, 0.0]], dtype=np.float64)


def _orthogonal_axis(vec):
    vec = _safe_unit(vec)
    basis = np.array([1.0, 0.0, 0.0])
    if abs(np.dot(vec, basis)) > 0.9:
        basis = np.array([0.0, 0.0, 1.0])
    return _safe_unit(np.cross(vec, basis))


def _smplh_parents(kintree_table, num_joints):
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


def _smplh_joints(data, vertices, num_skinning_joints):
    joints = np.asarray(data["J"], dtype=np.float64)
    if len(joints) == num_skinning_joints:
        return joints

    if "J_regressor" in data:
        regressor = np.asarray(data["J_regressor"], dtype=np.float64)
        if regressor.shape[0] == num_skinning_joints:
            return regressor @ vertices

    raise ValueError(
        "SMPL-H model has incompatible joint and skinning-weight counts: "
        f"{len(joints)} joints for {num_skinning_joints} weight columns")


def _estimate_body_root_transform(frames, joints, gravity_vec, body_translation,
                                  body_rotation=None, align_body_root=True):
    if body_rotation is not None:
        root_R = np.asarray(body_rotation, dtype=np.float64).reshape(3, 3)
        root_t = (_estimate_body_translation(frames, joints, root_R)
                  if body_translation is None
                  else np.asarray(body_translation, dtype=np.float64))
        return root_R, root_t

    up_dir = -_safe_unit(gravity_vec)
    upright_R = _rotation_between(np.array([0.0, 1.0, 0.0]), up_dir)

    if align_body_root and len(frames) >= 2:
        root_R = _align_body_yaw_to_wrist_span(frames, joints, upright_R, up_dir)
    else:
        root_R = upright_R

    root_t = (_estimate_body_translation(frames, joints, root_R)
              if body_translation is None
              else np.asarray(body_translation, dtype=np.float64))
    return root_R, root_t


def _align_body_yaw_to_wrist_span(frames, joints, upright_R, up_dir):
    if "L" not in frames or "R" not in frames:
        return upright_R

    rest_span = upright_R @ (
        joints[_SMPLH_LEFT_WRIST] - joints[_SMPLH_RIGHT_WRIST])
    target_span = frames["L"]["origin"] - frames["R"]["origin"]

    rest_span = _project_to_plane(rest_span, up_dir)
    target_span = _project_to_plane(target_span, up_dir)
    if np.linalg.norm(rest_span) < 1e-8 or np.linalg.norm(target_span) < 1e-8:
        return upright_R

    rest_span = _safe_unit(rest_span)
    target_span = _safe_unit(target_span)
    signed_cross = np.dot(up_dir, np.cross(rest_span, target_span))
    signed_dot = np.clip(np.dot(rest_span, target_span), -1.0, 1.0)
    yaw = np.arctan2(signed_cross, signed_dot)
    return _axis_angle_to_matrix(up_dir, yaw) @ upright_R


def _project_to_plane(vec, normal):
    normal = _safe_unit(normal)
    vec = np.asarray(vec, dtype=np.float64)
    return vec - normal * np.dot(vec, normal)


def _estimate_body_translation(frames, joints, root_R):
    pairs = []
    for side, (_, _, wrist) in _SMPLH_ARM_CHAINS.items():
        if side in frames:
            pairs.append((frames[side]["origin"], root_R @ joints[wrist]))
    if not pairs:
        return np.zeros(3, dtype=np.float64)
    target_mid = np.mean([p[0] for p in pairs], axis=0)
    rest_mid = np.mean([p[1] for p in pairs], axis=0)
    return target_mid - rest_mid


def _solve_arm_chain_rotations(arm_chain, target_wrist, target_wrist_global_R):
    shoulder_p = arm_chain["shoulder_p"]
    elbow_p = arm_chain["elbow_p"]
    wrist_p = arm_chain["wrist_p"]
    upper_rest = arm_chain["upper_rest"]
    forearm_rest = arm_chain["forearm_rest"]
    upper_len = arm_chain["upper_len"]
    forearm_len = arm_chain["forearm_len"]

    target_vec = target_wrist - shoulder_p
    min_reach = abs(upper_len - forearm_len) + 1e-6
    max_reach = upper_len + forearm_len - 1e-6
    target_dist = np.clip(np.linalg.norm(target_vec), min_reach, max_reach)
    target_dir = _safe_unit(target_vec, fallback=_safe_unit(wrist_p - shoulder_p))
    target_wrist_clamped = shoulder_p + target_dir * target_dist

    elbow_pole = elbow_p - shoulder_p
    pole = elbow_pole - target_dir * np.dot(elbow_pole, target_dir)
    pole = _safe_unit(pole, fallback=_orthogonal_axis(target_dir))

    elbow_along = ((upper_len * upper_len - forearm_len * forearm_len
                    + target_dist * target_dist) / (2.0 * target_dist))
    elbow_height = np.sqrt(max(upper_len * upper_len - elbow_along * elbow_along,
                               0.0))
    target_elbow = shoulder_p + target_dir * elbow_along + pole * elbow_height

    target_upper = target_elbow - shoulder_p
    target_forearm = target_wrist_clamped - target_elbow

    shoulder_R = _rotation_between(upper_rest, target_upper)
    elbow_global_R = _rotation_between(forearm_rest, target_forearm)
    elbow_R = shoulder_R.T @ elbow_global_R
    wrist_R = elbow_global_R.T @ target_wrist_global_R

    fit_diag = {
        "wrist_reach_error": np.linalg.norm(target_wrist - target_wrist_clamped),
        "target_wrist": target_wrist,
        "fitted_wrist": target_wrist_clamped,
        "rotation_magnitude": (
            _rotation_angle(shoulder_R)
            + _rotation_angle(elbow_R)
            + _rotation_angle(wrist_R)
        ),
    }
    return shoulder_R, elbow_R, wrist_R, fit_diag


def _smplh_hand_frame_local_from_joints(joints, side):
    _, R = _smplh_hand_frame_from_joints(joints, side)
    return R


def _smplh_hand_frame_from_joints(joints, side):
    side = str(side).upper()[0]
    if side == "L":
        wrist_i = _SMPLH_LEFT_WRIST
        middle_i = _SMPLH_LEFT_MIDDLE1
        pinky_i = _SMPLH_LEFT_PINKY1
    else:
        wrist_i = _SMPLH_RIGHT_WRIST
        middle_i = _SMPLH_RIGHT_MIDDLE1
        pinky_i = _SMPLH_RIGHT_PINKY1

    if max(wrist_i, middle_i, pinky_i) >= len(joints):
        return np.zeros(3, dtype=np.float64), np.eye(3)

    wrist = joints[wrist_i]
    middle = joints[middle_i]
    pinky = joints[pinky_i]

    x = middle - wrist
    x = _safe_unit(x, fallback=np.array([1.0, 0.0, 0.0]))
    pinky_dir = pinky - wrist
    z = np.cross(x, pinky_dir)
    z = _safe_unit(z, fallback=np.array([0.0, 0.0, 1.0]))

    # Match the side-specific normal convention used in keypoint frame extraction.
    if side == "L":
        z = -z

    y = _safe_unit(np.cross(z, x), fallback=np.array([0.0, 1.0, 0.0]))
    z = _safe_unit(np.cross(x, y), fallback=z)
    return wrist, np.column_stack([x, y, z])


def _rotation_angle(R):
    trace = np.trace(R)
    cos_theta = np.clip((trace - 1.0) * 0.5, -1.0, 1.0)
    return float(np.arccos(cos_theta))


def _joint_skinning_transforms(joints, parents, local_rots):
    globals_ = _joint_global_transforms(joints, parents, local_rots)
    rest_globals = _joint_global_transforms(
        joints, parents, np.tile(np.eye(3), (len(joints), 1, 1)))
    return globals_ @ np.linalg.inv(rest_globals)


def _joint_global_transforms(joints, parents, local_rots):
    num_joints = len(joints)
    globals_ = np.zeros((num_joints, 4, 4), dtype=np.float64)

    for j in range(num_joints):
        parent = int(parents[j])
        rel_t = joints[j] if parent < 0 else joints[j] - joints[parent]

        local = np.eye(4, dtype=np.float64)
        local[:3, :3] = local_rots[j]
        local[:3, 3] = rel_t

        if parent < 0:
            globals_[j] = local
        else:
            globals_[j] = globals_[parent] @ local

    return globals_


def _skin_vertices(vertices, weights, joint_transforms):
    transforms = np.einsum("vj,jab->vab", weights,
                           joint_transforms[:weights.shape[1]])
    homogeneous = np.concatenate(
        [vertices, np.ones((len(vertices), 1), dtype=np.float64)], axis=1)
    return np.einsum("vab,vb->va", transforms, homogeneous)[:, :3]


def _make_gravity_arrow_viz(vertices, gravity_vec, color):
    gravity_dir = _safe_unit(gravity_vec)
    up_dir = -gravity_dir

    head_center = vertices[np.argmax(vertices @ up_dir)]
    side_dir = _orthogonal_axis(up_dir)

    body_span = np.ptp(vertices, axis=0).max()
    arrow_len = max(body_span * 0.16, 0.15)
    shaft_len = arrow_len * 0.72
    head_len = arrow_len - shaft_len
    shaft_r = arrow_len * 0.025
    head_r = shaft_r * 2.6

    start = head_center + up_dir * (arrow_len * 0.22) + side_dir * (arrow_len * 0.55)
    shaft_end = start + gravity_dir * shaft_len
    tip = start + gravity_dir * arrow_len

    shaft = trimesh.creation.cylinder(
        radius=shaft_r, segment=[start, shaft_end], sections=24)
    shaft.visual.vertex_colors = color

    cone = trimesh.creation.cone(radius=head_r, height=head_len, sections=32)
    transform = np.eye(4, dtype=np.float64)
    transform[:3, :3] = _rotation_between(np.array([0.0, 0.0, 1.0]), gravity_dir)
    transform[:3, 3] = tip - gravity_dir * (head_len * 0.5)
    cone.apply_transform(transform)
    cone.visual.vertex_colors = color

    return trimesh.util.concatenate([shaft, cone])
