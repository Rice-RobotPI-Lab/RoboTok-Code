"""Depth-ground DB keypoints with MoGe-2 metric depth estimation.

For each clip, decodes video frames, runs MoGe-2 depth estimation, and
places WiLoR's 3D hand poses into metric space via a rigid per-hand
transform (uniform scale + wrist translation).  WiLoR outputs all hands
at a single canonical 3D hand size, so the full articulation/pose is
preserved exactly — only the hand's position and metric scale change.

Per hand per frame, one robust Z is sampled from the depth map (median of
reliable joints after MAD outlier rejection), and the wrist is back-projected
to metric 3D.  A fixed anatomical scale (~0.08m wrist-to-MCP9) maps WiLoR's
canonical hand size to real-world meters — the same scale for every frame,
free of depth/focal noise.  The entire WiLoR pose is then rigidly transformed:

    kpts_3d_metric = (wilor_kpts_3d - wrist) * scale + wrist_metric

Temporal smoothing operates on the 3D wrist position per hand (scale is
constant), never on individual joint coordinates — within-frame articulation
is untouched.

HaWoR-infilled frames get their wrist position and scale by linear
interpolation from bracketing grounded frames, applied to the infilled
pose.

Every hand is guaranteed to be grounded: if reliable joints fail, all
visible joints are tried.  If that also fails, the hand falls back to the
other hand's depth and scale, then to the clip median.

Clips whose MP4 is missing are dropped from the output entirely.

Input:  selected_clips.keypoints_per_frame (private clip database via the
        `db` module, not distributed with this repo — see README)
Output: selected_clips.depth_grounded_keypoints (same schema, grounded kpts_3d)

Usage:
    python 0depth_ground_keypoints.py                    # process all unprocessed clips
    python 0depth_ground_keypoints.py -n 500             # next 500 unprocessed videos
    python 0depth_ground_keypoints.py --default-gravity  # skip GeoCalib gravity estimation
    python 0depth_ground_keypoints.py --reprocess        # redo already-processed clips
"""
import argparse
import json
import sys
import time
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch

# `db` is the private clip-database module, not distributed with this repo
# (see README). This pipeline step only runs against the private database;
# later steps run from the exported .pt artifacts.
import db  # noqa: E402


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

def _log(msg: str):
    print(msg, flush=True)


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

from encoder_utils import FRAMES_DIR as DEFAULT_FRAMES_DIR  # noqa: E402

MOGE2_MODEL_NAME = "Ruicheng/moge-2-vitl"

MIN_DEPTH = 0.01  # meters — floor for any assigned Z value
DEFAULT_GRAVITY = np.array([0.0, 1.0, 0.0], dtype=np.float32)


# ---------------------------------------------------------------------------
# MoGe-2 depth estimator  (adapted from the EgoInfinity pipeline — see README)
# ---------------------------------------------------------------------------

@dataclass
class DepthResult:
    depth: np.ndarray          # (H, W) metric depth in meters
    focal_length_px: float     # estimated focal length in pixels


class MoGe2Estimator:

    def __init__(self, model_name: str = MOGE2_MODEL_NAME,
                 device: str = "cuda", resolution_level: int = 9):
        from moge.model.v2 import MoGeModel
        self.device = torch.device(device)
        self.model = MoGeModel.from_pretrained(model_name).to(self.device)
        self.model.eval()
        self.resolution_level = resolution_level

    @torch.no_grad()
    def estimate(self, image_bgr: np.ndarray) -> DepthResult:
        H, W = image_bgr.shape[:2]
        rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        t = torch.tensor(rgb / 255, dtype=torch.float32,
                         device=self.device).permute(2, 0, 1)
        out = self.model.infer(t, resolution_level=self.resolution_level,
                               apply_mask=True, force_projection=True,
                               use_fp16=True)
        depth = out["depth"].cpu().numpy()
        mask = out["mask"].cpu().numpy()
        intrinsics = out["intrinsics"].cpu().numpy()
        depth[~mask] = 0.0
        fx_px = float(intrinsics[0, 0]) * W
        fy_px = float(intrinsics[1, 1]) * H
        focal = (fx_px + fy_px) / 2
        return DepthResult(depth=depth, focal_length_px=focal)


# ---------------------------------------------------------------------------
# Depth sampling  (adapted from the EgoInfinity pipeline — see README)
# ---------------------------------------------------------------------------

RELIABLE_JOINT_IDS = [0, 5, 9, 13, 17]  # wrist + 4 MCP joints
ALL_JOINT_IDS = list(range(21))


def _adaptive_patch_size(kpts_2d_px: np.ndarray,
                         visible_mask: np.ndarray) -> int:
    vis = kpts_2d_px[visible_mask]
    if len(vis) < 2:
        return 5
    bbox_diag = np.linalg.norm(vis.max(axis=0) - vis.min(axis=0))
    return max(3, min(9, int(round(bbox_diag * 0.05))))


def sample_depth_at_joints(depth_map: np.ndarray,
                           joints_2d: np.ndarray,
                           joint_ids: list,
                           patch_size: int = 5) -> np.ndarray:
    H, W = depth_map.shape
    half = patch_size // 2
    depths = np.full(len(joint_ids), np.nan)
    for i, jid in enumerate(joint_ids):
        x, y = joints_2d[jid]
        xi, yi = int(round(x)), int(round(y))
        if xi < 0 or xi >= W or yi < 0 or yi >= H:
            continue
        x0, x1 = max(0, xi - half), min(W, xi + half + 1)
        y0, y1 = max(0, yi - half), min(H, yi + half + 1)
        patch = depth_map[y0:y1, x0:x1]
        valid = patch[patch > MIN_DEPTH]
        if len(valid) > 0:
            depths[i] = float(np.median(valid))
    return depths


def _reject_depth_outliers(joint_ids: list,
                           depths: np.ndarray) -> Dict[int, float]:
    """Keep only inlier depths using MAD (median absolute deviation)."""
    valid_mask = ~np.isnan(depths) & (depths > MIN_DEPTH)
    valid_ids = [jid for jid, m in zip(joint_ids, valid_mask) if m]
    valid_d = depths[valid_mask]

    if len(valid_d) <= 2:
        return {jid: float(d) for jid, d in zip(valid_ids, valid_d)}

    med = float(np.median(valid_d))
    mad = float(np.median(np.abs(valid_d - med)))
    if mad < 1e-6:
        return {jid: float(d) for jid, d in zip(valid_ids, valid_d)}

    thresh = 3.0 * mad
    return {jid: float(d)
            for jid, d in zip(valid_ids, valid_d)
            if abs(d - med) <= thresh}


# ---------------------------------------------------------------------------
# Rigid hand transform: uniform scale + wrist translation
# ---------------------------------------------------------------------------

@dataclass
class HandGrounding:
    wrist_metric: np.ndarray   # (3,) metric 3D wrist position
    scale: float               # uniform WiLoR-to-meters scale
    wilor_kpts_3d: np.ndarray  # (21, 3) original WiLoR pose


@dataclass
class SmoothedHand:
    kpts_3d: np.ndarray        # (21, 3) final metric keypoints
    wrist_metric: np.ndarray   # (3,) smoothed wrist (for infilled interp)
    scale: float               # smoothed scale (for infilled interp)


def _reconstruct_metric_hand(wilor_kpts_3d: np.ndarray,
                              wrist_metric: np.ndarray,
                              scale: float) -> np.ndarray:
    relative = wilor_kpts_3d - wilor_kpts_3d[0]
    return relative * scale + wrist_metric


ANATOMICAL_WRIST_TO_MCP9 = 0.08  # meters — average adult wrist-to-MCP9


def _compute_anatomical_scale(wilor_kpts_3d: np.ndarray) -> float:
    """Fixed anatomical scale: map WiLoR's canonical hand to ~0.08m wrist-to-MCP9."""
    wilor_hand = float(np.linalg.norm(
        wilor_kpts_3d[9] - wilor_kpts_3d[0]))
    if wilor_hand > 1e-6:
        return ANATOMICAL_WRIST_TO_MCP9 / wilor_hand
    return 1.0


# ---------------------------------------------------------------------------
# Temporal smoothing (wrist position + scale, not per-joint Z)
# ---------------------------------------------------------------------------

def _smooth_hand_trajectory(
        frames: Dict[int, HandGrounding]) -> Dict[int, SmoothedHand]:
    """Median-filter (wrist_metric, scale) across frames, then reconstruct.

    Window scales with frame count (3-7, always odd).  Articulation is
    untouched — only the rigid placement is smoothed.
    """
    frame_idxs = sorted(frames.keys())
    n = len(frame_idxs)
    window = min(7, max(3, n // 3 * 2 + 1))
    if window % 2 == 0:
        window += 1

    wrists = np.stack([frames[fi].wrist_metric for fi in frame_idxs])
    scales = np.array([frames[fi].scale for fi in frame_idxs])

    if n < window:
        result = {}
        for i, fi in enumerate(frame_idxs):
            gf = frames[fi]
            result[fi] = SmoothedHand(
                kpts_3d=_reconstruct_metric_hand(
                    gf.wilor_kpts_3d, gf.wrist_metric, gf.scale),
                wrist_metric=gf.wrist_metric.copy(),
                scale=gf.scale)
        return result

    half = window // 2
    result = {}
    for i, fi in enumerate(frame_idxs):
        lo = max(0, i - half)
        hi = min(n, i + half + 1)
        w_smooth = np.median(wrists[lo:hi], axis=0)
        s_smooth = float(np.median(scales[lo:hi]))
        gf = frames[fi]
        result[fi] = SmoothedHand(
            kpts_3d=_reconstruct_metric_hand(
                gf.wilor_kpts_3d, w_smooth, s_smooth),
            wrist_metric=w_smooth,
            scale=s_smooth)
    return result


# ---------------------------------------------------------------------------
# Infilled frame interpolation (wrist + scale, not per-joint Z)
# ---------------------------------------------------------------------------

def _find_brackets(frame_idxs: List[int],
                   target: int) -> Tuple[int, int]:
    lo = frame_idxs[0]
    hi = frame_idxs[-1]
    for fi in frame_idxs:
        if fi <= target:
            lo = fi
        if fi >= target:
            hi = fi
            break
    return lo, hi


def _interpolate_infilled_hand(
        smoothed: Dict[int, SmoothedHand],
        grounded_idxs: List[int],
        target_fi: int,
        infilled_kpts_3d: np.ndarray) -> Optional[np.ndarray]:
    if not smoothed:
        return None

    if target_fi <= grounded_idxs[0]:
        ref = smoothed[grounded_idxs[0]]
        return _reconstruct_metric_hand(
            infilled_kpts_3d, ref.wrist_metric, ref.scale)
    if target_fi >= grounded_idxs[-1]:
        ref = smoothed[grounded_idxs[-1]]
        return _reconstruct_metric_hand(
            infilled_kpts_3d, ref.wrist_metric, ref.scale)

    lo_fi, hi_fi = _find_brackets(grounded_idxs, target_fi)
    if lo_fi == hi_fi:
        ref = smoothed[lo_fi]
        return _reconstruct_metric_hand(
            infilled_kpts_3d, ref.wrist_metric, ref.scale)

    alpha = (target_fi - lo_fi) / (hi_fi - lo_fi)
    lo_s = smoothed[lo_fi]
    hi_s = smoothed[hi_fi]
    wrist_interp = (1 - alpha) * lo_s.wrist_metric + alpha * hi_s.wrist_metric
    scale_interp = (1 - alpha) * lo_s.scale + alpha * hi_s.scale

    return _reconstruct_metric_hand(infilled_kpts_3d, wrist_interp, scale_interp)


# ---------------------------------------------------------------------------
# Frame decoding
# ---------------------------------------------------------------------------

def decode_keypoint_frames(mp4_path: Path,
                           frame_indices: List[int]) -> Dict[int, np.ndarray]:
    if not frame_indices:
        return {}
    sorted_indices = sorted(frame_indices)
    frames = {}
    cap = cv2.VideoCapture(str(mp4_path))
    if not cap.isOpened():
        return {}
    cap.set(cv2.CAP_PROP_POS_FRAMES, sorted_indices[0])
    pos = sorted_indices[0]
    for target in sorted_indices:
        while pos < target:
            cap.grab()
            pos += 1
        ret, frame = cap.read()
        if ret:
            frames[target] = frame
        pos += 1
    cap.release()
    return frames


# ---------------------------------------------------------------------------
# Per-clip grounding
# ---------------------------------------------------------------------------

MIN_VALID_JOINTS = 2


def _round_3d(pts: np.ndarray) -> list:
    return [[round(float(v), 5) for v in pt] for pt in pts]


def _try_ground_hand(hand: dict, depth_map: np.ndarray,
                     W: int, H: int, focal: float, cx: float, cy: float,
                     joint_set: list) -> Optional[HandGrounding]:
    """Sample depth, back-project wrist, apply fixed anatomical scale."""
    wilor_3d = np.array(hand["kpts_3d"])
    kpts_2d = np.array(hand["kpts_2d"])
    kpts_2d_px = kpts_2d * np.array([W, H])

    visible = ((kpts_2d_px[:, 0] >= 0) & (kpts_2d_px[:, 0] < W) &
               (kpts_2d_px[:, 1] >= 0) & (kpts_2d_px[:, 1] < H))

    candidates = [j for j in joint_set if visible[j]]
    if not candidates:
        return None

    patch_sz = _adaptive_patch_size(kpts_2d_px, visible)
    depths = sample_depth_at_joints(depth_map, kpts_2d_px, candidates, patch_sz)
    valid_depths = _reject_depth_outliers(candidates, depths)

    if len(valid_depths) < MIN_VALID_JOINTS:
        return None

    z_hand = float(np.median(list(valid_depths.values())))
    z_hand = max(z_hand, MIN_DEPTH)

    wrist_metric = np.array([
        (kpts_2d_px[0, 0] - cx) * z_hand / focal,
        (kpts_2d_px[0, 1] - cy) * z_hand / focal,
        z_hand])

    scale = _compute_anatomical_scale(wilor_3d)

    return HandGrounding(wrist_metric=wrist_metric, scale=scale,
                         wilor_kpts_3d=wilor_3d.copy())


def ground_clip(clip: dict, frames_dir: Path,
                estimator: MoGe2Estimator,
                geocalib_model=None,
                default_gravity: Optional[np.ndarray] = None
                ) -> Optional[Tuple[dict, dict]]:
    """Depth-ground a single clip's kpts_3d.  Returns (clip, timings) or None."""
    vn = clip["video_number"]
    uid = clip["node_uid"]
    mp4 = frames_dir / str(vn) / f"{uid}.mp4"
    if not mp4.exists():
        _log(f"  [drop] MP4 missing: {mp4}")
        return None

    kpf = clip["keypoints_per_frame"]
    all_frame_keys = sorted(kpf.keys(), key=int)
    all_frame_indices = [int(fk) for fk in all_frame_keys]

    if not all_frame_indices:
        return None

    t_decode = time.time()
    decoded = decode_keypoint_frames(mp4, all_frame_indices)
    t_decode = time.time() - t_decode
    if not decoded:
        _log(f"  [drop] no frames decoded from {mp4}")
        return None

    first_fi = min(decoded.keys())
    H, W = decoded[first_fi].shape[:2]
    cx, cy = W / 2.0, H / 2.0

    depth_maps: Dict[int, np.ndarray] = {}
    focals: List[float] = []
    gravity_per_frame: Dict[int, np.ndarray] = {}
    t_depth = time.time()
    for fi in sorted(decoded.keys()):
        result = estimator.estimate(decoded[fi])
        depth_maps[fi] = result.depth
        focals.append(result.focal_length_px)
    t_depth = time.time() - t_depth

    t_gravity = time.time()
    if default_gravity is not None:
        gravity = np.asarray(default_gravity, dtype=np.float32)
        for fi in sorted(decoded.keys()):
            gravity_per_frame[fi] = gravity
    elif geocalib_model is not None:
        for fi in sorted(decoded.keys()):
            rgb = cv2.cvtColor(decoded[fi], cv2.COLOR_BGR2RGB)
            img_t = (torch.tensor(rgb, dtype=torch.float32)
                     .permute(2, 0, 1).div_(255.0).unsqueeze(0).cuda())
            gc_out = geocalib_model.calibrate(img_t)
            gravity_per_frame[fi] = gc_out["gravity"].vec3d[0].cpu().numpy()
    t_gravity = time.time() - t_gravity

    del decoded
    clip_focal = float(np.median(focals))

    # Phase 1: Ground WiLoR-extracted hands — reliable joints first.
    t_ground = time.time()
    grounded_frames: Dict[str, Dict[int, HandGrounding]] = {"L": {}, "R": {}}
    all_wrist_z: List[float] = []

    for fk in all_frame_keys:
        fi = int(fk)
        if fi not in depth_maps:
            continue

        for hid in ("L", "R"):
            for hand in kpf[fk].get(hid, []):
                if hand.get("infilled"):
                    continue

                grounding = _try_ground_hand(
                    hand, depth_maps[fi], W, H, clip_focal, cx, cy,
                    RELIABLE_JOINT_IDS)
                if grounding is None:
                    continue

                grounded_frames[hid][fi] = grounding
                hand["depth_grounded"] = True
                all_wrist_z.append(float(grounding.wrist_metric[2]))

    clip_median_wrist_z = (float(np.median(all_wrist_z))
                           if all_wrist_z else None)

    # Phase 1b: Guarantee grounding for every non-infilled hand that
    # failed Phase 1.  Escalation order:
    #   (a) try ALL visible joints (not just reliable)
    #   (b) cross-hand Z from same frame
    #   (c) clip median Z
    for fk in all_frame_keys:
        fi = int(fk)
        if fi not in depth_maps:
            continue

        for hid in ("L", "R"):
            for hand in kpf[fk].get(hid, []):
                if hand.get("infilled") or hand.get("depth_grounded"):
                    continue

                wilor_3d = np.array(hand["kpts_3d"])
                kpts_2d_px = np.array(hand["kpts_2d"]) * np.array([W, H])

                # (a) Try ALL visible joints.
                grounding = _try_ground_hand(
                    hand, depth_maps[fi], W, H, clip_focal, cx, cy,
                    ALL_JOINT_IDS)
                if grounding is not None:
                    grounded_frames[hid][fi] = grounding
                    hand["depth_grounded"] = True
                    hand["depth_fallback"] = "all_joints"
                    continue

                # (b) Cross-hand: borrow Z from the other hand's wrist.
                other_hid = "R" if hid == "L" else "L"
                if fi in grounded_frames[other_hid]:
                    fallback_z = float(grounded_frames[other_hid][fi].wrist_metric[2])
                    fallback_tag = "cross_hand"
                # (c) Clip median Z.
                elif clip_median_wrist_z is not None:
                    fallback_z = clip_median_wrist_z
                    fallback_tag = "clip_median"
                else:
                    hand["depth_grounded"] = False
                    continue

                wrist_metric = np.array([
                    (kpts_2d_px[0, 0] - cx) * fallback_z / clip_focal,
                    (kpts_2d_px[0, 1] - cy) * fallback_z / clip_focal,
                    fallback_z])
                grounded_frames[hid][fi] = HandGrounding(
                    wrist_metric=wrist_metric,
                    scale=_compute_anatomical_scale(wilor_3d),
                    wilor_kpts_3d=wilor_3d.copy())
                hand["depth_grounded"] = True
                hand["depth_fallback"] = fallback_tag

    del depth_maps
    t_ground = time.time() - t_ground

    # Phase 2: Temporal smoothing (wrist position + scale, adaptive window).
    t_smooth = time.time()
    smoothed: Dict[str, Dict[int, SmoothedHand]] = {"L": {}, "R": {}}
    grounded_idxs: Dict[str, List[int]] = {"L": [], "R": []}
    for hid in ("L", "R"):
        if grounded_frames[hid]:
            smoothed[hid] = _smooth_hand_trajectory(grounded_frames[hid])
            grounded_idxs[hid] = sorted(smoothed[hid].keys())

    del grounded_frames
    t_smooth = time.time() - t_smooth

    # Phase 3: Write grounded results + interpolate infilled frames.
    for fk in all_frame_keys:
        fi = int(fk)
        hands = kpf[fk]
        for hid in ("L", "R"):
            for hand in hands.get(hid, []):
                if fi in smoothed[hid]:
                    hand["kpts_3d"] = _round_3d(smoothed[hid][fi].kpts_3d)

                elif hand.get("infilled") and smoothed[hid]:
                    infilled_3d = np.array(hand["kpts_3d"])

                    interp = _interpolate_infilled_hand(
                        smoothed[hid], grounded_idxs[hid], fi,
                        infilled_3d)
                    if interp is not None:
                        hand["kpts_3d"] = _round_3d(interp)
                        hand["depth_interpolated"] = True

    for fk in all_frame_keys:
        fi = int(fk)
        if fi in gravity_per_frame:
            kpf[fk]["gravity"] = [round(float(v), 5) for v in gravity_per_frame[fi]]

    clip["depth_focal"] = clip_focal
    clip["keypoints_per_frame"]["_meta"] = {
        "depth_focal": round(clip_focal, 2),
        "frame_width": W,
        "frame_height": H,
    }

    timings = {"decode": t_decode, "depth": t_depth, "gravity": t_gravity,
                "ground": t_ground, "smooth": t_smooth}
    _log(f"    {uid}: {len(all_frame_indices)}f — "
         f"decode {t_decode:.2f}s, depth {t_depth:.2f}s, gravity {t_gravity:.2f}s, "
         f"ground {t_ground:.2f}s, smooth {t_smooth:.2f}s")

    return clip, timings


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def _save_to_db(clips: List[dict], batch_label: str):
    """Write depth_grounded_keypoints for a batch of processed clips."""
    n = len(clips)
    if n == 0:
        return
    _log(f"  [{batch_label}] saving {n} clips to DB ...")
    t0 = time.time()
    for clip in clips:
        kpf_json = json.dumps(clip["keypoints_per_frame"])
        db.execute(
            "UPDATE selected_clips SET depth_grounded_keypoints = %s "
            "WHERE video_number = %s AND node_uid = %s",
            (kpf_json, int(clip["video_number"]), clip["node_uid"]))
    _log(f"  [{batch_label}] saved in {time.time()-t0:.1f}s")


def main():
    parser = argparse.ArgumentParser(
        description="Depth-ground keypoints via MoGe-2 (DB → DB)")
    parser.add_argument("-n", type=int, default=None,
                        help="Process only the first N video_numbers")
    parser.add_argument("--frames-dir", type=Path, default=DEFAULT_FRAMES_DIR,
                        help=f"Clip MP4 directory (default: {DEFAULT_FRAMES_DIR})")
    parser.add_argument("--save-every", type=int, default=50,
                        help="Commit to DB every N videos (default: 50)")
    parser.add_argument("--reprocess", action="store_true",
                        help="Re-process clips that already have depth_grounded_keypoints "
                             "(default: skip them)")
    parser.add_argument("--default-gravity", action="store_true",
                        help="Skip GeoCalib and write default camera-frame gravity "
                             f"{DEFAULT_GRAVITY.tolist()} for every decoded frame")
    args = parser.parse_args()

    _log(f"[main] args={vars(args)}")

    try:
        from moge.model.v2 import MoGeModel  # noqa: F401
    except ImportError:
        sys.exit("MoGe-2 not installed. Run:\n"
                 "  pip install git+https://github.com/microsoft/MoGe.git")

    # Load clips from DB — when -n is set, restrict the query to only the
    # first N video_numbers so we don't fetch 100k+ rows we won't use.
    _log("[main] querying clips from DB ...")
    t0 = time.time()

    skip_existing = not args.reprocess
    skip_cond = "AND depth_grounded_keypoints IS NULL " if skip_existing else ""

    if args.n is not None:
        VN_BATCH = 1000
        target_vns = []
        remaining = args.n
        last_vn = -1
        while remaining > 0:
            batch_size = min(remaining, VN_BATCH)
            vn_rows = db.query_clips(
                f"SELECT DISTINCT TOP {batch_size} video_number "
                f"FROM selected_clips "
                f"WHERE keypoints_per_frame IS NOT NULL "
                f"{skip_cond}"
                f"AND video_number > {last_vn} "
                f"ORDER BY video_number")
            if not vn_rows:
                break
            target_vns.extend(r["video_number"] for r in vn_rows)
            last_vn = target_vns[-1]
            remaining -= len(vn_rows)
            _log(f"  fetched {len(target_vns)} video_numbers so far "
                 f"({time.time()-t0:.1f}s)")
        if not target_vns:
            _log("[main] no clips to process")
            return
        video_numbers = sorted(target_vns)
    else:
        vn_rows = db.query_clips(
            f"SELECT DISTINCT video_number "
            f"FROM selected_clips "
            f"WHERE keypoints_per_frame IS NOT NULL "
            f"{skip_cond}"
            f"ORDER BY video_number")
        video_numbers = [r["video_number"] for r in vn_rows]

    if not video_numbers:
        _log("[main] no clips to process")
        return

    _log(f"[main] {len(video_numbers)} videos to process "
         f"(fetched video list in {time.time()-t0:.1f}s)")

    _log("[main] loading MoGe-2 ...")
    t0 = time.time()
    estimator = MoGe2Estimator()
    _log(f"[main] MoGe-2 ready in {time.time()-t0:.1f}s")

    default_gravity = None
    geocalib_model = None
    if args.default_gravity:
        default_gravity = DEFAULT_GRAVITY
        _log(f"[main] skipping GeoCalib; using default gravity "
             f"{DEFAULT_GRAVITY.tolist()}")
    else:
        from geocalib import GeoCalib
        _log("[main] loading GeoCalib ...")
        t0 = time.time()
        geocalib_model = GeoCalib(weights="pinhole").cuda().eval()
        _log(f"[main] GeoCalib ready in {time.time()-t0:.1f}s")

    pending_saves: List[dict] = []
    total_grounded = 0
    total_fallback = 0
    total_dropped = 0
    total_failed = 0
    total_interpolated = 0
    total_clips = 0
    cumulative_timings: Dict[str, float] = {
        "decode": 0.0, "depth": 0.0, "gravity": 0.0,
        "ground": 0.0, "smooth": 0.0}
    t_start = time.time()

    for vi, vn in enumerate(video_numbers, 1):
        vn_clips = db.query_clips(
            f"SELECT video_number, node_number, node_uid, keypoints_per_frame "
            f"FROM selected_clips "
            f"WHERE keypoints_per_frame IS NOT NULL "
            f"{skip_cond}"
            f"AND video_number = {vn} "
            f"ORDER BY node_number")
        total_clips += len(vn_clips)
        vn_grounded = 0

        for clip in vn_clips:
            out = ground_clip(clip, args.frames_dir, estimator,
                              geocalib_model, default_gravity)
            if out is None:
                total_dropped += 1
                continue

            result, timings = out
            pending_saves.append(result)
            for k in cumulative_timings:
                cumulative_timings[k] += timings.get(k, 0.0)

            for fk, frame in result["keypoints_per_frame"].items():
                if fk == "_meta":
                    continue
                for hid in ("L", "R"):
                    for h in frame.get(hid, []):
                        if h.get("depth_grounded") is True:
                            vn_grounded += 1
                            if h.get("depth_fallback"):
                                total_fallback += 1
                        elif h.get("depth_grounded") is False:
                            total_failed += 1
                        elif h.get("depth_interpolated"):
                            total_interpolated += 1

        total_grounded += vn_grounded
        elapsed = time.time() - t_start
        _log(f"[{vi}/{len(video_numbers)}] vn={vn}: "
             f"{len(vn_clips)} clips, {vn_grounded} hands grounded "
             f"(elapsed {elapsed:.0f}s)")

        if vi % args.save_every == 0 or vi == len(video_numbers):
            _save_to_db(pending_saves, "checkpoint")
            pending_saves.clear()

    _log(f"\n[main] done in {time.time()-t_start:.0f}s")
    _log(f"  grounded:     {total_grounded} hand-frames "
         f"({total_fallback} via fallback)")
    _log(f"  interpolated: {total_interpolated} hand-frames (infilled)")
    _log(f"  failed:       {total_failed} hand-frames")
    _log(f"  dropped:      {total_dropped} clips (MP4 missing / no frames)")
    _log(f"  cumulative timings:")
    for k, v in cumulative_timings.items():
        _log(f"    {k:>8s}: {v:.1f}s")


if __name__ == "__main__":
    main()
