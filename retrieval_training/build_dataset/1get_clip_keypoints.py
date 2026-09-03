"""Dump selected_clips keypoints to a single .pt file for offline DTW/CKNNA work.

Paginates `selected_clips` (keypoints_per_frame NOT NULL), keeps only the fields
needed for trajectory experiments, and writes one combined file via torch.save.

**Incremental by default:** if the output file already exists, only fetches videos
not yet in the cache and merges them in. Use --full-rebuild to ignore the cache.

Output: outputs/training_data/clip_keypoints.pt — list[dict] sorted by (video_number, node_number):
    {
        "video_number":        int,
        "node_number":         int,
        "node_uid":            str,    # required to join with encoder_features/*.pt
        "keypoints_per_frame": dict,
        "left_hand_presence":  float,
        "right_hand_presence": float,
    }

Usage:
    python 1get_clip_keypoints.py                      # incremental update (or full if no cache)
    python 1get_clip_keypoints.py --full-rebuild       # ignore cache, fetch everything
    python 1get_clip_keypoints.py --retry-missing-cache-rows
                                                       # fetch DB rows absent from cache
    python 1get_clip_keypoints.py -n 50                # first 50 video_numbers
    python 1get_clip_keypoints.py --max-video 5000     # skip listing, iterate video_number=1..5000
    python 1get_clip_keypoints.py -o PATH              # override output path
    python 1get_clip_keypoints.py --check              # health-check the DB and exit
"""

import argparse
import copy
import os
import sys
import time
from pathlib import Path

import torch


def _log(msg):
    print(msg, flush=True)

# ---------------------------------------------------------------------------
# Env and DB client
# ---------------------------------------------------------------------------
# This pipeline step reads from the private clip database (`db` module, not
# distributed with this repo — see README). Later steps run from the
# exported .pt artifacts.
_REPO_ROOT = Path(
    os.environ.get("ABMR_PROJECT_ROOT", Path(__file__).resolve().parents[2])
).expanduser()

DEFAULT_OUT_PATH = _REPO_ROOT / "outputs" / "training_data" / "clip_keypoints.pt"
DEPTH_GROUNDED_OUT_PATH = _REPO_ROOT / "outputs" / "training_data" / "depth_grounded_clip_keypoints.pt"

SELECT_COLS = (
    "video_number,node_number,node_uid,keypoints_per_frame,"
    "left_hand_presence,right_hand_presence"
)

DEPTH_GROUNDED_SELECT_COLS = (
    "video_number,node_number,node_uid,"
    "depth_grounded_keypoints AS keypoints_per_frame,"
    "left_hand_presence,right_hand_presence"
)

from db import query as _db_query, query_clips as _db_query_clips


# ---------------------------------------------------------------------------
# Health check
# ---------------------------------------------------------------------------

def _health_check():
    _log("[health] SELECT TOP 1 video_number ...")
    t0 = time.time()
    try:
        r = _db_query("SELECT TOP 1 video_number FROM selected_clips")
        _log(f"[health]   OK in {time.time()-t0:.2f}s, got {len(r)} row(s): {r}")
        return True
    except Exception as e:
        _log(f"[health]   FAILED after {time.time()-t0:.1f}s: {type(e).__name__}: {e}")
        return False


# ---------------------------------------------------------------------------
# Cache helpers
# ---------------------------------------------------------------------------

def _load_existing_cache(path):
    """Load existing .pt cache. Returns (rows, cached_video_numbers_set)."""
    if not path.exists():
        return [], set()
    _log(f"[cache] loading existing {path}...")
    t0 = time.time()
    rows = torch.load(path, weights_only=False)
    cached_vns = {r["video_number"] for r in rows}
    _log(f"[cache] loaded {len(rows)} rows from {len(cached_vns)} videos "
         f"in {time.time()-t0:.1f}s")
    return rows, cached_vns


def _merge_rows(existing_rows, new_rows):
    """Merge existing and new rows, deduplicating by (video_number, node_number).

    New rows win on conflict (handles clips updated in the DB since last fetch).
    """
    by_key = {(r["video_number"], r["node_number"]): r for r in existing_rows}
    for r in new_rows:
        by_key[(r["video_number"], r["node_number"])] = r
    merged = sorted(by_key.values(), key=lambda r: (r["video_number"], r["node_number"]))
    return merged


def _row_key(row):
    return (row["video_number"], row["node_number"])


def _cache_keys(rows):
    return {_row_key(r) for r in rows}


def _missing_db_keys(db_key_rows, cached_keys):
    """Return DB row keys that are absent from the local cache."""
    missing = []
    seen = set()
    for row in db_key_rows:
        key = _row_key(row)
        if key in cached_keys or key in seen:
            continue
        missing.append(key)
        seen.add(key)
    return missing


def _group_keys_by_video(keys):
    grouped = {}
    for video_number, node_number in keys:
        grouped.setdefault(video_number, []).append(node_number)
    return {vn: sorted(node_numbers) for vn, node_numbers in sorted(grouped.items())}


def _load_body_pose_estimator(checkpoint_path=None):
    from body_pose_est import load_model

    _log("[body] loading body pose estimator ...")
    t0 = time.time()
    model, device = load_model(checkpoint_path)
    _log(f"[body] loaded on {device} in {time.time()-t0:.1f}s")
    return model, device


def _overwrite_kpts_with_body_frame(row, body_result):
    """Return a copy of row with kpts_3d replaced by body-frame keypoints."""
    out = copy.deepcopy(row)
    kpf = out.get("keypoints_per_frame")
    if not isinstance(kpf, dict):
        return out

    hands_body = body_result.get("hands_body", {})
    for frame_key, frame_hands in hands_body.items():
        frame = kpf.get(frame_key)
        if not isinstance(frame, dict):
            continue
        for hand_id in ("L", "R"):
            pts = frame_hands.get(hand_id) if isinstance(frame_hands, dict) else None
            if pts is None:
                continue
            entries = frame.get(hand_id)
            if not isinstance(entries, list) or not entries:
                continue
            for entry_idx, entry in enumerate(entries):
                if not isinstance(entry, dict) or "kpts_3d" not in entry:
                    continue
                if entry_idx == 0:
                    body_pts = pts
                else:
                    body_pts = _transform_extra_hand_entry_to_body(
                        entry["kpts_3d"], body_result)
                    if body_pts is None:
                        continue
                entry["kpts_3d"] = (
                    body_pts.tolist() if hasattr(body_pts, "tolist") else body_pts
                )
    return out


def _transform_extra_hand_entry_to_body(kpts_3d, body_result):
    """Transform non-primary hand detections with the clip-level body frame."""
    if "R_cb" not in body_result or "t_cb" not in body_result:
        return None

    import numpy as np
    from body_pose_est import (
        FLIP_HAND_Y_ABOUT_WRIST,
        FLIP_HAND_Z_ABOUT_WRIST,
        _flip_about_wrist,
    )

    pts_cam = np.asarray(kpts_3d, dtype=np.float64)
    pts_cam = _flip_about_wrist(
        pts_cam,
        flip_y=FLIP_HAND_Y_ABOUT_WRIST,
        flip_z=FLIP_HAND_Z_ABOUT_WRIST,
    )
    R_cb = np.asarray(body_result["R_cb"], dtype=np.float64)
    t_cb = np.asarray(body_result["t_cb"], dtype=np.float64)
    R_bc = R_cb.T
    t_bc = -R_bc @ t_cb
    return (R_bc @ pts_cam.T).T + t_bc


def _convert_rows_to_body_frame(rows, body_pose_model, body_pose_device,
                                estimator=None):
    """Run body pose estimation on rows, skipping clips that cannot be converted."""
    if estimator is None:
        from body_pose_est import estimate_body_frame as estimator

    converted = []
    skipped = 0
    t0 = time.time()
    for row in rows:
        ident = (f"vn={row.get('video_number')} "
                 f"node_number={row.get('node_number')} "
                 f"node={row.get('node_uid')}")
        try:
            body_result = estimator(row, body_pose_model, body_pose_device)
        except Exception as exc:
            skipped += 1
            _log(f"[body] skip {ident}: {type(exc).__name__}: {exc}")
            continue
        if body_result is None:
            skipped += 1
            _log(f"[body] skip {ident}: no body frame estimated")
            continue
        converted.append(_overwrite_kpts_with_body_frame(row, body_result))

    _log(f"[body] converted {len(converted)}/{len(rows)} rows "
         f"(skipped {skipped}) in {time.time()-t0:.1f}s")
    return converted


# ---------------------------------------------------------------------------
# Fetch
# ---------------------------------------------------------------------------

def _list_video_numbers():
    """Return the sorted list of distinct video_numbers in selected_clips."""
    _log("[list] fetching distinct video_numbers ...")
    t0 = time.time()
    rows = _db_query("SELECT DISTINCT video_number FROM selected_clips ORDER BY video_number")
    uniq = [r["video_number"] for r in rows]
    _log(f"[list] done in {time.time()-t0:.1f}s: {len(uniq)} distinct videos "
         f"(max={uniq[-1] if uniq else None})")
    return uniq


def _fetch_video_rows(video_number, select_cols=SELECT_COLS,
                      where_col="keypoints_per_frame", max_retries=5):
    """Fetch all keypoint rows for a single video, with retries on timeout."""
    for attempt in range(max_retries):
        t0 = time.time()
        try:
            data = _db_query_clips(
                f"SELECT {select_cols} FROM selected_clips "
                f"WHERE video_number = %s AND {where_col} IS NOT NULL "
                "ORDER BY node_number",
                (video_number,))
            return data, time.time() - t0
        except Exception as e:
            dt = time.time() - t0
            _log(f"[vid {video_number}]   attempt {attempt+1}/{max_retries} "
                 f"failed after {dt:.1f}s: {type(e).__name__}: {e}")
            if attempt == max_retries - 1:
                raise
            backoff = 2 ** attempt
            _log(f"[vid {video_number}]   sleeping {backoff}s before retry")
            time.sleep(backoff)


def _list_row_keys_for_videos(video_numbers, where_col="keypoints_per_frame"):
    """Return DB row keys for rows with available keypoints in the video range."""
    if not video_numbers:
        return []

    vn_set = set(video_numbers)
    lo, hi = min(vn_set), max(vn_set)
    _log(f"[retry] scanning DB row keys for video_number {lo}..{hi} "
         f"({len(vn_set)} selected videos) where {where_col} IS NOT NULL")
    t0 = time.time()
    rows = _db_query_clips(
        "SELECT video_number,node_number,node_uid FROM selected_clips "
        f"WHERE video_number >= %s AND video_number <= %s AND {where_col} IS NOT NULL "
        "ORDER BY video_number,node_number",
        (lo, hi))
    rows = [r for r in rows if r["video_number"] in vn_set]
    _log(f"[retry] DB key scan found {len(rows)} rows in {time.time()-t0:.1f}s")
    return rows


def _fetch_video_rows_by_node_numbers(video_number, node_numbers,
                                      select_cols=SELECT_COLS,
                                      where_col="keypoints_per_frame",
                                      max_retries=5,
                                      chunk_size=200):
    """Fetch specific node_number rows for one video, with retries."""
    if not node_numbers:
        return [], 0.0

    out = []
    total_dt = 0.0
    for start in range(0, len(node_numbers), chunk_size):
        chunk = list(node_numbers[start:start + chunk_size])
        placeholders = ",".join(["%s"] * len(chunk))
        params = (video_number, *chunk)
        for attempt in range(max_retries):
            t0 = time.time()
            try:
                data = _db_query_clips(
                    f"SELECT {select_cols} FROM selected_clips "
                    f"WHERE video_number = %s AND {where_col} IS NOT NULL "
                    f"AND node_number IN ({placeholders}) "
                    "ORDER BY node_number",
                    params)
                total_dt += time.time() - t0
                out.extend(data)
                break
            except Exception as e:
                dt = time.time() - t0
                _log(f"[vid {video_number}]   missing-row attempt "
                     f"{attempt+1}/{max_retries} failed after {dt:.1f}s: "
                     f"{type(e).__name__}: {e}")
                if attempt == max_retries - 1:
                    raise
                backoff = 2 ** attempt
                _log(f"[vid {video_number}]   sleeping {backoff}s before retry")
                time.sleep(backoff)
    return out, total_dt


def _fetch_missing_rows_for_keys(missing_keys, out_path=None, save_every=500,
                                 existing_rows=None, select_cols=SELECT_COLS,
                                 where_col="keypoints_per_frame",
                                 body_pose_model=None, body_pose_device=None):
    """Fetch only specific missing cache row keys grouped by video."""
    grouped = _group_keys_by_video(missing_keys)
    rows = []
    total = len(grouped)
    t_start = time.time()
    for i, (vn, node_numbers) in enumerate(grouped.items(), 1):
        data, dt = _fetch_video_rows_by_node_numbers(
            vn, node_numbers, select_cols=select_cols, where_col=where_col)
        if body_pose_model is not None:
            data = _convert_rows_to_body_frame(data, body_pose_model, body_pose_device)
        rows.extend(data)
        _log(f"[retry] {i}/{total} vn={vn}: requested {len(node_numbers)} "
             f"missing rows, fetched +{len(data)} rows in {dt:.1f}s "
             f"(total {len(rows)} rows, elapsed {time.time()-t_start:.0f}s)")
        if out_path is not None and (i % save_every == 0 or i == total):
            to_write = _merge_rows(existing_rows, rows) if existing_rows else rows
            _log(f"[retry]   checkpoint: saving {len(to_write)} rows to {out_path}")
            out_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(to_write, out_path)
    return rows


def _fetch_rows_for_videos(video_numbers, out_path=None, save_every=500,
                           existing_rows=None, select_cols=SELECT_COLS,
                           where_col="keypoints_per_frame",
                           body_pose_model=None, body_pose_device=None):
    """Fetch full rows for each video_number one at a time.

    If `out_path` is provided, incrementally writes progress after every
    `save_every` videos so a crash late in the run does not lose everything.
    When `existing_rows` is supplied, checkpoints merge new rows with the
    existing cache before writing so the output file remains complete.
    Videos with no rows are silently skipped.
    """
    rows = []
    total = len(video_numbers)
    t_start = time.time()
    printed_tree = False
    for i, vn in enumerate(video_numbers, 1):
        data, dt = _fetch_video_rows(vn, select_cols=select_cols, where_col=where_col)
        if not printed_tree and data:
            kpf = data[0]["keypoints_per_frame"]
            _log(f"[keypoints tree] vn={vn}, node={data[0].get('node_uid','?')}")
            frame_keys = [k for k in kpf.keys() if k.isdigit()]
            meta_keys = [k for k in kpf.keys() if not k.isdigit()]
            if meta_keys:
                _log(f"  non-frame keys: {meta_keys}")
                for mk in meta_keys:
                    val = kpf[mk]
                    if isinstance(val, dict):
                        _log(f"    {mk}: {{keys={list(val.keys())}}}")
                    else:
                        _log(f"    {mk}: {val}")
            for frame_key in sorted(frame_keys, key=int)[:3]:
                frame = kpf[frame_key]
                _log(f"  frame {frame_key}: hands={list(frame.keys())}")
                for hand, entries in frame.items():
                    if isinstance(entries, list):
                        if entries and isinstance(entries[0], dict):
                            _log(f"    {hand}: {len(entries)} entry(ies)")
                            _log(f"      keys={list(entries[0].keys())}")
                        else:
                            _log(f"    {hand}: {entries}")
                    else:
                        _log(f"    {hand}: {type(entries).__name__} = {entries}")
            _log(f"  ... ({len(frame_keys)} frames + {len(meta_keys)} meta keys)")
            printed_tree = True
        if body_pose_model is not None:
            data = _convert_rows_to_body_frame(data, body_pose_model, body_pose_device)
        rows.extend(data)
        _log(f"[fetch] {i}/{total} vn={vn}: +{len(data)} rows in {dt:.1f}s "
             f"(total {len(rows)} rows, elapsed {time.time()-t_start:.0f}s)")
        if out_path is not None and (i % save_every == 0 or i == total):
            to_write = _merge_rows(existing_rows, rows) if existing_rows else rows
            _log(f"[fetch]   checkpoint: saving {len(to_write)} rows to {out_path}")
            out_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(to_write, out_path)
    return rows


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Dump selected_clips keypoints to a .pt file")
    parser.add_argument("-n", type=int, default=None,
                        help="Only dump clips with video_number <= N")
    parser.add_argument("-o", "--out", type=Path, default=DEFAULT_OUT_PATH,
                        help=f"Output .pt path (default: {DEFAULT_OUT_PATH})")
    parser.add_argument("--max-video", type=int, default=None,
                        help="Skip listing; iterate video_number=from..max-video instead. "
                             "Use when the whole-table listing query is timing out.")
    parser.add_argument("--from-video", type=int, default=1,
                        help="Starting video_number when using --max-video (default: 1)")
    parser.add_argument("--check", action="store_true",
                        help="Run a minimal health check and exit")
    parser.add_argument("--save-every", type=int, default=500,
                        help="Checkpoint to disk every N videos (default: 500)")
    parser.add_argument("--full-rebuild", action="store_true",
                        help="Ignore existing cache and rebuild from scratch")
    parser.add_argument("--retry-missing-cache-rows", action="store_true",
                        help="Fetch DB rows with keypoints that are absent from "
                             "the existing local cache, even if their video is cached")
    parser.add_argument("--depth-grounded", action="store_true",
                        help="Pull from depth_grounded_keypoints column; "
                             "output to depth_grounded_clip_keypoints.pt")
    parser.add_argument("--body-pose-ckpt", type=Path, default=None,
                        help="Optional body pose estimator checkpoint for "
                             "--depth-grounded body-frame conversion")
    args = parser.parse_args()

    if args.depth_grounded:
        if args.out == DEFAULT_OUT_PATH:
            args.out = DEPTH_GROUNDED_OUT_PATH
        args._select_cols = DEPTH_GROUNDED_SELECT_COLS
        args._where_col = "depth_grounded_keypoints"
    else:
        args._select_cols = SELECT_COLS
        args._where_col = "keypoints_per_frame"

    _log(f"[main] python={sys.version.split()[0]}, args={vars(args)}")

    if not _health_check():
        _log("[main] health check failed — DB appears to be down.")
        sys.exit(1)

    if args.check:
        _log("[main] --check passed, exiting.")
        return

    # --- Determine which videos to cover ---
    if args.n is not None:
        all_video_numbers = list(range(args.from_video, args.n + 1))
        _log(f"[main] iterating video_number "
             f"{args.from_video}..{args.n} ({len(all_video_numbers)} videos)")
    elif args.max_video is not None:
        all_video_numbers = list(range(args.from_video, args.max_video + 1))
        _log(f"[main] iterating video_number "
             f"{args.from_video}..{args.max_video} ({len(all_video_numbers)} videos)")
    else:
        _log("[main] listing video_numbers...")
        all_video_numbers = _list_video_numbers()
        _log(f"[main] {len(all_video_numbers)} videos to cover")

    # --- Incremental: load cache and fetch only new videos or missing rows ---
    existing_rows = []
    missing_keys = None
    if not args.full_rebuild:
        existing_rows, cached_vns = _load_existing_cache(args.out)
        if existing_rows:
            if args.depth_grounded:
                _log("[main] WARNING: existing depth-grounded cache rows are assumed "
                     "to already be body-frame converted. Use --full-rebuild when "
                     "migrating an older cache.")
            if args.retry_missing_cache_rows:
                db_key_rows = _list_row_keys_for_videos(
                    all_video_numbers, where_col=args._where_col)
                missing_keys = _missing_db_keys(db_key_rows, _cache_keys(existing_rows))
                if not missing_keys:
                    _log(f"[main] cache has all DB rows for requested range "
                         f"({len(existing_rows)} cached rows). Nothing to retry.")
                    return
                retry_videos = len({vn for vn, _ in missing_keys})
                _log(f"[main] retrying {len(missing_keys)} missing cache rows "
                     f"across {retry_videos} videos")
            else:
                new_vns = sorted(set(all_video_numbers) - cached_vns)
                if not new_vns:
                    _log(f"[main] cache is up to date ({len(cached_vns)} videos, "
                         f"{len(existing_rows)} rows). Nothing to fetch.")
                    return
                _log(f"[main] {len(cached_vns)} videos cached, "
                     f"{len(new_vns)} new videos to fetch")
                all_video_numbers = new_vns
        elif args.retry_missing_cache_rows:
            _log("[main] --retry-missing-cache-rows requested but no existing "
                 "cache was found; falling back to normal selected-range fetch")

    body_pose_model = None
    body_pose_device = None
    if args.depth_grounded:
        body_pose_model, body_pose_device = _load_body_pose_estimator(args.body_pose_ckpt)

    # --- Fetch ---
    if missing_keys is not None:
        new_rows = _fetch_missing_rows_for_keys(
            missing_keys,
            out_path=args.out,
            save_every=args.save_every,
            existing_rows=existing_rows,
            select_cols=args._select_cols,
            where_col=args._where_col,
            body_pose_model=body_pose_model,
            body_pose_device=body_pose_device)
    else:
        new_rows = _fetch_rows_for_videos(all_video_numbers,
                                          out_path=args.out,
                                          save_every=args.save_every,
                                          existing_rows=existing_rows,
                                          select_cols=args._select_cols,
                                          where_col=args._where_col,
                                          body_pose_model=body_pose_model,
                                          body_pose_device=body_pose_device)

    # --- Merge (or just sort if full rebuild) ---
    if existing_rows:
        _log(f"[main] merging {len(existing_rows)} existing + {len(new_rows)} new rows...")
        rows = _merge_rows(existing_rows, new_rows)
    else:
        _log(f"[main] sorting {len(new_rows)} rows...")
        rows = sorted(new_rows, key=lambda r: (r["video_number"], r["node_number"]))

    _log(f"[main] writing final {args.out}...")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(rows, args.out)

    num_videos = len({r["video_number"] for r in rows})
    size_mb = args.out.stat().st_size / (1024 * 1024)
    _log(f"\n[main] done: wrote {len(rows)} rows from {num_videos} videos "
         f"to {args.out} ({size_mb:.1f} MB)")


if __name__ == "__main__":
    main()
