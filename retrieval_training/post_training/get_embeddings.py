"""Extract JEPA pooled and retrieval embeddings for all clips using a trained checkpoint."""

import argparse
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
TRAINING_DIR = SCRIPT_DIR.parent  # retrieval_training/
sys.path.insert(0, str(TRAINING_DIR))

import torch
from torch.utils.data import DataLoader

from config import Config
from data import VideoClipDataset, collate_fn, select_model_input
from model import TrajectoryRetrievalModel


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Path to .pt checkpoint (default: {run_dir}/best_model.pt)")
    parser.add_argument("-n", "--num-videos", type=int, default=-1,
                        help="Only encode clips from the first N videos (-1 = all)")
    # NB: kept as `--output-dir` for compatibility with the Slurm training
    # wrapper (which passes `--output-dir "$RUN_DIR"`). Internally this is the **run_dir** —
    # the per-run timestamped directory that holds the checkpoint, trajectory
    # artifact, and TB logs — not the YAML's `cfg.output_dir`, which is the
    # parent directory shared across runs and never holds these files directly.
    parser.add_argument("--output-dir", type=str, required=True,
                        help="Run directory (RUN_DIR). The per-run dir that "
                             "holds best_model.pt and trajectories_<design>.pt — "
                             "*not* the parent output_dir from the YAML.")
    parser.add_argument("--upsert-to-db", action="store_true",
                        help="Upsert embeddings to the database (default: skip)")
    args = parser.parse_args()

    cfg = Config.from_yaml(args.config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    run_dir = Path(args.output_dir)

    if args.checkpoint:
        ckpt_path = args.checkpoint
    elif (run_dir / "best_model.pt").exists():
        ckpt_path = str(run_dir / "best_model.pt")
    else:
        epoch_ckpts = sorted(run_dir.glob("checkpoint_epoch_*.pt"))
        if not epoch_ckpts:
            raise FileNotFoundError(f"No checkpoints found in {run_dir}")
        ckpt_path = str(epoch_ckpts[-1])
        print(f"best_model.pt not found, falling back to {epoch_ckpts[-1].name}")
    print(f"Loading checkpoint: {ckpt_path}")
    ckpt = torch.load(ckpt_path, weights_only=True, map_location=device)

    trajectory_mode = cfg.input_mode == "trajectory"
    if cfg.input_mode not in ("jepa", "trajectory"):
        raise ValueError(
            f"Unknown input_mode={cfg.input_mode!r}; expected 'jepa' or 'trajectory'"
        )

    # -- Trajectory artifact (mirrors train.py: reuse if present, else
    #    auto-build). In the slurm pipeline this script runs right after
    #    train.py, so the file is already in `run_dir`. For standalone
    #    invocations against an arbitrary checkpoint, build it here so the
    #    script "just works". Built before the model so we can derive the
    #    trajectory feature dim D for encoder_dim in trajectory mode.
    traj_path = run_dir / f"trajectories_{cfg.dtw_design}.pt"
    if not traj_path.exists():
        print(f"Building trajectories for design={cfg.dtw_design} -> {traj_path}")
        sys.path.insert(0, str(TRAINING_DIR / "build_dataset"))
        from build_trajectories import build_trajectories
        build_trajectories(
            data_dir=Path(cfg.data_dir),
            dtw_design=cfg.dtw_design,
            output_path=traj_path,
            require_jepa_features=cfg.input_mode == "jepa",
        )
    else:
        print(f"Reusing trajectories at {traj_path}")

    dataset = VideoClipDataset(
        cfg.data_dir,
        dtw_design=cfg.dtw_design,
        max_clip_frames=cfg.max_clip_frames,
        trajectories_path=str(traj_path),
        load_jepa_features=not trajectory_mode,
    )

    # In trajectory mode the head's input dim is the trajectory feature dim D
    # (must match the trained checkpoint), not the YAML's JEPA encoder_dim.
    effective_encoder_dim = (
        int(dataset.trajectories.shape[-1]) if trajectory_mode else cfg.encoder_dim
    )
    attention_dim = cfg.trajectory_model_dim if trajectory_mode else None
    # Construct with the same head-shape fields as train.py so the loaded
    # state_dict matches structurally. For a DTW-NN checkpoint this builds
    # the DTW layer + BN + MLP; for a cross-attn checkpoint it builds the
    # attention head as before.
    model = TrajectoryRetrievalModel(
        encoder_dim=effective_encoder_dim,
        retrieval_dim=cfg.retrieval_dim,
        num_heads=cfg.num_cross_attn_heads,
        num_projection_layers=cfg.num_projection_layers,
        num_cross_attn_layers=cfg.num_cross_attn_layers,
        use_positional_encoding=trajectory_mode and cfg.use_trajectory_positional_encoding,
        attention_dim=attention_dim,
        num_queries=cfg.num_queries,
        dtw_nn_layer=cfg.dtw_nn_layer,
        dtw_nn_num_nodes=cfg.dtw_nn_num_nodes,
        dtw_nn_prototype_length=cfg.dtw_nn_prototype_length,
        dtw_nn_gamma=cfg.dtw_nn_gamma,
    ).to(device)
    model.head.load_state_dict(ckpt["head_state_dict"])
    model.eval()

    if args.num_videos > 0:
        all_videos = sorted({dataset.clip_index[i]["video_number"] for i in dataset.indices})
        keep_videos = set(all_videos[:args.num_videos])
        dataset.indices = [i for i in dataset.indices
                           if dataset.clip_index[i]["video_number"] in keep_videos]
        print(f"Filtered to {len(dataset.indices)} clips from {len(keep_videos)} videos")

    dataloader = DataLoader(
        dataset,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        collate_fn=collate_fn,
        pin_memory=True,
        drop_last=False,
    )

    jepa_pooled = {}
    retrieval = {}
    videos_seen = set()
    last_video_count = 0
    total_steps = len(dataloader)
    print(f"Encoding {len(dataset)} clips across {total_steps} batches...")

    with torch.no_grad(), torch.amp.autocast("cuda", enabled=torch.cuda.is_available()):
        for step, batch in enumerate(dataloader, 1):
            feats, mask = select_model_input(batch, cfg.input_mode, device)
            clip_indices = batch["clip_indices"].numpy()

            out = model(feats, mask)
            z = out["z"].float().cpu()
            # In trajectory mode v_ref is a pooled trajectory, not a semantic
            # JEPA embedding — skip storing it.
            v_ref = None if trajectory_mode else out["v_ref"].float().cpu()

            for i, idx in enumerate(clip_indices):
                idx_int = int(idx)
                if v_ref is not None:
                    jepa_pooled[idx_int] = v_ref[i]
                retrieval[idx_int] = z[i]
                videos_seen.add(dataset.clip_index[idx_int]["video_number"])

            if len(videos_seen) >= last_video_count + 10 or step == total_steps:
                print(f"  step {step}/{total_steps} | {len(retrieval)} clips, {len(videos_seen)} videos encoded")
                last_video_count = len(videos_seen)

    print(f"Encoded {len(retrieval)} clips from {len(videos_seen)} videos")

    run_dir.mkdir(parents=True, exist_ok=True)
    torch.save(retrieval, run_dir / "ac_pooled_embeddings.pt")
    if trajectory_mode:
        print(f"Saved to {run_dir}/ac_pooled_embeddings.pt "
              "(trajectory mode: no jepa_pooled_embeddings.pt)")
    else:
        torch.save(jepa_pooled, run_dir / "jepa_pooled_embeddings.pt")
        print(f"Saved to {run_dir}/jepa_pooled_embeddings.pt and {run_dir}/ac_pooled_embeddings.pt")

    # --- Upsert to embeddings table ---
    if not args.upsert_to_db:
        print(f"--upsert-to-db not set; skipping DB upsert of {len(retrieval)} embeddings.")
        return

    # `db` is the private clip-database module, not distributed with this
    # repo (see README); --upsert-to-db only works against the private DB.
    from db import vec_to_sql, ensure_embedding_column, upsert_embeddings

    # Namespace the column by input mode so trajectory-mode vectors don't
    # clobber JEPA-mode ones for the same design.
    prefix = "retrieval_embedding_traj" if trajectory_mode else "retrieval_embedding"
    retrieval_col = f"{prefix}_{cfg.dtw_design}"
    print(f"Upserting {len(retrieval)} embeddings to DB (retrieval col: {retrieval_col})...")
    ensure_embedding_column(retrieval_col, cfg.retrieval_dim)

    upsert_rows = []
    for idx_int in sorted(retrieval.keys()):
        info = dataset.clip_index[idx_int]
        upsert_rows.append({
            "video_number": int(info["video_number"]),
            "node_uid": info["node_uid"],
            # NULL in trajectory mode — there is no JEPA pooled embedding.
            "jepa_pooled_embedding": (
                None if trajectory_mode else vec_to_sql(jepa_pooled[idx_int])
            ),
            "retrieval_embedding": vec_to_sql(retrieval[idx_int]),
        })
    upsert_embeddings(upsert_rows, retrieval_col)
    print(f"Upserted {len(upsert_rows)} rows to embeddings table")


if __name__ == "__main__":
    main()
