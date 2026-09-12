# RoboTok

**An Internet-Scale Data Engine for Human Demonstration Video Retrieval and
Dexterous Manipulation Learning**

Howard Qian¹, Yiting Chen¹, Yunfei Xie¹, Kejia Ren¹, Podshara Chanrungmaneekul¹,
Gaotian Wang¹, Bowen Wen², Chen Wei¹, Kaiyu Hang¹

¹ Rice University  ·  ² NVIDIA

Trains a retrieval model that embeds web video clips by 3D hand-motion
similarity, from torso-relative two-hand trajectories.

![Retrieval embedding space and example clips](assets/qualitative_activities.png)

## Install

```bash
pip install -r requirements.txt
```

Use `faiss-gpu` instead of `faiss-cpu` on CUDA machines. The DTW kernels are
numba CUDA kernels and need a GPU.

## Usage

```bash
cd retrieval_training
python train.py --config configs/default.yaml   # trains; auto-builds artifacts
python -m pytest tests/ -q                      # tests
```

Config paths are relative to the repo root; `ABMR_PROJECT_ROOT` overrides it.
The held-out split is a seeded random **clip-level** split, not per-video — see
`config.py` for the same-video caveat.

## Data

The 3D torso-relative hand trajectories that training, eval and tests run
from are in the accompanying Hugging Face repo
[Rice-RobotPI-Lab/robotok-public](https://huggingface.co/Rice-RobotPI-Lab/robotok-public)
(`eval_data/torso_relative_clip_keypoints.pt`). Place the file at

- `outputs/training_data/depth_grounded_clip_keypoints.pt` (default, or
  `clip_keypoints.pt` with `use_depth_grounded_keypoints: False`).

`train.py` builds the per-design trajectory and DTW-neighbor artifacts from it
on first run.

## Citation

```bibtex
@article{qian2026robotok,
  title     = {RoboTok: An Internet-Scale Data Engine for Human Demonstration
               Video Retrieval and Dexterous Manipulation Learning},
  author    = {Qian, Howard and Chen, Yiting and Xie, Yunfei and
               Ren, Kejia and Chanrungmaneekul, Podshara and Wang, Gaotian and
               Wen, Bowen and Wei, Chen and Hang, Kaiyu},
  journal   = {arXiv preprint arXiv:2609.03199},
  year      = {2026}
}
```

## License

MIT ([LICENSE](LICENSE)).
