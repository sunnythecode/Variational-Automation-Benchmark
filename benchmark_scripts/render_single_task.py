"""Render a single VAB task at a given init index and save the agentview RGB.

Replacement for the deleted upstream script. The old version loaded a
benchmark by name, fetched (bddl_file, init_states) for a task_id, and
also opened an HDF5 demo file to render the final demo state alongside.
With the new self-contained YAML format there is no separate init or
demo file -- everything lives in the task YAML.

Usage:
    python benchmark_scripts/render_single_task.py \\
        --task tasks/libero_object_target_pos_var20x20/pick_up_the_cream_cheese_and_place_it_in_the_basket.yaml \\
        --init-index 0
    # writes benchmark_tasks/<task_id>__init0.png
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

# Prefer VAB's libero over any LIBERO-PosVar editable install also on sys.path.
_REPO = Path(__file__).resolve().parents[1]
sys.path = [p for p in sys.path if "LIBERO-PosVar" not in p]
sys.path.insert(0, str(_REPO / "libero"))

from libero.vab import load_task  # noqa: E402


def render_task(task_path: Path, init_index: int, n_settle_steps: int, out_dir: Path) -> Path:
    task = load_task(task_path)
    env = task.make_env()
    try:
        obs = env.reset(init_index=init_index)
        # let physics settle for a few zero-action steps
        for _ in range(n_settle_steps):
            obs, _, _, _ = env.step(np.zeros(env.action_dim, dtype=np.float32))
        # Pick the first camera (typically agentview).
        first_cam = task.cameras[0]
        rgb = obs["images"][first_cam][::-1]   # MuJoCo Y-flip
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / f"{task.id}__init{init_index}.png"
        cv2.imwrite(str(out_path), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
        return out_path
    finally:
        env.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", required=True, type=Path,
                        help="Path to a task YAML under tasks/")
    parser.add_argument("--init-index", type=int, default=0)
    parser.add_argument("--settle-steps", type=int, default=5)
    parser.add_argument("--out-dir", type=Path,
                        default=_REPO / "benchmark_tasks")
    args = parser.parse_args()
    out = render_task(args.task, args.init_index, args.settle_steps, args.out_dir)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
