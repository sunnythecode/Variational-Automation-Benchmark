"""End-to-end smoke test for the VAB unified-config pipeline.

For each seed task: load YAML, build env, step the policy a few times with a
zero action, assert observation is the strict {images, proprio} shape, and
dump one agentview RGB to /tmp/vab_smoke/ for visual sanity-check.
"""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pytest

try:
    from libero.vab import load_task                  # pip install -e .
except ImportError:
    from libero.libero.vab import load_task           # run-from-source (CWD=repo)

SEED_TASKS_DIR = Path(__file__).resolve().parents[1] / "seed_tasks"
SEED_TASK_FILES = sorted(SEED_TASKS_DIR.glob("*.yaml"))

DUMP_DIR = Path("/tmp/vab_smoke")
DUMP_DIR.mkdir(parents=True, exist_ok=True)


@pytest.mark.parametrize("task_path", SEED_TASK_FILES, ids=lambda p: p.stem)
def test_seed_task_runs(task_path):
    task = load_task(task_path)
    env = task.make_env()
    try:
        for init_index in range(task.n_inits):
            obs = env.reset(init_index=init_index)
            assert set(obs.keys()) == {"images", "proprio"}, obs.keys()
            assert "agentview" in obs["images"]
            assert obs["images"]["agentview"].shape == (
                task.camera_height,
                task.camera_width,
                3,
            )
            for _ in range(20):
                obs, reward, done, info = env.step(np.zeros(env.action_dim))
            assert "success" in info
            assert "language" in info and info["language"] == task.language
        rgb = obs["images"]["agentview"][::-1]
        from matplotlib.image import imsave

        imsave(DUMP_DIR / f"{task_path.stem}_agentview.png", rgb)
    finally:
        env.close()
