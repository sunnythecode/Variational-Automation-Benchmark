"""Render an 800x600 video covering EVERY task YAML under tasks/ and
EVERY init variant per task. Each (task, init) clip mirrors
render_single_task: reset, settle for a few zero-action steps, capture
the first camera's RGB, hold for HOLD_FRAMES frames.

Output: videos/vab_all_tasks_all_seeds.mp4

Scope across the 6 committed suites:
    crate_washing                 1 task   x 5 inits
    object_all_variance          10 tasks x 50 inits
    object_target_basket_swap_variance  10 x 50
    object_target_permutation_variance  10 x 50
    object_target_pos_var20x20   10 tasks x 50 inits
    popcorn_production            1 task   x 50 inits
    ----------------------------------------------------
    42 tasks, 2055 (task, init) clips.
"""
from __future__ import annotations

import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[1]
VAB_LIBERO = str(_REPO / "libero")
sys.path = [p for p in sys.path if "LIBERO-PosVar" not in p]
sys.path.insert(0, VAB_LIBERO)

import cv2  # noqa: E402
import numpy as np  # noqa: E402

from libero.vab import load_task  # noqa: E402

TASKS_DIR = _REPO / "tasks"
OUT_PATH = _REPO / "videos" / "vab_all_tasks_all_seeds.mp4"
OUT_PATH.parent.mkdir(parents=True, exist_ok=True)

# Same canvas as the previous all-tasks video so frames stay 800x640.
RENDER_H, RENDER_W = 600, 800
BANNER_H = 40
SETTLE_STEPS = 3        # zero-action settling (matches render_single_task)
HOLD_FRAMES = 2         # frames to hold each init in the timeline
FPS = 15

YAMLS = sorted(TASKS_DIR.rglob("*.yaml"))
total_inits = 0
for yp in YAMLS:
    t = load_task(yp)
    total_inits += t.n_inits

print(f"[plan] {len(YAMLS)} tasks, {total_inits} total inits, "
      f"{HOLD_FRAMES} frames/init @ {RENDER_W}x{RENDER_H} {FPS}fps "
      f"=> ~{total_inits * HOLD_FRAMES / FPS:.0f} sec")


def annotate(rgb: np.ndarray, lines: list[str]) -> np.ndarray:
    h, w = rgb.shape[:2]
    banner = np.zeros((BANNER_H, w, 3), dtype=np.uint8)
    canvas = np.concatenate([rgb, banner], axis=0)
    for i, line in enumerate(lines):
        y = h + 16 + i * 16
        cv2.putText(canvas, line, (10, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.50,
                    (255, 255, 255), 1, cv2.LINE_AA)
    return canvas


def first_image(obs):
    return next(iter(obs["images"].values()))


fourcc = cv2.VideoWriter_fourcc(*"mp4v")
writer = cv2.VideoWriter(str(OUT_PATH), fourcc, FPS, (RENDER_W, RENDER_H + BANNER_H))

frames_written = 0
for yaml_path in YAMLS:
    task = load_task(yaml_path)
    task.camera_height = RENDER_H
    task.camera_width = RENDER_W
    suite = task.metadata.get("suite", yaml_path.parent.name)
    short_name = yaml_path.stem
    env = task.make_env()
    print(f"[task] {suite}/{short_name}  n_inits={task.n_inits}  action_dim={env.action_dim}")
    try:
        zero = np.zeros(env.action_dim, dtype=np.float32)
        for init_idx in range(task.n_inits):
            obs = env.reset(init_index=init_idx)
            for _ in range(SETTLE_STEPS):
                obs, _, _, _ = env.step(zero)
            img = first_image(obs)[::-1]
            frame = annotate(img, [
                f"suite: {suite}",
                f"task: {short_name}    init {init_idx+1}/{task.n_inits}",
            ])
            bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            for _ in range(HOLD_FRAMES):
                writer.write(bgr)
                frames_written += 1
    finally:
        env.close()

writer.release()
print(f"[done] {OUT_PATH}  ({frames_written} frames, {frames_written/FPS:.1f}s @ {FPS}fps)")
