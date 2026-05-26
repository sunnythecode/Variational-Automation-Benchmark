"""Record an 800x600 annotated MP4 covering EVERY task YAML under tasks/,
2 init variants per task. Banner shows suite + task + init + success.
"""
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[1]
VAB_LIBERO = str(_REPO / "libero")
# If a sibling LIBERO-PosVar install is on sys.path via .pth, prefer VAB's libero.
sys.path = [p for p in sys.path if "LIBERO-PosVar" not in p]
sys.path.insert(0, VAB_LIBERO)

import cv2  # noqa: E402
import numpy as np  # noqa: E402

from libero.vab import load_task  # noqa: E402

REPO = _REPO
TASKS_DIR = REPO / "tasks"
OUT_PATH = REPO / "videos" / "vab_all_tasks.mp4"
OUT_PATH.parent.mkdir(parents=True, exist_ok=True)

# Resolution: render straight at 800x600 (no upscale).
RENDER_H, RENDER_W = 600, 800
BANNER_H = 40
N_INITS_PER_TASK = 2
LOAD_FRAMES = 4          # static "loaded scene" beat
STEP_FRAMES = 18         # descend / lift loop
FPS = 15

YAMLS = sorted(TASKS_DIR.rglob("*.yaml"))
print(f"[plan] {len(YAMLS)} tasks, {N_INITS_PER_TASK} inits/task, "
      f"{LOAD_FRAMES + STEP_FRAMES} frames/clip @ {RENDER_W}x{RENDER_H} {FPS}fps")


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


def action_for_step(step: int, total: int, action_dim: int) -> np.ndarray:
    a = np.zeros(action_dim, dtype=np.float32)
    if action_dim >= 3:
        a[2] = -0.10 if step < total // 2 else 0.10
    if action_dim >= 10:                        # bimanual: mirror to robot1 z
        a[2 + 7] = -0.10 if step < total // 2 else 0.10
    return a


fourcc = cv2.VideoWriter_fourcc(*"mp4v")
writer = cv2.VideoWriter(str(OUT_PATH), fourcc, FPS, (RENDER_W, RENDER_H + BANNER_H))

total_frames = 0
for yaml_path in YAMLS:
    task = load_task(yaml_path)
    # Override camera resolution to render at 800x600 directly.
    task.camera_height = RENDER_H
    task.camera_width = RENDER_W
    suite = task.metadata.get("suite", yaml_path.parent.name)
    short_name = yaml_path.stem
    env = task.make_env()
    print(f"[task] {suite}/{short_name}  action_dim={env.action_dim}  n_inits={task.n_inits}")
    try:
        n = min(N_INITS_PER_TASK, task.n_inits)
        for init_idx in range(n):
            obs = env.reset(init_index=init_idx)
            for _ in range(LOAD_FRAMES):
                img = first_image(obs)[::-1]
                frame = annotate(img, [
                    f"suite: {suite}",
                    f"task: {short_name}    init {init_idx+1}/{n}    success: pending",
                ])
                writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
                total_frames += 1
            for step in range(STEP_FRAMES):
                obs, reward, done, info = env.step(
                    action_for_step(step, STEP_FRAMES, env.action_dim)
                )
                img = first_image(obs)[::-1]
                frame = annotate(img, [
                    f"suite: {suite}",
                    f"task: {short_name}    init {init_idx+1}/{n}    success: {info['success']}",
                ])
                writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
                total_frames += 1
    finally:
        env.close()

writer.release()
print(f"[done] {OUT_PATH}  ({total_frames} frames, {total_frames/FPS:.1f}s @ {FPS}fps)")
