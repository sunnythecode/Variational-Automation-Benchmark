#!/usr/bin/env python3
"""
Benchmark runner: pick-and-place on libero_object_target_pos_var20x20.

Supports swappable motion planners via --planner:
  --planner curobo   (default) uses libero/libero/vab/planning.py
  --planner alt                uses libero/libero/vab/planning_alt.py

Parallelise across N GPUs by launching one worker per GPU:
  CUDA_VISIBLE_DEVICES=0 python run_b1_benchmark.py --task-slice 0:2  > videos/b1/run0.log &
  CUDA_VISIBLE_DEVICES=1 python run_b1_benchmark.py --task-slice 2:4  > videos/b1/run1.log &
  ...

Or use the provided run_b1_parallel.sh to launch all workers automatically.

Each worker writes its own shard CSV (timings_gpu<N>.csv).
A final merge step (run_b1_merge.py) combines shards → timings.csv + summary.json.
"""
from __future__ import annotations

import argparse
import csv
import json
import logging
import os
import sys
import time
from pathlib import Path

import imageio
import numpy as np

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from libero.vab import load_task
from libero.vab.env import VABEnv

# Planner module is selected at runtime via --planner; resolved in main()
# and injected into run_episode via a module-level reference.
_planning = None  # set in main() before any episode runs

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
)
_log = logging.getLogger(__name__)

TASK_DIR   = REPO / "tasks" / "libero_object_target_pos_var20x20"
OUT_DIR    = REPO / "videos" / "b1"
ROBOT_FILE = "franka.yml"
VIDEO_FPS  = 20
VIDEO_CAM  = "agentview"

CSV_FIELDS = [
    "task", "init_index",
    "success", "completion_rate",
    "total_plan_ms", "total_exec_ms", "total_steps", "wall_ms",
    "seg0_plan_ms", "seg0_exec_ms", "seg0_ok",
    "seg1_plan_ms", "seg1_exec_ms", "seg1_ok",
    "seg2_plan_ms", "seg2_exec_ms", "seg2_ok",
    "seg3_plan_ms", "seg3_exec_ms", "seg3_ok",
    "seg4_plan_ms", "seg4_exec_ms", "seg4_ok",
    "seg5_plan_ms", "seg5_exec_ms", "seg5_ok",
    "seg6_plan_ms", "seg6_exec_ms", "seg6_ok",
    "fail_reason",
]


def _make_env(task_obj) -> VABEnv:
    return VABEnv(
        task=task_obj,
        has_renderer=False,
        has_offscreen_renderer=True,
        control_freq=VIDEO_FPS,
        render_gpu_device_id=-1,
        ignore_done=True,
        hard_reset=False,
        controller="JOINT_POSITION",
    )


def run_episode(env: VABEnv, task_obj, init_index: int) -> dict:
    wall_t0 = time.monotonic()
    obs = env.reset(init_index=init_index)

    frames     = []
    segments   = []
    total_steps = 0
    fail_reason = ""
    info_ref: dict = {}

    rbase         = _planning._robot_base_world(env)
    hover_z_r     = _planning.HOVER_Z_FINGERTIP    + _planning.FRANKA_HAND_TO_FINGERTIP_Z_M - rbase[2]
    transport_z_r = _planning.TRANSPORT_Z_FINGERTIP + _planning.FRANKA_HAND_TO_FINGERTIP_Z_M - rbase[2]

    obj_id       = task_obj.success.args["obj"]
    container_id = task_obj.success.args["container"]

    # Capture initial frame (flipped: robosuite uses OpenGL Y-up → looks upside-down)
    frames.append(obs["images"][VIDEO_CAM][::-1].copy())

    def step_capture(action):
        nonlocal obs, info_ref
        obs, _, _, info_ref = env.step(action)
        frames.append(obs["images"][VIDEO_CAM][::-1].copy())

    def exec_traj(traj_Tx7, g_cmd, tol=0.01, max_steps_per_wp=20, subsample=2):
        indices = list(range(0, len(traj_Tx7), subsample))
        if (len(traj_Tx7) - 1) not in indices:
            indices.append(len(traj_Tx7) - 1)
        steps = 0
        for i in indices:
            target_q = traj_Tx7[i]
            for _ in range(max_steps_per_wp):
                err = target_q - obs["proprio"]["joint_pos"]
                if np.max(np.abs(err)) < tol:
                    break
                step_capture(np.append(np.clip(err / 0.05, -1.0, 1.0), g_cmd))
                steps += 1
        return steps

    def grip(g_cmd, steps=20):
        for _ in range(steps):
            step_capture(np.append(np.zeros(7), g_cmd))
        return steps

    def plan_exec(label, axes, target_xyz, g_cmd, seg_idx):
        nonlocal total_steps, fail_reason
        t0 = time.monotonic()
        ok, traj, reason = _planning.plan_directed_linear(
            start_config     = obs["proprio"]["joint_pos"],
            target_pose      = (target_xyz, np.array([0.0, 1.0, 0.0, 0.0])),
            endpoint_mode    = "PROJECT_TO_TARGET",
            allowed_axes     = axes,
            orientation_mode = "LOCK",
            robot_file       = ROBOT_FILE,
        )
        plan_ms = (time.monotonic() - t0) * 1000

        if not ok:
            segments.append({"label": label, "seg": seg_idx,
                              "plan_ms": plan_ms, "exec_ms": 0.0,
                              "steps": 0, "ok": False})
            fail_reason = f"seg{seg_idx}({label}): {reason}"
            return False

        t0 = time.monotonic()
        steps = exec_traj(traj, g_cmd)
        exec_ms = (time.monotonic() - t0) * 1000
        total_steps += steps
        segments.append({"label": label, "seg": seg_idx,
                          "plan_ms": plan_ms, "exec_ms": exec_ms,
                          "steps": steps, "ok": True})
        return True

    # ── Segment sequence ──────────────────────────────────────────────────────
    def early_exit():
        return _build_result(segments, frames, info_ref, total_steps, wall_t0, fail_reason)

    GO = _planning.GRIPPER_OPEN
    GC = _planning.GRIPPER_CLOSED

    if not plan_exec("rise_to_hover", ["Z"],
                     np.array([0.0, 0.0, hover_z_r]), GO, 0):
        return early_exit()

    obj_r = _planning._world_to_robot(env.sim.data.body_xpos[env._obj_body_id[obj_id]], rbase)
    if not plan_exec("xy_to_object", ["X", "Y"],
                     np.array([obj_r[0], obj_r[1], 0.0]), GO, 1):
        return early_exit()

    grasp_pos, _ = _planning.compute_grasp_pose(env, obj_id, robot_base=rbase)
    if not plan_exec("descend_to_grasp", ["Z"], grasp_pos, GO, 2):
        return early_exit()

    total_steps += grip(GC, 20)

    if not plan_exec("lift", ["Z"],
                     np.array([0.0, 0.0, transport_z_r]), GC, 3):
        return early_exit()

    basket_r = _planning._world_to_robot(env.sim.data.body_xpos[env._obj_body_id[container_id]], rbase)
    if not plan_exec("xy_to_basket", ["X", "Y"],
                     np.array([basket_r[0], basket_r[1], 0.0]), GC, 4):
        return early_exit()

    place_pos, _ = _planning.compute_grasp_pose(env, container_id, robot_base=rbase)
    if not plan_exec("descend_to_place", ["Z"], place_pos, GC, 5):
        return early_exit()

    total_steps += grip(GO, 20)

    plan_exec("retract", ["Z"],
              np.array([0.0, 0.0, hover_z_r]), GO, 6)

    return _build_result(segments, frames, info_ref, total_steps, wall_t0, fail_reason)


def _build_result(segments, frames, info, total_steps, wall_t0, fail_reason):
    wall_ms        = (time.monotonic() - wall_t0) * 1000
    total_plan_ms  = sum(s["plan_ms"] for s in segments)
    total_exec_ms  = sum(s["exec_ms"] for s in segments)
    success        = bool(info.get("success", False))
    completion_rate = float(info.get("completion_rate", 1.0 if success else 0.0))
    return {
        "success": success, "completion_rate": completion_rate,
        "segments": segments, "frames": frames,
        "total_plan_ms": total_plan_ms, "total_exec_ms": total_exec_ms,
        "total_steps": total_steps, "wall_ms": wall_ms,
        "fail_reason": fail_reason,
    }


def result_to_csv_row(task_name, init_index, result):
    row = {
        "task": task_name, "init_index": init_index,
        "success": int(result["success"]),
        "completion_rate": f"{result['completion_rate']:.4f}",
        "total_plan_ms":   f"{result['total_plan_ms']:.1f}",
        "total_exec_ms":   f"{result['total_exec_ms']:.1f}",
        "total_steps":      result["total_steps"],
        "wall_ms":         f"{result['wall_ms']:.1f}",
        "fail_reason":      result.get("fail_reason", ""),
    }
    seg_map = {s["seg"]: s for s in result["segments"]}
    for i in range(7):
        s = seg_map.get(i, {})
        row[f"seg{i}_plan_ms"] = f"{s.get('plan_ms', 0.0):.1f}"
        row[f"seg{i}_exec_ms"] = f"{s.get('exec_ms', 0.0):.1f}"
        row[f"seg{i}_ok"]      = int(s.get("ok", False)) if s else ""
    return row


def save_video(frames, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with imageio.get_writer(
        str(path), fps=VIDEO_FPS, codec="libx264",
        quality=5, pixelformat="yuv420p",
        macro_block_size=None,
        ffmpeg_params=["-crf", "28"],
    ) as writer:
        for frame in frames:
            writer.append_data(frame)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--planner", default="curobo", choices=["curobo", "alt"],
                        help="Motion planner module: 'curobo' (planning.py) or "
                             "'alt' (planning_alt.py)")
    parser.add_argument("--task-slice", default="0:10",
                        help="Python slice of task list, e.g. '0:2' for first 2 tasks")
    parser.add_argument("--max-inits", type=int, default=None)
    parser.add_argument("--shard-id", type=int, default=0,
                        help="Used to name the shard CSV (timings_gpu<N>.csv)")
    parser.add_argument("--skip-video", action="store_true")
    args = parser.parse_args()

    # Resolve planner module and expose it as the module-level _planning reference
    import importlib
    global _planning
    module_name = ("libero.vab.planning" if args.planner == "curobo"
                   else "libero.vab.planning_alt")
    _planning = importlib.import_module(module_name)
    _log.info("Using planner module: %s", module_name)

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # Parse task slice
    parts = args.task_slice.split(":")
    t_start = int(parts[0]) if parts[0] else 0
    t_end   = int(parts[1]) if len(parts) > 1 and parts[1] else None
    task_yamls = sorted(TASK_DIR.glob("*.yaml"))[t_start:t_end]

    shard_csv = OUT_DIR / f"timings_{args.planner}_gpu{args.shard_id}.csv"
    _log.info("Worker planner=%s shard=%d tasks=%s n_tasks=%d csv=%s",
              args.planner, args.shard_id, args.task_slice, len(task_yamls), shard_csv.name)

    _log.info("Warming up planner (%s)...", args.planner)
    init_fn = getattr(_planning, "init_planner", None) or getattr(_planning, "_get_directed_planner", None)
    if init_fn:
        init_fn(ROBOT_FILE)
    _log.info("Planner ready.")

    csv_fh = open(shard_csv, "w", newline="")
    writer = csv.DictWriter(csv_fh, fieldnames=CSV_FIELDS)
    writer.writeheader()
    csv_fh.flush()

    ep_total = ep_ok = 0

    for task_yaml in task_yamls:
        task_name = task_yaml.stem
        _log.info("=== Task: %s ===", task_name)
        task_obj = load_task(str(task_yaml))
        env = _make_env(task_obj)

        n_inits = len(task_obj.inits)
        if args.max_inits:
            n_inits = min(n_inits, args.max_inits)

        for init_idx in range(n_inits):
            ep_total += 1
            t_ep = time.monotonic()
            try:
                result = run_episode(env, task_obj, init_index=init_idx)
            except Exception as exc:
                _log.error("CRASH %s init%d: %s", task_name, init_idx, exc)
                result = {
                    "success": False, "completion_rate": 0.0,
                    "segments": [], "frames": [], "total_plan_ms": 0.0,
                    "total_exec_ms": 0.0, "total_steps": 0,
                    "wall_ms": (time.monotonic() - t_ep) * 1000,
                    "fail_reason": f"CRASH: {exc}",
                }

            ep_ok += int(result["success"])
            _log.info(
                "  init%02d [%s] plan=%.0fms exec=%.0fms steps=%d wall=%.1fs "
                "(SR=%.0f%% %d/%d)",
                init_idx, "OK" if result["success"] else "FAIL",
                result["total_plan_ms"], result["total_exec_ms"],
                result["total_steps"], result["wall_ms"] / 1000,
                100.0 * ep_ok / ep_total, ep_ok, ep_total,
            )

            if not args.skip_video and result["frames"]:
                vpath = OUT_DIR / f"{task_name}__init{init_idx:02d}.mp4"
                try:
                    save_video(result["frames"], vpath)
                except Exception as exc:
                    _log.warning("Video save failed: %s", exc)

            writer.writerow(result_to_csv_row(task_name, init_idx, result))
            csv_fh.flush()

        env.close()
        del env

    csv_fh.close()
    _log.info("Shard done: SR=%.1f%% (%d/%d)", 100.0*ep_ok/ep_total, ep_ok, ep_total)


if __name__ == "__main__":
    main()
