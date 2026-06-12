# How to Benchmark Your Motion Planner Against cuRobo

This guide walks you through implementing a new motion planner and running a
fair comparison against the cuRobo baseline documented in `curobo_statistics.md`.

---

## What you are replacing

The pipeline has a single planning function:

```
plan_directed_linear(start_config, *, target_pose, allowed_axes, ...) -> (success, traj_Tx7, reason)
```

Everything else — the geometry helpers that compute grasp targets, the
P-controller execution loop, the episode orchestrator — is identical between
the two planners and lives in the shared infrastructure. You only need to
implement `plan_directed_linear`.

---

## Step 1 — Implement your planner in `planning_alt.py`

Open `libero/libero/vab/planning_alt.py`. There are two functions to implement:

### `init_planner(robot_file="franka.yml")`

Called once before the benchmark loop. Load model weights, compile kernels,
connect to a planning server, etc. The cuRobo baseline uses this to compile its
CUDA graph so the first real episode is not penalised. Do the same for any
one-time setup your planner needs.

```python
def init_planner(robot_file: str = "franka.yml") -> None:
    # Example: load your model
    global _my_planner
    _my_planner = MyPlanner.from_config(robot_file)
    _my_planner.compile()   # do any warm-up here
```

### `plan_directed_linear(...) -> (bool, np.ndarray | None, str)`

This is the core function. Read the full docstring in `planning_alt.py` — it
explains every argument. The short version:

**Input**
- `start_config`: current joint positions, shape `(7,)`, in radians
- `target_pose`: `(position (3,), quat_wxyz (4,))` goal in **panda_hand /
  robot-base frame** (not world frame)
- `allowed_axes`: list of axes the end-effector can move along, e.g. `["Z"]`
  or `["X", "Y"]`. Axes not listed must stay at their FK value.
- `orientation_mode`: always `"LOCK"` in the benchmark — maintain top-down
  orientation throughout the trajectory.
- `endpoint_mode`: always `"PROJECT_TO_TARGET"` — the goal position is the
  FK position with the allowed-axis components replaced by those of `target_pose`.

**Output — a 3-tuple**
```python
(success: bool, trajectory_Tx7: np.ndarray | None, failure_reason: str)
```
- `trajectory_Tx7`: shape `(T, 7)`, joint positions in radians, time-ordered.
  `T >= 2`. `None` on failure.
- The final row must reach the goal within the position tolerance you choose.
- Intermediate rows should minimise off-axis FK drift (this is the "linear
  constraint").
- Returning failure (`False, None, "reason"`) causes the episode to be marked
  as failed.

**Minimal example**
```python
def plan_directed_linear(start_config, *, target_pose, allowed_axes, ...):
    q0 = np.array(start_config[:7], dtype=np.float32)

    # 1. Compute goal joint config via IK
    goal_pos = _project_target(q0, target_pose, allowed_axes)  # your IK
    q_goal = my_ik(goal_pos, quat_wxyz=[0,1,0,0])
    if q_goal is None:
        return False, None, "ik_failed"

    # 2. Interpolate a joint-space trajectory
    T = 20  # number of waypoints
    traj = np.linspace(q0, q_goal, T)   # shape (T, 7)

    return True, traj, ""
```

---

## Step 2 — Understand the coordinate frame

All positions passed to `plan_directed_linear` are in the **robot-base frame**,
not the world frame. The two differ only by a translation:

```
robot_xyz = world_xyz - robot0_base_world_xyz
```

For all VAB table-arena tasks, `robot0_base_world_xyz ≈ [-0.6, 0, 0]`.

The grasp targets are in the **panda_hand FK frame** (not the fingertip frame).
The offset between them for a top-down grasp is:

```
panda_hand_z = fingertip_z + 0.1029 m
```

You do not need to compute grasp targets yourself — `compute_grasp_pose()` in
`planning_alt.py` does this for you and is called by the episode orchestrator.

---

## Step 3 — Run the benchmark

### Single-GPU quick test (1 task, 5 inits)

```bash
source .venv/bin/activate
python benchmark_scripts/run_b1_benchmark.py \
    --planner alt \
    --task-slice 0:1 \
    --max-inits 5 \
    --shard-id 0
```

This writes `videos/b1/timings_alt_gpu0.csv`.

### Full benchmark (5 GPUs in parallel, all 500 episodes)

```bash
bash benchmark_scripts/run_b1_parallel.sh alt
```

Monitor progress:
```bash
python benchmark_scripts/monitor_b1.py   # update monitor_b1.py glob to timings_alt_gpu*.csv
```

Or directly:
```bash
grep -h "init" videos/b1/run_alt_gpu*.log | wc -l
grep -h "init" videos/b1/run_alt_gpu*.log | tail -10
```

### Merge results

```bash
python benchmark_scripts/run_b1_merge.py --planner alt
```

This writes:
- `videos/b1/timings_alt.csv` — 500 rows, one per episode
- `videos/b1/summary_alt.json` — aggregate stats

---

## Step 4 — Compare against cuRobo

The cuRobo numbers are in `videos/b1/timings_curobo.csv` and
`videos/b1/summary_curobo.json` (rename from `timings.csv` / `summary.json`
if you ran the original benchmark before this guide existed).

### Fair comparison metrics

| Metric | Compare? | Notes |
|---|---|---|
| **Success rate** | Yes | Primary metric |
| **Plan time** | Yes | The thing being replaced; target < 841 ms median |
| Execution time | Report but caveat | Equal only if trajectory density matches |
| Wall time | Report but caveat | Depends on both plan and exec |
| Steps per episode | Informative | Proxy for trajectory density |

Execution time and steps will differ if your planner returns trajectories with
a different number of waypoints (T) than cuRobo's 21. The P-controller visits
every 2nd waypoint (`subsample=2`) and spends up to 20 steps converging on
each. A planner that returns 10 waypoints will execute roughly twice as fast as
one that returns 20 — but this is a difference in trajectory density, not
controller quality. Flag it in your comparison.

### Quick comparison script

```python
import csv, json
from pathlib import Path

OUT = Path("videos/b1")

def load(planner):
    rows = list(csv.DictReader(open(OUT / f"timings_{planner}.csv")))
    n    = len(rows)
    ok   = sum(int(r["success"]) for r in rows)
    plan = [float(r["total_plan_ms"]) for r in rows]
    exec_ = [float(r["total_exec_ms"]) for r in rows]
    import numpy as np
    return {
        "sr":          ok / n,
        "plan_median": float(np.median(plan)),
        "plan_p95":    float(np.percentile(plan, 95)),
        "exec_median": float(np.median(exec_)),
    }

cu  = load("curobo")
alt = load("alt")

print(f"{'Metric':<20} {'cuRobo':>10} {'Alt':>10} {'Delta':>10}")
print("-" * 52)
for k in cu:
    delta = alt[k] - cu[k]
    unit  = "%" if k == "sr" else " ms"
    scale = 100 if k == "sr" else 1
    print(f"{k:<20} {cu[k]*scale:>10.1f} {alt[k]*scale:>10.1f} {delta*scale:>+10.1f}{unit}")
```

---

## What not to change for a fair comparison

The following are **fixed** across both planners. Do not modify them:

| Parameter | Value | Location |
|---|---|---|
| Controller | `JOINT_POSITION` (delta, ±0.05 rad/action) | `VABEnv.__init__` |
| `subsample` | 2 | `execute_trajectory()` in `planning_alt.py` |
| `max_steps_per_wp` | 20 | `execute_trajectory()` in `planning_alt.py` |
| `tol` | 0.01 rad | `execute_trajectory()` in `planning_alt.py` |
| Gripper steps | 20 | `set_gripper()` in `planning_alt.py` |
| Init seeds | same 500 episodes | `tasks/libero_object_target_pos_var20x20/` |
| Segment sequence | 7 segments, same axes | `pick_and_place()` in `planning_alt.py` |
| Grasp geometry | centroid + mesh top-Z + −0.06 m | `compute_grasp_pose()` in `planning_alt.py` |

---

## cuRobo baseline targets

From `curobo_statistics.md`:

| Metric | cuRobo |
|---|---|
| Success rate | **98.6%** |
| Plan time median | 841 ms / episode (7 segments) |
| Plan time per segment | 109–157 ms |
| Execution time median | 14,568 ms / episode |
| Steps median | 666 / episode |

A new planner beats cuRobo if it achieves ≥ 98.6% success rate with a lower
median plan time. Execution time is expected to be similar assuming comparable
trajectory density.
