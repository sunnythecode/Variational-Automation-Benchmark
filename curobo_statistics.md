# cuRobo Baseline Statistics — libero_object_target_pos_var20x20

Benchmark of a cuRobo-based pick-and-place pipeline on the full
`libero_object_target_pos_var20x20` suite (10 tasks × 50 inits = 500 episodes).

---

## Results at a Glance

| Metric | Value |
|---|---|
| Episodes | 500 |
| **Success rate** | **98.6% (493 / 500)** |
| Avg wall time / episode | 16.4 ± 1.3 s |
| Avg plan time / episode | 926 ± 750 ms |
| Avg execution time / episode | 14,474 ± 1,106 ms |
| Avg steps / episode | 669 ± 40 |
| Planning share of wall time | 5.6% |
| Execution share of wall time | 88.0% |

The large plan-time std (750 ms) is caused by CUDA graph compilation on the
first episode per worker (~8 s); steady-state median plan time is **841 ms**.

---

## Per-Task Success Rates

| Object | SR | (n) |
|---|---|---|
| alphabet\_soup | 100.0% | 50 |
| bbq\_sauce | 100.0% | 50 |
| butter | 100.0% | 50 |
| chocolate\_pudding | 100.0% | 50 |
| cream\_cheese | 100.0% | 50 |
| **ketchup** | **86.0%** | 50 |
| milk | 100.0% | 50 |
| orange\_juice | 100.0% | 50 |
| salad\_dressing | 100.0% | 50 |
| tomato\_sauce | 100.0% | 50 |

Ketchup is the only object below 100%. It is a tall, narrow bottle; the
centroid-based grasp target lands the fingers near the label rather than a
stable grip band. The 7 failures are all mid-grasp drops.

---

## Per-Segment Timing

Each episode runs 7 axis-constrained segments in sequence. Timing below is
over successful completions of each segment.

| Seg | Label | Axes | Plan median | Plan p5–p95 | Exec median | Exec p5–p95 |
|---|---|---|---|---|---|---|
| 0 | rise\_to\_hover | Z | 120 ms | 110–146 ms | 90 ms | 80–98 ms |
| 1 | xy\_to\_object | XY | 110 ms | 102–135 ms | 2,154 ms | 1,499–3,499 ms |
| 2 | descend\_to\_grasp | Z | 115 ms | 105–170 ms | 2,809 ms | 2,193–3,339 ms |
| 3 | lift | Z | 112 ms | 104–141 ms | 1,921 ms | 1,512–2,638 ms |
| 4 | xy\_to\_basket | XY | 109 ms | 102–129 ms | 2,972 ms | 2,186–3,812 ms |
| 5 | descend\_to\_place | Z | 157 ms | 106–181 ms | 2,819 ms | 2,498–3,537 ms |
| 6 | retract | Z | 111 ms | 104–133 ms | 1,438 ms | 1,310–1,568 ms |

**Planning is fast and consistent (~110–160 ms/segment).** Execution variance is
highest on XY segments (seg1, seg4) because travel distance varies with the
randomised init poses.

---

## Wall-time Distribution (full episode)

| Percentile | Wall time |
|---|---|
| p5 | 14.4 s |
| p25 | 15.5 s |
| median | 16.4 s |
| p75 | 17.1 s |
| p95 | 18.5 s |
| min | 13.4 s |
| max | 24.7 s |

---

## Controller Details

The policy is a simple **P-controller** replaying cuRobo's joint-space
trajectory waypoints through robosuite's `JOINT_POSITION` controller.

### robosuite JOINT\_POSITION controller

- **Mode**: delta — each action component is a *change* in joint angle
- **Scale**: action ±1 → ±0.05 rad applied to the corresponding joint per step
- **Action dim**: 8 (7 arm joints + 1 gripper scalar)
- **Gripper convention**: −1 = open, +1 = closed (robosuite Panda)
- **Control frequency**: 20 Hz (one `env.step()` = 0.05 s of sim time)

### Waypoint-tracking loop (`execute_trajectory` in `libero/libero/vab/planning.py`)

```
for each waypoint i (every subsample-th):
    for up to max_steps_per_wp steps:
        error = target_joints[i] - current_joints
        if max(|error|) < tol:  break          # converged
        action = clip(error / 0.05, -1, 1)    # P-gain = 1/0.05
        env.step([action..., gripper_cmd])
```

| Parameter | Value | Effect |
|---|---|---|
| `subsample` | 2 | Visit every 2nd waypoint; halves steps vs tracking all |
| `max_steps_per_wp` | 20 | Hard cap on convergence steps per waypoint |
| `tol` | 0.01 rad | Per-joint convergence threshold |

With 21 cuRobo waypoints per segment and `subsample=2`, the controller visits
11 waypoints per segment. Execution time scales with object distance (more XY
travel → more steps) and robot's proximity to the target at each waypoint.

---

## cuRobo Hyperparameters

All parameters live in `_get_directed_planner()` in
`libero/libero/vab/planning.py`.

| Parameter | Value | Effect on plan time |
|---|---|---|
| `num_ik_seeds` | 32 | Seeds for IK warm-start; more → better coverage, slower first solve |
| `use_cuda_graph` | True | Compiles computation graph on first call (~7 s); all subsequent calls ~100 ms |
| `interpolation_dt` | 0.05 s | Time spacing between output waypoints; matches 20 Hz control rate |
| `position_tolerance` | 0.005 m | IK position convergence criterion |
| `orientation_tolerance` | 0.05 rad | IK orientation convergence criterion |
| `max_attempts` | 10 | Re-plans if first attempt fails (rare) |
| `self_collision_check` | False | Disabled — no obstacles in VAB table scenes |

The CUDA graph is compiled once per process and cached in `_planner_cache`.
Steady-state plan time is **~110–160 ms per segment** regardless of scene.

### Linear constraint API (`plan_directed_linear`)

Each segment is planned with `plan_directed_linear(...)` which wraps cuRobo's
`plan_pose` with:

- **`endpoint_mode="PROJECT_TO_TARGET"`** — goal is constructed by projecting
  the FK position onto the allowed axes toward the target; held axes stay fixed
- **`orientation_mode="LOCK"`** — orientation is held at the FK value throughout
  the trajectory (top-down grasp maintained for all segments)
- **`ToolPoseCriteria`** — non-terminal waypoints are penalised for off-axis
  drift, enforcing linear motion along allowed axes

---

## Code Locations

| What | File | Key symbol |
|---|---|---|
| Planner init & cache | `libero/libero/vab/planning.py` | `_get_directed_planner()` |
| Linear plan call | `libero/libero/vab/planning.py` | `plan_directed_linear()` |
| Waypoint execution loop | `libero/libero/vab/planning.py` | `execute_trajectory()` |
| Gripper actuation | `libero/libero/vab/planning.py` | `set_gripper()` |
| Episode orchestrator | `libero/libero/vab/planning.py` | `pick_and_place()` |
| Geometry helpers | `libero/libero/vab/planning.py` | `compute_grasp_pose()`, `_object_top_z_world()` |
| Benchmark runner | `benchmark_scripts/run_b1_benchmark.py` | `run_episode()`, `main()` |
| Parallel launch | `benchmark_scripts/run_b1_parallel.sh` | — |
| Result merge | `benchmark_scripts/run_b1_merge.py` | — |
| Raw episode data | `videos/b1/timings.csv` | 500 rows, one per episode |
| Aggregate stats | `videos/b1/summary.json` | Per-segment averages |
| Episode videos | `videos/b1/<task>__init<N>.mp4` | 128×128 agentview, 20 fps |

---

## Benchmark Conditions

- **Suite**: `tasks/libero_object_target_pos_var20x20/` (10 tasks, 50 inits each)
- **Robot**: Franka Panda (MountedPanda in robosuite)
- **Arena**: table (floor-mounted robot at world `[−0.6, 0, 0]`)
- **Hardware**: 5× Tesla V100-PCIE-32GB, one worker per GPU
- **cuRobo version**: v0.8 (cuda-core backend, sm\_70)
- **Robot config**: `robot_metadata/franka.yml`
