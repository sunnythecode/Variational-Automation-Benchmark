"""Alternative motion planner template for VAB pick-and-place.

Drop-in replacement for libero/libero/vab/planning.py.
Implement plan_directed_linear() below — everything else (geometry helpers,
execution loop, episode orchestrator) is unchanged from the cuRobo baseline.

The public API is identical to planning.py so the benchmark runner can swap
between the two with a single flag:

    python benchmark_scripts/run_b1_benchmark.py --planner alt

See how_to_benchmark_your_planner.md for a full walkthrough.
"""
from __future__ import annotations

import logging
import time
from typing import TYPE_CHECKING, Optional

import numpy as np

if TYPE_CHECKING:
    from .env import VABEnv

_log = logging.getLogger(__name__)

# ── Physical constants (do not change — shared with cuRobo baseline) ──────────
FRANKA_HAND_TO_FINGERTIP_Z_M = 0.0584 + 0.0445  # 0.1029 m
_FRANKA_TOOL_FRAME            = "panda_hand"
_AXIS_IDX                     = {"X": 0, "Y": 1, "Z": 2}

HOVER_Z_FINGERTIP    = 0.25
TRANSPORT_Z_FINGERTIP = 0.25
Z_OFFSET_FINGERTIP   = -0.06

HOVER_Z_HAND     = HOVER_Z_FINGERTIP    + FRANKA_HAND_TO_FINGERTIP_Z_M
TRANSPORT_Z_HAND = TRANSPORT_Z_FINGERTIP + FRANKA_HAND_TO_FINGERTIP_Z_M

# Robosuite Panda gripper: -1 = open, +1 = closed
GRIPPER_OPEN   = -1.0
GRIPPER_CLOSED =  1.0


# ── YOUR PLANNER — implement this section ─────────────────────────────────────

def init_planner(robot_file: str = "franka.yml") -> None:
    """Optional one-time initialisation (called once before the benchmark loop).

    Load model weights, compile kernels, connect to a planning server, etc.
    Raise RuntimeError if the planner cannot be initialised.

    The cuRobo baseline uses this to warm up its CUDA graph so that the first
    real episode is not penalised. Do the same for any compilation step your
    planner needs.
    """
    raise NotImplementedError(
        "init_planner() not implemented. "
        "See how_to_benchmark_your_planner.md."
    )


def plan_directed_linear(
    start_config: np.ndarray,
    *,
    target_pose: Optional[tuple] = None,
    allowed_axes: Optional[list] = None,
    explicit_direction: Optional[np.ndarray] = None,
    distance: Optional[float] = None,
    endpoint_mode: str = "PROJECT_TO_TARGET",
    orientation_mode: str = "LOCK",
    orientation_target: Optional[np.ndarray] = None,
    robot_file: str = "franka.yml",
) -> tuple:
    """Plan a single axis-constrained linear segment in panda_hand FK space.

    THIS IS THE FUNCTION YOU MUST IMPLEMENT.

    ── What it must do ────────────────────────────────────────────────────────
    Compute a joint-space trajectory from `start_config` to a goal that is
    reachable by moving only along `allowed_axes` (e.g. ["Z"] or ["X","Y"]).
    Held axes must remain at their FK value throughout the trajectory.

    The output trajectory will be tracked by a P-controller running at 20 Hz
    (see execute_trajectory below). Sparser trajectories (fewer waypoints) are
    faster to execute; denser ones track the linear path more closely.

    ── Coordinate frame ───────────────────────────────────────────────────────
    All positions are in the cuRobo / panda_hand FK frame, which is the robot-
    base frame (world frame minus the robot0_base world position).  The robot
    base sits at world [-0.6, 0, 0] in all VAB table-arena tasks.

    ── Arguments ──────────────────────────────────────────────────────────────
    start_config : (7,) or (8,) float array
        Current joint positions in radians. If length 8, the 8th element is
        the gripper and should be ignored.

    target_pose : ((3,) position, (4,) quat_wxyz) | None
        Goal end-effector pose in the panda_hand frame.
        Required when endpoint_mode == "PROJECT_TO_TARGET".

    allowed_axes : list of str, e.g. ["Z"] or ["X","Y"]
        Axes the end-effector is free to move along.  Held axes are fixed at
        their FK value computed from start_config.

    explicit_direction : (3,) unit vector | None
        Required when endpoint_mode == "DISTANCE".

    distance : float | None
        Metres to move along explicit_direction.  Required for "DISTANCE" mode.

    endpoint_mode : "PROJECT_TO_TARGET" | "DISTANCE"
        "PROJECT_TO_TARGET" (default) — goal position is constructed by taking
        the FK position and replacing the allowed-axis components with those
        from target_pose[0].
        "DISTANCE" — goal is FK position + distance * explicit_direction.

    orientation_mode : "LOCK" | "SLERP" | "TARGET_AT_END"
        "LOCK" (default and always used by the benchmark) — maintain FK
        orientation (top-down grasp) throughout.

    orientation_target : (4,) quat_wxyz | None
        Override goal orientation.  Ignored when orientation_mode == "LOCK".

    robot_file : str
        Robot config filename (always "franka.yml" in the benchmark).

    ── Return value ───────────────────────────────────────────────────────────
    (success, trajectory_Tx7, failure_reason)

    success : bool
        True if a valid trajectory was found.

    trajectory_Tx7 : np.ndarray shape (T, 7) | None
        Joint-space trajectory in radians.  T >= 2.  None on failure.
        - Rows are time-ordered waypoints.
        - Values must respect the Franka joint limits.
        - The final row must reach the goal within position_tolerance.
        - Intermediate rows should maintain the axis constraints (off-axis
          FK position drift should be minimised).

    failure_reason : str
        Short human-readable string describing why planning failed.
        Empty string on success.

    ── Minimal stub (replace with your implementation) ────────────────────────
    """
    # -------------------------------------------------------------------------
    # DELETE this block and replace with your planner.
    # -------------------------------------------------------------------------
    raise NotImplementedError(
        "plan_directed_linear() not implemented. "
        "See how_to_benchmark_your_planner.md for the contract and examples."
    )
    # -------------------------------------------------------------------------


# ── Geometry helpers (unchanged from cuRobo baseline — do not modify) ─────────

def _object_top_z_world(env: "VABEnv", obj_id: str) -> float:
    """World-frame Z of the highest point of obj_id's collision geometry."""
    body_id    = env._obj_body_id[obj_id]
    geom_start = env.sim.model.body_geomadr[body_id]
    geom_count = env.sim.model.body_geomnum[body_id]

    max_z = -np.inf
    for gi in range(geom_start, geom_start + geom_count):
        gtype = env.sim.model.geom_type[gi]
        xpos  = env.sim.data.geom_xpos[gi]
        xmat  = env.sim.data.geom_xmat[gi].reshape(3, 3)

        if gtype == 7:  # mesh
            dataid = env.sim.model.geom_dataid[gi]
            v0    = env.sim.model.mesh_vertadr[dataid]
            vn    = env.sim.model.mesh_vertnum[dataid]
            verts = env.sim.model.mesh_vert[v0: v0 + vn]
            world_z = (xmat @ verts.T)[2] + xpos[2]
            max_z = max(max_z, float(world_z.max()))
        elif gtype == 6:  # box
            s = env.sim.model.geom_size[gi]
            corners = np.array([[sx, sy, sz]
                                 for sx in (-s[0], s[0])
                                 for sy in (-s[1], s[1])
                                 for sz in (-s[2], s[2])], dtype=np.float32)
            world_z = (xmat @ corners.T)[2] + xpos[2]
            max_z = max(max_z, float(world_z.max()))
        else:
            max_z = max(max_z, float(xpos[2]) + float(env.sim.model.geom_size[gi, 0]))

    if max_z == -np.inf:
        max_z = float(env.sim.data.body_xpos[body_id][2])
    return max_z


def _robot_base_world(env: "VABEnv") -> np.ndarray:
    """World-frame position of the robot0_base body."""
    body_id = env.sim.model.body_name2id("robot0_base")
    return np.array(env.sim.data.body_xpos[body_id], dtype=np.float64)


def _world_to_robot(xyz_world: np.ndarray, robot_base: np.ndarray) -> np.ndarray:
    """World → robot-base frame (pure translation, no rotation in VAB table arenas)."""
    return np.asarray(xyz_world, dtype=np.float64) - robot_base


def compute_grasp_pose(
    env: "VABEnv",
    obj_id: str,
    z_offset: float = Z_OFFSET_FINGERTIP,
    robot_base: Optional[np.ndarray] = None,
) -> tuple:
    """Top-down panda_hand grasp target in robot-base frame for obj_id.

    Returns (position (3,), quat_wxyz (4,)).
    """
    if robot_base is None:
        robot_base = _robot_base_world(env)
    body_id    = env._obj_body_id[obj_id]
    centroid_w = np.array(env.sim.data.body_xpos[body_id])
    top_z_w    = _object_top_z_world(env, obj_id)
    centroid_r = _world_to_robot(centroid_w, robot_base)
    top_z_r    = top_z_w - robot_base[2]
    fingertip_z = top_z_r + z_offset
    hand_z      = fingertip_z + FRANKA_HAND_TO_FINGERTIP_Z_M
    position    = np.array([centroid_r[0], centroid_r[1], hand_z], dtype=np.float64)
    quat_wxyz   = np.array([0.0, 1.0, 0.0, 0.0], dtype=np.float64)
    return position, quat_wxyz


# ── Execution loop (unchanged — do not modify for a fair comparison) ──────────

def execute_trajectory(
    env: "VABEnv",
    traj_Tx7: np.ndarray,
    gripper_cmd: float,
    obs: dict,
    tol: float = 0.01,
    max_steps_per_wp: int = 20,
    subsample: int = 2,
) -> tuple:
    """Track a (T, 7) joint trajectory with the JOINT_POSITION P-controller.

    Visits every `subsample`-th waypoint, converges on each before advancing.
    The final waypoint is always included.

    Parameters match the cuRobo baseline exactly — do not change them.
    """
    info: dict = {}
    indices = list(range(0, len(traj_Tx7), subsample))
    if (len(traj_Tx7) - 1) not in indices:
        indices.append(len(traj_Tx7) - 1)
    for i in indices:
        target_q = traj_Tx7[i]
        for _ in range(max_steps_per_wp):
            error = target_q - obs["proprio"]["joint_pos"]
            if np.max(np.abs(error)) < tol:
                break
            action = np.clip(error / 0.05, -1.0, 1.0)
            obs, _, _, info = env.step(np.append(action, gripper_cmd))
    return obs, info


def set_gripper(
    env: "VABEnv",
    obs: dict,
    gripper_cmd: float,
    steps: int = 20,
) -> tuple:
    """Hold arm joints in place while actuating the gripper for `steps` steps."""
    info: dict = {}
    for _ in range(steps):
        action = np.append(np.zeros(7), gripper_cmd)
        obs, _, _, info = env.step(action)
    return obs, info


# ── Episode orchestrator (unchanged — do not modify) ─────────────────────────

def pick_and_place(
    env: "VABEnv",
    task,
    init_index: int,
    robot_file: str = "franka.yml",
) -> dict:
    """7-segment pick-and-place episode using plan_directed_linear.

    Segment sequence:
      0  rise_to_hover      Z only   open
      1  xy_to_object       XY only  open
      2  descend_to_grasp   Z only   open   → close gripper
      3  lift               Z only   closed
      4  xy_to_basket       XY only  closed
      5  descend_to_place   Z only   closed → open gripper
      6  retract            Z only   open

    Returns {success, completion_rate, segment_results}.
    """
    obj_id       = task.success.args["obj"]
    container_id = task.success.args["container"]

    obs = env.reset(init_index=init_index)
    segment_results: list = []

    rbase         = _robot_base_world(env)
    hover_z_r     = HOVER_Z_FINGERTIP    + FRANKA_HAND_TO_FINGERTIP_Z_M - rbase[2]
    transport_z_r = TRANSPORT_Z_FINGERTIP + FRANKA_HAND_TO_FINGERTIP_Z_M - rbase[2]

    def _plan_exec(label, axes, target_xyz, g_cmd):
        nonlocal obs
        ok, traj, reason = plan_directed_linear(
            start_config     = obs["proprio"]["joint_pos"],
            target_pose      = (target_xyz, np.array([0.0, 1.0, 0.0, 0.0])),
            endpoint_mode    = "PROJECT_TO_TARGET",
            allowed_axes     = axes,
            orientation_mode = "LOCK",
            robot_file       = robot_file,
        )
        if not ok:
            _log.warning("Segment %r failed: %s", label, reason)
            segment_results.append({"label": label, "success": False, "reason": reason})
            return False
        obs, info = execute_trajectory(env, traj, g_cmd, obs)
        segment_results.append({"label": label, "success": True, "steps": len(traj)})
        return True

    if not _plan_exec("rise_to_hover", ["Z"],
                      np.array([0.0, 0.0, hover_z_r]), GRIPPER_OPEN):
        return {"success": False, "completion_rate": 0.0, "segment_results": segment_results}

    obj_r = _world_to_robot(env.sim.data.body_xpos[env._obj_body_id[obj_id]], rbase)
    if not _plan_exec("xy_to_object", ["X", "Y"],
                      np.array([obj_r[0], obj_r[1], 0.0]), GRIPPER_OPEN):
        return {"success": False, "completion_rate": 0.0, "segment_results": segment_results}

    grasp_pos, _ = compute_grasp_pose(env, obj_id, robot_base=rbase)
    if not _plan_exec("descend_to_grasp", ["Z"], grasp_pos, GRIPPER_OPEN):
        return {"success": False, "completion_rate": 0.0, "segment_results": segment_results}

    obs, _ = set_gripper(env, obs, GRIPPER_CLOSED, steps=20)

    if not _plan_exec("lift", ["Z"],
                      np.array([0.0, 0.0, transport_z_r]), GRIPPER_CLOSED):
        return {"success": False, "completion_rate": 0.0, "segment_results": segment_results}

    basket_r = _world_to_robot(env.sim.data.body_xpos[env._obj_body_id[container_id]], rbase)
    if not _plan_exec("xy_to_basket", ["X", "Y"],
                      np.array([basket_r[0], basket_r[1], 0.0]), GRIPPER_CLOSED):
        return {"success": False, "completion_rate": 0.0, "segment_results": segment_results}

    place_pos, _ = compute_grasp_pose(env, container_id, robot_base=rbase)
    if not _plan_exec("descend_to_place", ["Z"], place_pos, GRIPPER_CLOSED):
        return {"success": False, "completion_rate": 0.0, "segment_results": segment_results}

    obs, info = set_gripper(env, obs, GRIPPER_OPEN, steps=20)
    _plan_exec("retract", ["Z"], np.array([0.0, 0.0, hover_z_r]), GRIPPER_OPEN)

    return {
        "success": info.get("success", False),
        "completion_rate": info.get("completion_rate", 0.0),
        "segment_results": segment_results,
    }
