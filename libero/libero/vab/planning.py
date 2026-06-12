"""Motion planning for VAB using cuRobo v0.8 constrained linear trajectories.

All planning operates in the panda_hand FK frame (not robosuite's grip_site frame).
The two frames differ by FRANKA_HAND_TO_FINGERTIP_Z_M along panda_hand's Z axis.
For a top-down grasp (quat_wxyz=[0,1,0,0]), panda_hand is directly ABOVE fingertips:
  panda_hand_z = fingertip_z + FRANKA_HAND_TO_FINGERTIP_Z_M

Install cuRobo v0.8 before using this module:
  uv pip install -e ~/graph-as-policy/third_party/curobo/
"""
from __future__ import annotations

import logging
import time
from typing import TYPE_CHECKING, Optional

import numpy as np

if TYPE_CHECKING:
    from .env import VABEnv

_log = logging.getLogger(__name__)

# ── cuRobo v0.8 optional imports ─────────────────────────────────────────────
_CUROBO_AVAILABLE = False
try:
    import torch
    from curobo.motion_planner import MotionPlanner, MotionPlannerCfg
    from curobo._src.state.state_joint import JointState as _CuJointState
    from curobo._src.types.pose import Pose as _CuPose
    from curobo._src.types.tool_pose import GoalToolPose
    from curobo._src.cost.cost_pose_metric import PoseCostMetric
    from curobo._src.cost.tool_pose_criteria import ToolPoseCriteria
    from curobo._src.types.device_cfg import DeviceCfg
    _CUROBO_AVAILABLE = True
except ImportError:
    _log.warning(
        "cuRobo v0.8 not found. Install with: "
        "uv pip install -e ~/graph-as-policy/third_party/curobo/"
    )

# ── Physical constants ────────────────────────────────────────────────────────
# panda_hand → fingertip offset (m): finger body 0.0584m + pad 0.0445m.
# Matches FRANKA_HAND_TO_FINGERTIP_Z_M in graph-as-policy/third_party/curobo_api.py.
FRANKA_HAND_TO_FINGERTIP_Z_M = 0.0584 + 0.0445  # 0.1029 m

_FRANKA_TOOL_FRAME = "panda_hand"
_AXIS_IDX = {"X": 0, "Y": 1, "Z": 2}

# Fingertip-frame heights (world Z, metres)
HOVER_Z_FINGERTIP    = 0.25   # safe transit height above table surface
TRANSPORT_Z_FINGERTIP = 0.25  # height while carrying an object
Z_OFFSET_FINGERTIP   = -0.06  # depth below object top to close gripper (from graph-as-policy)

# Equivalent panda_hand heights (for top-down grasp, panda_hand is above fingertip)
HOVER_Z_HAND     = HOVER_Z_FINGERTIP + FRANKA_HAND_TO_FINGERTIP_Z_M
TRANSPORT_Z_HAND = TRANSPORT_Z_FINGERTIP + FRANKA_HAND_TO_FINGERTIP_Z_M

# Gripper command convention (robosuite Panda): -1 = open, +1 = closed
GRIPPER_OPEN   = -1.0
GRIPPER_CLOSED =  1.0

# ── Planner singleton cache ───────────────────────────────────────────────────
_planner_cache: dict = {}


def _get_directed_planner(robot_file: str = "franka.yml"):
    """Lazy-init and cache a cuRobo v0.8 MotionPlanner (kinematics only, no collision).

    Ported verbatim from graph-as-policy/third_party/curobo_api.py::_get_directed_planner.
    interpolation_dt=0.15 gives ~7 waypoints/segment (vs ~21 at default 0.025s),
    reducing sim steps per segment. Max allowed by solver_trajopt_cfg.py is 0.2s.
    """
    if not _CUROBO_AVAILABLE:
        raise RuntimeError(
            "cuRobo v0.8 is required.\n"
            "Install: uv pip install -e ~/graph-as-policy/third_party/curobo/"
        )
    if robot_file in _planner_cache:
        return _planner_cache[robot_file]

    device_cfg = DeviceCfg()
    cfg = MotionPlannerCfg.create(
        robot=robot_file,
        self_collision_check=False,
        device_cfg=device_cfg,
        num_ik_seeds=32,
        position_tolerance=0.005,
        orientation_tolerance=0.05,
        use_cuda_graph=True,
    )
    planner = MotionPlanner(cfg)
    planner.trajopt_solver.config.interpolation_dt = 0.05  # match 20 Hz sim (1/control_freq)
    _planner_cache[robot_file] = planner
    return planner


def _build_linear_tool_pose_criteria(held_idx: set, orientation_mode: str, mode: str, device_cfg):
    """Path-constraint ToolPoseCriteria that prevents off-axis drift at intermediate poses.

    Ported verbatim from graph-as-policy/third_party/curobo_api.py.
    """
    if mode == "ORIENT_IN_PLACE":
        pos_non_term = [1.0, 1.0, 1.0]
        rot_non_term = [0.0, 0.0, 0.0]
    else:
        pos_non_term = [1.0 if i in held_idx else 0.0 for i in range(3)]
        if orientation_mode.upper() in ("LOCK", "SLERP"):
            rot_non_term = [1.0, 1.0, 1.0]
        else:
            rot_non_term = [0.0, 0.0, 0.0]

    non_terminal = [*pos_non_term, *rot_non_term]
    terminal = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
    use_projection = (len(held_idx) == 2) or (mode == "ORIENT_IN_PLACE")

    return ToolPoseCriteria(
        terminal_pose_axes_weight_factor=terminal,
        non_terminal_pose_axes_weight_factor=non_terminal,
        project_distance_to_goal=use_projection,
        device_cfg=device_cfg,
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
    """Plan a constrained linear trajectory with cuRobo v0.8 MotionPlanner.

    All positions are in panda_hand FK frame. Ported from graph-as-policy/
    third_party/curobo_api.py — v0.8 code path only.

    Args:
        start_config: (7,) or (8,) joint positions (gripper element ignored).
        target_pose: ((3,) position, (4,) quat_wxyz) goal in panda_hand frame.
        allowed_axes: Axes free to move, e.g. ["Z"] or ["X","Y"]. Omitted axes
            are held at their FK value.
        explicit_direction: Unit vector for DISTANCE mode.
        distance: Metres to move along explicit_direction (DISTANCE mode).
        endpoint_mode: "PROJECT_TO_TARGET" (default) or "DISTANCE".
        orientation_mode: "LOCK" holds FK orientation throughout (default),
            "SLERP" or "TARGET_AT_END" interpolates toward target orientation.
        orientation_target: Override target orientation (quat_wxyz); falls back
            to target_pose[1] or FK orientation.
        robot_file: cuRobo robot config filename ("franka.yml").

    Returns:
        (success: bool, trajectory_Tx7: np.ndarray | None, failure_reason: str)
    """
    if not _CUROBO_AVAILABLE:
        raise RuntimeError("cuRobo v0.8 required — see module docstring for install.")

    planner = _get_directed_planner(robot_file)
    planner.reset_seed()

    device_cfg = planner.config.device_cfg
    joint_names = planner.joint_names
    n_dof = len(joint_names)

    # Trim or pad to n_dof (7 for Franka)
    cfg = np.array(start_config, dtype=np.float32)
    if len(cfg) > n_dof:
        cfg = cfg[:n_dof]
    elif len(cfg) < n_dof:
        cfg = np.concatenate([cfg, np.zeros(n_dof - len(cfg), dtype=np.float32)])

    start_state = _CuJointState.from_numpy(
        joint_names=joint_names,
        position=np.expand_dims(cfg, 0),
        device_cfg=device_cfg,
    )

    kin = planner.compute_kinematics(start_state)
    fk_pos  = kin.tool_poses.position[0, 0, 0, :].cpu().numpy()
    fk_quat = kin.tool_poses.quaternion[0, 0, 0, :].cpu().numpy()  # wxyz

    mode = endpoint_mode.upper()

    if mode == "ORIENT_IN_PLACE":
        free_idx: set = set()
        held_idx: set = {0, 1, 2}
        goal_pos = fk_pos.copy()
    else:
        if allowed_axes is None:
            allowed_axes = ["X", "Y", "Z"]
        free_idx = {_AXIS_IDX[a.upper()] for a in allowed_axes if a.upper() in _AXIS_IDX}
        held_idx = {0, 1, 2} - free_idx
        goal_pos = fk_pos.copy()

        if mode == "DISTANCE":
            if explicit_direction is None or distance is None:
                return False, None, "DISTANCE mode requires explicit_direction and distance"
            d = np.array(explicit_direction, dtype=np.float32)
            norm = np.linalg.norm(d)
            if norm > 1e-6:
                d = d / norm
            goal_pos = fk_pos + distance * d
        else:  # PROJECT_TO_TARGET
            if target_pose is None:
                return False, None, "PROJECT_TO_TARGET mode requires target_pose"
            tgt_pos = np.array(target_pose[0], dtype=np.float32)
            for i in free_idx:
                goal_pos[i] = tgt_pos[i]

    if orientation_mode.upper() == "LOCK":
        goal_quat = fk_quat.copy()
    elif orientation_target is not None:
        goal_quat = np.array(orientation_target, dtype=np.float32)
    elif target_pose is not None:
        goal_quat = np.array(target_pose[1], dtype=np.float32)
    else:
        goal_quat = fk_quat.copy()

    # Hold-partial-pose weight vector: [rot_x, rot_y, rot_z, pos_x, pos_y, pos_z]
    hvw = [0.0] * 6
    if orientation_mode.upper() in ("LOCK", "SLERP"):
        hvw[0] = hvw[1] = hvw[2] = 1.0
    for i in held_idx:
        hvw[3 + i] = 1.0

    tool_frame = planner.tool_frames[0] if planner.tool_frames else _FRANKA_TOOL_FRAME

    pose_metric = None
    if any(v > 0 for v in hvw):
        hvw_t = device_cfg.to_device(torch.tensor(hvw, dtype=torch.float32))
        pose_metric = PoseCostMetric(hold_partial_pose=True, hold_vec_weight=hvw_t)

    goal_pos_t  = device_cfg.to_device(torch.tensor(goal_pos,  dtype=torch.float32).unsqueeze(0))
    goal_quat_t = device_cfg.to_device(torch.tensor(goal_quat, dtype=torch.float32).unsqueeze(0))
    goal_v2pose = _CuPose(position=goal_pos_t, quaternion=goal_quat_t, name=tool_frame)
    goal_tool_poses = GoalToolPose.from_poses({tool_frame: goal_v2pose})

    _log.debug(
        "plan_directed_linear axes=%s orient=%s hvw=%s goal=%s fk=%s",
        allowed_axes, orientation_mode, hvw,
        np.round(goal_pos, 4).tolist(), np.round(fk_pos, 4).tolist(),
    )

    if pose_metric is not None:
        planner.ik_solver.update_pose_cost_metric({tool_frame: pose_metric})
        planner.trajopt_solver.update_pose_cost_metric({tool_frame: pose_metric})

    path_criteria = _build_linear_tool_pose_criteria(held_idx, orientation_mode, mode, device_cfg)
    planner.update_tool_pose_criteria({tool_frame: path_criteria})

    t0 = time.monotonic()
    try:
        result = planner.plan_pose(goal_tool_poses, start_state, max_attempts=10)
    finally:
        # Always reset metrics so the cached planner is clean for next call.
        reset_metric = PoseCostMetric.reset_metric()
        planner.ik_solver.update_pose_cost_metric({tool_frame: reset_metric})
        planner.trajopt_solver.update_pose_cost_metric({tool_frame: reset_metric})
        planner.update_tool_pose_criteria(
            {tool_frame: ToolPoseCriteria(device_cfg=device_cfg)}
        )
        _log.info(
            "plan_directed_linear axes=%s orient=%s plan_ms=%.1f",
            allowed_axes, orientation_mode, (time.monotonic() - t0) * 1000,
        )

    if result is None:
        return False, None, "planner_returned_none"

    success = bool(
        result.success is not None and torch.any(result.success).item()
    )
    if not success:
        status = getattr(result, "status", None) or getattr(result, "failure_reason", "") or "unknown"
        return False, None, f"motion_gen_failed: {status}"

    try:
        traj_js = result.get_interpolated_plan()
        if traj_js is not None and traj_js.position is not None:
            traj = np.squeeze(traj_js.position.detach().cpu().numpy())
            if traj.ndim == 1:
                traj = traj.reshape(1, -1)
            return True, traj[:, :7], ""
    except Exception as exc:
        return False, None, f"trajectory_extraction: {exc}"

    return False, None, "no_interpolated_plan"


# ── Geometry helpers ──────────────────────────────────────────────────────────

def _object_top_z_world(env: "VABEnv", obj_id: str) -> float:
    """World-frame Z of the highest point of obj_id's geometry.

    Handles mesh geoms (MuJoCo type 7) and box primitives (type 6).
    Vertex positions are transformed to world frame via geom_xpos + geom_xmat.
    """
    body_id = env._obj_body_id[obj_id]
    geom_start = env.sim.model.body_geomadr[body_id]
    geom_count = env.sim.model.body_geomnum[body_id]

    max_z = -np.inf
    for gi in range(geom_start, geom_start + geom_count):
        gtype = env.sim.model.geom_type[gi]
        xpos  = env.sim.data.geom_xpos[gi]               # world position of geom origin
        xmat  = env.sim.data.geom_xmat[gi].reshape(3, 3) # world rotation of geom frame

        if gtype == 7:  # mesh
            dataid = env.sim.model.geom_dataid[gi]
            v0 = env.sim.model.mesh_vertadr[dataid]
            vn = env.sim.model.mesh_vertnum[dataid]
            verts = env.sim.model.mesh_vert[v0: v0 + vn]   # (N, 3) in geom local frame
            world_z = (xmat @ verts.T)[2] + xpos[2]
            max_z = max(max_z, float(world_z.max()))
        elif gtype == 6:  # box
            s = env.sim.model.geom_size[gi]                # half-extents (hx, hy, hz)
            corners = np.array([[sx, sy, sz]
                                 for sx in (-s[0], s[0])
                                 for sy in (-s[1], s[1])
                                 for sz in (-s[2], s[2])], dtype=np.float32)
            world_z = (xmat @ corners.T)[2] + xpos[2]
            max_z = max(max_z, float(world_z.max()))
        else:
            # Sphere / cylinder / capsule: approximate with geom origin + size[0] radius.
            max_z = max(max_z, float(xpos[2]) + float(env.sim.model.geom_size[gi, 0]))

    if max_z == -np.inf:
        max_z = float(env.sim.data.body_xpos[body_id][2])

    return max_z


def _robot_base_world(env: "VABEnv") -> np.ndarray:
    """World-frame position of the robot0_base body."""
    body_id = env.sim.model.body_name2id("robot0_base")
    return np.array(env.sim.data.body_xpos[body_id], dtype=np.float64)


def _world_to_robot(xyz_world: np.ndarray, robot_base: np.ndarray) -> np.ndarray:
    """Convert world-frame XYZ to cuRobo robot-base frame.

    The robot base axes are aligned with world axes (no base rotation in VAB),
    so the transform is a pure translation: robot_xyz = world_xyz - base_world_xyz.
    """
    return np.asarray(xyz_world, dtype=np.float64) - robot_base


def compute_grasp_pose(
    env: "VABEnv",
    obj_id: str,
    z_offset: float = Z_OFFSET_FINGERTIP,
    robot_base: Optional[np.ndarray] = None,
) -> tuple:
    """Top-down grasp target in cuRobo robot-base frame for obj_id.

    Uses ground-truth mesh geometry to find the world-Z top surface, applies
    z_offset (fingertip frame, negative = into object), then converts to
    panda_hand frame by adding FRANKA_HAND_TO_FINGERTIP_Z_M.

    Matches the z_offset used by graph-as-policy (same Franka Panda robot).

    Args:
        robot_base: World position of robot0_base body. Computed once per
            episode and passed in to avoid repeated model lookups.

    Returns:
        position (3,): panda_hand XYZ target in cuRobo robot-base frame
        quat_wxyz (4,): [0, 1, 0, 0] — 180° around X = top-down approach
    """
    if robot_base is None:
        robot_base = _robot_base_world(env)

    body_id    = env._obj_body_id[obj_id]
    centroid_w = np.array(env.sim.data.body_xpos[body_id])
    top_z_w    = _object_top_z_world(env, obj_id)

    # Convert centroid XY to robot frame; Z is only offset from world Z
    centroid_r = _world_to_robot(centroid_w, robot_base)
    top_z_r    = top_z_w - robot_base[2]

    fingertip_z = top_z_r + z_offset
    hand_z      = fingertip_z + FRANKA_HAND_TO_FINGERTIP_Z_M

    position  = np.array([centroid_r[0], centroid_r[1], hand_z], dtype=np.float64)
    quat_wxyz = np.array([0.0, 1.0, 0.0, 0.0], dtype=np.float64)  # top-down
    return position, quat_wxyz


# ── Trajectory execution ──────────────────────────────────────────────────────

def execute_trajectory(
    env: "VABEnv",
    traj_Tx7: np.ndarray,
    gripper_cmd: float,
    obs: dict,
    tol: float = 0.01,
    max_steps_per_wp: int = 20,
    subsample: int = 2,
) -> tuple:
    """Track a (T, 7) joint trajectory through the env with JOINT_POSITION control.

    Visits every `subsample`-th waypoint (matching graph-as-policy's subsampling
    approach) and converges on each before advancing. This keeps the robot on the
    planned linear path at each visited waypoint while reducing total env steps
    vs. tracking all T waypoints.

    The final waypoint is always included regardless of subsample.

    Args:
        env: VABEnv with controller="JOINT_POSITION".
        traj_Tx7: (T, 7) absolute joint positions from plan_directed_linear.
        gripper_cmd: GRIPPER_OPEN or GRIPPER_CLOSED.
        obs: Current obs dict from reset() or previous step().
        tol: Per-joint convergence threshold in radians.
        max_steps_per_wp: Max env steps to spend converging on each waypoint.
        subsample: Visit every Nth waypoint. Final waypoint always included.

    Returns:
        (final_obs, final_info)
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
    """Hold joints in place while actuating the gripper for `steps` env steps."""
    info: dict = {}
    for _ in range(steps):
        current_q = obs["proprio"]["joint_pos"]
        error = np.zeros_like(current_q)  # zero arm delta = hold
        action = np.clip(error / 0.05, -1.0, 1.0)
        action = np.append(action, gripper_cmd)
        obs, _, _, info = env.step(action)
    return obs, info


# ── Full pick-and-place orchestrator ─────────────────────────────────────────

def pick_and_place(
    env: "VABEnv",
    task,
    init_index: int,
    robot_file: str = "franka.yml",
) -> dict:
    """One-shot pick-and-place episode using cuRobo axis-constrained linear planning.

    Reads the target object and container from task.success.args. Each motion
    segment is planned with plan_directed_linear using PROJECT_TO_TARGET and
    LOCK orientation:

        Seg 0: Rise to hover        — Z only, gripper open
        Seg 1: XY to above object   — XY only, gripper open
        Seg 2: Descend to grasp     — Z only, gripper open
           Close gripper (20 steps)
        Seg 3: Lift to transport    — Z only, gripper closed
        Seg 4: XY to above basket   — XY only, gripper closed
        Seg 5: Descend into basket  — Z only, gripper closed
           Open gripper (20 steps)
        Seg 6: Retract to hover     — Z only, gripper open

    Requires env created with controller="JOINT_POSITION".

    Returns dict: {success, completion_rate, segment_results}
    """
    obj_id       = task.success.args["obj"]
    container_id = task.success.args["container"]

    obs = env.reset(init_index=init_index)
    segment_results: list = []

    # Compute robot base once — world → robot-frame offset for all planning calls.
    rbase = _robot_base_world(env)
    hover_z_r     = HOVER_Z_FINGERTIP    + FRANKA_HAND_TO_FINGERTIP_Z_M - rbase[2]
    transport_z_r = TRANSPORT_Z_FINGERTIP + FRANKA_HAND_TO_FINGERTIP_Z_M - rbase[2]

    def _plan_exec(label: str, axes: list, target_xyz: np.ndarray, g_cmd: float):
        nonlocal obs
        joints = obs["proprio"]["joint_pos"]
        success, traj, reason = plan_directed_linear(
            start_config     = joints,
            target_pose      = (target_xyz, np.array([0.0, 1.0, 0.0, 0.0])),
            endpoint_mode    = "PROJECT_TO_TARGET",
            allowed_axes     = axes,
            orientation_mode = "LOCK",
            robot_file       = robot_file,
        )
        if not success:
            _log.warning("Segment %r planning failed: %s", label, reason)
            segment_results.append({"label": label, "success": False, "reason": reason})
            return False, {}
        obs, info = execute_trajectory(env, traj, g_cmd, obs)
        segment_results.append({"label": label, "success": True, "steps": len(traj)})
        return True, info

    # ── Seg 0: Rise to hover ──────────────────────────────────────────────────
    ok, _ = _plan_exec(
        "rise_to_hover", ["Z"],
        np.array([0.0, 0.0, hover_z_r]),   # XY held by PROJECT_TO_TARGET
        GRIPPER_OPEN,
    )
    if not ok:
        return {"success": False, "completion_rate": 0.0, "segment_results": segment_results}

    # ── Seg 1: XY to above object ─────────────────────────────────────────────
    obj_r = _world_to_robot(env.sim.data.body_xpos[env._obj_body_id[obj_id]], rbase)
    ok, _ = _plan_exec(
        "xy_to_object", ["X", "Y"],
        np.array([obj_r[0], obj_r[1], 0.0]),  # Z held
        GRIPPER_OPEN,
    )
    if not ok:
        return {"success": False, "completion_rate": 0.0, "segment_results": segment_results}

    # ── Seg 2: Descend to grasp ───────────────────────────────────────────────
    grasp_pos, _ = compute_grasp_pose(env, obj_id, robot_base=rbase)
    ok, _ = _plan_exec(
        "descend_to_grasp", ["Z"],
        grasp_pos,   # XY held; Z → grasp_Z_hand (robot frame)
        GRIPPER_OPEN,
    )
    if not ok:
        return {"success": False, "completion_rate": 0.0, "segment_results": segment_results}

    # ── Close gripper ─────────────────────────────────────────────────────────
    obs, _ = set_gripper(env, obs, GRIPPER_CLOSED, steps=20)

    # ── Seg 3: Lift to transport height ──────────────────────────────────────
    ok, _ = _plan_exec(
        "lift", ["Z"],
        np.array([0.0, 0.0, transport_z_r]),  # XY held
        GRIPPER_CLOSED,
    )
    if not ok:
        return {"success": False, "completion_rate": 0.0, "segment_results": segment_results}

    # ── Seg 4: XY to above basket ─────────────────────────────────────────────
    basket_r = _world_to_robot(env.sim.data.body_xpos[env._obj_body_id[container_id]], rbase)
    ok, _ = _plan_exec(
        "xy_to_basket", ["X", "Y"],
        np.array([basket_r[0], basket_r[1], 0.0]),  # Z held
        GRIPPER_CLOSED,
    )
    if not ok:
        return {"success": False, "completion_rate": 0.0, "segment_results": segment_results}

    # ── Seg 5: Descend into basket ────────────────────────────────────────────
    place_pos, _ = compute_grasp_pose(env, container_id, robot_base=rbase)
    ok, info = _plan_exec(
        "descend_to_place", ["Z"],
        place_pos,   # XY held; Z → place_Z_hand (robot frame)
        GRIPPER_CLOSED,
    )
    if not ok:
        return {"success": False, "completion_rate": 0.0, "segment_results": segment_results}

    # ── Open gripper ──────────────────────────────────────────────────────────
    obs, info = set_gripper(env, obs, GRIPPER_OPEN, steps=20)

    # ── Seg 6: Retract ────────────────────────────────────────────────────────
    ok, info = _plan_exec(
        "retract", ["Z"],
        np.array([0.0, 0.0, HOVER_Z_HAND]),
        GRIPPER_OPEN,
    )

    return {
        "success": info.get("success", False),
        "completion_rate": info.get("completion_rate", 0.0),
        "segment_results": segment_results,
    }
