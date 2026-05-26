"""Bimanual VAB env for the crate_washing scene.

Subclasses robosuite's TwoArmEnv with env_configuration='single-arm-parallel'
to wire up two single-arm Pandas. The scene MJCF (loaded by
CrateWashingArena) is self-contained: it bakes in the 11-crate stack, the
washing-machine fixture, and the robot platform. We add no movable objects;
inits only need to seat the two robots and let physics settle.

v1 scope: scene + dual-arm control + strict-obs filtering + a simple
"top crate lifted" success check. Multi-stage / sequenced goals are
deferred to a stage-aware SuccessSpec.
"""
from __future__ import annotations

import os
from typing import Any, Dict, Optional

import numpy as np
import robosuite as suite
from robosuite.environments.manipulation.two_arm_env import TwoArmEnv
from robosuite.models.tasks import ManipulationTask

from libero import envs as _libero_envs  # noqa: F401
from libero.envs.arenas.crate_washing_arena import CrateWashingArena

from . import predicates as _predicates
from .schema import Task


_PROPRIO_KEYS = {
    f"robot{idx}_{suffix}"
    for idx in (0, 1)
    for suffix in ("joint_pos", "joint_vel", "eef_pos", "eef_quat",
                   "gripper_qpos")
}


class VABBimanualEnv(TwoArmEnv):
    def __init__(
        self,
        task: Task,
        has_renderer: bool = False,
        has_offscreen_renderer: bool = True,
        control_freq: int = 20,
        render_gpu_device_id: int = -1,
        ignore_done: bool = True,
        hard_reset: bool = True,
        **kwargs,
    ):
        self.task = task
        if task.arena.name != "crate_washing":
            raise ValueError(
                f"VABBimanualEnv only supports arena.name='crate_washing'"
                f" (got {task.arena.name!r})"
            )
        if not task.robots or len(task.robots) != 2:
            raise ValueError(
                f"VABBimanualEnv requires exactly 2 robots in YAML;"
                f" got {len(task.robots) if task.robots else 0}"
            )

        # crate_washing has its own `robot_platform` body in the MJCF; no
        # RethinkMount pedestal is needed. Use OnTheGroundPanda (gripper +
        # arm only) to match the original deleted Libero_Crate_Washing class.
        robot_classes = [
            f"OnTheGround{spec.name.capitalize()}" for spec in task.robots
        ]
        # Use the first robot's controller for both (per robosuite TwoArmEnv).
        controller_configs = suite.load_controller_config(
            default_controller=task.robots[0].controller
        )

        self._obj_body_id: Dict[str, int] = {}
        self._last_success: bool = False
        self._current_init_index: int = task.default_init_index

        self.use_object_obs = False
        self.reward_scale = 1.0
        self.reward_shaping = False

        super().__init__(
            robots=robot_classes,
            env_configuration="single-arm-parallel",
            controller_configs=controller_configs,
            mount_types="default",
            gripper_types="default",
            initialization_noise=None,
            use_camera_obs=True,
            has_renderer=has_renderer,
            has_offscreen_renderer=has_offscreen_renderer,
            render_camera="frontview",
            render_collision_mesh=False,
            render_visual_mesh=True,
            render_gpu_device_id=render_gpu_device_id,
            control_freq=control_freq,
            horizon=task.horizon,
            ignore_done=ignore_done,
            hard_reset=hard_reset,
            camera_names=list(task.cameras),
            camera_heights=task.camera_height,
            camera_widths=task.camera_width,
            camera_depths=task.camera_depth,
            camera_segmentations=None,
            renderer="mujoco",
            **kwargs,
        )

    def _load_model(self):
        super()._load_model()
        mujoco_arena = CrateWashingArena()

        # Robot base positions + yaw match the deleted Libero_Crate_Washing
        # constants. Two OnTheGroundPandas on the scene's robot_platform body,
        # rotated 180 deg so the grippers face the crate stack.
        base_positions = [(1.74, -0.20, 0.76), (1.74, 0.20, 0.76)]
        base_yaw = np.pi
        for robot, pos in zip(self.robots, base_positions):
            robot.robot_model.set_base_xpos(pos)
            robot.robot_model.set_base_ori((0.0, 0.0, base_yaw))

        mujoco_arena.set_origin([0.0, 0.0, 0.0])

        # Add an `agentview` camera framing both arms + crate stack so we get
        # a useful view from the canonical camera name. Inherited from the
        # deleted BimanualBDDLBaseDomain._setup_camera.
        mujoco_arena.set_camera(
            camera_name="agentview",
            pos=[1.0, -1.5, 1.6],
            quat=[0.816937, 0.5495927, -0.0975822, -0.1450502],
        )

        self.model = ManipulationTask(
            mujoco_arena=mujoco_arena,
            mujoco_robots=[r.robot_model for r in self.robots],
            mujoco_objects=[],
        )

    def _setup_references(self):
        super()._setup_references()
        self._obj_body_id = {}
        # The scene XML names every crate body crate_box_<n>; ditto fixtures.
        for body_name in ("crate_box_11", "crate_machine", "robot_platform"):
            try:
                self._obj_body_id[body_name] = self.sim.model.body_name2id(body_name)
            except (KeyError, ValueError):
                pass

    def _setup_observables(self):
        observables = super()._setup_observables()
        for key in list(observables.keys()):
            if key in _PROPRIO_KEYS:
                continue
            if key.endswith("_image") or key.endswith("_depth"):
                continue
            observables[key].set_enabled(False)
            observables[key].set_active(False)
        return observables

    def reset(self, init_index: Optional[int] = None):
        super().reset()
        if init_index is None:
            init_index = self.task.default_init_index
        self._current_init_index = init_index
        # No movable objects to reposition for crate_washing; physics settles.
        self.sim.forward()
        self._last_success = False
        return self._filter_obs(self._get_observations(force_update=True))

    def step(self, action):
        obs, reward, done, info = super().step(action)
        self._last_success = bool(self._check_success())
        info = dict(info) if info else {}
        info["success"] = self._last_success
        info["language"] = self.task.language
        return self._filter_obs(obs), reward, done, info

    def _check_success(self) -> bool:
        return _predicates.evaluate(
            self.sim,
            self._obj_body_id,
            self.task.success.predicate,
            dict(self.task.success.args),
        )

    def reward(self, action=None) -> float:
        return 1.0 if self._check_success() else 0.0

    def _filter_obs(self, raw: Dict[str, Any]) -> Dict[str, Any]:
        images: Dict[str, np.ndarray] = {}
        for cam in self.task.cameras:
            if f"{cam}_image" in raw:
                images[cam] = raw[f"{cam}_image"]
            if self.task.camera_depth and f"{cam}_depth" in raw:
                images[f"{cam}_depth"] = raw[f"{cam}_depth"]
        proprio: Dict[str, np.ndarray] = {}
        for idx in (0, 1):
            for suffix in ("joint_pos", "joint_vel", "eef_pos", "eef_quat",
                           "gripper_qpos"):
                src = f"robot{idx}_{suffix}"
                if src in raw:
                    proprio[f"robot{idx}_{suffix}"] = np.asarray(raw[src], dtype=np.float32)
        return {"images": images, "proprio": proprio}

    @property
    def language_instruction(self) -> str:
        return self.task.language

    @property
    def action_dim(self) -> int:
        return sum(r.action_dim for r in self.robots)
