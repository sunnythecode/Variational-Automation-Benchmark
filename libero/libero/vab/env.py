"""VAB env: thin SingleArmEnv subclass that strips privileged observables.

Agent-facing observation is RGB(+depth) cameras + robot proprioception only.
Object names, poses, and segmentation masks are NEVER returned from
``reset`` / ``step``. Success checks happen inside ``_check_success`` using
ground-truth body positions, but that result surfaces only as a single
boolean in ``info['success']`` -- not as a pose dict.
"""
from __future__ import annotations

import os
from typing import Any, Dict, Optional

import numpy as np
import robosuite as suite
from robosuite.environments.manipulation.single_arm_env import SingleArmEnv
from robosuite.models.tasks import ManipulationTask

# Importing libero.envs triggers registration of OBJECTS_DICT, custom arenas,
# and the MountedPanda / OnTheGroundPanda robots into ROBOT_CLASS_MAPPING.
from libero import envs as _libero_envs  # noqa: F401
from libero.envs.arenas import (
    CoffeeTableArena,
    EmptyArena,
    KitchenTableArena,
    LivingRoomTableArena,
    StudyTableArena,
    TableArena,
)
from libero.envs.objects import get_object_fn

from . import predicates as _predicates
from ._arena_table import resolve_arena
from .schema import Task

_ASSETS_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "assets")
)


_PROPRIO_KEYS = (
    "robot0_joint_pos",
    "robot0_joint_vel",
    "robot0_eef_pos",
    "robot0_eef_quat",
    "robot0_gripper_qpos",
)


class VABEnv(SingleArmEnv):
    def __init__(
        self,
        task: Task,
        has_renderer: bool = False,
        has_offscreen_renderer: bool = True,
        control_freq: int = 20,
        render_gpu_device_id: int = -1,
        ignore_done: bool = True,
        hard_reset: bool = True,
        controller: Optional[str] = None,
        **kwargs,
    ):
        self.task = task
        self._arena_spec = resolve_arena(task.arena.name)

        scene_xml_rel = task.arena.scene_xml or self._arena_spec["scene_xml"]
        self._arena_xml_path = os.path.join(_ASSETS_DIR, scene_xml_rel)
        self._scene_properties = {
            **self._arena_spec["scene_properties"],
            **task.arena.scene_properties,
        }

        robot_class_name = f"{self._arena_spec['robot_prefix']}{task.robot.name.capitalize()}"
        controller_configs = suite.load_controller_config(
            default_controller=controller or task.robot.controller
        )

        self._mujoco_objects = []
        self._obj_body_id: Dict[str, int] = {}
        self._obj_joint_name: Dict[str, str] = {}
        self._last_success: bool = False
        self._current_init_index: int = task.default_init_index

        # Strict-obs policy: tell robosuite we do not want object observables.
        self.use_object_obs = False
        self.reward_scale = 1.0
        self.reward_shaping = False

        super().__init__(
            robots=[robot_class_name],
            env_configuration="default",
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

        base_offset = self.robots[0].robot_model.base_xpos_offset[
            self._arena_spec["base_xpos_key"]
        ]
        if callable(base_offset):
            xpos = base_offset(self._arena_spec["table_full_size"][0])
        else:
            xpos = base_offset
        self.robots[0].robot_model.set_base_xpos(xpos)

        arena_cls = self._arena_spec["arena_cls"]
        arena_kwargs: Dict[str, Any] = {
            "xml": self._arena_xml_path,
            **self._scene_properties,
        }
        if arena_cls in (TableArena, KitchenTableArena, StudyTableArena):
            arena_kwargs["table_full_size"] = self._arena_spec["table_full_size"]
            arena_kwargs["table_offset"] = self._arena_spec["table_offset"]
        if arena_cls is TableArena:
            arena_kwargs["table_friction"] = (0.6, 0.005, 0.0001)

        mujoco_arena = arena_cls(**arena_kwargs)
        mujoco_arena.set_origin([0, 0, 0])

        mujoco_arena.set_camera(
            camera_name="agentview",
            pos=self._arena_spec["agentview_pos"],
            quat=self._arena_spec["agentview_quat"],
        )

        self._mujoco_objects = []
        for spec in self.task.objects:
            cls = get_object_fn(spec.asset)
            self._mujoco_objects.append(cls(name=spec.id))

        self.model = ManipulationTask(
            mujoco_arena=mujoco_arena,
            mujoco_robots=[r.robot_model for r in self.robots],
            mujoco_objects=self._mujoco_objects,
        )

    def _setup_references(self):
        super()._setup_references()
        self._obj_body_id = {}
        self._obj_joint_name = {}
        for obj in self._mujoco_objects:
            self._obj_body_id[obj.name] = self.sim.model.body_name2id(obj.root_body)
            if not obj.joints:
                raise ValueError(
                    f"Object {obj.name!r} (asset {obj.__class__.__name__}) has no joints"
                    f" -- VAB v1 requires free-jointed objects so init poses are settable."
                )
            self._obj_joint_name[obj.name] = obj.joints[-1]

    def _setup_observables(self):
        observables = super()._setup_observables()
        for key in list(observables.keys()):
            if key in _PROPRIO_KEYS:
                # Robosuite leaves ``robot0_joint_pos`` ``active=False`` by
                # default (sin/cos encoding is preferred). Explicitly turn
                # it on so the proprio dict actually contains every entry
                # in ``_PROPRIO_KEYS``.
                observables[key].set_enabled(True)
                observables[key].set_active(True)
                continue
            if key.endswith("_image") or key.endswith("_depth"):
                continue
            observables[key].set_enabled(False)
            observables[key].set_active(False)
        return observables

    def _apply_init(self, init_index: int):
        if init_index >= len(self.task.inits):
            raise IndexError(
                f"init_index={init_index} out of range for {len(self.task.inits)} inits"
            )
        for obj_id, pose_xyzw in self.task.inits[init_index].items():
            x, y, z, qx, qy, qz, qw = pose_xyzw
            mujoco_pose = np.array([x, y, z, qw, qx, qy, qz], dtype=np.float64)
            self.sim.data.set_joint_qpos(self._obj_joint_name[obj_id], mujoco_pose)
        self.sim.forward()

    def reset(self, init_index: Optional[int] = None):
        super().reset()
        if init_index is None:
            init_index = self.task.default_init_index
        self._current_init_index = init_index
        self._apply_init(init_index)
        self._last_success = False
        # Drop any stateful-predicate scratch (e.g. pack_all_into's
        # delivery set). With hard_reset=True the sim is rebuilt and
        # this attribute is gone anyway; the pop keeps hard_reset=False
        # callers correct.
        self.sim.__dict__.pop("_packing_state", None)
        return self._filter_obs(self._get_observations(force_update=True))

    def step(self, action):
        obs, reward, done, info = super().step(action)
        self._last_success = bool(self._check_success())
        info = dict(info) if info else {}
        info["success"] = self._last_success
        info["language"] = self.task.language
        progress = self.delivery_progress()
        if progress is not None:
            delivered, total = progress
            info["completion_rate"] = (
                float(delivered) / float(total) if total else 0.0
            )
        else:
            info["completion_rate"] = 1.0 if self._last_success else 0.0
        return self._filter_obs(obs), reward, done, info

    def _check_success(self) -> bool:
        return _predicates.evaluate(
            self.sim,
            self._obj_body_id,
            self.task.success.predicate,
            dict(self.task.success.args),
            joint_names=self._obj_joint_name,
        )

    def delivery_progress(self) -> Optional[tuple]:
        """Return ``(delivered, total)`` for multi-target predicates.

        Read from ``sim._packing_state`` (populated by ``pack_all_into``).
        ``None`` for every predicate that doesn't track per-target
        progress, in which case callers should fall back to the binary
        success verdict.
        """
        state = getattr(self.sim, "_packing_state", None)
        if state is None:
            return None
        return (len(state["delivered"]), len(state["objs"]))

    def reward(self, action=None) -> float:
        return 1.0 if self._check_success() else 0.0

    def _filter_obs(self, raw: Dict[str, Any]) -> Dict[str, Any]:
        images: Dict[str, np.ndarray] = {}
        for cam in self.task.cameras:
            key_img = f"{cam}_image"
            if key_img in raw:
                images[cam] = raw[key_img]
            if self.task.camera_depth:
                key_depth = f"{cam}_depth"
                if key_depth in raw:
                    images[f"{cam}_depth"] = raw[key_depth]
        proprio: Dict[str, np.ndarray] = {}
        for src, dst in (
            ("robot0_joint_pos", "joint_pos"),
            ("robot0_joint_vel", "joint_vel"),
            ("robot0_eef_pos", "eef_pos"),
            ("robot0_eef_quat", "eef_quat"),
            ("robot0_gripper_qpos", "gripper_qpos"),
        ):
            if src in raw:
                proprio[dst] = np.asarray(raw[src], dtype=np.float32)
        return {"images": images, "proprio": proprio}

    @property
    def language_instruction(self) -> str:
        return self.task.language

    @property
    def action_dim(self) -> int:
        return self.robots[0].action_dim

    # ── Planning helpers ──────────────────────────────────────────────────────

    def get_object_pose(self, obj_id: str) -> tuple:
        """Live world-frame (position (3,), quaternion_wxyz (4,)) of obj_id."""
        body_id = self._obj_body_id[obj_id]
        return (
            np.array(self.sim.data.body_xpos[body_id], dtype=np.float64),
            np.array(self.sim.data.body_xquat[body_id], dtype=np.float64),
        )

    def get_eef_pose(self) -> tuple:
        """Live world-frame (position (3,), quaternion_xyzw (4,)) of the grip site."""
        from scipy.spatial.transform import Rotation
        eef_id = self.robots[0].eef_site_id
        pos = np.array(self.sim.data.site_xpos[eef_id], dtype=np.float64)
        mat = np.array(self.sim.data.site_xmat[eef_id], dtype=np.float64).reshape(3, 3)
        quat_xyzw = Rotation.from_matrix(mat).as_quat()
        return pos, quat_xyzw
