from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field, PrivateAttr, field_validator, model_validator


class ArenaSpec(BaseModel):
    name: str
    scene_xml: Optional[str] = None
    scene_properties: Dict[str, Any] = Field(default_factory=dict)


class RobotSpec(BaseModel):
    name: str = "panda"
    controller: str = "OSC_POSE"


class ObjectSpec(BaseModel):
    id: str
    asset: str


class SuccessSpec(BaseModel):
    predicate: str
    args: Dict[str, Any] = Field(default_factory=dict)


Pose7 = List[float]


class Task(BaseModel):
    id: str
    language: str
    arena: ArenaSpec
    robot: RobotSpec = Field(default_factory=RobotSpec)
    robots: Optional[List[RobotSpec]] = None     # bimanual: list of 2; else None
    cameras: List[str] = Field(default_factory=lambda: ["agentview", "robot0_eye_in_hand"])
    camera_height: int = 128
    camera_width: int = 128
    camera_depth: bool = False
    objects: List[ObjectSpec]
    inits: List[Dict[str, Pose7]]
    default_init_index: int = 0
    success: SuccessSpec
    horizon: int = 500
    metadata: Dict[str, Any] = Field(default_factory=dict)

    _source_path: Optional[Path] = PrivateAttr(default=None)

    @field_validator("inits")
    @classmethod
    def _check_inits_nonempty(cls, v):
        if not v:
            raise ValueError("inits must contain at least one initial state")
        for i, init in enumerate(v):
            for obj_id, pose in init.items():
                if len(pose) != 7:
                    raise ValueError(
                        f"inits[{i}][{obj_id}] must be a 7-vector [x,y,z, qx,qy,qz,qw],"
                        f" got length {len(pose)}"
                    )
        return v

    @model_validator(mode="after")
    def _cross_check(self):
        if self.default_init_index >= len(self.inits):
            raise ValueError(
                f"default_init_index={self.default_init_index} out of range for"
                f" {len(self.inits)} inits"
            )
        declared_ids = {o.id for o in self.objects}
        for i, init in enumerate(self.inits):
            for obj_id in init.keys():
                if obj_id not in declared_ids:
                    raise ValueError(
                        f"inits[{i}] references unknown object id {obj_id!r};"
                        f" declared objects: {sorted(declared_ids)}"
                    )
        return self

    @property
    def n_inits(self) -> int:
        return len(self.inits)

    def make_env(self, **overrides):
        if self.robots and len(self.robots) >= 2:
            from .bimanual_env import VABBimanualEnv

            return VABBimanualEnv(task=self, **overrides)
        from .env import VABEnv

        return VABEnv(task=self, **overrides)

    model_config = {"arbitrary_types_allowed": True}
