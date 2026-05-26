"""Arena name -> robosuite primitives table.

Constants extracted from the deleted ``envs/problems/libero_*_manipulation.py``
subclasses. Each entry tells the loader which arena class to instantiate, what
default scene XML / textures to use, what the robot's base offset should be, and
which robot-mount variant is appropriate.
"""
from __future__ import annotations

from typing import Any, Dict

from libero.envs.arenas import (
    CoffeeTableArena,
    EmptyArena,
    KitchenTableArena,
    LivingRoomTableArena,
    StudyTableArena,
    TableArena,
)

ARENA_TABLE: Dict[str, Dict[str, Any]] = {
    "table": {
        "arena_cls": TableArena,
        "scene_xml": "scenes/libero_tabletop_base_style.xml",
        "table_full_size": (1.0, 1.2, 0.05),
        "table_offset": (0.0, 0.0, 0.90),
        "z_offset": -0.04,
        "robot_prefix": "Mounted",
        "base_xpos_key": "table",
        "scene_properties": {"floor_style": "light-gray", "wall_style": "light-gray-plaster"},
        "agentview_pos": [0.6586131746834771, 0.0, 1.6103500240372423],
        "agentview_quat": [0.6380177736282349, 0.3048497438430786, 0.30484986305236816, 0.6380177736282349],
    },
    "kitchen": {
        "arena_cls": KitchenTableArena,
        "scene_xml": "scenes/libero_kitchen_tabletop_base_style.xml",
        "table_full_size": (1.0, 1.2, 0.05),
        "table_offset": (0.0, 0.0, 0.90),
        "z_offset": -0.04,
        "robot_prefix": "Mounted",
        "base_xpos_key": "kitchen_table",
        "scene_properties": {"floor_style": "gray-ceramic", "wall_style": "yellow-linen"},
        "agentview_pos": [0.6586131746834771, 0.0, 1.6103500240372423],
        "agentview_quat": [0.6380177736282349, 0.3048497438430786, 0.30484986305236816, 0.6380177736282349],
    },
    "study": {
        "arena_cls": StudyTableArena,
        "scene_xml": "scenes/libero_study_base_style.xml",
        "table_full_size": (1.0, 1.2, 0.05),
        "table_offset": (-0.2, 0.0, 0.867),
        "z_offset": -0.04,
        "robot_prefix": "Mounted",
        "base_xpos_key": "study_table",
        "scene_properties": {"floor_style": "light-gray", "wall_style": "light-gray-plaster"},
        "agentview_pos": [0.4586131746834771, 0.0, 1.6103500240372423],
        "agentview_quat": [0.6380177736282349, 0.3048497438430786, 0.30484986305236816, 0.6380177736282349],
    },
    "living_room": {
        "arena_cls": LivingRoomTableArena,
        "scene_xml": "scenes/libero_living_room_tabletop_base_style.xml",
        "table_full_size": (0.70, 1.6, 0.024),
        "table_offset": (0.0, 0.0, 0.41),
        "z_offset": -0.014,
        "robot_prefix": "OnTheGround",
        "base_xpos_key": "living_room_table",
        "scene_properties": {"floor_style": "wood-plank", "wall_style": "light-gray-plaster"},
        "agentview_pos": [0.6065773716836134, 0.0, 0.96],
        "agentview_quat": [0.6182166934013367, 0.3432307541370392, 0.3432314395904541, 0.6182177066802979],
    },
    "coffee_table": {
        "arena_cls": CoffeeTableArena,
        "scene_xml": "scenes/libero_coffee_table_base_style.xml",
        "table_full_size": (0.70, 1.6, 0.024),
        "table_offset": (0.0, 0.0, 0.41),
        "z_offset": -0.014,
        "robot_prefix": "OnTheGround",
        "base_xpos_key": "coffee_table",
        "scene_properties": {"floor_style": "wood-plank", "wall_style": "light-gray-plaster"},
        "agentview_pos": [1.5, 0.0, 0.9],
        "agentview_quat": [0.56, 0.43, 0.43, 0.56],
    },
    "floor": {
        "arena_cls": EmptyArena,
        "scene_xml": "scenes/libero_floor_base_style.xml",
        "table_full_size": None,
        "table_offset": (0.0, 0.0, -0.035),
        "z_offset": -0.025,
        "robot_prefix": "OnTheGround",
        "base_xpos_key": "empty",
        "scene_properties": {"floor_style": "light-gray", "wall_style": "light-gray-plaster"},
        "agentview_pos": [0.8965773716836134, 0.0, 0.65],
        "agentview_quat": [0.6182166934013367, 0.3432307541370392, 0.3432314395904541, 0.6182177066802979],
    },
}


def resolve_arena(name: str) -> Dict[str, Any]:
    if name not in ARENA_TABLE:
        raise KeyError(
            f"Unknown arena {name!r}. Available: {sorted(ARENA_TABLE.keys())}"
        )
    return ARENA_TABLE[name]
