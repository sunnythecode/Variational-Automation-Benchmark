"""Port the legacy BDDL+pruned_init suites to self-contained VAB YAMLs.

Loads BDDL files from a directory tree, parses them, samples N init poses
per task within the region bounds declared in the BDDL, then emits one
YAML per task into the output directory.

We *cannot* recover the exact original .pruned_init values (the pickles
reference numpy._core which crashes our numpy 1.26). Instead we sample
uniform-random (x, y) within each region's declared range, which keeps
the spatial-coverage semantics of the original suite intact even if not
bit-identical.

Usage:
    python tools/port_legacy_bddl.py \\
        --src-rev df2e536^ \\
        --suites libero_object_target_pos_var20x20 ... \\
        --out tasks/ \\
        --n-inits 10 \\
        --seed 0
"""
from __future__ import annotations

import argparse
import re
import subprocess
from collections import OrderedDict
from pathlib import Path
from typing import Dict, List, Tuple

import yaml

# ---------------------------------------------------------------------------
# Arena name from the BDDL `(problem <Name>)` -> our YAML `arena.name`.
PROBLEM_TO_ARENA = {
    "libero_floor_manipulation": "floor",
    "libero_living_room_tabletop_manipulation": "living_room",
    "libero_coffee_table_manipulation": "coffee_table",
    "libero_kitchen_tabletop_manipulation": "kitchen",
    "libero_study_tabletop_manipulation": "study",
    "libero_tabletop_manipulation": "table",
    "libero_kitchen_popcorn_production": "kitchen",
}

# Per-arena workspace surface z (where objects rest). Matches values from
# the deleted envs/problems/libero_*_manipulation.py constants.
ARENA_SURFACE_Z = {
    "floor": -0.025,             # floor_offset (0,0,-0.035) + nominal pad
    "living_room": 0.43,         # table_offset (0,0,0.41) + ~0.02
    "coffee_table": 0.43,
    "kitchen": 0.93,             # table_offset (0,0,0.90) + ~0.03
    "study": 0.90,               # table_offset (-0.2,0,0.867)
    "table": 0.93,
}

# Per-arena (x, y) workspace center -- in the new env, the arena is placed at
# workspace_offset and inits are absolute world coords. The BDDL regions are
# relative to the table center. We add the table_offset to recover world coords.
ARENA_WORKSPACE_OFFSET = {
    "floor": (0.0, 0.0),
    "living_room": (0.0, 0.0),
    "coffee_table": (0.0, 0.0),
    "kitchen": (0.0, 0.0),
    "study": (-0.2, 0.0),
    "table": (0.0, 0.0),
}

# ---------------------------------------------------------------------------
# BDDL parsing

_PROBLEM_RE = re.compile(r"\(define\s+\(problem\s+([\w_]+)\)", re.IGNORECASE)
_LANG_RE = re.compile(r"\(:language\s+([^)]+)\)", re.IGNORECASE)
_REGIONS_BLOCK_RE = re.compile(r"\(:regions(.+?)\)\s*\(:fixtures", re.DOTALL | re.IGNORECASE)
_REGION_RE = re.compile(
    r"\(\s*([\w_]+)\s*"
    r"\(:target\s+([\w_]+)\s*\)"
    r"(?:\s*\(:ranges\s*\(\s*\(([\-\d\.eE+\s]+)\)\s*\)\s*\))?"
    r"(?:\s*\(:yaw_rotation\s*\(\s*\(([\-\d\.eE+\s]+)\)\s*\)\s*\))?",
    re.DOTALL,
)
_OBJECTS_BLOCK_RE = re.compile(r"\(:objects(.+?)\)\s*\(:obj_of_interest", re.DOTALL | re.IGNORECASE)
_OBJECT_LINE_RE = re.compile(r"([\w_]+)\s*-\s*([\w_]+)")
_INIT_BLOCK_RE = re.compile(r"\(:init(.+?)\)\s*\(:goal", re.DOTALL | re.IGNORECASE)
_INIT_FACT_RE = re.compile(r"\(\s*([\w_]+)\s+([\w_]+)\s+([\w_]+)\s*\)")
_GOAL_BLOCK_RE = re.compile(r"\(:goal(.+?)\)\s*\)\s*$", re.DOTALL | re.IGNORECASE)
_GOAL_PRED_RE = re.compile(r"\(\s*([\w_]+)\s+([\w_]+)(?:\s+([\w_]+))?\s*\)")


def _strip_inst_suffix(name: str) -> str:
    """`cream_cheese_1` -> `cream_cheese`."""
    m = re.match(r"^(.+?)_\d+$", name)
    return m.group(1) if m else name


def parse_bddl(text: str) -> dict:
    problem = _PROBLEM_RE.search(text)
    if not problem:
        raise ValueError("no problem name")
    problem_name = problem.group(1).lower()

    lang_m = _LANG_RE.search(text)
    language = lang_m.group(1).strip() if lang_m else ""

    regions: Dict[str, dict] = OrderedDict()
    regions_block_m = _REGIONS_BLOCK_RE.search(text)
    if regions_block_m:
        for m in _REGION_RE.finditer(regions_block_m.group(1)):
            name, target, ranges, yaw = m.group(1), m.group(2), m.group(3), m.group(4)
            entry = {"target": target}
            if ranges:
                entry["ranges"] = [float(x) for x in ranges.split()]
            if yaw:
                entry["yaw"] = [float(x) for x in yaw.split()]
            regions[name] = entry

    objects: Dict[str, str] = OrderedDict()
    objects_block_m = _OBJECTS_BLOCK_RE.search(text)
    if objects_block_m:
        for m in _OBJECT_LINE_RE.finditer(objects_block_m.group(1)):
            instance, category = m.group(1), m.group(2)
            objects[instance] = category

    init_facts: List[Tuple[str, str, str]] = []
    init_block_m = _INIT_BLOCK_RE.search(text)
    if init_block_m:
        for m in _INIT_FACT_RE.finditer(init_block_m.group(1)):
            init_facts.append((m.group(1).lower(), m.group(2), m.group(3)))

    goal_preds: List[Tuple[str, str, str]] = []
    goal_block_m = _GOAL_BLOCK_RE.search(text)
    if goal_block_m:
        for m in _GOAL_PRED_RE.finditer(goal_block_m.group(1)):
            pred = m.group(1).lower()
            if pred in {"and", "sequence"}:
                continue
            goal_preds.append((pred, m.group(2), m.group(3) or ""))

    return {
        "problem_name": problem_name,
        "language": language,
        "regions": regions,
        "objects": objects,
        "init_facts": init_facts,
        "goal_preds": goal_preds,
    }


# ---------------------------------------------------------------------------
# Init sampling

def sample_pose(rng, region: dict, arena: str, z: float | None = None) -> List[float]:
    """Sample a 7-vec [x,y,z, qx,qy,qz,qw] in world coords for a region."""
    ranges = region.get("ranges")
    if ranges is None:
        # Falls through for regions whose target is an object (e.g. contain_region
        # on basket_1). The caller never asks us to place such regions; only the
        # objects-on-table regions provide ranges.
        raise ValueError("region has no ranges")
    x_lo, y_lo, x_hi, y_hi = ranges
    x = rng.uniform(x_lo, x_hi)
    y = rng.uniform(y_lo, y_hi)
    ox, oy = ARENA_WORKSPACE_OFFSET[arena]
    wz = ARENA_SURFACE_Z[arena] if z is None else z
    yaw_low, yaw_high = region.get("yaw", [0.0, 0.0])
    yaw = rng.uniform(yaw_low, yaw_high)
    qx, qy, qz, qw = 0.0, 0.0, float(__import__("math").sin(yaw / 2)), float(__import__("math").cos(yaw / 2))
    return [x + ox, y + oy, wz, qx, qy, qz, qw]


# ---------------------------------------------------------------------------
# Goal predicate translation

def translate_goal(goal_preds: List[Tuple[str, str, str]]) -> dict:
    """Map BDDL goal predicates onto our SuccessSpec registry."""
    if not goal_preds:
        raise ValueError("no goal predicates")
    pred, a, b = goal_preds[0]
    a = _strip_inst_suffix(a)
    if pred == "in":
        # (In cream_cheese_1 basket_1_contain_region) -> contained_in
        container = _strip_inst_suffix(re.sub(r"_contain_region$", "", b))
        return {"predicate": "contained_in", "args": {"obj": a, "container": container,
                                                     "xy_tol": 0.10, "z_low": -0.05, "z_high": 0.25}}
    if pred == "on":
        surface = _strip_inst_suffix(b)
        return {"predicate": "on_top_of", "args": {"obj": a, "surface": surface,
                                                  "xy_tol": 0.15, "z_min": 0.01, "z_max": 0.30}}
    if pred == "lifted":
        return {"predicate": "lifted_above", "args": {"obj": a, "z_min": 0.10}}
    raise ValueError(f"unsupported goal predicate: {pred}")


# ---------------------------------------------------------------------------
# YAML emission

def bddl_to_yaml(text: str, *, suite: str, task_name: str, n_inits: int, rng) -> dict:
    parsed = parse_bddl(text)
    arena = PROBLEM_TO_ARENA[parsed["problem_name"]]

    obj_list = [
        {"id": _strip_inst_suffix(inst), "asset": cat}
        for inst, cat in parsed["objects"].items()
    ]

    # Build placement plan: instance -> region with ranges
    placement: Dict[str, str] = {}
    for pred, inst, region_full in parsed["init_facts"]:
        if pred != "on":
            continue
        # region_full is e.g. "floor_target_object_region"; strip arena prefix
        if region_full.startswith(arena + "_"):
            region_key = region_full[len(arena) + 1:]
        else:
            region_key = region_full
        if region_key in parsed["regions"] and "ranges" in parsed["regions"][region_key]:
            placement[inst] = region_key

    inits: List[dict] = []
    for _ in range(n_inits):
        init = {}
        for inst, region_key in placement.items():
            init[_strip_inst_suffix(inst)] = [
                round(v, 6) for v in sample_pose(rng, parsed["regions"][region_key], arena)
            ]
        inits.append(init)

    return {
        "id": f"{suite}.{task_name}",
        "language": parsed["language"],
        "arena": {"name": arena},
        "robot": {"name": "panda", "controller": "OSC_POSE"},
        "cameras": ["agentview", "robot0_eye_in_hand"],
        "camera_height": 128,
        "camera_width": 128,
        "camera_depth": False,
        "objects": obj_list,
        "inits": inits,
        "default_init_index": 0,
        "success": translate_goal(parsed["goal_preds"]),
        "horizon": 500,
        "metadata": {"suite": suite, "source": "ported_from_bddl"},
    }


# ---------------------------------------------------------------------------
# Git reading

def git_show(rev: str, path: str) -> str:
    return subprocess.check_output(["git", "show", f"{rev}:{path}"], text=True)


def list_bddl_in_suite(rev: str, suite_dir: str) -> List[str]:
    out = subprocess.check_output(["git", "show", f"{rev}:{suite_dir}"], text=True)
    files = []
    for line in out.splitlines():
        line = line.strip()
        if line.endswith(".bddl"):
            files.append(line)
    return files


def main():
    import sys, random

    ap = argparse.ArgumentParser()
    ap.add_argument("--src-rev", default="df2e536^",
                   help="git rev to read BDDLs from (the commit before deletion)")
    ap.add_argument("--src-dir", default="libero/libero/bddl_files",
                   help="path under the rev where suites live")
    ap.add_argument("--suites", nargs="+", required=True)
    ap.add_argument("--out", default="tasks", help="output dir")
    ap.add_argument("--n-inits", type=int, default=10)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    rng = random.Random(args.seed)

    out_root = Path(args.out)
    written = 0
    for suite in args.suites:
        suite_path = f"{args.src_dir}/{suite}"
        bddls = list_bddl_in_suite(args.src_rev, suite_path)
        if not bddls:
            print(f"  [skip] {suite}: no bddl files", file=sys.stderr)
            continue
        out_dir = out_root / suite
        out_dir.mkdir(parents=True, exist_ok=True)
        for bddl_filename in bddls:
            text = git_show(args.src_rev, f"{suite_path}/{bddl_filename}")
            task_name = Path(bddl_filename).stem
            try:
                yaml_obj = bddl_to_yaml(text, suite=suite, task_name=task_name,
                                       n_inits=args.n_inits, rng=rng)
            except Exception as e:
                print(f"  [fail] {suite}/{task_name}: {e}", file=sys.stderr)
                continue
            out_path = out_dir / f"{task_name}.yaml"
            with out_path.open("w") as f:
                yaml.safe_dump(yaml_obj, f, sort_keys=False, default_flow_style=None)
            written += 1
            print(f"  [ok] {out_path}")
    print(f"[done] wrote {written} task YAMLs to {out_root}/")


if __name__ == "__main__":
    main()
