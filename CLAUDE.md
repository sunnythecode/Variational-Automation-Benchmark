# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Setup

```bash
uv venv .venv --python 3.10
source .venv/bin/activate
uv pip install -r requirements.txt
uv pip install easydict termcolor   # undeclared deps required at runtime
uv pip install -e .
python .venv/lib/python3.10/site-packages/robosuite/scripts/setup_macros.py  # one-time
```

The package installs as `libero` from `libero/` (editable). `robosuite==1.4.0` is the pinned simulation backend. The `easydict` and `termcolor` packages are required but missing from `requirements.txt`. The robosuite macros script suppresses startup warnings and only needs to be run once per venv.

## Commands

```bash
# Run smoke tests
pytest tests/test_smoke.py -v

# Render a single task at a given seed -> benchmark_tasks/<task_id>__init<n>.png
python benchmark_scripts/render_single_task.py \
    --task tasks/libero_object_packing/pack_all_objects_v00.yaml \
    --init-index 0

# Render all tasks -> videos/vab_all_trials.mp4
python benchmark_scripts/render_all_tasks_video.py

# Derive a suite from an existing one (e.g. after editing source YAMLs)
python tools/build_object_packing.py
python tools/build_permutation_packing.py
```

## Architecture

### The YAML-first task model

Each task is a single self-contained YAML under `tasks/<suite>/<task>.yaml`. The schema is defined in `libero/libero/vab/schema.py` (Pydantic `Task` model). Loading is just:

```python
from libero.vab import load_task
task = load_task("tasks/...")
env = task.make_env()          # dispatches to VABEnv or VABBimanualEnv
```

`Task.make_env()` inspects `task.robots` (list, bimanual) vs `task.robot` (single) and returns the right env class.

### Env classes (`libero/libero/vab/`)

- **`env.py` — `VABEnv`**: single-arm tasks; subclasses robosuite `SingleArmEnv`. Enforces the strict observation contract: only `{"images": {cam: uint8[H,W,3]}, "proprio": {joint_pos, joint_vel, eef_pos, eef_quat, gripper_qpos}}`. All other robosuite observables are disabled in `_setup_observables`.
- **`bimanual_env.py` — `VABBimanualEnv`**: crate_washing only; subclasses `TwoArmEnv`. Proprio keys are `robot0_*` / `robot1_*`. No movable objects — the arena MJCF bakes in the crate stack.
- Both envs call `_predicates.evaluate(...)` in `_check_success` and surface only `info["success"]` (bool) and `info["completion_rate"]` (float, multi-target only).

### Success predicates (`libero/libero/vab/predicates.py`)

Stateless predicates take `(sim, body_ids, **args) -> bool`. Stateful predicates (currently `pack_all_into`) declare a `joint_names` kwarg; `evaluate()` introspects the signature and forwards it. `pack_all_into` stores its delivery set on `sim._packing_state` and teleports delivered objects to a graveyard pose `(50, 50, 5)` so they cannot regress. Add new predicates with `@register("name")`.

### Arena mapping (`libero/libero/vab/_arena_table.py`)

`ARENA_TABLE` maps arena name strings (`"floor"`, `"table"`, `"kitchen"`, `"study"`, `"living_room"`, `"coffee_table"`, `"crate_washing"`) to dicts containing the arena class, scene XML path (relative to `libero/libero/assets/`), robot prefix (`Mounted` vs `OnTheGround`), camera poses, and table geometry. `resolve_arena(name)` is the only lookup; arenas not in the table raise `KeyError` immediately.

### Suite relationships

`libero_object_packing` is derived from `libero_object_all_variance` (same inits, different success predicate). `permutation_packing` is derived from `libero_object_target_permutation_variance`. Running `tools/build_object_packing.py` or `tools/build_permutation_packing.py` regenerates derived suites from their sources — run these after any init-pose update to keep suites in sync.

### Coordinate conventions

- Init poses in YAML: `[x, y, z, qx, qy, qz, qw]` (xyzw quaternion).
- MuJoCo `set_joint_qpos` expects `[x, y, z, qw, qx, qy, qz]` — `VABEnv._apply_init` handles this conversion.
- Predicate geometry uses `sim.data.body_xpos` and `sim.data.body_xquat` (MuJoCo wxyz convention internally).

### Smoke test note

`tests/test_smoke.py` looks for YAMLs under `seed_tasks/` (not `tasks/`). That directory doesn't exist in the repo yet, so the parametrized test currently collects zero cases. To test a task, load and step it directly in a script or add seed YAMLs there.
