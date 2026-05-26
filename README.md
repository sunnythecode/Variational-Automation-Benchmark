# Variational Automation Benchmark (VAB)

Self-contained-YAML manipulation benchmark. One file per task: arena, robot,
objects, initial states, success predicate, language. No BDDL, no sidecar
init file, no privileged information surfaced to the agent.

## Suites

52 tasks across 7 suites under `tasks/`:

| Suite | Tasks × Seeds | Description |
| --- | --- | --- |
| `libero_object_target_pos_var20x20` | 10 × 50 | LIBERO-Object pick-and-place with a 20×20 grid of target-basket positions to measure spatial generalisation. |
| `libero_object_target_permutation_variance` | 10 × 50 | Target basket and distractor objects permuted across the workspace. |
| `libero_object_target_basket_swap_variance` | 10 × 50 | Target basket swapped with distractor basket on each trial. |
| `libero_object_all_variance` | 10 × 50 | Combined variance suite: position, permutation, and basket-swap perturbations applied jointly. |
| `libero_object_packing` | 10 × 50 | Multi-item packing variant of `libero_object_all_variance` — every grocery item on the floor must end up in the basket. Uses the stateful `pack_all_into` predicate (delivered items teleport to a far graveyard pose so they cannot regress; success when all 6 delivered). Reports both `success` (binary) and `completion_rate` (`delivered/6`). |
| `libero_popcorn_production` | 1 × 50 | Single-task multi-stage suite — place frypan on stove → turn on → turn off → remove. **v1 success check is single-stage (`on_top_of`)**; full sequence pending stage-aware `SuccessSpec`. |
| `libero_crate_washing` | 1 × 5 | Bimanual single-task suite — two Franka Pandas lift the top crate of an 11-crate stack onto a washing-machine table. **v1 success check is `lifted_above`**; full `Lifted → Placed` sequence pending. |

Per-object init poses were ported from the upstream `.pruned_init` blobs
(see [Porting from the legacy format](#porting-from-the-legacy-format))
so trials match the original benchmark verbatim.

## Quick start

```bash
git clone https://github.com/ehehee/Variational-Automation-Benchmark.git
cd Variational-Automation-Benchmark
pip install -r requirements.txt
pip install -e .
```

```python
import numpy as np
from libero.vab import load_task

task = load_task("tasks/libero_object_target_pos_var20x20/pick_up_the_cream_cheese_and_place_it_in_the_basket.yaml")
env  = task.make_env()

for i in range(task.n_inits):              # 50 seeds for single-arm tasks
    obs = env.reset(init_index=i)
    # obs.keys() == {"images", "proprio"} -- nothing else.
    for _ in range(50):
        obs, reward, done, info = env.step(np.zeros(env.action_dim))
    print(f"seed {i}: success={info['success']}")
env.close()
```

## Observation contract (strict)

```python
obs = {
    "images":  {camera_name: uint8[H,W,3], ...},   # one per camera in YAML
    "proprio": {
        "joint_pos":    float32[7],
        "joint_vel":    float32[7],
        "eef_pos":      float32[3],
        "eef_quat":     float32[4],   # xyzw
        "gripper_qpos": float32[2],
    },
}
```

For bimanual tasks (e.g. `crate_washing`) proprio keys are
`robot0_*` and `robot1_*`; `action_dim` is 14.

No object names, no ground-truth poses, no segmentation masks. Success is
returned only as `info["success"]: bool` from `env.step`. Multi-target
predicates (e.g. `pack_all_into`) additionally surface partial credit via
`info["completion_rate"]: float` (0..1) and
`env.delivery_progress() -> tuple[int, int] | None`.

## Task YAML

```yaml
id: libero_object_target_pos_var20x20.pick_up_the_cream_cheese_and_place_it_in_the_basket
language: "Pick the cream cheese and place it in the basket"

arena:
  name: floor              # floor | living_room | kitchen | study | coffee_table | table | crate_washing
  # scene_xml: optional override (path under libero/libero/assets/)
  # scene_properties: { floor_style: ..., wall_style: ... }   # optional

robot:
  name: panda
  controller: OSC_POSE

# For bimanual tasks (crate_washing) use `robots:` (list of 2) instead of `robot:`.

cameras: [agentview, robot0_eye_in_hand]
camera_height: 128
camera_width: 128
camera_depth: false

objects:
  - { id: cream_cheese, asset: cream_cheese }   # asset = registered OBJECTS_DICT key
  - { id: basket,       asset: basket }
  - ...

inits:                                          # 50 seeds for single-arm, 5 for crate
  - cream_cheese: [x, y, z, qx, qy, qz, qw]     # xyzw quaternion
    basket:       [x, y, z, qx, qy, qz, qw]
    ...
  - ...
default_init_index: 0

success:
  predicate: contained_in
  args: { obj: cream_cheese, container: basket, xy_tol: 0.10, z_low: -0.05, z_high: 0.25 }

horizon: 500
metadata: { suite: libero_object_target_pos_var20x20, source: ported_from_pruned_init }
```

## Success predicates

Registered in `libero/libero/vab/predicates.py`. Add new ones by appending to
the `PREDICATES` registry; each takes `(sim, body_ids, **args) -> bool`.

| predicate | required args | optional args |
| --- | --- | --- |
| `contained_in`  | `obj`, `container`            | `xy_tol`, `z_low`, `z_high` |
| `on_top_of`     | `obj`, `surface`              | `xy_tol`, `z_min`, `z_max` |
| `near`          | `obj_a`, `obj_b`              | `threshold` |
| `oriented_like` | `obj`, `quat` (xyzw)          | `tol_deg` |
| `lifted_above`  | `obj`, `z_min`                | — |
| `pack_all_into` | `objs` (list), `container`    | `xy_tol`, `z_low`, `z_high` |

`pack_all_into` is **stateful**: it tracks delivery on `sim._packing_state`
and teleports each delivered object to a far graveyard pose so it cannot
regress (matches the retired LIBERO-PosVar `Libero_Grocery_Packing`
behavior). Partial progress is exposed via
`env.delivery_progress() -> (delivered, total)` and `info["completion_rate"]`.
State is reset on every `env.reset()`. Predicates that need joint-level
state mutations can opt in by declaring a `joint_names` kwarg —
`evaluate` introspects the signature and forwards the per-object
free-joint name table only when requested.

## Rendering / inspection

```bash
# Single task at a chosen seed -> benchmark_tasks/<task_id>__init<n>.png
python benchmark_scripts/render_single_task.py \
    --task tasks/libero_popcorn_production/kitchen_scene9_popcorn_production.yaml \
    --init-index 0

# Every task × every seed -> videos/vab_all_trials.mp4 (42 MB, ~4.5 min @ 800x600)
python benchmark_scripts/render_all_tasks_video.py
```

A pre-rendered MP4 covering all 2,052 trials is checked in at
`videos/vab_all_trials.mp4`.

## Smoke test

```bash
pytest tests/test_smoke.py -v
```

## Layout

```
benchmark_scripts/
├── render_single_task.py          # one task × one seed -> PNG
└── render_all_tasks_video.py      # every task × every seed -> MP4
libero/libero/
├── assets/                         # scene XMLs, textures, scanned/CAD meshes (runtime-essential)
├── envs/                           # robosuite primitives: arenas/, robots/, objects/
└── vab/                            # YAML loader + strict-obs env
    ├── schema.py
    ├── loader.py
    ├── env.py                      # single-arm
    ├── bimanual_env.py             # crate_washing
    ├── predicates.py
    └── _arena_table.py
tasks/                              # 52 task YAMLs across 7 suites
tests/test_smoke.py
tools/                              # one-shot porting tools
├── port_legacy_bddl.py             # BDDL -> YAML scaffold (no inits)
├── apply_extracted_inits.py        # patch YAMLs with extracted world poses
└── record_video.py                 # 2-seed-per-task sampler video
videos/                             # pre-rendered MP4s
```

## Porting from the legacy format

Original tasks live as `(scene.bddl, init_states.pruned_init)` pairs under
upstream LIBERO-PosVar. The two-step recovery pipeline (used once to
generate `tasks/`):

1. **Read `.pruned_init`** — the pickles reference `numpy._core` which
   segfaults under numpy 1.26. Convert with `uv` in an ephemeral env:
   ```bash
   uv run --python 3.11 --with 'numpy>=2.0,<2.3' --with 'torch>=2.6' \
       tools/dump_pruned_init.py        # writes NPY per task
   ```
2. **Extract world poses** — replay each NPY state through the legacy
   `ControlEnv` (still installed editably as LIBERO-PosVar) and read
   per-object `sim.data.body_xpos` + `body_xquat`. JSON dropped per task.
3. **Patch YAMLs** —
   ```bash
   python tools/apply_extracted_inits.py
   ```

## Roadmap

- Stage-aware `SuccessSpec` so `popcorn_production` and `crate_washing`
  can express their full multi-stage goals (e.g. `Sequence(...)`).
- Action spaces beyond `OSC_POSE`.

## License

MIT.
