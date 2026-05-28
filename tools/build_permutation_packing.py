"""Generate the permutation_packing task suite from libero_object_target_permutation_variance.

Sibling of :mod:`build_object_packing`. Where ``libero_object_packing``
derives from ``libero_object_all_variance`` (position + permutation +
basket-swap perturbations applied jointly), this suite isolates the
permutation axis: only target basket and distractor positions are
permuted across trials, every other initial-condition factor is held
fixed. Comparing the two suites measures how much of a packing policy's
robustness comes from generalising over object permutations alone vs.
the combined variance distribution.

Each output YAML reuses its source's arena, robot, cameras, objects,
and inits verbatim; only ``id``, ``language``, ``success``, ``horizon``,
and ``metadata`` change. Re-run after any init-pose refresh upstream to
keep the suites in sync.

Usage:
    python3 tools/build_permutation_packing.py
"""
from __future__ import annotations

from pathlib import Path

import yaml

REPO = Path(__file__).resolve().parents[1]
SRC = REPO / "tasks" / "libero_object_target_permutation_variance"
DST = REPO / "tasks" / "permutation_packing"

LANGUAGE = "Pick all the objects and place them in the basket"
CONTAINER = "basket"
HORIZON = 1200


def build_one(src_path: Path, dst_path: Path) -> None:
    raw = yaml.safe_load(src_path.read_text())
    obj_ids = [o["id"] for o in raw["objects"]]
    if CONTAINER not in obj_ids:
        raise ValueError(
            f"{src_path.name}: expected a '{CONTAINER}' object, got {obj_ids}"
        )
    objs = [o for o in obj_ids if o != CONTAINER]

    target_label = src_path.stem.replace("pick_up_the_", "").replace(
        "_and_place_it_in_the_basket", ""
    )
    raw["id"] = f"permutation_packing.pack_all_{target_label}_scene"
    raw["language"] = LANGUAGE
    raw["success"] = {
        "predicate": "pack_all_into",
        "args": {
            "objs": objs,
            "container": CONTAINER,
            "xy_tol": 0.10,
            "z_low": -0.05,
            "z_high": 0.25,
        },
    }
    raw["horizon"] = HORIZON
    md = dict(raw.get("metadata") or {})
    md["suite"] = "permutation_packing"
    md["source"] = f"derived_from_libero_object_target_permutation_variance/{src_path.name}"
    md["n_inits"] = len(raw["inits"])
    raw["metadata"] = md

    dst_path.write_text(yaml.safe_dump(raw, sort_keys=False))


def main() -> None:
    DST.mkdir(parents=True, exist_ok=True)
    sources = sorted(SRC.glob("*.yaml"))
    if not sources:
        raise FileNotFoundError(f"no source YAMLs under {SRC}")
    for src in sources:
        target_label = src.stem.replace("pick_up_the_", "").replace(
            "_and_place_it_in_the_basket", ""
        )
        dst = DST / f"pack_all_{target_label}_scene.yaml"
        build_one(src, dst)
        print(f"{src.name} -> {dst.relative_to(REPO)}")


if __name__ == "__main__":
    main()
