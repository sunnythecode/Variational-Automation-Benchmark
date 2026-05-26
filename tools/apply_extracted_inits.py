"""Apply extracted world-pose inits (under /tmp/vab_init_world_poses/) to the
task YAMLs under tasks/. Replaces each YAML's `inits:` block with the per-
object 7-vec poses recovered from the original .pruned_init states.

Run from the VAB repo root.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import yaml

REPO = Path(__file__).resolve().parents[1]
INIT_ROOT = Path("/tmp/vab_init_world_poses")
TASKS_ROOT = REPO / "tasks"

# Suites whose YAMLs we mutate. crate_washing has no movable objects in
# its YAML (the crates are baked into the MJCF), so we skip it.
SUITES = [
    "libero_object_target_pos_var20x20",
    "libero_object_target_permutation_variance",
    "libero_object_target_basket_swap_variance",
    "libero_object_all_variance",
    "libero_popcorn_production",
]


def apply_one(yaml_path: Path, json_path: Path) -> bool:
    if not json_path.exists():
        print(f"  [skip] no json for {yaml_path}")
        return False
    data = yaml.safe_load(yaml_path.read_text())
    extracted = json.loads(json_path.read_text())

    yaml_obj_ids = [o["id"] for o in data["objects"]]
    extracted_obj_ids = set(extracted["obj_ids"])

    missing_in_yaml = extracted_obj_ids - set(yaml_obj_ids)
    missing_in_extracted = set(yaml_obj_ids) - extracted_obj_ids
    if missing_in_yaml:
        print(f"  [warn] {yaml_path.name}: extracted has objects not in YAML: {missing_in_yaml}")
    if missing_in_extracted:
        print(f"  [warn] {yaml_path.name}: YAML has objects not in extracted: {missing_in_extracted}")

    new_inits: list[dict[str, list[float]]] = []
    for init in extracted["inits"]:
        row: dict[str, list[float]] = {}
        for obj_id in yaml_obj_ids:
            if obj_id in init:
                row[obj_id] = init[obj_id]
        new_inits.append(row)
    data["inits"] = new_inits
    data["default_init_index"] = 0
    # Update metadata to reflect provenance.
    md = data.setdefault("metadata", {})
    md["source"] = "ported_from_pruned_init"
    md.pop("v1_note", None)
    md["n_inits"] = len(new_inits)
    yaml_path.write_text(
        yaml.safe_dump(data, sort_keys=False, default_flow_style=None)
    )
    return True


def main():
    written = 0
    for suite in SUITES:
        json_dir = INIT_ROOT / suite
        yaml_dir = TASKS_ROOT / suite
        if not yaml_dir.exists():
            print(f"[skip] no yaml dir for {suite}")
            continue
        for yaml_path in sorted(yaml_dir.glob("*.yaml")):
            stem = yaml_path.stem
            # popcorn stem is lower-case; JSON keeps original BDDL casing.
            candidates = [json_dir / f"{stem}.json",
                          json_dir / f"{stem.upper()}.json",
                          json_dir / f"KITCHEN_SCENE9_popcorn_production.json"
                          if "popcorn" in suite else None]
            json_path = next((c for c in candidates if c and c.exists()), None)
            if json_path is None:
                # try case-insensitive
                hits = [p for p in json_dir.glob("*.json")
                        if p.stem.lower() == stem.lower()]
                json_path = hits[0] if hits else None
            if json_path is None:
                print(f"  [miss] {yaml_path.name}: no json")
                continue
            if apply_one(yaml_path, json_path):
                written += 1
                print(f"  [ok] {yaml_path.relative_to(REPO)}  <- {json_path.name}")
    print(f"[done] updated {written} YAMLs")


if __name__ == "__main__":
    main()
