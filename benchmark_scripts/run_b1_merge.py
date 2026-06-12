#!/usr/bin/env python3
"""Merge shard CSVs from parallel benchmark workers → timings.csv + summary.json.

Usage:
  python run_b1_merge.py              # merges curobo shards (timings_curobo_gpu*.csv)
  python run_b1_merge.py --planner alt  # merges alt shards (timings_alt_gpu*.csv)
"""
import argparse, csv, json, sys
from pathlib import Path
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument("--planner", default="curobo", choices=["curobo", "alt"])
args = parser.parse_args()

OUT_DIR = Path(__file__).resolve().parent.parent / "videos" / "b1"

shards = sorted(OUT_DIR.glob(f"timings_{args.planner}_gpu*.csv"))
if not shards:
    print("No shard CSVs found in", OUT_DIR); sys.exit(1)

all_rows = []
for s in shards:
    with open(s) as f:
        all_rows.extend(list(csv.DictReader(f)))

# Sort by task then init_index for a clean final CSV
all_rows.sort(key=lambda r: (r["task"], int(r["init_index"])))

merged_csv = OUT_DIR / f"timings_{args.planner}.csv"
with open(merged_csv, "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=all_rows[0].keys())
    w.writeheader()
    w.writerows(all_rows)
print(f"Merged {len(all_rows)} rows → {merged_csv}")

# Summary
def _f(key):
    return [float(r[key]) for r in all_rows if r.get(key, "") not in ("", "0.0")]

n = len(all_rows)
n_ok = sum(int(r["success"]) for r in all_rows)
summary = {
    "total_episodes": n,
    "n_success": n_ok,
    "success_rate": n_ok / n if n else 0.0,
    "avg_plan_ms": float(np.mean(_f("total_plan_ms"))),
    "std_plan_ms": float(np.std(_f("total_plan_ms"))),
    "avg_exec_ms": float(np.mean(_f("total_exec_ms"))),
    "std_exec_ms": float(np.std(_f("total_exec_ms"))),
    "avg_wall_s":  float(np.mean(_f("wall_ms"))) / 1000,
    "std_wall_s":  float(np.std(_f("wall_ms"))) / 1000,
    "avg_steps":   float(np.mean(_f("total_steps"))),
}
for i in range(7):
    pv = _f(f"seg{i}_plan_ms")
    ev = _f(f"seg{i}_exec_ms")
    summary[f"seg{i}_avg_plan_ms"] = float(np.mean(pv)) if pv else 0.0
    summary[f"seg{i}_avg_exec_ms"] = float(np.mean(ev)) if ev else 0.0

summary_path = OUT_DIR / f"summary_{args.planner}.json"
summary_path.write_text(json.dumps(summary, indent=2))

print(f"\n{'='*60}")
print(f"BENCHMARK SUMMARY")
print(f"  Episodes:      {n}")
print(f"  Success rate:  {summary['success_rate']*100:.1f}%  ({n_ok}/{n})")
print(f"  Avg plan time: {summary['avg_plan_ms']:.0f} ± {summary['std_plan_ms']:.0f} ms")
print(f"  Avg exec time: {summary['avg_exec_ms']:.0f} ± {summary['std_exec_ms']:.0f} ms")
print(f"  Avg wall time: {summary['avg_wall_s']:.1f} ± {summary['std_wall_s']:.1f} s")
print(f"  Avg steps:     {summary['avg_steps']:.0f}")
print(f"\n  Per-segment plan (avg ms):")
for i in range(7):
    print(f"    seg{i}: plan={summary[f'seg{i}_avg_plan_ms']:.0f}ms  exec={summary[f'seg{i}_avg_exec_ms']:.0f}ms")
print(f"\n  Results: {merged_csv}")
print(f"  Summary: {summary_path}")
print('='*60)
