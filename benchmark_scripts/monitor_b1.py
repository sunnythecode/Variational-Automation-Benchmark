#!/usr/bin/env python3
"""Print benchmark progress every 5 seconds by reading shard CSVs live."""
import csv, glob, time
from pathlib import Path

OUT_DIR = Path(__file__).resolve().parent.parent / "videos" / "b1"
TOTAL = 500

while True:
    shards = sorted(OUT_DIR.glob("timings_gpu*.csv"))
    rows = []
    for s in shards:
        try:
            with open(s) as f:
                rows.extend(list(csv.DictReader(f)))
        except Exception:
            pass

    done = len(rows)
    ok   = sum(int(r["success"]) for r in rows)
    sr   = 100.0 * ok / done if done else 0.0
    pct  = 100.0 * done / TOTAL

    print(f"\r[{time.strftime('%H:%M:%S')}]  {done:>3}/{TOTAL} episodes  "
          f"({pct:5.1f}%)  SR={sr:.1f}%  ({ok} success)", end="", flush=True)

    if done >= TOTAL:
        print("\nDone!")
        break
    time.sleep(5)
