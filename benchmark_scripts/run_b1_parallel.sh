#!/usr/bin/env bash
# Launch 5 parallel benchmark workers, one per GPU, 2 tasks each.
# Usage: bash run_b1_parallel.sh [curobo|alt]   (default: curobo)
set -e
PLANNER="${1:-curobo}"
REPO="$(cd "$(dirname "$0")/.." && pwd)"
VENV="$REPO/.venv/bin/activate"
OUT="$REPO/videos/b1"
SCRIPT="$REPO/benchmark_scripts/run_b1_benchmark.py"

mkdir -p "$OUT"

echo "Launching 5 workers (planner=$PLANNER)..."
for GPU in 0 1 2 3 4; do
    START=$((GPU * 2))
    END=$((START + 2))
    LOG="$OUT/run_${PLANNER}_gpu${GPU}.log"
    CUDA_VISIBLE_DEVICES=$GPU bash -c "
        source '$VENV' && \
        python '$SCRIPT' \
            --planner ${PLANNER} \
            --task-slice ${START}:${END} \
            --shard-id ${GPU}
    " > "$LOG" 2>&1 &
    echo "  GPU $GPU → tasks ${START}:${END}  PID=$!  log=$LOG"
done

echo ""
echo "All workers launched. Monitor with:"
echo "  tail -f $OUT/run_${PLANNER}_gpu*.log | grep 'init\|SR=\|=== Task'"
echo ""
echo "When done, merge results with:"
echo "  source .venv/bin/activate && python benchmark_scripts/run_b1_merge.py --planner ${PLANNER}"
