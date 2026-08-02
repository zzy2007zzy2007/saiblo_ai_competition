#!/bin/bash
# Run multiple sequential collection processes in parallel.
# Each process handles a chunk of games with different seed range.

OUTDIR="value_data_v2"
mkdir -p "$OUTDIR"

# Total: 5000 games, split across 8 workers = 625 each
CHUNK=625
WORKERS=8

for i in $(seq 0 $((WORKERS - 1))); do
    START=$((i * CHUNK))
    python code/my_ai/collect_value_seq.py \
        training_history/ss_20260709_191050/gen_0030.pt \
        training_history/ss_20260710_155327/gen_0101.pt \
        --games $CHUNK --seed $START \
        --output-dir "$OUTDIR" \
        --save-every 100 \
        > "value_data_v2/worker_${i}.log" 2>&1 &
    echo "Started worker $i (seed=$START, games=$CHUNK)"
done

echo "All $WORKERS workers launched. Waiting for completion..."
wait
echo "All workers done."

# Final count
TOTAL_FILES=$(ls "$OUTDIR"/*.npz 2>/dev/null | wc -l)
echo "Total .npz files: $TOTAL_FILES"
