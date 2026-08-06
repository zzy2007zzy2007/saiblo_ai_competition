#!/usr/bin/env bash
# bundle-MCTS AlphaZero 多 batch 训练循环。
#
# 每个 batch: 用当前模型自对弈采集 10 局（128 迭代/深度 4, 10 worker 并行）
#   -> 训练（数据目录累计, init = 上一 batch 的 checkpoint）-> 保存 az_az{b}.pt。
# 手动打断安全：已完成 batch 的模型和自对弈数据都已落盘，随时可恢复。
#
# 用法:
#   bash code/my_ai/az_intent/run_az_batches.sh [总batch数] [起始batch]
set -u

REPO="D:/2026智能体大赛_新2"
PY="D:/anaconda3/envs/pytorch-gpu/python.exe"
DATA="$REPO/training_history/az_intent/az_selfplay_data"
N_BATCHES="${1:-12}"
START_BATCH="${2:-0}"

cd "$REPO"
mkdir -p "$DATA"

prev="$REPO/training_history/az_intent/gen0120_warm.pt"
if [ "$START_BATCH" -gt 0 ]; then
    prev="$REPO/training_history/az_intent/az_az$((START_BATCH - 1)).pt"
fi

echo "[run] total batches=$N_BATCHES start=$START_BATCH data=$DATA"
for b in $(seq "$START_BATCH" $((START_BATCH + N_BATCHES - 1))); do
    echo "==================== [batch $b] 开始 ===================="
    echo "[batch $b] 采集 10 局 (128 iter / depth 4, 10 workers) seed=$b ..."
    "$PY" code/my_ai/az_intent/az_selfplay.py \
        --checkpoint "$prev" \
        --games 10 --workers 10 \
        --iterations 128 --max-depth-rounds 4 --max-rounds 512 \
        --out-dir "$DATA" --seed "$b"
    ckpt="$REPO/training_history/az_intent/az_az$b.pt"
    echo "[batch $b] 训练 (init=$prev) -> $ckpt ..."
    "$PY" code/my_ai/az_intent/az_train.py \
        --init "$prev" --data-dir "$DATA" --checkpoint "$ckpt" --epochs 5
    echo "[batch $b] 完成 -> $ckpt"
    prev="$ckpt"
done
echo "[run] 全部 $N_BATCHES 个 batch 完成"
