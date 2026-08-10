#!/usr/bin/env bash
# bundle-MCTS AlphaZero 多 batch 迭代训练循环（第二轮方案，拆分双网 + 加权标签 + 缩放）。
#
# 每个 batch: 用当前模型（上一 batch 训练出的拆分 checkpoint）自对弈采集 10 局
#   （128 迭代/深度 4, 10 worker 并行）-> 训练（策略=当前 batch、价值=累计数据,
#   --split 双网, --label-scale 6）-> 保存 az_r{b}.pt。
# 手动打断安全：已完成 batch 的模型和自对弈数据都已落盘，随时可恢复。
#
# 用法:
#   bash code/my_ai/az_intent/run_az_batches.sh [总batch数] [起始batch] [init_checkpoint] [迭代数] [native_engine=0/1] \
#       [data_dir] [ckpt_dir] [games] [workers]
set -u

REPO="D:/2026智能体大赛_新2"
PY="D:/anaconda3/envs/pytorch-gpu/python.exe"
N_BATCHES="${1:-6}"
START_BATCH="${2:-9}"
INIT="${3:-$REPO/training_history/az_intent/az_r3.pt}"
ITERATIONS="${4:-128}"
NATIVE="${5:-0}"
DATA="${6:-$REPO/training_history/az_intent/az_selfplay_data}"
CKPT_DIR="${7:-$REPO/training_history/az_intent}"
GAMES="${8:-10}"
WORKERS="${9:-10}"

cd "$REPO"
mkdir -p "$DATA" "$CKPT_DIR"

prev="$INIT"
echo "[run] batches=$N_BATCHES start=$START_BATCH init=$prev iters=$ITERATIONS native=$NATIVE games=$GAMES workers=$WORKERS data=$DATA ckpt=$CKPT_DIR"
for b in $(seq "$START_BATCH" $((START_BATCH + N_BATCHES - 1))); do
    echo "==================== [batch $b] 开始 ===================="
    BATCH_DIR="$DATA/batch$b"
    NATIVE_ARG=""
    if [ "$NATIVE" = "1" ]; then NATIVE_ARG="--native-engine"; fi
    echo "[batch $b] 采集 $GAMES 局 ($ITERATIONS iter / depth 4, $WORKERS workers) seed=$b -> $BATCH_DIR [native=$NATIVE]..."
    "$PY" code/my_ai/az_intent/az_selfplay.py \
        --checkpoint "$prev" \
        --games "$GAMES" --workers "$WORKERS" \
        --iterations "$ITERATIONS" --max-depth-rounds 4 --max-rounds 512 \
        --out-dir "$BATCH_DIR" --seed "$b" $NATIVE_ARG
    ckpt="$CKPT_DIR/az_r$b.pt"
    echo "[batch $b] 训练 (init=$prev) -> $ckpt (policy=$BATCH_DIR, value=累计, split, scale=6)..."
    "$PY" code/my_ai/az_intent/az_train.py \
        --init "$prev" --policy-dir "$BATCH_DIR" --data-dir "$DATA" \
        --checkpoint "$ckpt" --epochs 5 --tau 20 --label-scale 6 --split \
        --max-value-batches 16 --device auto
    echo "[batch $b] 完成 -> $ckpt"
    prev="$ckpt"
done
echo "[run] 全部 $N_BATCHES 个 batch 完成"
