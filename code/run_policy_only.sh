#!/usr/bin/env bash
# 策略-only 迭代训练：冻结价值头，只训策略（价值网在 split 架构里独立，天然不动）。
#
# 每个 batch: 用当前 checkpoint（策略=上一批训练后，价值=固定不变）自对弈采集 GAMES 局
#   -> az_train --split --policy-only 只更新策略 -> 保存 az_p{b}.pt
#
# 用法:
#   bash code/run_policy_only.sh [batch数] [起始batch] [init] [data目录] [ckpt目录] \
#       [每批局数] [workers] [迭代数] [随机动作注入概率x]
set -u

REPO="D:/2026智能体大赛_新2"
PY="D:/anaconda3/envs/pytorch-gpu/python.exe"
N="${1:-9}"
START="${2:-2}"
INIT="${3:-$REPO/training_history/az_fixed/az_p1.pt}"
DATA="${4:-$REPO/training_history/az_fixed/data_polonly}"
CKPT_DIR="${5:-$REPO/training_history/az_fixed}"
GAMES="${6:-16}"
WORKERS="${7:-16}"
ITER="${8:-256}"
RAND="${9:-0.0}"

cd "$REPO" || exit 1
prev="$INIT"
echo "[polonly] batches=$N start=$START init=$prev data=$DATA games=$GAMES workers=$WORKERS iters=$ITER rand=$RAND"
for b in $(seq "$START" $((START + N - 1))); do
    BATCH_DIR="$DATA/batch$b"
    ckpt="$CKPT_DIR/az_p$b.pt"
    if [ -f "$ckpt" ]; then
        echo "==================== [batch $b] 已有 $ckpt，跳过 ===================="
        prev="$ckpt"
        continue
    fi
    echo "==================== [batch $b] 采集开始 ===================="
    "$PY" code/my_ai/az_intent/az_selfplay.py \
        --checkpoint "$prev" --games "$GAMES" --workers "$WORKERS" \
        --iterations "$ITER" --max-depth-rounds 4 --max-rounds 512 \
        --out-dir "$BATCH_DIR" --seed $((40000 + b * 1000)) --native-engine \
        --t-class 0.5 --t-pos 1.0 --k 24 --random-action-prob "$RAND"
    echo "==================== [batch $b] 策略训练 (init=$prev) -> $ckpt ===================="
    "$PY" code/my_ai/az_intent/az_train.py \
        --init "$prev" --policy-dir "$BATCH_DIR" --data-dir "$DATA" \
        --checkpoint "$ckpt" --epochs 5 --split --policy-only \
        --max-value-batches 1 --device auto
    echo "[batch $b] 完成 -> $ckpt"
    prev="$ckpt"
done
echo "[polonly] 全部完成，最后一个模型: $prev"
