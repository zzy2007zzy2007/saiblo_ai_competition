#!/usr/bin/env bash
# 复刻"原始 AZ 批循环"（~2-3 周前那版），**只改三处**：
#   ① 采集掺 2% 随机动作注入（--random-action-prob 0.02）
#   ② 开 BatchNorm（init 用 BN checkpoint；不折叠）
#   ③ 标签用 kgeo（--label-mode abs --label-weight kgeo --tau 50 --label-scale 1.5）
#
# 其余与原始版一致：
#   - az_train --split：**策略网和价值网都训**（不是 value-only）
#   - 双池：policy_dir = 本 batch，data_dir = 累积（--value-passes 3）
#   - 采集 256 迭代 / depth 4 / C++ 引擎 / 16 workers（**不**用 HOLD-skip，保持与原始一致）
#   - 每 batch 采集 32 局、训练 5 epochs、共 5 个 batch
#   - 网络结构不变（不做任何"类/位置拆分"）
#
# 每 batch 结束时评价一次 32 局 vs rule_v4（给出强度曲线）。
#
# 用法：bash code/run_az_bn_kgeo.sh [rounds=5] [games/batch=32] [epochs=5]
set -u
REPO="D:/2026智能体大赛_新2"; PY="D:/anaconda3/envs/pytorch-gpu/python.exe"
cd "$REPO" || exit 1

ROUNDS="${1:-5}"
GAMES="${2:-32}"
EPOCHS="${3:-5}"
WORK="${WORK:-training_history/az_bn_kgeo}"
DATA="$WORK/data"
CKPT="$WORK"
INIT="training_history/az_fixed/gen0120_bn_init.pt"   # BN checkpoint → 网络开 BN
INJECT="${INJECT:-0.02}"
mkdir -p "$DATA" "$CKPT"

echo "[bnkgeo] rounds=$ROUNDS games/batch=$GAMES epochs=$EPOCHS inject=$INJECT"
echo "[bnkgeo] init=$INIT (BN)  形态: split（策略+价值都训，非 value-only）"
echo "[bnkgeo] 采集 256 迭代/depth4/C++/16workers（无 HOLD-skip）"

prev="$INIT"
for b in $(seq 1 "$ROUNDS"); do
  echo "########## BATCH $b / $ROUNDS ##########"
  echo "--- 采集 $GAMES 局（掺 $INJECT 随机注入）---"
  "$PY" code/my_ai/az_intent/az_selfplay.py \
      --checkpoint "$prev" --games "$GAMES" --workers 16 \
      --iterations 256 --max-depth-rounds 4 --max-rounds 512 \
      --out-dir "$DATA/batch$b" --seed $((80000 + b * 1000)) \
      --native-engine --random-action-prob "$INJECT" \
      --t-class 0.5 --t-pos 0.3 --k 24

  echo "--- 训练（split：策略+价值，BN，kgeo 标签，$EPOCHS epochs，value-passes 3）---"
  "$PY" code/my_ai/az_intent/az_train.py \
      --init "$prev" --policy-dir "$DATA/batch$b" --data-dir "$DATA" \
      --checkpoint "$CKPT/az_b$b.pt" --epochs "$EPOCHS" --split \
      --label-mode abs --label-weight kgeo --tau 50 --label-scale 1.5 \
      --value-passes 3 --device auto
  prev="$CKPT/az_b$b.pt"
done

echo "--- 全部批次的最终评价：64 局 vs rule_v4 ---"
"$PY" code/my_ai/az_intent/eval.py \
    --checkpoint "$prev" --opponent rule_v4 --bundle-mcts --native-engine \
    --iterations 256 --max-depth-rounds 4 \
    --t-class 0.5 --t-pos 0.3 --k 24 --games 64 --workers 16
echo "############ az_bn_kgeo 全部 $ROUNDS 个 batch 完成 ############"
