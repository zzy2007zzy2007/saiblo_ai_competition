#!/usr/bin/env bash
# 位置网迭代循环（docs/az_three_net_split_plan.md）——只训位置网，类网/价值网冻结。
#
# 设计（与用户确认）：
#   * 每轮采 400 局 → 过滤成紧凑数据集（2.8% 体量）→ **删掉本轮原始 pkl**（15GB/轮，磁盘放不下；
#     重采 36 分钟，且 compact 保留了位置损失需要的全部字段）→ 从上一轮**温启动**训 8 epoch；
#   * 累积池：`--policy-dir` 指向所有轮 compact 的父目录（rglob 自动并集，无需 merge）；
#   * **主判据永远是 vs 第 0 轮**（未训练的原始位置网）的镜像配对——因为该台 SE=0，
#     对着"上一轮"比的话纯漂移也会涨分，会自己骗自己；
#   * 每 2 轮跑一次 vs rule_v4（带廉价搜索）当"没跑偏"的粗锚。
#
# 用法：bash code/run_posnet_loop.sh [轮数]   （默认 5）
export PYTHONIOENCODING=utf-8
set -u
REPO="D:/2026智能体大赛_新2"; PY="D:/anaconda3/envs/pytorch-gpu/python.exe"
cd "$REPO" || exit 1
ROUNDS="${1:-5}"
CK3=training_history/az_fixed/three_mix_r10p_vw_pol_frozen.pt   # 第 0 轮（未训练的位置网）
W=training_history/posnet_loop
GAMES=400
LAMBDA_ANCHOR=30          # Phase 2 实测：有效 anchor/CE ≈1%，且绝对血量差方向最好
EPOCHS=8

mkdir -p "$W"
echo "### 位置网迭代循环：$ROUNDS 轮 × $GAMES 局，λ_anchor=$LAMBDA_ANCHOR，$EPOCHS epoch ###"

PREV="$CK3"
for r in $(seq 1 "$ROUNDS"); do
  SEED=$((920000 + r * 1000))
  RAW="$W/r${r}_raw"; CMP="$W/pool/r${r}"
  echo
  echo "########## 第 $r/$ROUNDS 轮 ##########"

  echo "--- 采集 $GAMES 局（用第 $((r-1)) 轮的策略做搜索，种子 $SEED）---"
  "$PY" code/my_ai/az_intent/az_selfplay.py --checkpoint "$PREV" --games "$GAMES" \
      --workers 16 --iterations 256 --max-depth-rounds 2 --max-rounds 512 --native-engine \
      --skip-hold-search --search-mode pos-only --skip-single-candidate \
      --out-dir "$RAW" --seed "$SEED" --t-class 0.5 --t-pos 0.3 --k 24 \
      || { echo "!! 第 $r 轮采集失败"; exit 1; }

  echo "--- 过滤成紧凑数据集 ---"
  "$PY" code/my_ai/az_intent/filter_pos_samples.py --src "$RAW" --out "$CMP" \
      || { echo "!! 第 $r 轮过滤失败"; exit 1; }
  N=$(find "$CMP" -name "*.pkl" | wc -l)
  if [ "$N" -lt 1 ]; then echo "!! 第 $r 轮 compact 为空，拒绝删 raw"; exit 1; fi
  echo "--- 删除本轮原始 pkl（$RAW，已过滤）---"
  rm -rf "$RAW"

  echo "--- 训练位置网（从 $PREV 温启动，累积池 = $W/pool）---"
  "$PY" code/my_ai/az_intent/az_train.py --pos-only-net --init "$PREV" \
      --policy-dir "$W/pool" --checkpoint "$W/pos_r$r.pt" \
      --epochs "$EPOCHS" --batch-size 32 --lr 1e-3 \
      --lambda-anchor "$LAMBDA_ANCHOR" --pos-single \
      || { echo "!! 第 $r 轮训练失败"; exit 1; }
  PREV="$W/pos_r$r.pt"

  echo "--- 主判据：第 $r 轮 vs 第 0 轮（不搜索镜像配对 64 pair）---"
  "$PY" _tmp_position_choice_ab.py --checkpoint "$PREV" --opponent-checkpoint "$CK3" \
      --mode raw --pairs 64 --workers 16 --seed 0

  if [ $((r % 2)) -eq 0 ]; then
    echo "--- 粗锚：第 $r 轮 vs rule_v4（带廉价搜索，64 局）---"
    "$PY" code/my_ai/az_intent/eval.py --checkpoint "$PREV" --opponent rule_v4 \
        --bundle-mcts --native-engine --iterations 256 --max-depth-rounds 4 --k 24 \
        --t-class 0.5 --t-pos 0.3 --search-mode pos-only --skip-single-candidate \
        --games 64 --workers 16
  fi
done
echo
echo "### 位置网迭代循环完成：$ROUNDS 轮 ###"
