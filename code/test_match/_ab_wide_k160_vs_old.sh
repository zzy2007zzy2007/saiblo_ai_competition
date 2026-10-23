#!/usr/bin/env bash
# **免费广度**：采样 1 + 候选 160 vs 现状（采样 16 + 候选 24）
#
# 由来（2026-10-23 确定性发现）：
#   rollout 里双方不出招 ⇒ `advance_round()` 确定性 ⇒ **采样 1/16/64 得到同一个叶子**，
#   实测分数逐位相同、耗时 177/919/3548 ms。也就是说部署配置里的 `FLAT_ROLLOUTS=16`
#   是把 94% 的算力浪费在重复计算上。
#   ⇒ 同样的时间预算，可以改买**广度**：k=160、采样 1（实测 930 ms，与旧配置 919 ms 相当）。
#
# 本臂：A = k=160 / samples=1 / h16     （6.7× 候选，同成本）
#       B = k=24  / samples=16 / h16    （**当前部署口径**，不是上一臂的 s=1）
#
# 32 局配对（16 图 × 2 座位），jobs=8。
set -uo pipefail
cd "$(dirname "$0")/../.." || exit 1
PY=D:/anaconda3/envs/pytorch-gpu/python.exe
SEEDS=$(seq 1260 1275 | tr '\n' ' ')

BASE="AZAI_CKPT=checkpoints/posnet_A_k5_m32.pt;AZAI_ITERS=256;AZAI_DEPTH=4;AZAI_SAMPLE_MULT=15;AZAI_MODE=pos-only;AZAI_POSPIN=mc_quiet;AZAI_MC_EVERY=1;AZAI_MC_ROOTONLY=1;AZAI_MC_CELLS=4;AZAI_SKIP1=1;AZAI_TCLASS=0.5;AZAI_TPOS=1.0;AZAI_TEMP=1e-6;AZAI_POSPRIOR=off;AZAI_SEARCH=flat;AZAI_FLAT_HORIZON=16;AZAI_FLAT_BATCH=32;AZAI_FLAT_ROUNDS=39000;AZAI_MC_HORIZON=16;AZAI_MC_SCORE=net;AZAI_MC_NET_SAMPLES=4"
A_WIDE="$BASE;AZAI_K=160;AZAI_FLAT_ROLLOUTS=1"
B_OLD="$BASE;AZAI_K=24;AZAI_FLAT_ROLLOUTS=16"

echo "等待「随机出招对照」结束… ($(date +%H:%M:%S))"
for _ in $(seq 1 200); do
  marker=$(ls -d training_history/runs/*ab_rollout_random* 2>/dev/null | tail -1)
  if [ -n "$marker" ] && grep -q "随机出招对照结束" "$marker/output.log" 2>/dev/null; then
    echo "上一场已结束 ($(date +%H:%M:%S))"; break
  fi
  sleep 30
done

echo ""
echo "############ 免费广度：k=160/s=1(A) vs k=24/s=16(B)  32 局配对  $(date +%H:%M:%S)"
$PY -u code/test_match/ab_search_modes.py --paired-sides --seeds $SEEDS --jobs 8 \
    --out-dir training_history/ab_wide_k160_vs_old \
    --flat-env="$A_WIDE" --mcts-env="$B_OLD"
echo ""
echo "广度对照结束 ($(date +%H:%M:%S))"
