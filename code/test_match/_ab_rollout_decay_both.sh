#!/usr/bin/env bash
# **双方随机 + 衰减 vs 部署口径**（2026-10-23 用户要求）
#
# 与前一条（只随机对手）的唯一差别：`AZAI_FLAT_RANDOM_SIDE=both`
#   · both   = 我方也参与随机（但**只在衰减的早期回合**，之后迅速安静）
#   · 之前 const p=0.5 的 both 版把局面搅成 churn ⇒ 34.4% 显著更差；
#     衰减后实测塔数变化只有 1-3（const 是 5-9），叶子状态接近值网训练分布。
#
# 成本对齐：A（both+decay 采样8）= 1125 ms ≈ B（idle 采样16，部署口径）= 919 ms
# 32 局配对（16 图 × 2 座位），jobs=8。
set -uo pipefail
cd "$(dirname "$0")/../.." || exit 1
PY=D:/anaconda3/envs/pytorch-gpu/python.exe
SEEDS=$(seq 1340 1355 | tr '\n' ' ')

BASE="AZAI_CKPT=checkpoints/posnet_A_k5_m32.pt;AZAI_ITERS=256;AZAI_DEPTH=4;AZAI_K=24;AZAI_SAMPLE_MULT=15;AZAI_MODE=pos-only;AZAI_POSPIN=mc_quiet;AZAI_MC_EVERY=1;AZAI_MC_ROOTONLY=1;AZAI_MC_CELLS=4;AZAI_SKIP1=1;AZAI_TCLASS=0.5;AZAI_TPOS=1.0;AZAI_TEMP=1e-6;AZAI_POSPRIOR=off;AZAI_SEARCH=flat;AZAI_FLAT_HORIZON=16;AZAI_FLAT_BATCH=32;AZAI_FLAT_ROUNDS=39000;AZAI_MC_HORIZON=16;AZAI_MC_SCORE=net;AZAI_MC_NET_SAMPLES=4"
A_BOTH="$BASE;AZAI_FLAT_ROLLOUT_MODE=random;AZAI_FLAT_ROLLOUTS=8;AZAI_FLAT_RANDOM_SIDE=both;AZAI_FLAT_PROB_MODE=decay;AZAI_FLAT_DECAY_C=0.6;AZAI_FLAT_DECAY_K=3.0;AZAI_FLAT_ACT_PROB=0.6;AZAI_FLAT_SEED=17"
B_IDLE="$BASE;AZAI_FLAT_ROLLOUT_MODE=idle;AZAI_FLAT_ROLLOUTS=16"

echo "等待「衰减随机（只对手）对照」结束… ($(date +%H:%M:%S))"
for _ in $(seq 1 200); do
  marker=$(ls -d training_history/runs/*ab_rollout_decay_vs_idle* 2>/dev/null | tail -1)
  if [ -n "$marker" ] && grep -q "衰减随机对照结束" "$marker/output.log" 2>/dev/null; then
    echo "上一场已结束 ($(date +%H:%M:%S))"; break
  fi
  sleep 30
done

echo ""
echo "############ 双方随机+衰减(A) vs 部署口径(B)  32 局配对  $(date +%H:%M:%S)"
$PY -u code/test_match/ab_search_modes.py --paired-sides --seeds $SEEDS --jobs 8 \
    --out-dir training_history/ab_rollout_decay_both \
    --flat-env="$A_BOTH" --mcts-env="$B_IDLE"
echo ""
echo "双方随机衰减对照结束 ($(date +%H:%M:%S))"
