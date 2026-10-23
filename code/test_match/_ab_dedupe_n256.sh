#!/usr/bin/env bash
# **大 N + 去重加权**（2026-10-23，修 N=128 暴露的统计问题）
#
# 问题：C=.3/k=1 下大量样本重复。实测有效轨迹占比：
#   N=32  → 503/1536 = 33%
#   N=128 → 2400/6144 = **12%**（88% 是重复！）
#   ⇒ 朴素均值被重复的"静止"轨迹主导，N 越大排序越退化：N=64 68.8% → N=128 46.9%。
#
# 修法：`rollout_dedupe_weight=1` —— 按轨迹 token 分组，每组只算 1 份权重。
# 本臂：A = N=256 + 前缀树 + **去重加权**   B = 当前冠军（静止 + 采样1）
#   成本：树在 C=.3/k=1 下节点数饱和，N=256 与 N=128 同价（~7.4 s/决策）。
set -uo pipefail
cd "$(dirname "$0")/../.." || exit 1
PY=D:/anaconda3/envs/pytorch-gpu/python.exe
SEEDS=$(seq 1620 1635 | tr '\n' ' ')

BASE="AZAI_CKPT=checkpoints/posnet_A_k5_m32.pt;AZAI_ITERS=256;AZAI_DEPTH=4;AZAI_SAMPLE_MULT=15;AZAI_MODE=pos-only;AZAI_POSPIN=mc_quiet;AZAI_MC_EVERY=1;AZAI_MC_ROOTONLY=1;AZAI_MC_CELLS=4;AZAI_SKIP1=1;AZAI_TCLASS=0.5;AZAI_TPOS=1.0;AZAI_TEMP=1e-6;AZAI_POSPRIOR=off;AZAI_SEARCH=flat;AZAI_K=160;AZAI_FLAT_HORIZON=16;AZAI_FLAT_BATCH=32;AZAI_FLAT_ROUNDS=39000;AZAI_MC_HORIZON=16;AZAI_MC_SCORE=net;AZAI_MC_NET_SAMPLES=4"
A="$BASE;AZAI_FLAT_ROLLOUT_MODE=random;AZAI_FLAT_ROLLOUTS=256;AZAI_FLAT_RANDOM_SIDE=opponent;AZAI_FLAT_PROB_MODE=decay;AZAI_FLAT_DECAY_C=0.3;AZAI_FLAT_DECAY_K=1.0;AZAI_FLAT_ACT_PROB=0.3;AZAI_FLAT_SHARE_PREFIX=1;AZAI_FLAT_DEDUPE=1;AZAI_FLAT_SEED=17"
B="$BASE;AZAI_FLAT_ROLLOUT_MODE=idle;AZAI_FLAT_ROLLOUTS=1"

echo "等待「全枚举」结束… ($(date +%H:%M:%S))"
for _ in $(seq 1 200); do
  marker=$(ls -d training_history/runs/*ab_enum300* 2>/dev/null | tail -1)
  if [ -n "$marker" ] && grep -q "全枚举对照结束" "$marker/output.log" 2>/dev/null; then
    echo "上一场已结束 ($(date +%H:%M:%S))"; break
  fi
  sleep 30
done

echo ""
echo "############ 大N+去重加权(A) vs 冠军(B)  32 局配对  $(date +%H:%M:%S)"
$PY -u code/test_match/ab_search_modes.py --paired-sides --seeds $SEEDS --jobs 8 \
    --out-dir training_history/ab_dedupe_n256 \
    --flat-env="$A" --mcts-env="$B"
echo ""
echo "去重加权对照结束 ($(date +%H:%M:%S))"
