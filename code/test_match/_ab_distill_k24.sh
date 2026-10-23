#!/usr/bin/env bash
# **只换位置图（保留类头）vs 现冠军，k=24**（2026-10-23 第 2 轮）
#
# 第 1 轮：把 `action_map_conv` + `policy_heads` 都换成蒸馏网 ⇒ k=24 只有 **12.5%**（-11.62hp）。
# 假设：类头（`policy_heads`）是"禁闪电子博弈"里训出的经济偏好 ⇒ 在完整对局里把类选择带偏。
# 本臂隔离该因素：**只换 `action_map_conv`（位置图）**，类头与主干全保留我们的。
set -uo pipefail
cd "$(dirname "$0")/../.." || exit 1
PY=D:/anaconda3/envs/pytorch-gpu/python.exe
SEEDS=$(seq 1700 1715 | tr '\n' ' ')

COMMON="AZAI_ITERS=256;AZAI_DEPTH=4;AZAI_K=24;AZAI_SAMPLE_MULT=15;AZAI_MODE=pos-only;AZAI_POSPIN=mc_quiet;AZAI_MC_EVERY=1;AZAI_MC_ROOTONLY=1;AZAI_MC_CELLS=4;AZAI_SKIP1=1;AZAI_TCLASS=0.5;AZAI_TPOS=1.0;AZAI_TEMP=1e-6;AZAI_POSPRIOR=off;AZAI_SEARCH=flat;AZAI_FLAT_HORIZON=16;AZAI_FLAT_BATCH=32;AZAI_FLAT_ROUNDS=39000;AZAI_MC_HORIZON=16;AZAI_MC_SCORE=net;AZAI_MC_NET_SAMPLES=4;AZAI_FLAT_ROLLOUT_MODE=idle;AZAI_FLAT_ROLLOUTS=1"
A="AZAI_CKPT=checkpoints/prior_distill_smoke.pt;$COMMON"
B="AZAI_CKPT=checkpoints/posnet_A_k5_m32.pt;$COMMON"

echo "############ 只换位置图(A) vs 现冠军(B)，k=24  32 局配对  $(date +%H:%M:%S)"
$PY -u code/test_match/ab_search_modes.py --paired-sides --seeds $SEEDS --jobs 8 \
    --out-dir training_history/ab_distill_k24 \
    --flat-env="$A" --mcts-env="$B"
echo ""
echo "pos-only 拼接 k24 对照结束 ($(date +%H:%M:%S))"
