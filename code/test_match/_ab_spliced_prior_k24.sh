#!/usr/bin/env bash
# **拼接先验 vs 现冠军，k=24**（2026-10-23 夜，用户设想验证）
#
# 先验质量（固定候选集 + 固定教师 posnet_A.value_state，`diag_prior_fair.py`）：
#   · 我们 posnet_A：建塔通道 **-0.113~-0.122**（负相关）、闪电 **+0.607**
#   · SDK 蒸馏 grafted_sdk_A：建塔 +0.022~+0.031、闪电 +0.554
#   · **拼接 spliced_prior_A**（塔/降级/基地升级用蒸馏行，闪电+超武用我们行，主干用我们的）：
#     建塔 **+0.04~+0.22**（全部转正）、闪电 **+0.607**（原样保留）
#
# 本臂：A = spliced_prior_A（k=24）  B = posnet_A（k=24，现冠军）
#   k=24 的用意：广度小 ⇒ 候选覆盖面取决于**先验**，先验质量才会体现为棋力。
set -uo pipefail
cd "$(dirname "$0")/../.." || exit 1
PY=D:/anaconda3/envs/pytorch-gpu/python.exe
SEEDS=$(seq 1660 1675 | tr '\n' ' ')

COMMON="AZAI_ITERS=256;AZAI_DEPTH=4;AZAI_K=24;AZAI_SAMPLE_MULT=15;AZAI_MODE=pos-only;AZAI_POSPIN=mc_quiet;AZAI_MC_EVERY=1;AZAI_MC_ROOTONLY=1;AZAI_MC_CELLS=4;AZAI_SKIP1=1;AZAI_TCLASS=0.5;AZAI_TPOS=1.0;AZAI_TEMP=1e-6;AZAI_POSPRIOR=off;AZAI_SEARCH=flat;AZAI_FLAT_HORIZON=16;AZAI_FLAT_BATCH=32;AZAI_FLAT_ROUNDS=39000;AZAI_MC_HORIZON=16;AZAI_MC_SCORE=net;AZAI_MC_NET_SAMPLES=4;AZAI_FLAT_ROLLOUT_MODE=idle;AZAI_FLAT_ROLLOUTS=1"
A="AZAI_CKPT=checkpoints/spliced_prior_A.pt;$COMMON"
B="AZAI_CKPT=checkpoints/posnet_A_k5_m32.pt;$COMMON"

echo "############ 拼接先验(A) vs 现冠军(B)，k=24  32 局配对  $(date +%H:%M:%S)"
$PY -u code/test_match/ab_search_modes.py --paired-sides --seeds $SEEDS --jobs 8 \
    --out-dir training_history/ab_spliced_prior_k24 \
    --flat-env="$A" --mcts-env="$B"
echo ""
echo "拼接先验 k24 对照结束 ($(date +%H:%M:%S))"
