#!/usr/bin/env bash
# **仲裁评估器：quiet vs net**（都开升级，配对同图交换座位）
#
# 动机（2026-10-22 autograd 实测）：
#   · 安静 rollout 的分数 = 终局 HP 差 ⇒ **结构上看不见金币**；
#   · 值网 `∂V/∂(我方金币) = +0.0274`（16/16 局面为正）、`∂V/∂(敌方金币) = −0.0562`
#     ⇒ **它看得见钱，且方向正确**（虽然幅度偏小：300 币 ≈ 0.55 HP）。
#   ⇒ 因此"把叶节点评分换成值网"应当能**抑制乱花钱**（少升级、多攒钱），
#     这正是 `diag_upg_damage` 发现的核心病灶（我们金币中位 46 vs 亚军 128、p90 429）。
#
# 单变量：只换仲裁评分器（两侧都 h=64、都开升级 targets=4）。
# 机制指标（比胜率更早给信号）：升级次数、金币轨迹（用 code/test_match/diag_upg_damage.py 口径）。
set -uo pipefail
cd "$(dirname "$0")/../.." || exit 1
PY=D:/anaconda3/envs/pytorch-gpu/python.exe
SEEDS=$(seq 900 915 | tr '\n' ' ')

BASE="AZAI_CKPT=checkpoints/posnet_A_k5_m32.pt;AZAI_ITERS=256;AZAI_DEPTH=4;AZAI_K=24;AZAI_SAMPLE_MULT=15;AZAI_MODE=pos-only;AZAI_POSPIN=mc_quiet;AZAI_MC_EVERY=1;AZAI_MC_HORIZON=64;AZAI_MC_ROOTONLY=1;AZAI_MC_CELLS=4;AZAI_MC_UPG=1;AZAI_MC_UPG_TARGETS=4;AZAI_SKIP1=1;AZAI_TCLASS=0.5;AZAI_TPOS=1.0;AZAI_TEMP=1e-6;AZAI_POSPRIOR=off;AZAI_SEARCH=flat;AZAI_FLAT_ROLLOUTS=16;AZAI_FLAT_HORIZON=64;AZAI_FLAT_BATCH=32;AZAI_FLAT_ROUNDS=39000"
A_NET="$BASE;AZAI_MC_SCORE=net;AZAI_MC_NET_SAMPLES=4"
B_QUIET="$BASE"

echo "############ 仲裁评分器 net(A) vs quiet(B)，升级都开，32 局配对  $(date +%H:%M:%S)"
$PY -u code/test_match/ab_search_modes.py --paired-sides --seeds $SEEDS --jobs 5 \
    --out-dir training_history/ab_mc_net_vs_quiet_upg \
    --flat-env="$A_NET" --mcts-env="$B_QUIET"
echo ""
echo "评估器对照结束 ($(date +%H:%M:%S))"
