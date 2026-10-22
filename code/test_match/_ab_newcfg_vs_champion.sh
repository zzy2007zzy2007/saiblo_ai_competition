#!/usr/bin/env bash
# **新配方 vs 现任冠军**（决定要不要换部署配置）
#
# 新配方 A = 值网评分@h64 + **不升级**
#   · 值网评分：`net vs quiet` 32 局配对 = **71.9%**（23W-9L），机制"少升级多建塔"；
#   · 不升级：三次实验一致证明升级是负收益
#     （quiet 下 23.4% / 18.8%；值网下 31.2%，CI 全在 0 以下）。
# 现任冠军 B = 安静 rollout@H256 + 不升级（今天之前的部署配置）
#
# 单变量意图：评分器（net vs quiet）。视界 64 vs 256 是既有冠军的配置，故保留不改
#   —— 若 A 赢，说明"换评分器"本身就够；视界可另开一条臂再优化。
#
# 32 局配对（16 图 × 2 座位），jobs=8（16 个 bridge 吃满 16 物理核）。
set -uo pipefail
cd "$(dirname "$0")/../.." || exit 1
PY=D:/anaconda3/envs/pytorch-gpu/python.exe
SEEDS=$(seq 1020 1035 | tr '\n' ' ')

COMMON="AZAI_CKPT=checkpoints/posnet_A_k5_m32.pt;AZAI_ITERS=256;AZAI_DEPTH=4;AZAI_K=24;AZAI_SAMPLE_MULT=15;AZAI_MODE=pos-only;AZAI_POSPIN=mc_quiet;AZAI_MC_EVERY=1;AZAI_MC_ROOTONLY=1;AZAI_MC_CELLS=4;AZAI_SKIP1=1;AZAI_TCLASS=0.5;AZAI_TPOS=1.0;AZAI_TEMP=1e-6;AZAI_POSPRIOR=off;AZAI_SEARCH=flat;AZAI_FLAT_ROLLOUTS=16;AZAI_FLAT_HORIZON=64;AZAI_FLAT_BATCH=32;AZAI_FLAT_ROUNDS=39000"
A_NEW="$COMMON;AZAI_MC_HORIZON=64;AZAI_MC_SCORE=net;AZAI_MC_NET_SAMPLES=4"
B_CHAMPION="$COMMON;AZAI_MC_HORIZON=256"

echo "############ 新配方(net@h64,不升级) vs 冠军(quiet@H256,不升级)  32 局配对  $(date +%H:%M:%S)"
$PY -u code/test_match/ab_search_modes.py --paired-sides --seeds $SEEDS --jobs 8 \
    --out-dir training_history/ab_newcfg_vs_champion \
    --flat-env="$A_NEW" --mcts-env="$B_CHAMPION"
echo ""
echo "新配方对照结束 ($(date +%H:%M:%S))"
