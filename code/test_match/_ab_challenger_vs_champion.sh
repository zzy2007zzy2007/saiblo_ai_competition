#!/usr/bin/env bash
# **新配置 vs 现有冠军**（正面对打，配对同图交换座位）
#
# A（挑战者）= cells4 + 升级(全目标) + **值网评分 @h64**
#      来由：2026-10-22 `net vs quiet` 32 局配对实测 **71.9%**（23W-9L），
#            机制上"少升级(15.1%→13.4%)、多建塔(1888→2057)"，符合"值网看得见金币"的预测
#            （autograd: ∂V/∂我方金币 = +0.0274，16/16 局面为正）。
# B（现任冠军）= cells4 + 安静rollout@H256 + **不升级**（今天之前的部署配置）
#
# 设计要点：
#   · `--paired-sides`：16 张图 × 2 座位 = 32 局，同图交换座位（座位/地图方差在图内相消）
#   · jobs=5（5 局 × 3 进程 = 15，符合"总进程 ≤16"）
#   · 主指标：配对胜率 + 血量差；机制指标：升级率、建塔数
set -uo pipefail
cd "$(dirname "$0")/../.." || exit 1
PY=D:/anaconda3/envs/pytorch-gpu/python.exe
SEEDS=$(seq 960 975 | tr '\n' ' ')

COMMON="AZAI_CKPT=checkpoints/posnet_A_k5_m32.pt;AZAI_ITERS=256;AZAI_DEPTH=4;AZAI_K=24;AZAI_SAMPLE_MULT=15;AZAI_MODE=pos-only;AZAI_POSPIN=mc_quiet;AZAI_MC_EVERY=1;AZAI_MC_ROOTONLY=1;AZAI_MC_CELLS=4;AZAI_SKIP1=1;AZAI_TCLASS=0.5;AZAI_TPOS=1.0;AZAI_TEMP=1e-6;AZAI_POSPRIOR=off;AZAI_SEARCH=flat;AZAI_FLAT_ROLLOUTS=16;AZAI_FLAT_HORIZON=64;AZAI_FLAT_BATCH=32;AZAI_FLAT_ROUNDS=39000"
# 挑战者：值网评分 + 全目标升级 + h64
A_CHALLENGER="$COMMON;AZAI_MC_HORIZON=64;AZAI_MC_SCORE=net;AZAI_MC_NET_SAMPLES=4;AZAI_MC_UPG=1;AZAI_MC_UPG_TARGETS=4"
# 现任：安静 rollout 长视界、不升级
B_CHAMPION="$COMMON;AZAI_MC_HORIZON=256"

echo "############ 新配置(net+升级) vs 现冠军(quiet+不升级)  32 局配对  $(date +%H:%M:%S)"
$PY -u code/test_match/ab_search_modes.py --paired-sides --seeds $SEEDS --jobs 5 \
    --out-dir training_history/ab_challenger_vs_champion \
    --flat-env="$A_CHALLENGER" --mcts-env="$B_CHAMPION"
echo ""
echo "挑战者对照结束 ($(date +%H:%M:%S))"
