#!/usr/bin/env bash
# **升级在「值网评分」下还有害吗？**（单变量：只切 AZAI_MC_UPG）
#
# 背景（2026-10-22 两场结果）：
#   ① `net vs quiet`（都开升级、都 h64、32 局配对）= **net 71.9%** ⇒ 值网评分明显更好；
#      机制：升级率 15.1%→13.4%、建塔 1888→2057（值网看得见金币成本）。
#   ② `挑战者(net+升级+h64) vs 冠军(quiet+H256+不升级)` = **31.2%** ⇒ 挑战者落后。
#      ⇒ 挑战者同时改了三处（评分器/视界/升级），而"升级有害"此前只在 quiet 评分下测过
#        （quiet 下：23.4%、18.8%，血量差 −10、−13）。
#
# 本实验回答唯一剩下的问题：**换成值网评分之后，升级还值不值？**
#   A = 值网@h64 + 升级(全目标)
#   B = 值网@h64 + 不升级            ← 只差 `AZAI_MC_UPG` 这一个开关
#
# 判读：
#   · A ≥ 50% ⇒ 值网把升级"救活了" ⇒ 新配方 = net + 升级，可继续与冠军比；
#   · A < 50%  ⇒ 升级就是负收益（与评分器无关）⇒ 新配方 = net + **不升级**，拿它去和冠军比。
set -uo pipefail
cd "$(dirname "$0")/../.." || exit 1
PY=D:/anaconda3/envs/pytorch-gpu/python.exe
SEEDS=$(seq 990 1005 | tr '\n' ' ')

BASE="AZAI_CKPT=checkpoints/posnet_A_k5_m32.pt;AZAI_ITERS=256;AZAI_DEPTH=4;AZAI_K=24;AZAI_SAMPLE_MULT=15;AZAI_MODE=pos-only;AZAI_POSPIN=mc_quiet;AZAI_MC_EVERY=1;AZAI_MC_ROOTONLY=1;AZAI_MC_CELLS=4;AZAI_SKIP1=1;AZAI_TCLASS=0.5;AZAI_TPOS=1.0;AZAI_TEMP=1e-6;AZAI_POSPRIOR=off;AZAI_SEARCH=flat;AZAI_FLAT_ROLLOUTS=16;AZAI_FLAT_HORIZON=64;AZAI_FLAT_BATCH=32;AZAI_FLAT_ROUNDS=39000;AZAI_MC_HORIZON=64;AZAI_MC_SCORE=net;AZAI_MC_NET_SAMPLES=4"
A_UPG="$BASE;AZAI_MC_UPG=1;AZAI_MC_UPG_TARGETS=4"
B_NOUPG="$BASE"

echo "############ 值网评分下：升级(A) vs 不升级(B)  32 局配对  $(date +%H:%M:%S)"
$PY -u code/test_match/ab_search_modes.py --paired-sides --seeds $SEEDS --jobs 8 \
    --out-dir training_history/ab_upg_under_net \
    --flat-env="$A_UPG" --mcts-env="$B_NOUPG"
echo ""
echo "升级(值网下)对照结束 ($(date +%H:%M:%S))"
