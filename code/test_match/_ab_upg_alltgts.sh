#!/usr/bin/env bash
# **升级目标**对照：A = 枚举全部 4 个升级目标（HEAVY/QUICK/MORTAR/PRODUCER），B = 不升级。
#
# 背景（2026-10-22）：
#   · 修好 `int('op')` 之后，`cells4 + MC_UPG=1` **显著差于**不升级（34 局 26.5%，血量差 −9.6，CI 全在 0 下）；
#   · 原因怀疑在**目标选择**：旧代码永远取 `_tgts[0]` = **HEAVY**，而亚军的奖励是
#     Quick +100 / Sniper +400、对 Heavy **不给**；
#   · 实测同一局面下仲裁给 QUICK 0.25 > HEAVY 0.20 ⇒ 放开目标后它**会**选 QUICK。
#   ⇒ 本臂检验："升级有害" 是 "升级本身有害"，还是 "统一升成 Heavy 有害"。
#
# A 比 B 只多两个开关：AZAI_MC_UPG=1 + AZAI_MC_UPG_TARGETS=4。
# 用 --paired-sides：16 张图 × 2 座位 = 32 局，同图交换座位（座位效应实测 +19~31pp，必须消掉）。
set -uo pipefail
cd "$(dirname "$0")/../.." || exit 1
PY=D:/anaconda3/envs/pytorch-gpu/python.exe
SEEDS=$(seq 800 815 | tr '\n' ' ')

BASE="AZAI_CKPT=checkpoints/posnet_A_k5_m32.pt;AZAI_ITERS=256;AZAI_DEPTH=4;AZAI_K=24;AZAI_SAMPLE_MULT=15;AZAI_MODE=pos-only;AZAI_POSPIN=mc_quiet;AZAI_MC_EVERY=1;AZAI_MC_HORIZON=256;AZAI_MC_ROOTONLY=1;AZAI_MC_CELLS=4;AZAI_SKIP1=1;AZAI_TCLASS=0.5;AZAI_TPOS=1.0;AZAI_TEMP=1e-6;AZAI_POSPRIOR=off;AZAI_SEARCH=flat;AZAI_FLAT_ROLLOUTS=16;AZAI_FLAT_HORIZON=64;AZAI_FLAT_BATCH=32;AZAI_FLAT_ROUNDS=39000"
A_ALLTGT="$BASE;AZAI_MC_UPG=1;AZAI_MC_UPG_TARGETS=4"
B_NONE="$BASE"

echo "等待「自对弈座位控制」结束… ($(date +%H:%M:%S))"
for _ in $(seq 1 400); do
  marker=$(ls -d training_history/runs/*ab_seat_control* 2>/dev/null | tail -1)
  if [ -n "$marker" ] && grep -q "座位控制结束" "$marker/output.log" 2>/dev/null; then
    echo "上一场已结束 ($(date +%H:%M:%S))"
    break
  fi
  sleep 30
done

echo ""
echo "############ 升级全目标(A) vs 不升级(B) 32 局  $(date +%H:%M:%S)"
$PY -u code/test_match/ab_search_modes.py --paired-sides --seeds $SEEDS --jobs 5 \
    --out-dir training_history/ab_upg_alltgts \
    --flat-env="$A_ALLTGT" --mcts-env="$B_NONE"
echo ""
echo "升级目标对照结束 ($(date +%H:%M:%S))"
