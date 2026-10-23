#!/usr/bin/env bash
# **rollout 随机出招 vs 现状（双方不出招）**（2026-10-23，用户指出关键缺失）
#
# 发现：我们的 rollout 是 `advance_round()`，**双方都不出招、纯确定性** ⇒
#   实测 samples=1/16/64 的分数逐位相同（0.092069…），耗时 177/919/3548 ms
#   ⇒ `FLAT_ROLLOUTS=16` 是**用 16 倍算力重复算同一个数**（两个座位的进度文件哈希完全相同即为证据）。
#   亚军的 `simulate_round(rng)` 是**带随机性**的（`rs::FastRng rng(rollout_seed)`）。
#
# 本臂：A = 随机出招（每方每回合以 0.5 概率随机建塔/降级）+ 采样 8
#       B = 现状（双方不出招）+ 采样 1（因为采样 16 与 1 等价，用 1 省算力）
#   A 贵约 4x（829 ms vs 207 ms/决策），先回答"随机性到底有没有用"；若有用，再做同预算对照。
#
# 32 局配对（16 图 × 2 座位），jobs=8。
set -uo pipefail
cd "$(dirname "$0")/../.." || exit 1
PY=D:/anaconda3/envs/pytorch-gpu/python.exe
SEEDS=$(seq 1220 1235 | tr '\n' ' ')

BASE="AZAI_CKPT=checkpoints/posnet_A_k5_m32.pt;AZAI_ITERS=256;AZAI_DEPTH=4;AZAI_K=24;AZAI_SAMPLE_MULT=15;AZAI_MODE=pos-only;AZAI_POSPIN=mc_quiet;AZAI_MC_EVERY=1;AZAI_MC_ROOTONLY=1;AZAI_MC_CELLS=4;AZAI_SKIP1=1;AZAI_TCLASS=0.5;AZAI_TPOS=1.0;AZAI_TEMP=1e-6;AZAI_POSPRIOR=off;AZAI_SEARCH=flat;AZAI_FLAT_HORIZON=16;AZAI_FLAT_BATCH=32;AZAI_FLAT_ROUNDS=39000;AZAI_MC_HORIZON=16;AZAI_MC_SCORE=net;AZAI_MC_NET_SAMPLES=4"
A_RAND="$BASE;AZAI_FLAT_ROLLOUT_MODE=random;AZAI_FLAT_ROLLOUTS=8;AZAI_FLAT_ACT_PROB=0.5;AZAI_FLAT_SEED=17"
B_IDLE="$BASE;AZAI_FLAT_ROLLOUT_MODE=idle;AZAI_FLAT_ROLLOUTS=1"

echo "############ rollout: 随机出招(A) vs 双方不出招(B)  32 局配对  $(date +%H:%M:%S)"
$PY -u code/test_match/ab_search_modes.py --paired-sides --seeds $SEEDS --jobs 8 \
    --out-dir training_history/ab_rollout_random_vs_idle \
    --flat-env="$A_RAND" --mcts-env="$B_IDLE"
echo ""
echo "随机出招对照结束 ($(date +%H:%M:%S))"
