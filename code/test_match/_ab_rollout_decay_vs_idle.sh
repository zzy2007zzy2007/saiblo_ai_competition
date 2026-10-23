#!/usr/bin/env bash
# **衰减随机 rollout vs 静止**（2026-10-23 用户提议）
#
# 背景：
#   · `rollout_mode=random` 恒定 p=0.5 ⇒ 双方反复建拆塔(churn)，叶子状态被搅烂 ⇒
#     实测 11W-21L = 34.4%，血量差 -7.47（CI[-12.3,-2.6]）**显著更差**；
#   · 诊断（code/test_match/diag_random_rollout_ops.py）：16 回合里塔数变化 5~9、金币差 176~413，
#     而 HP 几乎不动（20→21）⇒ 噪声污染而非"更真实的模拟"；
#   · 亚军的随机性在**敌方蚂蚁移动/传送**（`move_phase(rng)`/`teleport_phase(rng)`），
#     我方始终按候选计划走 ⇒ 我们改为**只随机对手**。
#
# 用户提议：只在**第一次决策**随机，或按 p(r)=C·e^(−r/k) 衰减。
#   实测叶子偏移：first(C=0.6) 塔数变化 0-1；decay(C=.6,k=3) 1-3；const 5-9 ⇒ 衰减后明显"贴近静止"。
#
# 本臂：A = 只随机对手 + decay(C=0.6,k=3) + 采样4
#       B = 静止 + 采样1（现状）
set -uo pipefail
cd "$(dirname "$0")/../.." || exit 1
PY=D:/anaconda3/envs/pytorch-gpu/python.exe
SEEDS=$(seq 1300 1315 | tr '\n' ' ')

BASE="AZAI_CKPT=checkpoints/posnet_A_k5_m32.pt;AZAI_ITERS=256;AZAI_DEPTH=4;AZAI_K=24;AZAI_SAMPLE_MULT=15;AZAI_MODE=pos-only;AZAI_POSPIN=mc_quiet;AZAI_MC_EVERY=1;AZAI_MC_ROOTONLY=1;AZAI_MC_CELLS=4;AZAI_SKIP1=1;AZAI_TCLASS=0.5;AZAI_TPOS=1.0;AZAI_TEMP=1e-6;AZAI_POSPRIOR=off;AZAI_SEARCH=flat;AZAI_FLAT_HORIZON=16;AZAI_FLAT_BATCH=32;AZAI_FLAT_ROUNDS=39000;AZAI_MC_HORIZON=16;AZAI_MC_SCORE=net;AZAI_MC_NET_SAMPLES=4"
A_DECAY="$BASE;AZAI_FLAT_ROLLOUT_MODE=random;AZAI_FLAT_ROLLOUTS=4;AZAI_FLAT_RANDOM_SIDE=opponent;AZAI_FLAT_PROB_MODE=decay;AZAI_FLAT_DECAY_C=0.6;AZAI_FLAT_DECAY_K=3.0;AZAI_FLAT_ACT_PROB=0.6;AZAI_FLAT_SEED=17"
B_IDLE="$BASE;AZAI_FLAT_ROLLOUT_MODE=idle;AZAI_FLAT_ROLLOUTS=1"

echo "等待「广度对照」结束… ($(date +%H:%M:%S))"
for _ in $(seq 1 200); do
  marker=$(ls -d training_history/runs/*ab_wide_k160* 2>/dev/null | tail -1)
  if [ -n "$marker" ] && grep -q "广度对照结束" "$marker/output.log" 2>/dev/null; then
    echo "上一场已结束 ($(date +%H:%M:%S))"; break
  fi
  sleep 30
done

echo ""
echo "############ 衰减随机出招(A) vs 静止(B)  32 局配对  $(date +%H:%M:%S)"
$PY -u code/test_match/ab_search_modes.py --paired-sides --seeds $SEEDS --jobs 8 \
    --out-dir training_history/ab_rollout_decay_vs_idle \
    --flat-env="$A_DECAY" --mcts-env="$B_IDLE"
echo ""
echo "衰减随机对照结束 ($(date +%H:%M:%S))"
