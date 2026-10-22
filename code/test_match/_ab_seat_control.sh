#!/usr/bin/env bash
# **自对弈座位控制**：A 与 B 配置**完全相同**，唯一目的是量化"先/后手"的座位效应。
#
# 为什么需要它（2026-10-22）：
#   三条内部对打的座位拆分都显示 **P1 系统性占优**：
#     · 仲裁评估器（64 局）：A 在 P0 = 40.6%，A 在 P1 = 71.9%  ⇒ +31pp
#     · T1+T2（128 局）   ：A 在 P0 = 54.7%，A 在 P1 = 73.4%  ⇒ +19pp
#   这两条臂的 A/B 配置都不同，所以不能排除"是策略交互而非座位"。
#   **自对弈（A≡B）是唯一干净的判定**：若 P1 仍赢 ~65%，座位效应坐实；若 ~50%，则上面是交互。
#
# 设计：16 张地图 × 2 个座位 = 32 局，`--paired-sides` ⇒ 同一张图打两遍、交换座位，
#       座位差在**图内**计算，地图方差直接消掉。
set -uo pipefail
cd "$(dirname "$0")/../.." || exit 1
PY=D:/anaconda3/envs/pytorch-gpu/python.exe
SEEDS=$(seq 700 715 | tr '\n' ' ')

SAME="AZAI_CKPT=checkpoints/posnet_A_k5_m32.pt;AZAI_ITERS=256;AZAI_DEPTH=4;AZAI_K=24;AZAI_SAMPLE_MULT=15;AZAI_MODE=pos-only;AZAI_POSPIN=mc_quiet;AZAI_MC_EVERY=1;AZAI_MC_HORIZON=256;AZAI_MC_ROOTONLY=1;AZAI_MC_CELLS=4;AZAI_SKIP1=1;AZAI_TCLASS=0.5;AZAI_TPOS=1.0;AZAI_TEMP=1e-6;AZAI_POSPRIOR=off;AZAI_SEARCH=flat;AZAI_FLAT_ROLLOUTS=16;AZAI_FLAT_HORIZON=64;AZAI_FLAT_BATCH=32;AZAI_FLAT_ROUNDS=39000"

echo "等待「有升级 vs 无升级」结束… ($(date +%H:%M:%S))"
for _ in $(seq 1 400); do
  marker=$(ls -d training_history/runs/*ab_upg1_vs_none* 2>/dev/null | tail -1)
  if [ -n "$marker" ] && grep -q "升级对照结束" "$marker/output.log" 2>/dev/null; then
    echo "上一场已结束 ($(date +%H:%M:%S))"
    break
  fi
  sleep 30
done

echo ""
echo "############ 自对弈座位控制：16 图 × 2 座位 = 32 局  $(date +%H:%M:%S)"
$PY -u code/test_match/ab_search_modes.py --paired-sides --seeds $SEEDS --jobs 5 \
    --out-dir training_history/ab_seat_control \
    --flat-env="$SAME" --mcts-env="$SAME"
echo ""
echo "座位控制结束 ($(date +%H:%M:%S))"
