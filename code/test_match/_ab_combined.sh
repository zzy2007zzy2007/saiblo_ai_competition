#!/usr/bin/env bash
# 组合臂：**flat + MC_EVERY=1 + MC_HORIZON=64**，vs rule_v4，64 局。
#
# 动机：两个旋钮各自都被测出正向增益
#   · flat + MC_EVERY=1（H=256）= 1.000（A4，63-1-0；但 rule_v4 已见天花板）
#   · flat + MC_EVERY=5 削 H 到 32/64 = 0.953 / 0.970（HZ_h32 / HZ_h64，原 0.891）
# ⇒ 很可能叠加，值得单独跑一臂。
# 对照：A3（flat/e5/H256）=0.891、A4（flat/e1/H256）=1.000、HZ_h64（flat/e5/H64）=0.970。
set -uo pipefail
cd "$(dirname "$0")/../.." || exit 1
PY=D:/anaconda3/envs/pytorch-gpu/python.exe
SEEDS="7 7r 8 8r 9 9r 10 10r 11 11r 12 12r 13 13r 14 14r 15 15r 16 16r 17 17r 18 18r 19 19r 20 20r 21 21r 22 22r 23 23r 24 24r 25 25r 26 26r 27 27r 28 28r 29 29r 30 30r 31 31r 32 32r 33 33r 34 34r 35 35r 36 36r 37 37r 38 38r"

echo "等待 T1/T2 结束… ($(date +%H:%M:%S))"
for _ in $(seq 1 400); do
  marker=$(ls -d training_history/runs/*ab_t1t2* 2>/dev/null | tail -1)
  if [ -n "$marker" ] && grep -q "T1/T2 全部结束" "$marker/output.log" 2>/dev/null; then
    echo "T1/T2 已结束 ($(date +%H:%M:%S))"
    break
  fi
  sleep 30
done

export AZAI_CKPT=checkpoints/posnet_A_k5_m32.pt
export AZAI_ITERS=256 AZAI_DEPTH=4 AZAI_K=24 AZAI_SAMPLE_MULT=15
export AZAI_MODE=pos-only AZAI_POSPIN=mc_quiet
export AZAI_MC_EVERY=1 AZAI_MC_HORIZON=64 AZAI_MC_ROOTONLY=1
export AZAI_SKIP1=1 AZAI_TCLASS=0.5 AZAI_TPOS=1.0 AZAI_TEMP=1e-6
export AZAI_POSPRIOR=off AZAI_TRACE=1 AZAI_VERIFY=1
export AZAI_SEARCH=flat
export AZAI_FLAT_ROLLOUTS=16 AZAI_FLAT_HORIZON=64 AZAI_FLAT_BATCH=32 AZAI_FLAT_ROUNDS=39000

echo ""
echo "############ COMBO: flat + MC_EVERY=1 + MC_HORIZON=64  ($(date +%H:%M:%S))"
$PY -u _tmp_ladder.py --tag="COMBO_flat_e1_h64" --jobs=8 \
    --ai0=code/test_match/az_bridge_ai.py --ai1=code/test_match/rv4_pkg/main.py \
    $SEEDS
echo ""
echo "组合臂结束 ($(date +%H:%M:%S))"
