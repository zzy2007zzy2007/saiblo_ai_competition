#!/usr/bin/env bash
# 对手 = **亚军本体**（ai_cpp_lure_v4.exe），我们 = **flat + MC_EVERY=1 + MC_CELLS=4**（当前最强臂）。
#
# 为什么是 cells4 而不是 A4：
#   · A4（flat/every1/cells0）打 rule_v4 = 63-1-0 = 1.000，分辨率已耗尽；
#   · T2 内部对打（cells4 vs cells0）早期 10 胜 2 负、血量 +8.08（95%CI [+3.0,+13.2]）⇒ cells4 更强。
#
# ⚠️ 标定基准（2026-10-21 用户澄清）：**rule_v4 不是官方示例，是用户本人参赛的 AI，最终第 16 名（共 50）**。
#   所以：0.656/0.7344 ≈ 16 名档；A4 的 1.000 = 首次明显越过这一档。
#
# 32 局 = 16 seed × 双向镜像；8 并行。每局约 20–30 分钟（双方各 ~10–12 分钟搜索）
# ⇒ 4 波 ≈ 2 小时。
set -uo pipefail
cd "$(dirname "$0")/../.." || exit 1
PY=D:/anaconda3/envs/pytorch-gpu/python.exe
RUNNERUP="其他版本ai/saiblo-30th-AI/Game1/antgame_ai_cpp/cpp_lure_v4/build/ai_cpp_lure_v4.exe"
SEEDS="7 7r 8 8r 9 9r 10 10r 11 11r 12 12r 13 13r 14 14r 15 15r 16 16r 17 17r 18 18r 19 19r 20 20r 21 21r 22 22r"

# ── 等 T1/T2 结束 ─────────────────────────────────────────────────────────
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
export AZAI_MC_EVERY=1 AZAI_MC_HORIZON=256 AZAI_MC_ROOTONLY=1
export AZAI_MC_CELLS=4
export AZAI_SKIP1=1 AZAI_TCLASS=0.5 AZAI_TPOS=1.0 AZAI_TEMP=1e-6
export AZAI_POSPRIOR=off AZAI_TRACE=1 AZAI_VERIFY=1
export AZAI_SEARCH=flat
export AZAI_FLAT_ROLLOUTS=16 AZAI_FLAT_HORIZON=64 AZAI_FLAT_BATCH=32 AZAI_FLAT_ROUNDS=39000

echo ""
echo "############ flat + every1 + CELLS4  vs 亚军（ai_cpp_lure_v4）32 局  $(date +%H:%M:%S)"
$PY -u _tmp_ladder.py --tag="VS_RUNNERUP_cells4" --jobs=8 \
    --ai0=code/test_match/az_bridge_ai.py --ai1="$RUNNERUP" \
    $SEEDS
echo ""
echo "对亚军（cells4）测试结束 ($(date +%H:%M:%S))"
