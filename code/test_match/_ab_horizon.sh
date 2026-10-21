#!/usr/bin/env bash
# MC 仲裁 rollout 水平（AZAI_MC_HORIZON）扫描：32 / 64 / 128，对照 = A3 的 H256。
#
# 动机（用户 2026-10-21）：现在 H=256，但 mid-game 到终局只有两百多回合 ⇒ 256 很可能已经
# 等价于"推到终局"。若 32/64 的效果一样，则每个仲裁回合能省 4–8 倍 rollout 时间。
# 历史：docs/experiment_log.md 2026-10-10 有 AB_TAIL_T32/T64 两批，但 result 均为 `_待填_`
# 且 run 目录已被清理 ⇒ 实质等于没测过。
#
# 设计：只改 AZAI_MC_HORIZON，其余与 A3（flat + MC_EVERY=5）完全一致 ⇒ 单变量。
# 每个臂 64 局（32 seed × 双向镜像），vs rule_v4（外部锚点，与四臂同尺子）。
# 对照：A3_flat_every5 = H256（同种子同对手，已在四臂里跑完）。
set -uo pipefail
cd "$(dirname "$0")/../.." || exit 1
PY=D:/anaconda3/envs/pytorch-gpu/python.exe
SEEDS="7 7r 8 8r 9 9r 10 10r 11 11r 12 12r 13 13r 14 14r 15 15r 16 16r 17 17r 18 18r 19 19r 20 20r 21 21r 22 22r 23 23r 24 24r 25 25r 26 26r 27 27r 28 28r 29 29r 30 30r 31 31r 32 32r 33 33r 34 34r 35 35r 36 36r 37 37r 38 38r"

# ── 等四臂实验结束（避免超订 CPU）──────────────────────────────────────────
echo "等待 ab_2x2_mc_coverage 结束… ($(date +%H:%M:%S))"
for _ in $(seq 1 240); do
  marker=$(ls -d training_history/runs/*ab_2x2_mc_coverage* 2>/dev/null | tail -1)
  if [ -n "$marker" ] && grep -q "全部四臂结束" "$marker/output.log" 2>/dev/null; then
    echo "四臂已结束，开始水平扫描 ($(date +%H:%M:%S))"
    break
  fi
  sleep 30
done

export AZAI_CKPT=checkpoints/posnet_A_k5_m32.pt
export AZAI_ITERS=256 AZAI_DEPTH=4 AZAI_K=24 AZAI_SAMPLE_MULT=15
export AZAI_MODE=pos-only AZAI_POSPIN=mc_quiet
export AZAI_MC_EVERY=5 AZAI_MC_ROOTONLY=1
export AZAI_SKIP1=1 AZAI_TCLASS=0.5 AZAI_TPOS=1.0 AZAI_TEMP=1e-6
export AZAI_POSPRIOR=off AZAI_TRACE=1 AZAI_VERIFY=1
export AZAI_SEARCH=flat
export AZAI_FLAT_ROLLOUTS=16 AZAI_FLAT_HORIZON=64
export AZAI_FLAT_BATCH=32 AZAI_FLAT_ROUNDS=39000

for h in 32 64 128; do
  export AZAI_MC_HORIZON=$h
  echo ""
  echo "############################################################"
  echo "## HORIZON $h : AZAI_MC_HORIZON=$h  ($(date +%H:%M:%S))"
  echo "############################################################"
  $PY -u _tmp_ladder.py --tag="HZ_h${h}" --jobs=8 \
      --ai0=code/test_match/az_bridge_ai.py --ai1=code/test_match/rv4_pkg/main.py \
      $SEEDS
done
echo ""
echo "水平扫描结束 ($(date +%H:%M:%S))"
