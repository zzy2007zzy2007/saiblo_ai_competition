#!/usr/bin/env bash
# 严格模式"干净局"回归（快档；2026-10-27 追加）
#
# 背景：机器上同时有 8 局 A/B 在跑（parent 的 rng_ab_controlled），canonical 配置的一局要
# 30-40 分钟，且**会被外部清理杀掉**（`illegal_strict_smoke` 那次 exit=127 就是 18:40:47 被
# 外部 kill 的，只剩 1 局跑到 round 295）。所以这里用"canonical 的低配版"把墙钟压到 3-5 分钟/局：
#   ITERS 256→64、K 24→12、SAMPLE_MULT 15→4，其余（mode/pospin/mc_every/horizon/rootonly/
#   skip1/tclass/tpos/temp/posprior）与 `ab_search_modes.py:DEFAULT_MCTS_ENV` 一致。
# ⚠️ 口径：**op 的合法性与搜索强度无关**，这份配置只用来验证"没有非法操作 + 判词=engine"；
# 部署口径的读数由 `illegal_strict_smoke`（canonical）+ 120 局历史重放审计负责。
set -uo pipefail
cd "$(dirname "$0")/../.." || exit 1
PY=D:/anaconda3/envs/pytorch-gpu/python.exe
RU="其他版本ai/saiblo-30th-AI/Game1/antgame_ai_cpp/cpp_lure_v4/build/ai_cpp_lure_v4.exe"
BR=code/test_match/az_bridge_ai.py
export MVS_TIMEOUT_SECONDS=900

LITE="AZAI_CKPT=checkpoints/posnet_A_k5_m32.pt;AZAI_ITERS=64;AZAI_DEPTH=4;AZAI_K=12;\
AZAI_SAMPLE_MULT=4;AZAI_MODE=pos-only;AZAI_POSPIN=mc_quiet;AZAI_MC_EVERY=5;\
AZAI_MC_HORIZON=256;AZAI_MC_ROOTONLY=1;AZAI_SKIP1=1;AZAI_TCLASS=0.5;AZAI_TPOS=1.0;\
AZAI_TEMP=1e-6;AZAI_POSPRIOR=off"

export DSH_PROGRESS_TAG=illegal_strict_lite
bash code/run_logged.sh illegal_strict_lite \
  "$PY" -u code/test_match/match_via_sdk.py \
    --ai0="$BR" --ai1="$RU" --ai0-env="$LITE" 8201 8201r 8202 8202r

export DSH_PROGRESS_TAG=illegal_cold_lite
bash code/run_logged.sh illegal_cold_lite_control \
  "$PY" -u code/test_match/match_via_sdk.py --cold-illegal \
    --ai0="$BR" --ai1="$RU" --ai0-env="$LITE" 8201 8201r

echo "=== _illegal_smoke_lite.sh 结束 $(date '+%Y-%m-%d %H:%M:%S') ==="
