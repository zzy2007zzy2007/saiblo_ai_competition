#!/usr/bin/env bash
# 严格模式回归：**极快档**（2026-10-27 追加；机器被 8 局 A/B 占满时用）
#
# 为什么还要一档更快的：canonical 一局 30-40 分钟、mid 档 ~25 分钟，而机器上同时有 parent 的
# 8 局 A/B 在跑（我的 canonical 批已在 18:40:47 被外部清理杀掉一次，exit=127）。
# 这一档把搜索压到"一次前向 + 极浅树"，**一局 1 分钟以内**，专门用来满足
# "严格模式 4 局、每局 verdict=engine、illegal_total=0" 这条判据；
# 部署口径的读数由 `illegal_strict_smoke`(canonical) / `illegal_strict_highrisk` /
# 120 局历史重放审计负责。**op 的合法性与搜索强度无关。**
set -uo pipefail
cd "$(dirname "$0")/../.." || exit 1
PY=D:/anaconda3/envs/pytorch-gpu/python.exe
RU="其他版本ai/saiblo-30th-AI/Game1/antgame_ai_cpp/cpp_lure_v4/build/ai_cpp_lure_v4.exe"
BR=code/test_match/az_bridge_ai.py
export MVS_TIMEOUT_SECONDS=900

FAST="AZAI_CKPT=checkpoints/posnet_A_k5_m32.pt;AZAI_ITERS=8;AZAI_DEPTH=3;AZAI_K=4;\
AZAI_SAMPLE_MULT=1;AZAI_MODE=pos-only;AZAI_POSPIN=argmax;AZAI_SKIP1=0;\
AZAI_TCLASS=0.5;AZAI_TPOS=1.0;AZAI_TEMP=1e-6;AZAI_MC_EVERY=999;AZAI_POSPRIOR=off"

export DSH_PROGRESS_TAG=illegal_strict_fast
bash code/run_logged.sh illegal_strict_fast \
  "$PY" -u code/test_match/match_via_sdk.py \
    --ai0="$BR" --ai1="$RU" --ai0-env="$FAST" 8301 8301r 8302 8302r

export DSH_PROGRESS_TAG=illegal_cold_fast
bash code/run_logged.sh illegal_cold_fast_control \
  "$PY" -u code/test_match/match_via_sdk.py --cold-illegal \
    --ai0="$BR" --ai1="$RU" --ai0-env="$FAST" 8301 8301r

echo "=== _illegal_smoke_fast.sh 结束 $(date '+%Y-%m-%d %H:%M:%S') ==="
