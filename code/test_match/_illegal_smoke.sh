#!/usr/bin/env bash
# 非法操作严格模式回归（2026-10-27；报告 docs/illegal_op_strict_mode_20261028.md）
#
# 目的：默认 `cold_handle_rule_illegal=False`（= 官方判题口径）之后，用真实对局验证
#   ① 没有非法操作（`illegal_total=0`）；
#   ② 每局判词都是 `verdict=engine`（跑到引擎真终局，不是超时/中断）；
#   ③ 严格模式真的抓得住非法操作（故障注入 2 局：严格 vs 旧行为）；
#   ④ 改动不改变结果（同 seed 的 cold 对照局应与严格局逐字段一致）。
#
# 每批都走 `code/run_logged.sh` ⇒ `training_history/runs/<ts>_<name>/{cmd.txt,meta.txt,output.log}`。
#
# 用法： bash code/test_match/_illegal_smoke.sh          # 全部 4 批（顺序跑，避免抢 CPU）
#        bash code/test_match/_illegal_smoke.sh fast     # 只跑故障注入那批（2 分钟级）
set -uo pipefail
cd "$(dirname "$0")/../.." || exit 1
PY=D:/anaconda3/envs/pytorch-gpu/python.exe
RU="其他版本ai/saiblo-30th-AI/Game1/antgame_ai_cpp/cpp_lure_v4/build/ai_cpp_lure_v4.exe"
BR=code/test_match/az_bridge_ai.py
ONLY="${1:-all}"

# 机器上还有别的对局在跑 ⇒ 把单次决策的读超时放宽（它只是"卡死检测"门限，不影响判词）
export MVS_TIMEOUT_SECONDS=900

# 部署口径的 canonical 配置（与 code/test_match/ab_search_modes.py 的 DEFAULT_MCTS_ENV 逐字相同）
CANON="AZAI_CKPT=checkpoints/posnet_A_k5_m32.pt;AZAI_ITERS=256;AZAI_DEPTH=4;AZAI_K=24;\
AZAI_SAMPLE_MULT=15;AZAI_MODE=pos-only;AZAI_POSPIN=mc_quiet;AZAI_MC_EVERY=5;\
AZAI_MC_HORIZON=256;AZAI_MC_ROOTONLY=1;AZAI_SKIP1=1;AZAI_TCLASS=0.5;AZAI_TPOS=1.0;\
AZAI_TEMP=1e-6;AZAI_POSPRIOR=off"

# 便宜配置 + **把高风险改写路径全部打开**：候选注入（oracle）、去降级（no_downgrade）、
# 显式升级候选（mc_upg）、HOLD（light 门槛）。搜索故意调小 ⇒ 只测"op 合不合法"。
HIGHRISK="AZAI_CKPT=checkpoints/posnet_A_k5_m32.pt;AZAI_ITERS=32;AZAI_K=8;AZAI_SAMPLE_MULT=1;\
AZAI_MODE=pos-only;AZAI_POSPIN=argmax;AZAI_SKIP1=1;AZAI_TEMP=1e-6;AZAI_SEARCH=flat;\
AZAI_FLAT_ROLLOUTS=4;AZAI_FLAT_HORIZON=8;AZAI_FLAT_BATCH=8;AZAI_FLAT_ROUNDS=39000;\
AZAI_MC_UPG=1;AZAI_NO_DOWNGRADE=1;AZAI_ORACLE_INJECT=${RU};AZAI_ORACLE_TIMEOUT=2.5"

run() {   # run <实验名> <命令...>
  bash code/run_logged.sh "$@"
}

if [ "$ONLY" = "all" ] || [ "$ONLY" = "fast" ]; then
  # ── 批 4（先跑，2 分钟级）：故障注入 —— 证明"严格模式真的判负 / 旧行为真的吞掉" ──────
  run illegal_fault_inject_strict \
    "$PY" -u code/test_match/match_via_sdk.py --rounds=15 \
      --ai0="$RU" --ai1=code/test_match/faulty_ai.py 7501
  run illegal_fault_inject_cold \
    "$PY" -u code/test_match/match_via_sdk.py --cold-illegal --rounds=15 \
      --ai0="$RU" --ai1=code/test_match/faulty_ai.py 7501
fi

if [ "$ONLY" = "all" ]; then
  # ── 批 1：严格模式 × 部署配置 × 4 局（2 张图 × 2 个座位）vs 亚军 ────────────────────
  export DSH_PROGRESS_TAG=illegal_strict_smoke
  run illegal_strict_smoke \
    "$PY" -u code/test_match/match_via_sdk.py \
      --ai0="$BR" --ai1="$RU" --ai0-env="$CANON" 8101 8101r 8102 8102r

  # ── 批 2：严格模式 × 高风险改写路径（注入/去降级/显式升级）× 2 局 ────────────────────
  export DSH_PROGRESS_TAG=illegal_strict_highrisk
  run illegal_strict_highrisk \
    "$PY" -u code/test_match/match_via_sdk.py \
      --ai0="$BR" --ai1="$RU" --ai0-env="$HIGHRISK" 8103 8103r

  # ── 批 3：旧行为（cold=True）对照 —— 同一个 seed，用来证明"0 非法时结果逐字段相同" ────
  export DSH_PROGRESS_TAG=illegal_cold_control
  run illegal_cold_control \
    "$PY" -u code/test_match/match_via_sdk.py --cold-illegal \
      --ai0="$BR" --ai1="$RU" --ai0-env="$CANON" 8101
fi

echo "=== _illegal_smoke.sh 结束 $(date '+%Y-%m-%d %H:%M:%S') ==="
