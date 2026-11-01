#!/usr/bin/env bash
# shape_pin vs pin+flat A/B（2026-11-02）：16 局 = seeds 100..107 × 2 座位（镜像配对）。
#
# 两臂只差 `AZAI_CAND_MODE`（A=shape_pin，B=pin）；其余 = 部署 BASE（`_us_vs_external.sh:68`
# **逐字相同**，脚本里有断言）。阶段 2 = flat rollout 评估器，两臂同一套参数
# （rollouts=64/horizon=16/batch=32/random+opponent+decay(1,3)/rounds=49152/seed=53/
#  share_prefix=1/eval=net/k=48），由 shape_pin 侧 stderr 里的
# `[shape_pin] engine=flat ...` 行证明（与 pin/unified 的 stderr 同款）。
#
# 驱动 = `ab_search_modes.py`（它用 `match_via_sdk.py` 把 ai0/ai1 都指 az_bridge_ai.py，
# 各自带 `--ai0-env` / `--ai1-env`）：
#   · A 臂（shape_pin）= `--flat-env`（ab_search_modes 里叫 "flat"，本批 A 臂坐该槽）；
#   · B 臂（pin）      = `--mcts-env`。
#   · `--paired-sides`：每个 seed 打两局，A 侧各坐 P0/P1 ⇒ 抵消先手偏。
#
# 用法（长任务用 WMI 起 detached，见本文件末尾注释 / docs/shape_pin_mode_20261102.md）：
#   bash code/run_logged.sh shape_pin_ab bash code/test_match/_shape_pin_ab.sh
set -uo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO" || exit 1
export PYTHONIOENCODING=utf-8
P="${PYTHON:-D:/anaconda3/envs/pytorch-gpu/python.exe}"
JOBS="${JOBS:-6}"
SEEDS="${SEEDS:-100 101 102 103 104 105 106 107}"
RETRY="${RETRY:-2}"
GAME_TIMEOUT_S="${GAME_TIMEOUT_S:-10800}"
OUTDIR="match_results/shape_pin_ab"

# BASE：与 `_us_vs_external.sh:68` 逐字相同（历史批次可比性的锚点）
BASE="$(sed -n 's/^BASE="\(.*\)"$/\1/p' code/test_match/_us_vs_external.sh)"
[ -n "$BASE" ] || { echo "[shape_pin_ab] !! 读不到 _us_vs_external.sh 的 BASE" >&2; exit 2; }
case "$BASE" in *AZAI_CAND_MODE*) echo "[shape_pin_ab] !! BASE 里已含 AZAI_CAND_MODE ⇒ 停" >&2; exit 2 ;; esac

# 两臂只差这一处：A=shape_pin、B=pin。其余（含 AZAI_CAND_VERBOSE=1 只为日志）逐字相同。
FLAT_ENV="$BASE;AZAI_CAND_MODE=shape_pin;AZAI_CAND_VERBOSE=1"
MCTS_ENV="$BASE;AZAI_CAND_MODE=pin;AZAI_CAND_VERBOSE=1"

mkdir -p "$OUTDIR"
echo "=============================================================="
echo "[shape_pin_ab] A(shape_pin) vs B(pin+flat)   16 局 = $(echo $SEEDS | wc -w) seeds × 2 座位"
echo "[shape_pin_ab] seeds=[$SEEDS]  jobs=$JOBS  retry=$RETRY  game_timeout=${GAME_TIMEOUT_S}s"
echo "[shape_pin_ab] BASE 与 _us_vs_external.sh:68 逐字相同（${#BASE} 字符）"
echo "[shape_pin_ab] ckpt=checkpoints/posnet_A_k5_m32.pt  flat: rollouts=64/horizon=16/batch=32"
echo "[shape_pin_ab] $(date '+%F %T')"
echo "=============================================================="

# MVS_STDERR_SUFFIX=1：同 seed 两局（P0/P1 并发）的 stderr 必须带 _p0/_p1 后缀，否则互相覆盖
#   （match_via_sdk.py 只在该开关=1 时把 DSH_PROGRESS_SUFFIX 拼进 stderr 文件名）。
export MVS_STDERR_SUFFIX=1

"$P" -u code/test_match/ab_search_modes.py \
    --seeds $SEEDS --jobs "$JOBS" --paired-sides \
    --out-dir "$OUTDIR" \
    --flat-env "$FLAT_ENV" --mcts-env "$MCTS_ENV" \
    --resume --retry-invalid "$RETRY" --game-timeout-s "$GAME_TIMEOUT_S" \
    2>&1 | tee "$OUTDIR/_run.log"
rc=${PIPESTATUS[0]}
echo "[shape_pin_ab] ab_search_modes 结束 rc=$rc  $(date '+%F %T')"
exit "$rc"
