#!/usr/bin/env bash
# unified_noupg（候选层删升级） vs pin+flat A/B（2026-11-02）。
# 16 局 = seeds 100..107 × 2 座位（镜像配对）。
#
# A 臂 = `AZAI_CAND_MODE=unified_noupg` = 原 unified（阶段1 bandit+游标）+ **阶段1结束、top-k
#   已生成之后、阶段2之前**删掉含升级(op12)的候选（兜底：全删 ⇒ HOLD）。阶段1/阶段2 都一字不动。
# B 臂 = `AZAI_CAND_MODE=pin` + BASE 自带 `AZAI_FLAT_ROLLOUT_MODE=random`（现役）。
# 两臂唯一差别 = 候选生成（unified+删升级 vs pin）；flat rollout 参数 = 部署 BASE 逐字相同。
#
# 驱动 = `ab_search_modes.py`（ai0/ai1 都指 az_bridge_ai.py，各自 env）：
#   · A 臂 = `--flat-env`；B 臂 = `--mcts-env`；`--paired-sides` 抵消先手偏。
#
# 用法（WMI detached，见 docs/lightning_rollout_mode_20261102.md 末尾注释）：
#   bash code/run_logged.sh noupg_ab bash code/test_match/_noupg_ab.sh
set -uo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO" || exit 1
export PYTHONIOENCODING=utf-8
P="${PYTHON:-D:/anaconda3/envs/pytorch-gpu/python.exe}"
JOBS="${JOBS:-8}"
SEEDS="${SEEDS:-100 101 102 103 104 105 106 107}"
RETRY="${RETRY:-2}"
GAME_TIMEOUT_S="${GAME_TIMEOUT_S:-10800}"
OUTDIR="match_results/noupg_ab"

# BASE：与 `_us_vs_external.sh:68` 逐字相同
BASE="$(sed -n 's/^BASE="\(.*\)"$/\1/p' code/test_match/_us_vs_external.sh)"
[ -n "$BASE" ] || { echo "[noupg_ab] !! 读不到 _us_vs_external.sh 的 BASE" >&2; exit 2; }
case "$BASE" in
    *AZAI_CAND_MODE*) echo "[noupg_ab] !! BASE 里已含 AZAI_CAND_MODE ⇒ 停" >&2; exit 2 ;;
esac

FLAT_ENV="$BASE;AZAI_CAND_MODE=unified_noupg;AZAI_CAND_VERBOSE=1"
MCTS_ENV="$BASE;AZAI_CAND_MODE=pin;AZAI_CAND_VERBOSE=1"

mkdir -p "$OUTDIR"
echo "=============================================================="
echo "[noupg_ab] A(unified_noupg) vs B(pin+random)   16 局 = $(echo $SEEDS | wc -w) seeds × 2 座位"
echo "[noupg_ab] seeds=[$SEEDS]  jobs=$JOBS  retry=$RETRY  game_timeout=${GAME_TIMEOUT_S}s"
echo "[noupg_ab] BASE 与 _us_vs_external.sh:68 逐字相同（${#BASE} 字符）"
echo "[noupg_ab] A=unified 阶段1 后删含升级(op12)候选（兜底 HOLD）；B=pin；rollout 都 random"
echo "[noupg_ab] $(date '+%F %T')"
echo "=============================================================="

# 并发批次 stderr 唯一前缀（shape_pin 用 _p<side>，lightning 用 _lightning_p<side>，noupg 用 _noupg_p<side>）
export MVS_STDERR_SUFFIX=1
export AB_SEARCH_SUFFIX_TAG="_noupg"

"$P" -u code/test_match/ab_search_modes.py \
    --seeds $SEEDS --jobs "$JOBS" --paired-sides \
    --out-dir "$OUTDIR" \
    --flat-env "$FLAT_ENV" --mcts-env "$MCTS_ENV" \
    --resume --retry-invalid "$RETRY" --game-timeout-s "$GAME_TIMEOUT_S" \
    2>&1 | tee "$OUTDIR/_run.log"
rc=${PIPESTATUS[0]}
echo "[noupg_ab] ab_search_modes 结束 rc=$rc  $(date '+%F %T')"
exit "$rc"
