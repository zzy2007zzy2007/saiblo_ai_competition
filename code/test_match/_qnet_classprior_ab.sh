#!/usr/bin/env bash
# 实验 A：Q 网当类先验（AZAI_QNET_CLASS_PRIOR=1）vs pin+flat A/B（2026-11-02）。
# 两个 Q 网 checkpoint 并排：terminal / kgeo50。
#
# A 臂 = `AZAI_CAND_MODE=unified` + `AZAI_QNET_CKPT=<qnet>` + `AZAI_QNET_CLASS_PRIOR=1`
#   （bandit 的 Q[c] 初始 = zscore(Q(s,a_c))，见 unified_cand.py:_reset_decision）。
# B 臂 = `AZAI_CAND_MODE=pin` + BASE 自带 random rollout（现役）。
# 其余 = 部署 BASE（`_us_vs_external.sh:68` 逐字相同）。阶段1/阶段2 参数两臂一致。
#
# 用法：
#   # 小批量（4 局 = 2 seeds × 2 座位）
#   SEEDS="100 101" bash code/test_match/_qnet_classprior_ab.sh
#   # 16 局（8 seeds × 2 座位；--resume 复用已跑局）
#   bash code/test_match/_qnet_classprior_ab.sh
#   日志：bash code/run_logged.sh qnet_classprior_ab bash code/test_match/_qnet_classprior_ab.sh
set -uo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO" || exit 1
export PYTHONIOENCODING=utf-8
P="${PYTHON:-D:/anaconda3/envs/pytorch-gpu/python.exe}"
JOBS="${JOBS:-4}"
SEEDS="${SEEDS:-100 101 102 103 104 105 106 107}"
RETRY="${RETRY:-1}"
GAME_TIMEOUT_S="${GAME_TIMEOUT_S:-10800}"
OUTROOT="match_results/qnet_classprior_ab"
QNETS="${QNETS:-terminal kgeo50}"

BASE="$(sed -n 's/^BASE="\(.*\)"$/\1/p' code/test_match/_us_vs_external.sh)"
[ -n "$BASE" ] || { echo "[qcp] !! 读不到 _us_vs_external.sh 的 BASE" >&2; exit 2; }
case "$BASE" in
    *AZAI_CAND_MODE*) echo "[qcp] !! BASE 里已含 AZAI_CAND_MODE ⇒ 停" >&2; exit 2 ;;
esac

mkdir -p "$OUTROOT"
echo "=============================================================="
echo "[qcp] 实验A（Q 网类先验）vs pin+random   qnets=[$QNETS]"
echo "[qcp] seeds=[$SEEDS]  jobs=$JOBS  retry=$RETRY"
echo "[qcp] BASE 与 _us_vs_external.sh:68 逐字相同（${#BASE} 字符）"
echo "[qcp] $(date '+%F %T')"
echo "=============================================================="

for QNET in $QNETS; do
    CKPT="training_history/qnet_action/qnet_${QNET}.pt"
    TAG="qcp_${QNET}"
    OUTDIR="$OUTROOT/${QNET}"
    mkdir -p "$OUTDIR"
    FLAT_ENV="$BASE;AZAI_CAND_MODE=unified;AZAI_CAND_VERBOSE=1;AZAI_QNET_CKPT=$CKPT;AZAI_QNET_CLASS_PRIOR=1"
    MCTS_ENV="$BASE;AZAI_CAND_MODE=pin;AZAI_CAND_VERBOSE=1"
    export MVS_STDERR_SUFFIX=1
    export AB_SEARCH_SUFFIX_TAG="_$TAG"
    echo ""
    echo "[qcp] ── $QNET → $OUTDIR  ckpt=$CKPT ──"
    echo "[qcp] A env: ... AZAI_QNET_CKPT=$CKPT AZAI_QNET_CLASS_PRIOR=1"
    "$P" -u code/test_match/ab_search_modes.py \
        --seeds $SEEDS --jobs "$JOBS" --paired-sides \
        --out-dir "$OUTDIR" \
        --flat-env "$FLAT_ENV" --mcts-env "$MCTS_ENV" \
        --resume --retry-invalid "$RETRY" --game-timeout-s "$GAME_TIMEOUT_S" \
        2>&1 | tee "$OUTDIR/_run.log"
    rc=${PIPESTATUS[0]}
    echo "[qcp] $QNET 结束 rc=$rc  $(date '+%F %T')"
done

echo ""
echo "[qcp] 分析："
echo "  $P -u code/test_match/analyze_qnet_prior_ab.py --out-dir $OUTROOT/terminal --stderr-tag qcp_terminal --label-a 'A(classprior-term)' --seeds $(echo $SEEDS | tr ' ' ',') --json $OUTROOT/terminal/summary.json"
echo "  $P -u code/test_match/analyze_qnet_prior_ab.py --out-dir $OUTROOT/kgeo50  --stderr-tag qcp_kgeo50  --label-a 'A(classprior-kgeo50)' --seeds $(echo $SEEDS | tr ' ' ',') --json $OUTROOT/kgeo50/summary.json"
