#!/usr/bin/env bash
# Q 网偏置叠加版 + 游标48 组合 vs pin（terminal Q 网，2026-11-02）。
# 3 臂（terminal 只）：
#   combo        = AZAI_QNET_CLASS_PRIOR=1 + AZAI_QNET_S2_MIX=<W>（叠加版）
#   steps48prior = AZAI_CAND_STEPS=48 + AZAI_QNET_CLASS_PRIOR=1（游标放宽 + 类先验）
#   steps48mix   = AZAI_CAND_STEPS=48 + AZAI_QNET_S2_MIX=<W>（游标放宽 + s2 混合）
# B 臂 = AZAI_CAND_MODE=pin（现役）。其余 = 部署 BASE（_us_vs_external.sh:68 逐字）。
#
# 用法：
#   SEEDS="100 101" W="0.5" bash code/test_match/_qnet_combo_ab.sh    # 4 局/臂看方向
#   W 默认 0.5（应改成小批量里表现较好的那个 w）
set -uo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO" || exit 1
export PYTHONIOENCODING=utf-8
P="${PYTHON:-D:/anaconda3/envs/pytorch-gpu/python.exe}"
JOBS="${JOBS:-2}"
SEEDS="${SEEDS:-100 101 102 103 104 105 106 107}"
RETRY="${RETRY:-1}"
GAME_TIMEOUT_S="${GAME_TIMEOUT_S:-10800}"
OUTROOT="match_results/qnet_combo_ab"
ARMS="${ARMS:-combo steps48prior steps48mix}"
W="${W:-0.5}"

BASE="$(sed -n 's/^BASE="\(.*\)"$/\1/p' code/test_match/_us_vs_external.sh)"
[ -n "$BASE" ] || { echo "[qcombo] !! 读不到 _us_vs_external.sh 的 BASE" >&2; exit 2; }
case "$BASE" in
    *AZAI_CAND_MODE*) echo "[qcombo] !! BASE 里已含 AZAI_CAND_MODE ⇒ 停" >&2; exit 2 ;;
esac

mkdir -p "$OUTROOT"
echo "=============================================================="
echo "[qcombo] 叠加版 + 游标48 组合 vs pin+random   arms=[$ARMS]  w=$W"
echo "[qcombo] seeds=[$SEEDS]  jobs=$JOBS  retry=$RETRY"
echo "[qcombo] BASE 与 _us_vs_external.sh:68 逐字相同（${#BASE} 字符）"
echo "[qcombo] $(date '+%F %T')"
echo "=============================================================="

for ARM in $ARMS; do
    CKPT="training_history/qnet_action/qnet_terminal.pt"
    case "$ARM" in
        combo)        QEXTRA=";AZAI_QNET_CLASS_PRIOR=1;AZAI_QNET_S2_MIX=$W"; STEPS="" ;;
        steps48prior) QEXTRA=";AZAI_QNET_CLASS_PRIOR=1";                      STEPS=";AZAI_CAND_STEPS=48" ;;
        steps48mix)   QEXTRA=";AZAI_QNET_S2_MIX=$W";                          STEPS=";AZAI_CAND_STEPS=48" ;;
        *) echo "[qcombo] !! ARM=$ARM 不认识（combo|steps48prior|steps48mix）" >&2; exit 2 ;;
    esac
    TAG="qcombo_${ARM}"
    OUTDIR="$OUTROOT/${ARM}"
    mkdir -p "$OUTDIR"
    FLAT_ENV="$BASE;AZAI_CAND_MODE=unified;AZAI_CAND_VERBOSE=1;AZAI_QNET_CKPT=$CKPT$QEXTRA$STEPS"
    # 对手 = 原版 unified（只差那个 Q 网/游标开关；BASE 其余逐字一致）⇒ 直接看 delta
    MCTS_ENV="$BASE;AZAI_CAND_MODE=unified;AZAI_CAND_VERBOSE=1"
    export MVS_STDERR_SUFFIX=1
    export AB_SEARCH_SUFFIX_TAG="_$TAG"
    echo ""
    echo "[qcombo] ── $ARM → $OUTDIR ──"
    echo "[qcombo] A env: ... AZAI_QNET_CKPT=$CKPT$QEXTRA$STEPS"
    "$P" -u code/test_match/ab_search_modes.py \
        --seeds $SEEDS --jobs "$JOBS" --paired-sides \
        --out-dir "$OUTDIR" \
        --flat-env "$FLAT_ENV" --mcts-env "$MCTS_ENV" \
        --resume --retry-invalid "$RETRY" --game-timeout-s "$GAME_TIMEOUT_S" \
        2>&1 | tee "$OUTDIR/_run.log"
    rc=${PIPESTATUS[0]}
    echo "[qcombo] $ARM 结束 rc=$rc  $(date '+%F %T')"
done

echo ""
echo "[qcombo] 分析："
for ARM in $ARMS; do
    echo "  $P -u code/test_match/analyze_qnet_prior_ab.py --out-dir $OUTROOT/${ARM} --stderr-tag qcombo_${ARM} --label-a 'combo($ARM-w$W)' --seeds $(echo $SEEDS | tr ' ' ',') --json $OUTROOT/${ARM}/summary.json"
done
