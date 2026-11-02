#!/usr/bin/env bash
# 实验 B：s2 分数混合 Q 网值（AZAI_QNET_S2_MIX=w）vs pin+flat A/B（2026-11-02）。
# 两个 Q 网 checkpoint（terminal / kgeo50）× 权重 w 扫描。
#
# A 臂 = `AZAI_CAND_MODE=unified` + `AZAI_QNET_CKPT=<qnet>` + `AZAI_QNET_S2_MIX=<w>`
#   （阶段2 排序分 s2' = zscore(s2) + w×zscore(Q(s,a_candidate))，见 flat_rollout_greedy.py:search）。
# B 臂 = `AZAI_CAND_MODE=pin` + BASE 自带 random rollout（现役）。
# 其余 = 部署 BASE（`_us_vs_external.sh:68` 逐字相同）。阶段1/阶段2 参数两臂一致。
#
# 用法：
#   # 小批量（4 局 = 2 seeds × 2 座位），只扫 w=0.3,1.0 两个 Q 网
#   SEEDS="100 101" W="0.3 1.0" bash code/test_match/_qnet_s2mix_ab.sh
#   # 全 w 扫描（4 局）
#   SEEDS="100 101" bash code/test_match/_qnet_s2mix_ab.sh
#   # 16 局补某个 (qnet,w)：QNETS=terminal W=0.5 bash code/test_match/_qnet_s2mix_ab.sh
#   日志：bash code/run_logged.sh qnet_s2mix_ab bash code/test_match/_qnet_s2mix_ab.sh
set -uo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO" || exit 1
export PYTHONIOENCODING=utf-8
P="${PYTHON:-D:/anaconda3/envs/pytorch-gpu/python.exe}"
JOBS="${JOBS:-4}"
SEEDS="${SEEDS:-100 101 102 103 104 105 106 107}"
RETRY="${RETRY:-1}"
GAME_TIMEOUT_S="${GAME_TIMEOUT_S:-10800}"
OUTROOT="match_results/qnet_s2mix_ab"
QNETS="${QNETS:-terminal kgeo50}"
W="${W:-0.1 0.3 0.5 1.0}"

BASE="$(sed -n 's/^BASE="\(.*\)"$/\1/p' code/test_match/_us_vs_external.sh)"
[ -n "$BASE" ] || { echo "[qsm] !! 读不到 _us_vs_external.sh 的 BASE" >&2; exit 2; }
case "$BASE" in
    *AZAI_CAND_MODE*) echo "[qsm] !! BASE 里已含 AZAI_CAND_MODE ⇒ 停" >&2; exit 2 ;;
esac

mkdir -p "$OUTROOT"
echo "=============================================================="
echo "[qsm] 实验B（s2 混合 Q 网值）vs pin+random   qnets=[$QNETS]  w=[$W]"
echo "[qsm] seeds=[$SEEDS]  jobs=$JOBS  retry=$RETRY"
echo "[qsm] BASE 与 _us_vs_external.sh:68 逐字相同（${#BASE} 字符）"
echo "[qsm] $(date '+%F %T')"
echo "=============================================================="

for QNET in $QNETS; do
  for WW in $W; do
    CKPT="training_history/qnet_action/qnet_${QNET}.pt"
    TAG="qsm_${QNET}_w${WW}"
    OUTDIR="$OUTROOT/${QNET}_w${WW}"
    mkdir -p "$OUTDIR"
    FLAT_ENV="$BASE;AZAI_CAND_MODE=unified;AZAI_CAND_VERBOSE=1;AZAI_QNET_CKPT=$CKPT;AZAI_QNET_S2_MIX=$WW"
    MCTS_ENV="$BASE;AZAI_CAND_MODE=pin;AZAI_CAND_VERBOSE=1"
    export MVS_STDERR_SUFFIX=1
    export AB_SEARCH_SUFFIX_TAG="_$TAG"
    echo ""
    echo "[qsm] ── $QNET w=$WW → $OUTDIR  ckpt=$CKPT ──"
    echo "[qsm] A env: ... AZAI_QNET_CKPT=$CKPT AZAI_QNET_S2_MIX=$WW"
    "$P" -u code/test_match/ab_search_modes.py \
        --seeds $SEEDS --jobs "$JOBS" --paired-sides \
        --out-dir "$OUTDIR" \
        --flat-env "$FLAT_ENV" --mcts-env "$MCTS_ENV" \
        --resume --retry-invalid "$RETRY" --game-timeout-s "$GAME_TIMEOUT_S" \
        2>&1 | tee "$OUTDIR/_run.log"
    rc=${PIPESTATUS[0]}
    echo "[qsm] $QNET w=$WW 结束 rc=$rc  $(date '+%F %T')"
  done
done

echo ""
echo "[qsm] 分析（每个 (qnet,w) 一条）："
for QNET in $QNETS; do
  for WW in $W; do
    echo "  $P -u code/test_match/analyze_qnet_prior_ab.py --out-dir $OUTROOT/${QNET}_w${WW} --stderr-tag qsm_${QNET}_w${WW} --label-a 'B(s2mix-$QNET-w$WW)' --seeds $(echo $SEEDS | tr ' ' ',') --json $OUTROOT/${QNET}_w${WW}/summary.json"
  done
done
