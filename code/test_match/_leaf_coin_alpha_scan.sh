#!/usr/bin/env bash
# 叶子金币项 α 扫描：unified(+AZAI_LEAF_COIN_ALPHA=α) vs pin+flat（2026-11-02）。
#
# A 臂 = `AZAI_CAND_MODE=unified` + `AZAI_LEAF_COIN_ALPHA=<α>`（叶子评估加成
#   `V(leaf) + α×(my_coins/300)`，见 flat_rollout_greedy.py:_leaf_values）；
# B 臂 = `AZAI_CAND_MODE=pin`（现役 pin+flat，BASE 自带 random rollout）。
# 其余 = 部署 BASE（`_us_vs_external.sh:68` 逐字相同，脚本里有断言）。
#
# 初扫（默认）：每个 α 跑 2 seeds × 2 座位 = 4 局（seeds 100 101）；看方向后，
# 对有明显信号的 α 用 `SEEDS="100 101 102 103 104 105 106 107"` 补到 16 局（--resume 复用已跑局）。
#
# 并行纪律：`--jobs 8` 只在一个 α 的 4/16 局内并行；不同 α **串行**跑（本脚本 for 循环）。
# 用法：
#   bash code/run_logged.sh leaf_coin_alpha_scan bash code/test_match/_leaf_coin_alpha_scan.sh
#   单 α 补 16 局：ALPHAS="0.037" SEEDS="100 101 102 103 104 105 106 107" bash code/test_match/_leaf_coin_alpha_scan.sh
set -uo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO" || exit 1
export PYTHONIOENCODING=utf-8
P="${PYTHON:-D:/anaconda3/envs/pytorch-gpu/python.exe}"
JOBS="${JOBS:-8}"
SEEDS="${SEEDS:-100 101 102 103}"
RETRY="${RETRY:-1}"
GAME_TIMEOUT_S="${GAME_TIMEOUT_S:-7200}"
ALPHAS="${ALPHAS:-0 0.02 0.037 0.07 0.15 0.30}"
OUTROOT="match_results/leaf_coin_alpha"

# BASE：与 `_us_vs_external.sh:68` 逐字相同（历史批次可比性的锚点）
BASE="$(sed -n 's/^BASE="\(.*\)"$/\1/p' code/test_match/_us_vs_external.sh)"
[ -n "$BASE" ] || { echo "[lca] !! 读不到 _us_vs_external.sh 的 BASE" >&2; exit 2; }
case "$BASE" in
    *AZAI_CAND_MODE*) echo "[lca] !! BASE 里已含 AZAI_CAND_MODE ⇒ 停" >&2; exit 2 ;;
esac
case "$BASE" in
    *AZAI_FLAT_ROLLOUT_MODE=random*) ;;
    *) echo "[lca] !! BASE 里没有 AZAI_FLAT_ROLLOUT_MODE=random ⇒ 停" >&2; exit 2 ;;
esac

mkdir -p "$OUTROOT"
echo "=============================================================="
echo "[lca] α 扫描：unified(+leaf_coin_alpha) vs pin+random"
echo "[lca] alphas=[$ALPHAS]  seeds=[$SEEDS]  jobs=$JOBS  retry=$RETRY"
echo "[lca] BASE 与 _us_vs_external.sh:68 逐字相同（${#BASE} 字符）"
echo "[lca] $(date '+%F %T')"
echo "=============================================================="

for ALPHA in $ALPHAS; do
    TAG="lca${ALPHA}"
    OUTDIR="$OUTROOT/a${ALPHA}"
    mkdir -p "$OUTDIR"
    # A 臂：unified + 叶子金币项（α 覆盖 BASE 里没有的键；AZAI_CAND_VERBOSE=1 只为行为日志）
    FLAT_ENV="$BASE;AZAI_CAND_MODE=unified;AZAI_CAND_VERBOSE=1;AZAI_LEAF_COIN_ALPHA=$ALPHA"
    MCTS_ENV="$BASE;AZAI_CAND_MODE=pin;AZAI_CAND_VERBOSE=1"
    # 每个 α 用唯一 stderr 后缀，避免串行批次之间互相覆盖（并方便分析端按 α 归档）
    export MVS_STDERR_SUFFIX=1
    export AB_SEARCH_SUFFIX_TAG="_$TAG"
    echo ""
    echo "[lca] ── α=$ALPHA → $OUTDIR  seeds=[$SEEDS] jobs=$JOBS ──"
    echo "[lca] A env: ... AZAI_CAND_MODE=unified AZAI_LEAF_COIN_ALPHA=$ALPHA"
    echo "[lca] B env: ... AZAI_CAND_MODE=pin"
    "$P" -u code/test_match/ab_search_modes.py \
        --seeds $SEEDS --jobs "$JOBS" --paired-sides \
        --out-dir "$OUTDIR" \
        --flat-env "$FLAT_ENV" --mcts-env "$MCTS_ENV" \
        --resume --retry-invalid "$RETRY" --game-timeout-s "$GAME_TIMEOUT_S" \
        2>&1 | tee "$OUTDIR/_run.log"
    rc=${PIPESTATUS[0]}
    echo "[lca] α=$ALPHA 结束 rc=$rc  $(date '+%F %T')"
done

echo ""
echo "=============================================================="
echo "[lca] 全部 α 跑完 $(date '+%F %T')"
echo "[lca] 分析："
echo "  $P -u code/test_match/analyze_leaf_coin_alpha.py \\"
echo "      --alphas $(echo $ALPHAS | tr ' ' ',') --seeds $(echo $SEEDS | tr ' ' ',') \\"
echo "      --json $OUTROOT/summary.json"
echo "=============================================================="
