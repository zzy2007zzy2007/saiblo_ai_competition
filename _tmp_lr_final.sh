#!/usr/bin/env bash
# 终版复跑：最终代码（含"重跑前再判一次"的防御）把四个场景整套跑一遍。
#   V1 只重跑"新版"那一侧（HEAD 版 `LR_V1_OLD` 的日志是确定性的、已在仓库里，不重复跑）。
source "$(dirname "${BASH_SOURCE[0]}")/_tmp_lr_env.sh"

echo ""
echo "===== V1-new（终版，默认参数）：对比已有的 HEAD 版日志 LR_V1_OLD ====="
rm -rf match_results/ladder_logs/LR_V1_NEW
bash code/run_logged.sh ladder_resilience_v1_new \
    "$PY" -u _tmp_ladder.py --tag=LR_V1_NEW --jobs=1 \
    --ai0="$RV4" --ai1="$NOOP" 920 920r
"$PY" _tmp_lr_cmp.py LR_V1_OLD LR_V1_NEW 920 920r
V1RC=$?

echo ""
echo "===== V2（终版）====="
bash _tmp_lr_v2.sh
V2RC=$?

echo ""
echo "===== V3（终版）====="
bash _tmp_lr_v3.sh
V3RC=$?

echo ""
echo "===== V4（终版）====="
bash _tmp_lr_v4.sh
V4RC=$?

echo ""
echo "[final] 结束 $(date '+%Y-%m-%d %H:%M:%S')  v1=$V1RC v2=$V2RC v3=$V3RC v4=$V4RC"
