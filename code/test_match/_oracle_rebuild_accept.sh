#!/usr/bin/env bash
# 任务 A 验收：冠军 oracle 的「会话自动重建 + 回放追赶」到底救回多少覆盖度（2026-10-29）
#
# 两个臂（其余环境与 E2 **逐字相同**：BASE、对手=冠军、注入=冠军、seeds 7..14×2 座位）：
#   ARM=rebuild  ：只开**崩溃后重建**（`AZAI_ORACLE_REBUILD_MAX=6`，不主动重启）
#                  ⇒ 预期：第 ~239 回合它 abort → 重建（回放 ~239 回合、~20s）→ 同一回合又崩
#                  → 6 次用完 ⇒ 后半局无注入。**这一臂是用来量"回放治不了"的**。
#   ARM=restart  ：开**主动重启**（`AZAI_ORACLE_RESTART_EVERY=150`，B 会话不重启）
#                  ⇒ 预期：每 150 回合换一张白纸，会话**永不崩** ⇒ 覆盖度 ~100%。
#                  ⚠️ 代价（报告里必须写）：新会话的影子局面从零攒 ⇒ 它几乎一律 HOLD。
#   ARM=both     ：两个开关都开（生产配置 E4 用的就是它）
#
# 用法：
#   ARM=restart SEEDS="7 8 9" JOBS=3 bash code/run_logged.sh oracle_fix_restart \
#       bash code/test_match/_oracle_rebuild_accept.sh
#   ARM=rebuild SEEDS="7" JOBS=1 bash code/run_logged.sh oracle_fix_rebuild \
#       bash code/test_match/_oracle_rebuild_accept.sh
#
# 读数（跑完自动打）：
#   D:/anaconda3/envs/pytorch-gpu/python.exe -u code/test_match/analyze_xoracle.py \
#       --tag <TAG> --stderr match_results/us_vs_external/<TAG>/stderr
set -uo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO" || exit 1
export EXP=E2                       # 注入冠军动作 vs 冠军（唯一变量就是重建/重启那两个开关）
export OPP=champion
ARM="${ARM:-restart}"
case "$ARM" in
    rebuild)
        export TAG="${TAG:-ORACLE_FIX_REBUILD}"
        export AZAI_ORACLE_REBUILD_MAX="${AZAI_ORACLE_REBUILD_MAX:-6}"
        export AZAI_ORACLE_RESTART_EVERY=0 ;;
    restart)
        export TAG="${TAG:-ORACLE_FIX_RESTART}"
        export AZAI_ORACLE_REBUILD_MAX="${AZAI_ORACLE_REBUILD_MAX:-6}"
        export AZAI_ORACLE_RESTART_EVERY="${AZAI_ORACLE_RESTART_EVERY:-150}"
        export AZAI_ORACLE_RESTART_EVERY_B=0 ;;   # B 会话（E4 才用）不重启
    both)
        export TAG="${TAG:-ORACLE_FIX_BOTH}"
        export AZAI_ORACLE_REBUILD_MAX="${AZAI_ORACLE_REBUILD_MAX:-6}"
        export AZAI_ORACLE_RESTART_EVERY="${AZAI_ORACLE_RESTART_EVERY:-150}" ;;
    *) echo "[accept] !! ARM=$ARM 不认识（rebuild|restart|both）" >&2; exit 2 ;;
esac
export SEEDS="${SEEDS:-7 8 9}"
export JOBS="${JOBS:-3}"
echo "[accept] ARM=$ARM TAG=$TAG SEEDS=$SEEDS JOBS=$JOBS"
echo "[accept] AZAI_ORACLE_REBUILD_MAX=${AZAI_ORACLE_REBUILD_MAX:-<默认6>} " \
     "AZAI_ORACLE_RESTART_EVERY=${AZAI_ORACLE_RESTART_EVERY:-0}"
exec bash code/test_match/_us_vs_external.sh
