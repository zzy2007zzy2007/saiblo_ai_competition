#!/usr/bin/env bash
# 冒烟（2026-10-28）：`AZAI_UPG_BUNDLES=4` 到底有没有把升级候选塞进 flat rollout。
#
# 只做三件事（跑 40 回合就够，不占预算）：
#   ① 不崩：整局能跑完（`--rounds=40` ⇒ verdict=hp_tiebreak 是预期的）；
#   ② 通道真的进了：AI stderr 里有 `[upg] round=… 加入 k 条升级候选 targets=[…]`；
#   ③ 评估器真的在选：有 `[upg] sel … hit=1` 行，且没有 `illegal` / `[upg] ⚠️`。
# ⚠️ 用 seed 3（避开 A/B/C 三臂的 seeds 7..14）⇒ 不会污染/覆盖那批 stderr 归档。
#
# 用法：bash code/test_match/_upg_bundles_smoke.sh [N] [SEED] [ROUNDS]
set -uo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO" || exit 1
export PYTHONIOENCODING=utf-8

P="${PYTHON:-D:/anaconda3/envs/pytorch-gpu/python.exe}"
UPG_N="${1:-4}"
SEED="${2:-3}"
ROUNDS="${3:-40}"
OUT="match_results/upgrade_bundles"
mkdir -p "$OUT"

# BASE 从驱动脚本里**读**出来（不复制）：保证"冒烟的配置 = 正式批的配置"
BASE="$(sed -n 's/^BASE="\(.*\)"$/\1/p' code/test_match/_inject_vs_runnerup.sh)"
[ -n "$BASE" ] || { echo "[smoke] !! 读不到 BASE"; exit 2; }
ENV0="$BASE;AZAI_UPG_BUNDLES=$UPG_N"     # ai0 = 开了开关
ENV1="$BASE"                              # ai1 = 对照（开关关）

echo "=============================================================="
echo "[smoke] AZAI_UPG_BUNDLES=$UPG_N  seed=$SEED rounds=$ROUNDS  $(date '+%Y-%m-%d %H:%M:%S')"
echo "[smoke] ai0 env = $ENV0"
echo "=============================================================="

# 顺手把同 seed 的旧 stderr 挪走（只挪不删）⇒ 下面 grep 的必然是本轮
mkdir -p "$OUT/_stale"
for f in match_results/ai?_az_bridge_ai_seed${SEED}.stderr.log; do
    [ -f "$f" ] || continue
    mv -f "$f" "$OUT/_stale/" 2>/dev/null && echo "[smoke] 旧 stderr 挪走: $(basename "$f")"
done

LOG="$OUT/smoke_seed${SEED}_upg${UPG_N}.log"
"$P" -u code/test_match/match_via_sdk.py --rounds="$ROUNDS" --trace-ops \
    --ai0=code/test_match/az_bridge_ai.py --ai1=code/test_match/az_bridge_ai.py \
    --ai0-env="$ENV0" --ai1-env="$ENV1" "$SEED" > "$LOG" 2>&1
RC=$?
echo "[smoke] match_via_sdk rc=$RC  ($(date '+%H:%M:%S'))"
grep -h "RESULT seed" "$LOG" | sed 's/^/[smoke] /'

echo ""
echo "-------- [smoke] 我方（ai0，开关开）stderr 里的 [upg] 行 --------"
F0="match_results/ai0_az_bridge_ai_seed${SEED}.stderr.log"
if [ -f "$F0" ]; then
    echo "[smoke] 启用行：";      grep -h '\[upg\] 升级候选已启用' "$F0" | head -2 | sed 's/^/  /'
    echo "[smoke] 加入行（前 6 条）："; grep -h '加入 .* 条升级候选' "$F0" | head -6 | sed 's/^/  /'
    echo "[smoke] 选中行（前 6 条）："; grep -h '\[upg\] sel ' "$F0" | head -6 | sed 's/^/  /'
    echo "[smoke] 统计行（最后 1 条）："; grep -h '\[upg\] 统计' "$F0" | tail -1 | sed 's/^/  /'
    echo ""
    printf '[smoke] 计数: 决策=%s  有候选=%s  塞入条数合计=%s  选中=%s  异常=%s\n' \
        "$(grep -c '加入 .* 条升级候选' "$F0")" \
        "$(grep -c '加入 [1-9]' "$F0")" \
        "$(grep -o '加入 [0-9]* 条' "$F0" | awk '{s+=$2} END{print s+0}')" \
        "$(grep -c '\[upg\] sel .*hit=1' "$F0")" \
        "$(grep -c '\[upg\] ⚠️' "$F0")"
else
    echo "[smoke] !! 找不到 $F0"
fi
echo "-------- [smoke] 对照（ai1，开关关）应当**一条 [upg] 都没有** --------"
F1="match_results/ai1_az_bridge_ai_seed${SEED}.stderr.log"
if [ -f "$F1" ]; then
    printf '[smoke] ai1 的 [upg] 行数 = %s（应为 0）\n' "$(grep -c '\[upg\]' "$F1")"
    printf '[smoke] ai0/ai1 illegal 行数 = %s / %s\n' \
        "$(grep -c 'ILLEGAL' "$F0" 2>/dev/null || echo 0)" "$(grep -c 'ILLEGAL' "$F1" 2>/dev/null || echo 0)"
fi
echo "[smoke] 完 $(date '+%H:%M:%S')"
