#!/usr/bin/env bash
# 定向验收：**手工杀掉 oracle 子进程**，看桥接能不能"重建 + 回放 + 重试本回合"（2026-10-29）
#
# 为什么需要这个：冠军什么时候"自然崩"取决于它影子账本里的基地血什么时候归零，
# 与 seed/局面有关（实测有的局到 340 回合还没崩）⇒ 光靠自然崩**没法在验收窗口内**保证触发到重建路径。
# 这个脚本按 **PID 精确**杀掉**本实验自己起的** oracle（magica_v3_O0.exe）子进程，
# 复现"读包时进程已退出（OracleDead）"这一条，然后看：
#   ① `[oracle] ⚠️ 第 N 回合查询失败`（dead+1）
#   ② `[oracle] 重建会话 #1 at round=N（回放 N 回合，耗时 X s）`
#   ③ `[oracle] ↻ 第 N 回合：会话已重建，重试本回合`
#   ④ 本局 `asks` 覆盖度是否仍然 ≈100%（= 只丢了一个回合的候选）
#
# ⚠️ 只杀**本脚本自己起的**进程：先读桥接 stderr 里的 `init: ... pid=<bridge>`，
#    再用 WMI 找 `ParentProcessId=<bridge>` 的 magica 进程。**绝不按镜像名/通配符批量杀。**
#
# 用法：
#   SEED=10 KILL_AT_ROUND=40 ROUNDS=90 bash code/run_logged.sh oracle_kill_test \
#       bash code/test_match/_oracle_kill_rebuild_test.sh
set -uo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO" || exit 1
export PYTHONIOENCODING=utf-8
SEED="${SEED:-10}"
KILL_AT_ROUND="${KILL_AT_ROUND:-40}"
ROUNDS="${ROUNDS:-90}"
export TAG="${TAG:-ORACLE_KILL_TEST}"
export EXP=E2 OPP=champion SEEDS="$SEED" JOBS=1 RETRY=0 ROUNDS="$ROUNDS"
export LADDER_TIMEOUT="${LADDER_TIMEOUT:-3600}" MVS_TIMEOUT_SECONDS="${MVS_TIMEOUT_SECONDS:-300}"
export AZAI_ORACLE_REBUILD_MAX="${AZAI_ORACLE_REBUILD_MAX:-6}"
export AZAI_ORACLE_RESTART_EVERY="${AZAI_ORACLE_RESTART_EVERY:-0}"     # 本测要的是"崩了才重建"
LOG="match_results/ladder_logs/$TAG/$SEED.log"
BRIDGE_LOG="match_results/ai0_az_bridge_ai_seed${SEED}_${TAG}.stderr.log"

echo "[kill-test] TAG=$TAG seed=$SEED rounds=$ROUNDS 计划在第 $KILL_AT_ROUND 回合**精确杀** oracle 子进程"

# 后台起实验
( bash code/test_match/_us_vs_external.sh ) > "match_results/ladder_logs/_kill_${TAG}.out" 2>&1 &
LADDER_SH=$!
echo "[kill-test] 实验 shell pid=$LADDER_SH"

# 等桥接 stderr 出现 init 行（拿 bridge pid）
BRIDGE_PID=""
for _ in $(seq 1 60); do
    if [ -f "$BRIDGE_LOG" ]; then
        BRIDGE_PID="$(sed -n 's/.*\[azai\] init: player=[01] seed=[0-9]* pid=\([0-9]*\).*/\1/p' "$BRIDGE_LOG" | head -1)"
        [ -n "$BRIDGE_PID" ] && break
    fi
    sleep 2
done
if [ -z "$BRIDGE_PID" ]; then
    echo "[kill-test] !! 60s 内没拿到 bridge pid（$BRIDGE_LOG）⇒ 放弃" >&2
    kill "$LADDER_SH" 2>/dev/null
    exit 3
fi
echo "[kill-test] bridge pid=$BRIDGE_PID"

# 等局面推进到 KILL_AT_ROUND
for _ in $(seq 1 120); do
    R="$(grep -o '\[round [0-9]*\] towers' "$LOG" 2>/dev/null | tail -1 | grep -o '[0-9]*' | tail -1)"
    [ -n "${R:-}" ] && [ "$R" -ge "$KILL_AT_ROUND" ] && break
    sleep 2
done
echo "[kill-test] 当前回合 R=${R:-?} ⇒ 开始找 oracle 子进程（ParentProcessId=$BRIDGE_PID 的 magica）"

# 精确找：父进程 = bridge pid 的 magica_v3_O0.exe
ORACLE_PID="$(powershell -NoProfile -Command "(Get-CimInstance Win32_Process -Filter \"ParentProcessId=$BRIDGE_PID\" | Where-Object { \$_.Name -eq 'magica_v3_O0.exe' } | Select-Object -First 1).ProcessId" 2>/dev/null | tr -d '\r\n ')"
if [ -z "${ORACLE_PID:-}" ]; then
    echo "[kill-test] !! 没找到 ParentProcessId=$BRIDGE_PID 的 magica 子进程 ⇒ 列出候选只看不杀：" >&2
    powershell -NoProfile -Command "Get-CimInstance Win32_Process -Filter \"ParentProcessId=$BRIDGE_PID\" | Select-Object ProcessId,Name | Format-Table -AutoSize" >&2
    kill "$LADDER_SH" 2>/dev/null
    exit 4
fi
echo "[kill-test] ★ 精确杀 oracle pid=$ORACLE_PID（它就是本实验 bridge 的子进程）"
powershell -NoProfile -Command "Stop-Process -Id $ORACLE_PID -Force" 2>&1 | head -3
date '+%H:%M:%S'

# 等实验结束（最多 12 分钟）
for _ in $(seq 1 72); do
    kill -0 "$LADDER_SH" 2>/dev/null || break
    sleep 10
done
wait "$LADDER_SH" 2>/dev/null
echo "[kill-test] 实验结束 $(date '+%H:%M:%S')"
echo ""
echo "-------- 桥接日志里的 oracle 事件（验收就看这几行）--------"
grep -E '查询失败|重建会话|重建次数|重试本回合|主动重启' "$BRIDGE_LOG" 2>/dev/null | tail -10
echo ""
echo "-------- 该局的注入统计 --------"
grep '注入统计:' "$BRIDGE_LOG" 2>/dev/null | tail -1 | cut -c1-420
echo ""
echo "-------- RESULT --------"
grep -h 'RESULT seed' "$LOG" 2>/dev/null | tail -1
