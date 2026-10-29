#!/usr/bin/env bash
# 验证 4：`--watchdog-stall-min 1` 生效（~1 分钟内**只**精确杀掉卡住的那一局）
#
# 卡住的构造 = `_tmp_hangai.py`（假 AI，`sleep 600`）+ 空操作假 AI：
#   · 该局卡在 round 0，progress/日志 mtime 都不再前进 ⇒ 看门狗 60s 阈值命中；
#   · 同时并发跑一个**对照批次**（真对局，没有任何新开关）⇒ 证明别人的局不受影响；
#   · 跑前/跑后各查一次**别人那批真实对局**（`US_VS_CHAMPION_BASE` ladder PID 58044）
#     ⇒ 一个都没少（只读进程表，绝不 kill）。
source "$(dirname "${BASH_SOURCE[0]}")/_tmp_lr_env.sh"

HANG="$ROOT_W/_tmp_hangai.py"
[ -f "$HANG" ] || { echo "[v4] !! 假 AI 不存在: $HANG" >&2; exit 2; }
export LR_HANG_PIDFILE="$ROOT_W/_tmp_lr_v4_hangpids.txt"
rm -f "$LR_HANG_PIDFILE"

TAG_WD=LR_V4_WD
TAG_CTL=LR_V4_CTL
rm -rf "match_results/ladder_logs/$TAG_WD" "match_results/ladder_logs/$TAG_CTL"

echo "===== 跑前进程普查（别人那批真实对局）====="
"$PY" _tmp_lr_pids.py 58044 --label "US_VS_CHAMPION_BASE ladder（跑前）" | tee _tmp_lr_v4_census_before.txt

echo ""
echo "===== 并发：对照批次（真对局）先起，随后起看门狗批次（假 AI 挂住）====="
bash code/run_logged.sh ladder_resilience_v4_ctl \
    "$PY" -u _tmp_ladder.py --tag=$TAG_CTL --jobs=1 \
    --ai0="$RV4" --ai1="$NOOP" 996 > _tmp_lr_v4_ctl.out 2>&1 &
CTL=$!
sleep 3

bash code/run_logged.sh ladder_resilience_v4_watchdog \
    "$PY" -u _tmp_ladder.py --tag=$TAG_WD --jobs=1 \
    --watchdog-stall-min 1 --retry-invalid 1 \
    --ai0="$HANG" --ai1="$NOOP" 995
RC=$?
wait $CTL
CTL_RC=$?

echo ""
echo "===== 跑后进程普查（别人那批真实对局 + 我的假 AI 有没有漏网）====="
"$PY" _tmp_lr_pids.py 58044 --label "US_VS_CHAMPION_BASE ladder（跑后）" | tee _tmp_lr_v4_census_after.txt
if [ -s "$LR_HANG_PIDFILE" ]; then
    PIDS="$(tr '\n' ' ' < "$LR_HANG_PIDFILE")"
    echo "--- 假 AI 自报的 PID: $PIDS（若还活着就按**这些确切 PID** 补刀；绝不按镜像名/通配符）"
    "$PY" _tmp_lr_pids.py $PIDS --label "假 AI（跑后）" || true
    for p in $PIDS; do taskkill //PID "$p" //T //F 2>/dev/null || true; done
fi

echo ""
echo "===== 对照批次（别的局不受影响）====="
echo "--- 对照批次 rc=$CTL_RC，它的日志尾:"
tail -5 _tmp_lr_v4_ctl.out | sed 's/^/  /'
grep -h "RESULT seed=" "match_results/ladder_logs/$TAG_CTL"/*.log 2>/dev/null | sed 's/^/  /'
echo "--- 看门狗批次（应 0 局 engine；两局日志都没有 RESULT 行）:"
grep -h "RESULT seed=" "match_results/ladder_logs/$TAG_WD"/*.log 2>/dev/null | sed 's/^/  /'
echo "--- 看门狗批次日志大小（空壳 ⇒ 被精确杀掉）:"
ls -l "match_results/ladder_logs/$TAG_WD"/*.log | sed 's/^/  /'
echo ""
echo "[v4] 结束 $(date '+%Y-%m-%d %H:%M:%S')  rc=$RC"
exit $RC
