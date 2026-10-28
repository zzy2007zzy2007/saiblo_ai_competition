#!/usr/bin/env bash
# 冒烟：**注入在"我们 vs 亚军本体"的真实对局里到底有没有生效**（2026-10-28）
#
# 目的（这是"静默失效"的唯一可信判据，报告 docs/inject_vs_runnerup_20261028.md §1.3/§3.1）：
#   1. `[oracle] 注入已启用` 必须出现，且 `exe=` 是 **D:/ 形态**（Git-bash 的 /d/... 会让
#      `Path.is_absolute()` 判 False ⇒ 注入整局静默不启用，历史踩过一次）；
#   2. `asks > 0`（真的问到亚军了），且**没有** `注入已停用 / 永久停用 / 查询失败`；
#   3. 两个座位（我们执 P0 的 `7`、我们执 P1 的 `7r`）都要过 —— P0/P1 的协议读序不同。
#
# 只打 30 回合（`--rounds=30`）⇒ 几分钟出结果。**这一局不计入 A/B 读数**（不是完整对局）。
# 为什么是 30 而不是更少：bridge 每 **25 个决策**才打一次 `注入累计`，而收尾报告在对局被
# bridge 硬杀时不保证写出来 ⇒ 少于 26 回合就只能看到 `注入已启用`、看不到 asks/timeout/dead。
#
# 用法：
#   bash code/run_logged.sh inject_vs_ru_smoke bash code/test_match/_inject_vs_runnerup_smoke.sh
set -uo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO" || exit 1
export PYTHONIOENCODING=utf-8

P="${PYTHON:-D:/anaconda3/envs/pytorch-gpu/python.exe}"
RU_REL="其他版本ai/saiblo-30th-AI/Game1/antgame_ai_cpp/cpp_lure_v4/build/ai_cpp_lure_v4.exe"
RU="$(pwd -W | sed 's#/$##')/$RU_REL"
OUT="match_results/inject_vs_ru/smoke"
mkdir -p "$OUT"

BASE="AZAI_CKPT=checkpoints/posnet_A_k5_m32.pt;AZAI_ITERS=256;AZAI_DEPTH=4;AZAI_SAMPLE_MULT=15;AZAI_MODE=pos-only;AZAI_POSPIN=mc_quiet;AZAI_MC_EVERY=1;AZAI_MC_ROOTONLY=1;AZAI_MC_CELLS=4;AZAI_SKIP1=1;AZAI_TCLASS=0.5;AZAI_TPOS=1.0;AZAI_TEMP=1e-6;AZAI_POSPRIOR=off;AZAI_SEARCH=flat;AZAI_MC_HORIZON=16;AZAI_MC_SCORE=net;AZAI_MC_NET_SAMPLES=4;AZAI_FLAT_BATCH=32;AZAI_FLAT_HORIZON=16;AZAI_FLAT_SHARE_PREFIX=1;AZAI_K=48;AZAI_FLAT_ROLLOUTS=64;AZAI_FLAT_ROLLOUT_MODE=random;AZAI_FLAT_RANDOM_SIDE=opponent;AZAI_FLAT_PROB_MODE=decay;AZAI_FLAT_DECAY_C=1.0;AZAI_FLAT_DECAY_K=3.0;AZAI_FLAT_ROUNDS=49152;AZAI_FLAT_SEED=53"
ENV_A="$BASE;AZAI_ORACLE_INJECT=$RU;AZAI_ORACLE_TIMEOUT=15"

case "$RU" in
    D:/*) ;;
    *) echo "[smoke] !! RU 不是 D:/ 形态: $RU" >&2; exit 2 ;;
esac

for v in $(env | sed -n 's/^\(AZAI_[A-Za-z_0-9]*\)=.*/\1/p'); do unset "$v"; done
for kv in $(echo "$ENV_A" | tr ';' '\n'); do [ -n "$kv" ] && export "$kv"; done
echo "[smoke] exe=$RU"
echo "[smoke] AZAI_ORACLE_INJECT=$AZAI_ORACLE_INJECT timeout=$AZAI_ORACLE_TIMEOUT"
echo "[smoke] $(date '+%Y-%m-%d %H:%M:%S')  开始（30 回合 × 2 座位）"

"$P" -u code/test_match/match_via_sdk.py --rounds=30 --trace-ops \
    --ai0=code/test_match/az_bridge_ai.py --ai1="$RU" 7 7r 2>&1 | tee "$OUT/smoke.log"
rc=${PIPESTATUS[0]}

cp -f match_results/ai0_az_bridge_ai_seed7.stderr.log "$OUT/" 2>/dev/null
cp -f match_results/ai1_az_bridge_ai_seed7.stderr.log "$OUT/" 2>/dev/null

echo ""
echo "================ 断言 ================"
fail=0
for seat in 0 1; do
    f="$OUT/ai${seat}_az_bridge_ai_seed7.stderr.log"
    if [ ! -f "$f" ]; then echo "!! 缺 stderr: $f"; fail=1; continue; fi
    en=$(grep -c '注入已启用' "$f")
    # ⚠️ 路径判定：bridge 打的是 `str(Path(...))` ⇒ Windows 上可能是 `D:\...` 反斜杠形态。
    #    真正的要求是"Windows 绝对路径"（`Path.is_absolute()` 为真），不是斜杠方向
    #    —— 反斜杠形态同样能 Popen 成功（历史踩的坑是 Git-bash 的 `/d/...`）。
    path=$(grep -o 'exe=[^ ]*' "$f" | head -1 | cut -d= -f2-)
    norm=$(echo "$path" | tr '\\' '/')
    # ⚠️ 前置空格：统计行里还有 `hit_rate_asks=84.0%`，不锚定会误抓到它
    asks=$(grep -o ' asks=[0-9]*' "$f" | tail -1 | tr -d ' ' | cut -d= -f2)
    dead=$(grep -o 'dead=[0-9]*' "$f" | tail -1 | cut -d= -f2)
    tmo=$(grep -o 'timeout=[0-9]*' "$f" | tail -1 | cut -d= -f2)
    bad=$(grep -c '注入已停用\|永久停用\|查询失败\|OracleDesync\|OracleDead\|OracleTimeout' "$f")
    steps=$(grep -c '\[oracle\] inject round=' "$f")
    echo "[座位 P$seat] 注入已启用=$en  exe=$path  asks=$asks  timeout=$tmo  dead=$dead " \
         "失败标记=$bad  每决策日志=$steps 行"
    [ "$en" -ge 1 ] || { echo "  !! P$seat 没有『注入已启用』⇒ 注入整局没生效"; fail=1; }
    case "$norm" in
        D:/*) ;;
        *) echo "  !! P$seat 的 exe 不是 Windows 绝对路径（D:/ 或 D:\\）⇒ 静默失效"; fail=1 ;;
    esac
    [ "$steps" -gt 0 ] || { echo "  !! P$seat 一条每决策注入日志都没有 ⇒ 注入没真的跑"; fail=1; }
    [ "${asks:-0}" -gt 0 ] || { echo "  !! P$seat asks=0"; fail=1; }
    [ "$bad" -eq 0 ] || { echo "  !! P$seat 有注入失败标记 $bad 行"; fail=1; }
    echo "  注入统计: $(grep '注入统计\|注入累计' "$f" | tail -1)"
done
echo ""
if [ "$fail" -eq 0 ]; then echo "[smoke] ✅ 全部断言通过（注入在两个座位都真的生效）"; else echo "[smoke] ❌ 有断言失败"; fi
echo "[smoke] $(date '+%Y-%m-%d %H:%M:%S') rc=$rc"
exit $fail
