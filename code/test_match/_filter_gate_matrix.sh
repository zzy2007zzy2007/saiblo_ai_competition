#!/usr/bin/env bash
# P1「过滤后整块合法性闸门」的组合矩阵（2026-10-28；报告 docs/oracle_inject_ab_20261028.md §3）
#
# 目的：`docs/illegal_op_strict_mode_20261028.md` §5.3 那个真非法操作来自**事后删 op**，
# 所以在**注入开启**的前提下，把每个"事后过滤"开关各跑一局，验证：
#   ① `illegal_total=0`（AI 侧 `illegal-summary`）且日志里没有 `ILLEGAL` 行；
#   ② 判词 `verdict=engine`（不是 `illegal_strict`）；
#   ③ 该触发闸门的组合（NO_DOWNGRADE / BAN_OPS=13）能看到 `🛡️ [gate] ... 整块丢弃` 的证据。
#
# 4 局**并行**（各占一套 seed/座位，stderr 名唯一 ⇒ 不互相覆盖）：
#   no_downgrade   seed 8103     我方 p0  ← §5.3 的 r61 复现局
#   ban_ops13      seed 8103r    我方 p1  ← §5.3 的 r244 复现局
#   light_min      seed 8201     我方 p0
#   build_min      seed 8202     我方 p0
#
# 配置 = `_illegal_smoke.sh` 的 HIGHRISK（便宜搜索 + 高风险改写路径全开；
# **op 的合法性与搜索强度无关**）+ 各组合的一个开关。
set -uo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO" || exit 1
export PYTHONIOENCODING=utf-8
export MVS_STDERR_SUFFIX=1        # 每局 stderr 带 DSH_PROGRESS_SUFFIX 后缀（防覆盖）
export MVS_TIMEOUT_SECONDS=900    # 只是"卡死检测"门限，不影响判词

P="${PYTHON:-D:/anaconda3/envs/pytorch-gpu/python.exe}"
BR=code/test_match/az_bridge_ai.py
RU="其他版本ai/saiblo-30th-AI/Game1/antgame_ai_cpp/cpp_lure_v4/build/ai_cpp_lure_v4.exe"
OUT=match_results/filter_gate_matrix
mkdir -p "$OUT"

# HIGHRISK 底（逐字照抄 _illegal_smoke.sh，只少了它自带的 AZAI_NO_DOWNGRADE=1 —— 这里按组合给）
HR="AZAI_CKPT=checkpoints/posnet_A_k5_m32.pt;AZAI_ITERS=32;AZAI_K=8;AZAI_SAMPLE_MULT=1;AZAI_MODE=pos-only;AZAI_POSPIN=argmax;AZAI_SKIP1=1;AZAI_TEMP=1e-6;AZAI_SEARCH=flat;AZAI_FLAT_ROLLOUTS=4;AZAI_FLAT_HORIZON=8;AZAI_FLAT_BATCH=8;AZAI_FLAT_ROUNDS=39000;AZAI_MC_UPG=1;AZAI_ORACLE_INJECT=${RU};AZAI_ORACLE_TIMEOUT=2.5"

run_one() {   # run_one <tag> <seed token> <组合 env>
    local tag="$1" seed="$2" extra="$3"
    echo "[matrix] 开局 tag=$tag seed=$seed 组合=$extra -> $OUT/$tag.log"
    DSH_PROGRESS_TAG=filter_gate_matrix DSH_PROGRESS_SUFFIX="_${tag}" \
        "$P" -u code/test_match/match_via_sdk.py \
        "--ai0=$BR" "--ai1=$RU" "--ai0-env=$HR;$extra" "$seed" \
        > "$OUT/$tag.log" 2>&1
    echo "[matrix] 结束 tag=$tag rc=$?"
}

echo "[matrix] 4 局并行开始 $(date '+%Y-%m-%d %H:%M:%S')"
run_one no_downgrade 8103  "AZAI_NO_DOWNGRADE=1"    &
run_one ban_ops13    8103r "AZAI_BAN_OPS=13"        &
run_one light_min    8201  "AZAI_LIGHT_MIN_COINS=180" &
run_one build_min    8202  "AZAI_BUILD_MIN_COINS=90"  &
wait
echo "[matrix] 4 局并行结束 $(date '+%Y-%m-%d %H:%M:%S')"

echo ""
echo "================ RESULT 行 ================"
grep -h "RESULT seed" "$OUT"/*.log 2>/dev/null | sort

echo ""
echo "================ 每局读数（非法 / 闸门 / 注入）================"
for tag in no_downgrade ban_ops13 light_min build_min; do
    echo "── $tag ──────────────────────────────────────────────"
    # AI 侧：非法汇总 + 过滤闸门 + 注入统计（stderr 名字里带了 _<tag> 后缀）
    for f in match_results/ai?_az_bridge_ai_seed*_${tag}.stderr.log; do
        [ -f "$f" ] || continue
        echo "  [$(basename "$f")]"
        grep -h "illegal-summary\|过滤闸门累计\|🛡️ \[gate\] 过滤后整块非法\|注入统计" "$f" \
            | tail -4 | sed 's/^/    /'
    done
    # 裁判侧：一行 ILLEGAL 都不许有
    local_n=$(grep -c "ILLEGAL " "$OUT/$tag.log" 2>/dev/null || true)
    echo "    裁判侧 ILLEGAL 行数 = ${local_n:-0}"
done

echo ""
echo "================ 判据汇总 ================"
# ⚠️ `illegal-summary` / `过滤闸门累计` 都是**累计口径**、对局中每 25 决策打一条 ⇒
#    每个文件只取**最后一条**，再跨文件相加（第一版按行相加 ⇒ 把累计值重复计数，读数虚高）。
fail=0
for tag in no_downgrade ban_ops13 light_min build_min; do
    ill=0; gchk=0; gdrop=0
    for f in match_results/ai?_az_bridge_ai_seed*_${tag}.stderr.log; do
        [ -f "$f" ] || continue
        v=$(grep -ho "illegal_total=[0-9]*" "$f" | tail -1 | sed 's/.*=//')
        ill=$((ill + ${v:-0}))
        g=$(grep -ho "过滤闸门累计: checks=[0-9]* dropped=[0-9]*" "$f" | tail -1)
        if [ -n "$g" ]; then
            gchk=$((gchk + $(echo "$g" | sed 's/.*checks=\([0-9]*\).*/\1/')))
            gdrop=$((gdrop + $(echo "$g" | sed 's/.*dropped=\([0-9]*\).*/\1/')))
        fi
    done
    bridge_ill=$(grep -ho "illegal=[0-9]*" "$OUT/$tag.log" 2>/dev/null | sed 's/.*=//' | awk '{s+=$1} END{print s+0}')
    bridge_ill_lines=$(grep -c "ILLEGAL " "$OUT/$tag.log" 2>/dev/null || true)
    verdict=$(grep -ho "verdict=[a-z_]*" "$OUT/$tag.log" 2>/dev/null | head -1 | sed 's/.*=//')
    printf "  %-14s verdict=%-14s 裁判侧illegal=%s(行数%s) AI侧illegal_total=%s 闸门=checks %s / dropped %s\n" \
           "$tag" "${verdict:-?}" "${bridge_ill:-0}" "${bridge_ill_lines:-0}" "$ill" "$gchk" "$gdrop"
    [ "${bridge_ill:-0}" = "0" ] || fail=1
    [ "${bridge_ill_lines:-0}" = "0" ] || fail=1
    [ "${ill:-0}" = "0" ] || fail=1
    [ "${verdict:-x}" = "engine" ] || fail=1
done
echo ""
if [ "$fail" = "0" ]; then
    echo "[matrix] ✅ 全部组合 0 非法操作"
else
    echo "[matrix] ❌ 有组合产生非法操作（见上）"
fi
exit $fail
