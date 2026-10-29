#!/usr/bin/env bash
# E1 续跑：我们（**不注入**）vs 冠军 `magica_v3_O0.exe`，**只补没跑完的局**（2026-10-29）
#
# 背景（上一批被中断，见 docs/us_vs_champion_20261029.md §2）：
#   2026-10-28 的 E1 批（tag=US_VS_CHAMPION_BASE, seeds 7..14 × 2 座位 = 16 局）跑到一半，
#   上一个 subagent 死掉、**整棵进程树在 2026-10-29 02:16:47 被掐掉**，留下 5 局有效 + 3 局 timeout + 8 局空壳。
#   本脚本按"日志里没有 `verdict=engine` 就补跑"的 resume 语义只补剩下的局，**已完成的 5 局一个字节都不碰**。
#
# 复用点（与历史读数同源，勿改）：
#   * 驱动 = `_tmp_ladder.py` → `code/test_match/match_via_sdk.py`（本项目所有 ladder 批次都用它）；
#   * 配置 = `BASE`（与 `_us_vs_external.sh` / `_inject_vs_runnerup.sh` / `_oracle_inject_ab.sh` **逐字相同**，下面有断言）；
#   * token 语义 = 无后缀 `9` ⇒ 我们执 **P0**；`9r` ⇒ 镜像，我们执 **P1**；
#   * stderr 命名 = `MVS_STDERR_SUFFIX=1` + `DSH_PROGRESS_SUFFIX=_<TAG>`（**与 2026-10-28 那批逐字相同**，
#     否则新老 stderr 落不到同一套文件名上，`_batch_table.py` 读不到）。
#
# 三条硬约束（都是被坑出来的，别改）：
#   1. **不重跑已完成局**：todo 由 `grep -q 'verdict=engine'` 派生，绝不按"我觉得这局不算"重跑
#      （一局 35–45 分钟，白跑一轮就是 3 小时）。
#   2. **单局超时放宽到 3 小时**（`LADDER_TIMEOUT=10800`）：冠军是 O0 编译 + 全项目最重的 MCTS。
#      2026-10-28 那批没设它（默认 7200s），9r/10/10r 三局正是**被 2 小时墙钟掐断**的
#      ⇒ 这次的 timeout 不会再"白跑"（那一批的 3 个 timeout 局就是本脚本要补的）。
#   3. **并发时 stderr 会互相覆盖**：`match_results/ai?_<label>_seed<N>...stderr.log` 不按 tag 分目录，
#      同 seed 的两个座位 / 不同批次会写同一个文件。本批用 `MVS_STDERR_SUFFIX=1` 唯一化，
#      并在开跑前把**本批待补 token 的旧 stderr**挪到 `match_results/champion_e1/_attempts/<时间戳>/`
#      （只挪不删 ⇒ 上一轮 timeout 局的证据可复查；只挪本批 tag 后缀的文件 ⇒ 绝不碰别的实验）。
#
# ⚠️ **禁止**按镜像名/通配符批量 kill（用户明确要求）。本脚本不 kill 任何进程。
# ⚠️ E1 是**纯基线**：注入 / 安慰剂 / 升级 / PIN_K 一概不许带（下面有断言）。
#
# 用法（推荐用 run_logged 包一层，日志落盘）：
#   bash code/run_logged.sh us_vs_champion_e1_resume bash code/test_match/_us_vs_champion_e1_resume.sh
#   续跑：同一条命令原样再敲（已 verdict=engine 的局会跳过）
#   只看要跑什么不跑：DRY=1 bash code/test_match/_us_vs_champion_e1_resume.sh
#   冒烟：ROUNDS=24 JOBS=2 SEEDS="7 7r" TAG=US_VS_CHAMPION_SMOKE3 bash code/test_match/_us_vs_champion_e1_resume.sh
set -uo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO" || exit 1
export PYTHONIOENCODING=utf-8

P="${PYTHON:-D:/anaconda3/envs/pytorch-gpu/python.exe}"
# CPU 纪律（2026-10-29 用户指定）：机器上还有 PIN_K2_VS_RU 约 4 局在飞 ⇒ 本批开 5 局，
#   两边合计 ≤10 局并行（32 逻辑核）。2026-10-28 那批失控到 11–12 局并行 ⇒ 单局慢 7 倍 + 3 局读超时。
JOBS="${JOBS:-5}"
RETRY="${RETRY:-0}"            # 默认**不自动重试**：失败的局要人看着原因再决定（"别硬凑"）
SEEDS="${SEEDS:-7 7r 8 8r 9 9r 10 10r 11 11r 12 12r 13 13r 14 14r}"
TAG="${TAG:-US_VS_CHAMPION_BASE}"
ROUNDS="${ROUNDS:-}"           # 冒烟/截断用；空 = 完整对局（512 回合上限）
DRY="${DRY:-0}"
OUTROOT="${OUTROOT:-match_results/champion_e1}"

# 冠军是 O0 编译（慢）：单局 3 小时上限；单次读包 30 min（卡死检测）。
#   ⚠️ 后者只决定"多慢算卡死"，不是每步预算（我们的搜索是预算制，不吃墙钟）。
export LADDER_TIMEOUT="${LADDER_TIMEOUT:-10800}"
export MVS_TIMEOUT_SECONDS="${MVS_TIMEOUT_SECONDS:-1800}"
# stderr 唯一化（与 2026-10-28 那批**逐字相同**）
export MVS_STDERR_SUFFIX=1
export DSH_PROGRESS_SUFFIX="_${TAG}"

CHAMP_REL="其他版本ai/ant-war2-magica-v3/magica_v3_O0.exe"
ROOT_W="$(pwd -W | sed 's#/$##')"
CHAMP="$ROOT_W/$CHAMP_REL"
LOGDIR="match_results/ladder_logs/$TAG"

# 配置基线（任务书逐字；与 `_us_vs_external.sh` / `_inject_vs_runnerup.sh` 的 BASE 相同）
BASE="AZAI_CKPT=checkpoints/posnet_A_k5_m32.pt;AZAI_ITERS=256;AZAI_DEPTH=4;AZAI_SAMPLE_MULT=15;AZAI_MODE=pos-only;AZAI_POSPIN=mc_quiet;AZAI_MC_EVERY=1;AZAI_MC_ROOTONLY=1;AZAI_MC_CELLS=4;AZAI_SKIP1=1;AZAI_TCLASS=0.5;AZAI_TPOS=1.0;AZAI_TEMP=1e-6;AZAI_POSPRIOR=off;AZAI_SEARCH=flat;AZAI_MC_HORIZON=16;AZAI_MC_SCORE=net;AZAI_MC_NET_SAMPLES=4;AZAI_FLAT_BATCH=32;AZAI_FLAT_HORIZON=16;AZAI_FLAT_SHARE_PREFIX=1;AZAI_K=48;AZAI_FLAT_ROLLOUTS=64;AZAI_FLAT_ROLLOUT_MODE=random;AZAI_FLAT_RANDOM_SIDE=opponent;AZAI_FLAT_PROB_MODE=decay;AZAI_FLAT_DECAY_C=1.0;AZAI_FLAT_DECAY_K=3.0;AZAI_FLAT_ROUNDS=49152;AZAI_FLAT_SEED=53"

echo "=============================================================="
echo "[e1r] TAG=$TAG  冠军 exe=$CHAMP"
echo "[e1r] seeds=$SEEDS  jobs=$JOBS  retry=$RETRY  rounds=${ROUNDS:-<完整>}  dry=$DRY"
echo "[e1r] LADDER_TIMEOUT=${LADDER_TIMEOUT}s  MVS_TIMEOUT_SECONDS=${MVS_TIMEOUT_SECONDS}s"
echo "[e1r] MVS_STDERR_SUFFIX=$MVS_STDERR_SUFFIX  DSH_PROGRESS_SUFFIX=$DSH_PROGRESS_SUFFIX"
echo "[e1r] $(date '+%Y-%m-%d %H:%M:%S')"
echo "=============================================================="

# ── 前置断言（任一失败立刻退出，别白跑几小时）─────────────────────────────────
case "$CHAMP" in
    D:/*) echo "[e1r] ✓ 冠军 exe 路径是 D:/ 形态（Windows 绝对路径）" ;;
    *)    echo "[e1r] !! 冠军 exe 不是 D:/ 形态: $CHAMP" >&2; exit 2 ;;
esac
[ -f "$CHAMP" ] || { echo "[e1r] !! 冠军 exe 不存在: $CHAMP" >&2; exit 2; }
# 冠军是 mingw 动态链接的：缺 DLL ⇒ 0xC0000139 秒退（历史坑，见 docs/us_vs_champion_20261028.md §1.1）
for d in libstdc++-6.dll libgcc_s_seh-1.dll libwinpthread-1.dll; do
    [ -f "$(dirname "$CHAMP")/$d" ] || { echo "[e1r] !! 缺 $d（冠军会 0xC0000139 直接死）" >&2; exit 2; }
done
echo "[e1r] ✓ 冠军 exe + 3 个 mingw DLL 都在"

# BASE 必须与历史批次逐字相同 —— 否则"配置可比/可配对"这句话就不成立
BASE_REF="$(sed -n 's/^BASE="\(.*\)"$/\1/p' code/test_match/_us_vs_external.sh)"
if [ "$BASE" = "$BASE_REF" ]; then
    echo "[e1r] ✓ BASE 与 _us_vs_external.sh 逐字相同（${#BASE} 字符）"
else
    echo "[e1r] !! BASE 与历史 BASE 不同 ⇒ 停（先对齐配置）" >&2
    echo "  本次: ${#BASE} 字符" >&2
    echo "  参考: ${#BASE_REF} 字符" >&2
    exit 3
fi

# ── 只跑"日志里还没有 verdict=engine"的 token ────────────────────────────────
remaining_tokens() {
    local out="$1" tok
    for tok in $SEEDS; do
        if ! grep -q "verdict=engine" "$out/$tok.log" 2>/dev/null; then
            printf '%s ' "$tok"
        fi
    done
}
# 已完成（有 verdict=engine）的 token ⇒ 本脚本**绝不重跑**，也绝不动它们的 stderr
done_tokens() {
    local out="$1" tok
    for tok in $SEEDS; do
        if grep -q "verdict=engine" "$out/$tok.log" 2>/dev/null; then
            printf '%s ' "$tok"
        fi
    done
}

mkdir -p "$LOGDIR" "$OUTROOT"
DONE="$(done_tokens "$LOGDIR")"
TODO="$(remaining_tokens "$LOGDIR")"
echo "[e1r] ✅ 已完成（不重跑）: ${DONE:-<无>}"
echo "[e1r] ⏳ 待补跑: ${TODO:-<无>}"
[ -n "${TODO// /}" ] || { echo "[e1r] 没有待跑的局了（全部 verdict=engine）"; exit 0; }
[ "$DRY" = "1" ] && { echo "[e1r] DRY=1 ⇒ 只打印不跑"; exit 0; }

# ── 动前先把**待补 token** 的旧 stderr 挪走（只挪不删；只挪本批 tag 后缀）────────────
# 为什么按 (座位, seed) 精确算而不是整目录搬：同一个 seed 有两个座位两个文件，其中
# **已完成局**那一份是证据（例如 seed 9 的 `ai1_champion_seed9` 属于已完成的 `9`），绝不能动。
stash_stderr() {
    local why="$1" dest tok base seat
    dest="$OUTROOT/_attempts/$why"
    mkdir -p "$dest"
    for tok in $TODO; do
        base="${tok%r}"
        if [ "$tok" = "$base" ]; then seat=0; else seat=1; fi
        for f in "match_results/ai${seat}_az_bridge_ai_seed${base}_${TAG}.stderr.log" \
                 "match_results/ai$((1-seat))_champion_seed${base}_${TAG}.stderr.log"; do
            [ -f "$f" ] || continue
            mv -f "$f" "$dest/" 2>/dev/null && echo "[e1r]   旧 stderr 挪走: $(basename "$f") -> $dest/"
        done
    done
}

# ── 归档：本批两名 AI 的 stderr → `<OUTROOT>/stderr/`（去掉 `_<TAG>` 后缀，用历史规范名）──
# 冠军那份**极其**冗长（实测 ~0.2MB/回合 ⇒ 一局 100MB+；`ANTWAR_MAGICA_LOG_LEVEL` 在 O0 版里无效，
# 见 §1.2）⇒ gzip 留存（约 10:1）；我们自己的原样存（`_batch_table.py` / 老工具直接读）。
archive_stderr() {
    local dir="$OUTROOT/stderr" f base
    mkdir -p "$dir"
    for f in match_results/ai?_*_seed*_"${TAG}".stderr.log; do
        [ -f "$f" ] || continue
        base="$(basename "$f")"
        base="${base/_${TAG}/}"
        case "$base" in
            *champion*|*magica*)
                gzip -1 -c "$f" > "$dir/${base}.gz" 2>/dev/null \
                    && echo "[e1r]   冠军 stderr 已 gzip: ${base}.gz" ;;
            *)
                cp -f "$f" "$dir/$base" 2>/dev/null ;;
        esac
    done
    echo "[e1r] stderr 归档 -> $dir（$(ls -1 "$dir" 2>/dev/null | wc -l) 个文件）"
}

# ── 环境：清掉继承来的 AZAI_*，再逐条 export BASE（否则会继承别的臂的开关）──────────
for v in $(env | sed -n 's/^\(AZAI_[A-Za-z_0-9]*\)=.*/\1/p'); do unset "$v"; done
for kv in $(echo "$BASE" | tr ';' '\n'); do
    [ -n "$kv" ] && export "$kv"
done
# E1 纯基线自检
[ -z "${AZAI_ORACLE_INJECT:-}" ]     || { echo "[e1r] !! E1 竟然带着 AZAI_ORACLE_INJECT" >&2; exit 4; }
[ -z "${AZAI_ORACLE_PLACEBO:-}" ]    || { echo "[e1r] !! E1 竟然带着 AZAI_ORACLE_PLACEBO" >&2; exit 4; }
[ -z "${AZAI_ONLY_EXTRA:-}" ]        || { echo "[e1r] !! E1 竟然带着 AZAI_ONLY_EXTRA" >&2; exit 4; }
[ -z "${AZAI_ONLY_EXTRA_RANDOM:-}" ] || { echo "[e1r] !! E1 竟然带着 AZAI_ONLY_EXTRA_RANDOM" >&2; exit 4; }
[ -z "${AZAI_UPG_BUNDLES:-}" ]       || { echo "[e1r] !! E1 竟然带着 AZAI_UPG_BUNDLES" >&2; exit 4; }
[ -z "${AZAI_PIN_K:-}" ]             || { echo "[e1r] !! E1 竟然带着 AZAI_PIN_K" >&2; exit 4; }
[ -z "${AZAI_MC_UPG:-}" ]            || { echo "[e1r] !! E1 竟然带着 AZAI_MC_UPG" >&2; exit 4; }
[ "${MVS_STDERR_SUFFIX}" = "1" ]     || { echo "[e1r] !! MVS_STDERR_SUFFIX != 1" >&2; exit 4; }
echo "[e1r] ✓ E1 纯基线自检：注入/安慰剂/only-extra/升级/PIN_K 全部为空"
echo "[e1r] 环境自检 ✓  AZAI_CKPT=$AZAI_CKPT  AZAI_K=$AZAI_K  AZAI_FLAT_ROUNDS=$AZAI_FLAT_ROUNDS  AZAI_FLAT_SEED=$AZAI_FLAT_SEED"

stash_stderr "$(date '+%Y%m%d_%H%M%S')"

attempt=0
while [ "$attempt" -le "$RETRY" ]; do
    TODO="$(remaining_tokens "$LOGDIR")"
    if [ -z "${TODO// /}" ]; then
        echo "[e1r] 没有待跑的局（全部 verdict=engine）"
        break
    fi
    echo ""
    echo "[e1r] 第 $((attempt+1)) 轮：待跑 = $TODO  $(date '+%H:%M:%S')"
    for tok in $TODO; do
        [ -f "$LOGDIR/$tok.log" ] || continue
        mkdir -p "$LOGDIR/_attempts"
        cp -f "$LOGDIR/$tok.log" "$LOGDIR/_attempts/$tok.attempt$attempt.log" 2>/dev/null
    done
    t0=$(date +%s)
    "$P" -u _tmp_ladder.py --tag="$TAG" --jobs="$JOBS" \
        --ai0=code/test_match/az_bridge_ai.py --ai1="$CHAMP" \
        ${ROUNDS:+--rounds="$ROUNDS"} $TODO 2>&1 | tee -a "$LOGDIR/_run.log"
    rc=${PIPESTATUS[0]}
    t1=$(date +%s)
    echo "[e1r] 第 $((attempt+1)) 轮结束 rc=$rc  $(( (t1-t0)/60 )) min  $(date '+%H:%M:%S')"
    attempt=$((attempt+1))
    [ "$rc" -ne 0 ] && echo "[e1r] ⚠️ ladder rc=$rc（继续按 resume 逻辑补跑）"
done

archive_stderr

# ── 收尾：逐局读数 + 三条硬断言（verdict=engine / illegal=0 / 冠军侧 stderr 非空）──────
echo ""
echo "-------- [$TAG] 逐局读数 --------"
printf '  %-4s %-4s %-9s %-6s %-11s %-6s %-7s %-6s %-6s\n' \
    "tok" "我们" "verdict" "回合" "hp(我,敌)" "Δhp" "ill(裁判)" "ill(AI)" "冠军stderr"
fails=0
for tok in $SEEDS; do
    base="${tok%r}"
    if [ "$tok" = "$base" ]; then seat=0; else seat=1; fi
    resl="$(grep -h "RESULT seed" "$LOGDIR/$tok.log" 2>/dev/null | tail -1)"
    verdict="$(echo "$resl" | grep -o 'verdict=[a-z_]*' | cut -d= -f2)"
    rounds="$(echo "$resl" | grep -o 'rounds=[0-9]*' | cut -d= -f2)"
    hp0="$(echo "$resl" | sed -n 's/.*base_hp=\([0-9]*\),\([0-9]*\).*/\1/p')"
    hp1="$(echo "$resl" | sed -n 's/.*base_hp=\([0-9]*\),\([0-9]*\).*/\2/p')"
    ill="$(echo "$resl" | grep -o 'illegal=[0-9]*' | cut -d= -f2)"
    if [ "$seat" = "0" ]; then hp_ours="$hp0"; hp_opp="$hp1"; else hp_ours="$hp1"; hp_opp="$hp0"; fi
    f="$OUTROOT/stderr/ai${seat}_az_bridge_ai_seed${base}.stderr.log"
    if [ -f "$f" ]; then
        our_ill="$(grep -o 'illegal_total=[0-9]*' "$f" | tail -1 | cut -d= -f2)"
    else
        our_ill="-"
    fi
    cf="$OUTROOT/stderr/ai$((1-seat))_champion_seed${base}.stderr.log.gz"
    if [ -f "$cf" ]; then st_size="$(wc -c < "$cf" | tr -d ' ')"; else st_size="-"; fi
    printf '  %-4s %-4s %-9s %-6s %-11s %-6s %-7s %-6s %-6s\n' \
        "$tok" "P$seat" "${verdict:-无}" "${rounds:-?}" "${hp_ours:-?},${hp_opp:-?}" \
        "$( [ -n "${hp_ours:-}" ] && [ -n "${hp_opp:-}" ] && echo $((hp_ours - hp_opp)) || echo '?')" \
        "${ill:-?}" "${our_ill:-?}" "${st_size}"
    [ "${verdict:-}" = "engine" ] || { echo "    !! $tok verdict=${verdict:-无}（不是 engine）"; fails=$((fails+1)); }
    [ "${ill:-x}" = "0" ] || { echo "    !! $tok 裁判侧 illegal=${ill:-?}（必须 0）"; fails=$((fails+1)); }
    [ "${our_ill:-x}" = "0" ] || { echo "    !! $tok 我方 illegal_total=${our_ill:-?}（必须 0）"; fails=$((fails+1)); }
done
echo ""
echo "-------- [$TAG] RESULT 行 --------"
for tok in $SEEDS; do
    grep -h "RESULT seed" "$LOGDIR/$tok.log" 2>/dev/null | sed "s/^/  [$tok] /"
done
echo ""
if [ "$fails" -eq 0 ]; then
    echo "[e1r] ✅ [$TAG] 断言全过：verdict=engine / illegal=0（裁判侧+AI 侧）"
else
    echo "[e1r] ❌ [$TAG] 有 $fails 条断言失败（见上面的 !!）"
fi
echo "[e1r] 读数："
echo "  $P -u code/test_match/_batch_table.py --tag $TAG --stderr $OUTROOT/stderr --opp-label champion"
exit "$fails"
