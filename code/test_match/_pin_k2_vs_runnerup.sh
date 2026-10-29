#!/usr/bin/env bash
# `AZAI_PIN_K=2` A/B（预注册 16 局）：**我们（放宽类轴）对打亚军本体**（2026-10-28）
#
# 报告：`docs/pin_k2_ab_20261028.md`（预注册判据在报告 §0，**跑之前**写定）。
#
# 设计（预注册，见报告 §0.1–§0.6）：
#   * 实验臂 E = `BASE` + **只多** `AZAI_PIN_K=2`；
#   * 基线臂 B = `BASE`，**复用** `INJECT_VS_RU_B`（16 局，2W-14L，血量差 −20.12）——
#     前提是配置/seed/座位/token **逐字同源**（本脚本用一条 `BASE` 逐字断言 + 同一份 seed 清单
#     把这件事钉死；`analyze_pin_k2_ab.py` 再逐 token 核一遍 token 集合与座位）；
#   * 对手 = **亚军本体**（同一份 exe）；seeds `7..14 × 2 座位` = 16 局，与 B 臂**逐 token 同源**；
#   * 驱动 = `_tmp_ladder.py` + `match_via_sdk.py`（与历史所有"我们 vs 亚军"批次**同一个 harness**
#     ⇒ 日志格式 / token 语义 / `RESULT` 行 / stderr 归档全部同源，读数可配对）。
#
# 为什么照抄 `_us_vs_external.sh` 而不是新写一套：那支已经把三件被踩过的事做对了 ——
#   ① `--resume` 语义（跳过已有 `verdict=engine` 的 token，只补跑无效局，最多 RETRY 轮）；
#   ② **并发时 stderr 会被同名覆盖** ⇒ `MVS_STDERR_SUFFIX=1` + `DSH_PROGRESS_SUFFIX=_<TAG>`
#      （`match_results/ai?_<label>_seed<N>.stderr.log` **不按 tag 分目录**，同 seed 互相覆盖，
#      我们踩过两次）；
#   ③ `BASE` 与历史批次逐字断言 + 臂的开关前置断言（"静默失效"会被伪装成"没效果"）。
#
# ⚠️ **禁止**按镜像名/通配符批量 kill（用户明确要求）。本脚本自己不 kill 任何进程；
#    要中止就 Ctrl-C 或用 PID 精确处理。
# ⚠️ **不开** AZAI_ORACLE_INJECT / AZAI_ORACLE_PLACEBO / AZAI_UPG_BUNDLES / AZAI_MC_UPG /
#    AZAI_ONLY_EXTRA*（下面有断言，带了就 exit 4）。
#
# CPU 纪律：机器上可能**同时**还有别的批次在跑（2026-10-28 是 `US_VS_CHAMPION_BASE`，jobs=8）
#   ⇒ 默认 `JOBS=4`。确认没有别的批次时再 `JOBS=8`（16 核：每局 = 我们 1 进程 + 对手 1 进程）。
#
# 用法：
#   # 正式 16 局（跑完再敲一遍 = 只补没 verdict=engine 的局）
#   bash code/run_logged.sh pin_k2_ab bash code/test_match/_pin_k2_vs_runnerup.sh
#   # 冒烟（截断回合 + 打开逐决策候选日志）
#   ROUNDS=60 JOBS=2 SEEDS="7 7r" FLAT_VERBOSE=1 TAG=PIN_K2_SMOKE \
#       bash code/test_match/_pin_k2_vs_runnerup.sh
set -uo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO" || exit 1
export PYTHONIOENCODING=utf-8

P="${PYTHON:-D:/anaconda3/envs/pytorch-gpu/python.exe}"
JOBS="${JOBS:-4}"
RETRY="${RETRY:-2}"
PIN_K="${PIN_K:-2}"
SEEDS="${SEEDS:-7 7r 8 8r 9 9r 10 10r 11 11r 12 12r 13 13r 14 14r}"
ROUNDS="${ROUNDS:-}"                       # 冒烟/截断用；空 = 完整对局（512 回合上限）
FLAT_VERBOSE="${FLAT_VERBOSE:-0}"          # 1 ⇒ 每决策打一行 `[flat] 候选 N 个`（只进日志，零影响）
# ⚠️ 这两个是**harness 的"多久判死"参数，不是搜索配置**（我们的搜索是预算制，不吃墙钟），
#   所以两臂允许不同；它们只决定"一局/一次读包最多等多久"，不影响任一对局的走法。
#   · `LADDER_TIMEOUT`：单局墙钟上限。B 臂用默认 7200s（实测 31-46 min，从未触发）；
#     本臂放宽到 10800s，因为 PIN_K=2 的决策更贵（p50 0.56→~2.9 s），且多局并行时会互相挤。
#   · `MVS_TIMEOUT_SECONDS`：**单次读包**上限（卡死检测）。**与 B 臂同为默认 300 s**
#     （B 臂那批 `_inject_vs_runnerup.sh` 没设这个变量 ⇒ 默认值）——两臂口径一致。
#     实测本臂最慢单决策 37 s（冒烟/首轮），300 s 有 8 倍余量。
export LADDER_TIMEOUT="${LADDER_TIMEOUT:-10800}"
export MVS_TIMEOUT_SECONDS="${MVS_TIMEOUT_SECONDS:-300}"

RU_REL="其他版本ai/saiblo-30th-AI/Game1/antgame_ai_cpp/cpp_lure_v4/build/ai_cpp_lure_v4.exe"
ROOT_W="$(pwd -W | sed 's#/$##')"
# ⚠️ 必须是 **Windows 绝对路径（D:/…）**：Git-bash 的 `/d/...` 在 Windows 上
#    `Path.is_absolute()` 判 False ⇒ 会拼成 `<repo>/d/...` ⇒ 找不到 exe。
RU="$ROOT_W/$RU_REL"

TAG="${TAG:-PIN_K2_VS_RU}"
OUTROOT="match_results/pin_k2_vs_ru"
LOGDIR="match_results/ladder_logs/$TAG"

# 配置基线：与 `_inject_vs_runnerup.sh:48` 的 BASE **逐字相同**（断言在下面）
BASE="AZAI_CKPT=checkpoints/posnet_A_k5_m32.pt;AZAI_ITERS=256;AZAI_DEPTH=4;AZAI_SAMPLE_MULT=15;AZAI_MODE=pos-only;AZAI_POSPIN=mc_quiet;AZAI_MC_EVERY=1;AZAI_MC_ROOTONLY=1;AZAI_MC_CELLS=4;AZAI_SKIP1=1;AZAI_TCLASS=0.5;AZAI_TPOS=1.0;AZAI_TEMP=1e-6;AZAI_POSPRIOR=off;AZAI_SEARCH=flat;AZAI_MC_HORIZON=16;AZAI_MC_SCORE=net;AZAI_MC_NET_SAMPLES=4;AZAI_FLAT_BATCH=32;AZAI_FLAT_HORIZON=16;AZAI_FLAT_SHARE_PREFIX=1;AZAI_K=48;AZAI_FLAT_ROLLOUTS=64;AZAI_FLAT_ROLLOUT_MODE=random;AZAI_FLAT_RANDOM_SIDE=opponent;AZAI_FLAT_PROB_MODE=decay;AZAI_FLAT_DECAY_C=1.0;AZAI_FLAT_DECAY_K=3.0;AZAI_FLAT_ROUNDS=49152;AZAI_FLAT_SEED=53"
# ★ 本实验的**唯一**变量
EXTRA=";AZAI_PIN_K=$PIN_K"
[ "$FLAT_VERBOSE" = "1" ] && EXTRA="$EXTRA;AZAI_FLAT_VERBOSE=1"

# 并发写同一个 stderr 文件名会把上一局覆盖掉 ⇒ 本批的文件名唯一（含 TAG 后缀）
export MVS_STDERR_SUFFIX=1
export DSH_PROGRESS_SUFFIX="_${TAG}"

echo "=============================================================="
echo "[pk2] 臂 = E(PIN_K=$PIN_K) vs 基线 B(复用 INJECT_VS_RU_B)"
echo "[pk2] 对手 exe = $RU"
echo "[pk2] TAG=$TAG  seeds=$SEEDS  jobs=$JOBS  retry=$RETRY  rounds=${ROUNDS:-<完整>}"
echo "[pk2] LADDER_TIMEOUT=${LADDER_TIMEOUT}s  MVS_TIMEOUT_SECONDS=${MVS_TIMEOUT_SECONDS}s"
echo "[pk2] $(date '+%Y-%m-%d %H:%M:%S')"
echo "=============================================================="

# ── 前置断言（任一失败立刻退出，别白跑几小时）─────────────────────────────────
case "$RU" in
    D:/*) echo "[pk2] ✓ 亚军 exe 路径是 D:/ 形态（Windows 绝对路径）" ;;
    *)   echo "[pk2] !! 亚军 exe 不是 D:/ 形态，终止: $RU" >&2; exit 2 ;;
esac
[ -f "$RU" ] || { echo "[pk2] !! 亚军 exe 不存在: $RU" >&2; exit 2; }
for d in libstdc++-6.dll libgcc_s_seh-1.dll libwinpthread-1.dll; do
    [ -f "$(dirname "$RU")/$d" ] || { echo "[pk2] !! 缺 $d（亚军会 0xC0000139 直接死）" >&2; exit 2; }
done
echo "[pk2] ✓ 亚军 exe + 3 个 DLL 都在"

# BASE 必须与 `_inject_vs_runnerup.sh` 逐字相同 —— 否则"与 B 臂可比/可配对"这句话就不成立
BASE_REF="$(sed -n 's/^BASE="\(.*\)"$/\1/p' code/test_match/_inject_vs_runnerup.sh)"
if [ "$BASE" = "$BASE_REF" ]; then
    echo "[pk2] ✓ BASE 与 _inject_vs_runnerup.sh 逐字相同（${#BASE} 字符）"
else
    echo "[pk2] !! BASE 与历史 BASE 不同 ⇒ 停（先对齐配置）" >&2
    echo "  本次: ${#BASE} 字符" >&2
    echo "  参考: ${#BASE_REF} 字符" >&2
    exit 3
fi

# seed 清单必须与 B 臂**逐 token**相同（配对的前提）
B_SEEDS_REF="7 7r 8 8r 9 9r 10 10r 11 11r 12 12r 13 13r 14 14r"
if [ "$SEEDS" = "$B_SEEDS_REF" ]; then
    echo "[pk2] ✓ seed 清单与 INJECT_VS_RU_B 逐 token 相同（16 局，8 seed × 2 座位）"
else
    echo "[pk2] ⚠️ seed 清单与 B 臂不同（$SEEDS）⇒ 只能做非配对比较，报告里必须写明"
fi

# 该 tag 里还没有 verdict=engine 的 token（= 需要跑的）
remaining_tokens() {
    local out="$1" tok
    for tok in $SEEDS; do
        if ! grep -q "verdict=engine" "$out/$tok.log" 2>/dev/null; then
            printf '%s ' "$tok"
        fi
    done
}

# 把上一批同 seed 的 stderr 挪到一边（**只挪本脚本自己的后缀**，别的实验的文件绝不能动）
stash_stderr() {
    local why="$1" dest f
    dest="$OUTROOT/_stale/$why"
    mkdir -p "$dest"
    for f in match_results/ai?_*_seed{7,8,9,10,11,12,13,14}*"${DSH_PROGRESS_SUFFIX}".stderr.log; do
        [ -f "$f" ] || continue
        mv -f "$f" "$dest/" 2>/dev/null && echo "[pk2]   旧 stderr 挪走: $(basename "$f") -> $dest/"
    done
}

# 归档：我方 stderr → `<OUTROOT>/<TAG>/stderr/`，**去掉 TAG 后缀**，用历史规范名
# （这样 `analyze_inject_vs_runnerup.py` / `_batch_table.py` / `az_bridge_ai` 的旧工具都能直接读）
archive_stderr() {
    local dir="$OUTROOT/$TAG/stderr" f base
    mkdir -p "$dir"
    for f in match_results/ai?_*_seed{7,8,9,10,11,12,13,14}*"${DSH_PROGRESS_SUFFIX}".stderr.log; do
        [ -f "$f" ] || continue
        base="$(basename "$f")"
        base="${base/${DSH_PROGRESS_SUFFIX}/}"
        cp -f "$f" "$dir/$base" 2>/dev/null
    done
    echo "[pk2] stderr 归档 -> $dir（$(ls -1 "$dir" 2>/dev/null | wc -l) 个文件）"
}

# 逐局读数 + **四条硬断言**（verdict=engine / 裁判侧 illegal=0 / 我方 illegal_total=0 / terminal）
finish() {
    local tok base seat f resl verdict rounds hp0 hp1 ill wall tmo our_ill en=0 fails=0 dir
    dir="$OUTROOT/$TAG/stderr"
    echo ""
    echo "-------- [$TAG] 逐局读数 --------"
    printf '  %-4s %-4s %-9s %-6s %-11s %-6s %-8s %-8s %-7s\n' \
        "tok" "我们" "verdict" "回合" "hp(我,敌)" "Δhp" "ill(裁判)" "ill(AI)" "墙钟"
    for tok in $SEEDS; do
        base="${tok%r}"
        if [ "$tok" = "$base" ]; then seat=0; else seat=1; fi
        resl="$(grep -h "RESULT seed" "$LOGDIR/$tok.log" 2>/dev/null | tail -1)"
        verdict="$(echo "$resl" | grep -o 'verdict=[a-z_]*' | cut -d= -f2)"
        rounds="$(echo "$resl" | grep -o 'rounds=[0-9]*' | cut -d= -f2)"
        hp0="$(echo "$resl" | sed -n 's/.*base_hp=\([0-9]*\),\([0-9]*\).*/\1/p')"
        hp1="$(echo "$resl" | sed -n 's/.*base_hp=\([0-9]*\),\([0-9]*\).*/\2/p')"
        ill="$(echo "$resl" | grep -o 'illegal=[0-9]*' | cut -d= -f2)"
        tmo="$(echo "$resl" | grep -o 'terminal=[A-Za-z]*' | cut -d= -f2)"
        if [ "$seat" = "0" ]; then hp_ours="$hp0"; hp_opp="$hp1"; else hp_ours="$hp1"; hp_opp="$hp0"; fi
        f="$dir/ai${seat}_az_bridge_ai_seed${base}.stderr.log"
        if [ -f "$f" ]; then
            our_ill="$(grep -o 'illegal_total=[0-9]*' "$f" | tail -1 | cut -d= -f2)"
        else
            our_ill="-"
        fi
        wall="$(grep -o "^\[$tok\] rc=[-0-9]* *[0-9.]* min" "$LOGDIR/_run.log" 2>/dev/null \
                | tail -1 | sed 's/.* //;s/ min//')"
        printf '  %-4s %-4s %-9s %-6s %-11s %-6s %-8s %-8s %-7s\n' \
            "$tok" "P$seat" "${verdict:-无}" "${rounds:-?}" "${hp_ours:-?},${hp_opp:-?}" \
            "$( [ -n "${hp_ours:-}" ] && [ -n "${hp_opp:-}" ] && echo $((hp_ours - hp_opp)) || echo '?')" \
            "${ill:-?}" "${our_ill:-?}" "${wall:-?}m"
        # ── 断言（§0.6）──
        [ "${verdict:-}" = "engine" ] || { echo "    !! $tok verdict=${verdict:-无}（不是 engine）"; fails=$((fails+1)); }
        [ "${ill:-x}" = "0" ] || { echo "    !! $tok 裁判侧 illegal=${ill:-?}（必须 0）"; fails=$((fails+1)); }
        [ "${our_ill:-x}" = "0" ] || { echo "    !! $tok 我方 illegal_total=${our_ill:-?}（必须 0）"; fails=$((fails+1)); }
        [ "${tmo:-}" = "True" ] || { echo "    !! $tok terminal=${tmo:-?}（必须 True）"; fails=$((fails+1)); }
        # PIN_K 生效旁证：stderr 里应当有 >1 个候选的 `[flat] 候选 N 个`（只在 FLAT_VERBOSE=1 时有）
        if [ "$FLAT_VERBOSE" = "1" ] && [ -f "$f" ]; then
            en="$(grep -o '\[flat\] 候选 [0-9]* 个' "$f" | sed 's/[^0-9]//g' | sort -n | tail -1)"
            [ -n "$en" ] && echo "    ·· $tok 单局最大候选数 = $en（PIN_K=$PIN_K）"
        fi
    done
    echo ""
    echo "-------- [$TAG] RESULT 行 --------"
    for tok in $SEEDS; do
        grep -h "RESULT seed" "$LOGDIR/$tok.log" 2>/dev/null | sed "s/^/  [$tok] /"
    done
    echo ""
    if [ "$fails" -eq 0 ]; then
        echo "[pk2] ✅ [$TAG] 四条硬断言全过：verdict=engine / illegal=0（裁判侧+AI 侧）/ terminal=True"
    else
        echo "[pk2] ❌ [$TAG] 有 $fails 条断言失败（见上面的 !!）"
    fi
    return "$fails"
}

# ── 起跑 ─────────────────────────────────────────────────────────────────────
export_env="$BASE$EXTRA"
mkdir -p "$LOGDIR"
# ⚠️ 先把继承来的 AZAI_* 全部清掉，再 export 本臂的 —— 否则别的批次的开关会被**继承**进来
#    （历史踩过：两臂变成同一臂，整批读数作废）
for v in $(env | sed -n 's/^\(AZAI_[A-Za-z_0-9]*\)=.*/\1/p'); do unset "$v"; done
for kv in $(echo "$export_env" | tr ';' '\n'); do
    [ -n "$kv" ] && export "$kv"
done

# 本臂的开关前置断言（少了这些，"静默失效"会伪装成"没效果"）
[ "${AZAI_PIN_K:-}" = "$PIN_K" ] || { echo "[pk2] !! AZAI_PIN_K=${AZAI_PIN_K:-<空>} != $PIN_K" >&2; exit 4; }
[ "${AZAI_PIN_K}" -ge 2 ] || { echo "[pk2] !! AZAI_PIN_K=$AZAI_PIN_K < 2 ⇒ 补丁整段不执行（等于基线臂）" >&2; exit 4; }
[ -z "${AZAI_ORACLE_INJECT:-}" ]   || { echo "[pk2] !! 竟然带着 AZAI_ORACLE_INJECT" >&2; exit 4; }
[ -z "${AZAI_ORACLE_PLACEBO:-}" ]  || { echo "[pk2] !! 竟然带着 AZAI_ORACLE_PLACEBO" >&2; exit 4; }
[ -z "${AZAI_UPG_BUNDLES:-}" ]     || { echo "[pk2] !! 竟然带着 AZAI_UPG_BUNDLES" >&2; exit 4; }
[ -z "${AZAI_MC_UPG:-}" ]          || { echo "[pk2] !! 竟然带着 AZAI_MC_UPG" >&2; exit 4; }
[ -z "${AZAI_ONLY_EXTRA:-}" ]      || { echo "[pk2] !! 竟然带着 AZAI_ONLY_EXTRA" >&2; exit 4; }
echo "[pk2] ✓ 环境自检：AZAI_PIN_K=$AZAI_PIN_K  AZAI_K=$AZAI_K  AZAI_FLAT_ROUNDS=$AZAI_FLAT_ROUNDS" \
     " AZAI_FLAT_SEED=$AZAI_FLAT_SEED  AZAI_ORACLE_INJECT=${AZAI_ORACLE_INJECT:-<空>}" \
     " AZAI_FLAT_VERBOSE=${AZAI_FLAT_VERBOSE:-<未设>}"

stash_stderr "${TAG}_$(date '+%Y%m%d_%H%M%S')"

attempt=0
while [ "$attempt" -le "$RETRY" ]; do
    todo="$(remaining_tokens "$LOGDIR")"
    if [ -z "${todo// /}" ]; then
        echo "[pk2] [$TAG] 没有待跑的局（全部 verdict=engine）"
        break
    fi
    echo ""
    echo "[pk2] [$TAG] 第 $((attempt+1)) 轮：待跑 = $todo  $(date '+%H:%M:%S')"
    for tok in $todo; do
        [ -f "$LOGDIR/$tok.log" ] || continue
        mkdir -p "$LOGDIR/_attempts"
        cp -f "$LOGDIR/$tok.log" "$LOGDIR/_attempts/$tok.attempt$attempt.log" 2>/dev/null
    done
    t0=$(date +%s)
    "$P" -u _tmp_ladder.py --tag="$TAG" --jobs="$JOBS" \
        --ai0=code/test_match/az_bridge_ai.py --ai1="$RU" \
        ${ROUNDS:+--rounds="$ROUNDS"} $todo 2>&1 | tee -a "$LOGDIR/_run.log"
    rc=${PIPESTATUS[0]}
    t1=$(date +%s)
    echo "[pk2] [$TAG] 第 $((attempt+1)) 轮结束 rc=$rc  $(( (t1-t0)/60 )) min  $(date '+%H:%M:%S')"
    attempt=$((attempt+1))
    [ "$rc" -ne 0 ] && echo "[pk2] [$TAG] ⚠️ ladder rc=$rc（继续按 resume 逻辑补跑）"
done

archive_stderr
finish
RC_FIN=$?

echo ""
echo "=============================================================="
echo "[pk2] 结束 $(date '+%Y-%m-%d %H:%M:%S')  TAG=$TAG  断言失败数=$RC_FIN"
echo "[pk2] 读数："
echo "  $P -u code/test_match/analyze_pin_k2_ab.py \\"
echo "      --arm-e $LOGDIR \\"
echo "      --arm-b match_results/ladder_logs/INJECT_VS_RU_B \\"
echo "      --stderr-e $OUTROOT/$TAG/stderr \\"
echo "      --stderr-b match_results/inject_vs_ru/B/stderr"
echo "=============================================================="
exit "$RC_FIN"
