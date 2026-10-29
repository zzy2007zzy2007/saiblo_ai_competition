#!/usr/bin/env bash
# 外部标尺：**我们（任意 AZAI_* 配置）直接对打外部黑盒 AI**（2026-10-28）
#
# 这一支脚本服务四个实验（用 `EXP=` 选；其余环境逐字相同 ⇒ 读数可配对比较）：
#   E1  = 我们（**不注入**）vs 冠军            ← 本文件第一次跑的就是它（`docs/us_vs_champion_20261028.md`）
#   E2  = 我们 + 注入**冠军**动作 vs 冠军        （`AZAI_ORACLE_INJECT=<冠军>`，`AZAI_ORACLE_TIMEOUT=20`）
#   E4  = 候选集**只保留**「冠军动作 + 亚军动作」两个，由评估器裁决（`AZAI_ONLY_EXTRA=1`）
#   E4R = 同上候选集，但**随机挑一个**（`AZAI_ONLY_EXTRA_RANDOM=1`，固定种子可复现）
#   （E4/E4R 的对手用 `OPP=champion|runnerup` 选，各 16 局）
#
# 设计（照抄历史所有"我们 vs 亚军"批次的 harness，理由见
# `_inject_vs_runnerup.sh` 文件头）：驱动 = `_tmp_ladder.py` + `match_via_sdk.py`，
# 日志格式/token 语义/stderr 归档/`RESULT` 行全部沿用 ⇒ 与历史读数**同源**。
#   * 每臂 16 局 = seeds 7..14 × 2 座位（`7` = 我们执 P0；`7r` = 镜像，我们执 P1）；
#   * `--resume` 语义：跳过已有 `verdict=engine` 的 token，只补跑无效局（最多 RETRY 轮）；
#   * 每局 stderr 先挪到一边（否则上一批同一个 seed 的日志会被误读成本批的）。
#
# 冠军（magica_v3_O0）特有的三条工程约束（**踩过/量过**，别改）：
#   1. **O0 编译 + 全项目最重的 MCTS** ⇒ 单局墙钟上限从 7200s 放宽到 `LADDER_TIMEOUT=10800s`，
#      单次读包超时（卡死检测）从 300s 放到 `MVS_TIMEOUT_SECONDS=1800s`。
#      两者都只是"别把慢局掐断"，**不是每步预算**（我们的搜索是预算制，不吃墙钟）。
#   2. **只接受 `\n`**（magica 的换行处理在 `match_via_sdk.normalize_lines` / `is_magica` 里，
#      bridge 侧已内建）；这里只用官方 bridge，**不要**自己起进程喂它。
#   3. 它的 `RESULT` 里标签是 **`champion`**（不是文件名 `magica_v3_O0`）——
#      `_tmp_ladder._archive_stderr` 按**文件名 stem** 过滤 labels ⇒ 会**跳过**冠军那份 stderr
#      （打印"没找到"）。所以归档在本脚本里**自己**做（glob 两种名字）。
#
# ⚠️ **禁止**按镜像名/通配符批量 kill（用户明确要求）。本脚本不 kill 任何进程。
# ⚠️ E1 是**纯基线**：`AZAI_ORACLE_INJECT` / `AZAI_ONLY_EXTRA*` 必须为空/0（下面有断言）。
#
# 用法：
#   bash code/run_logged.sh us_vs_champion_baseline bash code/test_match/_us_vs_external.sh
#   续跑：同一条命令原样再敲（已 verdict=engine 的局会跳过）
#   冒烟：ROUNDS=24 JOBS=2 SEEDS="7 7r" bash code/test_match/_us_vs_external.sh
set -uo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO" || exit 1
export PYTHONIOENCODING=utf-8

P="${PYTHON:-D:/anaconda3/envs/pytorch-gpu/python.exe}"
# CPU 纪律：2026-10-28 机器上还有别的实验在跑（升级诊断臂 8 局并行）⇒ 本批只开 4 局。
JOBS="${JOBS:-4}"
RETRY="${RETRY:-1}"
EXP="${EXP:-E1}"
OPP="${OPP:-champion}"
SEEDS="${SEEDS:-7 7r 8 8r 9 9r 10 10r 11 11r 12 12r 13 13r 14 14r}"
ROUNDS="${ROUNDS:-}"                       # 冒烟/截断用；空 = 完整对局（512 回合上限）
# 冠军是 O0 编译（慢）：单局 3 小时上限；单次读包 30 min（卡死检测）。都可被环境覆盖。
export LADDER_TIMEOUT="${LADDER_TIMEOUT:-10800}"
export MVS_TIMEOUT_SECONDS="${MVS_TIMEOUT_SECONDS:-1800}"

CHAMP_REL="其他版本ai/ant-war2-magica-v3/magica_v3_O0.exe"
RU_REL="其他版本ai/saiblo-30th-AI/Game1/antgame_ai_cpp/cpp_lure_v4/build/ai_cpp_lure_v4.exe"
ROOT_W="$(pwd -W | sed 's#/$##')"
# ⚠️ 必须是 **Windows 绝对路径（D:/…）**：Git-bash 的 `/d/...` 在 Windows 上
#   `Path.is_absolute()` 判 False ⇒ `_oracle_init` 把它拼成 `<repo>/d/...` ⇒ Popen 抛
#   FileNotFoundError ⇒ **注入整局静默不启用，而 A/B 看起来照跑**（踩过一次，
#   见 docs/oracle_inject_ab_20261028.md §4.1.1 第 2 条）。`pwd -W` 给的就是 D:/ 形态。
CHAMP="$ROOT_W/$CHAMP_REL"
RU="$ROOT_W/$RU_REL"
case "$OPP" in
    champion) OPP_EXE="$CHAMP" ;;
    runnerup) OPP_EXE="$RU" ;;
    *) echo "[uvx] !! OPP=$OPP 不认识（champion|runnerup）" >&2; exit 2 ;;
esac

# 配置基线（任务书逐字；与 `_inject_vs_runnerup.sh` / `_oracle_inject_ab.sh` 的 BASE **逐字相同**）
BASE="AZAI_CKPT=checkpoints/posnet_A_k5_m32.pt;AZAI_ITERS=256;AZAI_DEPTH=4;AZAI_SAMPLE_MULT=15;AZAI_MODE=pos-only;AZAI_POSPIN=mc_quiet;AZAI_MC_EVERY=1;AZAI_MC_ROOTONLY=1;AZAI_MC_CELLS=4;AZAI_SKIP1=1;AZAI_TCLASS=0.5;AZAI_TPOS=1.0;AZAI_TEMP=1e-6;AZAI_POSPRIOR=off;AZAI_SEARCH=flat;AZAI_MC_HORIZON=16;AZAI_MC_SCORE=net;AZAI_MC_NET_SAMPLES=4;AZAI_FLAT_BATCH=32;AZAI_FLAT_HORIZON=16;AZAI_FLAT_SHARE_PREFIX=1;AZAI_K=48;AZAI_FLAT_ROLLOUTS=64;AZAI_FLAT_ROLLOUT_MODE=random;AZAI_FLAT_RANDOM_SIDE=opponent;AZAI_FLAT_PROB_MODE=decay;AZAI_FLAT_DECAY_C=1.0;AZAI_FLAT_DECAY_K=3.0;AZAI_FLAT_ROUNDS=49152;AZAI_FLAT_SEED=53"

# ── 四个实验臂的差别**只有**这几行环境变量（其余逐字相同）─────────────────────────
# ⚠️ `AZAI_ORACLE_TIMEOUT=20`（亚军那批是 15）：冠军是 O0 编译，单次查询明显更慢。
# ⚠️ 双 oracle：A = 冠军（`AZAI_ORACLE_INJECT`），B = 亚军（`AZAI_ORACLE_INJECT_B`），
#    两者都走 `RunnerUpOracle`（它只是"按官方协议喂局面、收操作"的通用外壳，不绑定亚军）。
E4_ENV=";AZAI_ONLY_EXTRA=1;AZAI_ORACLE_INJECT=$CHAMP;AZAI_ORACLE_TIMEOUT=20;AZAI_ORACLE_INJECT_B=$RU;AZAI_ORACLE_TIMEOUT_B=20"
case "$EXP" in
    E1)  TAG="${TAG:-US_VS_${OPP^^}_BASE}";      EXTRA="" ;;
    E2)  TAG="${TAG:-US_VS_${OPP^^}_INJ}"
         EXTRA=";AZAI_ORACLE_INJECT=$CHAMP;AZAI_ORACLE_TIMEOUT=20" ;;
    E4)  TAG="${TAG:-ONLYEXTRA_VS_${OPP^^}}";    EXTRA="$E4_ENV" ;;
    E4R) TAG="${TAG:-ONLYEXTRA_RAND_VS_${OPP^^}}"
         EXTRA="$E4_ENV;AZAI_ONLY_EXTRA_RANDOM=1" ;;
    *)   echo "[uvx] !! EXP=$EXP 不认识（E1|E2|E4|E4R）" >&2; exit 2 ;;
esac
# E2 的对手只能是冠军（"注入冠军动作 vs 冠军"）
if [ "$EXP" = "E2" ] && [ "$OPP" != "champion" ]; then
    echo "[uvx] !! E2 的对手必须是冠军（注入的是冠军的动作）" >&2; exit 2
fi

# stderr / 进度文件名加后缀：2026-10-28 机器上**同时**有别的实验在跑，而
# `match_results/ai?_<label>_seed<N>.stderr.log` **不按 tag 分目录** ⇒ 同 seed 会互相覆盖
# （历史踩过，`docs/oracle_inject_design.md` §4.3）。加后缀后本批的日志文件名唯一。
export MVS_STDERR_SUFFIX=1
export DSH_PROGRESS_SUFFIX="_${TAG}"

OUTROOT="match_results/us_vs_external"
LOGDIR="match_results/ladder_logs/$TAG"

echo "=============================================================="
echo "[uvx] EXP=$EXP  对手=$OPP  对手 exe=$OPP_EXE"
echo "[uvx] TAG=$TAG  seeds=$SEEDS  jobs=$JOBS  retry=$RETRY  rounds=${ROUNDS:-<完整>}"
echo "[uvx] LADDER_TIMEOUT=${LADDER_TIMEOUT}s  MVS_TIMEOUT_SECONDS=${MVS_TIMEOUT_SECONDS}s"
echo "[uvx] $(date '+%Y-%m-%d %H:%M:%S')"
echo "=============================================================="

# ── 前置断言（任一失败立刻退出，别白跑几小时）─────────────────────────────────
case "$OPP_EXE" in
    D:/*) echo "[uvx] ✓ 对手 exe 路径是 D:/ 形态（Windows 绝对路径）" ;;
    *)    echo "[uvx] !! 对手 exe 不是 D:/ 形态: $OPP_EXE" >&2; exit 2 ;;
esac
[ -f "$OPP_EXE" ] || { echo "[uvx] !! 对手 exe 不存在: $OPP_EXE" >&2; exit 2; }
if [ "$OPP" = "runnerup" ]; then
    # 亚军是动态链接的，缺 DLL 会 0xC0000139 直接死
    for d in libstdc++-6.dll libgcc_s_seh-1.dll libwinpthread-1.dll; do
        [ -f "$(dirname "$OPP_EXE")/$d" ] || { echo "[uvx] !! 缺 $d" >&2; exit 2; }
    done
fi
[ -f "$CHAMP" ] || { echo "[uvx] !! 冠军 exe 不存在: $CHAMP" >&2; exit 2; }
echo "[uvx] ✓ 对手 exe（+ 需要的 DLL）都在"

# BASE 必须与历史批次逐字相同 —— 否则"配置可比/可配对"这句话就不成立
BASE_REF="$(sed -n 's/^BASE="\(.*\)"$/\1/p' code/test_match/_inject_vs_runnerup.sh)"
if [ "$BASE" = "$BASE_REF" ]; then
    echo "[uvx] ✓ BASE 与 _inject_vs_runnerup.sh 逐字相同（${#BASE} 字符）"
else
    echo "[uvx] !! BASE 与历史 BASE 不同 ⇒ 停（先对齐配置）" >&2
    echo "  本次: ${#BASE} 字符" >&2
    echo "  参考: ${#BASE_REF} 字符" >&2
    exit 3
fi

# 该实验里还没有 verdict=engine 的 token（= 需要跑的）
remaining_tokens() {
    local out="$1" tok
    for tok in $SEEDS; do
        if ! grep -q "verdict=engine" "$out/$tok.log" 2>/dev/null; then
            printf '%s ' "$tok"
        fi
    done
}

# 把上一批同 seed 的 stderr 挪到一边（只挪不删 ⇒ 历史留存；避免"读错批"）
stash_stderr() {
    local why="$1" dest f
    dest="$OUTROOT/_stale/$why"
    mkdir -p "$dest"
    for f in match_results/ai?_*_seed{7,8,9,10,11,12,13,14}*.stderr.log; do
        [ -f "$f" ] || continue
        # 只挪**本脚本自己**的后缀（别的实验的文件绝不能动）
        case "$f" in
            *"_${TAG}"*) ;;
            *) continue ;;
        esac
        mv -f "$f" "$dest/" 2>/dev/null && echo "[uvx]   旧 stderr 挪走: $(basename "$f") -> $dest/"
    done
}

# 归档：本批两名 AI 的 stderr → `<OUTROOT>/<TAG>/stderr/`（**去掉后缀**，用历史规范名，
# 这样 `analyze_inject_vs_runnerup.py --stderr` 能直接读）。
# ⚠️ 冠军的标签是 `champion`，`_tmp_ladder._archive_stderr` 按 stem（`magica_v3_O0`）过滤会跳过它 ⇒ 这里自己做。
archive_stderr() {
    local dir="$OUTROOT/$TAG/stderr" f base raw_mb
    mkdir -p "$dir"
    for f in match_results/ai?_*_seed{7,8,9,10,11,12,13,14}*.stderr.log; do
        [ -f "$f" ] || continue
        case "$f" in
            *"_${TAG}"*) ;;
            *) continue ;;
        esac
        base="$(basename "$f")"
        base="${base/_${TAG}/}"
        case "$base" in
            # ⚠️ 冠军自己的日志**极其**冗长（实测 ~0.2MB/回合 ⇒ 一局 100MB+：它的 MCTS 调试行
            #   是 O0 编译里写死的，`ANTWAR_MAGICA_LOG_LEVEL` 那个环境变量在 O0 里**不起作用**，
            #   见 §2.2）。⇒ 冠军那份 gzip 留存（诊断够用），我们自己的原样存（老工具直接读）。
            *champion*|*magica*)
                gzip -1 -c "$f" > "$dir/${base}.gz" 2>/dev/null \
                    && raw_mb=$((($(wc -c < "$f")) / 1048576)) \
                    && echo "[uvx]   冠军 stderr 已 gzip: ${base}.gz（原始 ${raw_mb} MB）" ;;
            *)
                cp -f "$f" "$dir/$base" 2>/dev/null ;;
        esac
    done
    echo "[uvx] stderr 归档 -> $dir（$(ls -1 "$dir" 2>/dev/null | wc -l) 个文件）"
}

# 逐局读数 + **三条硬断言**（verdict=engine / illegal=0 / 冠军侧 stderr 非空）
finish() {
    local tok base seat f resl dir verdict rounds hp0 hp1 ill wall st_size our_size fails=0
    dir="$OUTROOT/$TAG/stderr"
    echo ""
    echo "-------- [$TAG] 逐局读数 --------"
    printf '  %-4s %-4s %-9s %-6s %-11s %-6s %-7s %-6s %-6s %-9s %-8s %-8s %-6s %-6s\n' \
        "tok" "我们" "verdict" "回合" "hp(我,敌)" "Δhp" "ill(裁判)" "ill(AI)" "墙钟" "冠军stderr" "注入已启用" "asks" "to" "dead"
    for tok in $SEEDS; do
        base="${tok%r}"
        if [ "$tok" = "$base" ]; then seat=0; else seat=1; fi
        resl="$(grep -h "RESULT seed" "$LOGDIR/$tok.log" 2>/dev/null | tail -1)"
        verdict="$(echo "$resl" | grep -o 'verdict=[a-z_]*' | cut -d= -f2)"
        rounds="$(echo "$resl" | grep -o 'rounds=[0-9]*' | cut -d= -f2)"
        # base_hp 是 (p0,p1)；我们那一侧的 HP 要按座位取
        hp0="$(echo "$resl" | sed -n 's/.*base_hp=\([0-9]*\),\([0-9]*\).*/\1/p')"
        hp1="$(echo "$resl" | sed -n 's/.*base_hp=\([0-9]*\),\([0-9]*\).*/\2/p')"
        ill="$(echo "$resl" | grep -o 'illegal=[0-9]*' | cut -d= -f2)"
        if [ "$seat" = "0" ]; then hp_ours="$hp0"; hp_opp="$hp1"; else hp_ours="$hp1"; hp_opp="$hp0"; fi
        f="$dir/ai${seat}_az_bridge_ai_seed${base}.stderr.log"
        if [ -f "$f" ]; then
            our_ill="$(grep -o 'illegal_total=[0-9]*' "$f" | tail -1 | cut -d= -f2)"
            our_size="$(wc -c < "$f" | tr -d ' ')"
        else
            our_ill="-"; our_size="-"
        fi
        # 2026-10-29：**注入真的启用了吗**（E2/E4/E4R 的逐局断言，父任务要求）。
        #   为什么必须逐局查：`AZAI_ORACLE_INJECT` 写成 `/d/...` 形态、exe 缺 DLL、或会话启用后
        #   第一次读包就超时 ⇒ 那一局**退化成 E1 但看不出来**（对局照跑、日志照出）⇒ 主读数被稀释。
        #   判据 = stderr 里有 `注入已启用` **且** 收尾统计的 `asks>0`。
        inj_on="-"; inj_asks="-"; inj_to="-"; inj_dead="-"
        if [ "$EXP" != "E1" ] && [ -f "$f" ]; then
            # ⚠️ 前置空格：`hit_rate_asks=84.0%` 也会被 `asks=` 抓到，不锚定会读错
            inj_asks="$(grep -o ' asks=[0-9]*' "$f" | tail -1 | tr -d ' ' | cut -d= -f2)"
            inj_on="$(grep -c '注入已启用' "$f")"
            # 2026-10-29：**会话中途死掉**才是 E2 的真风险（单次读包超时 1 次 = 整个会话永久停用
            #   ⇒ 后半局悄悄退化成 E1）。所以把 timeout/dead 也逐局列出来（E4 显示的是最后一个会话）。
            _stat="$(grep '注入统计:' "$f" | tail -1)"
            inj_to="$(echo "$_stat" | grep -o 'timeout=[0-9]*' | head -1 | cut -d= -f2)"
            inj_dead="$(echo "$_stat" | grep -o 'dead=[0-9]*' | head -1 | cut -d= -f2)"
        fi
        # 冠军侧 stderr（名字有两种可能：标签 `champion` 或文件名 stem `magica_v3_O0`；归档是 .gz）
        st_size="-"
        for cf in "$dir/ai$((1-seat))_champion_seed${base}.stderr.log.gz" \
                  "$dir/ai$((1-seat))_magica_v3_O0_seed${base}.stderr.log.gz" \
                  "$dir/ai$((1-seat))_champion_seed${base}.stderr.log" \
                  "$dir/ai$((1-seat))_magica_v3_O0_seed${base}.stderr.log"; do
            [ -f "$cf" ] && st_size="$(wc -c < "$cf" | tr -d ' ')"
        done
        wall="$(grep -o "^\[$tok\] rc=[-0-9]* *[0-9.]* min" "$LOGDIR/_run.log" 2>/dev/null \
                | tail -1 | sed 's/.* //;s/ min//')"
        printf '  %-4s %-4s %-9s %-6s %-11s %-6s %-7s %-6s %-6s %-9s %-8s %-8s %-6s %-6s\n' \
            "$tok" "P$seat" "${verdict:-无}" "${rounds:-?}" "${hp_ours:-?},${hp_opp:-?}" \
            "$( [ -n "${hp_ours:-}" ] && [ -n "${hp_opp:-}" ] && echo $((hp_ours - hp_opp)) || echo '?')" \
            "${ill:-?}" "${our_ill:-?}" "${wall:-?}m" "${st_size}" "${inj_on}" "${inj_asks}" \
            "${inj_to}" "${inj_dead}"
        # ── 断言 ──
        [ "${verdict:-}" = "engine" ] || { echo "    !! $tok verdict=${verdict:-无}（不是 engine）"; fails=$((fails+1)); }
        [ "${ill:-x}" = "0" ] || { echo "    !! $tok 裁判侧 illegal=${ill:-?}（必须 0）"; fails=$((fails+1)); }
        [ "${our_ill:-x}" = "0" ] || { echo "    !! $tok 我们自己的 illegal_total=${our_ill:-?}（必须 0）"; fails=$((fails+1)); }
        if [ "$EXP" != "E1" ]; then
            [ "${inj_on:-0}" -ge 1 ] 2>/dev/null || { echo "    !! $tok 没有『注入已启用』⇒ 这一局其实是 E1，无效"; fails=$((fails+1)); }
            [ "${inj_asks:-0}" -gt 0 ] 2>/dev/null || { echo "    !! $tok 注入 asks=${inj_asks:-?} ⇒ 一次都没问成，无效"; fails=$((fails+1)); }
            # 会话中途超时 = 从那一刻起退化成 E1（不算硬失败，但**必须**在报告里点名）
            [ "${inj_to:-0}" = "0" ] 2>/dev/null || echo "    ⚠️ $tok 注入会话 timeout=${inj_to} 次 ⇒ 那一局后半段已退化成 E1（读数字段要打折）"
            [ "${inj_dead:-0}" = "0" ] 2>/dev/null || echo "    ⚠️ $tok 注入会话 dead=${inj_dead}（读端错位/进程死）⇒ 同上"
        fi
        if [ "$OPP" = "champion" ]; then
            [ "${st_size:-0}" != "0" ] && [ "${st_size}" != "-" ] || { echo "    !! $tok 冠军侧 stderr 缺失/为空"; fails=$((fails+1)); }
        fi
    done
    echo ""
    echo "-------- [$TAG] RESULT 行 --------"
    for tok in $SEEDS; do
        grep -h "RESULT seed" "$LOGDIR/$tok.log" 2>/dev/null | sed "s/^/  [$tok] /"
    done
    echo ""
    if [ "$fails" -eq 0 ]; then
        echo "[uvx] ✅ [$TAG] 三条硬断言全过：verdict=engine / illegal=0（裁判侧+AI 侧）/ 冠军侧 stderr 非空"
    else
        echo "[uvx] ❌ [$TAG] 有 $fails 条断言失败（见上面的 !!）"
    fi
    return "$fails"
}

# ── 起跑 ─────────────────────────────────────────────────────────────────────
export_env="$BASE$EXTRA"
mkdir -p "$LOGDIR"
# ⚠️ 先把继承来的 AZAI_* 全部清掉，再 export 本臂的 —— 否则臂之间会**互相继承开关**
#    （两臂变成同一臂，整批读数作废）。外部 AI 忽略 AZAI_*，所以给对手带上也无害。
for v in $(env | sed -n 's/^\(AZAI_[A-Za-z_0-9]*\)=.*/\1/p'); do unset "$v"; done
for kv in $(echo "$export_env" | tr ';' '\n'); do
    [ -n "$kv" ] && export "$kv"
done

# 每个实验的开关前置断言（少了这些断言，一类"静默失效"就会伪装成"没效果"）
case "$EXP" in
    E1)
        [ -z "${AZAI_ORACLE_INJECT:-}" ] || { echo "[uvx] !! E1 竟然带着 AZAI_ORACLE_INJECT" >&2; exit 4; }
        [ -z "${AZAI_ORACLE_PLACEBO:-}" ] || { echo "[uvx] !! E1 竟然带着 AZAI_ORACLE_PLACEBO" >&2; exit 4; }
        [ -z "${AZAI_ONLY_EXTRA:-}" ] || { echo "[uvx] !! E1 竟然带着 AZAI_ONLY_EXTRA" >&2; exit 4; }
        [ -z "${AZAI_ONLY_EXTRA_RANDOM:-}" ] || { echo "[uvx] !! E1 竟然带着 AZAI_ONLY_EXTRA_RANDOM" >&2; exit 4; }
        echo "[uvx] ✓ E1 纯基线自检：没有任何注入 / only-extra 开关"
        ;;
    E2)
        [ "${AZAI_ORACLE_INJECT:-}" = "$CHAMP" ] || { echo "[uvx] !! E2 的 AZAI_ORACLE_INJECT 不等于冠军 exe" >&2; exit 4; }
        [ "${AZAI_ORACLE_TIMEOUT:-}" = "20" ] || { echo "[uvx] !! E2 的 AZAI_ORACLE_TIMEOUT != 20" >&2; exit 4; }
        [ -z "${AZAI_ONLY_EXTRA:-}" ] || { echo "[uvx] !! E2 竟然带着 AZAI_ONLY_EXTRA" >&2; exit 4; }
        echo "[uvx] ✓ E2 自检：注入=冠军 exe，timeout=20s"
        ;;
    E4|E4R)
        [ "${AZAI_ONLY_EXTRA:-}" = "1" ] || { echo "[uvx] !! $EXP 的 AZAI_ONLY_EXTRA != 1" >&2; exit 4; }
        [ "${AZAI_ORACLE_INJECT:-}" = "$CHAMP" ] || { echo "[uvx] !! $EXP 的 A 会话不是冠军" >&2; exit 4; }
        [ "${AZAI_ORACLE_INJECT_B:-}" = "$RU" ] || { echo "[uvx] !! $EXP 的 B 会话不是亚军" >&2; exit 4; }
        if [ "$EXP" = "E4R" ]; then
            [ "${AZAI_ONLY_EXTRA_RANDOM:-}" = "1" ] || { echo "[uvx] !! E4R 的 AZAI_ONLY_EXTRA_RANDOM != 1" >&2; exit 4; }
        else
            [ -z "${AZAI_ONLY_EXTRA_RANDOM:-}" ] || { echo "[uvx] !! E4 竟然带着 AZAI_ONLY_EXTRA_RANDOM" >&2; exit 4; }
        fi
        echo "[uvx] ✓ $EXP 自检：候选集=只保留额外候选（A=冠军 B=亚军），random=${AZAI_ONLY_EXTRA_RANDOM:-0}"
        ;;
esac
echo "[uvx] 环境自检 ✓  AZAI_ORACLE_INJECT=${AZAI_ORACLE_INJECT:-<空>}  AZAI_ONLY_EXTRA=${AZAI_ONLY_EXTRA:-<未设>}  AZAI_FLAT_SEED=$AZAI_FLAT_SEED  AZAI_K=$AZAI_K"

stash_stderr "${TAG}_$(date '+%Y%m%d_%H%M%S')"

attempt=0
while [ "$attempt" -le "$RETRY" ]; do
    todo="$(remaining_tokens "$LOGDIR")"
    if [ -z "${todo// /}" ]; then
        echo "[uvx] [$TAG] 没有待跑的局（全部 verdict=engine）"
        break
    fi
    echo ""
    echo "[uvx] [$TAG] 第 $((attempt+1)) 轮：待跑 = $todo  $(date '+%H:%M:%S')"
    for tok in $todo; do
        [ -f "$LOGDIR/$tok.log" ] || continue
        mkdir -p "$LOGDIR/_attempts"
        cp -f "$LOGDIR/$tok.log" "$LOGDIR/_attempts/$tok.attempt$attempt.log" 2>/dev/null
    done
    t0=$(date +%s)
    "$P" -u _tmp_ladder.py --tag="$TAG" --jobs="$JOBS" \
        --ai0=code/test_match/az_bridge_ai.py --ai1="$OPP_EXE" \
        ${ROUNDS:+--rounds="$ROUNDS"} $todo 2>&1 | tee -a "$LOGDIR/_run.log"
    rc=${PIPESTATUS[0]}
    t1=$(date +%s)
    echo "[uvx] [$TAG] 第 $((attempt+1)) 轮结束 rc=$rc  $(( (t1-t0)/60 )) min  $(date '+%H:%M:%S')"
    attempt=$((attempt+1))
    [ "$rc" -ne 0 ] && echo "[uvx] [$TAG] ⚠️ ladder rc=$rc（继续按 resume 逻辑补跑）"
done

archive_stderr
finish
RC_FIN=$?

echo ""
echo "=============================================================="
echo "[uvx] 结束 $(date '+%Y-%m-%d %H:%M:%S')  TAG=$TAG  断言失败数=$RC_FIN"
echo "[uvx] 读数："
echo "  $P -u code/test_match/analyze_inject_vs_runnerup.py \\"
echo "      --arm $EXP=$LOGDIR \\"
echo "      --stderr $EXP=$OUTROOT/$TAG/stderr \\"
echo "      --json $OUTROOT/$TAG/summary.json"
echo "=============================================================="
exit "$RC_FIN"
