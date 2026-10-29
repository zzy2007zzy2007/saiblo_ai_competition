#!/usr/bin/env bash
# E2：我们 **+ 注入冠军动作** vs 冠军（2026-10-28；**先把脚本备好，正式跑由用户点**）
#
# 唯一变量 = `AZAI_ORACLE_INJECT`（A 会话 = 冠军本体）。其余与 E1（`_us_vs_external.sh` 的 EXP=E1）
# **逐字相同**：BASE 配置、seeds 7..14 × 2 座位、对手 = 冠军、jobs=4。
#
# ⚠️ 三条已踩过的坑（`docs/us_vs_champion_20261028.md` §2）：
#   1. `AZAI_ORACLE_INJECT` 必须是 **D:/ 形态的 Windows 绝对路径**：`/d/...` 会让
#      `Path.is_absolute()` 判 False ⇒ 拼成 `<repo>/d/...` ⇒ Popen FileNotFoundError
#      ⇒ **注入整局静默失效，而对局照跑**（驱动已断言路径形态 + stderr 里有"注入已启用"）。
#   2. `AZAI_ORACLE_TIMEOUT=20`（不是 15）：冠军是 O0 编译、单次查询明显比亚军慢；
#      超时一次 = 整个会话永久停用（读端错位）⇒ 后半局退化成 E1，主读数被稀释。
#   3. 冠军 exe 靠 **PATH** 找 mingw DLL（它目录里没有 DLL）；`match_via_sdk.start_ai` 已给
#      magica 单独前置一个可用的 DLL 目录 ⇒ 不要绕过 `start_ai` 自己起进程。
#
# 用法：
#   bash code/run_logged.sh inject_champion_vs_champion bash code/test_match/_e2_inject_vs_champion.sh
#   续跑：原样再敲（已 verdict=engine 的局会跳过）
#
# 逐回合记录：E2 只有一个会话（A=冠军）⇒ 每回合一行 `[xor] round=.. champ=.. ru=NA ..`
# （格式见 `docs/us_vs_champion_20261028.md` §6）。归档后：
#   D:/anaconda3/envs/pytorch-gpu/python.exe -u code/test_match/analyze_xoracle.py \
#       --tag US_VS_CHAMPION_INJ --stderr match_results/us_vs_external/US_VS_CHAMPION_INJ/stderr
set -uo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO" || exit 1
export EXP=E2
export OPP=champion
exec bash code/test_match/_us_vs_external.sh
