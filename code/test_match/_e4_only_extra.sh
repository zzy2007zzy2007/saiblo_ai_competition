#!/usr/bin/env bash
# E4：候选集**只保留**「冠军动作 + 亚军动作」两个，由评估器裁决（2026-10-28；**先备脚本不跑**）
#
# 相对 E1 只多四个环境变量（其余逐字相同，见 `_us_vs_external.sh` 的 `E4_ENV`）：
#   AZAI_ONLY_EXTRA=1                   候选集 = 只有额外塞进去的 bundle（网络 K 条不进候选）
#   AZAI_ORACLE_INJECT=<冠军>            A 会话 = 冠军
#   AZAI_ORACLE_TIMEOUT=20              冠军慢（O0）
#   AZAI_ORACLE_INJECT_B=<亚军> + AZAI_ORACLE_TIMEOUT_B=20   B 会话 = 亚军
#
# 对手用 `OPP=` 选（**两个都要跑**，各 16 局）：
#   OPP=champion  → bash code/test_match/_e4_only_extra.sh
#   OPP=runnerup  → OPP=runnerup bash code/test_match/_e4_only_extra.sh
#
# 读数注意（报告 §5.3 会说清）：候选集只有 1~2 个 ⇒ A 臂统计里的 `hit_sel`（"额外候选被搜索
# 选中"）在这里**按构造恒为 1**，不能当"注入命中率"读；要看 `[onlyextra]` 的 `ncand` 直方图。
#
# 另外会**白拿**一组研究数据（2026-10-28 用户要求）：两个会话同时在跑 ⇒ 每回合一行
# `[xor] round=.. champ=.. ru=.. agree=.. dup_c=.. dup_r=.. chosen=.. hit_c=.. hit_r=..`
# （`docs/us_vs_champion_20261028.md` §6）。归档后这样读：
#   D:/anaconda3/envs/pytorch-gpu/python.exe -u code/test_match/analyze_xoracle.py \
#       --tag ONLYEXTRA_VS_<对手> --stderr match_results/us_vs_external/ONLYEXTRA_VS_<对手>/stderr
#
# 用法：
#   bash code/run_logged.sh only_extra_vs_champion bash code/test_match/_e4_only_extra.sh
set -uo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO" || exit 1
export EXP=E4
export OPP="${OPP:-champion}"
exec bash code/test_match/_us_vs_external.sh
