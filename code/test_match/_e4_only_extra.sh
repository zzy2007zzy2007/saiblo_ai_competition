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
# ── 2026-10-29：E4 的两条 oracle 加固（`docs/e4_only_extra_20261029.md` §1）──────────────
#   A 会话（冠军）：**主动重启**（默认每 150 回合）。为什么必须开：
#     它的影子局面从第 1 个"我们不执行的决策"起就分叉 ⇒ `force_pass_` 一置上就整局 HOLD、
#     再往后影子里的基地血归零 ⇒ abort（`apply_operations.called_after_terminal`）。
#     不重启 ⇒ 第 ~239 回合起 A 就永久消失 ⇒ 候选集塌成 1 个 ⇒ **E4 与 E4-rand 按构造相同**。
#   B 会话（亚军）：**不重启**（它是无状态查询器，重启只会把回合号搞乱、引入偏差）。
export AZAI_ORACLE_REBUILD_MAX="${AZAI_ORACLE_REBUILD_MAX:-6}"
export AZAI_ORACLE_RESTART_EVERY="${AZAI_ORACLE_RESTART_EVERY:-150}"
export AZAI_ORACLE_RESTART_EVERY_B="${AZAI_ORACLE_RESTART_EVERY_B:-0}"
# ── 2026-10-29：过夜批次的三件套（`docs/ladder_resilience_20261029.md`）───────────────
#   ⚠️ 阈值取 90 分钟（不是 1）：1 分钟是验证开关用的，会误杀"单回合很慢"的好局。
#   ⚠️ 6 小时墙钟（不是 3）：夜里休眠时墙钟会跳，2-3 小时会把好局误杀；靠 retry/sweep 补跑。
export LADDER_TIMEOUT="${LADDER_TIMEOUT:-21600}"
export LADDER_EXTRA_ARGS="${LADDER_EXTRA_ARGS:---retry-invalid 2 --final-sweep --watchdog-stall-min 90}"
# 单次读包超时 = 300s（任务书口径）：300s 只当"真死锁"门限；冠军在 force_pass 状态下单次 ~80ms，
# 不存在"慢查询被误杀"的风险。
export MVS_TIMEOUT_SECONDS="${MVS_TIMEOUT_SECONDS:-300}"
exec bash code/test_match/_us_vs_external.sh

