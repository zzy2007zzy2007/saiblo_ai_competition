#!/usr/bin/env bash
# E4-rand：**同一候选集**（冠军动作 + 亚军动作），但**随机挑一个**（2026-10-28；先备脚本不跑）
#
# 与 E4 的唯一差别 = `AZAI_ONLY_EXTRA_RANDOM=1`（隐含 `AZAI_ONLY_EXTRA=1`）：
#   选层**不看评估器分数**，用 `sha256("onlyextra|<seed>|<dec_n>")` 的独立随机流挑一个
#   ⇒ 同一 (seed, 决策序号) 恒得同一个候选，**完全可复现**（`AZAI_ONLY_EXTRA_SEED` 可换种子）。
#
# 它和 E4 组成一对**对照臂**，用来把"两个外部动作本身有多好"和"评估器从两个里挑对贡献多少"分开：
#   E4   = 评估器裁决（上限的一侧）
#   E4R  = 随机裁决（下界：候选集的平均质量）
#
# 对手用 `OPP=` 选（两个都要跑，各 16 局）：
#   bash code/run_logged.sh only_extra_rand_vs_champion bash code/test_match/_e4r_only_extra_rand.sh
#   OPP=runnerup bash code/test_match/_e4r_only_extra_rand.sh
set -uo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO" || exit 1
export EXP=E4R
export OPP="${OPP:-champion}"
# 2026-10-29：与 `_e4_only_extra.sh` **逐字相同**的两条 oracle 加固 + 过夜三件套
#   （必须一致，否则 E4 与 E4-rand 的"唯一差别 = 挑法"这句话就不成立）。
export AZAI_ORACLE_REBUILD_MAX="${AZAI_ORACLE_REBUILD_MAX:-6}"
export AZAI_ORACLE_RESTART_EVERY="${AZAI_ORACLE_RESTART_EVERY:-150}"
export AZAI_ORACLE_RESTART_EVERY_B="${AZAI_ORACLE_RESTART_EVERY_B:-0}"
export LADDER_TIMEOUT="${LADDER_TIMEOUT:-21600}"
export LADDER_EXTRA_ARGS="${LADDER_EXTRA_ARGS:---retry-invalid 2 --final-sweep --watchdog-stall-min 90}"
export MVS_TIMEOUT_SECONDS="${MVS_TIMEOUT_SECONDS:-300}"
exec bash code/test_match/_us_vs_external.sh
