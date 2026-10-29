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
exec bash code/test_match/_us_vs_external.sh
