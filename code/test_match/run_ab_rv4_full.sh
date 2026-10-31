#!/usr/bin/env bash
# 一键跑「4 臂 × rule_v4」+「实验 B：steps_per_shape」队列（**可 detached 启动**）。
#
# 为什么单独一个启动器：`_value_ab_vs_rv4.sh` 靠环境变量配置；而要用
# `Invoke-CimMethod Win32_Process Create`（不继承父进程环境、也不随 agent turn 结束而死）
# 起进程时，环境必须在脚本里写死。
#
# 配置（2026-11-01 用户定）：
#   · 搜索 = 统一候选 + flat 阶段 2，其余 = 部署 BASE（`_us_vs_external.sh:68`）+ `AZAI_CAND_MODE=unified`
#   · 6 个配置：incumbent / kgeo50 / abs50 / relk50（4 个值网臂）
#               + steps24 / steps48（值网=现役，只改 `AZAI_CAND_STEPS`）
#   · seeds 100..103 × 2 座位 = **每配置 8 局**，共 48 局（想扩到 32 局/配置就改 SEEDS）
#   · JOBS=4（每局 3 个 OS 进程 + 采集 12 工人 ⇒ 约 16 核满载）
#
# 启动（从任何地方，detached）：
#   bash code/test_match/run_ab_rv4_full.sh
# 日志：match_results/value_ab_rv4/AB_RV4_FULL/_run.log（驱动自身）+ _times.tsv（每局墙钟）
set -uo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO" || exit 1
export PYTHONIOENCODING=utf-8
export JOBS="${JOBS:-4}"
export SEEDS="${SEEDS:-100 100r 101 101r 102 102r 103 103r}"
export TAG="${TAG:-AB_RV4_FULL}"
export ARM_NAMES="${ARM_NAMES:-incumbent kgeo50 abs50 relk50 steps24 steps48}"
mkdir -p "match_results/value_ab_rv4/$TAG"
# 加一个时间戳 banner，便于判断"这次是不是真的起来了"
{
  echo "===== [launcher] $(date '+%Y-%m-%d %H:%M:%S') 启动"
  echo "===== [launcher] JOBS=$JOBS SEEDS=[$SEEDS] ARMS=[$ARM_NAMES]"
} >> "match_results/value_ab_rv4/$TAG/_run.log"
exec bash code/test_match/_value_ab_vs_rv4.sh
