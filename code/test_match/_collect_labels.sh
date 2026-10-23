#!/usr/bin/env bash
# 批量采集位置先验标签（2026-10-23 夜，S4 数据基础）
#
# 第 1/2 轮已证明：官方 SDK 蒸馏的位置图**在完整对局里不能用**
#   · 位置图+类头都换：k=24 → 12.5%（-11.62hp）
#   · **只换位置图**：k=24 → 12.5%（-11.53hp）⇒ 与类头无关，位置图本身就是坏的
#   ⇒ 转向用**我们自己的搜索**（枚举+值网排序，打 k=1 是 100% 胜）当教师重训位置头。
#
# 本脚本：8 个进程并行，每个跑 5 局 ⇒ 40 局；每局约 9 min（单进程）⇒ 约 50-60 min。
set -uo pipefail
cd "$(dirname "$0")/../.." || exit 1
PY=D:/anaconda3/envs/pytorch-gpu/python.exe
mkdir -p training_history/prior_labels
echo "############ 标签采集：40 局（8 并行 × 5 局）  $(date +%H:%M:%S)"
for w in 0 1 2 3 4 5 6 7; do
  (
    for j in 0 1 2 3 4; do
      seed=$((9100 + w * 10 + j))
      $PY -u code/test_match/collect_prior_labels.py "$seed" 2>&1 | tail -2
    done
  ) &
done
wait
echo ""
echo "采集完成：$(ls -1 training_history/prior_labels/*.npz 2>/dev/null | wc -l) 个文件  $(date +%H:%M:%S)"
du -sh training_history/prior_labels 2>/dev/null || true
