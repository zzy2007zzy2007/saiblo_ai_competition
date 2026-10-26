#!/usr/bin/env bash
# 批量：16 局亚军对局的"先验召回"统计（每局约 2-4 分钟，串行约 40-60 分钟）
set -uo pipefail
cd "$(dirname "$0")/../.." || exit 1
export AZAI_TORCH_THREADS=2
PY=D:/anaconda3/envs/pytorch-gpu/python.exe
echo "############ 先验对亚军走法的召回（16 局）  $(date +%H:%M:%S)"
$PY -u code/test_match/recall_on_runnerup_v3.py match_results/ladder_logs/VS_RUNNERUP_cells4/10.log match_results/ladder_logs/VS_RUNNERUP_cells4/10r.log match_results/ladder_logs/VS_RUNNERUP_cells4/11.log match_results/ladder_logs/VS_RUNNERUP_cells4/11r.log match_results/ladder_logs/VS_RUNNERUP_cells4/12.log match_results/ladder_logs/VS_RUNNERUP_cells4/12r.log match_results/ladder_logs/VS_RUNNERUP_cells4/13.log match_results/ladder_logs/VS_RUNNERUP_cells4/13r.log match_results/ladder_logs/VS_RUNNERUP_cells4/14.log match_results/ladder_logs/VS_RUNNERUP_cells4/14r.log match_results/ladder_logs/VS_RUNNERUP_cells4/15.log match_results/ladder_logs/VS_RUNNERUP_cells4/15r.log match_results/ladder_logs/VS_RUNNERUP_cells4/16.log match_results/ladder_logs/VS_RUNNERUP_cells4/16r.log match_results/ladder_logs/VS_RUNNERUP_cells4/17.log match_results/ladder_logs/VS_RUNNERUP_cells4/17r.log
echo "结束 $(date +%H:%M:%S)"