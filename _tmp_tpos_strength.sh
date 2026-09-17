#!/usr/bin/env bash
# t_pos 扫描的强度臂：同一对局台，只换位置采样温度（两侧共用 ⇒ 镜像对称）。
# 目的：看"把探索量用合法手段补回来"能不能把 search-vs-raw 从 1.3750 拉回 1.8125。
# 标定依据（_tmp_prior_quality.py --t-pos-list）：t_pos=0.3 -> 有效候选 10；
# 1.0 -> 173；4.0 -> 263（≈ 修复前那个 100x 尺度错配等价的全探索）。
set -u
cd "$(dirname "$0")"
PY=D:/anaconda3/envs/pytorch-gpu/python.exe
export PYTHONIOENCODING=utf-8
CK=training_history/az_fixed/three_mix_r10p_vw_pol_frozen.pt
for TP in 1.0 2.0 4.0; do
  echo "########## t_pos=$TP ##########"
  "$PY" -u _tmp_position_choice_ab.py --checkpoint "$CK" --mode search \
      --pairs 16 --workers 8 --seed 0 --t-pos "$TP"
done
