#!/usr/bin/env bash
# 公共环境 + 路径常量。被 `_tmp_lr_v*.sh` source。
#
# ⚠️ 为什么验证批次用 **rule_v4(Python) vs 空操作假 AI**（2026-10-29 现场调整，实测 40s/局）：
#   机器上正有一批真实的 16 局（`US_VS_CHAMPION_BASE`, jobs=5）+ 一个 40 worker 的 coevo 在跑，
#   驱动层验证只需要"跑真对局、拿到 verdict=engine、别把机器占满"。试过两种更慢的：
#     · `_illegal_smoke_fast.sh` 的 FAST bridge 档 ⇒ 双方每回合都空过，512 回合**僵局**
#       ⇒ `verdict=hp_tiebreak`（不是 engine，正是"僵局伪装"那条坑），单局 ~34 min；
#     · 亚军 C++ ×2 ⇒ 每回合 ~3s、单局 ~25 min。
#   最终选 rule_v4 vs `_tmp_noopai.py`：noop 不防守 ⇒ rule_v4 在 ~213 回合推平基地 ⇒
#   **引擎真的判终局**（`verdict=engine`）、整局 ~40s、纯 CPU 不占 GPU。对"驱动层"的验证效力不变
#   （驱动只认日志里的 RESULT/verdict）。
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")" || exit 1

export PYTHONIOENCODING=utf-8
PY=D:/anaconda3/envs/pytorch-gpu/python.exe
RV4=code/test_match/rv4_pkg/main.py                     # 我们的 rule_v4（Python 规则 AI）
NOOP=_tmp_noopai.py                                     # 空操作假 AI（必输，几十~两百回合被推平）
RU="其他版本ai/saiblo-30th-AI/Game1/antgame_ai_cpp/cpp_lure_v4/build/ai_cpp_lure_v4.exe"
ROOT_W="$(pwd -W | sed 's#/$##')"                       # git-bash 的 /d/... 不能喂给 Windows Python

# 单次读包卡死检测（bridge 侧兜底）。看门狗验证要在 ~1 min 内先动手，所以放宽到 15 min。
export MVS_TIMEOUT_SECONDS=900

for f in "$PY" "$RV4" "$NOOP"; do
    [ -f "$f" ] || { echo "[lr] !! 缺文件: $f" >&2; exit 2; }
done
echo "[lr] PY=$PY"
echo "[lr] 对局 = rule_v4 vs 空操作假 AI: $RV4  ×  $NOOP（单局 ~40s、真终局 verdict=engine）"
echo "[lr] ROOT_W=$ROOT_W"
echo "[lr] 开始 $(date '+%Y-%m-%d %H:%M:%S')"
