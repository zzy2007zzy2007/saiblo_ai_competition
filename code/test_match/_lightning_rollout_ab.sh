#!/usr/bin/env bash
# unified(+闪电随机 rollout) vs pin(正常 random rollout) A/B（2026-11-02）。
# 16 局 = seeds 100..107 × 2 座位（镜像配对）。
#
# 两臂唯一差别 = 候选生成（A=unified / B=pin）+ 随机动作的"侧+类"：
#   · A 臂 = `AZAI_CAND_MODE=unified` + `AZAI_FLAT_ROLLOUT_MODE=lightning`
#     （lightning 语义：rollout 的 16 回合里**我方**以 decay(1,3) 概率随机放闪电，
#       落点 = 我方合法高地格随机，cd==0 且金币≥90 才放，否则空过；对手不动。见
#       `code/my_ai/az_intent/flat_rollout_greedy.py:_decide_actions`）；
#   · B 臂 = `AZAI_CAND_MODE=pin` + BASE 自带的 `AZAI_FLAT_ROLLOUT_MODE=random`
#     （对手随机建塔/降级，现役行为）。
#   其余 flat rollout 参数 = 部署 BASE（`_us_vs_external.sh:68` **逐字相同**，脚本里有断言）：
#   k=48/rollouts=64/horizon=16/batch=32/prob=decay(1,3)/rounds=49152/seed=53/
#   share_prefix=1/eval=net/ckpt=posnet_A_k5_m32.pt。
#
# 驱动 = `ab_search_modes.py`（它用 `match_via_sdk.py` 把 ai0/ai1 都指 az_bridge_ai.py，
# 各自带 `--ai0-env` / `--ai1-env`）：
#   · A 臂（unified+lightning）= `--flat-env`（ab_search_modes 里叫 "flat"，本批 A 臂坐该槽）；
#   · B 臂（pin）               = `--mcts-env`。
#   · `--paired-sides`：每个 seed 打两局，A 侧各坐 P0/P1 ⇒ 抵消先手偏。
#
# 用法（长任务用 WMI 起 detached，见本文件末尾注释 / docs/lightning_rollout_mode_20261102.md）：
#   bash code/run_logged.sh lightning_rollout_ab bash code/test_match/_lightning_rollout_ab.sh
set -uo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO" || exit 1
export PYTHONIOENCODING=utf-8
P="${PYTHON:-D:/anaconda3/envs/pytorch-gpu/python.exe}"
JOBS="${JOBS:-8}"
SEEDS="${SEEDS:-100 101 102 103 104 105 106 107}"
RETRY="${RETRY:-2}"
GAME_TIMEOUT_S="${GAME_TIMEOUT_S:-10800}"
OUTDIR="match_results/lightning_rollout_ab"

# BASE：与 `_us_vs_external.sh:68` 逐字相同（历史批次可比性的锚点）
BASE="$(sed -n 's/^BASE="\(.*\)"$/\1/p' code/test_match/_us_vs_external.sh)"
[ -n "$BASE" ] || { echo "[lightning_ab] !! 读不到 _us_vs_external.sh 的 BASE" >&2; exit 2; }
case "$BASE" in
    *AZAI_CAND_MODE*) echo "[lightning_ab] !! BASE 里已含 AZAI_CAND_MODE ⇒ 停" >&2; exit 2 ;;
esac
case "$BASE" in
    *AZAI_FLAT_ROLLOUT_MODE=random*) ;;
    *) echo "[lightning_ab] !! BASE 里没有 AZAI_FLAT_ROLLOUT_MODE=random ⇒ 停" >&2; exit 2 ;;
esac

# 两臂只差候选生成 + 随机动作侧/类；其余（含 AZAI_CAND_VERBOSE=1 只为日志）逐字相同。
#   ⚠️ A 臂的 `AZAI_FLAT_ROLLOUT_MODE=lightning` 必须**排在 BASE 之后**（`_parse_env` 按 `;`
#     拆成 dict，后面的键覆盖前面的 ⇒ lightning 覆盖 BASE 里的 random）。
FLAT_ENV="$BASE;AZAI_CAND_MODE=unified;AZAI_FLAT_ROLLOUT_MODE=lightning;AZAI_CAND_VERBOSE=1"
MCTS_ENV="$BASE;AZAI_CAND_MODE=pin;AZAI_CAND_VERBOSE=1"

mkdir -p "$OUTDIR"
echo "=============================================================="
echo "[lightning_ab] A(unified+lightning) vs B(pin+random)   16 局 = $(echo $SEEDS | wc -w) seeds × 2 座位"
echo "[lightning_ab] seeds=[$SEEDS]  jobs=$JOBS  retry=$RETRY  game_timeout=${GAME_TIMEOUT_S}s"
echo "[lightning_ab] BASE 与 _us_vs_external.sh:68 逐字相同（${#BASE} 字符）"
echo "[lightning_ab] ckpt=checkpoints/posnet_A_k5_m32.pt  flat: rollouts=64/horizon=16/batch=32"
echo "[lightning_ab] A rollout_mode=lightning（我方随机放闪电）；B rollout_mode=random（对手随机建塔/降级）"
echo "[lightning_ab] $(date '+%F %T')"
echo "=============================================================="

# MVS_STDERR_SUFFIX=1：同 seed 两局（P0/P1 并发）的 stderr 必须带 _p0/_p1 后缀，否则互相覆盖
#   （match_via_sdk.py 只在该开关=1 时把 DSH_PROGRESS_SUFFIX 拼进 stderr 文件名）。
# AB_SEARCH_SUFFIX_TAG=_lightning：与并发批次（如 shape_pin）的 stderr 区分开，避免互相覆盖。
export MVS_STDERR_SUFFIX=1
export AB_SEARCH_SUFFIX_TAG="_lightning"

"$P" -u code/test_match/ab_search_modes.py \
    --seeds $SEEDS --jobs "$JOBS" --paired-sides \
    --out-dir "$OUTDIR" \
    --flat-env "$FLAT_ENV" --mcts-env "$MCTS_ENV" \
    --resume --retry-invalid "$RETRY" --game-timeout-s "$GAME_TIMEOUT_S" \
    2>&1 | tee "$OUTDIR/_run.log"
rc=${PIPESTATUS[0]}
echo "[lightning_ab] ab_search_modes 结束 rc=$rc  $(date '+%F %T')"
exit "$rc"
