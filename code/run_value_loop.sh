#!/usr/bin/env bash
# 价值头迭代训练循环（见 docs/az_iterative_value_loop_plan.md）
#
# 每轮：
#   1) 用「az_r10 策略 + 当前价值头」的 mix 做**搜索采集**（256 迭代 / depth 2 +
#      --skip-hold-search，实测 ≈2.2 分钟/局），**直接写轻量 npz**（无 pkl/无转换）
#   2) 把本轮 npz 按子目录合并进**累积池**（每个子目录一个 merged_*.npz）
#   3) 价值头 **warm-start**（--keep-value）在累积池上流式续训（--stream）
#   4) 记录 search-vs-raw（配对、灵敏）；每 VS_RULE_EVERY 轮做一次 64 局 vs rule_v4
#      —— 后者是**绝对水平**的唯一可信读数（成功标准：>50%）
#
# 用法：
#   bash code/run_value_loop.sh [rounds=8] [games/round=300] [iters=256] [depth=2] \
#                               [epochs/round=3] [work_dir=training_history/az_loop]
set -u
REPO="D:/2026智能体大赛_新2"; PY="D:/anaconda3/envs/pytorch-gpu/python.exe"
cd "$REPO" || exit 1

ROUNDS="${1:-8}"
GAMES="${2:-300}"
ITERS="${3:-256}"
DEPTH="${4:-2}"
EPOCHS="${5:-3}"
WORK="${6:-training_history/az_loop}"
SUBGAMES="${SUBGAMES:-50}"          # 每个采集子目录的局数 => 合并块大小（50 局 ≈1.6GB 内存）
VS_RULE_EVERY="${VS_RULE_EVERY:-3}" # 每 N 轮做一次 vs rule_v4
SEED_SRC="${SEED_SRC:-training_history/az_fixed/warm_polonly}"  # 初始池子的来源

POOL="$WORK/pool"
POLICY=training_history/az_fixed/az_r10.pt
VW=training_history/az_fixed/vw_pol_frozen.pt      # 起始价值头（29.7% 基线那个）
mkdir -p "$POOL" "$WORK"

echo "[loop] rounds=$ROUNDS games/round=$GAMES iters=$ITERS depth=$DEPTH epochs/round=$EPOCHS"
echo "[loop] work=$WORK pool(blocks)=${SUBGAMES}局  vs_rule_every=$VS_RULE_EVERY"

# ── 0) 用现有 336 局 polonly 给池子做种（只做一次）─────────────────────────
if [ ! -f "$WORK/.seeded" ]; then
  echo "########## SEED POOL：合并 $SEED_SRC ##########"
  "$PY" code/my_ai/az_intent/merge_az_batches.py \
      --src "$SEED_SRC" --out "$POOL" --layout flat --seed-div 10000000
  touch "$WORK/.seeded"
fi

for k in $(seq 1 "$ROUNDS"); do
  echo "########## ROUND $k / $ROUNDS ##########"
  ROUND_DIR="$WORK/round$k"
  rm -rf "$ROUND_DIR"; mkdir -p "$ROUND_DIR"

  # 1) 采集用的 mix = az_r10 策略 + 当前价值头
  "$PY" code/my_ai/az_intent/convert_no_bn_to_bn.py \
      --input "$VW" --output "$WORK/cur_bn.pt" > "$WORK/convert.log" 2>&1
  "$PY" code/my_ai/az_intent/make_mix_checkpoint.py \
      --policy "$POLICY" --value "$WORK/cur_bn.pt" --out "$WORK/mix_collect.pt" \
      > "$WORK/mix.log" 2>&1
  echo "[loop] collect model -> $WORK/mix_collect.pt"

  # 2) 采集（分若干子目录，使合并块 <= SUBGAMES 局）
  NSUB=$(( (GAMES + SUBGAMES - 1) / SUBGAMES ))
  for s in $(seq 1 "$NSUB"); do
    g=$(( GAMES / NSUB )); [ "$s" -eq "$NSUB" ] && g=$(( GAMES - (NSUB-1)*(GAMES/NSUB) ))
    echo "[loop]   collect sub$s/$NSUB: $g games (seed base $((900000 + k*100 + s)))"
    "$PY" code/my_ai/az_intent/az_selfplay.py \
        --checkpoint "$WORK/mix_collect.pt" --games "$g" --workers 16 \
        --iterations "$ITERS" --max-depth-rounds "$DEPTH" --max-rounds 512 \
        --out-dir "$ROUND_DIR/sub$s" --seed $((900000 + k*100 + s)) \
        --native-engine --skip-hold-search --write-npz \
        --t-class 0.5 --t-pos 0.3 --k 24
  done

  # 3) 本轮 -> 累积池（每子目录一个 merged_*.npz）
  "$PY" code/my_ai/az_intent/merge_az_batches.py \
      --src "$ROUND_DIR" --out "$POOL" --layout dirs --name-tag "r${k}_"

  # 4) warm-start 续训（流式）
  VW_NEW="$WORK/vw_round$k.pt"
  echo "[loop] train stream (warm-start from $VW) -> $VW_NEW"
  "$PY" code/my_ai/az_intent/value_warmup.py \
      --hotstart "$VW" --keep-value --stream \
      --data-dir "$POOL" --checkpoint "$VW_NEW" \
      --epochs "$EPOCHS" --skip-collect --freeze-backbone --device auto \
      --label-mode abs --label-weight kgeo --tau 50 --label-scale 1.5
  VW="$VW_NEW"

  # 5) 指标：search-vs-raw（配对、灵敏）
  echo "[loop] METRIC search-vs-raw (round $k)"
  "$PY" code/my_ai/az_intent/eval.py \
      --checkpoint "$WORK/mix_collect.pt" --self-raw-opponent --bundle-mcts --native-engine \
      --iterations "$ITERS" --max-depth-rounds "$DEPTH" \
      --t-class 0.5 --t-pos 0.3 --k 24 --games 32 --workers 16

  # 5b) 绝对水平：每 VS_RULE_EVERY 轮
  if [ $(( k % VS_RULE_EVERY )) -eq 0 ]; then
    echo "[loop] METRIC vs rule_v4 64 局（绝对水平，round $k）"
    "$PY" code/my_ai/az_intent/eval.py \
        --checkpoint "$WORK/mix_collect.pt" --opponent rule_v4 --bundle-mcts --native-engine \
        --iterations "$ITERS" --max-depth-rounds "$DEPTH" \
        --t-class 0.5 --t-pos 0.3 --k 24 --games 64 --workers 16
  fi
done
echo "############ 价值头迭代循环完成（$ROUNDS 轮）############"
