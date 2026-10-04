# 「做一轮循环」的可照抄配方（2026-10-05 只读调研，file:line 已核）

> 来源：无上下文 subagent 的只读调研（引用到的 file:line 我自己抽查过关键几处）。
> **强度：中**（代码/日志可复核；尚未实跑）。⚠️ 这是**执行配方**，不是结论；
> 任何用它做的实验**必须**先写预注册（含样本量）并走机械门。

## 0. 一个被纠正的前提（重要）

**`collect_value_prior.py` 没有搜索**：它不 import `bundle_mcts`，只用策略网前向
（`:87,:122-125`）+ `decode_network_output(temperature=0.0)`（`:195`）⇒ 走子 = **贪婪 1-ply**。
它的 argparse（`:215-235`）里**没有** `k/sample-mult/search-mode/skip1` —— 不是"默认值"，是**不存在**。
⇒ **位置头的训练数据是在"无搜索"轨迹上算出来的**，而部署是 `k24/sm15/pos-only` 搜索
⇒ 目标分布与部署时实际评估的落点**漂移**（钉 argmax 只对齐"哪一类"，没对齐"哪一格"）。
（官方计划也把"搜索版 collect_value_prior"列为**未实施的第二阶段**：`docs/az_posnet_v2_plan.md:384`。）

## 1. 数据：**两个目标、两个脚本**（一次采不回来）

| 目标 | 产出脚本 | 写在哪 | 关键字段 |
|---|---|---|---|
| **价值头**（终局 HP 差）| `az_selfplay.py` | `az_selfplay_seed*.pkl` | `v_p0=clip((hp0-hp1)/HP_SCALE,-1,1)`，逐帧 `value_target`（`az_selfplay.py:428-433`，pkl 键 `:392-405`，`:490` dump）|
| **位置头**（1-ply 价值先验）| `collect_value_prior.py` | `.npz` | `board,stats,player,head,cls,cell,adv,cnt,weight,*_meta`（`:292-301`）；`adv` = 每个候选格 post-state 过价值网取负（`:176-177`）|

- **顺序是强制的**：`adv` 由**价值网**产生 ⇒ 改了价值头必须**重采**位置数据，否则位置头在模仿旧价值头。
- ⚠️ 采集别加 `--write-npz`（建缓存只 glob `az_selfplay_seed*.pkl`，`train_value_net.py:72`）。
- ⚠️ `--skip-single-candidate` 会改变 `mcts.rng` 抽签 ⇒ 带/不带它的对局**不可复现**（`az_selfplay.py:582-585`），**不许混比**。

## 2. 训价值头（只换 `value_state`，class/pos 逐位不动 —— 有内建路径）

```
PY=D:/anaconda3/envs/pytorch-gpu/python.exe
# ① 采集（100 局 ≈17 min，400 局 ≈68 min；--workers 按机器纪律 ≤8）
$PY -u code/my_ai/az_intent/az_selfplay.py --checkpoint <当前三网 ckpt> --games 100 --workers 8 --seed 2 \
    --iterations 256 --max-depth-rounds 4 --t-class 0.5 --t-pos 1.0 --k 24 --sample-mult 15 \
    --search-mode pos-only --skip-single-candidate --native-engine --out-dir training_history/vprior/data_r2_100
# ② 建缓存（几分钟）
$PY -u code/my_ai/az_intent/train_value_net.py --ckpt <当前三网 ckpt> \
    --data training_history/vprior/data_r2_100 --cache training_history/vprior/vcache_r2_100 --build-cache-only
# ③ 只训价值头（≈2-3 min/4 epoch）
$PY -u code/my_ai/az_intent/train_value_net.py --ckpt <当前三网 ckpt> \
    --cache training_history/vprior/vcache_r2_100 --epochs 4 --lr 3e-4 --label-mode terminal \
    --freeze-bn --out training_history/vprior/posnet_r2_val.pt
```
- **`--ckpt` 必须传"当前三网 ckpt"**（不是 `three_mix_...`）：`load_three_models` 载入三网、
  class/pos `requires_grad=False` 冻住（`train_value_net.py:286-291`），`save_three_net` 只换 value
  （`:315-320/:346-352`）。**传错会静默退回旧 `pos_state`**（与当前 ckpt 有 14/94 张量不同）。
- `--label-mode terminal`：与现有价值头**同口径**（单变量）。
  `abs` 有"抄 `stats[1]`"捷径（见 `docs/next_step_survey_20261005.md`），
  `rel` 与搜索读法不匹配（要配 `AZAI_REL2ABS` = **协议变更**）。

## 3. 训位置头（在**新价值头**之上）

```
$PY -u code/my_ai/az_intent/collect_value_prior.py --checkpoint training_history/vprior/posnet_r2_val.pt \
    --games 400 --workers 8 --seed 910101 --out training_history/vprior/vp_r2_k5_m32.npz \
    --multi-class-k 5 --max-cells 32 --row-weight-lambda 0.2
$PY -u code/my_ai/az_intent/train_value_prior.py --ckpt training_history/vprior/posnet_r2_val.pt \
    --data training_history/vprior/vp_r2_k5_m32.npz --epochs 60 --batch-size 256 --lr 1e-3 --seed 0 \
    --target-tau 1.0 --t-pos 1.0 --out training_history/vprior/posnet_r2_full.pt
```
- 冻结生效：class/value `requires_grad=False` + `.eval()`，只 `pos_model.train()`（`:106-114`），
  loss 只前向 `pos_model`（`:137`）⇒ **class/value 只被原样写回**，所以 `--ckpt` 传带新价值头的那个。
- ⚠️ `--val-frac 0.1` 是**按行**切（`:102-104`）⇒ 同局的行会同时进 train/val，**val CE 偏乐观**
  （对比 `train_value_net` 是**按局**切）。

## 4. 拼接：**不需要额外脚本**

两个训练器**自己就写三个 state**（`train_value_net.py:315-320,346-352`、`train_value_prior.py:209-212`
→ `az_train.save_three_net`，`az_train.py:503-524`），键集合与现有 ckpt 一致（16 个顶层键，
**故意不写 `model_state`**）。只有"注入**外部**训的 value_state"才需要 ~10 行新脚本。

## 5. 成本（历史 run 实测）

| 步 | run | 用时 |
|---|---|---|
| `az_selfplay` 400 局（pos-only+skip1+k24）| `inject_ex02_400g`（`experiment_log.md:3045`）| **4090 s** |
| `collect_value_prior` 400 局 | `vprior_collect400`（`:2445`）| **3540 s** |
| `train_value_net` 3 epoch | `valnet_train_clean`（`:3108`）| **643 s / 两臂** |
| `train_value_prior` 60 epoch（K=5，351k 行）| `vp_train_three_arms`（`:3650`）| **6435 s / 三臂**（A 臂 ~93 min）|

⇒ 完整一轮 ≈ **4.3–5.4 h**；**最小一轮**（100 局 + 只训价值头）≈ **25 min**。

## 6. 最小一轮能回答什么 / 不能回答什么

- **能**：① 新策略数据是否让价值头在**留出局**上变好；② "只换 `value_state`"的管线是否干净。
- **不能**：棋力是否提升（100 局 ≈ 100 个独立结局；val MSE **不是**泛化读数，验证集只有 20 局）。

## 7. 坑（每条有出处的简表）

1. 价值网的 **BN 从未训过**（`num_batches_tracked` 全 0）⇒ 必须 `--freeze-bn`（默认开）；
   否则 `resblocks.5.bn1.running_var` 1→1.473e4（`experiment_log.md:3127`）。
2. 验证集 **20 局**、terminal 标签局内恒定 ⇒ 18,778 个 val 样本背后只有 20 个独立结局。
3. 采集（贪婪、无搜索）与部署（k24/sm15 搜索）**目标分布漂移**（§0）。
4. 顺序强制：改价值头 ⇒ 必须重采位置数据（§1）。
5. `rel` 要配 `AZAI_REL2ABS` + `--label-scale`，属**协议变更**，与现基线不可比。
6. 价值头是**裸线性**（无 tanh），搜索读时套 tanh（`AZAI_VALUE_TANH` 默认 1）⇒ 换标签口径时要小心幅度。
7. `--build-cache-only` 仍需传 `--ckpt`（required）。
8. 机器纪律：`--workers` 不要超过 8，且**不要与对战同时跑**。
