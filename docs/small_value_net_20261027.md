# 小值网（`width=32 / blocks=3`）离线训练与评估

> 2026-10-27。前置文档：`docs/value_net_pipeline_survey.md`（管线调研，本文**照它的 §4.3/§4.5 执行**，不重复探索）。
> 目标：用 `lowq_f64` 的终局标签训一个 **103,137 参数**的 `ValueNetwork(32,3)`（现役独立值网是 542,913 参数），
> 看它能不能在 **10 s 决策预算**里替掉 rollout 的叶子值网，从而换出更强的随机化。
>
> **所有数字都来自本文列出的真实命令输出**；日志全部落盘（`training_history/logs/`、`training_history/runs/`）。

---

## 0. 结论（TL;DR）

| 判据 | 门槛 | 32/3 实测（3 种子） | 现役 / 老基线 | 判定 |
|---|---|---|---|---|
| **留出 R²**（按局切最后 500 局 = 32,000 行） | **≥ 0.45** | **0.4711 / 0.4707 / 0.4730**，均值 **0.4716**±0.0010 | 现任 `posnet_A_k5_m32` = **0.4533**；64/6 重训 = **0.4701**；朴素 = **0.3932** | ✅ 过关（**且≥ 64/6**） |
| **排序一致性** `ρ(tanh学生, tanh老师)` | **≥ 0.95** | **0.9598 / 0.9621 / 0.9590**，均值 **0.9603** | — | ✅ 过关（**贴线**，仅高 0.01） |
| 前向耗时（batch 64，4 线程） | 越快越好 | **25.7 ms**（5 次中位） | 64/6 = **143.4 ms** ⇒ **5.13×** | ✅ 与 5.6× 估计一致 |

**一句话**：**离线过关** —— 容量从 542,913 砍到 103,137（5.3×）**没有损失精度**（R² 反而比 64/6 高 0.0015，落在 3 种子噪声内），
单帧前向从 2.24 ms 降到 0.40 ms。**但它现在还不能"直接替换"**：

1. **形状耦合没解**（调研 §5.1）：小值网只能作为**独立网**挂在消费侧，塞不进三网 ckpt；
2. 两处消费点写的是 `value_model(board, stat)["value"]`（`az_intent/value_batch.py:55`、`az_intent/flat_rollout_greedy.py:284`），
   而 `ValueNetwork` 返回**裸张量** ⇒ 必须加一层适配（**且必须把 `y_std/y_mean` 折进 `value_head.2`**，否则量纲错 2.6×）；
3. **ρ 贴着门槛线（0.960 vs 0.95）**，且历史有 **5 次"离线更准、实战更差"**（`docs/dl_feasibility_20261026.md:49-55`）
   ⇒ 离线过关**只说明"值得花 128 局去证伪"**，不能当"可以上线"。

---

## 1. 第 1 步：给训练脚本加宽度/深度参数

### 1.1 改了什么

| 文件 | 改动 | 默认行为 |
|---|---|---|
| `code/my_ai/entity_transformer/train_value_stream.py` | ① 加 `--width`（默认 64）/ `--blocks`（默认 6）；② `ValueNetwork()` → `ValueNetwork(latent_dim=a.width, num_resblocks=a.blocks)`；③ 两处 `"config": {"latent_dim": 64, "num_resblocks": 6}` → 用参数；④（附）加 `--snap-dir`（默认空 = 仍写进 `--out-dir`） | **不变** |
| `code/my_ai/entity_transformer/eval_value_ckpt.py` | L62 写死的 `ValueNetwork()` → 读 ckpt 的 `config` | **不变**（老 ckpt 的 config 就是 64/6） |
| `code/my_ai/entity_transformer/graft_value_ckpt.py` | L47 写死的 `ValueNetwork()` → 读被嫁网的 `config` | **不变** |

> `--snap-dir` 是第 4 个改动（任务只列了 3 个）：快照默认写 `--out-dir`，而本次 `--out-dir` 在 `checkpoints/` 里
> ⇒ 30 个快照会变成 git 未跟踪文件堆在 `checkpoints/`。加这个开关把快照引去 `training_history/`（gitignore），**不传参数时与原逻辑逐字节一致**。

### 1.2 "默认行为不变"的**证据**（不是口头保证）

三个历史 ckpt 的 `config` 全是 64/6 ⇒ `ValueNetwork(64,6)` 与 `ValueNetwork()` 完全同构：

```
lowq_value_seed1.pt          config={'latent_dim': 64, 'num_resblocks': 6}         val_r2=0.4701  params=542913
lowq_value_seed2.pt          config={'latent_dim': 64, 'num_resblocks': 6}         val_r2=0.4688  params=542913
full/lowq_value_seed0.pt     config={'latent_dim': 64, 'num_resblocks': 6}         val_r2=0.4581  params=542913
```

改完之后**用改后的 `eval_value_ckpt.py` 重跑 64/6 老 ckpt**，仍然得到文档里的那个数：

```
[eval] lowq_value_seed1.pt
  data=lowq_f64 label=label.npy val_rows=32000 (games 4500..4999)
  R2=0.4701  MAE=0.2153  label_std=0.3842
```
（`docs/checkpoints.md:258` 记的 `val R²=0.4701`，**逐位一致** ⇒ 评估路径没被改动影响。）

### 1.3 冒烟（先证明代码路径能走通，再烧 15 分钟）

拿 `lowq_f64` 的**最后 40 局**（2,560 行）造子集，跑 1 epoch：

```
[data] N=2560 games=5000 label=hp
[data] train=2240 val=320 (val games 4995..4999)
[net] seed0 ValueNetwork(width=32, blocks=3) params=103,137
  [seed 0 ep  1] train_mse=0.9342 val_R2=0.0709 val_MAE=0.3879  * (5s)
[seed 0] best val_R2=0.0709 (ep 1) -> ...\_smoke_small32\ckpt\lowq_value_seed0.pt
```

```
config = {'latent_dim': 32, 'num_resblocks': 3}      ← config 落盘正确
n tensors 48
[eval] lowq_value_seed0.pt  →  R2=0.0710  MAE=0.3878   ← 独立 eval 复现训练自报的 0.0709
```

---

## 2. 第 2 步：训练 3 个种子

### 2.1 命令（`run_logged.sh`，日志落盘）

```bash
bash code/run_logged.sh small_value_32_3 \
  D:/anaconda3/envs/pytorch-gpu/python.exe -u code/my_ai/entity_transformer/train_value_stream.py \
  --data D:/2026智能体大赛_新2_distill/training_history/distill_top2/lowq_f64 \
  --label-file label.npy --width 32 --blocks 3 \
  --seeds 0 1 2 --epochs 10 --batch-size 512 --shard-games 50 --val-games 500 \
  --out-dir checkpoints/small32_3 \
  --snap-dir training_history/distill_top2/small32_3_snaps \
  --log-file training_history/logs/train_small32_3.log
```

- **`exit=0`，用时 911 s（15.2 min）** —— 落在调研预估的 10–18 min 内。
- 日志：`training_history/logs/train_small32_3.log`、`training_history/runs/20261027_125902_small_value_32_3/output.log`
- **唯一偏离任务书的地方**：`--out-dir` 用 `checkpoints/small32_3/` 而不是 `checkpoints/`。
  原因：任务的输出名是 `lowq_value_seed{0,1,2}.pt`，而 `checkpoints/lowq_value_seed1.pt`/`seed2.pt` **是已入库的 64/6 基线**
  （`git ls-files` 确认被跟踪）⇒ 直接写 `checkpoints/` 会**覆盖掉要拿来比的那三个基线**。子目录保持同名，不撞车。
- 环境约束：本次只用 **1 个训练进程**（GPU，峰值 1,065 MiB / 8,151 MiB）、CPU 线程上限 4（`OMP_NUM_THREADS=4`）；
  未 kill 任何进程、未删/移任何数据文件。

### 2.2 逐 epoch 留出 R²（最后 500 局，32,000 行）

| epoch | seed0 R² / MAE | seed1 R² / MAE | seed2 R² / MAE |
|---|---|---|---|
| 1 | 0.4448 / 0.2205 | 0.4273 / 0.2258 | 0.4532 / 0.2200 |
| 2 | 0.4683 / 0.2159 | 0.4605 / 0.2181 | 0.4621 / 0.2177 |
| 3 | 0.4545 / 0.2196 | 0.4659 / 0.2163 | 0.4670 / 0.2158 |
| 4 | 0.4391 / 0.2233 | 0.4668 / 0.2169 | 0.4690 / 0.2154 |
| 5 | 0.4680 / 0.2157 | 0.4684 / 0.2161 | 0.4668 / 0.2164 |
| 6 | **0.4711** / 0.2154 | 0.4563 / 0.2189 | 0.4670 / 0.2168 |
| 7 | 0.4700 / 0.2148 | 0.4601 / 0.2185 | 0.4703 / 0.2160 |
| 8 | 0.4702 / 0.2159 | 0.4696 / 0.2159 | 0.4709 / 0.2154 |
| 9 | 0.4538 / 0.2189 | **0.4707** / 0.2156 | 0.4684 / 0.2162 |
| 10 | 0.4692 / 0.2163 | 0.4702 / 0.2144 | **0.4730** / 0.2148 |

- **best（与 64/6 基线同协议：取 10 个 epoch 的留出最高值）**：`0.4711 / 0.4707 / 0.4730`，**均值 0.4716，种子间 pstdev 0.0010**。
- **epoch10（不做 best-of 选择）**：`0.4692 / 0.4702 / 0.4730`，**均值 0.4708** —— 说明上面的 0.4716 **不是** best-of-10 的挑选效应堆出来的。
- 波动：单 epoch 在 0.439–0.473 之间跳 **±0.01**（比种子间差异大 10 倍）⇒ **单 epoch 读数不能当结论**，这也是历史"3 种子只差 0.002"的正确读法：种子稳定性 ≠ epoch 稳定性。

### 2.3 独立复算（不用训练脚本自带的评估）

```
[eval] lowq_value_seed0.pt  →  R2=0.4710  MAE=0.2154
[eval] lowq_value_seed1.pt  →  R2=0.4707  MAE=0.2156
[eval] lowq_value_seed2.pt  →  R2=0.4730  MAE=0.2148
（data=lowq_f64 label=label.npy val_rows=32000 games 4500..4999 label_std=0.3842）
```
与训练自报逐位一致 ⇒ 数字不是评估代码的产物。

### 2.4 与三档基线比（**同一留出集、同一指标**）

| 配置 | 参数 | 留出 R² | 相对 32/3 |
|---|---|---|---|
| 朴素（常数 = 训练集均值） | 1 | 0.3932 | −0.0784 |
| 现役值网 `posnet_A_k5_m32.pt` 的 `value_state`（部署在用的那个） | 553,313 | 0.4533 | −0.0183 |
| `ValueNetwork(64,6)` 重训（`lowq_value_seed1/2` + 文档 seed0） | 542,913 | 0.4701 / 0.4688（+0.4708）⇒ ≈0.4699 | −0.0017 |
| **`ValueNetwork(32,3)` 本次 3 种子** | **103,137** | **0.4716 ± 0.0010** | — |

**解读**：把主干容量砍到 1/5.3，R² **没有掉**（+0.0017 落在种子噪声 0.001–0.002 内）。
这与调研 §4.4 的判断一致：**0.47 的天花板由标签性质（从当前局面预测几十回合后的终局血量差）决定，不是容量不足**。
⇒ "小网会不会学不动"这个风险**已被证伪**，剩下的风险全是"换了网会不会改变搜索行为"（见 §3）和"实战会不会变差"（见 §6）。

---

## 3. 第 3 步：学生 vs 老师的排序一致性

### 3.1 脚本

新脚本 `code/test_match/eval_small_value.py`（约 200 行，含注释/自检）：

```bash
D:/anaconda3/envs/pytorch-gpu/python.exe -u code/test_match/eval_small_value.py \
  --student-ckpt checkpoints/small32_3/lowq_value_seed0.pt \
  --data D:/2026智能体大赛_新2_distill/training_history/distill_top2/lowq_f64 \
  --val-games 500 --n-frames 20000 \
  --out-json training_history/logs/eval_small_value_seed0.json
```

- 留出集 = `lowq_f64` **按局切**的最后 500 局（32,000 行），**随机抽 20,000 帧**（`--seed 20261027`，无放回，可复现）；
- **老师** = `checkpoints/posnet_A_k5_m32.pt` 的 `value_state`（`load_three_models()` 的第 3 个网，即现役值网）；
- **学生** = `checkpoints/small32_3/lowq_value_seed{0,1,2}.pt`（独立 `ValueNetwork`）；
- 两侧都换算到**部署真正消费的量** `tanh(v)` 再算 Spearman/Pearson。

三个坑都在脚本里显式处理了：
1. **量纲**：学生 ckpt 里存了 `y_mean/y_std`（z-score 训练），脚本做 `v = raw * y_std + y_mean` 折回标签尺度
   ——等价于 `graft_value_ckpt.py` 把系数折进 `value_head.2`。**自检**见 §3.3（老师/学生对 label 的 R² 都是 0.45 量级，说明两边同尺度）；
2. **输出形态**：老师返回 **dict**（取 `["value"]`），学生返回**裸张量**（不能照抄 `["value"]`）；
3. **别按帧切**：全程用 `game_id.npy` 推局边界。

### 3.2 结果（3 个种子，同一批 20,000 帧）

| 学生 | **Spearman ρ(tanh)** | Pearson r(tanh) | Pearson r(裸值) | 平均 \|Δtanh\| | 符号不一致 |
|---|---|---|---|---|---|
| `seed0` | **0.9598** | 0.9701 | 0.9713 | 0.0507 | 9.03% |
| `seed1` | **0.9621** | 0.9708 | 0.9717 | 0.0484 | 8.32% |
| `seed2` | **0.9590** | 0.9696 | 0.9707 | 0.0504 | 9.40% |
| **均值** | **0.9603** | 0.9702 | 0.9712 | 0.0498 | 8.92% |

**门槛 ρ ≥ 0.95 ⇒ 三个种子全部过关**（但只高 0.009–0.012，属于"贴线通过"，不是"轻松通过"）。

### 3.3 量纲自检（确认上面的 ρ 不是"两个错的量纲碰巧相关"）

```
老师 raw  R2 vs label = 0.4496 | tanh = 0.4423
学生 raw  R2 vs label = 0.4707 | tanh = 0.4667
```

老师 raw 的 R²=0.4496 与全量 32k 上的 0.4533 同量级（20k 子样本的抽样噪声）⇒ **老师的裸输出已经是标签尺度**，
学生的 `raw*y_std+y_mean` 折回后也是标签尺度，两边 tanh 后才能对得上（`tanh` 后 R² 只掉 0.004，说明值域本来就在 ±1 内）。

### 3.4 ⚠️ 那 9% 的"符号不一致"要紧吗？—— 分档看就没了

值网输出是**连续分**，大多数帧都挤在 0 附近，符号翻转本身不代表"看反了局面"。按**老师的把握程度**分档：

| 子集 | 帧数 | 占比 | 符号不一致 | ρ |
|---|---|---|---|---|
| 全部 | 20,000 | 100% | 8.3–9.4% | 0.959–0.962 |
| `\|tanh_t\| ≥ 0.05` | 16,126 | 80.6% | 2.65–3.65% | 0.973 |
| `\|tanh_t\| ≥ 0.20` | 6,572 | 32.9% | **0.00–0.02%** | 0.964–0.967 |
| `\|tanh_t\| ≥ 0.40` | 1,478 | 7.4% | **0.00%** | 0.956–0.963 |

⇒ **在老师"有态度"（\|tanh\|≥0.2，占 1/3 的帧）的局面上，学生几乎从不把谁领先看反（≤0.02%）**；
分歧全部集中在 `\|tanh\|<0.2` 的势均力敌区，而那一区的值本来就接近 0、对 argmax 的排序贡献也小。
（注：ρ 在分档子集上并不升高，甚至略降 —— 因为截断后动态范围变小、秩更容易被噪声打乱。**这是裁剪效应，不是学生在该区更差。**）

---

## 4. 前向耗时实测（32/3 vs 64/6，batch 64，同线程数）

脚本：`code/test_match/bench_value_forward.py`（**同进程、同线程、背靠背**，两侧都用真实 ckpt 权重；CPU 计时）。

```
torch 2.12.0.dev20260408+cu128  threads=4  cuda_avail=True
student lowq_value_seed0.pt: ValueNetwork(32,3) params=103,137
teacher posnet_A_k5_m32.pt: value_state(AntWarNetwork) params=553,313
 batch    64/6 ms    32/3 ms  speedup
    64     136.20      21.28    6.40x
    32      76.57      18.02    4.25x
   128     279.42      59.32    4.71x
     1      13.54       2.28    5.95x
```

**batch=64 重复 5 次**（机器上当时还有 8 局对局在跑，CPU 负载 38–42%，绝对值偏慢且抖）：

| repeat | 64/6 ms | 32/3 ms | speedup |
|---|---|---|---|
| 1 | 123.86 | 24.75 | 5.01× |
| 2 | 166.55 | 32.46 | 5.13× |
| 3 | 149.36 | 25.72 | 5.81× |
| 4 | 136.54 | 40.32 | 3.39× |
| 5 | 143.40 | 18.91 | 7.58× |
| **中位** | **143.40** | **25.72** | **5.13×** |

- **每帧成本（batch 64，4 线程）**：64/6 = **2.24 ms/帧** → 32/3 = **0.40 ms/帧**。
- 比值 **3.4×–7.6×（中位 5.1×）**，与调研里 2 线程的 **5.59×**、任务书里的 **5.6×(=4.9× 保守版)** 一致。
  **抖动来自机器负载（8 局对局），不是配置差异** —— 两侧永远在同一进程里前后脚测，比值才是可用量。
- ⚠️ **"有效叶子 17→62"本文没有实测**：值网前向只占决策预算的一部分，`flat_rollout_greedy` 的 `_leaf_values` 里
  `encode_observation`（特征抽取）是**固定成本**，所以 5.1× 的前向加速**不可能**等比例变成 5.1× 的叶子数
  （17×5.1=87 是上界，62 对应的固定开销占比约 60%）。要拿这个数只能进 rollout 计时，属于下一步。

---

## 5. 判据核对

| # | 判据 | 门槛 | 实测 | 判定 |
|---|---|---|---|---|
| 1 | 留出 R²（按局切，32k 行） | ≥ 0.45 | **0.4716**（0.4707–0.4730） | ✅ |
| 2 | 不低于现役 0.4533 | — | +0.018 | ✅ |
| 3 | 不明显低于 64/6 的 0.4701 | — | +0.0015 | ✅ |
| 4 | Spearman ρ(tanh) vs 老师 | ≥ 0.95 | **0.9603**（0.9590–0.9621） | ✅（贴线） |
| 5 | 前向是否真的更快 | — | 5.1×（中位），2.24→0.40 ms/帧 | ✅ |

**⇒ "离线过关"成立**（R²≥0.45 且 ρ≥0.95）。

**但"能作为替代"要说清楚是哪一层意思：**

- ✅ **容量层**：32/3 完全够用，砍到 103k 参数**没有任何精度代价**。"必须先保住大网"的顾虑可以放下。
- ⚠️ **行为层**：ρ=0.96 意味着**约 4% 的排序信息变了**；在 `|tanh|≥0.2` 的局面上几乎不翻转，但搜索的 argmax 只关心最高分的少数叶子
  ——**尾部行为没有被这个指标覆盖**。要证伪只能跑对局。
- ❌ **工程层（现在还不能换）**：
  1. 小值网**塞不进三网 ckpt**（`load_three_models` 用同一组 `latent_dim/num_resblocks` 建三个网，调研 §5.1）；
  2. 两处消费点写死 `["value"]`：`az_intent/value_batch.py:55`、`az_intent/flat_rollout_greedy.py:284`，而 `ValueNetwork` 返回**裸张量**；
  3. 走独立网这条路**必须**把 `y_std/y_mean` 折进 `value_head.2`（`graft_value_ckpt.py:68-76` 的写法）。
     本次学生是 `y_std=0.3861`、`y_mean=-0.0004`（z-score 训练）⇒ **不折的话输出会被放大 1/0.386≈2.6×**（就是调研 §5.3 那个历史事故）：
     部署侧对值做 `tanh`，2.6× 放大会把 `|v|>0.4` 的叶子**全部推饱和**，同时把"网络叶子"与"终局叶子（±1）"的相对尺度弄错 ⇒ 排序被压平。
     （这正是"学生 vs 老师"必须先折回再算的原因；顺带说清楚：Spearman 对**任意严格单调**变换不变，
     所以**纯缩放/平移不会改变 ρ**，但 **Pearson 与 R² 一定会错** —— 报告里两个都给了，就是为了让这个坑可见。）

---

## 6. 下一步建议（按性价比排序）

1. **接 rollout 计时（半天，零风险）**：只把 `_leaf_values`（`flat_rollout_greedy.py:284`）换成"独立 `ValueNetwork` + dict 适配 + 折归一化"，
   **不改任何搜索逻辑**，在同一 `--time-budget 10` 下量"有效叶子数"与"单步耗时"。
   这是把 5.1× 前向加速变成"叶子 17→62"的**唯一验证方式**，也是后续一切收益的分母。**推荐先做这个。**
2. **128 局配对（必须做，约 1–2 h）**：`AZAI_CKPT` 不变、只换叶子值网（32/3 vs 现状 64/6 `value_state`）。
   判据按项目惯例 ≥128 局、报告 95% CI；**门槛建议"不显著变差"而不是"显著变好"**——因为历史 5 次"离线更准、实战更差"。
   如果这一步掉了，**直接停**，不要再花时间调小网。
3. **只有当 (2) 通过后**，才值得去吃 5.1× 换来的随机化红利（把省下的预算加到 rollout 宽度/深度）。
4. **如果 (2) 掉了，备选方案（按顺序）**：
   - **48/4 折中**：约 1.7× 参数、前向约 2.6× 快（估），ρ 应该能到 0.97+，是"要么 32/3、要么 48/4"的自然折中；
   - **蒸馏而非重训（调研方案 B）**：先给 `lowq_f64` 的 320k 帧打上**老师的 tanh 输出**（纯推理，仓库无现成脚本，
     照 `cpp_test/check_incumbent_value.py` 改 15 行），再用 `--label-file teacher_tanh.npy` 重训。
     目标直接是"别改变老师"，ρ 会明显高于 0.96 —— 代价是学生上限被老师锁死（老师 R² 只有 0.45）。
     **注意**：本次结果说明"重训已经不损失精度"，所以蒸馏的价值**只在行为对齐**，不在精度。
5. **不建议**：再加大数据量（`lowq_f256/full`）。调研 §4.2 已有结论：terminal 标签 64 帧/局就够，帧数翻 4 倍 R² 反而更低。

---

## 7. 复现（全部命令）

```powershell
$env:PYTHONIOENCODING="utf-8"; $env:OMP_NUM_THREADS="4"
$PY = "D:\anaconda3\envs\pytorch-gpu\python.exe"

# ① 训练（15 min，日志落盘 + 记入 docs/experiment_log.md）
& "D:\Program Files\Git\bin\bash.exe" code/run_logged.sh small_value_32_3 `
  D:/anaconda3/envs/pytorch-gpu/python.exe -u code/my_ai/entity_transformer/train_value_stream.py `
  --data D:/2026智能体大赛_新2_distill/training_history/distill_top2/lowq_f64 `
  --label-file label.npy --width 32 --blocks 3 --seeds 0 1 2 --epochs 10 `
  --batch-size 512 --shard-games 50 --val-games 500 `
  --out-dir checkpoints/small32_3 --snap-dir training_history/distill_top2/small32_3_snaps `
  --log-file training_history/logs/train_small32_3.log

# ② 固定留出集独立评估（应得 0.4710 / 0.4707 / 0.4730）
foreach ($s in 0,1,2) {
  & $PY -u code/my_ai/entity_transformer/eval_value_ckpt.py `
    --ckpt checkpoints/small32_3/lowq_value_seed$s.pt `
    --data D:/2026智能体大赛_新2_distill/training_history/distill_top2/lowq_f64 `
    --label-file label.npy --val-games 500 --device cpu
}

# ③ 学生 vs 老师（应得 ρ=0.9598/0.9621/0.9590）
foreach ($s in 0,1,2) {
  & $PY -u code/test_match/eval_small_value.py `
    --student-ckpt checkpoints/small32_3/lowq_value_seed$s.pt `
    --data D:/2026智能体大赛_新2_distill/training_history/distill_top2/lowq_f64 `
    --n-frames 20000 --out-json training_history/logs/eval_small_value_seed$s.json
}

# ④ 前向耗时（batch 64：应得 5.1× 上下，机器负载越轻越接近纯比值）
& $PY -u code/test_match/bench_value_forward.py `
  --student-ckpt checkpoints/small32_3/lowq_value_seed0.pt --threads 4 --batch 64 32 128
```

产物：
- `checkpoints/small32_3/lowq_value_seed{0,1,2}.pt`（各 429,587 B，`ValueNetwork(32,3)`，103,137 参数，
  `config={latent_dim:32,num_resblocks:3}`、`y_mean=-0.0004`、`y_std=0.3861`；sha256 见 `docs/checkpoints.md`）
- 17 个 best 快照：`training_history/distill_top2/small32_3_snaps/`
- 指标 json：`training_history/logs/eval_small_value_seed{0,1,2}.json`

---

## 8. 约束遵守说明

| 约束 | 实际情况 |
|---|---|
| 不干扰正在跑的实验 | 未 kill/暂停任何进程；训练只用 **1 个进程**，CPU 线程上限 **4**（`OMP_NUM_THREADS=4`），显存峰值 **1,065 MiB / 8,151**（与 `train_prior_z` 共存） |
| 不改默认行为 | 见 §1.2（64/6 老 ckpt 重跑 **0.4701**，与文档逐位一致） |
| 不改 `az_intent/*` 核心搜索代码 | 只**读**了 `value_batch.py` / `flat_rollout_greedy.py` 用来写 §5 的结论，**没有改动** |
| 不删/移数据 | 只新增；训练写的是 `checkpoints/small32_3/`，**没有覆盖**已入库的 64/6 基线 |
| 日志落盘 | `training_history/logs/train_small32_3.log`、`runs/20261027_125902_small_value_32_3/output.log`、`eval_small_value_seed{0,1,2}.log/.json`、`bench_value_forward.log` |
| 已知没做 | ① rollout 里的叶子数/耗时实测；② 128 局配对；③ 与"部署时实时 encode 出来的 stats"逐维比对（调研 §5.2 遗留项） |
