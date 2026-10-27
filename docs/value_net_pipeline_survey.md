# 仓库内「价值网训练管线」现状调研

> 2026-10-27。**只读调研**，不改动 `code/my_ai/**`、不删不移任何数据。
> 目的：在动手训「width=32 / blocks=3 的小值网」之前，先摸清仓库里**已经有的**数据、脚本、checkpoint 与历史读数，避免重复造轮子。
>
> **所有数字都来自真实命令输出或仓库文档（标注了出处）**；查不到的写「未找到」。
> 复现本文探针的命令见 §6。

---

## 0. 先给结论（TL;DR）

| 问题 | 答案 |
|---|---|
| 有没有现成的值网训练管线？ | **有，而且不止一条**。`code/my_ai/entity_transformer/train_value_stream.py`（`ValueNetwork` on memmap）+ `code/my_ai/distill_top2/train_value_distill.py`（`AntWarNetwork` value_state）已经能直接吃仓库里的数据。 |
| 现成脚本能不能**不加修改**就训 32/3？ | **不能**。`train_value_stream.py` 第 164 行硬编码 `ValueNetwork()`（=64/6），`eval_value_ckpt.py:62` 与 `graft_value_ckpt.py:47` 同样硬编码。改动量 ≈ **3 处 + 2 个 argparse**。 |
| 数据够不够？ | **够**。`lowq_f64/`（320,000 行，6.2 GB，可全量进内存）就够用；`lowq_full/`（4,280,296 行，82.5 GB memmap）更大但慢。 |
| 有没有"现成的 32/3 值网"直接可用？ | **没有**。`checkpoints/` 里所有 **board 输入**的 value 网都是 **64/6**（`AntWarNetwork` 554,990 元素 / `ValueNetwork` 542,913 参数）；少数非 64/6 的是**实体 Transformer / Q8 MLP**，输入格式不同，不能直接替换。 |
| 最大的坑是什么？ | **形状耦合**：`load_three_models()` 用 ckpt 顶层的 `latent_dim`/`num_resblocks` **同时**建 class/pos/value 三网 ⇒ 值网**不能**比策略网窄。小值网只能走"独立 `ValueNetwork`"的路，不能塞进现有三网 ckpt。 |

---

## 1. 数据集清单

> 位置：`training_history/distill_top2/`（下称 `$D`）。
> 探针方式：`.npz` 用 `np.load` 逐键读；`>500 MB` 的 `.npy` 一律 `mmap_mode='r'` 只取 shape/dtype + 抽样（**每个文件最多前 200,000 行**算统计量，故大文件的 min/max 是抽样值，已在表中标注 `*`）。

### 1.1 A 族：`(board, stats, value)` —— **与当前网络输入格式一致**（可直接喂 `AntWarNetwork` / `ValueNetwork`）

输入格式：`board (N,28,19,19) float16` + `stats (N,42) float32`。
这正是 `AntWarNetwork.BOARD_CHANNELS=28 / BOARD_SIZE=19 / STATS_DIM=42`（`code/my_ai/network.py:131-133`）与 `ValueNetwork` 的输入约定。

| 文件 | 大小 | 键 → 形状 / dtype | `value` 语义 | 样本数 | 生成脚本（溯源） | mtime |
|---|---|---|---|---|---|---|
| `entity_ds_128_board.board.npy` | **2.4 GiB** | `(root)` → `[127578,28,19,19] f16` | —（**只有 board**） | 127,578 | **未找到**（见注①） | 10-18 10:11 |
| `entity_ds_128.npz` | 9.4 MB | `tower_cont[127578,32,3]f32` `tower_type[..,32]i16` `tower_pos[..,32]i16` `tower_mask[..,32]bool` `ant_cont[..,48,4]f32` `ant_kind/ant_behavior/ant_pos/ant_mask[..,48]` `stats[..,42]f32` `value[127578]f32` `seed i32` `round i16` `player i8` | **亚军手写价值 `evaluate_terminal` 的原始量纲**：min 624.3 / mean 10,470 / max 17,600 | 127,578（128 局，seed 920–1047） | `code/my_ai/entity_transformer/build_dataset.py`（`--raw-dir ru_ent_raw`，源 = 桥接 dump） | 10-18 09:14 |
| `entity_ds_128_board.npz` | 9.4 MB | 与上面**逐键同形同统计量**（各键 min/max/mean 全一致） | 同上 | 127,578 | 疑为 `entity_ds_128.npz` 的同份拷贝，**未在代码中找到写它的脚本** | 10-18 10:11 |
| `ent8_board.board.npy` | 165.6 MB | `[8192,28,19,19] f16` | —（只有 board） | 8,192 | 同上，**未找到** | 10-18 00:25 |
| `value_ds_ent8.npz` | 4.1 MB | `board[8192,28,19,19]f16` `stats[8192,42]f32` `value[8192]f32` `seed` `round` `player` | 亚军 `evaluate_terminal`，min 1,532 / mean 10,350 / max 16,360 | 8,192（8 局，seed 910–917） | `code/my_ai/distill_top2/gen_value_dataset.py` | 10-17 23:21 |
| `value_ds_12g.npz` | 4.1 MB | 上列 + `tower_hp[..,2,19,19]f16` `tower_type[..,2,19,19]i16` | 同上，min **−168.5** / mean 7,545 / max 11,020 | 5,942（12 局，seed 100–111） | 同上 | 10-17 14:04 |
| `value_ds_24g.npz` | 8.6 MB | 同 `12g` | 同上，min **−328.5** / mean 7,497 / max 12,750 | 12,356（24 局，seed 100–123） | 同上 | 10-17 14:43 |
| `value_ds_runnerup_selfplay.npz` | 16.4 MB | `board / stats / value / seed / round / player` | 同上，min 2,223 / mean 10,360 / max 16,280 | 32,768（32 局，seed 850–881，**亚军自对弈**） | 同上（换 raw dump 来源） | 10-17 20:14 |
| `sim_value_ds_128.npz` | 4.7 MB | 同 `entity_ds_128` 的实体键 + `stats` + `value` | 同 `entity_ds_128`（同一批 127,578 行） | 127,578 | `code/my_ai/entity_transformer/build_sim_value_dataset.py`（把 128 局投影成"**单边**"：只留己方塔+敌方蚁） | 10-18 17:02 |
| `rollout_ds_test.npz` | 4.7 MB | `board[184368,28,19,19]f16` `stats[184368,42]f32` `value[184368]f32` … | 亚军 `-DLURE_DUMP_VALUE` 单边 rollout 值，min 7,994 / mean 10,520 / max 11,900 | 184,368（**仅 2 局**，seed 1100–1101） | ⚠️ **未找到生成脚本**；⚠️ **`stats` 退化**（只有前 4 维非零，其余全 0，见 §5.2） | 10-18 09:51 |
| `_smoke.npz` / `_smoke2.npz` / `_smoke_bridge.npz` / `_partial_check.npz` | 0.05–0.54 MB | 同 A 族 | 冒烟用 | 60 / 80 / 936 / 418 | `gen_value_dataset.py --smoke` 等 | 10-17 |

注①：**仓库里没有任何 `.py` 引用 `*.board.npy` 或 `entity_ds_128_board.board.npy`**（全仓 grep 结果为空）。
`build_dataset.py` 只有 `--with-board` 开关但代码里没写 `*.board.npy` 文件 ⇒ **这批 board 张量是一次性脚本产物，生成脚本未找到**。
**但已验证它与 `entity_ds_128.npz` 逐行 1:1 对齐**（用 ch4=己方塔/ch5=敌方塔的格子数 == `tower_mask` 之和）：

```
rows with towers: [24 25 26 27 28 29] tower counts: [2 2 2 2 2 2]
  row     24: ch4=2 ch5=0 sum=2 | tower_mask=2  tower_types=[0]
  row     25: ch4=0 ch5=2 sum=2 | tower_mask=2  tower_types=[0]
  row  50000: ch4=2 ch5=2  tower_mask=4  round=405 player=0
  row 127000: ch4=2 ch5=2  tower_mask=4  round=223 player=0
```

### 1.2 B 族：5000 局低质量自对弈（`lowq_*`）—— **唯一"大数据"**

来源：`F:\...\training_history\lowq_e2`（5000 局原始采集，每局 npz + `ents_*.jsonl`）→ 合并成 `lowq_e2_merged`（91.3 GB）→ 拷到本仓库的列式 memmap。
策略 = **M6c 裸策略 + ε=2% 随机动作，C++ 引擎**（`docs/prereg_20261019_low_quality_value.md` §2）。

| 路径 | 键 → 形状 / dtype | 说明 | 规模 | 生成脚本 | mtime |
|---|---|---|---|---|---|
| `lowq_full/board.npy` | `[4280296,28,19,19] f16`（**82.5 GB**） | 逐帧全量 board | 4,280,296 帧 / 5,000 局 | `entity_transformer/merge_lowq_data.py`（`meta.json`: `total=4280296, src=F:\...\lowq_e2`） | 10-19 11:18 |
| `lowq_full/stats.npy` | `[4280296,42] f16` | 42 维 stats | 同上 | 同上 | 10-19 |
| `lowq_full/{hp_us,hp_opp}.npy` | `[N] int16` | **逐帧**血量（**不是标签**，别拿它现算） | | | |
| `lowq_full/{round,player}.npy` | `int16` / `int8` | 回合 / 视角 | | | |
| `lowq_full/tower_{pos,type,hp,mask}.npy` | `[N,32]` | 32 槽塔实体 | | | |
| `lowq_full/ant_{cont,kind,behavior,pos,mask}.npy` | `[N,64,{4,}]` | 64 槽蚁实体 | | | |
| `lowq_full/game_offsets.npy` | `[5001] int64` | **局边界**（按局切验证集必须用它） | | | |
| `lowq_full/value_win.npy` | `[4280296] f32` | **终局血量差 / 20（clip ±1，平局分胜负给 ±0.1）**；实测 min −1 / max 1 / mean 0 / **std 0.3757** / p5 −0.6 / p95 0.6 | | `merge_lowq_data.py` | |
| `lowq_full/label_geo_tau{5,10,50,100,200}.npy` | `[N] f32` | **相对**标签 = 指数加权未来血量差 − 当前；std 0.024 / 0.036 / 0.076 / 0.098 / 0.121 | | `build_rel_labels.py --mode rel --weight geo --save-full` | |
| `lowq_full/label_kgeo_tau{5,10,30,50,100,200}.npy` | `[N] f32` | 同上，`kgeo` 权重；std 0.042 / 0.059 / 0.094 / 0.114 / 0.143 / 0.169 | | 同上 `--weight kgeo` | |
| `lowq_f64/board.npy` | `[320000,28,19,19] f16`（**6.2 GB**） | **每局均匀抽 64 帧**（抽样种子硬编码 `default_rng(12345)`，`subsample_frames.py:58`） | 320,000 行（"可全量进内存"） | `subsample_frames.py`（`meta.json`: `games=5000, frames_per_game=64`） | 10-19 11:18 |
| `lowq_f64/{stats,hp_us,hp_opp,frame_idx,game_id}.npy` | `stats[320000,42] f16`、`hp_*[320000] i16`、`frame_idx i64`、`game_id i32` | 无 `game_offsets` ⇒ 用 `game_id` 反推局边界 | | 同上 | |
| `lowq_f64/label.npy` | `[320000] f32` | **= 终局血量差/20**（min −1 / max 1 / **std 0.3859** / p5 −0.65 / p95 0.65） | | 同上 | |
| `lowq_f64/label_{geo,kgeo}_tau*.npy` | `[320000] f32` | 9 个 rel 标签（std 0.024–0.169） | | `build_rel_labels.py` | |
| `lowq_f256/` | `board[1280000,28,19,19]`（**24.7 GB**）+ `stats` + `label.npy` + 8 个 rel 标签 | 每局 256 帧 | 1,280,000 行 | `build_frame_subset.py --frames-per-game 256` | 10-19 20:17 |
| `lowq_f64_ent.npz` | 实体键 + `stats[320000,42]f16` + `value` | 同 `lowq_f64` 帧的**实体特征**版；`value` ≈ 同 terminal 标签（抽样 std 0.431*） | 320,000 | `entity_transformer/build_lowq_entity_dataset.py` | 10-21 20:21 |
| `lowq_f256_ent.npz` | 同上 | 同上 | 1,280,000 | 同上 | 10-19 20:17 |

✅ **已抽查 board memmap 完整性**（防"大小正确、内容零填充"，见 §5.8）：`lowq_f64`（行 0/10 万/20 万/30 万）、`lowq_full`（行 0/200 万/400 万）、`entity_ds_128_board`（行 0/6 万/12.7 万）三份的 `nonzero_frac` 均 ≈ **0.100–0.104**，逐通道 max 一致，**未发现零填充区段**。

### 1.3 C 族：**不要用**（输入格式不兼容或不是价值标签）

| 路径 | 为什么不能用 |
|---|---|
| `q8/q8_ent.npz`(50,296) / `q8_ent_big.npz`(286,434) | 键是 `tower_*/ant_*/state_feats[63]/plan_feats[41]/value/std/n` —— **Q8 计划价值**（"这个 plan 值多少"），不是局面值；喂给 `ValueNetwork` 的输入都不对。训练脚本 `entity_transformer/train_q8_value.py` |
| `q8/q8_dataset*.jsonl`（27–607 MB） | 同上，原始 jsonl |
| `q8/gen_ds_{20,64}games.npz` | `labels[N,14]` + `state_feats[63]`，多标签生成数据，非标量值 |
| `prior_labels/*.jsonl` | 先验标签（class 分布），与值网无关 |
| `replay_dump/{replay,state_dump}_seed7.json` | 一局回放，装饰用 |
| `abl_*_raw/`（1.2 GB+661 MB+562 MB）、`ru_*_raw/`、`sp_*/` | **桥接 raw dump**（`raw_seed*.bin` 20.8 MB/局 + `ents_*.jsonl`）。是 §1.1/§1.2 各 npz 的**上游原料**，用它是"重新生成数据集"而不是"用现有数据集" |
| `lowq_full` 的 `ant_cont.npy`（2.1 GB）/`ant_pos.npy`（523 MB） | 实体特征，`ValueNetwork` 不吃 |

---

## 2. 训练脚本清单

> 判据：能产出「`(board 28×19×19, stats 42) → 标量价值`」的网。

### 2.1 主表

| # | 脚本 | 训练对象 | 数据要求 | 损失 | 评估指标 | 已知产出 ckpt |
|---|---|---|---|---|---|---|
| **S1 ★** | `code/my_ai/entity_transformer/train_value_stream.py` | `ValueNetwork`（**board+stats → 标量**，`code/my_ai/value_network.py`） | memmap 目录：`board.npy`/`stats.npy`/`label*.npy`/`game_offsets.npy`\|`game_id.npy` | **MSE on z-score 标签** `(y−y_mean)/y_std` | **按局**切最后 `--val-games` 局：`val_R2` / `val_MAE` | `checkpoints/lowq_value_seed{0,1,2}.pt`；`lowq_seed{s}_ep{e}_r{r}.pt` 快照 |
| **S2 ★** | `code/my_ai/distill_top2/train_value_distill.py` | `AntWarNetwork` 的 `value_state`（从 base ckpt warm start，`--train-mode head\|full`） | **单个 npz**：`board`/`stats`/`value`/`seed` | **MSE on z-score 标签** | 按 `--val-seeds` 切：`val_R2`（**原始尺度**）/`val_MAE` + ridge 线性基线 + 常数基线 | `$D/value_distill_gap_full.pt`、`value_distill_gap_head.pt`、`vd24_full_gap.pt`、`vd_runnerup_selfplay.pt`、`_curve_*.pt`、`_ent8_cnn.pt`（**都没进 `checkpoints/`**） |
| S3 | `code/my_ai/train_value_net.py`（`--small` = 32/2、`--stream`） | `ValueNetwork` | 每局 npz：`board`/`stats`/`value (T,2)=[hp_us,hp_opp]` | MSE on `(hp_us−hp_opp)/20`；`--tau N` 改指数加权未来 | 按**帧**切 `val_split`（⚠️ 会泄漏）`val_MAE` | `checkpoints/value_s2.pt`（`--stream`）、`training_history/value_2026*/` 8 个目录 |
| S4 | `code/my_ai/train_value_antwar.py` | `AntWarNetwork` 的 `value_state`（**直接写出可部署三网 ckpt**） | 目录下每局一个 npz：`board`/`stats`/`target` | **`MSE(tanh(raw), target)`**（与部署口径一致） | 按**局**切 `val_mae` | `checkpoints/grafted_sdk_A_vs2.pt`(0.3453)、`grafted_sdk_A_vsearch.pt`(0.0718)、`value_search_net.pt` |
| S5 | `code/my_ai/az_intent/train_value_net.py` | 三网 ckpt 的 `value_state`（class/pos 照抄，**只训 value**） | pkl 目录 → 自建 memmap 缓存 `(board,stats,player,value_target)` | MSE（`--freeze-backbone`/`--freeze-bn` 默认开/`--anchor-lambda`） | 整体 + **有塔/无塔分组** MSE 与 Pearson `r` | **`checkpoints/valnet_A2.pt`**、`training_history/inject_ex02/valnet_{C,r1,...}.pt` |
| S6 | `code/my_ai/entity_transformer/train.py` | 实体 Transformer 价值网（塔/蚁 token + 42 stats + CLS） | `entity_ds*.npz` / `lowq_*_ent.npz` + `--val-seeds` | MSE on z-score 标签 | 按 seed 切 `val_R2`/`val_MAE` | `entity_value_128_reduced.pt`(0.9969)、`entity_sim_value_small.pt`(0.9987)、`lowq_ent64_terminal.pt`(0.4717)、`entity_sim_value_xs.pt` |
| S7 | `code/my_ai/az_intent/az_train.py`（`--split`/`--value-only`）+ `value_warmup.py` | AZ 训练里的 `value_state` | 自对弈 pkl | MSE + `add_weighted_labels`（`terminal/rel/abs/mix` × `geo/kgeo`） | 训练日志 | 各 `az_fixed/*.pt` 的 `value_state` |
| ❌ | `code/my_ai/az_intent/train_value_prior.py` | **不是值网**：训位置图 `pos_state` 去拟合"价值先验" | `vp_*.npz`（含 `adv`） | 逐行位置 **CE** | CE / top1 命中 | `posnet_r1/A/A1/B` |

### 2.2 每个脚本的入口命令（照抄 `--help`/文档里已验证过的写法）

**S1（推荐起点）** —— 实测跑过的命令（`docs/experiment_log.md:11308-11316`，日志 `training_history/runs/20261019_131857_lowq_value_f64_3seed/`）：
```bash
D:/anaconda3/envs/pytorch-gpu/python.exe code/my_ai/entity_transformer/train_value_stream.py \
    --data D:/2026智能体大赛_新2_distill/training_history/distill_top2/lowq_f64 \
    --seeds 0 1 2 --epochs 10 --batch-size 512 --shard-games 50 --val-games 500 \
    --out-dir D:/2026智能体大赛_新2_distill/checkpoints --log-file <log>
```
- 关键参数：`--label {hp,win}`（默认 `hp`）；`--label-file label_kgeo_tau50.npy` 可换标签；`--shard-games`（I/O 与打乱单位）；`--val-games`（按局切）。
- 硬编码处：`model = ValueNetwork()`（L164）、`"config": {"latent_dim": 64, "num_resblocks": 6}`（L221/L229）。
- ⚠️ 标签**自动推断**：优先 `value_win.npy`，不存在则 `label.npy`（L79-85）。

**S2** —— 文档里的写法（`train_value_distill.py` 头注释）：
```bash
python -u code/my_ai/distill_top2/train_value_distill.py \
    --data training_history/distill_top2/value_ds_12g.npz \
    --ckpt checkpoints/grafted_sdk_A.pt --train-mode full \
    --out training_history/distill_top2/value_distill_gap.pt
```
默认 `--val-seeds 108-111`、`--epochs 60`、`--lr 1e-3`、`--batch-size 128`、`--patience 12`、`--value-pool gap`。

**S4** —— 文档里的写法（`train_value_antwar.py` 头注释）：
```bash
python code/my_ai/train_value_antwar.py --data-dirs data/s2_value_npz \
    --base checkpoints/grafted_sdk_A.pt --out checkpoints/grafted_sdk_A_vs2.pt \
    --epochs 30 --batch-size 256 --lr 1e-3 --val-games 12
```

**S5** —— 文档里的写法（`az_intent/train_value_net.py` 头注释）：
```bash
$PY -u code/my_ai/az_intent/train_value_net.py --ckpt training_history/az_fixed/three_mix_r10p_vw_pol_frozen.pt \
    --cache training_history/inject_ex02/value_cache --epochs 4 --lr 3e-4 \
    --out training_history/inject_ex02/valnet_r1.pt
```

**S6** —— 实测跑过的命令（`docs/experiment_log.md:11360`）：
```bash
D:/anaconda3/envs/pytorch-gpu/python.exe code/my_ai/entity_transformer/train.py \
    --data .../lowq_f64_ent.npz --val-seeds 4500-4999 --epochs 15 --batch-size 256 \
    --d-model 128 --n-layers 3 --n-heads 4 --out .../checkpoints/lowq_ent64_terminal.pt
```

### 2.3 三个**配套工具**（训完必用）

| 工具 | 作用 | 注意 |
|---|---|---|
| `code/my_ai/entity_transformer/eval_value_ckpt.py` | 在**固定留出集**（`lowq_f64` 最后 `--val-games 500` 局 = 32,000 行）上算 `R2 / MAE`，供跨配置可比 | ⚠️ L62 写死 `ValueNetwork()`；读 ckpt 的 `y_mean/y_std` 做反归一化 |
| `code/my_ai/entity_transformer/graft_value_ckpt.py` | 把 `ValueNetwork` 权重灌进三网 ckpt 的 `value_state`（`strict=False` 只覆盖 value 通路），**并把 `y_std/y_mean` 折进 `value_head.2`** | ⚠️ L47 写死 `ValueNetwork()`；形状不符只打印 `shape mismatch` **不报错** |
| `code/my_ai/entity_transformer/cpp_test/check_incumbent_value.py` | **现成的"现任值网尺子"**：`load_three_models(posnet_A_k5_m32.pt)` 在 `lowq_f64` 前 2 万行上打印 raw/tanh 的 min/max/std 与 `R2 vs label` | 只有 48 行，最容易改写成"学生 vs 老师"对比 |

---

## 3. 现有 checkpoint 战绩

`checkpoints/` 实测（`torch.load` + `count_parameters`）：

### 3.1 值网相关

| 文件 | 形态 | 架构 | 来历 | 已知质量读数 |
|---|---|---|---|---|
| **`valnet_A2.pt`** | 三网 ckpt（`class_state`/`pos_state`/`value_state` 各 94 张量 / 554,990 元素） | `AntWarNetwork(latent_dim=64, num_resblocks=6, num_heads=3, no_bn=False, value_pool=gap)`；`value_head.0.weight (64,128)` | `az_intent/train_value_net.py`，`valnet_src=az_fixed/three_mix_r10p_vw_pol_frozen.pt`，`valnet_label=terminal`，`valnet_epoch=2`，`valnet_cache=inject_ex02/value_cache` | ① 1-ply argmax 在官方启发式里的**分位 56.4%→61.2%**（SE 0.3，z≈12.5），**Spearman ρ 0.128→0.315**；② 有塔样本 MSE 0.06325→0.05794；③ **实战「只换价值网」18.4%→32.4%（+14.1pp，p=0.0003，256 局）** —— 项目历史上**最强的单条值网收益**（`docs/experiment_log.md:3547`） |
| `posnet_A_k5_m32.pt` | 同上 | 同上 | `train_value_prior.py` 产物；**`value_state` 是更早 value 线的网** | 当前搜索（M6c-ROOT / flat rollout）用的就是它的 `value_state`；在 `lowq_f64` 终局标签上 **R²=0.4533**（`docs/experiment_log.md:11374`） |
| `posnet_A_k5_m32_lowq.pt` | 同上 + `value_graft` 键 | value_state 换成 `lowq_value_seed0` | `graft_value_ckpt.py`（84/84 张量同名拷贝） | 128 局配对 vs rule_v4 = **0.7734**（95%CI [0.671,0.876]）vs 基线 0.7344 ⇒ **+3.9pp，CI 覆盖基线，不显著**（`experiment_log.md:11334-11337`） |
| `lowq_value_seed1.pt` / `seed2.pt` | 独立 `ValueNetwork`，84 张量 / 544,590 | `ValueNetwork(64,6)` = **542,913 参数** | `train_value_stream.py --data lowq_f64 --seeds 0 1 2` | **val R² = 0.4701 / 0.4688**（seed0=**0.4708**，但 ⚠️ **seed0 不在 `checkpoints/`**），MAE≈0.215；3 种子间只差 **0.002** |
| `value_s2.pt` | 独立 `ValueNetwork` | 542,913 | `train_value_net.py --stream`（目标=终局胜负） | `best_mae=0.0338` —— ⚠️ **按帧切分 ⇒ 泄漏**（文档明确：按局切是 0.3453，差 10 倍） |
| `value_search_net.pt` | `AntWarNetwork.model_state`（94/554,990） | 64/6/3 | `train_value_antwar.py --save-net` | `val_mae = 0.0718`（目标 = 搜索回传 `root_value`） |
| `grafted_sdk_A_vs2.pt` / `_vsearch.pt` | 三网 ckpt，只换 `value_state` | 64/6/3 | `train_value_antwar.py` | 按局 `val_mae` **0.3453**（终局胜负）/ **0.0718**（搜索值）；`_vsearch` 在"MC 评分器"角色读数 **0.7188（当时最好）**，但随后 4 个闸门把"价值网当仲裁"整体否决（`docs/action_net_retrain_index_raw.md:195-201`） |
| `entity_value_128_reduced.pt` | 实体 Transformer，54 张量 / 463,233 | `d_model=128, n_layers=3, n_heads=4, use_cells=False` | `entity_transformer/train.py --data entity_ds_128.npz`（train 96 局 / val 32 局） | **val R²=0.9969，MAE=64.9**（线性基线 0.9459/348）；归一化 `value = out*1750.318 + 10481.754`；⚠️ **训练未过 tanh**，并入 `AntWarNetwork` 体系须 `AZAI_VALUE_TANH=0` |
| `entity_sim_value_small.pt` | 实体 Transformer，42 张量 / **130,817** | `d_model=64, n_layers=2, n_heads=4` | `train.py --data sim_value_ds_128.npz` | **val R²=0.9987，MAE=41.1**；目标=复刻亚军手写价值，**不是更强** |
| `lowq_ent64_terminal.pt` | 实体 Transformer，463,233 | 同 `entity_value_128_reduced` | `train.py --data lowq_f64_ent.npz` | **val R²=0.4717，MAE=0.2149** |
| `q8_value.pt` / `q8_value_resid_z.pt` / `q8_ent_big.pt` | MLP `state_dim=63, plan_dim=41` | 非 board 输入 | `entity_transformer/train_q8_value.py` | Q8 计划价值，**与 board 值网不可互换** |

> ⚠️ `docs/checkpoints.md:258` 写 `lowq_value_seed{0,1,2}.pt` 三个都在库，**实测 `checkpoints/lowq_value_seed0.pt` 不存在**（只有 seed1/seed2）。要用 seed0 得去 `training_history/runs/20261019_131857_lowq_value_f64_3seed/` 或快照目录找。

### 3.2 参数规模基准（实测 `count_parameters()`）

```
ValueNetwork(latent_dim=64, blocks=6) =   542,913     ← 现状（独立小网形态）
ValueNetwork(latent_dim=32, blocks=3) =   103,137     ← 目标
ValueNetwork(latent_dim=32, blocks=2) =    84,577     ← create_small_value_network()
ValueNetwork(latent_dim=64, blocks=3) =   320,961
AntWarNetwork(latent_dim=64, blocks=6, heads=3) =   553,313   ← 现状（三网 value_state 形态）
AntWarNetwork(latent_dim=32, blocks=3, heads=3) =   107,361   ← 任务描述里的 "≈107k"
```

⇒ 任务里的 **"约 107k" 对应 `AntWarNetwork(32,3,3)`=107,361**；若走独立 `ValueNetwork`，32/3 是 **103,137**。两者相差 4k（策略头），**前向成本几乎全在主干上**，所以"107k vs 553k ⇒ 4.9×"这个估算对两种形态都成立。

### 3.3 前向耗时（实测，CPU，`torch.set_num_threads(2)`，batch=64）

```
torch 2.12.0.dev20260408+cu128  threads 2  cuda_avail True
config                                ms/call     params
ValueNetwork(64,6) batch=64            227.39    542,913     ← 与"实测 201 ms / batch64"同量级
ValueNetwork(32,3) batch=64             40.69    103,137
ValueNetwork(64,6) batch=32            218.63    542,913
ValueNetwork(32,3) batch=32             22.97    103,137
=> batch=64 speedup 64x6 -> 32x3 : 5.59x
=> batch=32 speedup 64x6 -> 32x3 : 9.52x
```

（测量时机器上还有 8 局对局在跑，只用了 2 线程；绝对值偏慢，**但比值可信**：**5.6–9.5×**，与任务里的 4.9× 一致。）

---

## 4. 蒸馏/重训可行性结论

### 4.1 一句话

**可以用现有数据 + 现有脚本（改 3 行）训出 32/3 小值网；数据集选 `lowq_f64`（320k 行，6.2 GB）；损失用 MSE；评估用「按局切的 val R²」+「与现任值网的 Spearman」。**

### 4.2 数据：用哪个

| 候选 | 样本量 | 标签 | 优点 | 缺点 |
|---|---|---|---|---|
| **`lowq_f64/` ★ 首选** | **320,000**（5000 局 × 64 帧） | `label.npy` = 终局血量差/20 | ✅ 6.2 GB 可全量进内存；✅ **已有 64/6 的同口径基线 R²=0.4701、3 种子稳定（±0.002）**；✅ 已有配套 eval 脚本（`eval_value_ckpt.py` 固定留出集） | 弱策略分布（M6c+ε2%）；局内标签恒定（一局一个值） |
| `lowq_f256/` | 1,280,000 | 同上 | 4× 样本 | 24.7 GB，训练慢 4×；历史证明**帧数对 terminal 标签没用**（全量帧 R²=0.4672 ≤ 64 帧 0.4701） |
| `lowq_full/` | 4,280,296 | 同上 + 10 个 rel 标签 | 最大 | 82.5 GB，单 epoch ≈ 89 min（64/6，实测 5332 s）；**没必要** |
| `entity_ds_128_board.board.npy` + `entity_ds_128.npz` | 127,578 | **亚军手写价值（~1e4 量纲）** | ✅ 唯一"128 局 + board"组合；✅ 目标函数**可表达性极高**（实体 Transformer R²=0.9969） | ❌ **标签不是当前搜索的口径**：历史实测"用亚军手写价值替换" 128 局只 **0.3906**（基线 0.7344，`experiment_log.md:11392`）⇒ **训它是做"容量验证"，不是做"替换"** |
| `value_ds_runnerup_selfplay.npz` | 32,768 | 亚军手写价值 | 单文件、S2 可直接吃 | 只有 **32 局**（1,024 行/局，局内标签恒定）⇒ 独立标签只有 32 个，验证集一换种子读数就跳 |

> **不需要重新生成数据**：上述标签全部已落盘。
> **唯一"没有现成数据"的场景**：如果你想直接**蒸馏现任值网的原始输出**（见 4.4 方案 B），需要一个"用现任值网给 `lowq_f64` 的 320k 帧打标"的步骤 —— 这只是**纯推理**（无对局、无引擎），仓库里没有现成脚本，但可照 `cpp_test/check_incumbent_value.py` 改 15 行（它已经在做同一件事，只是只跑了前 2 万行）。

### 4.3 脚本：用哪个 + 要改哪里

**用 `code/my_ai/entity_transformer/train_value_stream.py`（S1）**，因为它：
① 已经按**局**切验证集（防泄漏，项目里 `value_s2.pt` 就是踩了按帧切的坑）；
② 已经能**多种子**跑（历史教训：单种子在 ±20pp 上无效）；
③ 已经有 `--label-file`（换标签不用改代码）；
④ 内存/IO 已经调好（整片 `np.array` 批量预读，避开 memmap 惰性视图陷阱）。

**需要的最小改动（3 处 + 2 个参数）**：

```python
# ① L55 之后加两个参数
ap.add_argument("--width",  type=int, default=64, help="ValueNetwork latent_dim")
ap.add_argument("--blocks", type=int, default=6,  help="ValueNetwork num_resblocks")

# ② L164   model = ValueNetwork().to(dev)
model = ValueNetwork(latent_dim=a.width, num_resblocks=a.blocks).to(dev)

# ③ L221 与 L229  的 "config": {"latent_dim": 64, "num_resblocks": 6}
"config": {"latent_dim": a.width, "num_resblocks": a.blocks},
```
（`eval_value_ckpt.py:62` 的 `ValueNetwork()` 与 `graft_value_ckpt.py:47` 的 `ValueNetwork()` 也要同步改成读 `config`，否则**评估/嫁接环节直接崩或静默 0 覆盖**。）

**损失用哪个**：**MSE**（`nn.MSELoss`）。理由：
- 脚本已实现在**z-score 标签**上算 MSE（`(y−y_mean)/y_std`），terminal 标签 std=0.386 ⇒ z 化后 ≈ unit scale，梯度健康；
- 历史用过 MSE / `MSE(tanh(raw), target)` / 逐行位置 CE；值网这条线一直是 MSE，换成别的**没有证据收益**；
- ⚠️ 若改用 `label_geo_*`/`label_kgeo_*`（相对标签），历史是 **−13pp**（128 局 0.6016 vs 0.7344）⇒ **不要**。

**评估用哪个**：
1. **主指标**：`train_value_stream.py` 自带的**按局切 val R²**（最后 500 局 = 32,000 行）。同一口径下已有对照：朴素基线 0.3932 / 现任网 0.4533 / 64/6 新网 **0.4701**。**小网目标：R² ≥ 0.45（不低于现任 0.4533 太多）**。
2. **替换安全性指标（最关键）**：**小网 vs 现任值网的 Spearman ρ**，在**同一批留出行**上算，口径取部署侧实际消费的量 `tanh(raw)`。
   - 仓库里**没有**现成的"学生 vs 老师 Spearman"脚本（`grep -r spearman code/` 只命中 `train_prior_distill.py:37` 等**先验**相关脚本）；
   - 但可以**照抄** `code/my_ai/train_prior_distill.py:37` 的 `spearman()` 实现，配 `cpp_test/check_incumbent_value.py` 的加载方式（`load_three_models(ckpt)[2]` → `tanh(value)`），**约 20 行**；或直接算 `np.corrcoef(rank(student), rank(teacher))`。
   - **建议门槛：ρ(tanh 学生, tanh 老师) ≥ 0.95**；低于这个就别谈"无损替换"。这条门槛来自 `docs/dl_feasibility_20261026.md:105` 对"叶子值网"的判据（"与真实 rollout 值 Spearman ≥ 0.8"，我们对"与老师一致"要求更严）。
3. **可选**：`eval_value_ckpt.py --data lowq_f64 --val-games 500`，保证与历史读数**逐行同集**可比。

### 4.4 两条具体可行路径

**方案 A（今天的、零新数据、最稳）—— 同标签重训**

用 `lowq_f64` 的 terminal 标签，把 64/6 换成 32/3。**判据：按局 val R² 不要明显低于 0.4701（比如 ≥0.455），且 Spearman(小网, 现任网) ≥0.95。**
优点：完全复用历史基线与评估集，可解释性最好。
风险：从 542,913 → 103,137 是 **5.3× 容量缩减**；`lowq` 标签的 R² 天花板是 0.47（文档结论：**标签性质决定，不是容量不足**），所以**容量缩减大概率掉得很少** —— 这本身就是值得一测的问题。

**方案 B（更贴近目标、需 15 行新代码）—— 老师输出蒸馏**

1. 用 `load_three_models(checkpoints/posnet_A_k5_m32.pt)[2]`（或 `lowq_value_seed0`）在 `lowq_f64`（或 `lowq_full`）的 320k 帧上跑一遍，存 `teacher_tanh.npy`（float32，1.3 MB）。纯推理：GPU 上 320k 帧 ≈ 秒级；CPU 单线程 ≈ 320k × 3.5 ms ≈ **19 min**（用 batch 512 并行会快很多）。
2. 把 `train_value_stream.py --label-file teacher_tanh.npy` 直接喂（把 `label-file` 指向它即可，脚本会 `y_mean/y_std` 归一化后算 MSE）。
3. 判据同上（ρ ≥ 0.95，MSE on tanh 尺度尽量小）。

> 方案 B 的好处：**目标就是"别改变搜索决策"**，指标直接对齐目标；坏处：目标受老师上限约束（老师本身 R² 只有 0.47）。历史上"拟合得更准 ⇒ 棋力更差"出现过 5 次（`docs/dl_feasibility_20261026.md:49-55`），所以**离线指标只当筛选，最终必须跑 128 局配对**。

### 4.5 一条可直接复制运行的命令（方案 A，含预计耗时）

```powershell
# 前置：先按 §4.3 给 train_value_stream.py 加 --width/--blocks（3 处改动）
$env:PYTHONIOENCODING="utf-8"
D:\anaconda3\envs\pytorch-gpu\python.exe -u code/my_ai/entity_transformer/train_value_stream.py `
  --data D:/2026智能体大赛_新2_distill/training_history/distill_top2/lowq_f64 `
  --label-file label.npy `
  --width 32 --blocks 3 `
  --seeds 0 1 2 --epochs 10 --batch-size 512 --shard-games 50 --val-games 500 `
  --out-dir D:/2026智能体大赛_新2_distill/checkpoints `
  --log-file D:/2026智能体大赛_新2_distill/training_history/distill_top2/train_small32_3.log
```

**预计耗时**：
- **直接实测锚点**：同一命令、**64/6**、3 种子 × 10 epoch = **2,781 s（46 min）**（`docs/experiment_log.md:11311`，`- **exit**: 0，用时 2781s`）。
- 32/3 的每步计算量约为 1/5（前向实测 5.6×），且数据只有 6.2 GB（整片顺序读，非瓶颈）⇒ **估计 10–18 min**（3 种子 × 10 epoch 全部）。
- 若 GPU 被别的训练占着（当前 `nvidia-smi` 显示 282 MiB / 12% 占用），耗时按竞争程度线性放大。

**训完立刻能跑的评估**（同一留出集，与历史同口径）：
```powershell
D:\anaconda3\envs\pytorch-gpu\python.exe -u code/my_ai/entity_transformer/eval_value_ckpt.py `
  --ckpt D:/2026智能体大赛_新2_distill/checkpoints/lowq_value_seed0.pt `
  --data D:/2026智能体大赛_新2_distill/training_history/distill_top2/lowq_f64 `
  --label-file label.npy --val-games 500 --device cpu
```
（⚠️ `eval_value_ckpt.py:62` 写死 `ValueNetwork()`，同样要改成读 ckpt 的 `config`。）

---

## 5. 风险与坑（按"会直接毁掉一次实验"排序）

### 5.1 🔴 形状耦合：小值网**塞不进**现有三网 ckpt

`az_selfplay.load_three_models()`（`code/my_ai/az_intent/az_selfplay.py:104-117`）用 **ckpt 顶层同一组** `model_kwargs_from_ckpt()`（`latent_dim` / `num_resblocks` / `no_bn` / `gn`）建**三个**网，然后 `load_state_dict(ckpt["value_state"])`：
- 想给值网单独的 `latent_dim=32`？**不支持** —— `value_no_bn`/`value_gn`/`value_value_pool` 有独立键，但 **`latent_dim`/`num_resblocks` 没有**。
- 把整个三网 ckpt 换成 32/3 ⇒ **策略网也变小**，历史证明会掉棋力（8 次动作网重训全负，`docs/dl_feasibility_20261026.md:44`）。
- `graft_value_ckpt.py` 对形状不符**只打印 `shape mismatch` 不报错**（L58-62、L84-85）⇒ 会得到**一个 0 覆盖的"嫁接成功"文件**。

⇒ **唯一安全路径**：小值网走**独立 `ValueNetwork`**，在消费侧单独加载。需要适配的接口有两处（都按 `model(...)["value"]` 取值）：
- `code/my_ai/az_intent/value_batch.py:55`（`make_batched_value_fn`）
- `code/my_ai/az_intent/flat_rollout_greedy.py:284`（`_leaf_values`）

`ValueNetwork.forward` 返回**裸张量** `(B,)`（`value_network.py:77-95`），不是 dict ⇒ 直接塞会 `TypeError`。最小适配：包一层 `class Wrap(nn.Module): def forward(b,s): return {"value": inner(b,s).unsqueeze(-1)}`，或改两处为 `out = m(b,s); raw = out["value"] if isinstance(out, dict) else out`。

### 5.2 🔴 `stats` 语义/退化：不是每个 npz 的 `stats` 都能用

- `rollout_ds_test.npz` 的 `stats` **退化**：前 4 维非零，其余 38 维全 0（亚军单边 dump 只给了能提供的字段）⇒ 用它当验证集，跨局 val R² 只有 **0.624**（`docs/entity_transformer_value_results.md:26`）。**别拿它当留出集**。
- `sim_value_ds_128.npz` 同样是"单边投影"，只用于训练 `entity_sim_value_small`。
- `lowq_f64 / lowq_f256 / lowq_full / entity_ds_128 / value_ds_*` 的 `stats` 都是 42 维（`stats_mlp.0.weight` 形状 `(64,42)` 已核），来源是同一个 `FeatureExtractor.encode_observation`（与部署侧同一函数、同一 dump 链路）。⚠️ 但**没有**逐维比对"数据集里的 stats"与"部署时实时 encode 出来的 stats"（那需要跑对局，超出只读范围）—— 换数据集前建议抽 10 行做一次对比。

### 5.3 🔴 归一化 / tanh 口径：三处必须一致

- value 头是**裸线性**（`network.py:184-186` 注释：故意不加 tanh，因为目标可能超出 [-1,1]）。
- 部署侧**默认开 tanh**：`az_bridge_ai.py:472` `AZAI_VALUE_TANH` 默认 `"1"`；`value_batch.py:20` `tanh=True`；`flat_rollout_greedy.py` 同。
- `train_value_stream.py` / `train_value_distill.py` 都在 **z-score 标签**上训练 ⇒ 网输出是 z 尺度，**不是** tanh 前尺度。**必须把 `y_std/y_mean` 折进 `value_head` 最后一层**（`graft_value_ckpt.py:68-76` 已实现）。
- 历史 bug：漏了这一步 ⇒ 输出被放大 `1/y_std` 倍，terminal **2.6×**，rel **13×（直接让搜索失效）**（`docs/experiment_log.md:11399-11401`）。

### 5.4 🟠 BN 从未训练过

`three_mix_r10p_vw_pol_frozen.pt`（`valnet_A2` 的源、`posnet_A_k5_m32` 同一血统）的 `value_state` **所有 BN 都是初值**（`running_mean=0`、`running_var=1`、`num_batches_tracked=0`）⇒ 在 `eval()` 下等于恒等。对照：同 ckpt 的 `class_state` BN 是训过的（`num_batches_tracked=21475`）。
**后果**：直接 `model.train()` 微调会把 `resblocks.5.bn1.running_var` 从 1 推到 **1.47e4**，"val MSE 变好"里混进"BN 适配新分布"。
**用法**：`az_intent/train_value_net.py --freeze-bn` **默认开**（`_set_bn_eval`），别关。
（`ValueNetwork` 从零训不受影响 —— 它的 BN 本来就会随训练更新。但**跨网搬权重**时要注意两套 BN 语义不同。）

### 5.5 🟠 相对标签（`rel`/`kgeo`）部署要额外开关，且历史为负

- 语义：`label_t = 指数加权未来血量差均值 − 当前帧血量差`（`build_rel_labels.py:8-10`）。
- 部署必须 `AZAI_VALUE_TANH=0 + AZAI_REL2ABS=1.0`（`az_selfplay.py:187-188`：`value = value/rel2abs + stats[1]/HP_SCALE`）。
- **实战 128 局：`rel geo_tau50` = 0.6016**，比基线 0.7344 **低 13pp**（CI 不重叠）；与历史"rel 128 局 −7.0pp"方向一致（`experiment_log.md:11391-11395`）。⇒ **不要用相对标签**。

### 5.6 🟠 按帧切分会严重泄漏

同一个数字出现两次独立记录：
- `grafted_sdk_A_vs2`：按帧切 `val_mae 0.0338` vs 按局切 `0.3453`（**差 10 倍**，`docs/checkpoints.md:165`）；
- `value_s2.pt` 的 `best_mae=0.0338` 就是按帧切的产物 ⇒ 文档自己标注"只用它做实验记录"。
⇒ **一律按 `game_offsets.npy` / `game_id.npy` 切局**。

### 5.7 🟠 memmap 惰性视图会慢 30 倍

`np.asarray(ndarray_memmap)` 拿到的是**惰性视图**，训练中途逐 4 KB 缺页读 ⇒ 实测 **~56 MB/s**；改 `np.array(...)` **强制批量预读 1.5–2.5 GB/s**（`experiment_log.md:11402`，`train_value_stream.py:176-182` 已实现）。
另外磁盘类型：**D=SSD，E/F=HDD**；"散读帧"vs"整窗口顺序读"在机械盘上实测差 **30×**（抽帧 50 min → 3 min）。`lowq_full` 在 D 盘也要**顺序**读。

### 5.8 🟠 大文件"大小正确、内容零填充"

robocopy 被杀后会留下一个**大小完全正确但内容全零**的 `board.npy`，导致**一整轮训练报废**（`experiment_log.md:11403`）。修法：`copy_large.py`（分块 + 进度日志 + **抽样 sha256**）；结论是**"大文件只核对大小不够"**。
本次已抽查 `lowq_f64`(行 0/100k/200k/300k)、`lowq_full`(行 0/2M/4M)、`entity_ds_128_board`(行 0/60k/127k) 的 `nonzero_frac` 均 ≈0.10，**当前这三份都没被零填充**。但任何新拷的大文件都要重跑这个抽查（§6.2 的脚本可直接用）。

### 5.9 🟡 小数据集当留出集 = 噪声地板

- `value_ds_12g.npz`（12 局，`--val-seeds 108-111` 只剩 **4 局**）、`value_ds_ent8.npz`（8 局）、`_smoke*.npz`（1 局）—— **不能当验证集**。
- 历史教训：336 局时代"同配置换种子胜率 25%→50%"；47pp 的座位差；16 局 ±25pp ⇒ 判据至少 **128 局**（`docs/dl_feasibility_20261026.md:92`）。5000 局后 3 种子 R² 只差 **0.002**，这是"训练已稳定"的直接证据。

### 5.10 🟡 标签的"局内方差"决定帧数够不够（选数据集时必看）

| 标签 | total std | **局内 std** | 局间 std |
|---|---|---|---|
| terminal（终局血量差） | 0.384 | **0.0000** | 0.3665 |
| geo_tau10 | 0.0356 | 0.0354 | 0.0048 |
| geo_tau50 | 0.0760 | 0.0722 | 0.0198 |
| kgeo_tau50 | 0.1143 | 0.1042 | 0.0383 |

⇒ **terminal 标签一局只有一个值 ⇒ 64 帧/局就够**（全量 856 帧/局反而 R² 更低：0.4672 < 0.4701）；
⇒ **rel/kgeo 标签的方差 94–96% 在局内 ⇒ 必须全量帧**（64 帧只用掉 7.5% 的信号，历史已因此停止该实验）。
（`experiment_log.md:11342-11350`）

### 5.11 🟡 文档里一处**互相矛盾**，以代码/实测为准

`merge_lowq_data.py:12` 的注释写 `value_win.npy # ±0.5 终局胜负`，但：
- `train_value_stream.py:107-111` 的注释说它"**就是局末血量差/20**，平局且分胜负给 ±0.1"，且**明确警告不要用 `hp_us/hp_opp` 现算**；
- 实测 `value_win.npy`：min −1 / max 1 / **std 0.3757** / p5 −0.6 / p95 0.6 —— **不是 ±0.5 的胜负数**（胜负数只会有 ±0.5/0 三个值）。

⇒ **以 `train_value_stream.py` 的注释 + 实测分布为准**（血量差/20）。这类"注释过期"的坑，项目 CLAUDE.md 里也提醒过。

### 5.12 🟡 只读约束下**没做**的验证

- 没有跑任何对局、没有启动训练、没有 kill 任何进程；只跑了若干**只读**探针 + 1 个 2 线程的短基准（单次最长 < 1 min，CPU 峰值 2 线程）。
- 没有验证 `lowq_*` 与**当前部署 ckpt** 的局面分布重叠度（需要跑对局或用 rollout dump，超出只读范围）。
- `entity_ds_128_board.board.npy` 的**生成脚本未找到**（全仓无写端），只验证了它与 `entity_ds_128.npz` 逐行对齐。

---

## 6. 复现本文数字（自包含）

本文所有数字都由下面 4 段**只读**探针产出。把 `$PY` 换成 `D:\anaconda3\envs\pytorch-gpu\python.exe`，`$env:PYTHONIOENCODING="utf-8"`。

### 6.1 数据集 shape/dtype/统计量（§1）

```python
# probe_ds.py  —— 只读；大 npy 一律 mmap，每个文件最多前 200,000 行算统计量
import os, numpy as np
ROOT = r"D:\2026智能体大赛_新2_distill\training_history\distill_top2"

def line(a, n=200_000):
    if a.ndim == 0 or a.dtype == object:
        return "scalar/object"
    sub = np.array(a[:min(a.shape[0], n)])
    f = sub.reshape(-1)
    f = f[np.isfinite(f)] if f.dtype.kind == "f" else f
    s = "min=%.4g max=%.4g mean=%.4g std=%.4g" % (f.min(), f.max(), f.mean(), f.std())
    if a.ndim == 2 and a.shape[1] <= 48:                      # 2-D 小尾维打印逐列均值
        cm = np.nanmean(sub.astype(np.float64), axis=0)
        s += " | colmean=" + ",".join("%.3g" % x for x in cm)
    return s

for fn in sorted(os.listdir(ROOT)):
    p = os.path.join(ROOT, fn)
    if not os.path.isfile(p):
        continue
    if fn.endswith(".npy"):
        a = np.load(p, mmap_mode="r")                          # ← 关键：不整份读进内存
        print("%-32s (root) %-26s %s" % (fn, list(a.shape), line(a)))
    elif fn.endswith(".npz"):
        z = np.load(p, allow_pickle=True)
        print("### %-30s %.1f MB" % (fn, os.path.getsize(p) / 1e6))
        for k in z.files:
            a = z[k]
            print("   %-14s %-30s %s" % (k, "%s %s" % (list(a.shape), a.dtype), line(a)))
```

`lowq_*` 目录里的**一维标签**（`label*.npy` / `value_win.npy`）体积都 < 20 MB，直接**全量**读（`np.load`，不 mmap）取 min/max/std/p5/p95。

### 6.2 board memmap 完整性 + 与 npz 的对齐（§1.1 注①、§5.8）

```python
# probe_board.py —— 只读；防"大小正确、内容零填充"
import numpy as np, os
ROOT = r"D:\2026智能体大赛_新2_distill\training_history\distill_top2"

def probe(path, rows):
    a = np.load(path, mmap_mode="r")
    print("%s %s %s" % (os.path.basename(path), list(a.shape), a.dtype))
    for r in rows:
        blk = np.array(a[r:r + 64])
        print("   rows %8d nonzero_frac=%.4f" % (r, float((blk != 0).mean())))

probe(os.path.join(ROOT, "lowq_f64", "board.npy"), [0, 100_000, 200_000, 300_000])
probe(os.path.join(ROOT, "lowq_full", "board.npy"), [0, 2_000_000, 4_000_000])
b = np.load(os.path.join(ROOT, "entity_ds_128_board.board.npy"), mmap_mode="r")
probe(os.path.join(ROOT, "entity_ds_128_board.board.npy"), [0, 60_000, 127_000])

# 对齐：ch4=己方塔 / ch5=敌方塔 的格子数应 == tower_mask 之和
z = np.load(os.path.join(ROOT, "entity_ds_128.npz")); cnt = z["tower_mask"].sum(1)
for i in np.where(cnt > 0)[0][:6]:
    s = np.array(b[i]).astype(np.float32)
    print("row %6d ch4=%d ch5=%d sum=%d | tower_mask=%d"
          % (i, (s[4] > .5).sum(), (s[5] > .5).sum(),
             (s[4] > .5).sum() + (s[5] > .5).sum(), cnt[i]))
```

### 6.3 参数量 + checkpoint 元数据（§3）

```python
# probe_ckpt.py —— 只读；重点看 value_state 的张量数与 value_head 的形状
import sys, json, torch
from pathlib import Path
REPO = Path(r"D:\2026智能体大赛_新2_distill")
sys.path[:0] = [str(REPO / "Ant-Game"), str(REPO / "code")]
from my_ai.network import AntWarNetwork
from my_ai.value_network import ValueNetwork

for lat, nb in [(64, 6), (32, 3), (32, 2)]:
    print("ValueNetwork(%d,%d) = %d" % (lat, nb, ValueNetwork(lat, nb).count_parameters()))
for lat, nb, nh in [(64, 6, 3), (32, 3, 3)]:
    print("AntWarNetwork(%d,%d,%d) = %d" % (lat, nb, nh,
          AntWarNetwork(num_resblocks=nb, num_heads=nh, latent_dim=lat).count_parameters()))

for p in sorted((REPO / "checkpoints").glob("*.pt")):
    ck = torch.load(p, map_location="cpu", weights_only=False)
    if not isinstance(ck, dict):
        continue
    meta = {k: v for k, v in ck.items() if isinstance(v, (int, float, str, bool))}
    vs = ck.get("value_state") or ck.get("state_dict")
    n = sum(v.numel() for v in vs.values()) if isinstance(vs, dict) else 0
    print("%-34s %.2f MB  keys=%s  value_elems=%d" % (p.name, p.stat().st_size / 1e6, list(ck)[:6], n))
    for kk in ("value_head.0.weight", "initial_conv.0.weight"):
        if isinstance(vs, dict) and kk in vs:
            print("      %-22s %s" % (kk, tuple(vs[kk].shape)))
```

### 6.4 前向耗时（§3.3）

```python
# bench_value.py —— 只读；2 线程，避免抢占正在跑对局的 CPU
import sys, time, torch
from pathlib import Path
REPO = Path(r"D:\2026智能体大赛_新2_distill")
sys.path[:0] = [str(REPO / "Ant-Game"), str(REPO / "code")]
torch.set_num_threads(2)
from my_ai.value_network import ValueNetwork

def bench(lat, nb, bs, iters=6, warm=2):
    m = ValueNetwork(latent_dim=lat, num_resblocks=nb).eval()
    b, s = torch.randn(bs, 28, 19, 19), torch.randn(bs, 42)
    with torch.no_grad():
        for _ in range(warm): m(b, s)
        t0 = time.perf_counter()
        for _ in range(iters): m(b, s)
    return (time.perf_counter() - t0) / iters * 1000

for bs in (64, 32):
    a, c = bench(64, 6, bs), bench(32, 3, bs)
    print("batch=%d  64x6=%.2f ms  32x3=%.2f ms  speedup=%.2fx" % (bs, a, c, a / c))
```

### 6.5 文档/代码出处自查

```powershell
# 本文引用的每个 (文件:行号) 都可这样核（行号按 UTF-8 物理行）
$env:PYTHONIOENCODING="utf-8"
D:\anaconda3\envs\pytorch-gpu\python.exe -c @"
import io
p = r'D:\2026智能体大赛_新2_distill\docs\experiment_log.md'
ls = io.open(p, encoding='utf-8', errors='replace').read().split('\n')
for n in (11308, 11311, 11329, 11336, 11343, 11350, 11374, 11391, 11394, 11399, 11402, 11403, 3547):
    print(n, '|', ls[n-1][:120])
"@
```

> ⚠️ 引用行号时**不要用 `Get-Content` 的索引**：本仓库部分 md 含多字节字符，PowerShell 的默认解码会让行号整体偏移（实测 `docs/entity_transformer_value_results.md` 用 `Get-Content` 读出 33 行，实际 **44 行**）。以 `read` 工具 / Python 的 UTF-8 行号为准。

---

## 7. 建议的下一步（给主 agent）

1. **先跑方案 A**（§4.5 那条命令，约 10–18 min）：拿 `lowq_f64` + terminal 标签训 32/3 × 3 种子，看 val R² 掉多少。
2. **同时补一个 20 行的"学生 vs 老师 Spearman"脚本**（照 `train_prior_distill.py:37` 的 `spearman()` + `cpp_test/check_incumbent_value.py` 的加载方式），这是**唯一能回答"换网会不会改变决策"的离线指标**。
3. **容量验证（可选、便宜）**：在 **128 局那份数据**（`entity_ds_128_board.board.npy` 的 board + `entity_ds_128.npz` 的 `stats`/`value`/`seed`）上再训一个 32/3，看能不能达到 R²≈0.99。⚠️ 注意 `train_value_distill.py` 要求 `board` 与 `value` 在**同一个 npz** 里，而这里两份是分开的（2.4 GB board 在 `.npy`）⇒ 需要 ~30 行的自定义 loader 或先合成一个 npz（`board` float16 占 2.4 GB 内存，本机 16 GB 够但要小心）。若 32/3 也能到 R²≈0.99，就**彻底排除"容量不够"这个解释**。
4. 离线过关后，**必须**跑 128 局配对（`AZAI_CKPT` 不变、只换叶子值网）—— 历史 5 次"离线更准、实战更差"（`docs/dl_feasibility_20261026.md:49-55`），离线指标不能当结论。
