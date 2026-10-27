# 重要 checkpoint 清单（`checkpoints/`）

> 2026-10-06 建。**为什么有这份东西**：checkpoint 以前**全在 `.gitignore` 里**（`training_history/` 等被整目录忽略）
> ⇒ 既不进 git、也不进镜像、**只存在本机**，丢了没法从仓库复原；而"钉死 ckpt"是按**路径**钉的、不是按内容，
> 也没法核对"手头这份是不是当初那份"。
>
> **做法**：把**少量关键 ckpt 真拷进 `checkpoints/`**（这个目录**不在 `.gitignore` 里** ⇒ 会被 commit + 推镜像），
> 并在这里记下**来源路径 + 大小 + sha256 + 是什么**。
>
> ⚠️ **代价（务必记住）**：`.pt` 一旦 commit 就**永久留在 git 历史**（以后删文件也删不掉历史里的那份）
> ⇒ **只放真正关键的**；每个 ~6.5 MB 都是**一次性、永久**的成本。
> **大文件一律不入库**（GitHub 单文件硬限 100 MB）——只在这儿记哈希。

## 1. 已入库（`checkpoints/`）

| 文件 | 来源路径 | 大小 | 是什么 / 为什么留 |
|---|---|---|---|
| `posnet_A_k5_m32.pt` | `training_history/vprior/posnet_A_k5_m32.pt` | 6.5 MB | **钉死 ckpt**：当前协议的 `AZAI_CKPT`，**第一里程碑（M6c-ROOT）用的就是它**。血统见 `docs/next_step_survey_20261005.md` §1 |
| `three_mix_r10p_vw_pol_frozen.pt` | `training_history/az_fixed/three_mix_r10p_vw_pol_frozen.pt` | 6.5 MB | **三网拆分 ckpt**（A 的直接上级：`make_three_net_ckpt.py` 从 `mix_r10p_vw_pol_frozen.pt` 复制"三网同源"而来）|
| `valnet_A2.pt` | `training_history/inject_ex02/valnet_A2.pt` | 6.5 MB | **价值网**：`collect_value_prior.py` 采 `vp_A_k5_m32.npz` 时用它算 `adv`（npz 的 `ckpt_meta` 指向它）⇒ **复现 A 的标签需要它** |
| `posnet_Vk50.pt` | `training_history/vprior/posnet_Vk50.pt` | 6.5 MB | 价值先验**阶梯（候选 V，k=50）**的 ckpt |
| `posnet_C_class.pt` | `training_history/vprior/posnet_C_class.pt` | 6.5 MB | **候选 C**：把价值先验从位置轴扩到**类轴**的 ckpt |
| `posnet_M1_distill.pt` | `training_history/vprior/posnet_M1_distill.pt` | 6.5 MB | **M1（搜索→策略蒸馏）**的 ckpt |
| `qhead_M2.pt` | `training_history/vprior/qhead_M2.pt` | 41 KB | **Q 头（M2）**（很小，顺手带上）|
| `gen_0120_model.pt` | **从** `training_history/ga_ss_20260730_093908/gen_0120.pt` **抽取**（见 §2）| 4.4 MB | **血统起点**（官方 `ExampleAI` 蒸馏/初始化那一代）的**纯模型**；原文件 404 MB 里 **~99% 是 GA 训练状态**（`ga_pop` 种群 + `leaderboard`），不是模型。来源程序 = `ga_ss_train.py` |
| `posnet_A1_km1_m32.pt` | `training_history/vprior/posnet_A1_km1_m32.pt` | 6.4 MB | **位置轴单变量臂①**（与 A **只差 `pos_state`**，`class_state`/`value_state` 逐张量相同）；按 `memory/project_az_posnet_v2_two_stage.md`，它的**非闪电通道是随机初始化**的对照臂 ⇒ "位置先验 vs 近似无先验"的现成材料 |
| `posnet_B_km1.pt` | `training_history/vprior/posnet_B_km1.pt` | 6.4 MB | **位置轴单变量臂②**（同样只差 `pos_state`）⇒ 与 A / A1 构成**三臂单变量对照**（当年"判据待跑"，见同一份 memory）|
| `posnet_r1.pt` | `training_history/vprior/posnet_r1.pt` | 6.4 MB | **位置先验线的"第一次可复现提升"**（打 `rule_v4` **21.1% → 35.2%**、复现 oracle 收益的 **86%**，见 `experiment_log.md` 的 `vprior_r1_eval`）——**A/A1/B 的祖先**，血统不可无 |

> 上面"是什么"是按**文件名 + 预注册命名 + 用法**写的简要定位；细节看各自的 `docs/prereg_*.md`。
>
> **子网指纹**（`class_state` 相同 ⇒ 同一祖先；`pos_state` 才是变量）：2026-10-06 核过 ——
> `A: class=777fbd733e pos=1cae1a6d1a value=2b6b00a3d3`；
> `A1: 同 class、pos=7519f635be、同 value`；
> `B: 同 class、pos=89a73d975c、同 value`；
> `r1: 同 class、pos=f0cb9b9192、value=9e28a62c14`（r1/r2 的价值网也不同）。

### 校验（"手头这份是不是当初那份"）

在 `checkpoints/` 目录下跑（**Linux / Git-Bash / WSL**）：

```bash
sha256sum -c <<'EOF'
6fc38cb8ecbe9344c74c1eecfb3fd4bfeb5267960039d720deabf94cb3f235d9  posnet_A_k5_m32.pt
4c6a246e24391aa4145ec12f60d55f22aa0faf7a645ca76a4af2d54886c14e4f  three_mix_r10p_vw_pol_frozen.pt
4c73aa98bcb7a0ca70aa296bbceaa5824e218da9cdc19113539a3b9f23de30e5  valnet_A2.pt
d340ad42f9cbc84a0ae48e4976b61473dd946c76dc28e63d8dbefaa9b65dac8d  posnet_Vk50.pt
da7fc52a78ce683d02c38176f730dda6ba16a0e319cb04236af346a7d83db05d  posnet_C_class.pt
09123a5eae525ef277f6f3288fd0e17f1cc88905daaa188838487ac9535f8aa1  posnet_M1_distill.pt
0732163b307290ac295e8eec83f86ab8732b96438904f80ae9f050cb7f9d32f6  qhead_M2.pt
ea9d1964e9f33bd54ab2f7fe8b3cdf7be6d218b928fa19bcf96c1b293f1072a1  gen_0120_model.pt
a790dad06d211c971aa0cd6d39babbc12f74cf6ee83b0eff216d744582772aa5  posnet_A1_km1_m32.pt
4ec27094a23d52d6a5469d707490d5d3b133fb3661ad0c085f87cea15352dc8d  posnet_B_km1.pt
fb2553e558fe6b201d2232680291b791505388655caf894e21991bbfd6c4c265  posnet_r1.pt
EOF
```

⚠️ **这台机器（Windows）没有 `sha256sum`**（2026-10-06 实测）⇒ 用 PowerShell 等价写法：

```powershell
cd D:\2026智能体大赛_新2
Select-String -Path docs\checkpoints.md -Pattern '^([0-9a-f]{64})  (\S+)$' | ForEach-Object {
  $want = $_.Matches[0].Groups[1].Value; $name = $_.Matches[0].Groups[2].Value
  $got = (Get-FileHash "checkpoints\$name" -Algorithm SHA256).Hash.ToLower()
  "{0,-40} {1}" -f $name, $(if ($got -eq $want) { 'OK' } else { '!! 不符' })
}
```
（2026-10-06 用上面这段核过：**11/11 一致**。）

## 2. 大 ckpt：原文件不入库，但**把模型抽出来入库**

| 文件 | 原路径 | 大小 | sha256 |
|---|---|---|---|
| `gen_0120.pt`（**原文件，不入库**）| `training_history/ga_ss_20260730_093908/gen_0120.pt` | **404 MB** | `3046e77a0b5d3431defdc8ad9d17ac1e0995500438a322dd0a8d945b27176298` |

**为什么 404 MB 而模型只有 ~2 MB**（2026-10-06 用户追问后查明；数字按**实际序列化大小**）：
文件里只有 10 个键 —— **`ga_pop`（GA 种群，96 个个体）317.97 MB（75.1%）** ＋
**`leaderboard`（其实是 20 个参数列表）98.91 MB（23.4%）** ＋ `top2_params` 2.20 MB；
**真正的模型只有 `model_state`（2.21 MB）＋ `mean`（2.20 MB）**，合计约 **1%**。

- **来源程序 = `code/my_ai/ga_ss_train.py`**（**不是** `es_train.py`；判据：目录名 `ga_ss_{ts}` 与它写出的
  `ga_pop` 键，`es_train.py` 不写 `ga_pop`）。
- **哪个键才是"真模型"**（用户 2026-10-06 特意问过）：`ga_ss_train.save_checkpoint` 里
  `model_state = model.state_dict()`（**真权重**）；加载端是 `if "model_state" in ckpt:
  model.load_state_dict(sd)`，并**由模型反推 `mean`**（`mean = model.get_parameters_as_vector()`）
  ⇒ **`model_state` 是权威权重**（`mean` 是同一套权重的向量形式；实测两者取值分布几乎一致）。
  ⚠️ 又注：该程序里的 `top2_params` 只是 **compat shim**（`mean.copy()` + 分数 1.0），不是独立精英。

⇒ **已把纯模型抽出入库** = `checkpoints/gen_0120_model.pt`（**4.4 MB**，见 §1）。抽法：

```bash
python -u code/tools/extract_model_from_ckpt.py \
  --src training_history/ga_ss_20260730_093908/gen_0120.pt \
  --out checkpoints/gen_0120_model.pt
```

- 保留 `model_state`（权威）＋ `mean`（兼容只认 `mean` 的旧脚本，如 `collect_value_*.py`）
  ＋ 小元数据；**丢弃** `ga_pop` / `top2_params` / `leaderboard`。
- ⚠️ **哈希说明**：§1 记的是**本文件**的哈希；重跑上面命令得到的内容**逐张量相同**，但
  **字节不一定完全一样**（zip 内部归档名等序列化细节可能变）⇒ 别指望重跑得到同一个 sha256。
- ⚠️ **产物用于推理 / 血统参照，不能用来续训 GA/ES**（种群与 leaderboard 已丢；要续训得留原文件）。
- **同类可复用**：`training_history/ga_ss_*/gen_*.pt` 有 ~140 个、多半是同一个"几百 MB 里只有 2 MB 是模型"的形态，
  需要哪个就同一条命令抽（`--dry-run` 可先只看各键大小）。
- 更早的一些 ckpt（旧文档**只记了路径**）见 `docs/notable_checkpoints.md`（那份已作废，保留作考古）。

## 3. 还有哪些**没**进来（要用再拷）

`training_history/vprior/` 下的 `posnet_*` 变体**大部分没入库**（不然仓库会被撑爆）。2026-10-06 已入库的
"位置轴三臂 + 祖先" = `posnet_A_k5_m32` / `posnet_A1_km1_m32` / `posnet_B_km1` / `posnet_r1`（见 §1）。

**还没入库、但可能用到的（按需再拷）**：
- **位置轴更早的变体**：`posnet_r2`（与 r1 同价值网、pos_state 不同）、`posnet_tau0.25/0.5`；
- **价值标签视野阶梯**（该轴已按验收关闭，见寄存器）：`posnet_VRrel5/10/30`、`posnet_Vrel50/100/200`、`posnet_Vk35/70`、`posnet_Vk100`；
- **类轴/蒸馏线**：`posnet_M1b_masked`、`posnet_M1c`、`posnet_M1d`、`posnet_R_class16`、`posnet_V1_val20`、`posnet_v2_valfresh`、`posnet_Q1/Q1b_lr1e4/lr3e5/lr1e4_bn`；
- `az_fixed/`、`distill_data_v2/` 里的一些 ckpt。

要用哪个：`cp <原路径> checkpoints/` ⇒ 算哈希 ⇒ 在 §1 表里加一行 ⇒ **连同文件一起提交**。
（子网指纹可先对着 §1 末尾那三行核：`class_state` 相同 = 同一祖先。）

## 4. 加新 ckpt 的规矩（4 步）

1. `cp <原路径> checkpoints/`（用**原文件名**，便于和文档/日志对应）；
2. `sha256sum checkpoints/<file>`；
3. 在本文件 §1 表**加一行**（文件 / 来源 / 大小 / 哈希 / 一句话说明）；
4. **提交时别漏了文件本身**（`git add checkpoints/<file> docs/checkpoints.md`）——
   ⚠️ 只提交文档、忘记 `git add` 二进制，是这套做法最容易犯的错。

## `checkpoints/distill_sdk_A.pt`（2026-10-18，S1b 蒸馏产物）
- **来源**：`python code/distill/train.py distill_data/ --out checkpoints/distill_sdk_A.pt --epochs 30 --lambda-soft 0`
  （数据 = `distill_data/` 400 局 / 106,694 决策；监督 = SDK 评分反投影到 (class, position)；**未用软标签**
  因为 npz 里的 `head_logits` 是**旧模型自己的输出**（自指闭环）；日志 `training_history/distill_sdk_A.log`）
- **架构**：`create_model(num_heads=3)`，与部署 ckpt 的 `class_state/pos_state/value_state` **键完全一致（94/94）**
- **终值**：epoch30 `cls=0.6715 map=0.1491 tot=0.8239`（epoch1 为 1.0039/1.0137/2.0169）
- **嫁接产物** `checkpoints/grafted_sdk_A.pt`：`class_state/pos_state` ← 蒸馏网（94/94 覆盖），
  **`value_state` 保留原 `posnet_A_k5_m32.pt`**（本轮不动价值网，S2 再单独重训）
- **冒烟（双方禁闪电、pos-only+宽菜单、2 局）**：我方 **15.0 建 / 6.5 降 / 7.5 升** / 有操作回合 22.0
  ⇒ **升级从基线 2.9 跳到 7.5/局**（example 自身 5.5–6.0）⇒ 蒸馏确实注入了经济行为

## 头消融产物（2026-10-19，goal-2600a56a 第 1–2 轮；预注册 `docs/prereg_20261019_goal2_head_ablation.md`）
| 产物 | 训练 | 终值 | 说明 |
|---|---|---|---|
| `checkpoints/distill_maponly.pt` → `grafted_maponly.pt` | `--lambda-class 0`（只训动作图）| `map 0.1308`（比完整蒸馏 0.1491 更低，容量全给图）；`cls 42.3`（类头未约束、漂移）| 用于回答"那 +42pp 来自哪个头" |
| `checkpoints/distill_clsonly.pt` → `grafted_clsonly.pt` | `--lambda-map 0`（只训类头）| `cls 0.7023`；`map 24.9`（未约束）| 同上（互补）|
| `checkpoints/distill_phold10.pt` → `grafted_phold10.pt` | `--p-hold 0.1`（HOLD 降采样 10×）| 训练中 | 目的：让**类头分布可用**，为 `joint`（类交给学习）铺路 |
> ⚠️ 解读口径：λ=0 的那一头**不受约束会漂移** ⇒ 这些臂的准确含义是"**一个头受训 + 另一个头未被约束**"，
> 不是"另一个头保持原样"。若结果含糊，应改用小 λ（如 0.05）重训一版。

## 价值网（S2 / 搜索值）相关 ckpt（2026-10-09 第 38–51 轮）

**部署可直接加载的"三网 ckpt"**（`class_state`/`pos_state` 与冻结基线 `grafted_sdk_A.pt` **逐位相同**，**只换 `value_state`**）：

| 文件 | 大小 / sha256(前 16) | `value_state` 怎么来的 | 判据读数（vs 冻结基线、MC1 口径、128 局）|
|---|---|---|---|
| `checkpoints/grafted_sdk_A_vs2.pt` | 4.29 MB / `71616f81b4e2ceef` | `train_value_antwar.py`，目标 = **该局终局胜负**（按局 `val_mae 0.3453`）| 未单跑判据；其"MC 评分器"角色读数 **0.4531**（vs example、32 对）|
| `checkpoints/grafted_sdk_A_vsearch.pt` | 4.29 MB / `03e5f83a2a8bf52e` | `train_value_antwar.py`，目标 = **搜索回传的 `root_value`**（按局 `val_mae 0.0718`）| **0.7188**（当前最好；MC1+旧网 = 0.6797）|

**独立的训练网**（**不能**被 `load_three_models` 直接加载——它是 `ValueNetwork` 小网，键名与部署 `value_state` 交集为 0；仅供离线诊断/再训练）：

| 文件 | 大小 / sha256(前 16) | 说明 |
|---|---|---|
| `checkpoints/value_s2.pt` | 2.10 MB / `b9942fb2a5b7d718` | 终局胜负目标的小网（`train_value_net.py --stream` 产物，按局验证集会泄漏 ⇒ 只用它做实验记录）|
| `checkpoints/value_search_net.pt` | 2.14 MB / `cbfc767f11783169` | 搜索值目标的小网（`train_value_antwar.py --save-net` 产物）|

> **口径提醒**：判据只能由"部署形状的三网 ckpt"取得（`grafted_sdk_A_vs2/_vsearch`）；
> 离线 `val_mae` **按局切分**才有意义（按帧切分会泄漏，实测差 10 倍：0.0338 vs 0.3453）。
> 训练数据都在**训练专用 seed 段**（`10001..10030` 第一轮、`10031..10060` 第二轮）采集，**不含**判据段/随机段/未见段。

### 路 C（joint + 类搜索）用到的**混合体**（2026-10-09 第 57 轮）

| 文件 | 组成 | 为什么 |
|---|---|---|
| `checkpoints/grafted_phold10_vsearch.pt` | `class_state`/`pos_state` ← `grafted_phold10.pt`（HOLD 降采样 10× 重训 ⇒ **类头有分布**，joint 采样才不退化）；`value_state` ← `grafted_sdk_A_vsearch.pt`（**搜索值网**，离线对搜索值 MAE 0.0718 ⇒ 叶值能看见"花钱的代价"）| C1 首次冒烟（用 `grafted_phold10.pt` 的**陈旧旧网**当叶值）出现**建塔刷屏/僵持**（第 163 回合、≈100 座塔）⇒ 换叶值重试 |

> 生成工具：`code/tools/graft_value.py`（`graft_policy.py` 的**镜像**；后者只认蒸馏格式 `model_state`，对**部署格式** ckpt 会 0 覆盖并静默出错）。
> 三方逐位核对（class/pos 与策略来源、value 与价值来源）在写出时自动打印。

## 新目标（重训动作网拿 +10pp）的产物（2026-10-11，goal 第 1–4 轮）

### 已入库的 ckpt（`checkpoints/`）
| 文件 | 大小 | sha256(前 16) | 是什么 |
|---|---|---|---|
| `distill_sdk_A_e90.pt` | 6.37 MB | `a1345fa87dfe84be` | **路线 B**：把 incumbent 动作网的来源（SDK 蒸馏）从 30 epoch 拉到 **90 epoch**（末值 cls 0.6288 / map 0.1418，优于 30 epoch 的 0.6715/0.1491）|
| `grafted_sdk_A_e90.pt` | 4.30 MB | `849a501dfefcf82e` | **路线 B 的单变量候选**：`class_state`/`pos_state` ← e90 蒸馏网；**`value_state` ← 原 `posnet_A_k5_m32`（94/94 逐位相同）**；与 incumbent 核对：class 0/94、pos 0/94、value 94/94 ✓ |
| ~~`grafted_sdk_A_posroll_s1.pt`~~ | 6.45 MB | `747801d12308b3a5` | ⚠️ **已隔离（基于坏数据）**：路线 A 第 1 版候选（`pos_state` ← 安静 rollout 教师蒸馏；class/value 不变）。⚠️ **先验低**：教师未被学到（top1 ≈ 随机）⇒ 未投判据；等"宽格采样"重采后重训 |
| （对照）`grafted_sdk_A.pt` | 4.29 MB | `8c21f44d73bcefb3` | **incumbent**（= 冻结基线本身）|
| （对照）`distill_sdk_A.pt` | 6.37 MB | `e542ff209c83810c` | 30 epoch 版蒸馏网（incumbent 的动作网来源）|

### 数据（只在盘上，不入库）
| 路径 | 大小 | 说明 |
|---|---|---|
| `training_history/posroll/pilot30.npz` | 2.33 MB | 路线 A 首次试点（30 局）：**坐标是 (x,y)**（口径错误版，保留作对照）|
| `training_history/posroll/pilot30_yx.npz` | 2.33 MB | **正确口径 (row,col)=(y,x)** + `cnt`/`weight` ⇒ 训练用这份 |
| `training_history/posroll/pilot30.npz.summary.json` | <1 MB | 信号闸门摘要：`|top1−top2|` 中位 0.100、≥0.05 占 100%、完全平局仅 16.7% |

### 脚本
- `code/my_ai/az_intent/collect_pos_rollout_teacher.py`：**安静 rollout 位置教师**采集器（宽格采样：位置图前 2 格 + 分位分散，`--max-cells` 控制格数）；
- `code/my_ai/az_intent/train_pos_rollout.py`：**自写训练循环**（只训 `pos_state`，用 `az_train.save_three_net` 写三网 ckpt；每 epoch 打印**参数变化范数**——正是排查"训练器冻结"缺的那个观测量）；
- `code/test_match/audit_pos_head_vs_teacher.py`：位置头 vs 教师的一致性审计（含 dump 语义/配对/坐标三处口径坑的注释）。

### 判据读数（进行中）
- `AB_E90_vs_baseline_vs_baseline`（固定段 7..70、128 局、8 线程、incumbent 配置、**只换 ckpt**）——见 `docs/experiment_log.md` 第 79–81 轮。

### 路线 C / C′（类头 / 跨类教师）的产物（2026-10-11）
| 文件 | 大小 | sha256(前 16) | 说明 |
|---|---|---|---|
| `grafted_sdk_A_clsC.pt` | 6.44 MB | `fe894e680be8f570` | 路线 C：只换类头，AWR-BC 到"实际执行类"，10 epoch（**过拟合**：val 0.5706@ep2-4 → 0.8442）|
| `grafted_sdk_A_clsC_e3.pt` | 6.44 MB | `52ac48edd8627ba8` | 同上，3 epoch（val 最优区）；**闸门①不达标**（见 `docs/actionnet_retrain_summary.md` §3.4）|
| ~~`grafted_sdk_A_clsrank.pt`~~ | 6.44 MB | `f5921dcd8eb30649` | ⚠️ **已隔离**：路线 C′ 第 1 版，**基于坏数据**（局面不推进 + 坐标写反）⇒ 移到 `training_history/posroll/_BROKEN_2bugs/`，**勿用** |
| `grafted_clsonly_audit.pt` | 6.44 MB | `f967c5e26f84592d` | **机制审计用**只换类头对照（class 13/94 相同、pos/value 94/94 逐位相同）|

**数据（只在盘上，不入库）**：
- `data/posroll_c_policy/POSROLL_C` + `data/posroll_c_obs/POSROLL_C`：各 60 文件（22.4 MB / 555.2 MB）——
  incumbent 配置下 seed 22000–22029 双向 dump（冒烟的 seed 7 已移出）；
- `data/posroll_c_labeled`：60 个 pkl（559.8 MB）——ingest 产物，**28,629 样本 / 85,887 (头,类) 目标**（含 `chosen_cls`/`class_mask`/`value_target`）；
- `training_history/posroll/class30.npz`（0.81 MB）：跨类教师 30 局 ⇒ **15,360 行**；
- `training_history/posroll/pilot30.npz`（2.33 MB）：格级教师 30 局 ⇒ 46,080 行。

**脚本**：`collect_class_rollout_teacher.py`、`train_pos_class_rank.py`、`eval_class_head_agree.py`、
`train_pos_rollout.py`、`collect_pos_rollout_teacher.py`、`code/test_match/audit_pos_head_vs_teacher.py`。

### 更正与隔离（2026-10-11，用户追问逼出的两个采集器 bug）
- **作废并隔离**（移到 `training_history/posroll/_BROKEN_2bugs/`）：`pilot30*.npz`、`class30.npz`、`cls_verify2*.npz`、
  `verify2.npz`、`probe512.npz`、`grafted_sdk_A_posroll_s1.pt`、`grafted_sdk_A_clsrank.pt`
  —— 它们的采集器**局面从不推进**（只对 `st.clone()` 落子）且**坐标轴写反**（`_op_matches_cell` 只放对角格）。
- **修正后重采（进行中）**：`training_history/posroll2/cell90_h1.npz`（`CELLROLL_h1_90g`：90 局、**只采头 0**、
  `--max-cells 0` 全记录、真实对局推进、seed 25000+）。依据：真实对局日志统计"出手时平均 **1.24** 个动作、
  78% 单动作"（`{0:2453,1:467,2:115,3:13}`）⇒ 头 0 覆盖 ~77% 决策点，省下的时间换成 3 倍局数。
- **诊断工具**：`code/my_ai/az_intent/diag_pos_cells.py`（真实推进 + 统一 `[x,y]` + 轴序证明 +
  成功数直方图 + 按格类别成率 + 每头决策点占比）。

### 路线 r1 配方（价值先验 relabel，in-regime）产物（2026-10-11）
| 文件/数据 | 大小 | 说明 |
|---|---|---|
| `checkpoints/grafted_sdk_A_vprior2.pt` | ~6.4 MB | **候选**：warm start 自 incumbent、**只改 `pos_state`**；教师 = 冻结价值网在钉类（经济类 0–16）合法格上的 1-ply 优势 |
| `training_history/vprior2/pilot120.npz` | 0.16 GB | 采集：120 局、**862,134 行**（≈125k 个决策）、每行 ~30.4 格；类覆盖 0–16 共 17 类（**无闪电** ✓）|
| 工具 | — | `collect_value_prior.py`（新增 `--max-class`，把 pin 限制到经济类）、`train_pos_rollout.py`、`eval_map_vs_teacher.py` |

### 阶段 1（26,718 行）扫描产物（2026-10-12）
| 文件/目录 | 说明 |
|---|---|
| `training_history/visitmarg/ckpts30k/visitsweep_*.pt` | **16 个变体**的 best-val 产物（各 ~6.4 MB，均含完整顶层键 + 通过往返加载自检 ✓）|
| **`…/visitsweep_hard_anchor10.pt`** | **入围 A**：硬标签 + 锚定 incumbent（λ=1.0）⇒ val 命中 45.07%、全局扰动 0.0559（最小 ✓）|
| **`…/visitsweep_rank1_rest.pt`** | **入围 B**：目标 = rank-1 0.5 + 其余均分 0.5 ⇒ val 命中 44.17%、扰动 0.2525 ✓ |
| `training_history/visitmarg/rows30k.npz` | 训练集：**26,718 行**（388 dump ⇒ 有效行 14.7% ✓）|
| `training_history/visitmarg/sweep_30k.json` | 16 变体完整指标表（含"全局扰动"两列 ✓）|

## 实体 Transformer 价值网（2026-10-18，详见 `docs/entity_transformer_value_results.md`）

| 文件 | 大小 | sha256 | 是什么 |
|---|---|---|---|
| `entity_value_128_reduced.pt` | 1.87 MB | `704e9b4a1c66628d7726874bcc7e49989c4bf24c157b7d0d8d38b879157dd44a` | **实体 token Transformer 价值网**（塔/蚁 token + 42 stats + CLS；`d_model=128 / n_layers=3 / n_heads=4 / use_cells=False`）。训练数据 = 128 局亚军自对弈（`training_history/distill_top2/entity_ds_128.npz`，127,578 条，train 96 局 / val 32 局按 seed 切）。**val R²=0.9969、MAE=64.9**（线性基线 0.9459/348）。归一化：`value = out*1750.318 + 10481.754`。训练日志 `training_history/runs/20261018_115520_entity_value_128_reduced/`。⚠️ 训练**未过 tanh**，若并入 `AntWarNetwork` 体系部署须设 `AZAI_VALUE_TANH=0` |
| `entity_sim_value_small.pt` | 538 KB | `ab2a2b02d6b38ee3322db04e88b9cc57ad26945195a34317d43346b5c8396110` | **可塞进亚军的单边价值网**：同一模型但只吃"**己方塔 + 敌方蚁 + base_hp + coins**"（把上面的数据投影成 sim 能提供的形态，见 `build_sim_value_dataset.py`）。`d_model=64 / n_layers=2 / n_heads=4 / pos_mode=learned`，**130,817 参数**。**val R²=0.9987、MAE=41.1**。归一化同 `value = out*1750.318 + 10481.754`。⚠️ 目标是"复刻亚军自己的手写价值"，**不是更强**；用途=验证"网络能替代亚军价值且不掉强度" |

## 低质量大数据价值网 + M6c 嫁接版（2026-10-19，详见 `docs/prereg_20261019_low_quality_value.md`）

| 文件 | 大小 | 是什么 |
|---|---|---|
| `checkpoints/lowq_value_seed{0,1,2}.pt` | 各 2.15 MB | `ValueNetwork`（与 `AntWarNetwork` value 通路**逐层同构**）：5000 局**低质量**自对弈（M6c 裸策略 + ε=2% 随机动作，C++ 引擎）训练。数据 4.28M 帧 → 抽帧 64/局 = 320k 样本（`training_history/distill_top2/lowq_f64/`）。标签 = **局末血量差/20**（含 ±0.1 平局破平）。**val R²=0.4708 / 0.4701 / 0.4688（3 种子，种子间差 0.002）**，MAE≈0.215。对比：原 M6c 价值网在同验证集 R²=**0.4533** |
| `checkpoints/posnet_A_k5_m32_lowq.pt` | 6.5 MB | **嫁接版**：`posnet_A_k5_m32.pt` 的 `value_state` 换成 `lowq_value_seed0.pt`（84/84 张量同名拷贝，policy 头不动）。生成脚本 `code/my_ai/entity_transformer/graft_value_ckpt.py` |

⚠️ 关键读数：**种子间只差 0.002**（对比历史 336 局时代"同配置换种子胜率 25%→50%"的抽彩票现象）⇒ **5000 局量级下训练已经稳定**，噪声地板问题基本解决。

⚠️ R² 天花板 0.47 是**标签性质**决定的（从当前局面预测几十回合后的终局血量差），不是数据量或网络容量不足。

## 低质量数据资产（2026-10-19）

| 路径 | 大小 | 说明 |
|---|---|---|
| `F:\2026智能体大赛_新2\training_history\lowq_e2\` | 50.5 GB | 5000 局原始采集（每局一个 npz + `.ents.jsonl`：塔/蚁/效果/武器/金币/HP/回合 全字段） |
| `F:\2026智能体大赛_新2\training_history\lowq_e2_merged\` | 91.3 GB | 合并大数组（列式 memmap + `game_offsets.npy` 每局边界） |
| `D:\...\distill_top2\lowq_f64\` | 6.5 GB | **每局均匀抽 64 帧**的紧凑集（32 万样本，可全量进内存） |

⚙️ 磁盘：**D 是 SSD（954 GB）、E/F 都是 HDD（3726 GB）**。数据放 HDD 时顺序读 ~130 MB/s，随机读会退化成寻道风暴——同一任务"散读帧"与"整窗口顺序读"实测差 30 倍以上。

## 小值网 `ValueNetwork(32,3)`（2026-10-27，详见 `docs/small_value_net_20261027.md`）

**目的**：把 rollout 的叶子值网从 542,913 参数（64/6）换成 **103,137 参数（32/3）**，
换取前向 **5.1×**（batch64 实测 143.4 ms → 25.7 ms，4 线程），用于在 10 s 预算内加随机化。

| 文件 | 大小 | 是什么 |
|---|---|---|
| `checkpoints/small32_3/lowq_value_seed{0,1,2}.pt` | 各 **429,587 B** | `ValueNetwork(latent_dim=32, num_resblocks=3)`，**103,137 参数**。数据/标签/切分**与上面 64/6 那三个完全同口径**（`lowq_f64` + `label.npy` + 最后 500 局留出）。**val R²=0.4711 / 0.4707 / 0.4730（均值 0.4716，种子间 0.0010）**，MAE≈0.215 —— **不输 64/6 的 0.4701，远高于现役 `posnet_A_k5_m32` 的 0.4533**。ckpt 里带 `config={latent_dim:32,num_resblocks:3}` 与 `y_mean=-0.0004 / y_std=0.3861` |

⚠️ **这是独立 `ValueNetwork`，不能灌进三网 ckpt**（`load_three_models` 用同一组 `latent_dim/num_resblocks` 建三个网）⇒ 只能在消费侧单独加载，
且 `az_intent/value_batch.py:55` 与 `az_intent/flat_rollout_greedy.py:284` 写的是 `["value"]`，取值前要适配裸张量。
**部署时必须把 `y_std/y_mean` 折进 `value_head.2`**（照 `graft_value_ckpt.py:68-76`），否则输出放大 2.6×。

⚠️ 与现役值网的排序一致性：**Spearman ρ(tanh)=0.9598 / 0.9621 / 0.9590**（20,000 帧，同一留出集；门槛 0.95，**贴线通过**）。
在 `|tanh老师|≥0.2` 的帧上符号不一致只有 0.00–0.02%。

校验（2026-10-27 实测）：
```
fe643552461db0fccde21379b480bc55461cf730a65b8f336532ab92ab8082d7  small32_3/lowq_value_seed0.pt
42f60aeb1e6f6959517963623c01f3c2d7940e3fa602e0dd81ef0f7216e5987f  small32_3/lowq_value_seed1.pt
ddeedc3d5f6db8fd19953f7b36d77125f1bd9121d5798106d1d672d4ef965b04  small32_3/lowq_value_seed2.pt
```
（训练日志 `training_history/logs/train_small32_3.log`；`exit=0`，911 s，commit `ed5b28d`，`run_logged.sh` 名 `small_value_32_3`。）

### 2.7 `checkpoints/small32_3/v_32_3_folded.pt`（**部署可直接用**的折叠版，2026-10-27）

| 文件 | 大小 | 是什么 |
|---|---|---|
| `checkpoints/small32_3/v_32_3_folded.pt` | **429,075 B** | `lowq_value_seed0.pt` 的**归一化折叠版**：`value_head.2` 换成 `W*y_std` / `b*y_std+y_mean`（`y_std=0.386068, y_mean=-0.000408`），**输出直接是标签尺度**，部署侧 `AZAI_VALUE_TANH=1` 只做 tanh、不用改。仍带 `config={latent_dim:32,num_resblocks:3}`、`y_std/y_mean`、`y_folded=True`、`fold_info`、`source/val_r2/seed` |

- 生成：`code/test_match/fold_small_value_ckpt.py`（日志 `training_history/logs/fold_small32_3.log`）
- 自检：`max|folded - (raw*y_std+y_mean)| = 1.24e-07`（等价，逐位级）；1000 帧留出集
  `Spearman(tanh folded, tanh 现役) = 0.9600`、`Pearson=0.9708`、`R²(folded raw vs label)=0.4876`
- 反例（不折）：`Pearson` 掉到 0.9513、`tanh` 的 std 从 0.2489 变 0.4979（2.0×）、
  `|tanh|>0.9` 的帧从 0.00% 变 5.70% —— 就是调研 §5.3 那个"排序被压平"的事故
- 部署：`AZAI_VALUE_CKPT=checkpoints/small32_3/v_32_3_folded.pt`（空 = 关，行为逐位不变）。
  见 `code/my_ai/az_intent/value_override.py`、`docs/small_value_net_integration_20261027.md`

校验（2026-10-27 实测）：
```
76dfa446ac16704ef85e4a5d71c28775438ee3a3bba7116883d33ce92f7ef315  small32_3/v_32_3_folded.pt
```
