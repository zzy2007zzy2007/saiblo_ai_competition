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
