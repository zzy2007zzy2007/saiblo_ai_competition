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

> 上面"是什么"是按**文件名 + 预注册命名 + 用法**写的简要定位；细节看各自的 `docs/prereg_*.md`。

### 校验（"手头这份是不是当初那份"）

在 `checkpoints/` 目录下跑：

```bash
sha256sum -c <<'EOF'
6fc38cb8ecbe9344c74c1eecfb3fd4bfeb5267960039d720deabf94cb3f235d9  posnet_A_k5_m32.pt
4c6a246e24391aa4145ec12f60d55f22aa0faf7a645ca76a4af2d54886c14e4f  three_mix_r10p_vw_pol_frozen.pt
4c73aa98bcb7a0ca70aa296bbceaa5824e218da9cdc19113539a3b9f23de30e5  valnet_A2.pt
d340ad42f9cbc84a0ae48e4976b61473dd946c76dc28e63d8dbefaa9b65dac8d  posnet_Vk50.pt
da7fc52a78ce683d02c38176f730dda6ba16a0e319cb04236af346a7d83db05d  posnet_C_class.pt
09123a5eae525ef277f6f3288fd0e17f1cc88905daaa188838487ac9535f8aa1  posnet_M1_distill.pt
0732163b307290ac295e8eec83f86ab8732b96438904f80ae9f050cb7f9d32f6  qhead_M2.pt
EOF
```

## 2. 不入库、但记哈希（太大 / 已废弃）

| 文件 | 原路径 | 大小 | sha256 |
|---|---|---|---|
| `gen_0120.pt` | `training_history/ga_ss_20260730_093908/gen_0120.pt` | **404 MB** | `3046e77a0b5d3431defdc8ad9d17ac1e0995500438a322dd0a8d945b27176298` |

- `gen_0120` 是**血统起点**（官方 `ExampleAI` 蒸馏/初始化那一代），但 404 MB **超 GitHub 单文件硬限（100 MB）**
  ⇒ 只在库外保留，靠上面的哈希核对。
- 更早的一些 ckpt（旧文档**只记了路径**）见 `docs/notable_checkpoints.md`（那份文档已作废，保留作考古）。

## 3. 还有哪些**没**进来（要用再拷）

`training_history/vprior/` 下还有 **~24 个 `posnet_*` 变体**（各 6.5 MB，各对应某个臂），以及
`az_fixed/`、`distill_data_v2/` 里的一些 ckpt，**都没入库**（不然仓库会被撑爆）。
要用哪个：`cp <原路径> checkpoints/` ⇒ 算哈希 ⇒ 在 §1 表里加一行 ⇒ **连同文件一起提交**。

## 4. 加新 ckpt 的规矩（4 步）

1. `cp <原路径> checkpoints/`（用**原文件名**，便于和文档/日志对应）；
2. `sha256sum checkpoints/<file>`；
3. 在本文件 §1 表**加一行**（文件 / 来源 / 大小 / 哈希 / 一句话说明）；
4. **提交时别漏了文件本身**（`git add checkpoints/<file> docs/checkpoints.md`）——
   ⚠️ 只提交文档、忘记 `git add` 二进制，是这套做法最容易犯的错。
