# 备份目录空间审计 & "该不该用 git 存" 的结论

- 日期:2026-10-27
- 审计对象:`F:\智能体大赛备份_旧版`(7 个手动快照:备份2 ~ 备份8)
- 工具:`code/tools/backup_audit.py`、`code/tools/dedupe_scan2.py`、`code/tools/compare_copy_dirs.py`、`code/tools/summarize_dedupe.py`
- 原始结果:`docs/backup_audit/audit_result.json`、`docs/backup_audit/dedupe_result.json`

---

## 0. 一句话结论

**"用 git 存同一文件只存一次"这个理解是对的,但它不是这里的主要解药。**

这 1.81 TiB 里,源码只有 **174 MiB(占 0.0092%)**;
剩下的 99.99% 是模型权重、训练中间产物、采集数据;
而其中 **36% 的字节躺在名字就叫"没用的 / 副本 / 旧版本"的目录里**——
这是"同一个时刻的目录里就有多份副本",不是"历史版本存了多次"。
git 能去重**内容**,但去不掉**工作区里的重复路径**:你 checkout 出来还是两份。

> ⚠️ 但**不要被目录名骗了**:第 6 节的抽查发现,这 668.56 GiB 的 "没用的/副本" 里
> 有 **27.31 GiB 是在全部 7 个备份里独一份的源码/文档**,删了就真没了。
> 其中 `备份4\没用的` 其实是"当时整个项目被整份改名"进去的,不是垃圾堆。
> 真正能无脑删的只有 **266.20 GiB 的 `temp_*`** 和约 **290 GiB 的 `- 副本*` 目录**。

所以:git 用来管源码(已经在用了),空间靠**删中间产物 + 只留白名单 checkpoint**。

---

## 1. 现状:1.81 TiB / 214,289 个文件

| 版本 | 占用 | 文件数 |
| --- | ---: | ---: |
| 备份2 | 210.3 GiB | 46,654 |
| 备份3 | 285.5 GiB | 63,528 |
| 备份4 | 290.0 GiB | 57,964 |
| 备份5 | 296.0 GiB | 20,888 |
| 备份6 | 215.1 GiB | 2,427 |
| 备份7 | 275.2 GiB | 4,073 |
| 备份8 | 284.4 GiB | 18,755 |
| **合计** | **1856.48 GiB (1.81 TiB)** | **214,289** |

> F: 盘总共 3.7 TB,现在只剩 215 GB。这堆备份占了 F: 的一半以上。

### 1.1 按扩展名(事实摆在这里,别的都不用争了)

| 扩展名 | 占用 | 占比 | 文件数 |
| --- | ---: | ---: | ---: |
| `.npz` | 729.82 GiB | 39.3% | 19,600 |
| `.pth` | 474.72 GiB | 25.6% | 17,290 |
| `.json` | 346.43 GiB | 18.7% | 91,198 |
| `.pkl` | 261.37 GiB | 14.1% | 3,611 |
| `.pt` | 24.75 GiB | 1.3% | 278 |
| `.zip` | 16.85 GiB | 0.9% | 391 |
| `.txt` | 1.72 GiB | 0.1% | 42,763 |
| `.pyc` | 0.17 GiB | 0.0% | 9,240 |
| **`.py`** | **0.15 GiB** | **0.008%** | 14,237 |

**7 个版本的全部 Python 源码加起来 = 150 MiB。**
加上 C++ 头/源/脚本(`.cpp/.h/.hpp/.sh/.cu`)总共 **174 MiB**。

### 1.2 按"该不该留"分类

| 类别 | 占用 | 占比 | 文件数 | 判定 |
| --- | ---: | ---: | ---: | --- |
| `junk_dir`(路径含 没用的/副本/旧版本/copy) | **669.88 GiB** | 36.1% | 148,726 | 基本可删 |
| `ckpt`(模型权重) | 652.44 GiB | 35.1% | 19,634 | 留白名单,其余可删 |
| `temp_file`(`temp_*.npz` 训练中间产物) | **266.20 GiB** | 14.3% | 3,034 | 可删 |
| `dataset`(>10MB 的回放/数据) | 173.12 GiB | 9.3% | 1,212 | 按需 |
| `small`(源码/文档/零散小文件) | 94.85 GiB | 5.1% | 41,683 | 基本都留 |

### 1.3 各版本构成(GiB)

| 版本 | junk_dir | ckpt | temp_file | dataset | small | 合计 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| v2 | 97.6 | 65.6 | 0.0 | 4.1 | 42.9 | 210.3 |
| v3 | 136.3 | 93.3 | 0.0 | 4.4 | 51.6 | 285.5 |
| v4 | 237.4 | 49.3 | 0.0 | 3.2 | 0.1 | 290.0 |
| v5 | 135.8 | 63.7 | 88.7 | 7.7 | 0.1 | 296.0 |
| v6 | 0.4 | 74.7 | 88.7 | 51.2 | 0.1 | 215.1 |
| v7 | 31.2 | 104.0 | 88.7 | 51.2 | 0.1 | 275.2 |
| v8 | 31.2 | 201.8 | 0.0 | 51.3 | 0.1 | 284.4 |

> `temp_file` 三份 88.7 GiB 完全一样 → 这 88.7 GiB 被存了 3 遍。

---

## 2. 重复到底有多少(实测,不是猜)

### 2.1 判重上界(只用元数据就能确定的)

> 依据:**文件大小不同,内容必然不同。**
> 所以"大小唯一"的文件 100% 是独一无二的内容,不可能被任何去重手段省掉。

- 大小唯一的文件:7,157 个,**46.06 GiB** ← 这部分省不了
- 出现在"同大小分组"里的字节:**1810.42 GiB** ← 重复的上界

也就是说:**最多能省 1810 GiB,最少省 0**;真实值要靠内容比对。

### 2.2 抽样内容确认

对"最值钱"的 400 个同大小分组做 `(大小, 前 64KB sha1, 后 64KB sha1)` 指纹:

- 只读了 **196 MB** 的内容(F: 是 USB 机械盘,随机读仅 ~1.3 MB/s,全量哈希跑不完)
- 覆盖了 **137.67 GiB** 的分组
- **实测确认重复 126.69 GiB** → 抽查样本里 92% 是纯重复

最大的一组:**357 个文件内容完全相同,每个 90.0 MiB,合计 31.28 GiB。**
样例路径指向 `ga_nn_training\training_history\run_*\temp_genome_*.npz`。

### 2.3 "副本"目录有多离谱

用 `(相对路径, 文件大小)` 对比 `Ant-Game` 下的原目录与副本(只读元数据):

| 副本目录 | 占用 | 与原目录同名同大小 | 纯浪费 |
| --- | ---: | ---: | ---: |
| `training_pipeline - 副本` | 11.148 GiB | 186/199 文件,**100.0% 字节** | 11.147 GiB |
| `training_pipeline - 副本 (2)` | 18.249 GiB | 272/283 文件,**100.0% 字节** | 18.249 GiB |
| `training_pipeline - 副本 (3)` | 1.441 GiB | 56.2% | 0.810 GiB |

前两个是 **100% 一模一样的整份拷贝**,29.4 GiB 纯浪费,就在同一个版本里。
(顺便验证了"同名同大小"这个判据是可靠的——它把"100% 重复"准确挑出来了。)

### 2.4 "没用的"目录 Top 榜

| 目录(相对 `2026智能体大赛\`) | 占用 |
| --- | ---: |
| `没用的` | **304.48 GiB** |
| `没用的_2` | **68.57 GiB** |
| `download\旧版本对战` | 61.30 GiB |
| `ai_neural_network_new\没用的旧模型` | 53.92 GiB |
| `ai_neural_network_new - 副本 (2)` | 43.41 GiB |
| `Ant-Game\training_pipeline - 副本 (2)` | 36.50 GiB |
| `ai_neural_network_new - 副本 (3)` | 32.22 GiB |
| `ai_neural_network - 副本 (2)` | 31.23 GiB |
| `Ant-Game\training_pipeline - 副本` | 22.30 GiB |
| `Ant-Game\training_pipeline - 副本 (3)` | 2.88 GiB |

### 2.5 `temp_*` 中间产物

- 全部集中在 `ga_nn_training\training_history`,**266.20 GiB / 3,034 个文件**
- 单个版本(v5)内:**1,012 个 `temp_*` = 88.7 GiB**
- 单个 run 目录 `run_20260428_235337` 里就有 **702 个 `temp_genome_*.npz`,每个 90 MiB ≈ 61.7 GiB**
- 抽查 25 个只有 23 种内容 → 大部分 genome 确实不同(锦标赛 GA 每代都存一份权重,正常但极其浪费)

### 2.6 最新版本(v8)的 checkpoint 都在哪

v8 的 `.npz/.pth/.pt/.pkl` 共 **233.00 GiB**:

| 路径 | 占用 | 文件数 |
| --- | ---: | ---: |
| `Ant-Game\training_pipeline\custom_training_results` | 105.78 GiB | 4,663 |
| `Ant-Game\mcts_training_pipeline_new\training_results` | 23.54 GiB | 1,201 |
| `Ant-Game\training_pipeline - 副本 (2)\...` | 18.25 GiB | 203 |
| `Ant-Game\mcts_training_pipeline_new_with_rule\...` | 17.69 GiB | 919 |
| `Ant-Game\mcts_training_pipeline_summarized_state\...` | 15.04 GiB | 1,303 |
| `Ant-Game\training_pipeline - 副本\...` | 11.15 GiB | 124 |
| `Ant-Game\mcts_training_pipeline\training_results` | 10.83 GiB | 405 |
| `nn_distillation_new\models` | 10.81 GiB | 120 |

**`custom_training_results` 一个目录 105.78 GiB** —— 每个 iteration 存一份权重。
4 个 mcts pipeline 变体加起来 67.1 GiB,其中 3 个是同一套代码的实验分支。

---

## 3. 回到问题:用 git 存能省多少?

### 3.1 git 确实"同一内容只存一次",这是对的

三层机制:

1. **内容寻址**:blob 用 `sha1(文件内容)` 命名。同一内容出现在 1 个 commit 还是 100 个 commit、
   在 1 个路径还是 100 个路径,`.git/objects` 里都只有**一份**。跨 commit、跨目录都去重。
2. **zlib 压缩**:源码/文本通常压 2~4 倍。
3. **packfile delta**:相似的 blob 存成"一个基础版 + 差异",所以"改了几行的 50 个版本"几乎不占空间。

对**源码**,git 是最省的方式,没有更好的。当前仓库就是这个状态:
`.git` 打包后 420 MiB,存着全部提交历史。

### 3.2 但对这 1.8 TiB,git 是错的工具

| 问题 | 说明 |
| --- | --- |
| 体积不对等 | 源码 174 MiB = 总量的 0.009%。git 就算把源码省到 0,也救不回 1.8 TiB |
| 二进制不吃压缩 | `.npz` 本身就是 deflate 过的 zip 容器,git 的 zlib 再压 ≈ 无效 |
| "差一个字节 = 一整份" | `.pth/.npz` 只要变一点就存一个新 blob;权重是稠密浮点,二进制 delta 压缩收益很低 |
| repack 需要 2 倍空间 | `git gc` 要重打包,峰值 ≈ 1.8 TiB × 2 > F: 剩余 215 GB → **直接失败** |
| 单文件过大 | 单个 200 MiB 的 JSON 回放超过默认 delta 窗口,基本等于每版全量副本 |
| 日常操作变慢 | `git status`/`git add` 要 stat 21 万个文件 + 算 SHA1 |
| 用不上 | 你不会真的去 `git checkout` 一个 300 GiB 的版本 |

### 3.3 最关键的认知

git 优化的是**存储**,不是**工作区**。

`没用的` + `没用的_2` + 各种 `- 副本` 共 669.88 GiB,这些大多是**同一个 commit 里的重复路径**,
不是跨版本历史。git 会把同内容合并成一个 blob(存储省了),
但你 `checkout` 出来**仍然是两份 18.25 GiB 的文件**——工作区一个字节都没省。

而且你的痛点根本不是"历史存太多",是**"跑训练时的中间产物没有清理机制"**:
一个 run 目录 702 个 90 MiB 的权重 = 61.7 GiB,这个锅 git 不背。

### 3.4 什么样才该上 git(现在的仓库其实已经做对了)

当前 `.gitignore` 已经把 `data/ distill_data/ value_data/ training_history/ match_results/`
这些 280 GB 级的东西排除了 —— 这是正确的做法,继续维持。

---

## 4. 建议的执行方案(按性价比排序)

### 第一档:低风险,实测可自动回收 **500.60 GiB**

> ⚠️ **先看第 6 节**:最初我把 668.76 GiB 的 junk 目录整体判成"零风险 934.95 GiB",
> 后来做全局唯一性检查发现 **27.31 GiB 是全备份独一份、删了就真没了**,
> 而且 `备份4\没用的` 其实是**当时整个项目被整份改名**进去的,不是垃圾堆。
> 下面这张表已按检查结果重新分档。

`python code/tools/cleanup_backup.py`(**默认只列出,不删除**):

| 档位 | 内容 | 项数 | 占用 | 说明 |
| --- | --- | ---: | ---: | --- |
| 第 1 档 | `temp_*` 训练中间产物 | 3,044 文件 | **266.20 GiB** | GA 每代临时 genome,`备份5/6/7` 各存一份 |
| 第 2 档 | `*- 副本*`、`*- copy`、`没用的旧模型` | 45 项 | **234.40 GiB** | 全局比对 C 类(独一份)≈ 0,删了内容仍在别处 |
| **可自动合计** | — | — | **500.60 GiB** | — |
| 需人工 | `没用的` / `没用的_2` / `旧版本对战` | 5 项 | **434.34 GiB** | **工具绝不自动删**,见第 6 节 |

> 第 2 档的注意点:`没用的旧模型` 在 `备份2` 和 `备份3` 各有一份,
> 两份之间互有约 12.5 GiB 只在对方那里 —— **删一份安全,两份都删就丢内容**。

(A) 里最大的几项(还暴露了跨版本重复):

| 路径(相对 `智能体大赛备份N\`) | 占用 | 文件数 |
| --- | ---: | ---: |
| `备份4\2026智能体大赛\没用的` | 237.45 GiB | 57,393 |
| `备份5\2026智能体大赛\没用的_2` | 68.57 GiB | 10,538 |
| `备份5\2026智能体大赛\没用的` | 67.03 GiB | 8,408 |
| `备份3\...\ai_neural_network_new - 副本 (3)` | 32.22 GiB | 10,820 |
| `备份3\...\download\旧版本对战` | 30.90 GiB | 7,248 |
| `备份2\...\download\旧版本对战` | 30.39 GiB | 7,078 |
| `备份3\...\ai_neural_network_new\没用的旧模型` | 29.40 GiB | 9,636 |
| `备份2\...\ai_neural_network_new\没用的旧模型` | 24.51 GiB | 8,217 |
| `备份2/备份3\...\ai_neural_network_new - 副本 (2)` | 21.70 GiB ×2 | 7,470 ×2 |
| `备份7/备份8\...\Ant-Game\training_pipeline - 副本 (2)` | 18.25 GiB ×2 | 283 ×2 |
| `备份7/备份8\...\Ant-Game\training_pipeline - 副本` | 11.15 GiB ×2 | 199 ×2 |

> 注意 `备份7` 和 `备份8` 的 `training_pipeline - 副本(2)` 是 **18.25 GiB / 283 文件,完全一致** ——
> 同一个垃圾被两个快照各存一遍。

(B) 的分布更说明问题:同一批 `temp_*` 在 **备份5 / 6 / 7 各存一遍**:

| 目录(备份5/6/7 各一份) | 每份占用 |
| --- | ---: |
| `ga_nn_training\training_history\run_20260428_235337` | **61.67 GiB** |
| `...\run_20260428_221316` | 4.92 GiB |
| `...\run_20260428_213111` | 3.16 GiB |
| `...\run_20260428_215557` | 3.16 GiB |

### 第二档:砍历史版本 —— 预计降到 40 ~ 60 GiB

7 个快照里真正有信息量的只有 2~3 个。建议:

- 保留 **v8(最新)** 作为基线
- 保留 v5 / v6 里**只存在于旧版本**的东西(如早期 GA 代码、`simple_rl_example`、聊天记录)
- 其余整份删除
- 每个保留版本的 `custom_training_results` 只留 **`checkpoints/` 白名单**
  (CLAUDE.md 本来就要求"把关键的 checkpoint 拷进 checkpoints/ 并记录")

### 第三档:长效机制(防止再长回来)

1. **训练脚本加保留策略**:`keep_last_n` / 只存 best & last,`temp_*` 用完即删
   - 这一条能根治 266 GiB 那个问题
2. **`.gitignore` 之外再加一道**:训练输出统一写到 `runs/`,`runs/` 整个不进 git 也不进备份
3. **备份方式换成"源码 git + 数据白名单"**:
   - 源码/文档/配置 → git(F: 上没必要再留整份快照)
   - 数据/权重 → 只留 `checkpoints/` 里登记过的那几个,其余不留
4. **如果一定要保留全部历史又要省空间**:用**硬链接去重**(NTFS 原生支持)
   - 相同内容的多个路径共享同一份磁盘块,省空间且不需要任何工具链
   - 比 git 更合适的场景:这里重复的是"工作区里的重复路径"

### 关于"要不要给这堆东西建 git 历史"

**不建议。** 如果真想知道"当年那版代码长什么样",更划算的做法是:
把 7 个版本的 **源码 + 文档 + 聊天记录**(约 174 MiB + 1.8 GiB 文档)抽出来建一个小仓库,
按时间顺序做成 7 个 commit。这样:

- 存储 ≈ 几百 MiB(≈ 现在的 1/4000)
- 能 `git log` / `git diff` / `git checkout` 看任意一版
- 不需要碰任何 `.npz/.pth`

---

## 5. 复现方法

```bash
# 1. 分类审计(1.8TiB 全扫,约 6~10 分钟,只读元数据)
python code/tools/backup_audit.py

# 2. 判重上界 + 抽样内容指纹(只读 ~200MB)
python code/tools/dedupe_scan2.py

# 3. 原目录 vs 副本目录重合度(只读元数据)
python code/tools/compare_copy_dirs.py

# 4. 汇总
python code/tools/summarize_dedupe.py

# 5. 第一档清理(默认 dry-run,只列出)
python code/tools/cleanup_backup.py --tier1

# 6. 抽样取证:junk 目录里到底是什么
python code/tools/inspect_junk.py --tree --only 0     # 看目录结构
python code/tools/inspect_junk.py --read --only 2     # 抽小文本文件读内容

# 7. 全局唯一性:删了 junk 会不会丢内容
python code/tools/check_junk_uniqueness.py    # 版本内对比
python code/tools/global_junk_check.py        # 全备份 7 版本对比(A/B/C 三类)
```

---

## 6. 追加(2026-10-27):"没用的" 里面到底装了什么

原本只按目录名把它判成垃圾。实际打开看之后,**结论要修正**。

### 6.1 结构:它不是一堆零散垃圾,是"整个项目被改名"

`备份4` 的根目录只有两样东西:

| 目录 | 占用 |
| --- | ---: |
| `没用的` | 237.45 GiB |
| `distillation_trainer` | 52.41 GiB |

也就是说 **2026-04 那阵子,你把整个旧项目重命名成 `没用的`,然后开了一个新的
`distillation_trainer` 从头做蒸馏**。所以 `备份4\没用的` = **那一代的全量存档**,
不是"重复文件堆"。它内部结构和项目根目录完全一样:

| 内部目录 | 占用 | 是什么 |
| --- | ---: | --- |
| `ai_neural_network_new` | 33.10 GiB | 自对弈 BC/RL 训练框架,含 `没用的旧模型` 29.40 GiB |
| `ai_neural_network_new - 副本 (3)` | 32.22 GiB | 上面那份的副本 |
| `ai_neural_network_new - 副本 (2)` | 21.70 GiB | 副本 |
| `ai_neural_network` | 15.90 GiB | 第一版 NN(含 `model.pth` 10.11 MiB) |
| `ai_neural_network - 副本 (2)` | 15.62 GiB | 上面那份的副本 |
| `training` | 15.31 GiB | **`dataset.pkl` 单文件 15.3 GB**(行为克隆数据集) |
| `ga_training_new` | 14.31 GiB | 遗传算法训练 |
| `ga_training_new_2` | 14.31 GiB | 同一份 |
| `ga_training_new - 副本 (2)` / `- 副本` | 13.37 / 13.13 GiB | 副本 |
| `ga_training` / `- 副本 (2)` / `- 副本 (3)` | 10.90 / 10.92 / 10.90 GiB | 三份几乎一样 |
| `ai_dqn_new` | 5.40 GiB | DQN 版本 |
| 根目录散文件 | ~2 GiB | `蚁洋陷役2 - Saiblo.pdf/html`、`Trae聊天记录1~21.md`、`历史训练记录.txt`、`experiment_modified.zip`、`old_ai.zip` |

`备份5` 的两个 "没用的" 则是**训练实验目录的集合**:

- `备份5\没用的`: `ga_training - 副本 (2)/(3)`、`ga_training_new - 副本 (2)/副本`、
  `distillation_training`、`distillation_trainer copy`、`ga_nn_trainer`、`nn_trainer`、`ai_submission`
- `备份5\没用的_2`: `ga1_gen32_1`、`ga_nn_training_2`、`ga_nn_training_refactored`、
  `ga_training_new`、`ga_training_new_2`、`good_versions`、`my_ai_agent`、`my_ai_dev`、`my_ant_game`

**`没用的_2\good_versions` 值得单独提**:里面是**可以直接提交的 zip 包**
（`example.zip` 38 文件 / `v3.zip` 20.5 MB 91 文件 / `run_20260406_203812_gen99.zip` 101 文件），
这是当时的比赛提交产物,不该当垃圾删。

**`备份3\...\download\旧版本对战`** 也不是垃圾:里面是 `FalseCat_ / Happigy / kskvnkadvnj /
Xishiqing / zzy25` 五个对手的**平台对战回放**(文件名是 match_id,单个 3.8 MB 的完整
`op0/op1/round_state` 录制)。删了就没法复现当时的对局。

**只有 `备份3\...\ai_neural_network_new\没用的旧模型` 是干净可删的**:
29.40 GiB / 9,636 文件,by name+size 100% 都能在别处找到。

### 6.2 全局唯一性检查(A/B/C 三类)

对 junk 目录里的每个文件,按 `(文件名, 大小)` 去**全部 7 个版本**里找:

| junk 目录(相对路径) | 合计 | A 别处非 junk 也有 | B 只在别的 junk 里有 | C 全备份独一份 |
| --- | ---: | ---: | ---: | ---: |
| `没用的`(备份4+备份5) | 304.48 | 182.89 | 107.12 | **14.47** |
| `没用的_2` | 68.57 | 29.87 | 25.88 | **12.82** |
| `download\旧版本对战` | 61.30 | 11.45 | 49.85 | 0.00 |
| `ai_neural_network_new\没用的旧模型` | 53.92 | 41.41 | 12.50 | 0.00 |
| `ai_neural_network_new - 副本 (2)` | 43.41 | 31.02 | 12.38 | 0.00 |
| `Ant-Game\training_pipeline - 副本 (2)` | 36.50 | 36.50 | 0.00 | 0.00 |
| `ai_neural_network_new - 副本 (3)` | 32.22 | 25.92 | 6.30 | 0.00 |
| `ai_neural_network - 副本 (2)` | 31.23 | 31.21 | 0.02 | 0.00 |
| `Ant-Game\training_pipeline - 副本` | 22.30 | 22.29 | 0.00 | 0.00 |
| 其余小项 | — | — | — | ~0.02 |
| **合计** | **668.56** | **426.48** | **214.78** | **27.31** |

C 类(独一份)抽样看是什么 —— **几乎全是源码 / README / 聊天记录,不是大权重**:

```
智能体大赛备份4\...\没用的\analyze_replay.py
智能体大赛备份4\...\没用的\Trae聊天记录19.md
智能体大赛备份4\...\没用的\ai_neural_network_new\训练指令.txt
智能体大赛备份4\...\没用的\rank_traceing\rank_history.json
智能体大赛备份5\...\没用的\distillation_training\test_two_stage.py
智能体大赛备份5\...\没用的_2\ga_nn_training_refactored\README.md
智能体大赛备份5\...\没用的_2\ga_nn_training_refactored\ga_trainer.py
智能体大赛备份5\...\没用的_2\ga_nn_training_refactored\ai\two_stage_nn.py
智能体大赛备份5\...\没用的_2\ga_nn_training_refactored\ga\evaluator.py
智能体大赛备份3\...\ai_submission_neural_network - 副本 (3)\model.pth   (22.3 MiB)
```

> `ga_nn_training_refactored` 这一套(**重构版 GA 框架** + README)在全备份里只有一份,
> 藏在 `没用的_2` 里。这大概是这一轮抽查最有价值的一个发现。

### 6.3 修正后的建议

1. **`temp_*` 266.20 GiB 照删** —— 判断不变,纯粹是 GA 每代的临时 genome。
2. **`没用的旧模型`(29.40 GiB)、`training_pipeline - 副本*`(58.8 GiB)、
   各种 `ai_neural_network* - 副本*`** —— A 类占比接近 100%,可删。
3. **`没用的` / `没用的_2` 不要整目录删**。里面藏着:
   - `dataset.pkl` 15.3 GB(行为克隆数据集,重建要重新采集)
   - `good_versions` 的提交包
   - `ga_nn_training_refactored` 重构版框架(全备份唯一一份)
   - `Trae聊天记录19.md` 等文档
   正确做法:**先把这 27.31 GiB 的 C 类文件挑出来单独存**,再删剩下的。
4. `download\旧版本对战`(61.30 GiB)没有 C 类,但 A 类只有 11.45 GiB、
   B 类 49.85 GiB —— 意思是**备份2 和备份3 各存了一份,删掉其中一份即可省 30 GiB**,
   两份都删就没了。

> 注意:F: 是 USB 外接机械盘(WD Elements SE),随机读只有 ~1.3 MB/s。
> 任何全量内容哈希(214k 文件)会跑十几个小时,不要随便试。
> 上面这套流程刻意只读元数据 + 少量文件头,总读取量 < 300 MB。
