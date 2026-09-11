# 实验记录（Experiment Log）

**目的**：让每个实验都能被复现和追溯——跑的是哪份代码、什么命令、产出在哪、结果是什么。

**为什么记 `commit` + `dirty`**：同一份代码、同一条命令，工作区有没有未提交的改动会直接
改变结果。2026-09-09 追"34.4% 复现失败"时，最大的障碍就是分不清"变量"和"工作区脏"。

## 记录格式

每条包含：

| 字段 | 说明 |
|------|------|
| 时间 | 本地时间 |
| **commit** | `git rev-parse --short HEAD` + `dirty: N files`（**已跟踪文件被改但未提交**的数量；未跟踪的临时文件另计） |
| exit | 退出码 + 用时 |
| cmd | 完整可复制粘贴的命令 |
| output | 日志路径（`training_history/runs/<时间戳>_<名字>/output.log`） |
| result | 一行结论（需手动补） |

## 怎么用

```bash
bash code/run_logged.sh <实验名> <命令...>
```

脚本会自动抓 git 版本、把输出 tee 到 `training_history/runs/<时间戳>_<名字>/output.log`，
并在本文件末尾追加一条记录。跑完把 `result` 补上即可。

---

## 2026-09-09 — 关键实验补录（补录，非脚本自动生成）

### 裸策略基线（回答"§23 说策略进步了，到底进步在哪"）

- **commit**: `f442903`（当时工作区有未提交改动）
- **cmd**: `eval.py --checkpoint {training_history/az_fixed/az_r10.pt | --baseline-hotstart ga_ss/gen_0120.pt} --opponent rule_v4 --no-search --native-engine --games 32 --workers 16`
- **result**:
  - `az_r10` 裸策略 vs rule_v4 = **0W/32L = 0%**
  - `gen0120` 裸策略 vs rule_v4 = **1W/31L = 3.1%**
  - 结论：两个裸策略都≈0 → §23 的"az 策略进步了"应改写为"az 策略**作为搜索先验**更有效"（配原版价值头 18.8% → 34.4%）。

### 深度成本基准（回答"减深度能不能省时间"）

- **cmd**: `_bench_depth_cost.py --checkpoint mix_r10p_bn0v.pt --iterations 256 --depths 2 4 --reps 3`
- **result**: depth2 = 2.388s / depth4 = 2.584s（**只差 8%**）。成本在每次迭代的 `_expand`，不在树深。真正线性的杠杆是迭代数。

### 价值标签"两个 bug"复核（推翻 §22）

- **cmd**: 见 `/tmp/check_labels.py` 逻辑（旧 `(stats[24]-stats[25])*50` vs 新 `stats[1]` 在真实 pkl 上对照）
- **result**: 旧/新 `d[t]` **corr 0.999999**（差异仅 float16 舍入）；`stats[24]/[25]` 确实是我方/敌方 HP/50。**"两个 bug"是误诊，commit `53ffca6` 是等价变换**。详见结果文档 §22 修正。

### BN 消融（回答"BN 训练是否破坏价值头"）

- **训练 cmd**（两 arm 同 init、同数据，只差 `--keep-bn`）:
  ```bash
  value_warmup.py --hotstart training_history/az_fixed/gen0120_bn_init.pt \
      --data-dir training_history/az_intent/warm_data_cpp \
      --checkpoint training_history/az_fixed/vw_{nonbn,bn}_from_bninit.pt \
      --epochs 10 --skip-collect --device auto [--keep-bn]
  ```
- **测试 cmd**: `eval.py --checkpoint mix_r10p_{nonbn,bn}_frombninit.pt --opponent rule_v4 --bundle-mcts --iterations 256 --max-depth-rounds 4 --native-engine --t-class 0.5 --t-pos 0.3 --k 24 --games 32`
- **result**:
  - no_bn arm = **3W/29L = 9.4%**
  - BN arm = **2W/30L = 6.2%**
  - 结论：**BN 不是元凶**（两者都差）。但两 arm 都没复现 34.4% → 变量在别处。

### 控制实验：历史 34.4% checkpoint 重测（关键）

- **cmd**: `eval.py --checkpoint training_history/az_fixed/mix_r10p_bn0v.pt --opponent rule_v4 --bundle-mcts --iterations 256 --max-depth-rounds 4 --native-engine --t-class 0.5 --t-pos 0.3 --k 24 --games 32 --workers 14`
- **result**: **11W/21L = 34.4%**，与历史数字完全一致。
- 结论：**评估环境/代码没有漂移**。34.4% 的基准可信；问题在训练侧。

### 价值头离线指标对比（反直觉）

- **cmd**: 在 `warm_data_cpp` 40 局（32224 样本）上比较各价值头与真实标签的相关性
- **result**:

  | 价值头 | std | corr(target) | corr(历史好头) |
  |---|---|---|---|
  | `gen0120_warm_cpp`（历史 34.4%） | 0.327 | +0.799 | 1.000 |
  | `vw_nonbn_from_bninit`（新 9.4%） | 0.335 | **+0.822** | 0.976 |
  | `vw_bn_from_bninit`（新 6.2%） | 0.328 | +0.811 | 0.980 |
  | `az_r10` value（旧 9.4%） | 1.393 | +0.725 | 0.936 |

- 结论：**离线指标（随机状态上的相关性）完全无法预测实战强度**——新头 corr 甚至更高，实战差 3.7 倍。MCTS 依赖的是**同层兄弟动作之间的排序**，不是全局相关性。

### 实验 A 采集（已暂停）

- **cmd**: `az_selfplay.py --checkpoint training_history/az_fixed/mix_r10p_bn0v.pt --games 160 --workers 16 --iterations 256 --max-depth-rounds 4 --max-rounds 512 --out-dir training_history/az_fixed/data_goodvalue --seed 20000 --native-engine --t-class 0.5 --t-pos 0.3 --k 24`
- **result**: 完成 **60/160** 局后暂停（腾 CPU 给 BN 消融测试）。恢复用 `--seed 30000 --games 100`。
- 备注：每局约 30 分钟（两个搜索玩家互相防守，平均 432 回合）。

### §25 配方 GPU 重跑（待测试）

- **cmd**: `value_warmup.py --hotstart training_history/ga_ss_20260730_093908/gen_0120.pt --data-dir training_history/az_intent/warm_data_cpp --checkpoint training_history/az_fixed/gen0120_warm_cpp_gpu.pt --epochs 10 --skip-collect --device auto`
- **result**: 训练完成（value loss 0.0529）；`mix_r10p_geninit_gpu.pt` vs rule_v4 测试进行中。用途：区分"init 是关键"还是"GPU 训练有问题"。

## 2026-09-09 17:07:52 — vw_wcpp2_gpu

- **commit**: `7528ad3` (dirty: 15 files)
- **exit**: 0，用时 884s
- **cmd**:
  ```bash
  D:/anaconda3/envs/pytorch-gpu/python.exe code/my_ai/az_intent/value_warmup.py --hotstart training_history/ga_ss_20260730_093908/gen_0120.pt --data-dir training_history/az_intent/warm_data_cpp2 --checkpoint training_history/az_fixed/gen0120_warm_cpp2_gpu.pt --epochs 10 --skip-collect --device auto
  ```
- **output**: `training_history/runs/20260909_170752_vw_wcpp2_gpu/output.log`
- **result**: _待填_

## 2026-09-09 17:08:24 — vw_cpu_faithful

- **commit**: `7528ad3` (dirty: 15 files)
- **exit**: 0，用时 2009s
- **cmd**:
  ```bash
  D:/anaconda3/envs/pytorch-gpu/python.exe code/my_ai/az_intent/value_warmup.py --hotstart training_history/ga_ss_20260730_093908/gen_0120.pt --data-dir training_history/az_intent/warm_data_cpp --checkpoint training_history/az_fixed/gen0120_warm_cpp_cpu.pt --epochs 10 --skip-collect --device cpu
  ```
- **output**: `training_history/runs/20260909_170824_vw_cpu_faithful/output.log`
- **result**: _待填_

## 2026-09-09 17:43:02 — vw_gpu_no_tf32

- **commit**: `7528ad3` (dirty: 16 files)
- **exit**: 0，用时 488s
- **cmd**:
  ```bash
  D:/anaconda3/envs/pytorch-gpu/python.exe code/my_ai/az_intent/value_warmup.py --hotstart training_history/ga_ss_20260730_093908/gen_0120.pt --data-dir training_history/az_intent/warm_data_cpp --checkpoint training_history/az_fixed/gen0120_warm_cpp_gpu_notf32.pt --epochs 10 --skip-collect --device auto --no-tf32
  ```
- **output**: `training_history/runs/20260909_174302_vw_gpu_no_tf32/output.log`
- **result**: _待填_

---

## 2026-09-09 — 34.4% 复现失败：根因是"训练对计算路径的系统性浮点差异极其敏感"

### 排除的假设

| 假设 | 结论 |
|---|---|
| 数据用错（`warm_data_cpp` vs `warm_data_cpp2`） | **排除**——两份数据集逐位相同（同 seed、同 board、同 value_target） |
| BN 训练破坏价值头 | **排除**——no_bn 9.4% vs BN 6.2%，两者都差 |
| init 不同 | **部分**——ga_ss/gen_0120 15.6% vs gen0120_bn_init 9.4%，init 有影响但不是主因 |
| TF32 | **排除**——GPU+no-TF32 权重 corr 0.7156 ≈ GPU+TF32 0.7176 |
| GPU 计算有 bug | **排除**——CPU vs GPU 梯度相对误差 1.8e-5，比 CPU 换线程数的 2.4e-5 还小 |

### 核心证据

- **CPU 复现与历史好头逐位相同**（w_corr = 1.00000，maxdiff = 0.0000）→ 34.4% 是 CPU/默认线程路径的确定解
- **GPU 训练落在另一个解**（w_corr 0.7176）→ 15.6%
- 两次 GPU 训练彼此逐位相同（w_corr = 1.0）→ GPU 路径自身可复现
- 参数组散度：`policy_heads` corr 0.999（anchor loss 钉死输出），**`resblocks` 骨干 corr 0.246**（max|diff| 3.69 ≈ 20×std）→ 骨干自由漂移
- 但**两者都远离 init**（corr 均 ~0.51，GPU 甚至更近）→ **漂移量不是质量差异的原因**
- 梯度差分析：单批 ‖Δg‖/‖g‖ ≈ 1.4e-5~5.6e-5；**不同批次 Δg 的平均 cos = +0.31**，‖mean Δg‖/mean‖Δg‖ = **0.72** → 差异含**系统性偏置**，不是随机噪声。1e-5 × 5 万步 ≈ 0.5，可解释 w_corr 0.72 的散度。

### 结论

价值头训练在一个**欠约束的大空间**里自由优化（anchor loss 只约束输出、不约束权重），
因此对计算路径的**系统性浮点差异**极度敏感：CPU/GPU（或换线程数）会确定性但任意地
落到不同解，而这些解的搜索强度差 2 倍多（34.4% vs 15.6%）。

**"34.4%" 不是"正确的解"，只是 CPU/默认线程那条路径恰好走到的解。**

### 待验证 / 修法

- 正在验证：CPU + 8 线程是否也训出不同的解（若成立，则与设备无关）
- 修法候选：① 冻结骨干只训价值头；② 对 init 骨干加权重空间锚定（L2）
- 短期工程解：价值头统一用 **CPU + 固定线程数** 训练（逐位可复现）

## 2026-09-09 19:13:52 — vw_cpu_8threads

- **commit**: `7528ad3` (dirty: 16 files)
- **exit**: 0，用时 1307s
- **cmd**:
  ```bash
  D:/anaconda3/envs/pytorch-gpu/python.exe code/my_ai/az_intent/value_warmup.py --hotstart training_history/ga_ss_20260730_093908/gen_0120.pt --data-dir training_history/az_intent/warm_data_cpp --checkpoint training_history/az_fixed/gen0120_warm_cpp_cpu8.pt --epochs 10 --skip-collect --device cpu --threads 8
  ```
- **output**: `training_history/runs/20260909_191352_vw_cpu_8threads/output.log`
- **result**: _待填_

## 2026-09-09 19:44:07 — eval_hyb_gpub_cpuh

- **commit**: `2566b42` (dirty: 14 files)
- **exit**: 1，用时 710s
- **cmd**:
  ```bash
  D:/anaconda3/envs/pytorch-gpu/python.exe code/my_ai/az_intent/eval.py --checkpoint training_history/az_fixed/mix_r10p_hyb_gpubackbone_cpuhead.pt --opponent rule_v4 --bundle-mcts --iterations 256 --max-depth-rounds 4 --native-engine --t-class 0.5 --t-pos 0.3 --k 24 --games 32 --workers 8
  ```
- **output**: `training_history/runs/20260909_194407_eval_hyb_gpub_cpuh/output.log`
- **result**: _待填_

## 2026-09-09 19:44:10 — eval_hyb_cpub_gpuh

- **commit**: `2566b42` (dirty: 14 files)
- **exit**: 1，用时 708s
- **cmd**:
  ```bash
  D:/anaconda3/envs/pytorch-gpu/python.exe code/my_ai/az_intent/eval.py --checkpoint training_history/az_fixed/mix_r10p_hyb_cpubackbone_gpuhead.pt --opponent rule_v4 --bundle-mcts --iterations 256 --max-depth-rounds 4 --native-engine --t-class 0.5 --t-pos 0.3 --k 24 --games 32 --workers 8
  ```
- **output**: `training_history/runs/20260909_194410_eval_hyb_cpub_gpuh/output.log`
- **result**: _待填_

## 2026-09-09 19:56:31 — vw_frozen_cpu_bb

- **commit**: `2566b42` (dirty: 15 files)
- **exit**: 0，用时 695s
- **cmd**:
  ```bash
  D:/anaconda3/envs/pytorch-gpu/python.exe code/my_ai/az_intent/value_warmup.py --hotstart training_history/az_fixed/gen0120_warm_cpp_cpu.pt --data-dir training_history/az_intent/warm_data_cpp --checkpoint training_history/az_fixed/vw_frozen_cpubb.pt --epochs 10 --skip-collect --device cpu --freeze-backbone
  ```
- **output**: `training_history/runs/20260909_195631_vw_frozen_cpu_bb/output.log`
- **result**: _待填_

## 2026-09-09 19:56:38 — vw_frozen_gpu_bb

- **commit**: `2566b42` (dirty: 15 files)
- **exit**: 0，用时 694s
- **cmd**:
  ```bash
  D:/anaconda3/envs/pytorch-gpu/python.exe code/my_ai/az_intent/value_warmup.py --hotstart training_history/az_fixed/gen0120_warm_cpp_gpu.pt --data-dir training_history/az_intent/warm_data_cpp --checkpoint training_history/az_fixed/vw_frozen_gpubb.pt --epochs 10 --skip-collect --device cpu --freeze-backbone
  ```
- **output**: `training_history/runs/20260909_195638_vw_frozen_gpu_bb/output.log`
- **result**: _待填_

## 2026-09-09 20:08:29 — eval_frozen_cpubb

- **commit**: `2566b42` (dirty: 15 files)
- **exit**: 0，用时 4065s
- **cmd**:
  ```bash
  D:/anaconda3/envs/pytorch-gpu/python.exe code/my_ai/az_intent/eval.py --checkpoint training_history/az_fixed/mix_r10p_frozen_cpubb.pt --opponent rule_v4 --bundle-mcts --iterations 256 --max-depth-rounds 4 --native-engine --t-class 0.5 --t-pos 0.3 --k 24 --games 32 --workers 8
  ```
- **output**: `training_history/runs/20260909_200829_eval_frozen_cpubb/output.log`
- **result**: _待填_

## 2026-09-09 20:08:32 — eval_frozen_gpubb

- **commit**: `2566b42` (dirty: 15 files)
- **exit**: 0，用时 4131s
- **cmd**:
  ```bash
  D:/anaconda3/envs/pytorch-gpu/python.exe code/my_ai/az_intent/eval.py --checkpoint training_history/az_fixed/mix_r10p_frozen_gpubb.pt --opponent rule_v4 --bundle-mcts --iterations 256 --max-depth-rounds 4 --native-engine --t-class 0.5 --t-pos 0.3 --k 24 --games 32 --workers 8
  ```
- **output**: `training_history/runs/20260909_200832_eval_frozen_gpubb/output.log`
- **result**: _待填_

## 2026-09-09 21:17:49 — vw_azdata_frozen_gpu

- **commit**: `2566b42` (dirty: 15 files)
- **exit**: 0，用时 496s
- **cmd**:
  ```bash
  D:/anaconda3/envs/pytorch-gpu/python.exe code/my_ai/az_intent/value_warmup.py --hotstart training_history/ga_ss_20260730_093908/gen_0120.pt --data-dir training_history/az_fixed/warm_from_pkl --checkpoint training_history/az_fixed/vw_azdata_frozen_gpu.pt --epochs 10 --skip-collect --device auto --freeze-backbone
  ```
- **output**: `training_history/runs/20260909_211749_vw_azdata_frozen_gpu/output.log`
- **result**: _待填_

## 2026-09-09 21:17:51 — vw_azdata_frozen_cpu

- **commit**: `2566b42` (dirty: 15 files)
- **exit**: 0，用时 1294s
- **cmd**:
  ```bash
  D:/anaconda3/envs/pytorch-gpu/python.exe code/my_ai/az_intent/value_warmup.py --hotstart training_history/ga_ss_20260730_093908/gen_0120.pt --data-dir training_history/az_fixed/warm_from_pkl --checkpoint training_history/az_fixed/vw_azdata_frozen_cpu.pt --epochs 10 --skip-collect --device cpu --freeze-backbone
  ```
- **output**: `training_history/runs/20260909_211751_vw_azdata_frozen_cpu/output.log`
- **result**: _待填_

## 2026-09-09 21:19:12 — az_frozen_gpu

- **commit**: `016fac8` (dirty: 15 files)
- **exit**: 0，用时 1679s
- **cmd**:
  ```bash
  D:/anaconda3/envs/pytorch-gpu/python.exe code/my_ai/az_intent/az_train.py --init training_history/az_fixed/gen0120_bn_init.pt --policy-dir training_history/az_fixed/data/batch1 --data-dir training_history/az_fixed/data --checkpoint training_history/az_fixed/az_frozen_gpu.pt --epochs 5 --tau 50 --label-scale 6 --split --max-value-batches 16 --device auto --value-passes 3 --label-mode abs --freeze-backbone
  ```
- **output**: `training_history/runs/20260909_211912_az_frozen_gpu/output.log`
- **result**: _待填_

## 2026-09-09 21:19:26 — az_frozen_cpu

- **commit**: `016fac8` (dirty: 15 files)
- **exit**: 0，用时 2165s
- **cmd**:
  ```bash
  D:/anaconda3/envs/pytorch-gpu/python.exe code/my_ai/az_intent/az_train.py --init training_history/az_fixed/gen0120_bn_init.pt --policy-dir training_history/az_fixed/data/batch1 --data-dir training_history/az_fixed/data --checkpoint training_history/az_fixed/az_frozen_cpu.pt --epochs 5 --tau 50 --label-scale 6 --split --max-value-batches 16 --device cpu --value-passes 3 --label-mode abs --freeze-backbone
  ```
- **output**: `training_history/runs/20260909_211926_az_frozen_cpu/output.log`
- **result**: _待填_

## 2026-09-09 21:47:27 — az_frozen_terminal_gpu

- **commit**: `016fac8` (dirty: 16 files)
- **exit**: 0，用时 1329s
- **cmd**:
  ```bash
  D:/anaconda3/envs/pytorch-gpu/python.exe code/my_ai/az_intent/az_train.py --init training_history/az_fixed/gen0120_bn_init.pt --policy-dir training_history/az_fixed/data/batch1 --data-dir training_history/az_fixed/data --checkpoint training_history/az_fixed/az_frozen_terminal_gpu.pt --epochs 5 --split --max-value-batches 16 --device auto --value-passes 3 --label-mode terminal --freeze-backbone
  ```
- **output**: `training_history/runs/20260909_214727_az_frozen_terminal_gpu/output.log`
- **result**: _待填_

## 2026-09-09 21:47:27 — eval_azfrozen_gpu

- **commit**: `016fac8` (dirty: 16 files)
- **exit**: 0，用时 3794s
- **cmd**:
  ```bash
  D:/anaconda3/envs/pytorch-gpu/python.exe code/my_ai/az_intent/eval.py --checkpoint training_history/az_fixed/mix_r10p_azfrozen_gpu.pt --opponent rule_v4 --bundle-mcts --iterations 256 --max-depth-rounds 4 --native-engine --t-class 0.5 --t-pos 0.3 --k 24 --games 32 --workers 10
  ```
- **output**: `training_history/runs/20260909_214727_eval_azfrozen_gpu/output.log`
- **result**: _待填_

---

## 2026-09-09 — 冻结骨干消融：gen 数据有效、az 数据无效

### 背景

上一条发现"价值头联合训练对计算路径混沌敏感"，提出的修法是**冻结骨干、只训价值头**。
本轮在两种数据上验证，并测了标签尺度。

### 实验 A：gen 数据（`warm_data_cpp`，200 局）

| arm | 冻结骨干来源 | 价值头 | vs rule_v4 |
|-----|-------------|--------|-----------|
| 不冻结 | — | 联合训练 | 34.4%（CPU）/ **15.6%**（GPU） |
| `vw_frozen_cpubb` | CPU 解 | 重训 | **25.0%**（8W/24L） |
| `vw_frozen_gpubb` | **GPU 解**（那个"坏"的） | 重训 | **34.4%**（11W/21L） |

**结论**：冻结骨干后，**连 GPU 那条"坏"路径漂出来的骨干也能到 34.4%**（历史最好成绩）。
说明骨干本身不坏，坏的是"价值头追漂移骨干"的联合训练过程。
副产品：冻结后 CPU/GPU 训练结果一致（value_state corr 1.00000 vs 不冻结时的 0.72）→ 可复现。

### 实验 B：az 数据（`az_fixed/data`，160 局）+ 真实程序 `az_train.py`

命令（照搬 run_az_batches.sh 的配方 + `--freeze-backbone`）：

```bash
az_train.py --init az_fixed/gen0120_bn_init.pt --policy-dir az_fixed/data/batch1 \
    --data-dir az_fixed/data --epochs 5 --split --max-value-batches 16 \
    --value-passes 3 --label-mode {abs --label-scale 6 | terminal} --freeze-backbone
```

| arm | 标签 | vs rule_v4 |
|-----|------|-----------|
| `az_frozen_gpu` | abs, scale 6 | **6.2%**（2W/30L） |
| `az_frozen_terminal_gpu` | terminal | **3.1%**（1W/31L） |
| 对照：az 数据不冻结 | abs, scale 6 | 9.4% |

**结论：冻结骨干救不了 az 数据**——两个标签模式都更差。

### 价值头输出尺度分析（同一批 az 数据，label std = 0.222）

| 价值头 | 原始输出 std | tanh 后 | 饱和比例 |
|--------|-------------|---------|---------|
| 原版 `gen0120_warm_cpp`（34.4%） | **0.2118** | 0.2050 | 0% |
| 冻结 + abs scale6 | 0.7655 | 0.5327 | 1.7% |
| 冻结 + terminal | 0.0897 | 0.0886 | 0% |

原版好头的输出 std ≈ 标签 std（0.21 ≈ 0.22）——完美匹配；abs×6 过大（3.6×）、
terminal 过小（0.42×，R² 仅 0.32，因为终局标签每局共享、状态信号弱）。

### 总体结论

1. **冻结骨干**是"训练稳定性"的正解（gen 数据 25-34%、跨设备可复现），但**不是"数据没信号"的解**。
2. **az 自对弈数据是瓶颈**（§26 结论成立）：标签弱、胶着局多、决定性局仅 36%。
   在 az 数据上，不冻结 9.4%、冻结 6.2%/3.1%——怎么训都差。
3. 冻结骨干用的是 **gen 数据的骨干**，没有机会适配 az 状态分布，可能是它在 az 数据上更差的原因之一。
4. 下一步方向应回到**数据侧**（对手池 / 更多对局 / 混合数据），而不是继续调价值头训练。

## 2026-09-09 22:09:49 — eval_azfrozen_term

- **commit**: `016fac8` (dirty: 16 files)
- **exit**: 0，用时 3341s
- **cmd**:
  ```bash
  D:/anaconda3/envs/pytorch-gpu/python.exe code/my_ai/az_intent/eval.py --checkpoint training_history/az_fixed/mix_r10p_azfrozen_term.pt --opponent rule_v4 --bundle-mcts --iterations 256 --max-depth-rounds 4 --native-engine --t-class 0.5 --t-pos 0.3 --k 24 --games 32 --workers 8
  ```
- **output**: `training_history/runs/20260909_220949_eval_azfrozen_term/output.log`
- **result**: _待填_

## 2026-09-09 23:08:47 — collect_p1_batch1

- **commit**: `4cb014f` (dirty: 15 files)
- **exit**: 0，用时 2780s
- **cmd**:
  ```bash
  D:/anaconda3/envs/pytorch-gpu/python.exe code/my_ai/az_intent/az_selfplay.py --checkpoint training_history/az_fixed/mix_r10p_bn0v.pt --games 16 --workers 16 --iterations 256 --max-depth-rounds 4 --max-rounds 512 --out-dir training_history/az_fixed/data_polonly/batch1 --seed 40000 --native-engine --t-class 0.5 --t-pos 0.3 --k 24
  ```
- **output**: `training_history/runs/20260909_230847_collect_p1_batch1/output.log`
- **result**: _待填_

## 2026-09-09 23:55:15 — train_p1_batch1

- **commit**: `4cb014f` (dirty: 16 files)
- **exit**: 0，用时 302s
- **cmd**:
  ```bash
  D:/anaconda3/envs/pytorch-gpu/python.exe code/my_ai/az_intent/az_train.py --init training_history/az_fixed/mix_r10p_bn0v.pt --policy-dir training_history/az_fixed/data_polonly/batch1 --data-dir training_history/az_fixed/data_polonly --checkpoint training_history/az_fixed/az_p1.pt --epochs 5 --split --policy-only --max-value-batches 1 --device auto
  ```
- **output**: `training_history/runs/20260909_235515_train_p1_batch1/output.log`
- **result**: _待填_

## 2026-09-10 09:16:10 — polonly_resume

- **commit**: `4cb014f` (dirty: 16 files)
- **exit**: 0，用时 5618s
- **cmd**:
  ```bash
  bash code/resume_polonly.sh
  ```
- **output**: `training_history/runs/20260910_091610_polonly_resume/output.log`
- **result**: _待填_

## 2026-09-10 10:50:03 — eval_azp6_vs_rulev4

- **commit**: `28aec25` (dirty: 16 files)
- **exit**: 0，用时 2490s
- **cmd**:
  ```bash
  D:/anaconda3/envs/pytorch-gpu/python.exe code/my_ai/az_intent/eval.py --checkpoint training_history/az_fixed/az_p6.pt --opponent rule_v4 --bundle-mcts --iterations 256 --max-depth-rounds 4 --native-engine --t-class 0.5 --t-pos 0.3 --k 24 --games 32 --workers 16
  ```
- **output**: `training_history/runs/20260910_105003_eval_azp6_vs_rulev4/output.log`
- **result**: _待填_

## 2026-09-10 11:34:21 — svs_azp6_vs_init

- **commit**: `28aec25` (dirty: 16 files)
- **exit**: 0，用时 4722s
- **cmd**:
  ```bash
  D:/anaconda3/envs/pytorch-gpu/python.exe code/my_ai/az_intent/az_search_vs_search.py --a training_history/az_fixed/az_p6.pt --b training_history/az_fixed/mix_r10p_bn0v.pt --games 32 --workers 16 --iterations 256 --max-depth-rounds 4 --t-class 0.5 --t-pos 0.3 --native-engine
  ```
- **output**: `training_history/runs/20260910_113421_svs_azp6_vs_init/output.log`
- **result**: _待填_

## 2026-09-10 18:36:26 — eval_azp6_rand005

- **commit**: `28aec25` (dirty: 17 files)
- **exit**: 0，用时 1988s
- **cmd**:
  ```bash
  D:/anaconda3/envs/pytorch-gpu/python.exe code/my_ai/az_intent/eval.py --checkpoint training_history/az_fixed/az_p6.pt --opponent rule_v4 --bundle-mcts --iterations 256 --max-depth-rounds 4 --native-engine --t-class 0.5 --t-pos 0.3 --k 24 --random-action-prob 0.05 --games 32 --workers 16
  ```
- **output**: `training_history/runs/20260910_183626_eval_azp6_rand005/output.log`
- **result**: _待填_

## 2026-09-10 19:11:14 — eval_azp6_rand002

- **commit**: `28aec25` (dirty: 17 files)
- **exit**: 0，用时 1978s
- **cmd**:
  ```bash
  D:/anaconda3/envs/pytorch-gpu/python.exe code/my_ai/az_intent/eval.py --checkpoint training_history/az_fixed/az_p6.pt --opponent rule_v4 --bundle-mcts --iterations 256 --max-depth-rounds 4 --native-engine --t-class 0.5 --t-pos 0.3 --k 24 --random-action-prob 0.02 --games 32 --workers 16
  ```
- **output**: `training_history/runs/20260910_191114_eval_azp6_rand002/output.log`
- **result**: _待填_

## 2026-09-10 19:48:49 — polonly_to21_rand002

- **commit**: `28aec25` (dirty: 17 files)
- **exit**: 0，用时 37998s
- **cmd**:
  ```bash
  bash code/resume_polonly.sh 0.02 21
  ```
- **output**: `training_history/runs/20260910_194849_polonly_to21_rand002/output.log`
- **result**: _待填_

## 2026-09-11 06:22:20 — eval_azp21_vs_rulev4

- **commit**: `28aec25` (dirty: 17 files)
- **exit**: 0，用时 3727s
- **cmd**:
  ```bash
  D:/anaconda3/envs/pytorch-gpu/python.exe code/my_ai/az_intent/eval.py --checkpoint training_history/az_fixed/az_p21.pt --opponent rule_v4 --bundle-mcts --iterations 256 --max-depth-rounds 4 --native-engine --t-class 0.5 --t-pos 0.3 --k 24 --games 64 --workers 16
  ```
- **output**: `training_history/runs/20260911_062220_eval_azp21_vs_rulev4/output.log`
- **result**: _待填_

## 2026-09-11 09:27:30 — conv_polonly_npz

- **commit**: `28aec25` (dirty: 17 files)
- **exit**: 0，用时 1401s
- **cmd**:
  ```bash
  D:/anaconda3/envs/pytorch-gpu/python.exe code/my_ai/az_intent/convert_pkl_to_warm_npz.py --data-dir training_history/az_fixed/data_polonly --out-dir training_history/az_fixed/warm_polonly --workers 4
  ```
- **output**: `training_history/runs/20260911_092730_conv_polonly_npz/output.log`
- **result**: _待填_

## 2026-09-11 07:24:41 — svs_azp21_vs_init

- **commit**: `28aec25` (dirty: 17 files)
- **exit**: 0，用时 8829s
- **cmd**:
  ```bash
  D:/anaconda3/envs/pytorch-gpu/python.exe code/my_ai/az_intent/az_search_vs_search.py --a training_history/az_fixed/az_p21.pt --b training_history/az_fixed/mix_r10p_bn0v.pt --games 64 --workers 16 --iterations 256 --max-depth-rounds 4 --t-class 0.5 --t-pos 0.3 --native-engine
  ```
- **output**: `training_history/runs/20260911_072441_svs_azp21_vs_init/output.log`
- **result**: _待填_

---

## 2026-09-11 — 策略-only 循环（冻结价值头）到顶 + 2% 随机注入消融

### 背景

前面（09-09/09-10）确定了：价值头训练对计算路径敏感、az 数据标签弱、
"冻结骨干"能稳定训练但对弱数据无效。本轮测试另一条路线——
**冻结一个好价值头，只训策略**（把 AlphaZero 的自动价值训练改成手动+验证）。

### 实验 1：策略-only 循环（冻结 gen0120_bn_init 价值头）

每 batch：采集 16 局 @256 迭代/C++/16 workers（~45 min）+ `az_train --split --policy-only`
5 epochs（~5 min）。价值网逐位不变（已验 corr 1.000000）。

| 模型 | batch 数 | vs rule_v4 (seed0-31) | svs vs `mix_r10p_bn0v` |
|------|---------|----------------------|----------------------|
| `mix_r10p_bn0v`（起点） | 0 | **34.4%** (11W/21L) | — |
| `az_p6` | 6 | 28.1% (9W/23L) | **50.0%** (16W/16L) |
| `az_p21` | 21（含 2% 注入） | 17.2% (64 局)；seed0-31 = 15.6% | **48.4%** (31W/33L, 64 局)；seed0-31 = 46.9% |

**配对分析（az_p6 → az_p21，同 seed 0-31）**：
- svs：9 局输 / 8 局赢（p≈1.0）→ **完全对称，策略没变**
- vs rule_v4：8 局输 / 4 局赢（p≈0.39）→ 不显著

**结论**：
1. **"冻结价值头 + 只训策略"到顶了**——21 个 batch 后策略在 svs 上仍是 parity。
   原因：搜索的目标对策略已无新信息（策略和搜索的合法偏好一致率 99.9%）。
2. 唯一疑点：vs rule_v4 点估计 28.1% → 17.2%（64 局 σ≈±9.5pp，上界 ~27% 擦到 28.1%），
   而 svs 说"没变"。可能的解释：策略自我一致性原地踏步，但泛化到 rule_v4 略有损失
   （在自对弈风格里收窄）。幅度在噪声边缘，不下定论。
3. **瓶颈确认在价值头**：策略只在"搜索比它强"时才能学到东西，而搜索质量由价值头决定。

### 实验 2：2% / 5% 随机动作注入消融

`--random-action-prob x`（az_selfplay 采集侧 + eval.py 推理侧）：以概率 x 忽略搜索，
玩一个均匀随机的合法 bundle。

**采集侧效果**（决定性，是注入的目的）：

| 数据 | \|标签\|均值 | 决定性 |
|------|-------------|--------|
| batch1-6（无注入，96 局） | 0.218 | 45% |
| batch7-14（2% 注入，128 局） | 0.259 | **55%** |
| 全部 21 batch（336 局） | 0.250 | 62% |
| （参考）gen 裸策略 200 局 | 0.321 | 71% |

方向对，幅度有限。**训练损失在 batch7（注入开始）出现结构性跳变**：
起始 policy loss 0.54→0.65、末 0.37→0.44（注入让对局进入策略没见过的状态，
搜索目标更难拟合；是预期行为不是 bug）。对局长度不变（409-450 回合）。

**推理侧代价**（az_p6 vs rule_v4，同 seed 0-31 配对）：

| x | 战绩 | 胜率 |
|---|------|------|
| 0 | 9W/23L | 28.1% |
| **0.02** | 13W/19L | **40.6%**（6 输 / 10 赢，p≈0.45 不显著） |
| 0.05 | 3W/29L | 9.4%（6 局单向翻转，显著有害） |

**结论**：5% 明确有害（-19pp）；2% 的代价测不出来（点估计反而更好），
可以用于采集。但注意每 20 步 1 次乱走就能让 5% 的配置掉 19pp，
说明这个游戏的容错极低（对局在濒死血量上拉锯）。

### 方法论发现（重要）

**评估是完全确定性的**（temperature=0、固定 seed、注入 rng 按 seed 定死）——
**重复跑不会有任何变化**，要降噪只能**增加 seed 数量**（更多不同初始局面），不能靠重跑。
所以判断差异必须用**同 seed 配对**，而不是比较不同局数的总胜率。

### 数据侧观察（记录备查）

`_marginalize` 记录的是"网络采样的原始 intent"，**包含非法类别**（采样先按完整
softmax 选类、之后才降级成合法操作）：实测搜索目标的**82% 质量落在非法类别上**
（主要是 LIGHTNING，t_class=0.5 放大所致，见 §17）。训练 `_sample_policy_loss`
会把非法类别跳过，所以有效目标只剩 18%（以 HOLD 为主）。
这不影响正确性（用户确认高手也是 80-90% HOLD），但意味着**有效目标信息量小**。
