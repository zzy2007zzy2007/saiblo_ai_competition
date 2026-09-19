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
- **result**: 训练完成。warm_data_cpp2 ≡ warm_data_cpp（逐位相同的数据）；同配置两次 GPU 训练逐位相同 → GPU 路径自身可复现

## 2026-09-09 17:08:24 — vw_cpu_faithful

- **commit**: `7528ad3` (dirty: 15 files)
- **exit**: 0，用时 2009s
- **cmd**:
  ```bash
  D:/anaconda3/envs/pytorch-gpu/python.exe code/my_ai/az_intent/value_warmup.py --hotstart training_history/ga_ss_20260730_093908/gen_0120.pt --data-dir training_history/az_intent/warm_data_cpp --checkpoint training_history/az_fixed/gen0120_warm_cpp_cpu.pt --epochs 10 --skip-collect --device cpu
  ```
- **output**: `training_history/runs/20260909_170824_vw_cpu_faithful/output.log`
- **result**: **CPU 复现与历史好头逐位相同（w_corr 1.00000）** → 34.4% 是 CPU/默认线程路径的确定解

## 2026-09-09 17:43:02 — vw_gpu_no_tf32

- **commit**: `7528ad3` (dirty: 16 files)
- **exit**: 0，用时 488s
- **cmd**:
  ```bash
  D:/anaconda3/envs/pytorch-gpu/python.exe code/my_ai/az_intent/value_warmup.py --hotstart training_history/ga_ss_20260730_093908/gen_0120.pt --data-dir training_history/az_intent/warm_data_cpp --checkpoint training_history/az_fixed/gen0120_warm_cpp_gpu_notf32.pt --epochs 10 --skip-collect --device auto --no-tf32
  ```
- **output**: `training_history/runs/20260909_174302_vw_gpu_no_tf32/output.log`
- **result**: GPU+no-TF32 权重 corr 0.7156 ≈ GPU+TF32 的 0.7176 → **TF32 排除**（不是精度模式问题）

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
- **result**: **CPU 8 线程 vs CPU 默认 32 线程：骨干 corr 0.568** → 同一设备换线程数也训出不同解 → 与设备无关，是"对计算路径敏感"

## 2026-09-09 19:44:07 — eval_hyb_gpub_cpuh

- **commit**: `2566b42` (dirty: 14 files)
- **exit**: 1，用时 710s
- **cmd**:
  ```bash
  D:/anaconda3/envs/pytorch-gpu/python.exe code/my_ai/az_intent/eval.py --checkpoint training_history/az_fixed/mix_r10p_hyb_gpubackbone_cpuhead.pt --opponent rule_v4 --bundle-mcts --iterations 256 --max-depth-rounds 4 --native-engine --t-class 0.5 --t-pos 0.3 --k 24 --games 32 --workers 8
  ```
- **output**: `training_history/runs/20260909_194407_eval_hyb_gpub_cpuh/output.log`
- **result**: 中途停止：用户指出交叉互换设计有缺陷（价值头与自己的骨干共适应，换骨干必然失败，证明不了骨干好坏）

## 2026-09-09 19:44:10 — eval_hyb_cpub_gpuh

- **commit**: `2566b42` (dirty: 14 files)
- **exit**: 1，用时 708s
- **cmd**:
  ```bash
  D:/anaconda3/envs/pytorch-gpu/python.exe code/my_ai/az_intent/eval.py --checkpoint training_history/az_fixed/mix_r10p_hyb_cpubackbone_gpuhead.pt --opponent rule_v4 --bundle-mcts --iterations 256 --max-depth-rounds 4 --native-engine --t-class 0.5 --t-pos 0.3 --k 24 --games 32 --workers 8
  ```
- **output**: `training_history/runs/20260909_194410_eval_hyb_cpub_gpuh/output.log`
- **result**: 同上，中途停止

## 2026-09-09 19:56:31 — vw_frozen_cpu_bb

- **commit**: `2566b42` (dirty: 15 files)
- **exit**: 0，用时 695s
- **cmd**:
  ```bash
  D:/anaconda3/envs/pytorch-gpu/python.exe code/my_ai/az_intent/value_warmup.py --hotstart training_history/az_fixed/gen0120_warm_cpp_cpu.pt --data-dir training_history/az_intent/warm_data_cpp --checkpoint training_history/az_fixed/vw_frozen_cpubb.pt --epochs 10 --skip-collect --device cpu --freeze-backbone
  ```
- **output**: `training_history/runs/20260909_195631_vw_frozen_cpu_bb/output.log`
- **result**: 训练完成（冻结 CPU 骨干 + 重训价值头，value loss 0.0564）

## 2026-09-09 19:56:38 — vw_frozen_gpu_bb

- **commit**: `2566b42` (dirty: 15 files)
- **exit**: 0，用时 694s
- **cmd**:
  ```bash
  D:/anaconda3/envs/pytorch-gpu/python.exe code/my_ai/az_intent/value_warmup.py --hotstart training_history/az_fixed/gen0120_warm_cpp_gpu.pt --data-dir training_history/az_intent/warm_data_cpp --checkpoint training_history/az_fixed/vw_frozen_gpubb.pt --epochs 10 --skip-collect --device cpu --freeze-backbone
  ```
- **output**: `training_history/runs/20260909_195638_vw_frozen_gpu_bb/output.log`
- **result**: 训练完成（冻结 GPU 骨干，value loss 0.0582）

## 2026-09-09 20:08:29 — eval_frozen_cpubb

- **commit**: `2566b42` (dirty: 15 files)
- **exit**: 0，用时 4065s
- **cmd**:
  ```bash
  D:/anaconda3/envs/pytorch-gpu/python.exe code/my_ai/az_intent/eval.py --checkpoint training_history/az_fixed/mix_r10p_frozen_cpubb.pt --opponent rule_v4 --bundle-mcts --iterations 256 --max-depth-rounds 4 --native-engine --t-class 0.5 --t-pos 0.3 --k 24 --games 32 --workers 8
  ```
- **output**: `training_history/runs/20260909_200829_eval_frozen_cpubb/output.log`
- **result**: **8W/24L = 25.0%**（冻结 CPU 骨干 + 重训头）

## 2026-09-09 20:08:32 — eval_frozen_gpubb

- **commit**: `2566b42` (dirty: 15 files)
- **exit**: 0，用时 4131s
- **cmd**:
  ```bash
  D:/anaconda3/envs/pytorch-gpu/python.exe code/my_ai/az_intent/eval.py --checkpoint training_history/az_fixed/mix_r10p_frozen_gpubb.pt --opponent rule_v4 --bundle-mcts --iterations 256 --max-depth-rounds 4 --native-engine --t-class 0.5 --t-pos 0.3 --k 24 --games 32 --workers 8
  ```
- **output**: `training_history/runs/20260909_200832_eval_frozen_gpubb/output.log`
- **result**: **11W/21L = 34.4%**（冻结 GPU 骨干 + 重训头）→ 冻结骨干后连"坏"的 GPU 骨干也能到 34.4%：骨干本身不坏，坏的是联合训练

## 2026-09-09 21:17:49 — vw_azdata_frozen_gpu

- **commit**: `2566b42` (dirty: 15 files)
- **exit**: 0，用时 496s
- **cmd**:
  ```bash
  D:/anaconda3/envs/pytorch-gpu/python.exe code/my_ai/az_intent/value_warmup.py --hotstart training_history/ga_ss_20260730_093908/gen_0120.pt --data-dir training_history/az_fixed/warm_from_pkl --checkpoint training_history/az_fixed/vw_azdata_frozen_gpu.pt --epochs 10 --skip-collect --device auto --freeze-backbone
  ```
- **output**: `training_history/runs/20260909_211749_vw_azdata_frozen_gpu/output.log`
- **result**: 训练完成（旧 az 数据 + 冻结，输出 std 0.039 —— 价值饿死）

## 2026-09-09 21:17:51 — vw_azdata_frozen_cpu

- **commit**: `2566b42` (dirty: 15 files)
- **exit**: 0，用时 1294s
- **cmd**:
  ```bash
  D:/anaconda3/envs/pytorch-gpu/python.exe code/my_ai/az_intent/value_warmup.py --hotstart training_history/ga_ss_20260730_093908/gen_0120.pt --data-dir training_history/az_fixed/warm_from_pkl --checkpoint training_history/az_fixed/vw_azdata_frozen_cpu.pt --epochs 10 --skip-collect --device cpu --freeze-backbone
  ```
- **output**: `training_history/runs/20260909_211751_vw_azdata_frozen_cpu/output.log`
- **result**: 训练完成（输出 std 0.018 —— 饿死更严重）

## 2026-09-09 21:19:12 — az_frozen_gpu

- **commit**: `016fac8` (dirty: 15 files)
- **exit**: 0，用时 1679s
- **cmd**:
  ```bash
  D:/anaconda3/envs/pytorch-gpu/python.exe code/my_ai/az_intent/az_train.py --init training_history/az_fixed/gen0120_bn_init.pt --policy-dir training_history/az_fixed/data/batch1 --data-dir training_history/az_fixed/data --checkpoint training_history/az_fixed/az_frozen_gpu.pt --epochs 5 --tau 50 --label-scale 6 --split --max-value-batches 16 --device auto --value-passes 3 --label-mode abs --freeze-backbone
  ```
- **output**: `training_history/runs/20260909_211912_az_frozen_gpu/output.log`
- **result**: 训练完成（az_train + BN init + 冻结骨干 + abs×6 标签）

## 2026-09-09 21:19:26 — az_frozen_cpu

- **commit**: `016fac8` (dirty: 15 files)
- **exit**: 0，用时 2165s
- **cmd**:
  ```bash
  D:/anaconda3/envs/pytorch-gpu/python.exe code/my_ai/az_intent/az_train.py --init training_history/az_fixed/gen0120_bn_init.pt --policy-dir training_history/az_fixed/data/batch1 --data-dir training_history/az_fixed/data --checkpoint training_history/az_fixed/az_frozen_cpu.pt --epochs 5 --tau 50 --label-scale 6 --split --max-value-batches 16 --device cpu --value-passes 3 --label-mode abs --freeze-backbone
  ```
- **output**: `training_history/runs/20260909_211926_az_frozen_cpu/output.log`
- **result**: 训练完成

## 2026-09-09 21:47:27 — az_frozen_terminal_gpu

- **commit**: `016fac8` (dirty: 16 files)
- **exit**: 0，用时 1329s
- **cmd**:
  ```bash
  D:/anaconda3/envs/pytorch-gpu/python.exe code/my_ai/az_intent/az_train.py --init training_history/az_fixed/gen0120_bn_init.pt --policy-dir training_history/az_fixed/data/batch1 --data-dir training_history/az_fixed/data --checkpoint training_history/az_fixed/az_frozen_terminal_gpu.pt --epochs 5 --split --max-value-batches 16 --device auto --value-passes 3 --label-mode terminal --freeze-backbone
  ```
- **output**: `training_history/runs/20260909_214727_az_frozen_terminal_gpu/output.log`
- **result**: 训练完成（同上但 terminal 标签；标签 std 0.208/±0.8）

## 2026-09-09 21:47:27 — eval_azfrozen_gpu

- **commit**: `016fac8` (dirty: 16 files)
- **exit**: 0，用时 3794s
- **cmd**:
  ```bash
  D:/anaconda3/envs/pytorch-gpu/python.exe code/my_ai/az_intent/eval.py --checkpoint training_history/az_fixed/mix_r10p_azfrozen_gpu.pt --opponent rule_v4 --bundle-mcts --iterations 256 --max-depth-rounds 4 --native-engine --t-class 0.5 --t-pos 0.3 --k 24 --games 32 --workers 10
  ```
- **output**: `training_history/runs/20260909_214727_eval_azfrozen_gpu/output.log`
- **result**: **2W/30L = 6.2%**（az 数据 + 冻结 + BN init + abs×6）

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
- **result**: **1W/31L = 3.1%**（az 数据 + 冻结 + BN init + terminal）→ 旧数据上怎么训都差

## 2026-09-09 23:08:47 — collect_p1_batch1

- **commit**: `4cb014f` (dirty: 15 files)
- **exit**: 0，用时 2780s
- **cmd**:
  ```bash
  D:/anaconda3/envs/pytorch-gpu/python.exe code/my_ai/az_intent/az_selfplay.py --checkpoint training_history/az_fixed/mix_r10p_bn0v.pt --games 16 --workers 16 --iterations 256 --max-depth-rounds 4 --max-rounds 512 --out-dir training_history/az_fixed/data_polonly/batch1 --seed 40000 --native-engine --t-class 0.5 --t-pos 0.3 --k 24
  ```
- **output**: `training_history/runs/20260909_230847_collect_p1_batch1/output.log`
- **result**: 16 局采集完成，用时 **46.3 分钟**（256 迭代/深度4/C++/16 workers）

## 2026-09-09 23:55:15 — train_p1_batch1

- **commit**: `4cb014f` (dirty: 16 files)
- **exit**: 0，用时 302s
- **cmd**:
  ```bash
  D:/anaconda3/envs/pytorch-gpu/python.exe code/my_ai/az_intent/az_train.py --init training_history/az_fixed/mix_r10p_bn0v.pt --policy-dir training_history/az_fixed/data_polonly/batch1 --data-dir training_history/az_fixed/data_polonly --checkpoint training_history/az_fixed/az_p1.pt --epochs 5 --split --policy-only --max-value-batches 1 --device auto
  ```
- **output**: `training_history/runs/20260909_235515_train_p1_batch1/output.log`
- **result**: 策略-only 训练 5 分钟（policy loss 0.5103→0.3711，value loss 恒 0 = 价值网未动）→ az_p1.pt

## 2026-09-10 09:16:10 — polonly_resume

- **commit**: `4cb014f` (dirty: 16 files)
- **exit**: 0，用时 5618s
- **cmd**:
  ```bash
  bash code/resume_polonly.sh
  ```
- **output**: `training_history/runs/20260910_091610_polonly_resume/output.log`
- **result**: 补训 batch4（用已采数据）+ 继续到 batch5；产出 az_p4/az_p5

## 2026-09-10 10:50:03 — eval_azp6_vs_rulev4

- **commit**: `28aec25` (dirty: 16 files)
- **exit**: 0，用时 2490s
- **cmd**:
  ```bash
  D:/anaconda3/envs/pytorch-gpu/python.exe code/my_ai/az_intent/eval.py --checkpoint training_history/az_fixed/az_p6.pt --opponent rule_v4 --bundle-mcts --iterations 256 --max-depth-rounds 4 --native-engine --t-class 0.5 --t-pos 0.3 --k 24 --games 32 --workers 16
  ```
- **output**: `training_history/runs/20260910_105003_eval_azp6_vs_rulev4/output.log`
- **result**: **9W/23L = 28.1%**（基准 mix_r10p_bn0v = 34.4%）

## 2026-09-10 11:34:21 — svs_azp6_vs_init

- **commit**: `28aec25` (dirty: 16 files)
- **exit**: 0，用时 4722s
- **cmd**:
  ```bash
  D:/anaconda3/envs/pytorch-gpu/python.exe code/my_ai/az_intent/az_search_vs_search.py --a training_history/az_fixed/az_p6.pt --b training_history/az_fixed/mix_r10p_bn0v.pt --games 32 --workers 16 --iterations 256 --max-depth-rounds 4 --t-class 0.5 --t-pos 0.3 --native-engine
  ```
- **output**: `training_history/runs/20260910_113421_svs_azp6_vs_init/output.log`
- **result**: **16W/16L = 50.0%**（精确平手 → 6 个 batch 策略没变）

## 2026-09-10 18:36:26 — eval_azp6_rand005

- **commit**: `28aec25` (dirty: 17 files)
- **exit**: 0，用时 1988s
- **cmd**:
  ```bash
  D:/anaconda3/envs/pytorch-gpu/python.exe code/my_ai/az_intent/eval.py --checkpoint training_history/az_fixed/az_p6.pt --opponent rule_v4 --bundle-mcts --iterations 256 --max-depth-rounds 4 --native-engine --t-class 0.5 --t-pos 0.3 --k 24 --random-action-prob 0.05 --games 32 --workers 16
  ```
- **output**: `training_history/runs/20260910_183626_eval_azp6_rand005/output.log`
- **result**: **3W/29L = 9.4%**；配对（x=0→0.05）6 局单向从赢变输、0 反向 → **5% 随机动作代价 ≈ -19pp**

## 2026-09-10 19:11:14 — eval_azp6_rand002

- **commit**: `28aec25` (dirty: 17 files)
- **exit**: 0，用时 1978s
- **cmd**:
  ```bash
  D:/anaconda3/envs/pytorch-gpu/python.exe code/my_ai/az_intent/eval.py --checkpoint training_history/az_fixed/az_p6.pt --opponent rule_v4 --bundle-mcts --iterations 256 --max-depth-rounds 4 --native-engine --t-class 0.5 --t-pos 0.3 --k 24 --random-action-prob 0.02 --games 32 --workers 16
  ```
- **output**: `training_history/runs/20260910_191114_eval_azp6_rand002/output.log`
- **result**: **13W/19L = 40.6%**；配对 6 输/10 赢（p≈0.45 不显著）→ **2% 的代价测不出来**

## 2026-09-10 19:48:49 — polonly_to21_rand002

- **commit**: `28aec25` (dirty: 17 files)
- **exit**: 0，用时 37998s
- **cmd**:
  ```bash
  bash code/resume_polonly.sh 0.02 21
  ```
- **output**: `training_history/runs/20260910_194849_polonly_to21_rand002/output.log`
- **result**: 15 个 batch（batch7-21，含 2% 注入）全部完成，耗时 **10.6 小时** → az_p7..az_p21

## 2026-09-11 06:22:20 — eval_azp21_vs_rulev4

- **commit**: `28aec25` (dirty: 17 files)
- **exit**: 0，用时 3727s
- **cmd**:
  ```bash
  D:/anaconda3/envs/pytorch-gpu/python.exe code/my_ai/az_intent/eval.py --checkpoint training_history/az_fixed/az_p21.pt --opponent rule_v4 --bundle-mcts --iterations 256 --max-depth-rounds 4 --native-engine --t-class 0.5 --t-pos 0.3 --k 24 --games 64 --workers 16
  ```
- **output**: `training_history/runs/20260911_062220_eval_azp21_vs_rulev4/output.log`
- **result**: **11W/53L = 17.2%**（64 局）；seed0-31 子集 5W/27L = 15.6%；配对（vs az_p6）8 输/4 赢不显著

## 2026-09-11 09:27:30 — conv_polonly_npz

- **commit**: `28aec25` (dirty: 17 files)
- **exit**: 0，用时 1401s
- **cmd**:
  ```bash
  D:/anaconda3/envs/pytorch-gpu/python.exe code/my_ai/az_intent/convert_pkl_to_warm_npz.py --data-dir training_history/az_fixed/data_polonly --out-dir training_history/az_fixed/warm_polonly --workers 4
  ```
- **output**: `training_history/runs/20260911_092730_conv_polonly_npz/output.log`
- **result**: 336 局 pkl → npz：**287,546 样本，决定性 62%，|label| 均值 0.250**（旧 az 数据是 28%/0.17）

## 2026-09-11 07:24:41 — svs_azp21_vs_init

- **commit**: `28aec25` (dirty: 17 files)
- **exit**: 0，用时 8829s
- **cmd**:
  ```bash
  D:/anaconda3/envs/pytorch-gpu/python.exe code/my_ai/az_intent/az_search_vs_search.py --a training_history/az_fixed/az_p21.pt --b training_history/az_fixed/mix_r10p_bn0v.pt --games 64 --workers 16 --iterations 256 --max-depth-rounds 4 --t-class 0.5 --t-pos 0.3 --native-engine
  ```
- **output**: `training_history/runs/20260911_072441_svs_azp21_vs_init/output.log`
- **result**: **31W/33L = 48.4%**；配对（vs az_p6）9 输/8 赢完全对称 → **21 个 batch 后策略仍是平台**

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

## 2026-09-11 09:51:17 — train3_vheads_newdata

- **commit**: `28aec25` (dirty: 17 files)
- **exit**: 0，用时 1512s
- **cmd**:
  ```bash
  bash _tmp_train3.sh
  ```
- **output**: `training_history/runs/20260911_095117_train3_vheads_newdata/output.log`
- **result**: 三个价值头训练完成：A(vw+冻结) value loss 0.0526 / B(vw 不冻结) 0.0481 / C(az_train terminal, BN init)

## 2026-09-11 11:35:21 — train2_tau_newdata

- **commit**: `4daace4` (dirty: 17 files)
- **exit**: 0，用时 1990s
- **cmd**:
  ```bash
  bash _tmp_train_tau.sh
  ```
- **output**: `training_history/runs/20260911_113521_train2_tau_newdata/output.log`
- **result**: D(tau-abs scale2, BN init) / E(scale6) 训练完成（标签 std 0.425 / 1.274）

## 2026-09-11 10:18:19 — eval3_newdata_vheads

- **commit**: `4daace4` (dirty: 17 files)
- **exit**: 0，用时 6977s
- **cmd**:
  ```bash
  bash _tmp_eval3.sh
  ```
- **output**: `training_history/runs/20260911_101819_eval3_newdata_vheads/output.log`
- **result**: **A(vw+冻结) 10W/22L = 31.2%**；B(vw 不冻结) 4W/28L = 12.5%；C(az_train terminal) 3W/29L = 9.4%

## 2026-09-11 12:14:40 — eval2_tau_vheads

- **commit**: `4daace4` (dirty: 17 files)
- **exit**: 0，用时 4301s
- **cmd**:
  ```bash
  bash _tmp_eval_tau.sh
  ```
- **output**: `training_history/runs/20260911_121440_eval2_tau_vheads/output.log`
- **result**: **D(tau s2) 4W/28L = 12.5%；E(tau s6) 3W/29L = 9.4%**（两者都是 BN init）

## 2026-09-11 13:26:36 — c2_aztrain_nonbn

- **commit**: `4daace4` (dirty: 17 files)
- **exit**: 0，用时 2368s
- **cmd**:
  ```bash
  bash _tmp_c2.sh
  ```
- **output**: `training_history/runs/20260911_132636_c2_aztrain_nonbn/output.log`
- **result**: **9W/23L = 28.1%**（az_train + **no_bn** init + 冻结 + terminal）vs C 的 9.4% → **BN 是 az_train 路径的病根**（requires_grad=False 不冻结 running 统计）

## 2026-09-11 14:13:45 — svs_newval_vs_genval

- **commit**: `4daace4` (dirty: 17 files)
- **exit**: 0，用时 9022s
- **cmd**:
  ```bash
  D:/anaconda3/envs/pytorch-gpu/python.exe code/my_ai/az_intent/az_search_vs_search.py --a training_history/az_fixed/mix_r10p_pol_vonly_term_nonbn.pt --b training_history/az_fixed/mix_r10p_bn0v.pt --games 64 --workers 16 --iterations 256 --max-depth-rounds 4 --t-class 0.5 --t-pos 0.3 --native-engine
  ```
- **output**: `training_history/runs/20260911_141345_svs_newval_vs_genval/output.log`
- **result**: **34W/30L = 53.1%**（控制策略变量，只比价值头）→ 自对弈训的头与 gen 原版头**等价**，"28.1% vs 34.4%"是噪声

## 2026-09-11 16:49:14 — combined_data_2arms

- **commit**: `4daace4` (dirty: 17 files)
- **exit**: 0，用时 2493s
- **cmd**:
  ```bash
  bash _tmp_combined.sh
  ```
- **output**: `training_history/runs/20260911_164914_combined_data_2arms/output.log`
- **result**: F: vw 从零 + 冻结，在合并 536 局（gen 200 + 自对弈 336）上训练完成

## 2026-09-11 17:00:04 — eval_combined_head

- **commit**: `4daace4` (dirty: 21 files)
- **exit**: 0，用时 5882s
- **cmd**:
  ```bash
  bash _tmp_evalF.sh
  ```
- **output**: `training_history/runs/20260911_170004_eval_combined_head/output.log`
- **result**: **8W/24L = 25.0%**；配对（vs A=31.2%）7 输/5 赢不显著 → **合并 gen 数据没有帮助**

## 2026-09-11 16:59:54 — gn_2arms

- **commit**: `4daace4` (dirty: 21 files)
- **exit**: 0，用时 5958s
- **cmd**:
  ```bash
  bash _tmp_gn.sh
  ```
- **output**: `training_history/runs/20260911_165954_gn_2arms/output.log`
- **result**: GroupNorm 转换成功；H(GN+**解冻骨干**) value loss 0.0478 / I(GN+冻结) 0.0575 → 解冻拟合更好

---

## 2026-09-11 18:39:21 — eval_gn_free_H（补录：自动条目被并发写入覆盖）

- **commit**: 见 `training_history/runs/20260911_183921_eval_gn_free_H/meta.txt`
- **cmd**: `eval.py --checkpoint training_history/az_fixed/mix_r10p_gn_free.pt --opponent rule_v4 --bundle-mcts --iterations 256 --max-depth-rounds 4 --native-engine --t-class 0.5 --t-pos 0.3 --k 24 --games 32 --workers 16`
- **output**: `training_history/runs/20260911_183921_eval_gn_free_H/output.log`
- **result**: 见下方 §GroupNorm 小结（Arm H：GN + 解冻骨干）

## 2026-09-11 19:12:16 — tau_nonbn_2arms（补录）

- **cmd**: `az_train.py --init training_history/az_intent/gen0120_warm_cpp.pt（no_bn）--value-only --label-mode abs --label-scale {2,1} --tau 50 --freeze-backbone --device auto`（新数据 336 局）→ 转 BN → 合成
- **output**: `training_history/runs/20260911_191216_tau_nonbn_2arms/output.log`
- **result**: J(tau-abs scale2) / K(scale1) 两个价值头训练完成，**待对战测试**

## 2026-09-11 19:17:08 — tau_nonbn_s4（补录）

- **cmd**: 同 J/K 但 `--label-scale 4`
- **output**: `training_history/runs/20260911_191708_tau_nonbn_s4/output.log`
- **result**: L(tau-abs scale4) 训练完成，**待对战测试**

> 注意：`run_logged.sh` 用追加写入，若日志文件同时被编辑器/linter 重写，条目会丢失
> （这 3 条就是这样丢的）。以后可从 `training_history/runs/<ts>_<name>/cmd.txt` 恢复。

## 2026-09-11 18:39:21 — eval_gn_free_H

- **commit**: `4daace4` (dirty: 22 files)
- **exit**: 0，用时 2495s
- **cmd**:
  ```bash
  D:/anaconda3/envs/pytorch-gpu/python.exe code/my_ai/az_intent/eval.py --checkpoint training_history/az_fixed/mix_r10p_gn_free.pt --opponent rule_v4 --bundle-mcts --iterations 256 --max-depth-rounds 4 --native-engine --t-class 0.5 --t-pos 0.3 --k 24 --games 32 --workers 16
  ```
- **output**: `training_history/runs/20260911_183921_eval_gn_free_H/output.log`
- **result**: `mix_r10p_gn_free.pt` vs rule_v4 32 局 = **18.8%（6W/0D/26L）** → GroupNorm 骨干解冻仍远低于基线 29.7%

## 2026-09-11 19:17:08 — tau_nonbn_s4

- **commit**: `4daace4` (dirty: 22 files)
- **exit**: 0，用时 868s
- **cmd**:
  ```bash
  bash _tmp_tau4.sh
  ```
- **output**: `training_history/runs/20260911_191708_tau_nonbn_s4/output.log`
- **result**: L(tau-abs scale4) 价值头训练完成（value_loss 0.0481），`mix_r10p_tau4_nonbn.pt` 32 局 vs rule_v4 = **12.5%**（见下方 eval_tau4_L）

## 2026-09-11 19:12:16 — tau_nonbn_2arms

- **commit**: `4daace4` (dirty: 22 files)
- **exit**: 0，用时 1475s
- **cmd**:
  ```bash
  bash _tmp_tau_nonbn.sh
  ```
- **output**: `training_history/runs/20260911_191216_tau_nonbn_2arms/output.log`
- **result**: J(tau-abs scale2) / K(scale1) 两个价值头训练完成，产出 `mix_r10p_tau2_nonbn.pt` / `mix_r10p_tau1_nonbn.pt`；之后 eval_ijk 测得 vs rule_v4：**tau2 = 21.9%、tau1 = 9.4%**

## 2026-09-11 19:21:01 — eval_ijk

- **commit**: `e78fb0e` (dirty: 22 files)
- **exit**: 0，用时 7855s
- **cmd**:
  ```bash
  bash _tmp_eval_ijk.sh
  ```
- **output**: `training_history/runs/20260911_192101_eval_ijk/output.log`
- **result**: 三个价值头各 32 局 vs rule_v4：`gn_frozen` **6.2%**（2W/30L）、`tau2_nonbn` **21.9%**（7W/25L）、`tau1_nonbn` **9.4%**（3W/29L）→ 全部低于基线 29.7%，标签格式杠杆无效

## 2026-09-11 21:32:00 — eval_tau4_L

- **commit**: `e78fb0e` (dirty: 22 files)
- **exit**: 0，用时 3217s
- **cmd**:
  ```bash
  D:/anaconda3/envs/pytorch-gpu/python.exe code/my_ai/az_intent/eval.py --checkpoint training_history/az_fixed/mix_r10p_tau4_nonbn.pt --opponent rule_v4 --bundle-mcts --iterations 256 --max-depth-rounds 4 --native-engine --t-class 0.5 --t-pos 0.3 --k 24 --games 32 --workers 16
  ```
- **output**: `training_history/runs/20260911_213200_eval_tau4_L/output.log`
- **result**: `mix_r10p_tau4_nonbn.pt` vs rule_v4 32 局 = **12.5%（4W/0D/28L）**

## 2026-09-11 21:53:46 — pool_arms_5_retry

- **commit**: `953e4fe` (dirty: 16 files)
- **exit**: 0，用时 6047s
- **cmd**:
  ```bash
  bash _tmp_pool_arms.sh
  ```
- **output**: `training_history/runs/20260911_215346_pool_arms_5_retry/output.log`
- **result**: 5 种池化价值头训练完成，产出 `vw_pool_{gapmask,gapmax,region,grid,attn}.pt`（grid 最后一次 epoch loss=0.0614 anchor=0.0099 value=0.0516）

## 2026-09-11 23:35:16 — pool_arms_finish

- **commit**: `953e4fe` (dirty: 18 files)
- **exit**: 0，用时 479s
- **cmd**:
  ```bash
  bash _tmp_pool_finish.sh
  ```
- **output**: `training_history/runs/20260911_233516_pool_arms_finish/output.log`
- **result**: region/grid/attn 价值头转换 + mix 完成 → `mix_r10p_pool_{region,grid,attn}.pt`

## 2026-09-11 23:43:37 — pool_arms_finish2

- **commit**: `18dce3b` (dirty: 17 files)
- **exit**: 0，用时 111s
- **cmd**:
  ```bash
  bash _tmp_pool_finish2.sh
  ```
- **output**: `training_history/runs/20260911_234337_pool_arms_finish2/output.log`
- **result**: gapmax 价值头转换 + mix 完成 → `mix_r10p_pool_gapmax.pt`

## 2026-09-11 23:45:29 — pool64

- **commit**: `2b22086` (dirty: 16 files)
- **exit**: 0，用时 7757s
- **cmd**:
  ```bash
  bash _tmp_pool64.sh
  ```
- **output**: `training_history/runs/20260911_234529_pool64/output.log`
- **result**: 64 局 vs rule_v4：`vw_pol_frozen`（基线）**29.7%**（19W/45L）、`pool_gapmax` **40.6%**（26W/38L）→ gapmax 看着 +11pp，但随后 svs 判定为噪声

## 2026-09-12 01:00:07 — svs_followup

- **commit**: `2b22086` (dirty: 16 files)
- **exit**: 0，用时 18521s
- **cmd**:
  ```bash
  bash _tmp_svs_followup.sh
  ```
- **output**: `training_history/runs/20260912_010007_svs_followup/output.log`
- **result**: search-vs-search（gapmax vs 基线，256/4）：**A win rate = 48.4%（31W/0D/33L）** → 与 50% 无差异，**否决 gapmax 的 +11pp 优势**（第二次验证 vs rule_v4 小局数会出假阳性）

## 2026-09-12 01:55:35 — validate_gapmax

- **commit**: `2b22086` (dirty: 16 files)
- **exit**: 0，用时 26859s
- **cmd**:
  ```bash
  bash _tmp_validate.sh
  ```
- **output**: `training_history/runs/20260912_015535_validate_gapmax/output.log`
- **result**: 补齐 grid/attn 两臂 64 局 vs rule_v4：`pool_grid` **18.8%**（12W/52L）、`pool_attn` **26.6%**（17W/47L）→ 均低于基线 29.7%，池化杠杆全线无效

---

## 2026-09-12 — 价值头空间池化消融：无效（架构杠杆试完）

### 背景

价值头输入曾是 `spatial_feat.mean(dim=[2,3])`（全局平均 → 64 维）。地图是边长 10 的
正六边形（271 格，其中 90 格 VOID），全局平均约 1/3 是死空间；且位置信息被压掉。

### 实现

`network.py` 新增 `value_pool`（默认 `gap` = 原行为，**逐位向后兼容**，已用改动前的
network.py 对比验证 max|diff| = 0）：`gapmask / gapmax / region / grid / attn`。
掩码由 `VALID_CELLS` 固定生成，不进 state_dict、不参与训练。

### 结果（数据 336 局、冻结骨干、no_bn、terminal 标签；策略全部逐位相同 = az_r10）

| pool | 价值头输入 | 64 局 vs rule_v4 | svs（对基线）|
|------|-----------|-----------------|--------------|
| **gap（基线）** | 128 | **29.7%** | — |
| gapmax（平均+最大）| 192 | 40.6% | **48.4%（平手）** |
| gapmask | 128 | 28.1%（32 局）| — |
| attn | 128 | 26.6% | — |
| region（左右半场）| 192 | 25.0% | — |
| grid（4×4）| 1088 | 18.8% | — |

**结论**：**没有一种池化可靠地优于基线**。
- `gapmax` 在 vs rule_v4 上是 40.6%（看着 +11pp），但**配对分析 p≈0.30、svs 48.4% 平手**
  → 判为噪声。
- 空间信息越多越差（grid 1088 维最差 18.8%）：**骨干冻结时，价值头可训参数越多越容易
  过拟合噪声标签**。
- 仅"死区掩码"（gapmask）无用 → 死空间稀释不是瓶颈。

### ⚠️ 方法论结论（第二次验证）

**vs rule_v4 在小局数（32-64）上不可信**（对手是固定规则 AI，方差大，多次出现 +10pp
级别的假阳性)；**search-vs-search 是灵敏判据**（策略锁死、对打配对、方差小）。
之前 C2 头 vs 原版头也是同样情况（svs 53.1% 平手，而 vs rule_v4 差异被误读）。

### 六杠杆总结（全部在同一份自对弈数据上测过）

| 杠杆 | 最好结果 | 是否超越基线 29.7~31.2% |
|------|---------|----------------------|
| **数据质量**（旧 160 局/28% 决定性 → 新 336 局/62%）| **3.1% → 31.2%** | ✅ **唯一有效（10 倍）** |
| 数据量（336 → 536 局）| 25.0% | ❌ |
| 归一化（GroupNorm，冻结/解冻）| 6.2% / 18.8% | ❌ |
| 骨干解冻（no_bn / GN）| 12.5% / 18.8% | ❌ |
| 标签格式（tau-abs s1/2/4/6）| 9.4 / 21.9 / 12.5 / 9.4% | ❌ |
| 池化（5 种）| 18.8% ~ 40.6%（svs 平手）| ❌ |

### 下一步

**唯一剩下的方向是改变数据的性质**——对手池（打不同强度的对手 → 强度不对称 →
对局一边倒 → 血量差大 → 标签清晰），而不是继续调训练/架构。

## 2026-09-12 09:45:04 — pooltau_6arms

- **commit**: `87a5512` (dirty: 15 files)
- **exit**: 0，用时 15171s
- **cmd**:
  ```bash
  bash _tmp_pooltau.sh
  ```
- **output**: `training_history/runs/20260912_094504_pooltau_6arms/output.log`
- **result**: 6 臂（tau=50/scale=2/abs，32 局 vs rule_v4）：**gap 15.6%、gapmask 21.9%、gapmax 31.2%、region 15.6%、grid 9.4%、attn 18.8%**。对照 terminal 基线 29.7%（gap，64 局）。
  - ⚠️ 注意这不是单变量对比：`gap`+tau50(15.6%) 相对 `gap`+terminal(29.7%) 掉了 14pp，但 `gapmax`+tau50 反而到 31.2%。32 局的标准误 ≈8.8pp，15.6 vs 31.2 只差 1.8 SE → **不能下结论**，需 svs
  - 另外 tau-abs 有"抄 stats[1]"捷径（corr 0.937，见 plan 文档），label 线本身存疑
  - 判据待做：svs 64 局 `gapmax+tau50` vs terminal 基线

## 2026-09-12 13:57:55 — pooltau100_6arms

- **commit**: `b09c3c0` (dirty: 16 files)
- **exit**: 0，用时 14216s
- **cmd**:
  ```bash
  bash _tmp_pooltau100.sh
  ```
- **output**: `training_history/runs/20260912_135755_pooltau100_6arms/output.log`
- **result**: 6 臂（tau=100/scale=2/abs，32 局 vs rule_v4）：**gap 12.5%、gapmask 25.0%、gapmax 18.8%、region 18.8%、grid 21.9%、attn 9.4%**。
  - ⚠️ **臂间排名完全不稳**：tau50 里最好的 gapmax(31.2%) 在 tau100 掉到 18.8%；tau50 里最差的 grid(9.4%) 在 tau100 升到 21.9% → 再次印证 32 局粗筛是噪声主导，**单臂不能读**
  - ✅ **但聚合一（各池化均值）是稳的**：terminal 5 臂均值 ≈27.8%、tau50 6 臂均值 ≈18.8%、tau100 6 臂均值 ≈17.7% → **tau 标签整体比 terminal 低约 10pp，且两个 tau 值一致**
  - 与"tau-abs 有抄 stats[1] 捷径（corr 0.937）"的机制分析一致

## 2026-09-12 17:55:04 — poolweight_kgeo_rel

- **commit**: `6900bfd` (dirty: 16 files)
- **exit**: 0，用时 9312s
- **cmd**:
  ```bash
  bash _tmp_poolweight.sh
  ```
- **output**: `training_history/runs/20260912_175504_poolweight_kgeo_rel/output.log`
- **result**: 4 臂（32 局 vs rule_v4）：**kgeo_gap 46.9%（15W/17L）、kgeo_gapmax 12.5%、rel_gap 21.9%、rel_gapmax 25.0%**。
  - `kgeo_gap` 46.9% 是粗筛历史最高（terminal 基线 29.7%），且**全部臂都冻结骨干** → 不是"计算路径幸运盆地"那类不可复现产物
  - 但同一标签下 `kgeo_gapmax` 只有 12.5%（差 34pp）→ 32 局仍然是噪声主导，单臂不能读；判据待 svs（`svs_kgeo_gap_vs_term`）
  - `rel`（唯一把"抄当前血量差"捷径清零的模式）两臂 21.9/25.0%，**没有优于** terminal 基线

## 2026-09-12 20:30:26 — value_stats_only

- **commit**: `c78a68f` (dirty: 16 files)
- **exit**: 0，用时 4290s
- **cmd**:
  ```bash
  bash _tmp_statsonly.sh
  ```
- **output**: `training_history/runs/20260912_203026_value_stats_only/output.log`
- **result**: 2 臂（32 局 vs rule_v4）：**stats_term 15.6%（5W/27L）、stats_tau50 0.0%（0W/32L）**。
  - **棋盘输入对价值头是必要的**：`stats_term` 15.6% 明显低于同协议下 terminal 各池化臂的均值 ≈27.8%（gapmask 28.1/gapmax 40.6/region 25.0/grid 18.8/attn 26.6）→ 拿掉棋盘掉约 12pp
  - **`stats_tau50` 32 局全败 = 项目最差**，且可解释：stats-only（只剩 state）+ tau-abs（88% 方差可由 `stats[1]` 解释）叠加 → 价值头退化成"血量差复读机"（value≈2×stats[1]）→ 搜索变短视贪心、看不见威胁 → 全败
  - 对照：同标签配棋盘输入（`gap`+tau50）是 15.6% → **棋盘输入是防止价值头退化的关键**
  - 修正前一条对"棋盘无贡献"的错误解读（见 `docs/az_value_stats_only_plan.md`）

## 2026-09-12 21:42:50 — svs_kgeo_gap_vs_term

- **commit**: `e84694e` (dirty: 16 files)
- **exit**: 0，用时 8542s
- **cmd**:
  ```bash
  bash _tmp_svs_kgeo.sh
  ```
- **output**: `training_history/runs/20260912_214250_svs_kgeo_gap_vs_term/output.log`
- **result**: search-vs-search 64 局（`mix_r10p_kgeo_gap` A vs `mix_r10p_vw_pol_frozen` B）：
  **A = 60.9%（39W/0D/25L）**。两者策略逐位相同（az_r10）、池化相同（gap），
  唯一差别是价值头标签（`k·γ^k` tau50/abs/s1.5 vs terminal）→ 单变量。
  - **显著性：单侧 p = 0.052、双侧 0.103 → 未达显著**（64 局分辨不了 60% 的效应；要 p<0.05 需 ≥68 局、p<0.01 需 ~136 局）
  - ⚠️ **强先后手不对称**：A=P0 时 **46.9%**（15W/17L）、A=P1 时 **75.0%**（24W/8L），差 28.1pp（交互 z≈2.30、p≈0.021，但属事后拆分）
  - 对局整体后手 P1 胜率 41/64 = **64%**（后手优势明显）
  - → 60.9% 实为"执后手 +28pp、执先手 −3pp"的混合；已排配对补测 `_tmp_svs_kgeo_flip.sh`
    （同种子先后手取反，合成 64 对，配对分析抵消先后手效应）
  - 首轮 svs 结论：**方向有利但未定论**；真伪取决于配对补测

## 2026-09-13 00:05:19 — gen120rep_kgeo

- **commit**: `31e78c7` (dirty: 16 files)
- **exit**: 0，用时 4288s
- **cmd**:
  ```bash
  bash _tmp_gen120rep.sh
  ```
- **output**: `training_history/runs/20260913_000519_gen120rep_kgeo/output.log`
- **result**: 2 臂（gen120 200 局，32 局 vs rule_v4）：**kgeo_gen120 15.6%（5W/27L）、term_gen120 15.6%（5W/27L）**。
  - **同数据下 kgeo 与 terminal 完全相同** → kgeo 在 gen120 上**没有复现**（对照 polonly 上 kgeo 46.9% vs terminal 29.7%）
  - 且 gen120 数据在**当前冻结配方**下整体只有 15.6%，远低于 polonly 的 29.7% → 与历史上"gen120→34.4%"相反（那个是**不冻结 + CPU**配方）
  - 含义：**kgeo 的效果是数据依赖的**，不是通用改进（两臂 value loss 不同 0.0138 vs 0.0483，确认头确实训得不同，5W/27L 相同属 32 局巧合）

## 2026-09-13 01:17:33 — refeval_bn0v_64

- **commit**: `2d13259` (dirty: 16 files)
- **exit**: 0，用时 11657s
- **cmd**:
  ```bash
  bash _tmp_eval_bn0v_ref.sh
  ```
- **output**: `training_history/runs/20260913_011733_refeval_bn0v_64/output.log`
- **result**: 3 个价值头在**同一 64 局协议**下重测（策略全部逐位相同 = az_r10，只差价值头）：
  | 价值头 | vs rule_v4 |
  |---|---|
  | `mix_r10p_bn0v`（**历史"34.4%"最佳头**）| **28.1%（18W/46L）** |
  | `mix_r10p_vw_pol_frozen`（当前基线）| **29.7%（19W/45L）** |
  | `mix_r10p_kgeo_gap`（候选）| **50.0%（32W/32L）** |
  - ✅ **历史"34.4%"头并不更强**（28.1% ≈ 基线 29.7%）→ 历史 43-47%/34.4% 是**小样本产物**（32 局），**我们没有退步**，H3（评估漂移）成立
  - ✅ **kgeo_gap 在同协议下 +20.3pp**（50.0% vs 29.7%），两独立比例检验 z≈2.35、**p≈0.019**；与 svs 的 60.9%（p=0.052）方向一致 → **kgeo 有效信号得到两个判据的支持**
  - 判据状态：svs 未达显著但方向一致；vs rule_v4 64 局达 p≈0.019（但该判据历史上曾给假阳性，故仍需配对补测确认）

## 2026-09-13 04:32:50 — refeval_r69_history

- **commit**: `2d13259` (dirty: 16 files)
- **exit**: 0，用时 9147s
- **cmd**:
  ```bash
  bash _tmp_eval_r69ref.sh
  ```
- **output**: `training_history/runs/20260913_043250_refeval_r69_history/output.log`
- **result**: 历史坍缩模型在同协议 64 局下重测：
  | 模型 | 32 局（当年）| **64 局（本次）** |
  |---|---|---|
  | `az_mix_gen0120p_v50v` | 46.9% | **42.2%（27W/37L）** |
  | `az_mix_r69p_v50v` | 46.9% | **39.1%（25W/39L）** |
  - ✅ 比当前基线（29.7%）**高约 10-12pp** → **当年确实更强，"没有退步"的说法不成立**（修正 `az_collapsed_vs_healthy_paradox.md` 的前一次修正）
  - ⚠️ 但真正的"好头"是 **`v50`**（tau50/scale6/value-passes3），**不是**那个被记成"34.4% 最佳"的 `gen0120_warm_cpp_cpu`（同协议只有 28.1%）
  - 32 局的 46.9% → 64 局的 42.2%/39.1%：小样本把效应抬高了约 5-8pp（符合 ±8.8pp 标准误）
  - 与 `kgeo_gap`（50.0%）比：+7.8pp，但 64v64 检验 z≈0.89、p≈0.37，**不显著**

## 2026-09-13 07:05:23 — svs_kgeo_flip_paired

- **commit**: `3ef32ec` (dirty: 16 files)
- **exit**: 0，用时 8295s
- **cmd**:
  ```bash
  bash _tmp_svs_kgeo_flip.sh
  ```
- **output**: `training_history/runs/20260913_070523_svs_kgeo_flip_paired/output.log`
- **result**: 同种子 0-63 先后手取反重打，与首轮合成 64 个配对：
  | 轮次 | 胜率 | A=P0 | A=P1 |
  |---|---|---|---|
  | 首轮 | 60.9%（39W/25L）| 46.9% | 75.0% |
  | **配对补测** | **53.1%（34W/30L）** | 53.1% | 53.1% |
  | 合并 | **57.0%（73W/55L，n=128）** | — | — |
  - **配对 McNemar：2-0 22 对 / 1-1 29 对 / 0-2 13 对；不一致对 35，精确双侧 p = 0.176 → 不显著**
  - 合并 128 局单侧 p ≈ 0.056（仍未达 0.05；要在 57% 下达 p<0.05 需 ~138 局）
  - ⚠️ **首轮那个"强先后手不对称"（46.9% vs 75.0%）在补测中消失**（53.1% vs 53.1%）
    → 当时事后拆分的 p≈0.021 是小样本噪声，**再次印证小样本事后分层不能当证据**
  - **结论：kgeo_gap 的 svs 效应未获确立**（方向为正但不显著）；vs rule_v4 的 +20.3pp（p≈0.019）与它方向一致，但该判据历史上有假阳性记录

## 2026-09-13 10:08:34 — pv_2x2

- **commit**: `9f10b03` (dirty: 15 files)
- **exit**: 0，用时 1165s
- **cmd**:
  ```bash
  bash _tmp_policy_value_2x2.sh
  ```
- **output**: `training_history/runs/20260913_100834_pv_2x2/output.log`
- **result**: _待填_

## 2026-09-13 10:29:11 — continue_serial

- **commit**: `9420a5b` (dirty: 16 files)
- **exit**: 0，用时 10129s
- **cmd**:
  ```bash
  bash _tmp_continue_serial.sh
  ```
- **output**: `training_history/runs/20260913_102911_continue_serial/output.log`
- **result**: _待填_

## 2026-09-13 14:03:45 — v50pool_npz

- **commit**: `9420a5b` (dirty: 16 files)
- **exit**: 0，用时 12127s
- **cmd**:
  ```bash
  bash _tmp_v50pool_npz.sh
  ```
- **output**: `training_history/runs/20260913_140345_v50pool_npz/output.log`
- **result**: _待填_

## 2026-09-13 17:26:41 — v50label_2x2

- **commit**: `9420a5b` (dirty: 16 files)
- **exit**: 0，用时 10891s
- **cmd**:
  ```bash
  bash _tmp_v50label_2x2.sh
  ```
- **output**: `training_history/runs/20260913_172641_v50label_2x2/output.log`
- **result**: _待填_

## 2026-09-13 20:28:22 — polpool_kgeo_arm

- **commit**: `9420a5b` (dirty: 16 files)
- **exit**: 0，用时 4373s
- **cmd**:
  ```bash
  bash _tmp_polpool_kgeo.sh
  ```
- **output**: `training_history/runs/20260913_202822_polpool_kgeo_arm/output.log`
- **result**: _待填_

## 2026-09-13 22:13:07 — v50pool336_kgeo

- **commit**: `9420a5b` (dirty: 16 files)
- **exit**: 0，用时 5091s
- **cmd**:
  ```bash
  bash _tmp_v50pool336_kgeo.sh
  ```
- **output**: `training_history/runs/20260913_221307_v50pool336_kgeo/output.log`
- **result**: _待填_

## 2026-09-13 23:38:05 — seed_rep_kgeo

- **commit**: `558b23e` (dirty: 16 files)
- **exit**: 0，用时 9184s
- **cmd**:
  ```bash
  bash _tmp_seed_rep.sh
  ```
- **output**: `training_history/runs/20260913_233805_seed_rep_kgeo/output.log`
- **result**: _待填_

## 2026-09-14 02:12:07 — noise_floor_extra

- **commit**: `558b23e` (dirty: 16 files)
- **exit**: 0，用时 8379s
- **cmd**:
  ```bash
  bash _tmp_noise_floor_extra.sh
  ```
- **output**: `training_history/runs/20260914_021207_noise_floor_extra/output.log`
- **result**: _待填_

## 2026-09-14 08:41:52 — sraw_sweep

- **commit**: `ded55c9` (dirty: 17 files)
- **exit**: 0，用时 9350s
- **cmd**:
  ```bash
  bash _tmp_sraw_sweep.sh
  ```
- **output**: `training_history/runs/20260914_084152_sraw_sweep/output.log`
- **result**: _待填_

## 2026-09-14 — search-vs-raw 扫描：迭代/k/深度 的成本曲线（判据仪表定）

### 背景

用户提议：我们不需要"便宜的搜索与完整搜索一致"，只需要它**仍然打得过 raw**
（同一个模型不搜索的那一侧）——把"保真判据"换成"效果判据"。
台账显示该判据动态范围 50%（坏价值头）→ 90%（好价值头）。见
`docs/az_action_factorization_idea.md`。

### 结果（`mix_r10p_vw_pol_frozen`，32 局，每局是"同一模型搜索侧 vs raw 侧"→ 天然配对）

| 配置 | search-vs-raw |
|------|--------------|
| **256/4（基线）** | **93.8%（30W/2L）** |
| 128/4 | 78.1%（25W/7L） |
| 64/4 | 62.5%（20W/12L） |
| **32/4** | **25.0%（8W/24L）** ⚠️ |
| k=8 @256/4 | 75.0%（24W/8L） |
| **depth=2 @256** | **90.6%（29W/3L）** |

### 结论

1. **迭代数不能砍**：32 迭代下 search **比裸策略还差（25%）**——意味着
   "浅搜索 32/2"采集出来的数据，其标签质量是**负的**。
   → 之前计划文档里推荐的浅搜索采集配置**被实测否掉**。（要 ~80% 需 128 迭代，~90% 需 256。）
2. **k 不要动**：k=8 → 75.0%（-18.8pp）。这与"测量①：k=24 时实际只有 ~6 个不同候选"
   的推论矛盾 → 采样行为随模型而变，**该测量不具普适性**。
3. **✅ 深度是唯一免费的速度**：depth 2 → 90.6%（vs 93.8%），几乎无损而成本减半。

### 采集配置（修正后）

```
256 迭代 / **depth 2**（保持强度）+ `--skip-hold-search`（6.8×）
→ ~2.2 分钟/局 → 1000 局 ≈ 2.3 小时
```

**不用退到浅搜索也能上量**——这是"体量"路线上最关键的一个解锁。

### 方法论

`search-vs-raw` 作为"训练是否生效"的判据**经受住了检验**：同一模型、局内配对、
动态范围 50→94%，且循环里策略固定 ⇒ 该指标的变化只来自价值头。

## 2026-09-14 12:46:21 — value_loop

- **commit**: `4bcfd25` (dirty: 15 files)
- **exit**: 0，用时 167s
- **cmd**:
  ```bash
  bash code/run_value_loop.sh 8 300 256 2 3 training_history/az_loop
  ```
- **output**: `training_history/runs/20260914_124621_value_loop/output.log`
- **result**: _待填_

## 2026-09-14 — 价值头迭代循环首跑：**堆量反而削弱价值头**（配置不一致假说）

### 设计

`code/run_value_loop.sh`：每轮用「az_r10 策略 + 当前价值头」做搜索采集 300 局
→ 并入累积池 → 价值头 warm-start 流式续训 3 epoch（kgeo tau50/abs/s1.5，冻结骨干）
→ 记录 search-vs-raw（32 局）+ 每 3 轮一次 vs rule_v4（64 局）。
策略全程固定为 az_r10（纯价值头迭代）。

### 结果（在跑满 3 轮后被手动停止）

| 时点 | 池子 | search-vs-raw（32 局）| vs rule_v4（64 局，seed 0-63）|
|------|------|----------------------|------------------------------|
| 起点 `vw_pol_frozen` | 336 | 90.6% | **29.7%** |
| 训练 1 轮后 | 636 | 90.6% | — |
| 训练 2 轮后 | 936 | 87.5% | — |
| 训练 3 轮后 | 1236 | **81.2%** | **21.9%**（14W/50L）|

**两条独立指标方向一致：都在变差**（-9.4pp / -7.8pp）。
单条各约 1-2σ（32 局 σ≈5pp、64 局 σ≈5.7pp），合起来是可信的负面信号。

### 结论与主因假说

**堆量（在 kgeo + warm-start + 本采集配置下）不但没帮助，反而削弱了价值头。**

主因假说：**采集配置与原始数据不一致**——
原始 336 局是 **256 迭代 / depth 4**，而循环新采的 900 局是 **256 / depth 2**
（我为省一半时间改的）。池子因此成了"两种搜索强度"的混合物，而新数据占 2.7 倍
→ 价值头被拉向"较弱搜索下的价值"，在评测用的更强搜索上反而失准。

### 方法论副产物（两个坑，都已记）

1. **`search-vs-raw` 不适合当"进步"指标**：它起点就 90.6%，天花板只剩 ~9pp，
   32 局 σ≈5pp → **只能探测崩坏，探测不了变好**。（它这次确实探测到了下滑。）
   改进方向：用 **search-vs-search**（当前 mix vs 固定初始 mix）——量程 0-100%、配对。
2. **HOLD-skip 会稀释随机注入**：注入代码在"真正搜索"分支内，而 91% 回合被跳过
   → `--random-action-prob 0.02` 的有效比例只有 ~0.18%。当年那个 2% 换算到
   "有动作的回合"约等于 **22%**，所以要复现同等强度得用 **~0.2**。

### 保留资产

池子 1236 局（39 块，8.8GB）+ `vw_round1/2/3.pt`（每轮快照）+ 全部日志。

### 下一步（已提议）

**配置一致性对照**：新采 300 局 @ **256/depth4**（与原始同配置），
与已有的 300 局 @ depth2（R1 数据）分别训价值头对比 →
若 depth4 那条不再退化，则假说成立，以后采集必须 256/depth4。

## 2026-09-15 — svs 确认：迭代循环确实削弱了价值头（用户判断结果已明确，中途停止）

**设计**：`az_search_vs_search.py` A = 循环训练 2 轮后的价值头（`vw_round2`，就是测出
vs rule_v4 21.9% 的那个）vs B = 初始价值头（`vw_pol_frozen`，29.7%）。
**两边策略逐位相同（az_r10）**，唯一差别是价值头 → 干净配对。64 局，256/4。

**结果（跑 34/64 局后按用户判断停止）**：

| | |
|---|---|
| A 胜率 | **12W / 22L = 35.3%** |
| 先后手拆分 | A=P0 6W/11L（35.3%）、A=P1 6W/11L（35.3%）——**完全一致** |
| 显著性 | 34 局单侧 p≈0.044 |

**结论**：三条证据方向一致（svs 35.3%、search-vs-raw 90.6→81.2、vs rule_v4 29.7→21.9%）
→ **那套迭代（kgeo 标签 + warm-start + 256/depth2 采集的混池）确实在削弱价值头**。

### 方法论教训（三条）

1. **`search-vs-raw` 不能当"进步"指标**：起点就 90.6%，天花板只剩 ~9pp，32 局 σ≈5pp
   → 只能探测崩坏，探测不了变好。应用 **search-vs-search**（配对、量程 0-100%）。
2. **HOLD-skip 稀释随机注入**：注入在"真正搜索"分支内、而 91% 回合被跳过 →
   `--random-action-prob 0.02` 的实际有效比例只有 ~0.18%（当年那个 2% 换算到
   "有动作的回合"约 22%，要等效需 ~0.2）。
3. **合并脚本隔轮覆盖 bug**（已修）：原按子目录名分组，循环第 k+1 轮会覆盖第 k 轮。

## 2026-09-15 00:21:17 — az_bn_kgeo

- **commit**: `4164597` (dirty: 15 files)
- **exit**: 0，用时 28843s
- **cmd**:
  ```bash
  bash code/run_az_bn_kgeo.sh 5 32 5
  ```
- **output**: `training_history/runs/20260915_002117_az_bn_kgeo/output.log`
- **result**: _待填_

## 2026-09-15 — 原始批循环 + BN + kgeo + 2% 注入（5 batch × 32 局）

**动机（用户提出）**：复刻 ~2-3 周前的原始 AZ 批循环，只改三处——① 采集掺
**2% 随机注入**；② **开 BN**；③ 标签用 **kgeo**。策略网和价值网**都训**
（`--split`，非 value-only），网络结构不做拆分。

**配置**：`code/run_az_bn_kgeo.sh`，init=`gen0120_bn_init.pt`（BN），
采集 256 迭代/depth4/C++/16workers（**无 HOLD-skip**，保持与原始一致），
`az_train --split --label-mode abs --label-weight kgeo --tau 50 --label-scale 1.5
--value-passes 3`，每 batch 32 局、5 epochs、共 5 batch，最后 64 局 vs rule_v4。

**结果**：

| 项 | 值 |
|---|---|
| **最终 vs rule_v4（64 局）** | **29.7%（19W/0D/45L）** |
| 对照：当前 no_bn 基线 `vw_pol_frozen` | 29.7% |
| 对照：历史最好（`az_mix_gen0120p_v50v` / `r69p_v50v`）| 42.2% / 39.1% |
| 总耗时 | 8 小时（28843s）|

**训练损失**（每 batch 都收敛）：

| batch | policy CE | value MSE（末）|
|---|---|---|
| b1 | 0.628→0.532 | 0.0055 |
| b2 | 0.584→0.524 | 0.0048 |
| b3 | 0.568→0.506 | 0.0045 |
| b4 | 0.571→0.501 | 0.0045 |

**关键观察**：

1. **三处改动没有改变强度**（29.7% = 基线）。BN+kgeo+2%注入+联合训练 5 batch 后
   与当前 no_bn 基线**完全相同**。
2. **value MSE 低得反常（0.0045-0.0055 ⇒ R²≈0.96）**——远好于 no_bn 价值头
   （0.012-0.017，R²≈0.88）。但**棋力没动** → 又一次印证"**低 loss ≠ 好的搜索价值**"。
   结合"kgeo 标签本身 ~87% 方差可由'抄 stats[1] 当前血量差'解释"（见
   `az_pool_tau_label_plan.md`）：**BN 让价值头把这个捷径拟合得更彻底了，但学到的
   仍不是棋盘理解**。
3. **对局长度稳定**：每 batch 平均 408-430 回合（合计 419.3，中位 420），**逐 batch
   无趋势** → 印证"对局长度由游戏机制决定，与训练/搜索无关"。
4. **后手（P1）稳定占优**：合计 P1 91 胜 / P0 69 胜 = **57%**，与 svs 里测到的
   后手 64% 方向一致（呼应文档"搜索增强集中在后手"）。
5. **2% 注入没有明显提高决定性**：每 batch 47-72%（均值 57%），与历史无注入的
   45-62% 重叠（每 batch 仅 32 局，σ≈9pp）。

## 2026-09-15 — 机制诊断：价值头"用了多少棋盘信息"（stats-probe R²）+ 首次出现真正用棋盘的头

### 为什么要这个诊断

前面的教训：**value MSE 不能当价值头好坏**（BN+kgeo 那轮 MSE 0.0045/R²≈0.96，棋力却
纹丝不动）——因为它把"抄 `stats[1]`（当前血量差）"这个捷径拟合得更彻底而已。
所以需要一个**直接量"这个头用了多少棋盘"**的指标，而且不用打对局。

### 方法（`_tmp_stats_probe.py`，无需对局）

1. 取 60 局（51,184 个状态）真实局面，前向得到各价值头的输出 V；
2. 用一个 MLP（**只用 42 维 stats** → V）在 80% 上拟合、20% 上测 **R²**：
   **R² 越高 ⇒ V 越像"只是 stats 的函数"（棋盘无关）**；
3. 附带报 corr(V, hp_delta)（"复读当前血量差"的程度）。

### 结果

| checkpoint | V std | **stats-probe R²** | corr(V, hp_delta) |
|---|---|---|---|
| `mix_r10p_vw_pol_frozen`（29.7% 基线）| 0.151 | **+0.867** | +0.865 |
| `vw_kgeo_gap`（abs/kgeo）| 0.251 | **+0.920** | +0.922 |
| **`vw_kgeorel_s0`（kgeo+rel, scale 3）** | 0.201 | **+0.583** | **−0.426** |

### 结论（本轮最有价值的发现之一）

- **我们此前的价值头，输出方差 ~87-92% 能被 42 维 stats 单独解释——棋盘只贡献 8-13%。**
  这是"抄 stats[1] 捷径"的定量版本：**价值头基本就是个 stats 函数**。
- **`kgeo+rel` 把 stats-R² 压到 0.583 ⇒ 棋盘贡献升到 ~42%（3-4 倍）**，且
  corr(V, hp_delta) = −0.426（与其标签 −0.314 一致）→ **不是在复读血量差**，
  而是真的学到了"未来变化量"这个关系。
- **`kgeo+rel` 是这条线上第一个让价值头明显依赖棋盘信息的标签。**

### 标签侧的依据

`kgeo+rel` 同时去掉两样：`kgeo`（权重峰移到 k≈tau，排除当前帧）+ `rel`（去掉当前水平）
→ 标签与 hp_delta 的相关性从 `abs/kgeo` 的 **+0.853 变成 −0.314**，**无法靠抄输入拟合**。
（scale 取 3.0 使幅度与对照臂相当；corr 与 scale 无关。）

### 待办

① seeds 1、2 训完 → 看 stats-R² 的下降是否**跨种子稳定**；
② 强度判据：`kgeo+rel` vs 基线（svs 配对）+ vs rule_v4（64 局）。
"用了棋盘" ≠ "下得更好"，必须验强度。

## 2026-09-15 09:47:44 — kgeorel_strength

- **commit**: `bd89cb2` (dirty: 15 files)
- **exit**: 0，用时 20949s
- **cmd**:
  ```bash
  bash _tmp_kgeorel_strength.sh
  ```
- **output**: `training_history/runs/20260915_094744_kgeorel_strength/output.log`
- **result**: **机制确实被改了，强度一点没变。三个读数全部落在噪声内。**
  | 判据 | 结果 | 检验 |
  |---|---|---|
  | ① svs `kgeorel_s0` vs 基线（64 局）| **51.6%**（33W/31L）| 二项 p=0.90 |
  | ③ svs `kgeorel_s1` vs 基线（64 局）| **54.7%**（35W/29L）| 二项 p=0.53 |
  | ①+③ 合并（128 局）| **53.1%**（68W/60L）| 二项 p=0.54 |
  | ② vs rule_v4（64 局）| **21.9%**（14W/50L）| 基线同为 64 局 **29.7%**（19W/45L）|
  ② 的 -7.8pp 看着吓人，但两次跑用的是**同一批 seed（0..63）+ 同一先/后手交替**，
  所以可以逐局配对：**逐局一致率只有 54.7%，29/64 局翻转**（A 单独赢 12 局、B 单独赢 17 局），
  **McNemar 精确双侧 p=0.46** ⇒ 与基线无差异。**只换价值头就翻转近一半对局**，
  这正是本文档反复出现的"噪声地板"的又一次现身（`_tmp_paired_rule_v4.py`）。

  **结论**：`kgeo+rel` 把 stats-R² 从 0.87 压到 0.58、棋盘贡献 13%→42%，
  是一次真实且可复现的**机制**改变；但它**不转化为任何可测强度**
  （灵敏判据 svs 和绝对判据 vs rule_v4 都说没差别）。
  ⇒ **"价值头不听棋盘"不是强度瓶颈**，这条线到此可以收了。

## 2026-09-15 15:38:25 — search_axis_ablation

- **commit**: `e51d8af` (dirty: 18 files)
- **exit**: 0，用时 11363s
- **cmd**:
  ```bash
  bash _tmp_search_axis_ablation.sh
  ```
- **output**: `training_history/runs/20260915_153825_search_axis_ablation/output.log`
- **result**: 见下表。**搜索的增益 100% 在"位置"维，动作类维贡献为 0。**

| arm | 胜率（vs 同模型的 raw，32 局）| 候选空间 | 与 raw 相同 | 类差异 | 位置差异 | 我方每局回合 |
|---|---|---|---|---|---|---|
| ① joint（参照）| **93.8%**（30W/2L）| 99.7% | 97.2% | 0.0% | 2.6% | 388 |
| ② class-only | 53.1%（17W/15L）| 99.2% | **100.0%** | 0.0% | 0.0% | 375 |
| ③ pos-only(argmax) | **93.8%**（30W/2L）| **2.8%** | 97.2% | 0.0% | 2.8% | 393 |
| ④ pos-only(playable) | **0.0%（0W/32L）**| 83.0% | 14.4% | **56.1%** | 29.5% | 269 |

### 判读

1. **② 是"类维无用"的直接证据**：它 99.2% 的回合都有 ≥2 个候选（可以换类），
   但最终选择与 raw **逐回合 100.0% 相同** ⇒ 价值头**从未**认可过一个换类的候选。
   该 arm 实际退化成 raw 打 raw 的镜像对局，53.1% 就是先后手不对称 + 噪声（p=0.86）。
2. **③ 是"位置维承载全部增益"的直接证据**：候选空间只有 2.8%，却**完整复现 joint**
   （不只是同样 93.8%，是同样 30W/2L、连哪两局输都一样）。
3. **④ 不是公平的"只搜位置"测试，是"强制出招"干预**：把类钉成"第一个可执行的类"后，
   策略想 HOLD 时它仍会找一个类去执行 ⇒ 85.6% 的回合与 raw 不同（类差异 56.1%）。
   结果 **0W/32L，且我方血量每局都是 0**（先手 0/16、后手 0/16）——基地被打爆，对局还短 30%。
   ⇒ **策略"放弃这一回合"这个决定本身极有价值**，强制的替代动作是灾难性的。
4. 顺带的成本反例：④ 比 joint 慢 3 倍多（~89min vs ~34min / 32 局），
   而 ③ 比 joint 快。⇒ **成本跟着"候选多样性/树的大小"走，与"搜哪一维"无关**；
   ③ 的"快"是类头坍缩（argmax 常不可执行 ⇒ 只剩 1 个候选、树长不起来）的**副产品**。

### 对"动作类/位置拆开训练"这个想法的推论（`docs/az_action_factorization_idea.md`）

- "训位置模型：类固定、只搜位置"——该数据是**有效**的（③ 成立）；
- "训类模型：位置固定、只搜类"——该数据**不含任何"类该怎么改"的信号**（② 与 raw 逐回合相同），
  拿它训类模型只是把现有类头再学一遍。

**保留**：当前策略的类头**已坍缩**（原始 logits 峰值集中在闪电那一个类上），
"换类从未被选中"很可能是坍缩的后果而非普遍规律；类头训好后此结论需重测。

### ④ 的机制（补测 120 个真实局面，只看不打搜）

| 量 | 值 |
|---|---|
| raw 空手的回合 | **116/120 = 96.7%** |
| 三头**未掩码** argmax 全 HOLD | **0/120 = 0.0%** |
| raw 空手但 ④ 出招 | 111/120 = 92.5% |
| ④ 产生 DOWNGRADE | **0/120** |
| raw 的操作类型分布 | `{USE_LIGHTNING_STORM: 4}` |
| ④ 的操作类型分布 | **`{BUILD_TOWER: 254, USE_LIGHTNING_STORM: 4}`** |
| ④ 三个头钉到的类 | class 2 / 14 / 15 各 111 次（合计 30.8%×3），class 23 共 15，class 17 共 12 |

⇒ **④ 实际是"每回合狂盖基础塔"**（不是拆塔：DOWNGRADE 0 次），
与 raw 的"几乎全程空手、偶尔放一次闪电"完全不同。这解释了 0W/32L：
金币全砸在新塔上、升级与防守跟不上，我方基地每局被打爆。

⇒ 另一个重要事实：raw 的**未掩码** argmax **从未**三头全 HOLD，但它仍空手 96.7%——
说明策略指向的是**不可执行的类**（闪电在冷却/买不起），不是它想 HOLD。
这同时也是 `--skip-hold-search` 能生效的原因（它用的是**掩码后** argmax，
在本文局面集里 81% 命中 HOLD）。

## 2026-09-15 18:49:58 — search_speed_bench

- **commit**: `55908f5` (dirty: 17 files)
- **exit**: 0，用时 740s
- **cmd**:
  ```bash
  D:/anaconda3/envs/pytorch-gpu/python.exe _tmp_search_speed_bench.py training_history/az_fixed/mix_r10p_vw_pol_frozen.pt 96 256 4
  ```
- **output**: `training_history/runs/20260915_184958_search_speed_bench/output.log`
- **result**: 同一批 **96 个真实局面**、256/4 下横比（单进程单线程 ⇒ 只看比值）：

| mode | 总耗时 | 中位 | 展开次数 | sample_bundle | 前向 | root 候选数 | 相对 joint |
|---|---|---|---|---|---|---|---|
| joint | 81.6s | 782ms | 49.9 | 17948 | 257 | 7.62 | 1.00× |
| class-only | 72.7s | 751ms | 23.6 | 8486 | 257 | 2.98 | 0.89× |
| **pos-only(argmax)** | 71.8s | 698ms | 24.2 | 8715 | 257 | 1.72 | **0.88×** |
| pos-only(playable) | 489.6s | 5348ms | 250.2 | 90075 | 257 | 22.80 | **6.00×** |
| **skip-hold** | 17.1s | 0ms | 13.5 | 4860 | 48.2 | 2.64 | **0.21×**（78/96 跳过）|

### 结论

1. **"只搜位置快很多"不成立**：`pos-only(argmax)` 只快 **12%**；`pos-only(playable)` 反而慢 **6×**。
   真正的速度杠杆是 `--skip-hold-search`（本文 4.8×，81% 跳过率；文档记的 6.8× 对应
   ~85-91% 的 HOLD 率，机制一致 ⇒ 成功复现，说明量法可信）。
2. **各模式前向次数完全相同（257 = 1 root + 256 迭代）**：限制搜索轴**不减少模型评估次数**，
   因为 MCTS 每次搜索都跑满迭代。
3. **改哪一维都不省时间，因为开销在"共享的采样次数"上**：`sample_bundle` 次数 = **360 × 展开次数**
   （实测四种模式都精确满足：49.9→17948、23.6→8486、24.2→8715、250.2→90075），
   而总时间基本被这 360 次/展开的开销吃掉（skip-hold 只有 48.2 次前向≈0.1s，却仍花 17.1s）。
   类采样只是 24 类上的一次 `rng.choice`，而且 `_expand` 早已把它向量化预采样
   ⇒ 把它整个砍掉（pos-only）只省 **12%**。
   **保留**：ms/次采样在各模式间不是常数（3.5~8.6ms），因为展开发生在**不同局面**上
   （树的形状不同 ⇒ 落点的网络输出不同），所以"时间"不是采样次数的干净函数；
   能确定的是前向次数恒定、且成本主要跟着展开次数走。
4. **成本随候选多样性（root 候选数）走**：7.62→49.9 展开、1.72→24.2、22.80→250.2。
   ⇒ 与 ④ 比 joint 慢 6×、③ 比 joint 略快一致；**③ 的"快"是"argmax 类常不可执行
   ⇒ 只剩 1 个候选、树长不起来"的副产品，不是"只搜位置"本身的功劳**。
5. 方法论保留：本文前面的 arm 墙钟（~34/~33/~23/~89 分钟）是从快照推的，与实测比值
   对不齐（④ 实为 6× 而非 2.6×）⇒ **墙钟推断不可靠，比值要用同 state 横比测**。

## 2026-09-15 19:47:27 — forced_move_skip_bench

- **commit**: `905121c` (dirty: 19 files)
- **exit**: 0，用时 810s
- **cmd**:
  ```bash
  bash _tmp_forced_move_skip_bench.sh
  ```
- **output**: `training_history/runs/20260915_194727_forced_move_skip_bench/output.log`
- **result**: **端到端 8 局（pos-only）：775s → 35s = 22.1×**；两边都是 7W/1L、
  诊断面板一致（room 2.8%、与 raw 相同 97.2%、差异全在位置维）。

  ⚠️ **7W/1L 相同是巧合，不作证据**：跳过会改变 `mcts.rng` 之后抽到的数，
  对局实现本就会变（决策分布不变）。本项只当冒烟 + 测速。

### 前置的成本模型与等价性验证（决定性证据）

**成本模型**（三条 arm 全命中，误差 <3%）：

| 项 | 实测 |
|---|---|
| 一次 net_fn（encode 0.34 + 模型 2.20）| **2.539 ms** |
| 360×`sample_bundle`（一次展开的候选生成）| **~1.2 ms** |
| **256 次迭代 = 256 次前向** | **653 ms（占 ~85%）** |

⇒ 搜索成本 ≈ `257 × 2.539ms + 展开次数 × 3.7ms`。
验算：joint 836ms（实测 850）、pos-only 741（748）、skip-hold 172（178）。
⇒ **限制搜索轴最多只省 12% 的根因：它减的是那 15% 的候选生成。**

**等价性**（`_tmp_verify_forced_move_skip.py`，330 个真实局面 × 2 模式）：

| search_mode | 单候选局面 | 逐位不一致 | temperature=1.0 | 总耗时 | 单候选回合 |
|---|---|---|---|---|---|
| joint | 10/330（3.0%）| **0** | 0/60 | 279.2→271.6s（1.03×）| 775.0 → **5.0 ms** |
| pos-only(argmax) | 321/330（97.3%）| **0** | 0/60 | 247.3→**9.6s（25.83×）**| 706.8 → **4.8 ms** |

比对字段：`chosen_bundle` / `bundles` / `visit_policy` / `intent_counts`（660 次全等）。
**等价的边界**：这是"同一 RNG 状态下的单次搜索"层面的恒等式；整局层面 RNG 流推进步数
会变 ⇒ 实现的对局会变。详见 `docs/az_forced_move_skip_plan.md` §二。

### 采用方式

**opt-in（默认关）**，成对使用：

```bash
--search-mode pos-only --skip-single-candidate
```

不开默认的理由：joint 只命中 3%（收益 1.03×），却仍扰动 RNG 流 ⇒ 破坏
"同 commit + 同命令 → 同数字"的复现性，不划算；pos-only 命中 97% 且本身已 opt-in。

## 2026-09-15 20:09:42 — collection_cost_bench

- **commit**: `bae9b3a` (dirty: 16 files)
- **exit**: 0，用时 480s
- **cmd**:
  ```bash
  bash _tmp_collection_cost_bench.sh
  ```
- **output**: `training_history/runs/20260915_200942_collection_cost_bench/output.log`
- **result**: 三臂（16 局 / **16 worker** / 256 / depth2 / `--skip-hold-search` / `--write-npz` / 同 seed，
  即**现行生产配置** `run_value_loop.sh:65`）：

| arm | 配置 | 墙钟 | 相对 A | 样本数 | 每局样本 |
|---|---|---|---|---|---|
| **A 现行** | joint + skip-hold | 203s | 1.00× | 13476 | 842 |
| C | pos-only + skip-hold | 196s | 0.97× | 13318 | 832 |
| **B 新方案** | pos-only + skip-hold + **skip-single** | **81s** | **0.40×（2.51×）** | 13192 | 825 |

- **A→C（只换搜索轴）= 0.97×：几乎免费**。与前面的成本模型一致——轴只会省掉
  "候选生成"那 15%，而钱在 256 次迭代的前向上；这里 skip-hold 已经把大部分回合跳掉，
  剩下的回合又基本都是单候选，所以换轴本身几乎不动成本。
- **C→B（加单候选跳过）= 2.42×**，总 **2.51×**，样本数只差 -2%（对局长度方差，非数据丢失）。
- 换算上量（16 worker 线性外推，worker 一多会因抢核变慢、实际要打折）：
  **1000 局 ≈ 3.5h → 1.4h；3000 局 ≈ 10.6h → 4.2h**。
- ⚠️ 用途限制：pos-only 只适合采**价值头**数据（`--write-npz` 丢 bundles/visit，正好只用锚点）。
  采**策略**目标时 pos-only 的 visit 分布与 joint 不同，**不要用**。

### 数据质量：n=16 分辨不出来（不要把差异当结论）

每局 `mean|value_target|`（终局血量差）：

| arm | 每局均值 | 组内 SD | 范围 |
|---|---|---|---|
| A | 0.247 | 0.143 | [0.05, 0.55] |
| C | 0.275 | 0.134 | [0.05, 0.50] |
| B | 0.197 | 0.131 | [0.05, 0.45] |

组内 SD≈0.13 ⇒ 均值标准误 0.033 ⇒ 两臂最大差值 0.078 = **1.7σ（p≈0.09）**，
A↔B 只差 1.1σ ⇒ **三臂在 n=16 下统计上不可区分**。

- 机制上也不该有差别（pos-only 与 joint 在动作上逐位等价、skip 只是短路）
  ⇒ "数据质量相同"是**先验预期**；n=16 既无法推翻也无法证实，要证实需 ~100+ 局。
- 副产品：每局标签质量的组内 **CoV ≈ 55%**（SD 0.13 / 均值 0.2-0.28）
  ⇒ "拿每局标签质量去比较采集配置"这个做法本身噪声极大，必须上量。

## 2026-09-15 — 97% 空过：是策略判断还是类头/解码器坏了？（用户提出，已测）

**动机**：pos-only 的 97.3% 回合是"单候选 = 空 bundle"（见 `az_action_factorization_idea.md`
§回合构成分解）。若这是**类头/解码器**造成的，则位置数据被它卡住（每局只有 ~11 个决策点）；
若是**策略判断**，则位置数据天生就薄。用户经验判断"空过是合理策略"。

脚本：`_tmp_pass_is_strategy_or_collapse.py`（(a)(b)(c)）、`_tmp_pass_value_unbiased.py`
（(a) 的无偏版 + 仪器灵敏度）。样本：raw 自对弈 220 个局面（seed 3/11，stride 2），
其中 pass 回合 214、有选择的回合 6。

### (a) 1-ply 价值比较（无偏）

| 量 | 值 |
|---|---|
| `mean_c v_act(c) - v_pass` | **+0.0134**（SD 0.0207、SE 0.0020）|
| t 统计量 | **+6.73** |
| 偏好出招的回合占比 | **80.6%**（n=108）|
| 对照：max 版（有偏，天然偏正）| +0.0226 |
| 对照：min 版（反方向有偏）| **+0.0065（仍为正）** |

⇒ **不是选择偏差**：整条"出招"分布都相对空过上移（连最差的类都 +0.0065），
均值位移是**同类内跨位置散布（SD 0.0043）的 3 倍**。

**仪器灵敏度对照**（同名类、跨所有合法位置的价值散布，31 个局面）：
跨位置 SD **0.0043**、中位 0.0043。⇒ 价值头对位置的"分辨率"量级是 0.004，
而"出招 vs 空过"的位移是 0.013 ⇒ 位移超出仪器分辨率。
（反过来也是重要事实：位置搜索的增益正是靠 0.004 量级的偏好翻转了 93.8% 的对局
⇒ **价值差的数量级不能直接推断对局影响**。）

**保留**：绝对量级小（+0.0134 ≈ 0.27 HP），是"弱偏好"而非强信号。

### (b) 空过的成因拆解

| 量 | 值 |
|---|---|
| 头次（205 个有合法动作的 pass 回合 × 3 头）| 615 |
| **raw argmax == HOLD 的头次** | **0（一次都没有）** |
| raw argmax 在掩码外（非法类）| **477（77.6%）** |
| raw argmax 在掩码内但解不出动作 | 138（22.4%）|
| HOLD 的 logit 减去"最好的合法非 HOLD 类" | 均值 +0.067、中位 +0.159，**>0 占 78.5%** |

⇒ **空过从来不是"类头选了 HOLD"**：raw argmax **从不**是 HOLD，而是
"离线的类"（闪电在冷却/买不起）。解码器**故意不回退**（`decode_head` 的原始 argmax
非法就跳过这个头）⇒ 这个"首选离线"直接变成空过。
类头的 logit 排序实测是 **【不可用的类】> HOLD >【可用的类】**——可用动作排在最后，
是明显失准的类头；代价是 ~97% 的回合不动作。

### (c) 外部裁判 rule_v4（我们 29.7% vs 它 ~70%）

| 场景 | rule_v4 出招率 |
|---|---|
| 在**我们空过**的回合上 | **70.6%**（n=214）|
| 在我们有选择的回合上 | 100%（n=6）|
| 在所有采样回合上（基线）| 71.4%（n=220）|
| **我方 raw 策略在同样本上** | **2.7%** |

⇒ rule_v4 在我们空过的回合上**没有任何区别**（70.6% ≈ 基线 71.4%），它只是整体高频出招；
而我们的出招率比它低 **约 26 倍**。相对一个能打过我们 70/30 的对手，
"2.7% 出招"是**异常偏低**。

### 判读（三条信号的合力）

- (a) 正面、(c) 强负面信号 ⇒ **倾向"低出招率是缺陷"**，与用户经验判断相反；
- (b) 给出机制：**不是"想 HOLD"，而是"首选离线 + 解码器不回退"**；
- **但 (a) 只是弱偏好**（+0.0134），且 1-ply 用的是价值头自身 ⇒ **游戏级才能定论**。
- 未做的决定性测试：**arm ⑤** = 在空过回合上强制执行"价值头选出的"类（1-ply greedy），
  打 32 局。若 ≥50% ⇒ 空过是缺陷；若像 arm ④ 那样大败 ⇒ 价值头的弱偏好不兑现。

### 与"当前工作流是否正交"

查代码：`value_warmup` 的损失是 `anchor_loss(action_map + head_logits) + λ·value_loss`
（`value_warmup.py:272-277`）——**策略两个头都被 anchor 项按在采集时的输出上**，
真正在学的只有价值头。⇒ 当前工作流**结构上无法**修类头，用户"正交"的判断成立；
但**数据上耦合**：类头一旦修好，出招率 2.7%→~50%，位置数据预算涨 ~26×，
价值头的状态分布也会变。

## 2026-09-15 — 多出招能不能赢？四臂 A/B（不用搜索）—— **答：不能，空过是对的**

脚本 `_tmp_act_or_pass_ab.py`（独立诊断，未改生产代码）。镜像对战：我方用改过的解码、
对手用**现行 raw 解码**，同一策略网。每 seed 打两局（先后手互换）⇒ 配对内抵消。

**先说判据质量**：基线臂（mode=none）**32/32 个 pair 恰好 = 1.0、SE=0**
⇒ 该引擎给定 seed 与动作序列**完全确定**，两局是严格镜像，先后手不对称被精确抵消。
**任何偏离 1.0 都是纯信号。**

| arm | 干预 | 出招率 | 配对得分（1.0=持平）| 结果 |
|---|---|---|---|---|
| **none** | 基线（同 raw）| 2.8% | **1.0000**（SE=0）| — |
| **fallback** | 头 argmax 非法时回退到"掩码内 logit 最高的类"（经典 mask-then-argmax）| 2.8% | **1.0000** | **零行为改变** |
| **fallback-rand** | 触发条件同上，但类**随机**取合法类 | 99.1% | **0.0000**（16/16 双败）| 全败 |
| **value** | 空过时强制用价值网 1-ply 选最好的合法类 | 93.7% | **0.0625**（30/32 双败，t=−21.6）| 近乎全败 |

### fallback 臂"零改变"的机制（关键，纠正我前面的说法）

计数器（35808 个头次）：掩码 argmax == HOLD **90.5%**；剩下 9.5% 触发里
**100% 选的是类 17（闪电）**，而其中 **90.3% 解不出动作**（闪电在掩码内——
`intent_decoding` 只查冷却不查金币——但买不起、且没有可降级塔）。
⇒ **叠加 fallback 也救不了**，因为掩码本身把闪电当合法。

对照 fallback-rand 臂（23838 个头次）：掩码 argmax == HOLD **0.0%**、
触发里 **100% 是类 17**、其中 56.1% 解不出动作。差异的来源是**冷却状态**：

- **被动局**（基线/fallback）：策略一有机会就放闪电 ⇒ 闪电长期在冷却 ⇒ mask[17]=False
  ⇒ 掩码 argmax 落到 HOLD（HOLD 赢过所有可用类）⇒ 空过；
- **主动局**（fallback-rand）：我方每次把 17 换成随机类 ⇒ **闪电从没被真正放过**
  ⇒ 冷却恒为 0 ⇒ mask[17]=True ⇒ 掩码 argmax 恒为 17 ⇒ 又空过（或被我改成随机动作）。

⇒ **类头的顶选永远是闪电**，而闪电几乎永远不可执行 ⇒ 97% 空过被**双重锁死**。
空过**不是一个判断，而是这个头表达不出别的东西**。

### 结论

1. **三个独立的"多出招"干预全部惨败**（fallback-rand 0/64、value ≈3/64、加上 arm ④ 的 0/32）
   ⇒ **用户的经验判断成立：空过对这个模型是最优/最不坏的策略**，损失远小于可用动作的损失。
2. 前面 (a) 的"价值头偏好出招 +0.0134"**不兑现**：按它选类照样 30/32 双败。
   但 value(≈5%) ≫ fallback-rand(0%) ⇒ 价值头**有**信号，只是**远不够赢**。
3. 我前面"解码器不回退是主因"的说法**不完整**：回退了也落回闪电。**主因是类头把
   一个不可执行的类永久排在第一**。
4. **搜索的增益不来自"多出招"**（类差异 0.0%），而来自"**在罕见的出招回合上出得更好**"。
   ⇒ "多出招"是死路；"在那 ~11 个决策点上做得更好"才是位置轴的价值所在。

### 待核

本批镜像局里 **P0 赢 59.4%**（16 局 56.2%、32 局 59.4%），与之前记录的"P1 优势 57%"
方向相反。这条与结论无关（配对已抵消），但值得单独复核一次。

## 2026-09-15 — 更正：t_class 这条路瞄准的是错的靶子（用户提问后重推）

**起因**：用户问"调高 t_class 的意义是什么、我们想判断什么"。重推后发现我原来的推理
（"目标坍缩 → 调 t_class 恢复多样性 → 策略就能学"）**漏了关键一环：多样性必须同时是"更好的"**。

### 预检 ① 结果（真实 AZ 训练目标 = 根节点 visit 分布，60 个局面）

| t_class | 候选数 | **空 bundle 的 visit 质量占比** | bundle 大小 | visit 质量的类分布（前 6）|
|---|---|---|---|---|
| 0.50（当前）| 6.07 | **98.0%** | 0.84 | c17=97.6% c23=2.0% c21=0.1% |
| 1.00 | 20.55 | **84.4%** | 0.95 | c17=75.4% c23=10.3% c21=1.6% c16=1.4% |
| 1.50 | 20.55 | **50.7%** | 0.95 | c17=46.0% c23=12.3% c16=3.5% c21=3.4% |
| 2.00 | 20.55 | **34.4%** | 0.98 | c17=31.1% c23=12.0% c21=4.6% c16=4.4% |

- 比 §17 用**采样分布**估的 84.9%（t=0.5）高 13pp ⇒ **多出来的正是 PUCT 的再集中**，
  印证 §17 自己那句"真实搜索还有 MCTS/PUCT 进一步集中"。**这是首次用训练真正吃到的分布确认坍缩。**
- 所以"调大 t_class **能**把目标多样性打开"这一点**成立**。

### 但关键反证：先验放开之后，搜索还是选空过

同一批评估（t_class=1.0，256/4，真实对局）的诊断：

```
diag room(>=2 cands)=99.8~100%  same-as-raw=97.0~97.4%  diff[class=0 pos=6~9 both=4~5]
```

即：**根节点已有 20.55 个候选、里面真的包含建塔类，但搜索仍有 97.0~97.4% 的回合选择和
raw 完全相同**（类差异 0.0%，类+位置同时变仅 1.2%）。

⇒ **"空过"是价值头（经 MCTS）认可的选择，不是先验压制的结果。**
⇒ **我 2026-09-15 早先给的解释（"类变体先验 ~1/360、PUCT 探索项与先验成正比 ⇒ 机制上翻不上来"）
是不完整的、甚至方向错了**：先验放开之后仍然选空过。

### 因此：t_class 不是我们卡住的那一环

t=1.5 买到的是"**分布更分散的目标**"（空过从 98% 降到 51%，其余 49% 摊在约 19 个真实候选上），
**但空过仍是最大单峰** ⇒ 解码出的动作（argmax）基本仍是空过 ⇒ **行为和强度大概率都不变**。

**真正的静止不动点**是另一件事：类头**表达不出**好动作（顶选永远是买不起的闪电、且解码器不回退，
见 2026-09-15 的 act-or-pass 四臂），价值头**排不出**好动作（它的出招偏好经游戏验证亏光）
⇒ 自对弈只能产出"双方互相空过"的静止局面 ⇒ 数据里几乎没有"动作→结果"信号 ⇒ 自我强化。

### 判据也要换

"多样化的目标是更好还是只是更杂" **不能用 search-vs-raw 判**（它被卡在 90%+ 天花板，
两边都看不出区别——这是本轮预检 ② 的设计失误）。正确的工具是
**svs：t=1.5 的搜索 vs t=0.5 的搜索**（配对、灵敏）。

### 战略含义

既然类头和价值头**都**产不出好动作，好动作只能来自**外部**。

**⚠️ 2026-09-16 更正（用户指出）**：本节原先写着"`docs/distill_rule_v4.md`：从 rule_v4
蒸馏到 58-62% vs rule_v4，这是唯一一次超过 50%"。**这是错的，已撤回。**
`docs/distill_rule_v4.md` 是一份**方案设计**文档；12%/35%/58%/62% 那几个数字写在**一个 python
代码块里**、用来示意"监控输出长什么样"，同文档的"预期效果"表也明写"**预期值** ~50-60%"。
全库**没有** rule_v4 蒸馏的脚本、ckpt 或 run 目录；`training_history` 里带 `rulev4` 的只有两个
**对战评估**（azp6/azp21 vs rule_v4）。⇒ **rule_v4 从未被蒸馏过**（与用户记忆一致）。
权威说法仍是台账那句：**打败 rule_v4（>50%）从未达到**。
（教训：设计文档里的示例/预期数字**不是实测**；我当天把这条误读写进了本日志，
之后又回头引用它当"记录"，属自引自证，已撤回。）

⇒ "蒸馏 / 把 rule_v4 放进对手池"因此是**一个未验证的假设**（理由仍成立：类头需要一个
**好目标**，而自对弈产不出来；rule_v4 是更强、且可廉价查询的策略——项目早期确实做过
**从 example AI 蒸馏**，但那是个远弱于 rule_v4 的起点），**没有先例可援**。

## 2026-09-15 — 位置轴控制实验：增益来自"价值排名"，不是"偏离 argmax 本身"

脚本 `_tmp_position_choice_ab.py`（独立诊断，未改生产代码）。镜像配对（我方 vs 现行 raw 解码，
同一策略网，每 seed 两局先后手互换；基线臂恰好 1.0000/SE=0 ⇒ 引擎确定、配对精确抵消）。
**四个臂的候选集与触发条件完全相同**（都是 pos-only 搜索 `_expand` 出来的那批候选、
出招率全部 2.8%），唯一差别是**从候选里挑哪一个**：

| arm | 挑法 | 配对得分 | 胜率 | 结果 |
|---|---|---|---|---|
| raw | 不搜索，用现行 raw 解码 | 1.0000 (SE=0) | — | 基线 ✓ |
| **search** | 搜索的真实选择（visit 最大）| **1.8125** (SE=0.101, t=+8.06) | **90.6%** | 复现了搜索优势 ✓ |
| **random** | 候选里**均匀随机**挑 | **0.0000** (SE=0) | **0%** | **16/16 双败** |
| **prior** | 按候选**先验**抽，忽略 value/visit 排名 | **0.0000** (SE=0) | **0%** | **16/16 双败** |

### 判读

1. **"偏离 argmax 本身有用"被否证**：随机挑一个候选（都是**合法**动作，因为 `sample_bundle`
   已过 `can_apply_operation`）**输掉每一局**。所以偏离本身不是收益来源。
2. **收益 100% 来自价值/visit 排名**：同样 24 个候选、同样出招率，只有"价值选中的那个"
   能赢 ⇒ **价值头对位置的排序携带真实信息**，位置网络**有真信号可学**。
3. 附带证据：`search` 臂所选候选的**序号分布**（按先验降序）显示，352 个多候选回合里
   只有约 81 次选了先验最高的那个（index 0），其余**选中了更低先验的候选**
   ⇒ 搜索确实在用价值**压过先验**，而不是跟着先验走。
4. 这也修掉了我先前的另一个担心：价值头对位置的"分辨率"只有 SD 0.0043，
   我一度怀疑那个量级没有意义。本实验说明**那个量级的差异确实承载信息**。

### 对"先拆位置网络训练"的意义

用户方案（先只训位置网+价值网、类网留后）的第一步**通过了**：
位置轴是唯一一个「价值头会排序 + 采集便宜（pos-only+skip-single，~25×）+ 目标集中且决定性强」
三件都成立的轴，因此**值得投资**。
仍待定：价值网**是否一起训**（§23 记"训过的价值头有害" 12.5%/9.4% vs 原版价值 34.4%，
另有今天 kgeo+rel 与迭代循环的负面结果 ⇒ 倾向先冻住，做单变量）。

## 2026-09-16 08:47:53 — phase2_posnet

- **commit**: `aa3b556` (dirty: 18 files)
- **exit**: 0，用时 2194s
- **cmd**:
  ```bash
  bash _tmp_phase2_posnet.sh
  ```
- **output**: `training_history/runs/20260916_084753_phase2_posnet/output.log`
- **result**: _待填_

## 2026-09-16 09:26:11 — phase2_posnet_b

- **commit**: `aa3b556` (dirty: 20 files)
- **exit**: 0，用时 1541s
- **cmd**:
  ```bash
  bash _tmp_phase2_posnet_b.sh
  ```
- **output**: `training_history/runs/20260916_092611_phase2_posnet_b/output.log`
- **result**: _待填_

## 2026-09-16 10:55:39 — posnet_loop

- **commit**: `f2ca786` (dirty: 17 files)
- **exit**: 0，用时 16951s
- **cmd**:
  ```bash
  bash code/run_posnet_loop.sh 5
  ```
- **output**: `training_history/runs/20260916_105539_posnet_loop/output.log`
- **result**: _待填_

## 2026-09-16 17:19:44 — posnet_loop_svs

- **commit**: `c745731` (dirty: 18 files)
- **exit**: 0，用时 5089s
- **cmd**:
  ```bash
  bash _tmp_posnet_loop_svs.sh
  ```
- **output**: `training_history/runs/20260916_171944_posnet_loop_svs/output.log`
- **result**: 用**搜索对搜索**镜像重测 5 个 ckpt（第 k 轮搜索 vs 第 0 轮搜索，同 seed 配对）。
  该台 null 臂实测 **1.0000 / SE=0**（两侧用同一全局回合号定 seed ⇒ 对局逐位相同），
  所以它像 raw 镜像一样有零方差对照、但测的是**部署时真正跑的搜索**。

  | 轮 | 搜索镜像 | t | （对照）raw 镜像 |
  |---|---|---|---|
  | null | **1.0000（SE=0）** | — | 1.0000（SE=0）|
  | 1 | 1.0312 | +0.39 | 1.1406 |
  | 2 | **1.1094** | **+1.19** | **1.2969 (t=3.6)** |
  | 3 | 1.0781 | +1.00 | 1.0000 |
  | 4 | 1.0469 | +0.60 | 1.0469 |
  | 5 | **0.9688** | **−0.35** | 1.0312 |

  **结论**：两个判据**同向** ⇒ "raw 镜像测错东西（只看 argmax）"这个假设**作废**
  （第 1 轮搜索侧 1.0312 比 raw 侧 1.1406 还低）。两条曲线都**在第 2 轮见顶后衰减**，
  搜索侧无一显著点（最大 t=1.19）；第 5 轮搜索侧掉到 1.0 以下。
  第 4 轮那个 rule_v4=31.2%（p=0.13）随之可判为**噪声**。

## 2026-09-16 20:07:21 — posnet_convergence

- **commit**: `9cf9aed` (dirty: 19 files)
- **exit**: 0，用时 3004s
- **cmd**:
  ```bash
  bash _tmp_posnet_convergence.sh
  ```
- **output**: `training_history/runs/20260916_200721_posnet_convergence/output.log`
- **result**: 位置网（pos-only）30 epoch × 2 轮：pos_ce 7.117 → 6.290（uniform 基线 ln361 = 5.894）。
  **重要**：pos_ce 全程**高于均匀分布**，即位置网基本没学到东西。原因往下看——同一批数据上
  "网络 argmax vs 搜索选择"一致率：raw(未训) 0.63% / e8 15.62% / e30 28.11% / e80 28.04% /
  e30lr3 27.81%。参考面里**只涉及 1 个动作类（class 17 闪电，100%）** ⇒ 位置目标=闪电落点。
  一致率 28% 是**噪声上限**（同一局面重抽样只有 1.7% 自一致，见 `search_consistency_scaled2`），
  所以网络学到的所谓"28%"其实是把目标记成了数据集的众数/条件均值，不是局面的函数。
  ⇒ 位置线当时**没有合法前提**（目标不可复现）；这一结论在 z-score 修复后被推翻，见 `zscore_agreement`。

## 2026-09-17 07:48:25 — depth0_vs_raw

- **commit**: `9cf9aed` (dirty: 20 files)
- **exit**: 0，用时 100s
- **cmd**:
  ```bash
  bash -c export PYTHONIOENCODING=utf-8; D:/anaconda3/envs/pytorch-gpu/python.exe code/my_ai/az_intent/eval.py --checkpoint training_history/az_fixed/three_mix_r10p_vw_pol_frozen.pt --depth0 --self-raw-opponent --bundle-mcts --native-engine --iterations 256 --max-depth-rounds 4 --k 24 --t-class 0.5 --t-pos 0.3 --games 32 --workers 16
  ```
- **output**: `training_history/runs/20260917_074825_depth0_vs_raw/output.log`
- **result**: `three_mix_r10p_vw_pol_frozen.pt` **depth0（不搜索，直接用网络） vs raw-self：0W/0D/32L，胜率 0.0%**。
  诊断：our-turns=8976，room(>=2 候选)=86.4%，chosen==raw=36.1%，差异归因 class=19.2% /
  pos=1.0% / both=43.6%。⇒ depth0 的差异**绝大部分来自类别轴**，而类别轴被证明是塌缩的
  （见"搜索增益 100% 在位置轴"的消融），所以 depth0 输是意料之中；它同时也说明
  **搜索的价值就在于救回类别轴/位置轴的坏 prior**，而不是锦上添花。

## 2026-09-17 09:48:31 — pos_greedy_vs_raw

- **commit**: `9cf9aed` (dirty: 20 files)
- **exit**: 1，用时 171s
- **cmd**:
  ```bash
  bash -c export PYTHONIOENCODING=utf-8; D:/anaconda3/envs/pytorch-gpu/python.exe _tmp_pos_greedy_vs_raw.py --pairs 16 --workers 16
  ```
- **output**: `training_history/runs/20260917_094831_pos_greedy_vs_raw/output.log`
- **result**: 崩溃，`KeyError: 'value'` —— 三网拆分后 `ThreeNetPolicy` **故意不返回 value**，
  脚本里 `pol(b,s)["value"]` 的旧写法失效。修法：改用 `load_three_models` 的 vmodel 批量评估。

## 2026-09-17 09:51:22 — pos_greedy_vs_raw

- **commit**: `9cf9aed` (dirty: 20 files)
- **exit**: 1，用时 21s
- **cmd**:
  ```bash
  bash -c export PYTHONIOENCODING=utf-8; D:/anaconda3/envs/pytorch-gpu/python.exe _tmp_pos_greedy_vs_raw.py --pairs 16 --workers 16
  ```
- **output**: `training_history/runs/20260917_095122_pos_greedy_vs_raw/output.log`
- **result**: 同上，修完前重跑仍 `KeyError: 'value'`。

## 2026-09-17 09:52:50 — pos_greedy_vs_raw

- **commit**: `9cf9aed` (dirty: 20 files)
- **exit**: 0，用时 73s
- **cmd**:
  ```bash
  bash -c export PYTHONIOENCODING=utf-8; D:/anaconda3/envs/pytorch-gpu/python.exe _tmp_pos_greedy_vs_raw.py --pairs 16 --workers 16
  ```
- **output**: `training_history/runs/20260917_095250_pos_greedy_vs_raw/output.log`
- **result**: **位置维 1-ply 全局价值 argmax vs raw：1.3125（SE 0.1760）**；同台对照 pos-only MCTS = 1.8125。
  即"用价值网在每个候选位置上贪心、完全不做树搜索"就已显著优于 raw ⇒ 位置维的增益主要来自
  **价值网对的候选点排序**，而不是多次迭代的树展开。（后来 z-score 修复后重测降到 1.3750，见 `zscore_strength`。）

## 2026-09-17 10:19:29 — search_consistency_scaled

- **commit**: `9cf9aed` (dirty: 20 files)
- **exit**: 127，用时 4887s
- **cmd**:
  ```bash
  bash -c export PYTHONIOENCODING=utf-8; D:/anaconda3/envs/pytorch-gpu/python.exe _tmp_search_consistency_sweep.py --workers 8 --chunks 2 --max-ref 40
  ```
- **output**: `training_history/runs/20260917_101929_search_consistency_scaled/output.log`
- **result**: **无结果**（exit 127，用时 4887s；output.log 只有开头一行）。原因是我把每次搜索的
  耗时按 61s 估，实际 k×sample_mult 使得扩展成本随候选数放大（k200/it8192 ≈ 5.5 min/搜索），
  于是 10 个任务 ÷8 并行实际要 ~1.7 小时，我提前读了空结果并把它当成"还没跑完"。
  教训：**开跑前先按 k×sample_mult 折算单位成本**再报时长。改用较小 max-ref 重跑见下条。

## 2026-09-17 11:40:58 — search_consistency_scaled2

- **commit**: `9cf9aed` (dirty: 20 files)
- **exit**: 0，用时 934s
- **cmd**:
  ```bash
  bash -c export PYTHONIOENCODING=utf-8; D:/anaconda3/envs/pytorch-gpu/python.exe _tmp_search_consistency_sweep.py --workers 8 --chunks 2 --max-ref 16
  ```
- **output**: `training_history/runs/20260917_114058_search_consistency_scaled2/output.log`
- **result**: 同一局面重抽样两遍（只换 rng seed）的"选中是否一致"（n=16 参照局面，参照集已过滤
  掉 <2 候选的退化局面，退化率 0.0%）：

  | 配置 | 候选数 | 访问/候选 | bundle 同% | 逐头一致% | 头0/1/2 |
  |---|---|---|---|---|---|
  | base k24 it256 tp0.3 | 24.0 | 11 | **0.0%** | **0.0%** | 0/0/0 |
  | k64 it2048 tp0.3 | 64.0 | 32 | 6.2% | 2.1% | 6/0/0 |
  | k64 it2048 tp1.0 | 64.0 | 32 | 6.2% | 2.1% | 6/0/0 |
  | k128 it4096 tp1.0 | 128.0 | 32 | 12.5% | 4.2% | 12/0/0 |

  ⇒ 用户假设方向正确：**k ↑ 一致率 ↑**（0→2.1→4.2%），但**温度 tp0.3→1.0 完全没变化**
  （说明当时温度根本不起作用，因为尺度 bug 已经让分布接近均匀，见下条 `map_quality_effect`）。
  但即使在 k128 下一致率也只有 4.2% ⇒ 目标基本是噪声。**位置线当时没有可学的目标。**

## 2026-09-17 11:42:22 — search_consistency_k200

- **commit**: `9cf9aed` (dirty: 20 files)
- **exit**: 0，用时 1898s
- **cmd**:
  ```bash
  bash -c export PYTHONIOENCODING=utf-8; D:/anaconda3/envs/pytorch-gpu/python.exe _tmp_search_consistency_sweep.py --workers 2 --chunks 2 --max-ref 16 --labels k200
  ```
- **output**: `training_history/runs/20260917_114222_search_consistency_k200/output.log`
- **result**: **k=200 / it=8192 / tp=1.0（访问/候选 41）：bundle 同 56.2%，逐头一致 18.8%（头0 56%，
  头1/2 0%）。** 这是"候选数不够"假设的**决定性证据**：k 从 24 拉到 200，一致率 0% → 18.8%。
  但成本墙：k200/it8192 ≈ **5.5 min / 次搜索**，无法用于批量采集（k200 这一条就跑了 1898s，
  且只是 2 个 chunk × max-ref 16）。
  ⇒ 结论：**"靠加大 k/迭代提高一致率"方向正确但不可行**；必须从**先验质量**入手
  （当时先验塌成均匀，有效候选数 271，见下条）。头0 56% 而头1/2 0% 说明增益集中在闪电落点那一头。

## 2026-09-17 19:47:19 — map_quality_effect

- **commit**: `9cf9aed` (dirty: 20 files)
- **exit**: 0，用时 1268s
- **cmd**:
  ```bash
  bash _tmp_map_quality.sh
  ```
- **output**: `training_history/runs/20260917_194719_map_quality_effect/output.log`
- **result**: 用**不同训练程度**的地图（three_mix 未训 / e8 / e30 / e80）跑同一套一致率诊断：

  | 地图 | 候选数 | bundle 同% | 逐头一致% | 地图 top1 | 地图 top5 | n |
  |---|---|---|---|---|---|---|
  | three_mix(未训) | 24.0 | 2.5% | 0.8% | 0.4% | 1.9% | 40 |
  | e8 | 24.0 | 2.5% | 0.8% | - | - | 40 |
  | e30 | 24.0 | 5.0% | 1.7% | - | - | 40 |
  | e80 | 24.0 | 0.0% | 0.0% | - | - | 40 |

  **关键**：CE 从 16.7 训到 4.5（e80）、地图"变尖"了，但一致率**没有变好**（甚至 0.0%），
  且候选数**恒等于 24.0**。这直接暴露尺度 bug：采样器用 `map/100/t_pos`，而 CE 用 `map/t_pos`，
  两者差 100 倍 ⇒ **搜索里的采样器看到的是近似均匀的分布，地图训得多好都没用**
  （有效候选数 271.0，且 e80 之后**仍精确为 271.0**）。同时"地图 top1/top5"列也显示未训地图
  的 top1 只有 0.4%。⇒ 这就是 z-score 修复的动机。

## 2026-09-17 22:24:53 — zscore_agreement

- **commit**: `9cf9aed` (dirty: 21 files)
- **exit**: 0，用时 110s
- **cmd**:
  ```bash
  bash -c export PYTHONIOENCODING=utf-8; D:/anaconda3/envs/pytorch-gpu/python.exe _tmp_search_consistency_sweep.py --workers 4 --chunks 2 --max-ref 16 --labels base
  ```
- **output**: `training_history/runs/20260917_222453_zscore_agreement/output.log`
- **result**: **z-score 修复后同一套一致率诊断（base k24 it256 tp0.3，n=16）：bundle 同 87.5%，
  逐头一致 83.3%（头0 88% / 头1 75% / 头2 88%），候选数 22.3，退化率 0.0%。**
  对比修复前（`search_consistency_scaled2` 同一配置）：**bundle 0.0% / 逐头 0.0%**。

  ⚠️ **2026-09-18 更正（用户指出）**：我在这里原本写"搜索目标**第一次**从'抽样运气'变成
  '局面的函数'，位置蒸馏这才有了合法前提" —— **这句是过度解读，撤回**。这两个数是在
  **温度语义已经换过之后**、且 **t_pos=0.3 这个极尖档**上测的（有效候选 10/270、top1=0.47），
  候选被压在先验偏好的一小片区域里 ⇒ **两遍重抽自然一致**。它不是"搜索的决定变成了局面的
  函数"，而是**"采样器在复读地图自己的偏好"**；而那个偏好本身是错的（峰对齐价值 0/22、
  该配置打 rule_v4 只有 1.6%），所以它**不能当作"目标可学"的证据**。
  另一个易被误读的点：那一批参照局面的候选数平均 **22.3（不是 1）**，所以不是字面上
  "只能采到 argmax"；受限于现有数据，**"候选集几乎确定 + 访问真的做了决定（一个可复现但
  很差的偏好）" 与 "访问没决定、两遍都选了先验最高的那个" 这两种解释没能分开**。

  **一个未清的混杂（同样重要）**：一致率是同时被 ①两遍候选集的重叠度 和 ②访问的决定性
  卡住的。k=24 时 24 个候选散在 ~173 个有效格上 ⇒ 两遍平均只重叠两三个格子 ⇒ 一致率
  **结构性**上不去，与访问准不准无关。修复前那组扫描（k24/it256→k64/it2048→k128/it4096→
  k200/it8192：一致率 0%→2.1%→4.2%→**18.8%**）**把 k 与迭代数一起加大了，分不清是哪个
  起的作用**。⇒ "visit 目标不是局面的函数"这个结论**证据不足**，需要 k×迭代数二维扫描来判。

## 2026-09-17 22:24:53 — zscore_strength

- **commit**: `9cf9aed` (dirty: 21 files)
- **exit**: 0，用时 156s
- **cmd**:
  ```bash
  bash -c export PYTHONIOENCODING=utf-8; D:/anaconda3/envs/pytorch-gpu/python.exe _tmp_position_choice_ab.py --checkpoint training_history/az_fixed/three_mix_r10p_vw_pol_frozen.pt --mode search --pairs 16 --workers 8 --seed 0
  ```
- **output**: `training_history/runs/20260917_222453_zscore_strength/output.log`
- **result**: **z-score 修复后同台配对强度（search vs raw，16 pairs / 32 局）：
  配对得分 1.3750（SE 0.1250，t=+3.00），6/16 pair 偏离 1.0，净增 +6.0。**
  修复前同脚本同参数 = **1.8125**。⇒ **修复让搜索变弱了（1.81 → 1.38）**，但仍显著优于 parity。

  机制（**已更正**，见下条 tpos_prior_quality / tpos_strength）：早先我按
  `候选数分布(前5) = [(1, 11760), (24, 187), ...]` 里"98% 只剩 1 个候选"推断
  "搜索退化成 argmax" —— **这条是错的**。`(1, N)` 那一桶就是**空过回合**（根本没得选），
  `(24, N2)` 才是出招回合，而 N2/N ≈ 2.8% 恰好等于出招率：**出招回合永远是满 k=24 个候选**。
  正确机制是：尖先验把 24 次采样**全部限制在先验偏好的那一小片区域**里，而那片区域
  与价值的偏好不重合（峰对齐价值 **0/22**，见下条）⇒ 搜索搜不到价值最好的格子 ⇒ 强度掉。
  把 t_pos 调回近均匀就恢复全图搜索能力（1.0 → 1.8125，2.0/4.0 → 2.0000，见 tpos_strength）。
  ⇒ 那个 100× 尺度错配**等价于一个"温度极大 ⇒ 先验失效"的探索代理**（标定见下条）；
  修复本身是对的（目标首次可复现），但它不是终点：`t_pos` 语义现在才第一次有意义。

## 2026-09-17 23:01 — tpos_prior_quality（手工跑，未走 run_logged）

- **cmd**: `_tmp_prior_quality.py --ckpts .../three_mix_r10p_vw_pol_frozen.pt --ref-games 10 --max-ref 80 --t-pos-list 0.3,0.5,1.0,2.0,4.0`（1 分多钟）
- **result**: 地图的**采样集中度**与**峰是否对齐价值**随温度的变化（22 个可用参照局面）：

  | t_pos | top1 | top5 | 有效候选 | **峰对齐价值** |
  |---|---|---|---|---|
  | 0.3 | 0.47 | 0.79 | 10.0 | **0%** |
  | 0.5 | 0.21 | 0.45 | 47.7 | **0%** |
  | 1.0 | 0.04 | 0.14 | 172.8 | **0%** |
  | 2.0 | 0.01 | 0.05 | 242.1 | **0%** |
  | 4.0 | 0.01 | 0.03 | 263.2 | **0%** |

  1. **标定：修复前的 t_pos 是个完全失效的旋钮。** 早先我按"t_pos≈4 的有效候选 263 对上
     修复前的 271"推断"等价于 t_pos≈4" —— **那条推断错了**（两端都在饱和区，无法反解）。
     直接量了 `action_map` 在合法格上的 std（`_tmp_pos_scale_probe.py`，n=120 头槽位）：
     **std = 0.26**（中位数 0.258）。于是旧采样器的 logit 宽度 = `std(map)/(100·t_pos)`
     = **0.26/30 ≈ 0.0087**（t_pos=0.3）⇒ softmax 基本是平的。
     换算成新语义：`t_pos_new = t_pos_old × 100/std = t_pos_old × 381`
     ⇒ 旧的 0.3 / 1.0 相当于新的 **114 / 381**，也就是**无穷大**。
     这正好解释了 `map_quality_effect` 那次「tp0.3 → tp1.0 结果逐位相同」：**t_pos 当时不生效**。
  2. **地图的峰在任何温度下都不落在价值网偏好的格子上（0/22）**。所以修复不只是"把探索
     去掉了"，而是**把探索换成了一个方向错的信号**——这才是 1.8125 → 1.3750 的直接原因。
     反过来说，0% 对齐也正是"蒸馏有东西可学"的前提（地图现在的位置偏好是错的）。
  3. 因此 **t_pos=0.3 在修复后是一个"从未真正用过"的档位**（有效候选 10 / ~270 合法格，
     top1=0.47），而"轻度尖锐"区（t_pos≈0.5，48 个）从未被测试过 —— 见 `tpos_rule_v4`。
- **顺带**：本条的参照局面筛法改了（原先按"搜索候选数>=2"筛，修复后那样筛出来极小且
  系统性偏向探索量大的配置 ⇒ 改为按"头 0 的钉类有 >=2 个合法格"筛，推演只用 raw 策略），
  `_tmp_search_target_similarity.py` 的参照筛法同步改成"按被测配置自己筛"。

## 2026-09-17 23:34:42 — tpos_similarity

- **commit**: `fa649c3` (dirty: 20 files)
- **exit**: 0，用时 462s
- **cmd**:
  ```bash
  bash -c export PYTHONIOENCODING=utf-8; D:/anaconda3/envs/pytorch-gpu/python.exe -u _tmp_search_target_similarity.py --workers 5 --ref-games 8 --max-ref 40 --t-pos-list 0.3,0.5,1.0,2.0,4.0
  ```
- **output**: `training_history/runs/20260917_233442_tpos_similarity/output.log`
- **result**: 软目标可复现性 × t_pos（同一局面重抽两遍、只换 rng seed）。参照集=**被测配置自己**
  筛出的"有得选"回合（各配置走同一批轨迹，只是"记哪些回合"不同），所以每行都是 40 个可用参照：

  | t_pos | 先验有效候选 | bundle 同% | **目标余弦** | top5 重叠 | 选中格份额 |
  |---|---|---|---|---|---|
  | 0.3 | 10 | 80.0% | **0.966** | 78.8% | 49.8% |
  | 0.5 | 48 | 77.5% | 0.858 | 70.3% | 29.7% |
  | 1.0 | 173 | 42.5% | 0.350 | 34.0% | 11.2% |
  | 2.0 | 242 | 15.0% | 0.106 | 17.7% | 8.8% |
  | 4.0 | 263 | 2.5% | **0.034** | 15.0% | 4.9% |

  **与强度并排看是单调权衡，没有中间地带：**

  | t_pos | 强度 vs raw（下条）| 目标余弦 |
  |---|---|---|
  | 0.3 | 1.3750 (SE 0.1250) | 0.966 |
  | 1.0 | **1.8125 (SE 0.1008)** | 0.350 |
  | 2.0 | **2.0000（32/32 全胜）** | 0.106 |
  | 4.0 | **2.0000（32/32 全胜）** | 0.034 |

  ⇒ 我原先担心的"软指标比 argmax 宽松、可能救回来"**被否掉**：cosine 掉到 0.034（≈正交）、
  top5 15%（361 格里随机取 5 个是 1.4%，只是略高于随机）。
  **但这不等于"管线坏了"**：cosine 低的原因是 t_pos 大时**有很多都在 raw 之上的好格子**
  （选中格只占该头 visit 的 5% ⇒ 分布摊在 ~20 个格上），具体选哪个由采样运气定，
  但随便哪个都比 raw 强 ⇒ 强度高（32/32）而单次目标不可复现。
  即：不稳定反映的是「相对 raw 的优势是平的/多峰的」，不是噪声。
  代价是**逐样本目标不可学**（每次要教网络记 20 个任意格子），只有**跨样本的期望**才有意义。

## 2026-09-17 23:34:42 — tpos_strength

- **commit**: `fa649c3` (dirty: 20 files)
- **exit**: 0，用时 599s
- **cmd**:
  ```bash
  bash _tmp_tpos_strength.sh
  ```
- **output**: `training_history/runs/20260917_233442_tpos_strength/output.log`
- **result**: t_pos 扫描的强度臂（同一对局台，两侧共用同一 t_pos ⇒ 镜像严格对称）：

  | t_pos | 配对得分 | SE | t | 偏离 pair | 出招率 | 候选数分布 |
  |---|---|---|---|---|---|---|
  | 0.3（上条）| 1.3750 | 0.1250 | +3.00 | 6/16 | 2.8% | {1:11760, 24:187} |
  | 1.0 | **1.8125** | **0.1008** | **+8.06** | 13/16 | 2.8% | {1:12147, 24:350} |
  | 2.0 | **2.0000** | 0（32/32 全胜）| — | 16/16 | 2.8% | {1:11923, 24:342} |
  | 4.0 | **2.0000** | 0（32/32 全胜）| — | 16/16 | 2.8% | {1:11998, 24:347} |

  **① t_pos=1.0 把修复前那个 1.8125 逐位复现了**（SE 0.1008、t=+8.06 都一样）：所以那个
  100× 尺度错配**纯粹是个（非常贵的）探索代理**，现在换成一个有语义的旋钮能等价复现。
  **② t_pos=2.0/4.0 更强：32 战全胜 raw**（SE=0 是"每对都 2 分"导致的方差为 0，不是镜像抵消）。
  这比历史最好（`depth=2 @256` 的 90.6% = 29W/3L）还高一档。
  **③ 历史 90.6% 那个数字是在"错配探索"下测的，等价于 t_pos≈4**；换成 t_pos=2.0 后同台更强，
  说明**这个旋钮之前一直没被调过、我们一直是随机取的一个值**。
  ⚠️ 但"32/32 vs raw-self"**不能当作绝对强度的证据**（raw-self 弱），下一步必须去测 vs rule_v4。
  ⚠️ 另有一处未解的口径差：`pos_greedy_vs_raw`（1-ply 价值 argmax over **全部**合法格，
  确定性）只有 **1.3125**，低于"24 个采样候选里按 visit 选"的 2.0000。合理猜测是
  **visit-argmax 是比"对 270 个噪声估计取 max"更稳健的估计器**（后者有 winner's curse），
  但这只是猜测，未验证。

## 2026-09-17 23:47:12 — tpos_rule_v4

- **commit**: `448e001` (dirty: 17 files)
- **exit**: 0，用时 1067s
- **cmd**:
  ```bash
  bash _tmp_tpos_rule_v4.sh
  ```
- **output**: `training_history/runs/20260917_234712_tpos_rule_v4/output.log`
- **result**: t_pos 三档打 rule_v4，各 64 局（同 seed=0 ⇒ seeds 0..63、our_player=s%2，逐局配对可比）：

  | t_pos | 赢率 | W/D/L | 先验有效候选 | 诊断：与 raw 不同的回合占比 |
  |---|---|---|---|---|
  | 0.3 | **1.6%** | 1/0/63 | 10 | pos 0.6% |
  | 1.0 | **20.3%** | 13/0/51 | 173 | pos 1.6% |
  | 2.0 | **20.3%** | 13/0/51 | 242 | pos 2.3% |

  配对检验（`_tmp_paired_rule_v4.py`，McNemar 精确；class 差异恒为 0.0%，即所有差异都在位置轴）：
  * **0.3 vs 1.0：p = 0.0005，n_discordant = 12，12 个翻转全部朝向 1.0**（"只有 0.3 赢" = 0 局）
  * **1.0 vs 2.0：p = 1.0000，11:11 翻转** ⇒ 不可分辨（赢率相同，但逐局只有 65.6% 一致，
    所以不是同一批对局，只是同类）

  **① 那个 1.6% 的罕见坏结果来自"极尖先验"，不是来自 z-score 修复本身**：t_pos≥1.0
  就把历史水平（修复前 pos-only 的 18.8%）拿了回来。
  **② 必须更正：镜像台"32/32 vs raw-self"是假信号。** 换真强对手后 t_pos=2.0 只有 20.3%，
  落在历史 18.8%~29.7% 区间内 ⇒ **绝对强度没有提升**。这印证了项目早就记过的
  "search-vs-raw 不能当进步指标（起点就 90.6%，天花板只剩 ~9pp）"；我上一轮引用它是错的。
  **③ 1.6% 这个量级说明：尖先验不只是"少一点探索"，而是主动把决策推向错误的位置**
  （峰对齐价值 0/22，见 tpos_prior_quality）。
  **④ 仍未测**：t_pos∈(0.3, 1.0) 的"轻度尖锐"区间（用户提出的问题）；以及为什么本次
  pos-only 的 20.3% 低于历史 joint 的 29.7%（历史配对检验是 p=0.21，可能只是噪声）。

## 2026-09-18 00:06:18 — tpos_rule_v4_mild

- **commit**: `34896cd` (dirty: 17 files)
- **exit**: 0，用时 686s
- **cmd**:
  ```bash
  bash _tmp_tpos_rule_v4_mild.sh
  ```
- **output**: `training_history/runs/20260918_000618_tpos_rule_v4_mild/output.log`
- **result**: 补上"轻度尖锐"区间（用户提出的问题："0.3 是否只是太尖、轻度尖锐应该正常"）
  —— 答案：**轻度尖锐也不正常**。完整曲线（各 64 局打 rule_v4，同 seed 逐局配对）：

  | t_pos | top1 | top5 | 有效候选（共 ~270 合法格）| vs rule_v4 | W/D/L |
  |---|---|---|---|---|---|
  | 0.3 | 0.47 | 0.79 | 10 | **1.6%** | 1/0/63 |
  | 0.5 | 0.21 | 0.45 | 48 | **7.8%** | 5/0/59 |
  | 0.7 | 0.10 | 0.25 | 107 | **18.8%** | 12/0/52 |
  | 1.0 | 0.04 | 0.14 | 173 | **20.3%** | 13/0/51 |
  | 2.0 | 0.01 | 0.05 | 242 | **20.3%** | 13/0/51 |

  逐段 McNemar（翻转数 = "仅低 t_pos 赢 : 仅高 t_pos 赢"）：
  0.3→0.5：1:5（p=0.219）；0.5→0.7：2:9（p=0.065）；0.7→1.0：10:11（p=1.000）；
  1.0→2.0：11:11（p=1.000）。**每一段的翻转都朝向高温侧，平台从 t_pos≈0.7 开始。**

  **机制（比"先验方向反了"更准确）**：`k=24` 在两种温度下都只给 24 个候选，差别不是
  候选数量而是**这 24 个铺在哪**。高温时散布全图 = 24 次不同区域的探测；低温时挤在先验的
  那一小片 = 对同一个区域采样 24 次 —— 那个区域一旦不好就**逃不出去**。对空间相关的价值场
  这是**多样性塌缩**，是结构性的损害，不需要假设先验"反向"；而 `峰对齐价值 = 0/22` 只是
  说明先验也没有补偿性的好处。
  ⇒ **温度轴救不了"先验内容差"这个病**；平台 20.3% 与历史 18.8%~29.7% 一致 ⇒ 绝对强度未提升。
  ⇒ 据此把全项目的 `t_pos` 默认/硬编码值从 0.3 改为 1.0，见 `docs/az_t_pos_default_fix.md`。

## 2026-09-18 08:43:20 — grid2d_agreement

- **commit**: `eb2cfcc` (dirty: 19 files)
- **exit**: 1，用时 587s
- **cmd**:
  ```bash
  bash -c export PYTHONIOENCODING=utf-8; D:/anaconda3/envs/pytorch-gpu/python.exe -u _tmp_search_consistency_sweep.py --grid2d --workers 8 --chunks 2 --ref-games 8 --max-ref 24
  ```
- **output**: `training_history/runs/20260918_084320_grid2d_agreement/output.log`
- **result**: **中止**（exit 1 = 被按 PID 杀掉；用户判断这个实验不需要，14 个进程全部清理，剩余 0）。
  设计留档：在 t_pos=1.0 上做 k×迭代数二维扫描（k24 固定下 it 256/1024/4096，外加
  k96/it1024）以分离"一致率被候选集重叠度还是被访问噪声卡住"。
  用户给出了机制解释并已被我核对（k64 与 k128 访问/k 同为 32 而一致率 2.1%→4.2% ⇒ 重叠是
  独立因素；k128→k200 两者同时上升 ⇒ 跳到 18.8%；修复前四格数据全吻合），据此**不必再跑**。

## 2026-09-18 08:43:20 — grid2d_similarity

- **commit**: `eb2cfcc` (dirty: 19 files)
- **exit**: 1，用时 589s
- **cmd**:
  ```bash
  bash -c export PYTHONIOENCODING=utf-8; D:/anaconda3/envs/pytorch-gpu/python.exe -u _tmp_search_target_similarity.py --grid2d --workers 4 --ref-games 8 --max-ref 24
  ```
- **output**: `training_history/runs/20260918_084320_grid2d_similarity/output.log`
- **result**: **中止**（同上，与 grid2d_agreement 一起被杀）。

## 2026-09-18 09:05:10 — distill_prior_rule_v4

- **commit**: `eb2cfcc` (dirty: 19 files)
- **exit**: 0，用时 1261s
- **cmd**:
  ```bash
  bash _tmp_distill_prior_rule_v4.sh
  ```
- **output**: `training_history/runs/20260918_090510_distill_prior_rule_v4/output.log`
- **result**: 检验用户假设"现在的动作网被遗传/BC 训坏了，而直接拟合 SDK 启发式评分的蒸馏版
  位置图可能反而更好"。做法：把三网 ckpt 的 `pos_state` 换成蒸馏模型的 `state_dict`
  （**94/94 张量键名与形状完全对齐，0 不匹配** ⇒ 换位置网技术上是换一个键，
  `_tmp_make_swapped_pos_ckpt.py`）。打 rule_v4 各 64 局、同 seed 可与既有臂逐局配对：

  | 位置先验 | t_pos=1.0 | t_pos=0.3 |
  |---|---|---|
  | **我们的（原位置网）** | **20.3%**（13W/51L）| 1.6%（1W/63L）|
  | 蒸馏 V1（`distill_data/model.pt`）| 9.4%（6W/58L）| 3.1%（2W/62L）|
  | 蒸馏 V2（`distill_data_v2/model.pt`）| **1.6%**（1W/63L）| 0.0%（0W/64L）|

  配对 McNemar（同 seed 逐局）：
  * **我们 vs V2 @tp1.0：20.3% vs 1.6%，翻转 12:0，p = 0.0005**（高度显著）
  * 我们 vs V1 @tp1.0：翻转 11:4，p = 0.119（方向一致、不显著）
  * 我们 vs V1 @tp0.3：1.6% vs 3.1%，翻转 1:2，p = 1.000（一样烂）

  **⇒ 假设被否掉**：蒸馏版的位置图**不是**更好的先验，反而更差；我们的（看起来最"可疑"的）
  那个是五个候选里最好的。而且"锐化就崩"在三种先验上**都成立** ⇒ 这不是"我们的先验特别烂"，
  而是**手上没有任何先验好到能承受锐化**。

  **顺带两件更重要的事**：
  1. 🔴 **两个对齐"代理指标"都不能用来判断先验好坏**：逐格精确命中（原先 0/22）在"价值前 ~20 格
     近乎并列"时上限只有 ~5%，n=22 测不出东西——五个先验全 0/22 是判据的错；换成参数无关的
     **峰的价值分位**（随机 50%）后五个先验是 30.3 / 28.6 / 39.7 / 22.3 / 33.6%，**都显著优于
     随机，但排序与实际棋力排序不符**（V1 分位 28.6% 看着优于我们 30.3%，棋力却差得多）。
     ⇒ **要判断"先验好不好"，只有打棋力这一条读数有效。**（计划文档的主判据已据此改写。）
  2. ⚠️ 这条结果给"先验替代搜索"的终点**画出一个上界隐患**：那个终点最理想的先验就是
     **价值网自己**，而"用价值网的偏好筛小范围"恰好就是 `pos_greedy` 干的事
     （对全部合法格取价值 argmax）= **1.3125**，低于搜索的 2.0000。若这是真的，
     **再好的先验也追不上一"多候选 + 访问累积"的搜索**，因为搜索的强度可能来自
     "在许多近并列候选上做方差缩减"，而确定性的 top-k 复现不了这件事。
     （⚠️ 这两个数来自不同台子，正是那条未结的口径差。）

## 2026-09-18 13:50:10 — value_prior_ab

- **commit**: `90cb901` (dirty: 20 files)
- **exit**: 0，用时 1615s
- **cmd**:
  ```bash
  bash _tmp_value_prior_ab.sh
  ```
- **output**: `training_history/runs/20260918_135010_value_prior_ab/output.log`
- **result**: **八臂全部在同一个台子里**（`_tmp_position_choice_ab.py`：镜像配对、对手=同一 ckpt 的
  raw 解码、同 seed ⇒ 逐局可比）。新增三条臂靠给 `BundleMCTS` 加一个**可选**的位置先验回调
  `pos_prior_fn`（默认 None ⇒ 现有行为一字不变），以及 `--k` / `--pos-prior value`。

  | 臂 | 配对得分 | SE | t |
  |---|---|---|---|
  | A 基线（位置网先验，k=24，tp1.0）| 1.8125 | 0.1008 | +8.06 |
  | A 尖先验（tp0.3，k=24）| 1.3750 | 0.1250 | +3.00 |
  | A 缩 k=8（位置网先验）| 1.6875 | 0.1197 | +5.74 |
  | B **价值先验**（根节点，tp0.5，k=24）| 1.9375 | 0.0625 | +15.0 |
  | B **价值先验（tp0.5，k=8）** | **2.0000**（32/32）| 0 | — |
  | B **价值先验（tp1.0，k=8）** | **2.0000**（32/32）| 0 | — |
  | C 不搜索，纯 1-ply 价值 argmax（全合法格）| **1.3125** | 0.1760 | +1.78 |
  | D 同一候选集，直接按价值选（不看访问）| **1.8125** | 0.1008 | +8.06 |

  **① 那条未结的口径差结清了：`C = 1.3125`，SE 0.1760 —— 与另一个脚本
  （`_tmp_pos_greedy_vs_raw.py`）测的 1.3125 / 0.1760 逐位一致。** ⇒ **不是实现差异**：
  "不搜索、在全部合法格上做 1-ply 价值 argmax"真的只有 1.3125，而带搜索的是 1.8125。

  **② D = 1.8125，与 A 完全相同** ⇒ 在**同一个候选集**上，"按访问次数选"与"直接按价值选"
  结果一样 ⇒ **访问累积本身不是增益来源**。

  **③ 但 C（270 格菜单）1.3125 < D（24 个候选菜单）1.8125**，两者都是 1-ply 价值选择 ⇒
  **菜单变大反而更差**，与 winner's curse 的方向一致。（⚠️ C/D 除菜单大小外还有差异：
  C 是逐头顺序单操作、D 是整个 bundle，所以这条隔离不干净，见下"未清"。）

  **④ 用户的计划前提被验证**：**价值先验 + k=8 = 2.0000（32/32），而位置网先验 + k=8 = 1.6875**
  —— 用价值网当先验，k 从 24 缩到 8 不仅不掉强度、还更高（差 +0.3125，SE≈0.12，t≈2.6）。
  这正是"好先验 ⇒ 可以缩小候选集 ⇒ 搜索变便宜"。

  **⚠️ 三条必须一起说的限制**：
  1. **这个台子在 2.0 处封顶**（对手是同一 ckpt 的 raw）⇒ 1.9375 / 2.0000 / 2.0000 之间
     **分辨不出来**；"B tp0.5 k8 = 2.0 > B tp0.5 k24 = 1.9375" 不能当结论。要区分必须打 rule_v4。
  2. **价值先验目前是暴力算出来的**（根节点每个出招回合枚举钉类全部合法格、过价值网，约
     270 次前向）⇒ 它**现在并不便宜**：每次搜索 256 迭代 + ~270 次价值前向，比基线还贵。
     这次的收益是**信息**（"先验好了 k=8 就够"），**不是成本**。成本收益只有在先验由
     **一次前向**（训练出来的网络）给出时才兑现 —— 那正是计划的下一步。
  3. **未清**：C 与 D 除"菜单大小"外还差"逐头顺序单操作 vs 整个 bundle"，所以"菜单越大越差"
     这条隔离不干净。另：B 的价值先验只加在**根节点**（更深节点仍用网络先验），
     所以"价值先验的全部潜力"也未必测满。

## 2026-09-18 14:26:38 — value_prior_rule_v4

- **commit**: `e220009` (dirty: 19 files)
- **exit**: 0，用时 1512s
- **cmd**:
  ```bash
  bash _tmp_value_prior_rule_v4.sh
  ```
- **output**: `training_history/runs/20260918_142638_value_prior_rule_v4/output.log`
- **result**: 🔴 **本项目对 rule_v4 的第一次真实提升**。把"位置采样的分布"从位置网换成
  **价值网在钉类合法格上的 1-ply z-score**（只在根节点算，`eval.py --pos-prior value`；
  `BundleMCTS.pos_prior_fn` 是新增的可选回调，默认 None = 原行为）。各 64 局、同 seed=0：

  | 配置 | W/D/L | vs rule_v4 |
  |---|---|---|
  | 位置网先验，k=24，tp1.0 | 13/0/51 | 20.3%（历史基线）|
  | 位置网先验，k=8，tp1.0 | 10/0/54 | 15.6% |
  | **价值先验，k=24，tp0.5** | **25/0/39** | **39.1%** |
  | **价值先验，k=8，tp0.5** | **22/0/42** | **34.4%** |

  配对 McNemar（同 seed 逐局）：
  * 位置网 k24 vs **价值 k24**：**+18.8pp**，翻转 8:20，**p = 0.0357**
  * 位置网 k8 vs **价值 k8**：**+18.8pp**，翻转 6:18，**p = 0.0227**
  * 价值 k24 vs 价值 k8：+4.7pp，翻转 14:11，p = 0.69（**k 从 24 缩到 8 无可测代价**）
  * 位置网 k24 vs 位置网 k8：+4.7pp，翻转 11:8，p = 0.65

  ⇒ **① 价值先验稳定加约 +18.8pp**（39.1% 也超过历史最好 29.7% 约 9pp）。
  **② ⚠️ 撤回**（见 `value_prior_verify`）：本行原先写"k 从 24 缩到 8 没有可测代价"，
  那是 64 局的幸运抽样。128 局合计看 k=8 比 k=24 低 **10.2pp**，且在换 seed 那批上已与基线
  分辨不出 ⇒ **"缩 k 免费"不成立**。

  **⚠️ 三条限制（必须一起读）**：
  1. **价格**：价值先验目前是**暴力算**的（根节点每个出招回合枚举钉类全部合法格并过价值网，
     ~270 次前向）⇒ 它**现在更贵**，不是更便宜。收益是**信息**，成本收益要等先验由
     **一次前向**（训练出来的网络）给出才兑现 —— 那正是计划的下一步。
  2. **一个混杂**：该回调枚举的格子额外过了一遍 `can_apply_operation` 过滤（保证提出的
     动作可执行），而位置网采样不保证这点 ⇒ 部分增益来自"少浪费回合"而不是"位置更好"。
     对照臂 `--pos-prior uniform` 实测（seed0-63）= 26.6%，即 **+18.8pp 里约 +6.3pp 是这个**
     （见 `value_prior_verify`）。
  3. 单批 64 局、单个 seed 集（0..63）；因这数字远超历史最好，已另跑**换 seed（64..127）**
     的独立复核（`value_prior_verify`），结论以那一轮为准。

## 2026-09-18 14:53:47 — value_prior_verify

- **commit**: `e220009` (dirty: 20 files)
- **exit**: 0，用时 1939s
- **cmd**:
  ```bash
  bash _tmp_value_prior_verify.sh
  ```
- **output**: `training_history/runs/20260918_145347_value_prior_verify/output.log`
- **result**: 🔴 **独立复核 + 混杂对照。结论：主效应成立，但"缩 k 免费"被推翻。**

  (A) **混杂对照 `--pos-prior uniform`（seed 0..63）**：走完全一样的枚举 + `can_apply_operation`
  过滤，但权重全相同 ⇒ `17W/0D/47L = 26.6%`（同批位置网先验 = 20.3%）。⇒ 那 +18.8pp 里
  **约 +6.3pp 来自"只提出可执行动作 / 丢掉那个坏先验"**（不是先验质量的功劳），
  剩下约 +12.5pp 才是价值信息本身。（此臂只有 64 局，未换 seed 复核。）

  (B) **换 seed（64..127）**：

  | 配置 | W/D/L | 胜率 |
  |---|---|---|
  | 位置网先验 k24（新 seed 基线）| 14/0/50 | 21.9% |
  | 价值先验 k24 | 23/0/41 | 35.9% |
  | **价值先验 k8** | **13/0/51** | **20.3%** |

  **把两批 seed 合起来（共 128 局）**：

  | 配置 | 胜率 | 95% CI |
  |---|---|---|
  | 位置网先验 k24（基线）| 21.1%（27W/101L）| 14.9–29.0 |
  | **价值先验 k24** | **37.5%（48W/80L）** | 29.6–46.1 |
  | 价值先验 k8 | 27.3%（35W/93L）| 20.4–35.6 |

  * **价值 k24 vs 基线：+16.4pp，z = 2.93**（配对：seed0-63 p=0.036；seed64-127 p=0.108，
    两批**同向同量级**）⇒ **稳**。
  * 价值 k8 vs 基线：+6.2pp，z = 1.17 ⇒ **不显著**。
  * 价值 k24 vs k8：+10.2pp，z = 1.75；两批 seed 的配对**互相矛盾**
    （seed0-63 p=0.69"没代价"，seed64-127 p=0.087"明显更差"）。

  ⇒ **① 主结论修正为："把位置采样分布换成价值网的偏好，在 k=24 上可复现地 +16.4pp
  （128 局，z=2.93）"。** 这是本项目对 rule_v4 的第一次可复现提升。
  **② 撤回 `value_prior_rule_v4` 里"k 从 24 缩到 8 无可测代价"那句** —— 那是 64 局的幸运抽样；
  128 局看 k=8 比 k=24 低 10.2pp，且在换 seed 那批上已与基线分辨不出。
  **③ 对计划（docs/az_value_prior_plan.md）的意义**：真正"免费"的形态不是"缩 k"，而是
  **"把 270 次暴力前向换成 1 次网络前向，k 保持 24"** ⇒ 成本回到基线水平、期望 +16.4pp
  （这是上限；真实值取决于网络逼近 oracle 的程度）。"缩 k"是**另一个**问题，而且现有证据说明
  8 个候选不够用（很可能又是"候选挤在同一片区域"那个多样性塌缩，只是这次区域由价值挑）。

## 2026-09-18 16:37:45 — valprior_similarity

- **commit**: `0d398d6` (dirty: 18 files)
- **exit**: 0，用时 669s
- **cmd**:
  ```bash
  bash _tmp_valprior_similarity.sh
  ```
- **output**: `training_history/runs/20260918_163745_valprior_similarity/output.log`
- **result**: _待填_

## 2026-09-18 17:35:00 — vprior_collect400

- **commit**: `0d398d6` (dirty: 20 files)
- **exit**: 0，用时 3540s
- **cmd**:
  ```bash
  bash -c export PYTHONIOENCODING=utf-8; D:/anaconda3/envs/pytorch-gpu/python.exe -u code/my_ai/az_intent/collect_value_prior.py --checkpoint training_history/az_fixed/three_mix_r10p_vw_pol_frozen.pt --games 400 --workers 16 --seed 910301 --out training_history/vprior/vp_400.npz
  ```
- **output**: `training_history/runs/20260918_173500_vprior_collect400/output.log`
- **result**: 400 局 → **24972 个决策点**（62.4/局）、300934 回合，用时 3540s（比估的 35 分钟久），
  产物 `training_history/vprior/vp_400.npz` **19.3 MB**（压缩后约 0.8 KB/决策点）。
  **目标内容**：类分布 `[(17, 24972)]`（**全部是类 17 闪电**）、三头各 1/3、每行恒 271 格、
  目标内 adv 的 (极差/std) = 4.66 ⇒ 目标本身有信息。
  ⚠️ **通道覆盖仍只有类 17** —— 因为类头塌缩成闪电，三个头的 argmax 都是 17；
  这与"随机钉类"那个方案想解决的问题是同一个，但本方案**没有**触碰类轴。

## 2026-09-18 18:34:49 — vprior_train_r1

- **commit**: `cd50a65` (dirty: 19 files)
- **exit**: 0，用时 325s
- **cmd**:
  ```bash
  bash -c export PYTHONIOENCODING=utf-8; D:/anaconda3/envs/pytorch-gpu/python.exe -u code/my_ai/az_intent/train_value_prior.py --ckpt training_history/az_fixed/three_mix_r10p_vw_pol_frozen.pt --data training_history/vprior/vp_400.npz --epochs 60 --batch-size 256 --lr 1e-3 --tau 1.0 --t-pos 1.0 --out training_history/vprior/posnet_r1.pt
  ```
- **output**: `training_history/runs/20260918_183449_vprior_train_r1/output.log`
- **result**: 只训位置网（类网/价值网冻结），目标 `p=softmax(z(adv)/tau=1.0)`、
  预测 `q=softmax(z(action_map[c])/t_pos=1.0)`，两侧同一套合法格 z-score。60 epoch × 5s。

  | | val CE | val top1 命中 |
  |---|---|---|
  | 训练前 | **6.1264** | 0.72% |
  | 训练后（epoch 59，最佳）| **5.1983（−15.1%）** | **32.96%** |

  判据1 通过：CE 降到 5.1983，**已低于均匀基线 ln(271) = 5.60**（训练前 6.13 **高于**它——
  说明旧位置网又尖又偏、比均匀还差）。判据2 通过：top1 0.72% → 32.96%（随机的 89 倍）。
  ⇒ `training_history/vprior/posnet_r1.pt`。

## 2026-09-18 18:41:49 — vprior_r1_eval

- **commit**: `cd50a65` (dirty: 19 files)
- **exit**: 0，用时 885s
- **cmd**:
  ```bash
  bash _tmp_vprior_r1_eval.sh
  ```
- **output**: `training_history/runs/20260918_184149_vprior_r1_eval/output.log`
- **result**: 🔴 **本项目对 rule_v4 的第一次可复现绝对提升，而且成本与基线相同（一次前向）。**

  把 `posnet_r1.pt`（位置网已训成"价值偏好"）当先验，k=24 打 rule_v4，128 局（两批 seed）：

  | 位置先验 | 128 局胜率 | 95% CI | W/D/L |
  |---|---|---|---|
  | 未训练（基线）| **21.1%** | 14.9–29.0 | 27/0/101 |
  | **posnet_r1（训练后）** | **35.2%** | 27.4–43.8 | 45/0/83 |
  | 价值 oracle（暴力算）| 37.5% | 29.6–46.1 | 48/0/80 |

  * **训练后 vs 基线：+14.1pp，z = 2.53**（配对：seed0-63 翻转 10:18 p=0.185；
    seed64-127 翻转 10:20 p=0.099，两批**同向同量级**）。
  * **训练后 vs oracle：−2.3pp，z = −0.39 ⇒ 分不出来** ⇒ 网络把 oracle 的收益
    **复现了约 86%（14.1/16.4pp）**。
  * 呼应先验质量的软指标（`_tmp_prior_quality.py`，n=24）：
    **峰的价值分位 30.3% → 8.8%**（随机 50）、top5 重叠 5% → **23%**、
    质量落在价值 top10(tp0.3) 13% → 28%、有效候选 10.0 → 36.4（图更平了）。

  **关键点：oracle 需要每个出招回合暴力跑 ~270 次价值前向；训练后的位置网只要 1 次
  （与基线相同）⇒ 这 +14.1pp 是"免费"的。** 这正是计划的形态（§0 目标：同成本 +16.4pp）。
  ⚠️ 待独立复核（`vprior_verify`）：raw 镜像、复训（--seed 1）、换 128 个没见过的 seed。
  ⚠️ 交代清楚的口径：本评测 t_pos=1.0、k=24、pos-only + skip-single，与基线的那些臂逐字相同。

## 2026-09-18 18:57:59 — vprior_verify

- **commit**: `cd50a65` (dirty: 19 files)
- **exit**: 0，用时 1918s
- **cmd**:
  ```bash
  bash _tmp_vprior_verify.sh
  ```
- **output**: `training_history/runs/20260918_185759_vprior_verify/output.log`
- **result**: 🔴 **三件独立复核全部通过。**

  **A. raw 镜像（训后裸策略 vs 训前裸策略，完全不搜索）**：配对得分 **1.4688**
  （SE 0.1098，t = **+4.27**）。基线那一臂按构造是 1.0000 ⇒ **裸策略本身也变强了**，
  即增益不只是"先验帮搜索更好地探索"，位置图真的学到了东西。

  **B. 复训 `--seed 1`（不同 train/val 划分与训练顺序）**：最佳 val CE = 5.2008
  （第一轮 5.1983，几乎一样 ⇒ 训练稳）。评测：seed 0-63 **42.2%**（27W/37L）
  + seed 64-127 **39.1%**（25W/39L）= **128 局 40.6%**。

  **C. 换没见过的 game seed 128..255**（同一 ckpt `posnet_r1.pt`）：
  seed 128-191 **43.8%**（28W/36L）、seed 192-255 **54.7%**（35W/39L）= **128 局 49.2%**。

  **四批 seed 集汇总（各 64 局，同 seed 配对）**：

  | seed 集 | 未训练基线 | posnet_r1 | 差 | 配对 McNemar |
  |---|---|---|---|---|
  | 0-63 | 20.3% | 32.8% | +12.5pp | 10:18, p=0.185 |
  | 64-127 | 21.9% | 37.5% | +15.6pp | 10:20, p=0.099 |
  | 128-191 | 17.2% | **43.8%** | +26.6pp | 6:23, **p=0.0023** |
  | 192-255 | 14.1% | **54.7%** | +40.6pp | 4:30, **p<0.0001** |
  | **合计 256 局** | **18.4%**（47W）| **42.2%**（108W）| **+23.8pp** | z = **6.08** |

  ⚠️ **两条要如实标注**：
  1. **差距的大小随 seed 集单调上升（+12.5 → +40.6pp）**，而基线本身反而在下降
     （20.3% → 14.1%）。所以"提升幅度"**依赖地图/seed 集**、不是常数；但每一批都同向，
     后两批高度显著。
  2. 两轮独立训练（r1 35.2%、r1s1 40.6%，均在 seed 0-127）都**远高于基线 21.1%**，
     且都**不低于 oracle 的 37.5%**。一个可能机制：学出来的图**比 oracle 更平**
     （有效候选 36.4 vs 10.0），而"平"正是已知对搜索更友善的方向（尖会多样性塌缩）。

## 2026-09-18 19:32:29 — vprior_baseline_fresh

- **commit**: `38bc96d` (dirty: 18 files)
- **exit**: 0，用时 684s
- **cmd**:
  ```bash
  bash _tmp_vprior_baseline_fresh.sh
  ```
- **output**: `training_history/runs/20260918_193229_vprior_baseline_fresh/output.log`
- **result**: 补"未训练基线"在 seed 128..255 上的成绩（**必须先补这个才能解读训后的 49.2%**，
  因为 seed 集之间的波动本身就有十几 pp —— 这是本会话第四次遇到"单 seed 集不可比"）：
  seed 128-191 = **11W/53L = 17.2%**、seed 192-255 = **9W/55L = 14.1%**，
  合计 **20W/108L = 15.6%**（对比训后 posnet_r1 的 63W/65L = **49.2%**）。

## 2026-09-18 19:51:04 — tpos_curve_priors

- **commit**: `a726a45` (dirty: 17 files)
- **exit**: 0，用时 2749s
- **cmd**:
  ```bash
  bash _tmp_tpos_curve_priors.sh
  ```
- **output**: `training_history/runs/20260918_195104_tpos_curve_priors/output.log`
- **result**: 把"尖度"与"内容质量"分开：各自固定内容、只扫 t_pos（seeds 0-63、各 64 局、k=24）。

  | t_pos | 0.3 | 0.5 | 0.7 | 1.0 | 2.0 |
  |---|---|---|---|---|---|
  | 位置网先验（**内容差**，有效候选 10）| **1.6%** | 7.8% | 18.8% | 20.3% | 20.3% |
  | oracle（**内容对**）| 28.1% | 39.1% | — | 28.1% | **40.6%** |
  | learned（**内容对**，有效候选 36）| 17.2% | **42.2%** | — | 32.8% | 31.2% |

  与各自曲线内最高点比：oracle 的 tp0.3/tp1.0 是 −12.5pp（z=−1.50，不显著）；
  learned 的 tp0.3 是 **−25.0pp（z=−3.22，显著）**，tp1.0/tp2.0 是 −9~−11pp（不显著）。

  🔴 **结论：撤回我"图更平所以更好"的假说。** 用户的反例（"若平更好则完全均匀最好"）是对的。
  正确描述：**存在一条"尖度下限"（约 t_pos 0.5）**；低于它尖化有害，而**害处的大小取决于
  内容质量** —— 内容差时是灾难（1.6%），内容对时只有 10~15pp，且 tp≤0.3 之外测不到系统趋势
  （oracle 那条在 28~41% 之间非单调波动，全在 ~1.5σ 内）。
  ⇒ **"学出来的 ≈ oracle"的原因是内容对齐（峰的价值分位 8.8% vs 30.3%），不是"更平"**；
  平顶区间内的差异无关。
  **实用结论**：learned 先验在 t_pos ∈ {0.5, 1.0, 2.0} 上都可用（31~42%，彼此不可分），
  **t_pos ≤ 0.3 应当避免**；默认 1.0 不必改（tp0.5 的 42.2% vs tp1.0 的 32.8% 差 9.4pp、
  z=1.1，不足以据此改默认）。

## 2026-09-18 23:05:44 — vprior_tau_probe

- **commit**: `635d275` (dirty: 19 files)
- **exit**: 0，用时 2302s
- **cmd**:
  ```bash
  bash _tmp_vprior_tau_probe.sh
  ```
- **output**: `training_history/runs/20260918_230544_vprior_tau_probe/output.log`
- **result**: 🔴 **一次被推翻的预测。** 原假设（见 `train_value_prior.py` 文件头）：τ 近似重参数化
  —— 因为 `dL/dm=0 => m 正比于 z(adv)`（与 τ 无关），而采样侧解码器会重新 z-score、把尺度约掉，
  所以 τ 应当是空操作。用同一份数据、**同一个 `--seed 0`**（与 `posnet_r1` 逐字相同），
  唯一变量 = τ：

  | τ | 128 局打 rule_v4 |
  |---|---|
  | 0.25 | **30.5%**（29.7% + 31.2%）|
  | 0.5 | **36.7%**（31.2% + 42.2%）|
  | 1.0（`posnet_r1`）| **35.2%** |

  ⇒ **τ ∈ {0.5, 1.0} 不可分，τ=0.25 略差** ⇒ 实用上"别单独调 τ"这个结论成立。

  **但机制上我的预测错了**（无噪声检验，`_tmp_vprior_map_compare.py`，200 个决策点）：

  | | 原始地图 std | t_pos=1.0 下有效候选 |
  |---|---|---|
  | τ=1.0 | 1.454 | **176.3** |
  | τ=0.25 | 0.532 | **44.3** |

  归一化地图余弦 **0.698**、分布余弦 0.574、top1 同格 **34%** ⇒ **地图不是"只差一个尺度"**，
  τ=0.25 训出来的图**本征上就尖得多**。⇒ **τ 与 t_pos 耦合在同一根轴上**（都决定采样时的有效
  尖锐度），不是空操作。**错因**：那段代数假设网络能精确实现 `m 正比于 z(adv)`；实际卷积网络学的
  是受损失加权的近似、权重随 τ 变 ⇒ 实现的形状也随 τ 变。
  **顺带**：top1 命中随 τ 变小而升高（τ=1.0 **32.96%** → τ=0.5 **57.59%**），
  与"目标越尖、损失越集中在最好的那格"一致。
  ⚠️ **注意 CE 不能跨 τ 比**：目标的分布不同（CE = H(p) + KL）。换算后两个 τ 都拟合得很好
  （τ=0.25：CE 3.186 vs H(p)=ln23.5=3.157 ⇒ KL≈0.03；τ=1.0：5.198 vs ln170.5=5.139 ⇒ KL≈0.06）。

## 2026-09-18 23:46:42 — vprior_tau025_tp2

- **commit**: `cd09590` (dirty: 18 files)
- **exit**: 0，用时 402s
- **cmd**:
  ```bash
  bash -c export PYTHONIOENCODING=utf-8; D:/anaconda3/envs/pytorch-gpu/python.exe -u code/my_ai/az_intent/eval.py --checkpoint training_history/vprior/posnet_tau0.25.pt --opponent rule_v4 --games 64 --workers 16 --seed 0 --bundle-mcts --native-engine --iterations 256 --max-depth-rounds 4 --k 24 --t-class 0.5 --t-pos 2.0 --search-mode pos-only --skip-single-candidate
  ```
- **output**: `training_history/runs/20260918_234642_vprior_tau025_tp2/output.log`
- **result**: "τ 与 t_pos 在同一根轴上"的检验：把 **τ=0.25 训出来的网**（本征更尖、
  t_pos=1.0 下有效候选 44）放到 **t_pos=2.0** 上 ⇒ **26W/0D/38L = 40.6%**
  （同 seed 下它在 t_pos=1.0 是 **29.7%**）。**抬高 t_pos 补回 +10.9pp** ⇒ 同轴说法成立。

  **汇总成一个两因子模型**（横轴 = 有效候选数，各 64 局、seeds 0-63）：

  | 内容质量 | 有效候选 → vs rule_v4 |
  |---|---|
  | **差**（位置网先验）| 10 → **1.6%**、48 → **7.8%**、107 → 18.8%、173 → **20.3%**、242 → 20.3% |
  | **好**（learned `posnet_r1`）| 36 → 17.2%、85 → **42.2%**、176 → 32.8%、~240 → 31.2% |
  | **好**（τ=0.25 的网）| 44 → **29.7%**、~90@tp2.0 → **40.6%** |

  ⇒ ①**内容质量**决定天花板（差 ≈20%、好 ≈35-40%，两批 seed 共 256 局）；
  ②**有效尖锐度必须 ≳70-85**，否则掉到天花板之下，**掉多少取决于内容多差**
  （内容差 → 1.6%，内容好 → 17~30%）。
  ⇒ 这**救回了"尖锐度轴"直觉的一半**：轴确实存在、下限确实存在，但"越平越好"是错的
  （过下限之后由内容决定）。
  ⚠️ 单点都是 64 局（SE≈5.5pp），上面这些格子之间的差多数不显著；模型是**拟合**出来的，
  不是逐格验过的。

## 2026-09-19 10:31:05 — vprior_round2

- **commit**: `37f5570` (dirty: 17 files)
- **exit**: 0，用时 5365s
- **cmd**:
  ```bash
  bash _tmp_vprior_round2.sh
  ```
- **output**: `training_history/runs/20260919_103105_vprior_round2/output.log`
- **result**: 🔴 **没有复利 —— 两个读数都完全没动。**

  做法：用第一轮的 `posnet_r1` 自己采 400 局（**只换"谁在采"这一个变量**，数据量、配方逐字不变）
  → warm start 训 `posnet_r2` → 4 批 seed 共 256 局打 rule_v4 → raw 镜像。

  * **采集：27813 个决策点（69.5/局）**，比第一轮的 62.4/局多 ⇒ r1 打的局面里出招回合更多
    （学会了往哪儿放闪电、用得更频繁）⇒ **局面分布确实变了**，不是"什么都没变"。
  * 训练：`posnet_r1` 对着**第二轮目标**的"训练前"是 CE 5.3316 / top1 **9.71%**
    （第一轮时原网络对着第一轮目标是 6.1264 / 0.72%）⇒ **r1 已经和第二轮目标有一点点对齐**。
    训完 CE 5.1897 / top1 38.55%（第一轮 5.1983 / 32.96%），提升很小。

  | | 256 局 vs rule_v4 |
  |---|---|
  | 旧基线（未训练位置网）| **18.4%**（47W）|
  | 第一轮 `posnet_r1` | **42.2%**（108W）|
  | **第二轮 `posnet_r2`** | **41.0%**（105W）|

  * r2 vs r1：**−1.2pp，z = −0.27 ⇒ 不可分**（各批 seed：35.9 / 48.4 / 34.4 / 45.3，
    而 r1 是 32.8 / 37.5 / 43.8 / 54.7 —— 又是 seed 集异质性，合起来正好抵消）。
  * r2 vs 旧基线：**+22.6pp，z ≈ 5.8** ⇒ 仍然远高于基线，**第一轮那个大台阶是真的**。
  * **raw 镜像：r2 裸策略 vs r1 裸策略 = 1.0000（SE 0.1188，t = 0.00）⇒ 逐点相同。**

  **⇒ 结论：这套配方一次就把增益吃完了，再迭代不涨。** 最可能的解释（未单独验证）：
  目标 = **价值网的偏好**，而价值网全程冻结不变 ⇒ 先验只能逼近它，逼近之后循环没有新的
  信息源。⇒ **下一个天花板是价值网本身**（或"让目标比 1-ply 更深"）。
  这也与项目历史一致（之前 5 轮位置网循环都没复利），但这次**排除了我自己的两个设计错误**
  （跨轮累积陈旧目标、盘子太小）⇒ "不复利"这次是配方本身的性质，不是实现 bug。
  ⚠️ 本轮采集脚本还没有进度文件（`fec2fc2` 之后才加），所以中途看不见进度。

## 2026-09-19 12:15:59 — rv4_lightning

- **commit**: `22bdaa5` (dirty: 17 files)
- **exit**: 0，用时 1093s
- **cmd**:
  ```bash
  bash _tmp_rv4_lightning.sh
  ```
- **output**: `training_history/runs/20260919_121559_rv4_lightning/output.log`
- **result**: **rule_v4 闪电位置的迁移性检验**（用户提议）。两侧都是 rule_v4 血统、镜像配对、
  确定性 ⇒ NULL 臂应给出恰好 0.5000/SE=0。各 64 对（128 局）：

  | 臂 | 配对胜率 | t | 净胜局 | 位置被改 |
  |---|---|---|---|---|
  | NULL（我方也用原版）| **0.5000（SE=0）** | — | 0 | 0% |
  | `net`（我们训好的位置网 argmax，1 次前向）| **0.4922** | −0.17 | **−1** | 98.0% |
  | `value`（价值网 1-ply argmax，~270 次前向）| **0.3281** | **−4.09** | **−22** | 98.7% |

  **① NULL 在 128 局上完美抵消（0.5000 / SE=0）** ⇒ 台子与"镜像严格对称"成立。
  **② 我们改了 98% 的闪电落点，结果几乎一字不变（`net` 净 −1 局，t=−0.17）**
  ⇒ **我们的位置网与 rule_v4 的手写选点规则质量相当，但超不过它。**
  **③ 换成"价值网 1-ply 的精确最优"反而显著更差（−22 局，t=−4.09）**
  ⇒ **rule_v4 手写规则的落点明显优于 1-ply 价值 argmax。**
  这也是"取 max 会被 winner's curse 带偏"的**分布外**旁证（对照 `value_prior_ab` 的
  arm C 1.3125 vs 搜索 1.8125 —— 同一机制，另一套局面）。

  **⚠️ 混杂（必须一起读）**：我们的网络是在**我们自己智能体的局面分布**上训的，rule_v4 的局面
  不同 ⇒ 上面既是"迁移性"检验，也混着"分布外"这件事。
  **对最初问题的回答**：作为"给 rule_v4 提强度"的手段，**没有可利用的增益**；
  但作为对自己系统的旁证，它反而相当干净（见下条 `mcts` 臂）。

## 2026-09-19 12:38:09 — rv4_lightning_mcts

- **commit**: `22bdaa5` (dirty: 18 files)
- **exit**: 0，用时 592s
- **cmd**:
  ```bash
  bash -c export PYTHONIOENCODING=utf-8; D:/anaconda3/envs/pytorch-gpu/python.exe -u code/test_match/rule_v4_lightning_match.py --pos-source mcts --pairs 64 --workers 16 --seed 0
  ```
- **output**: `training_history/runs/20260919_123809_rv4_lightning_mcts/output.log`
- **result**: 补上"真正的搜索"那一臂（用户原话是"用**搜索**决定位置"；前两条是更便宜的近似）：
  用 `BundleMCTS`（k=24 / 迭代 256 / pos-only），做法是在 net_fn 外面套一层把 head 0 的
  argmax 强制成闪电（这样 `pos_pin="argmax"` 就把类钉在 17）、**不改搜索代码**。

  | arm（各 64 对 = 128 局）| 配对胜率 | t | 净胜局 |
  |---|---|---|---|
  | NULL | 0.5000（SE=0）| — | 0 |
  | `net`（位置网 argmax，1 次前向）| 0.4922 | −0.17 | −1 |
  | `value`（价值网 1-ply argmax）| **0.3281** | **−4.09** | **−22** |
  | **`mcts`（256 迭代 + 访问累积）** | **0.5391** | +1.04 | **+5** |

  ⇒ 顺序 `value(−22) ≪ net(−1) ≈ rule_v4 < mcts(+5)`。
  **关键读法：同一批候选、同一套局面，只把"取 max"换成"访问累积"，就从 −22 局变成 +5 局**
  ⇒ 这是"**visit-argmax 比 1-ply 硬 argmax 更稳健（winner's curse）**"的**分布外**旁证，
  与 `value_prior_ab` 的 arm C(1.3125) vs arm A(1.8125) 是同一机制、另一套局面。
  ⚠️ `mcts` 的 +5 局/128 局 **不显著**（t=1.04），只能说"方向为正、不能排除为零"。
  ⇒ **对"给 rule_v4 提强度"这个问题：没有可利用的增益**（最好也只是不显著的正向）。
  ⇒ **对自己的系统：这是一条干净的旁证 —— 搜索的增益载体是"访问平均"，不是"价值 argmax"。**

## 2026-09-19 13:01:52 — rv4_lightning_only

- **commit**: `650b48a` (dirty: 19 files)
- **exit**: 0，用时 3594s
- **cmd**:
  ```bash
  bash _tmp_rv4_lightning_only.sh
  ```
- **output**: `training_history/runs/20260919_130152_rv4_lightning_only/output.log`
- **result**: 🔴 **"只放闪电"的隔离检验 —— 把前面的结论翻转了。**

  用户提议：一方**只放闪电**（冷却 0 且金币够就放、否则空过；不建塔/升级/拆塔），
  位置用 rule_v4 的启发式规则，去打完整 rule_v4。**这样落点是唯一的决策 ⇒ 位置轴的干净隔离。**
  金币有被动收入（起始 50、+1.5/轮、闪电 90），所以不放塔也能攒到。
  顺带实测确认"能放就放"：400 轮一局里"冷却 0 且金币 ≥90"共 **11 次、11 次全放**，
  动作目录一次都没挡住（所以这个语义是准确的）。

  | 只放闪电，落点来自 | 配对胜率 | t（vs 0.5）| 赢/128 |
  |---|---|---|---|
  | A rule_v4 的规则 | 0.1719 | −10.97 | 22 |
  | B 我们的位置网（argmax）| 0.1953 | −8.82 | 25 |
  | C 价值网 1-ply argmax | 0.1797 | −8.93 | 23 |
  | **D 256 迭代搜索（访问累积）** | **0.3359** | **−3.93** | **43** |

  * **只放闪电打不过完整 rule_v4**（A 臂 17.2%，t=−10.97）—— 不放塔就守不住。
  * **但落点之间差别巨大：D 几乎把胜局翻倍（22 → 43）**，未配对 z = 3.20（p≈0.001）。
  * **B（我们的位置网）≈ C（1-ply 价值 argmax）≈ A（rule_v4 手写规则）**（22/23/25，全在噪声内）。

  ⇒ **同一个排序第三次出现**（前两次：镜像台 `value argmax 1.3125` vs 搜索 `1.8125`；
  完整规则下 `value −22 局` vs `mcts +5 局`）：
  **搜索（访问平均）的选点远好于任何"一次性"选点 —— 包括 rule_v4 的手写规则、我们的位置网、
  和 1-ply 价值 argmax。**
  ⇒ **这推翻了 `az_value_prior_plan.md` §0.3 的一半结论**：位置网确实把 **1-ply 价值**这个目标
  榨干了，但**位置这条线远没到顶** —— 搜索手里的位置知识比 1-ply 价值丰富得多，只是我们
  还没把它当目标。**"让目标更深（用搜索的根节点子值当目标）"从"值得一试"升级为"证据最强的一步"。**
  ⚠️ 局限：本脚本当时**没有逐局日志**，所以只能做未配对 z 检验（配对会更灵敏）；已补上。
  另：我们的网是在**我们自己局面分布**上训的，rule_v4 的局面不同。

## 2026-09-19 14:47:34 — cand_pick

- **commit**: `04e855a` (dirty: 17 files)
- **exit**: 0，用时 301s
- **cmd**:
  ```bash
  bash _tmp_cand_pick.sh
  ```
- **output**: `training_history/runs/20260919_144734_cand_pick/output.log`
- **result**: **官方启发式候选集 + 谁来选**（用户提出的设计）。候选集 = `ActionCatalog.build()`
  按 score 降序的前 K 个（**就是 ExampleAI 的脑子**：6 类生成器 → 去重 → 按分排序 →
  **一步 rollout 重排**），两侧拿**同一候选集**，一个随机选、一个让价值网 1-ply 选。
  ⇒ 这是"同一堆选项、谁排得更好"的受控 A/B（比"和 rule_v4 比一致率"干净）。

  | 臂 | 配对胜率 | t | 净胜局 |
  |---|---|---|---|
  | control 随机 vs 随机 | **0.5000（SE=0）** | — | 0 |
  | **主判据 价值网 vs 随机（k=8）** | **0.5469** | +0.83 | **+3** |
  | 价值网 vs 随机（k=24） | 0.5312 | +0.53 | +2 |
  | 参照 纯启发式（first）vs rule_v4 | **0.0000** | — | **−16** |
  | 绝对强度 价值网 vs rule_v4 | **0.0000** | — | **−32** |

  **① 主判据"读不出结论"**：value vs random 只 +3 局（t=0.83）。但**不能**据此说"价值网排不了"——
  另一种解释是"**启发式 top-8 本来就近乎等价**"，那样子谁重排都测不出差别。
  ⇒ 已补跑 **`first`（启发式第一名）vs `random`** 作天花板对照，**结论以那条为准**。

  **② 更扎眼的发现：只按启发式选的一方全败给 rule_v4（0.0000，−32 局）。**
  它的动作构成是 `BUILD_TOWER 93% / UPGRADE 5% / DOWNGRADE 1%`，**闪电 0%**；而 rule_v4 是
  `建塔 33% / 降级 25% / 升级 25% / 闪电 18%`。
  ⇒ **官方启发式压根不选闪电，而闪电是决定性的武器**（与"gen_0030 闪电流打 ExampleAI 90.6%"
  是同一件事；也印证用户说的"ExampleAI 不太爱放闪电"）。
  ⇒ 与 `rv4_lightning_only` 并排：**只建塔 = 0%、只闪电 = 17.2%、两者都做 = 最强**。
  ⇒ **类轴要学的是"闪电之外还要做经济/防御"的混合策略，不是"用别的动作替掉闪电"。**

  ⚠️ 设计细节：候选数在**空过回合恒为 1**（金币不够只能 hold），只有能出招的回合（约 30%）
  才有 ≥2 个 ⇒ 差异只出现在那些回合。k=8 时**闪电从不入候选**，所以这个实验天然在测非闪电
  动作。对局仍决定性（基地能被打到 0），所以"不放闪电就打不死人"这个担心不成立。

## 2026-09-19 14:53:16 — cand_pick_first

- **commit**: `04e855a` (dirty: 18 files)
- **exit**: 0，用时 53s
- **cmd**:
  ```bash
  bash -c export PYTHONIOENCODING=utf-8; D:/anaconda3/envs/pytorch-gpu/python.exe -u code/test_match/heuristic_candidates_match.py --our first --opp random --k 8 --pairs 32 --workers 16 --seed 0
  ```
- **output**: `training_history/runs/20260919_145316_cand_pick_first/output.log`
- **result**: 🔴 **把上一条的主判据判死了。** `first`（启发式**自己的第一名**）vs `random`
  （top-8 里随便挑）**= 0.4531（SE 0.0520，t = −0.90，净 −3 局）**。
  ⇒ **启发式 top-8 内部近乎等价** ⇒ 上一条"价值网在 top-8 里 +3 局（不显著）"**不能**读成
  "价值网排不了"——**那堆选项里本来就没有可排的东西**。

  **顺带的独立发现：官方启发式的 `score` 在 top-8 内部的排序基本是噪声**
  （它有用的是"筛出这 8 个"，不是"排出先后"）。

  ⇒ 要测"谁能排得更好"，候选池必须**大到质量上真有差距** ⇒ 已改用 **K=96（≈全部候选）**
  重跑三条（`cand_pick_k96`）：`first vs random` 作"池子里有没有质量差"的对照、
  `value vs random` 主判据、`value vs first` 头对头。

## 2026-09-19 14:59:28 — cand_pick_k96

- **commit**: `ad40cb7` (dirty: 18 files)
- **exit**: 0，用时 454s
- **cmd**:
  ```bash
  bash -c 
export PYTHONIOENCODING=utf-8; PY=D:/anaconda3/envs/pytorch-gpu/python.exe; S=code/test_match/heuristic_candidates_match.py
echo '########## first vs random (k=96=全部) —— 池子里到底有没有质量差 ##########'
$PY -u $S --our first --opp random --k 96 --pairs 32 --workers 16 --seed 0
echo '########## value vs random (k=96) —— 价值网能不能在大池子里挑 ##########'
$PY -u $S --our value --opp random --k 96 --pairs 32 --workers 16 --seed 0
echo '########## value vs first (k=96) —— 直接打启发式的选法 ##########'
$PY -u $S --our value --opp first --k 96 --pairs 32 --workers 16 --seed 0

  ```
- **output**: `training_history/runs/20260919_145928_cand_pick_k96/output.log`
- **result**: 🔴 **把候选池放大到 K=96（≈全部），并发现"对局级比较在这个游戏里不可传递"。**

  | 臂（k=96，各 32 对）| 配对胜率 | t | 净胜局 |
  |---|---|---|---|
  | `first`（启发式第一名）vs `random` | **0.6406** | **+2.06** | **+9** |
  | `value`（价值网）vs `random` | **0.5156** | +0.25 | **+1** |
  | `value` vs `first` | **0.6094** | **+2.03** | **+7** |

  * **池子里质量差是真的**（k=8 时没有、k=96 时 `first` 明显赢随机）⇒ "top-8 等价"只成立于小池子。
  * **但三条互相矛盾**：前两条给 `value < first`、第三条给 `value > first` ⇒ **不可传递**
    （这游戏有建塔/降级/升级/闪电的相互克制，可能是真的非传递性；也可能是别的混杂，未解释）。
  * ⇒ **对局不能当"谁排得更好"的判据**，改用局面级无噪声判据（见下条 `rank_quality`）。

  **顺带（独立发现）**：只按启发式选的一方**全败给 rule_v4（0.0000，−32 局）**，
  动作构成 `建塔 93%/升级 5%/降级 1%`、**闪电 0%**；rule_v4 是 `建塔33%/降级25%/升级25%/闪电18%`。
  ⇒ **官方启发式压根不选闪电**（与"gen_0030 闪电流打 ExampleAI 90.6%"、用户说的
  "ExampleAI 不太爱放闪电"一致）。与 `rv4_lightning_only` 并排：
  **只建塔 = 0% / 只闪电 = 17.2% / 两者都做 = 最强** ⇒ 类轴要学的是**混合**策略。

## 2026-09-19 15:22:46 — cand_pick_hl_save

- **commit**: `87c3dfb` (dirty: 18 files)
- **exit**: 0，用时 165s
- **cmd**:
  ```bash
  bash -c 
export PYTHONIOENCODING=utf-8; PY=D:/anaconda3/envs/pytorch-gpu/python.exe; S=code/test_match/heuristic_candidates_match.py
echo '########## hl_save vs rule_v4 (k=8) ##########'
$PY -u $S --our hl_save --opp rule_v4 --k 8 --pairs 32 --workers 16 --seed 0

  ```
- **output**: `training_history/runs/20260919_152246_cand_pick_hl_save/output.log`
- **result**: 🔴 **"rule_v4 的优势 = 官方启发式 + 闪电"这个假说被否掉；并定位出强度来源。**

  两代分解臂（都是"启发式做非闪电决策"，只加不同的闪电规则）：

  | 臂 | vs rule_v4 | 我方动作构成（建塔/降级/升级/闪电）|
  |---|---|---|
  | `hl`（能放就放，不攒钱）| （冒烟 2 对 −2）| **94% / 0% / 6% / 0%** ← 一次闪电都没放 |
  | `hl_save`（+ rule_v4 的"冷却≤2 就攒钱"）| **0.0000（−32 局）** | **85% / 2% / 2% / 11%** |
  | rule_v4 | — | **33% / 28% / 22% / 18%** |

  **① 冒烟就暴露了经济学约束**：`hl` 一直建塔（花钱）⇒ **金币永远攒不到 90 ⇒ 一次闪电都放不出来**。
  所以"启发式不选闪电"背后是"**一直建塔就买不起闪电**"。
  **② 但补上"攒钱"规则也全败（0.0000，−32 局）**，尽管它确实放出了闪电（11%）。
  **③ 关键在动作构成的不对称**：rule_v4 **降级 28% + 升级 22%**（拆塔换钱 → 去放闪电/去别处重建），
  而 SDK 启发式几乎只会建塔（85%）。
  ⇒ **rule_v4 的强度来自"一套活跃的经济循环"（建塔/升级/拆塔换钱/按冷却打闪电），
  不是"启发式 + 闪电"。**

  **对类轴计划的定位**：类轴要学的不是"每回合给动作类排序"，而是**执行这套动态经济**；
  而我们的智能体在动作空间里占的是**极端角落**（几乎只有闪电/空过）。
  这也**印证了类轴计划里"必须采到含降级/升级/建塔的数据"**（`az_class_axis_plan.md` §5(c)）：
  那些行为在我们的自对弈数据里**出现率≈0**，价值网自然判不了、类头自然也学不到。


## 2026-09-19 — 类轴第一步：`--inject-example-prob` 注入钩子 + 四条实测

> 手动/一次性命令，未走 `run_logged`。代码：`59796f3`（加钩子）、`807a8d6`（去过滤器）。
> 计划出处：`docs/az_class_axis_plan.md` §5(a)。

### 做了什么

`az_selfplay.py` 新增 `--inject-example-prob p`：以概率 p 把**搜索选的动作**换成
**ExampleAI 的动作**（`ai_example.py` 逐字逻辑：`bundles[1:8]` 里取 `(score, -len(ops))` 最大者）。
只改**实际执行**的动作；记录进样本的 `bundles`/`visit` 仍是搜索的 ⇒ **不污染策略目标**。
目的：把对局推向"有塔的局面"，给价值网/类头铺非闪电动作的覆盖。

### 发现 1（新）：**官方参考实现自己就是"建↔拆循环"的制造者**

`_tmp_probe_example_pick.py`（新增探针）：用 ExampleAI 的选法走一局，在"场上有塔"的决策点
打印 top-8 分数。直接看到机制：

```
r1  coins=35 塔=1  top8: build@4,9=24.0 ...            -> 选 BUILD
r2  coins=8  塔=2  n_bundles=5, b[0]=hold(0.0)
                   top8(b[1:8]): downgrade#2=-3.3, downgrade#0=-3.6, ...  -> 选 DOWNGRADE
r3  coins=35 塔=1  -> 又 BUILD
```

⇒ 金币耗尽时目录只剩 `[hold(0.0), downgrade(-3.3)]`，而 `ai_example.py` 取的是
`bundles[1:8]` —— **它主动跳过了 `bundles[0]` 那个 hold** ⇒ 拆一座塔换钱，下回合再建。
**真正出招的决策点里 26%(seed 3) / 39%(seed 0) 是降级。**

⇒ 这条**修正**了本会话早先的一个归因：拆塔有**两个独立来源**，不是一个。
  (a) 官方启发式 `bundles[1:8]` 会主动拆（上面这条）；
  (b) **我们自己的意图解码**：`code/my_ai/decoder.py:358-372` 写死的
      "选中 17-20 类但金币不足 ⇒ 用 16 通道的位置解码成 DOWNGRADE"。
  我们智能体真实轨迹里的拆塔是 (b)。

### 发现 2：**过滤器是 no-op**（所以在用户要求下删掉了）

中途我给注入加过 `_PRODUCTIVE` 过滤器（排除降级/超武），理由是发现 1。
用户要求保持 ExampleAI 原样，于是去掉。**实测：去掉零代价。**

| 配置（p=0.02，6 局 / 2536 回合）| 实际执行动作 |
|---|---|
| 带 `_PRODUCTIVE` 过滤器 | BUILD 164 / LIGHTNING 137 / DOWNGRADE 49 / UPGRADE 6 |
| **不带过滤器**（最终版）| **BUILD 164 / LIGHTNING 137 / DOWNGRADE 49 / UPGRADE 6** |

而且两边的 `az_progress_seed00000.txt` **逐字节相同**。
**原因**：ExampleAI 会注入降级的那些回合（2 塔 + 金币耗尽），我们自己的
意图解码 fallback 反正也会输出降级 ⇒ 两条路径同一动作。

### 发现 3：**注入比例**实测（修正一个我自己造成的假象）

早先我数的是样本里 `bundles[argmax(visit)]` = **搜索记录**，不是**实际执行**的动作
⇒ 得出过"p=0.05 全是降级"。改成从进度文件数实际执行后：

| 注入概率 | 回合数 | 出招回合 | 实际执行 | 建:拆 |
|---|---|---|---|---|
| p=0.02 | 2536 | 10.3% | BUILD 164 / LIGHTNING 137 / DOWNGRADE 49 / UPGRADE 6 | 3.3 : 1 |
| p=0.05 | 2396 | 18.2% | BUILD 384 / DOWNGRADE 161 / LIGHTNING 120 / UPGRADE 10 | 2.4 : 1 |

**两个都健康** ⇒ 用户"低比例就正常"的判断对，我的"与比例无关"撤回；但同时
`p=0.05` 会给 2.3x 的建塔覆盖（若 0.02 训完覆盖不够，这是下一个旋钮）。

### 发现 4（流程）：**playable 钉类不通**

`--pos-pin playable`（买不起闪电时不去降级、而是挑"真能执行"的类）+ **无注入**：
613 回合里 BUILD 704 / DOWNGRADE 700 / LIGHTNING 15 ⇒ **更糟的疯狂循环**。
⇒ "不用注入、靠 playable 就能产生建塔数据"的猜想被否掉。

### 发现 5（对照）：**p=0 时我们自己的策略只出闪电**，所以 trace 里所有非闪电动作都是注入

同一配置（`pos-only` + `pos-pin argmax` + `skip-single-candidate` + native）、只把
`--inject-example-prob` 设成 0，跑 3 局 / **1275 回合**：

| | 出招回合 | 实际执行 |
|---|---|---|
| **p=0（无注入，对照）** | 5.6% | **LIGHTNING:71，其余全 0** |
| p=0.02（正式采集） | 10.3% | BUILD 164 / LIGHTNING 137 / DOWNGRADE 49 / UPGRADE 6 |

⇒ ① **我们的策略在这个配置下的动作空间就是"闪电 + 空过"**（类头塌缩，与 §2 的诊断一致）；
② 所以采集 trace 里出现的 **BUILD/UPGRADE 必定是注入**，**DOWNGRADE 是意图解码的自动降级**
（对照里 0 降级 = 没有塔可拆）—— 这给了"读 trace"一把标尺；
③ 注入把出招回合占比从 5.6% 抬到 10.3%（≈ 翻倍），即注入确实在改变对局形态。

样例（第 1 局，`training_history/inject_ex02/data/az_progress_seed10098.txt`，383 回合、
winner=0、35 个出招回合，其余全是双方空过攒钱）：
`BUILD 24 / LIGHTNING 21 / DOWNGRADE 5 / UPGRADE 2`。可辨认的注入：
`r29 P1 [BUILD(13,9),BUILD(14,9)]`、`r88 P0 [UPGRADE(id=3->type=2),UPGRADE(id=5->type=2)]`
⇒ **不只是建塔，升级也进来了**；而且塔能活一段（r29 建的塔到 r63 才被拆，r88 升级的塔到 r98 才拆）。

### 下一步（判据口径先钉死，免得又"不可比"）

400 局采集已启动：`inject_ex02_400g`（p=0.02、不过滤、pos-only + skip-single-candidate +
native-engine、256 iters / depth 4、k=24、seed=1 ⇒ games 10000..10399、16 workers）。

**判据（训完后照打）：**
```bash
D:/anaconda3/envs/pytorch-gpu/python.exe -u _tmp_rank_quality.py \
    --ckpt training_history/<新 3 网 ckpt>.pt        # 其余全用默认
```
默认口径（与旧基线**逐字相同**才能比）：`--games 12 --workers 6 --k 96 --seed 31 --max-rounds 512`。

| | 数值 |
|---|---|
| 旧基线 ckpt | `training_history/az_fixed/three_mix_r10p_vw_pol_frozen.pt` |
| 旧基线 随机分位 | 49.6% (SE 0.4) |
| 旧基线 **价值网分位** | **56.4% (SE 0.5)** |
| 旧基线 Spearman ρ | 0.139 (SE 0.006) |
| 决策点数 | 4476（"有 ≥2 候选"的回合，候选池均 92） |

⚠️ **新 ckpt 的构造方式是关键**：`_tmp_rank_quality.py` 里那些对局是用 `--ckpt` 自己的
class/pos 网走出来的 ⇒ 要拿**干净 A/B**，新 ckpt 必须是 `three_mix_r10p_vw_pol_frozen.pt`
的 `class_state`/`pos_state` **照抄**、**只替换 `value_state`**（这样对局逐位相同，
唯一变量就是价值网）。

## 2026-09-19 17:46:49 — inject_ex02_400g

- **commit**: `807a8d6` (dirty: 17 files)
- **exit**: 0，用时 4090s
- **cmd**:
  ```bash
  D:/anaconda3/envs/pytorch-gpu/python.exe -u code/my_ai/az_intent/az_selfplay.py --checkpoint training_history/vprior/posnet_r1.pt --games 400 --workers 16 --seed 1 --iterations 256 --max-depth-rounds 4 --t-class 0.5 --t-pos 1.0 --k 24 --search-mode pos-only --skip-single-candidate --native-engine --inject-example-prob 0.02 --out-dir training_history/inject_ex02/data
  ```
- **output**: `training_history/runs/20260919_174648_inject_ex02_400g/output.log`
- **result**: ✅ 采集成功：**400 局全部打到 terminal**，**369,476 个决策点**（每局均 924），
  产物 `training_history/inject_ex02/data/az_selfplay_seed*.pkl` 共 **17 GB**。
  用时 68 分钟（"约 1 小时"的估计对上了）。

  **这批数据到底带来了什么 —— 用 `_tmp_inject_data_stats.py` 直接数 `board` 通道：**
  （通道 4 = 己方塔、通道 5 = 敌方塔，见 `SDK/utils/features.py:177-180`）

  | 数据集 | 决策点 | **场上有己方塔** | 有 ≥2 己方塔 | 场上有敌方塔 |
  |---|---|---|---|---|
  | **注入前**：`vprior/vp_400.npz`（训 `posnet_r1` 用的）| 24,972 | **0.0%** | 0.0% | **0.0%** |
  | **注入前**：`az_fixed/mixdata_mix13`（老训练数据）| 804/局 | **0.0%** | 0.0% | **0.0%** |
  | **注入后**：`inject_ex02/data`（本批）| **369,476** | **38.9%** | **28.2%** | **39.4%** |

  ⇒ 🔴 **决定性对比**：此前**所有**自对弈训练数据里，"场上有一座塔"的决策点是 **0.0%**
  （敌方塔也是 0.0% ⇒ 整盘棋从头到尾没有任何塔）。本批把"有塔局面"的覆盖从 **0 → 38.9%**、
  "≥2 塔"到 **28.2%**。这正是 §2"价值网对其他动作是瞎的"那条诊断的**直接证据**，
  也是 (a) 唯一想改变的东西。
  （通道索引可信度交叉验证：同一个通道在注入数据里是 38.9%、在两份老数据里是 0.0%
  ⇒ 不是"取错通道导致恒为 0"。）

  player0/1 样本数 184,738 / 184,738（完全均衡）；value_target 每局内标准差均 0.308
  （局间均值恒为 0，是"两玩家样本数相同、标签互为相反数"的构造使然，非 bug）。

## 2026-09-19 19:00:11 — valnet_train_arms

- **commit**: `aea221a` (dirty: 17 files)
- **exit**: 0，用时 780s
- **cmd**:
  ```bash
  bash -c export PYTHONIOENCODING=utf-8
echo '########## 臂 A：全量微调（lr 3e-4, 4 epochs）##########'
D:/anaconda3/envs/pytorch-gpu/python.exe -u code/my_ai/az_intent/train_value_net.py --ckpt training_history/az_fixed/three_mix_r10p_vw_pol_frozen.pt --cache training_history/inject_ex02/value_cache --epochs 4 --lr 3e-4 --out training_history/inject_ex02/valnet_A.pt
echo '########## 臂 B：--freeze-backbone（只训头）##########'
D:/anaconda3/envs/pytorch-gpu/python.exe -u code/my_ai/az_intent/train_value_net.py --ckpt training_history/az_fixed/three_mix_r10p_vw_pol_frozen.pt --cache training_history/inject_ex02/value_cache --epochs 6 --lr 1e-3 --freeze-backbone --out training_history/inject_ex02/valnet_B.pt
  ```
- **output**: `training_history/runs/20260919_190011_valnet_train_arms/output.log`
- **result**: 价值网在注入数据上微调，**有塔样本的误差改善明显大于整体**（这正是 (a) 想要的）：

  | 臂 | 可训参数 | 最好 val MSE | 有塔 MSE | 无塔 MSE | r |
  |---|---|---|---|---|---|
  | **训练前**（旧价值网）| — | 0.05815 | **0.06325** | 0.05503 | +0.526 |
  | A 全量微调 4ep（lr 3e-4）| 553,313 | **0.05517** @ep2 (−5.1%) | **0.05744** (−9.2%) | 0.05379 (−2.3%) | +0.556 |
  | B `--freeze-backbone` 6ep | 21,473 | 0.05616 @ep5 (−3.4%) | 0.05719 (−9.6%) | 0.05552 | +0.542 |

  **① 微调真的在学"有塔的局面"**：有塔样本 MSE 降 9%+，**大于整体降幅（3-5%）**，
  也大于无塔样本（2%）⇒ 改善集中在被注入出来的那部分。
  **② 收敛极快、随即过拟合**：ep1-2 就到位，之后 val 反弹（A 的 ep3/ep4 都变差）；
  `freeze-backbone` 只训 21k 参数也能拿到接近的效果 ⇒ 瓶颈不在容量。
  **③ ⚠️ 训练前发现 BN 陷阱**（见下条 `valnet_train_clean`）。

  ⚠️ **这个 MSE 读数的解释力有限**：terminal 标签是"每局一个值、按 player 签名"，
  一局内所有决策点同值 ⇒ 18,778 个 val 样本背后只有 **20 个独立结局**，
  r=+0.53 主要在说"它能看出谁占优"，不是"它对动作后果敏感"。

## 2026-09-19 19:13:32 — valnet_train_clean

- **commit**: `8eb7e07` (dirty: 19 files)
- **exit**: 0，用时 643s
- **cmd**:
  ```bash
  bash -c export PYTHONIOENCODING=utf-8
echo '########## 臂 A2：全量微调 + freeze-bn（terminal 标签，3 epochs）##########'
D:/anaconda3/envs/pytorch-gpu/python.exe -u code/my_ai/az_intent/train_value_net.py --ckpt training_history/az_fixed/three_mix_r10p_vw_pol_frozen.pt --cache training_history/inject_ex02/value_cache --epochs 3 --lr 3e-4 --freeze-bn --out training_history/inject_ex02/valnet_A2.pt
echo '########## 臂 C：abs+kgeo 标签（每帧有各自标签）+ freeze-bn，3 epochs ##########'
D:/anaconda3/envs/pytorch-gpu/python.exe -u code/my_ai/az_intent/train_value_net.py --ckpt training_history/az_fixed/three_mix_r10p_vw_pol_frozen.pt --cache training_history/inject_ex02/value_cache --epochs 3 --lr 3e-4 --freeze-bn --label-mode abs --label-weight kgeo --tau 20 --out training_history/inject_ex02/valnet_C.pt
  ```
- **output**: `training_history/runs/20260919_191332_valnet_train_clean/output.log`
- **result**: 🔴 **先修掉一个会污染归因的坑：源价值网的 BatchNorm 从来没训过。**

  `three_mix_r10p_vw_pol_frozen.pt` 里 **`value_state` 的所有 BN 都是初值**
  （`running_mean` 全 0、`running_var` 全 1、`num_batches_tracked=0`）
  ⇒ 在 `eval()` 下 BN 等于恒等（除以 sqrt(1+eps)）。
  对照：同 ckpt 的 `class_state`（策略网）BN 是训过的（`num_batches_tracked=21475`、
  `running_var` 到 1260）。

  后果：直接 `model.train()` 微调时 running 统计量朝新数据猛冲 ——
  实测 `resblocks.5.bn1.running_var` **1 → 1.473e4**，`resblocks.4/3/2` 依次 4401/2419/1882。
  那样"val MSE 变好"里会混进"BN 统计量适配了新分布"，不是纯"学会判有塔局面"。
  ⇒ 新增 `--freeze-bn`（**默认开**）：父模块仍 `train`，只把 BN 子模块切到 `eval`
  （`_set_bn_eval`），每 epoch 打印 BN var 极值确认冻住（实测整轮后仍 `=1`）。

  | 臂（3-6 epochs）| 最好 val MSE | 有塔 MSE | 备注 |
  |---|---|---|---|
  | **训练前** | 0.05815 | 0.06325 | — |
  | A2 全量微调 + freeze-bn 3ep | **0.05581** @ep2 | 0.05794 | 与不冻 BN 的 A（0.05517）几乎一样 |
  | C `abs + kgeo` 标签 + freeze-bn 3ep | **0.00308** @ep3 | 0.00347 | 见下 |

  **① 冻 BN 几乎不影响结果**（A2 0.05581 vs A 0.05517）⇒ 前面 A 的改善不是 BN 漂移带来的，
  结论不变。**② 臂 C 的 MSE 不能和 A 直接比**：`abs+kgeo` 标签**每帧各不相同**，
  而"训练前"的旧网在这个标签上 r 就已经 **+0.910**（MSE 0.00737 vs 常数 0.04131）
  ⇒ 这个标签本身好预测（未来 HP 均值与当前优势高度相关），MSE 降 58% 主要说明"拟合上了"，
  **不说明"对动作更敏感"**。它是否真的更有用，只能看判据（下条 `valnet_criterion`）。

## 2026-09-19 19:54:37 — k24_menu_guard

- **commit**: `8b45ad6` (dirty: 17 files)
- **exit**: 0，用时 220s
- **cmd**:
  ```bash
  bash -c export PYTHONIOENCODING=utf-8
echo '########## 零假设守卫：first vs first（两侧同策略 ⇒ 镜像必须恰好 0.5000）##########'
D:/anaconda3/envs/pytorch-gpu/python.exe -u code/test_match/heuristic_candidates_match.py --our first --opp first --k 24 --pairs 64 --workers 12 --seed 0
echo '########## 菜单可行性：random(top-24 里乱选) vs first(官方第一名) ##########'
D:/anaconda3/envs/pytorch-gpu/python.exe -u code/test_match/heuristic_candidates_match.py --our random --opp first --k 24 --pairs 64 --workers 12 --seed 0
  ```
- **output**: `training_history/runs/20260919_195437_k24_menu_guard/output.log`
- **result**: 🔴 **零假设守卫完美通过，但可行性守卫失败 ⇒ 用户提的"官方 top-24 菜单 + 价值网搜索"
  这个臂按原样**测不出东西**。**（64 对 / 128 局，K=24，镜像配对）

  | 臂 | 配对胜率 | t | 净胜局 |
  |---|---|---|---|
  | **① `first` vs `first`**（两侧同策略）| **0.5000（SE=0）** | — | **0** |
  | ② `random`（菜单里乱选）vs `first`（官方第一名）| **0.4922**（SE 0.0438）| **−0.18** | **−1** |

  **① 守卫通过**：0/64 个 pair 偏离 0.5，且两侧动作构成**逐项相同**
  ⇒ 镜像台子精确、两侧策略确实相同。
  **② 菜单内部的官方排序没有可测信息**：随机乱选 vs 永远选第一名，差 **−1 局（t=−0.18）**。
  更狠的是：随机那一侧的行为**变化很大**（降级 14% vs 对方 4%），**结果却完全不变**
  ⇒ 这份菜单里根本没有"选错要付代价"的选项。

  **结构性原因（顺带测出来的，比结论本身更有用）**：
  - 菜单大小 **中位 1、均值 3.5**（金币不够时空过是唯一合法）⇒ **大部分回合根本没得选**；
  - 菜单动作构成 **建塔 73~75%、闪电 ~0%** ⇒ 菜单里的差异主要是"建在哪一格"，
    而那些格子近乎等价。
  ⇒ 这也解释了为什么"per-class 的类轴"才是关键：这份菜单把**类轴（闪电 vs 不闪电）
    整条丢掉了**，只剩下近乎等价的同内选择。

  ⚠️ 注意守卫的边界：`random vs first` = 0.4922 ± 0.044 只证明"**官方排序**在这份菜单里没信息"
  （|Δ| 大致被 95% CI 限制在 ±9pp 内），**不等于"任何排序都没信息"**。但结合"中位 1 个候选 +
  选错无代价"，这个臂的**鉴别力**已经被结构性上限卡死 ⇒ 不该按原样投两小时。
  **修法**：把类轴放回菜单（官方 top-24 **∪ 评分最高的闪电候选**，harness 里已有 `value_mix`
  就是这个口径），这样"闪电 vs 不闪电"这个值 ~80pp 的取舍（`rv4_lightning_only`）才在菜单里。
  见 `docs/az_heuristic_menu_search_plan.md`。

## 2026-09-19 19:24:23 — valnet_criterion

- **commit**: `18b8ad2` (dirty: 18 files)
- **exit**: 0，用时 2109s
- **cmd**:
  ```bash
  bash -c export PYTHONIOENCODING=utf-8
echo '########## 守卫：基线 three_mix_r10p_vw_pol_frozen（应复现 56.4% / 49.6%）##########'
D:/anaconda3/envs/pytorch-gpu/python.exe -u _tmp_rank_quality.py --ckpt training_history/az_fixed/three_mix_r10p_vw_pol_frozen.pt
echo '########## 臂 A2：注入数据 + terminal + freeze-bn ##########'
D:/anaconda3/envs/pytorch-gpu/python.exe -u _tmp_rank_quality.py --ckpt training_history/inject_ex02/valnet_A2.pt
echo '########## 臂 C：注入数据 + abs+kgeo + freeze-bn ##########'
D:/anaconda3/envs/pytorch-gpu/python.exe -u _tmp_rank_quality.py --ckpt training_history/inject_ex02/valnet_C.pt
  ```
- **output**: `training_history/runs/20260919_192423_valnet_criterion/output.log`
- **result**: 🟢 **判据动了，而且幅度很大**（三臂 n 完全一致 = 9018，SE 各 0.3）：

  | ckpt | 价值网 1-ply argmax 分位 | Spearman ρ | 随机分位 |
  |---|---|---|---|
  | 基线 `three_mix_r10p_vw_pol_frozen` | **55.9%** | 0.128 | 49.5% |
  | **臂 A2**（注入数据 + `terminal` 标签 + 冻 BN）| **61.2%** | **0.315** | 49.5% |
  | 臂 C（注入数据 + `abs+kgeo` 标签 + 冻 BN）| **59.8%** | 0.233 | 49.5% |

  **① 主结论：注入数据显著改变了价值网的排序**（+5.3pp 分位、ρ 0.128→0.315，SE 0.3 ⇒ z≈12.5）。
  **② "数据覆盖"就是有效杠杆，不是"标签设计"**：A2 只换了数据、标签配方与产出旧价值网的
  value_warmup 逐字相同，却拿到最大的提升；反过来臂 C（换标签）拿到的更小（59.8 < 61.2）。
  这正好落在 `az_class_axis_plan.md` §5(a) 赌的那一条上（"新意在于从标签设计换到数据覆盖"）。
  **③ 守卫通过**："随机"那一列三臂**完全相同（49.5%）**⇒ 三臂的对局与决策点逐位相同，
  唯一变量确实是价值网（`class_state`/`pos_state` 已另验逐位照抄）。

  ⚠️ **但这条判据的"尺子"本身弱，不能当成"价值网变好了"的充分证据**（用户 2026-09-19 提出）：
  官方 score 在**大池子里**有信号（k=96 时 top-1 vs random = 0.6406, t=+2.06），但在
  **好端内部**是噪声（k=8 时 0.4531, t=−0.90）。所以分位上升只说明"价值网的排序朝官方 score
  靠了"，不等于"棋力会变好"。**对局级判据见下条 `k24_menu_guard` / `az_heuristic_menu_search_plan.md`。**

## 2026-09-19 20:03:46 — menu_lightning_arms

- **commit**: `a52be2a` (dirty: 18 files)
- **exit**: 2，用时 69s
- **cmd**:
  ```bash
  bash -c export PYTHONIOENCODING=utf-8
echo '########## 版本1 · 1-ply ##########'
echo '#### 1A 菜单天花板：random over (top-24 U 闪电) vs first ####'
D:/anaconda3/envs/pytorch-gpu/python.exe -u code/test_match/heuristic_candidates_match.py --our random --opp first --k 24 --menu-lightning --pairs 128 --workers 16 --seed 0 --ckpt training_history/az_fixed/three_mix_r10p_vw_pol_frozen.pt
echo '#### 1B 价值网(旧) over (top-24 U 闪电) vs first ####'
D:/anaconda3/envs/pytorch-gpu/python.exe -u code/test_match/heuristic_candidates_match.py --our value --opp first --k 24 --menu-lightning --pairs 128 --workers 16 --seed 0 --ckpt training_history/az_fixed/three_mix_r10p_vw_pol_frozen.pt
echo '#### 1C 价值网(新=A2) over (top-24 U 闪电) vs first ####'
D:/anaconda3/envs/pytorch-gpu/python.exe -u code/test_match/heuristic_candidates_match.py --our value --opp first --k 24 --menu-lightning --pairs 128 --workers 16 --seed 0 --ckpt training_history/inject_ex02/valnet_A2.pt
echo '########## 版本2 · 搜索（iters=64 depth=4）##########'
echo '#### 2D 搜索(旧价值网) over (top-24 U 闪电) vs first ####'
D:/anaconda3/envs/pytorch-gpu/python.exe -u code/test_match/heuristic_candidates_match.py --our search --opp first --k 24 --menu-lightning --search-iters 64 --search-depth 4 --pairs 128 --workers 16 --seed 0 --ckpt training_history/az_fixed/three_mix_r10p_vw_pol_frozen.pt
echo '#### 2E 搜索(新价值网=A2) over (top-24 U 闪电) vs first ####'
D:/anaconda3/envs/pytorch-gpu/python.exe -u code/test_match/heuristic_candidates_match.py --our search --opp first --k 24 --menu-lightning --search-iters 64 --search-depth 4 --pairs 128 --workers 16 --seed 0 --ckpt training_history/inject_ex02/valnet_A2.pt
  ```
- **output**: `training_history/runs/20260919_200346_menu_lightning_arms/output.log`
- **result**: _待填_

## 2026-09-19 20:13:26 — random_menu24_arms

- **commit**: `b3a9f97` (dirty: 18 files)
- **exit**: 127，用时 0s
- **cmd**:
  ```bash
  bash -c export PYTHONIOENCODING=utf-8
O=training_history/az_fixed/three_mix_r10p_vw_pol_frozen.pt; N=training_history/inject_ex02/valnet_A2.pt
echo '#### G 守卫：random vs random（期望恰好 0.5000）####'
$PY -u $S --our random --opp random --k 24 --menu-random 24 --pairs 32 --workers 16 --seed 0 --ckpt $O
echo '#### L 菜单活性：first(子集分最高) vs random(子集乱选) ####'
$PY -u $S --our first --opp random --k 24 --menu-random 24 --pairs 128 --workers 16 --seed 0 --ckpt $O
echo '#### P1 1-ply 头对头：value(新) vs value(旧) ####'
$PY -u $S --our value --opp value --k 24 --menu-random 24 --pairs 128 --workers 16 --seed 0 --ckpt-our $N --ckpt-opp $O
echo '#### S1 搜索(旧) vs random ####'
$PY -u $S --our search --opp random --k 24 --menu-random 24 --search-iters 64 --search-depth 4 --pairs 64 --workers 16 --seed 0 --ckpt-our $O
echo '#### S2 搜索(新) vs random ####'
$PY -u $S --our search --opp random --k 24 --menu-random 24 --search-iters 64 --search-depth 4 --pairs 64 --workers 16 --seed 0 --ckpt-our $N
echo '#### S3 主判据·搜索头对头：search(新) vs search(旧) ####'
$PY -u $S --our search --opp search --k 24 --menu-random 24 --search-iters 64 --search-depth 4 --pairs 128 --workers 16 --seed 0 --ckpt-our $N --ckpt-opp $O
  ```
- **output**: `training_history/runs/20260919_201326_random_menu24_arms/output.log`
- **result**: ❌ **启动失败（exit 127，0s）**——我把 `$PY`/`$S` 转义成了 `\$PY`，
  内层 `bash -c` 里这两个变量是空的（外层只 `PY=...` 没 `export`）⇒ 整条命令没跑。
  已用 `export` 重发为下面那条 `20:13:53` 的 run。**本条只作排错记录，无结果。**

## 2026-09-19 20:06:35 — menu_lightning_arms

- **commit**: `325966c` (dirty: 18 files)
- **exit**: 0，用时 608s
- **cmd**:
  ```bash
  bash -c export PYTHONIOENCODING=utf-8
echo '########## 版本1 · 1-ply ##########'
echo '#### 1A 菜单天花板：random over (top-24 U 闪电) vs first ####'
D:/anaconda3/envs/pytorch-gpu/python.exe -u code/test_match/heuristic_candidates_match.py --our random --opp first --k 24 --menu-lightning --pairs 128 --workers 16 --seed 0 --ckpt training_history/az_fixed/three_mix_r10p_vw_pol_frozen.pt
echo '#### 1B 价值网(旧) over (top-24 U 闪电) vs first ####'
D:/anaconda3/envs/pytorch-gpu/python.exe -u code/test_match/heuristic_candidates_match.py --our value --opp first --k 24 --menu-lightning --pairs 128 --workers 16 --seed 0 --ckpt training_history/az_fixed/three_mix_r10p_vw_pol_frozen.pt
echo '#### 1C 价值网(新=A2) over (top-24 U 闪电) vs first ####'
D:/anaconda3/envs/pytorch-gpu/python.exe -u code/test_match/heuristic_candidates_match.py --our value --opp first --k 24 --menu-lightning --pairs 128 --workers 16 --seed 0 --ckpt training_history/inject_ex02/valnet_A2.pt
echo '########## 版本2 · 搜索（iters=64 depth=4）##########'
echo '#### 2D 搜索(旧价值网) over (top-24 U 闪电) vs first ####'
D:/anaconda3/envs/pytorch-gpu/python.exe -u code/test_match/heuristic_candidates_match.py --our search --opp first --k 24 --menu-lightning --search-iters 64 --search-depth 4 --pairs 128 --workers 16 --seed 0 --ckpt training_history/az_fixed/three_mix_r10p_vw_pol_frozen.pt
echo '#### 2E 搜索(新价值网=A2) over (top-24 U 闪电) vs first ####'
D:/anaconda3/envs/pytorch-gpu/python.exe -u code/test_match/heuristic_candidates_match.py --our search --opp first --k 24 --menu-lightning --search-iters 64 --search-depth 4 --pairs 128 --workers 16 --seed 0 --ckpt training_history/inject_ex02/valnet_A2.pt
  ```
- **output**: `training_history/runs/20260919_200635_menu_lightning_arms/output.log`
- **result**: 🛑 **作废（被用户叫停的半途 run）——不是结果，是事故记录。**

  用户换掉菜单设计（改成"从合法动作里随机抽 24 个"，见下条 `random_menu24_arms`），
  所以这条跑到一半被我停掉。**但"停"没停干净，这是本节要记的教训**：

  * 我用 `TaskStop` 停的只是 `run_logged.sh` 那层 shell，**它下面的 `bash -c` → python 是孙进程，
    变成孤儿继续跑**；而且 `bash -c` 的多行脚本**默认不会因为某一行失败就中止**，
    所以它还继续往下跑了后面的臂（被发现时已经跑到 2D 臂）。
  * 于是它和随后 20:13:53 重发的 `random_menu24_arms` **并行**，机器上出现
    **34 个 python 进程（两套各 17 个）** —— 用户先看到的就是这个。
  * 清理：用 `taskkill /PID <bash -c 的 pid> /T /F` 杀掉**整棵树**才对
    （先误杀了 fork 出的子壳 44652，真正的根是 52632）。杀完确认只剩 17 个。

  ⚠️ **对结果无影响**：本台子用固定 seed、不依赖时序 ⇒ 并行只是拖慢速度，不改变数值。
  ⚠️ **教训**：这批台子的启动方式是 `run_logged.sh <name> bash -c "<多行脚本>"`，
  停要用进程树级别的杀法（`taskkill /T /F` + 按 ppid 分组核对），别只依赖 `TaskStop`。

  半途产出（**不可用**，仅存档）：1A 跑完、1B 中断、2D 刚起。
  `20260919_200346_menu_lightning_arms` 那条是同一套臂的更早一次启动，
  `--menu-lightning` 当时还没实现 ⇒ 5 个臂全部 argparse 报错（exit 2）。
