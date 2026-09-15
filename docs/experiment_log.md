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
