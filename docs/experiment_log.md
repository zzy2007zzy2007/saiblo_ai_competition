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

---

## 2026-09-09 — 冻结骨干修复：2×2 定位实验

### 设计

| 组合 | 骨干 | 价值头 | 数据 | 结果 |
|---|---|---|---|---|
| A | CPU 解 | 联合训练（不冻结） | warm_data_cpp | **34.4%** |
| D | GPU 解 | 联合训练（不冻结） | warm_data_cpp | **15.6%** |
| F1 | CPU 解（冻结） | 重训 | warm_data_cpp | **25.0%** |
| F2 | GPU 解（冻结） | 重训 | warm_data_cpp | **34.4%** |

（A/D 是 `value_warmup` 全参训练；F1/F2 是 `--freeze-backbone`，只训 21K 参数）

### 结论

1. **冻结骨干后两个骨干都给出好结果**（25% / 34.4%），包括那个"坏"的 GPU 骨干
2. **骨干本身不坏**——坏的是**联合训练**：骨干持续漂移，价值头在追移动靶，最终未收敛好
3. 冻结后 trainable 从 220 万降到 **2.1 万**，混沌敏感度大降 → 结果稳定，换设备/换骨干都不再翻车
4. 曾试图用"交叉互换"（换骨干/换头）定位，但**价值头与骨干是配套训练的**（§13 的策略-价值强耦合同理），换头必然失效 → 该设计被废弃，改用"冻结骨干 + 重训头"

### 工程解

价值头训练改用 `value_warmup.py --freeze-backbone`。
