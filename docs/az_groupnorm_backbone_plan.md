# 用 GroupNorm 解冻骨干训练价值头（2026-09-11）

## 动机

今天确认了两个互相冲突的约束：

| 归一化 | 骨干**不能**训练的原因 |
|--------|----------------------|
| `no_bn`（无归一化） | 激活尺度失控 → 特征坍缩（§20 的原始发现）|
| BatchNorm | running 统计在训练中漂移 → **训练/推理是两个不同函数**（实测：同样数据 + 冻结骨干，BN 9.4% vs no_bn 28.1%）|

而"冻结骨干"虽然绕过了两者（no_bn + 冻结 → 28-31%），但**天花板 = init 骨干的特征**，
而 `gen_0120` 的骨干是**为策略**（ES/GA）训的、不是为价值优化的。

**所以需要的是一种"逐样本"归一化**：不需要 running 统计（训练==推理，逐位相同）
同时控制激活尺度（防坍缩）。GroupNorm 满足这两条，且是卷积网络 + 小 batch 的标准选择
（Wu & He, *Group Normalization*, 2018）。

## 方案

给网络加归一化选项，用一个可切换的参数代替 `no_bn` 布尔：

```
norm = "bn"   → BatchNorm2d        （现状，训练/推理不一致）
norm = "none" → Identity           （现状的 no_bn，会坍缩）
norm = "gn"   → GroupNorm(8, C)    ← 新增
```

GroupNorm 与 BatchNorm 的**可学习参数形状相同**（每通道 γ、β），只是**没有 running 统计**，
所以从现有 no_bn 模型做转换很简单（和当初的 BN 迁移同套路）：
- conv 权重原样复制
- **conv bias → GN 的 β**，γ 初始化为 1
- 转换后输出**不等价**（GN 逐样本归一化会改变函数），需要重新训练

## 实验设计

数据：**合并数据 536 局**（gen 200 + 自对弈 336，决定性局数 ~350），
训练方法：`value_warmup`（价值头重新随机初始化）+ `--skip-collect`，10 epochs。

| arm | 归一化 | 骨干 | 目的 |
|-----|-------|------|------|
| 基线 | none | 冻结 | 31.2%（已知）|
| **H** | **gn** | **解冻** | **能不能突破 31%？** ← 核心问题 |
| I | gn | 冻结 | 对照：分离"归一化本身"和"解冻"的贡献 |
| J | gn | 解冻，输入用**自对弈 336 局** | 分离"数据"和"归一化" |

评价：az_r10 策略 + 各价值头 vs rule_v4 32 局（seed 0-31，配对可比；
基准 = 原版 gen 头 34.4%、A 31.2%）。

## 改动清单

1. `code/my_ai/network.py`
   - `ResBlock` / `AntWarNetwork` / `create_model` 增加 `gn: bool = False, gn_groups: int = 8`
   - `gn=True` 时 conv 无 bias + GroupNorm；checkpoint 写 `gn` / `gn_groups` 字段
2. `code/my_ai/az_intent/convert_no_bn_to_gn.py`（新）
   - no_bn 权重 → GN 权重（conv bias → GN β，γ=1）
3. 加载路径统一读 `ckpt.get("gn")` / `ckpt.get("gn_groups")`
   - `az_selfplay.load_model_from_ckpt` / `load_split_models` / `make_net_fn_from_ckpt`
   - `az_intent/train.build_model` / `load_hotstart`
   - `az_intent/eval.py`、`value_warmup.py`、`az_train.py` 的建网处
4. `value_warmup.py`：`--hotstart` 已支持任意 checkpoint；加 `--gn`（保留 GN 而不是折叠）
   —— 与已有的 `--keep-bn` 对称

## 判读

- **H 明显 >31%** → 冻结骨干确实是天花板，GroupNorm 解冻是正解 → 闭环里的价值头可以持续变强
- **H ≈ 31%** → 天花板不在骨干，而在数据/任务本身（价值头的信息上限）→ 转对手池方向
- **H <31%** → GroupNorm 训练本身有问题（需查实现）

## 风险

- GroupNorm 改变了网络结构 → **必须重新训练**，现有 checkpoint 不能直接用（转换后输出不等价）
- 所有加载路径都要读新字段，漏一处会导致加载失败（用 `ckpt.get(..., False)` 默认值兜底）
