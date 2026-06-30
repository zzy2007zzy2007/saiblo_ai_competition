# 模型行为诊断工具

`code/test_match/diagnose_model.py` — 分析模型在与 ExampleAI 对战时实际选择了哪些动作，用于诊断模型退化或行为异常。

## 用法

```bash
# 默认：10 局，只输出每局汇总
python code/test_match/diagnose_model.py <checkpoint.pt>

# 1 局 + 逐回合输出每个动作
python code/test_match/diagnose_model.py <checkpoint.pt> --games 1 --verbose

# 分析种群均值（非 top1）
python code/test_match/diagnose_model.py <checkpoint.pt> --games 5
```

## 选项

| 参数 | 默认 | 说明 |
|------|------|------|
| `ckpt` | 必填 | checkpoint 路径 |
| `--games` | 10 | 对局数 |
| `--verbose, -v` | 否 | 逐回合打印模型选择的动作 |

## 输出说明

- 每局结束后打印：种子、胜负、双方血量、回合数、操作数、HOLD 次数
- 汇总部分打印：
  - 胜率、平均回合数
  - **每回合平均操作数**（远小于 3 说明模型频繁 HOLD）
  - **HOLD 占比**（什么也不做的回合比例）
  - **操作类型分布**（BUILD / UPGRADE / DOWNGRADE / 超级武器 / 基地升级）

---

## 诊断记录

### 训练 3 gen_0015 top1 — 2026-06-30

**命令**：`python code/test_match/diagnose_model.py training_history/20260630_182516/gen_0015.pt --games 1 --verbose`

**结果**：

| 指标 | 值 |
|------|:----:|
| 胜率 | 1/1（100%，巧合） |
| 总回合 | 258 |
| 总操作 | 23 |
| **HOLD 占比** | **91.1%**（235/258 回合什么也不做） |
| 每回合操作数 | 0.089 |

**操作类型分布**：

| 操作 | 次数 | 占比 |
|------|:----:|:----:|
| BUILD（建 Basic 塔） | 19 | 82.6% |
| UPGRADE（升 Producer） | 4 | 17.4% |

**诊断结论**：

模型完全退化，表现如下：
- **91% 的回合选择 HOLD**，不执行任何操作
- 仅有的动作全是**在自家高地建 Basic 塔**，或**升级成 Producer**
- **从不使用**：超级武器、基地升级（出兵速度/蚂蚁血量）、降级/拆除
- **从不升级**：Heavy / Quick / Mortar 等进攻型防御塔分支

这种"只建塔挂机"策略在 50 局测试中胜率 0%，偶尔靠拖满 512 回合侥幸取胜。

---

# Checkpoint 对比工具

`code/test_match/compare_checkpoints.py` — 让两个 checkpoint 的 top1 个体直接对战，用于分析代际间的相对强弱关系。

## 用法

```bash
python code/test_match/compare_checkpoints.py <ckpt_a.pt> <ckpt_b.pt> --games 10 --workers 4
```

## 选项

| 参数 | 默认 | 说明 |
|------|------|------|
| `ckpt_a` | 必填 | 第一个 checkpoint |
| `ckpt_b` | 必填 | 第二个 checkpoint |
| `--games` | 10 | 对局数 |
| `--workers` | 4 | 并行进程数 |

## 对比记录

### 训练 4 各代 head-to-head vs gen_0008（10 局 4 进程）

```text
gen_0008 vs gen_0004:  gen_8 胜率 70%
gen_0008 vs gen_0005:  gen_8 胜率 40%
gen_0008 vs gen_0006:  gen_8 胜率 20%
gen_0008 vs gen_0007:  gen_8 胜率 70%
```

**结果解读**：各代塌缩模型之间存在石头剪刀布关系——gen_6 的塔阵型克制 gen_8，gen_8 克制 gen_4/7，但 gen_5 又能压制 gen_8。自对弈排名（种群内互打）与对阵 ExampleAI 的能力完全无关。

---

## 重要发现：自对弈 vs 真实对战能力完全无关

从训练 4 各代的测试数据可以明显看出，模型在自对弈中的表现（head-to-head 胜率）与对阵 ExampleAI 的胜率**几乎不相关**：

| 模型 | vs ExampleAI | vs gen_8 |
|:----:|:-----------:|:--------:|
| gen_4 | 0% (20局) | 30% (10局) |
| gen_5 | **65%** (20局) | **60%** (10局) |
| gen_6 | 40% (50局) | **80%** (10局) |
| gen_7 | **60%** (20局) | 30% (10局) |
| gen_8 | 40% (20局) | — |

- **gen_7**: ExampleAI 打 60%，打 gen_8 只有 30% — 高 Example 低内战
- **gen_6**: ExampleAI 只打 40%，打 gen_8 有 80% — 低 Example 高内战
- **gen_5**: 两样都高，gen_4: 两样都低——这两个算巧合对齐

**结论**：所有模型都塌缩到"建塔挂机"策略，但不同的塔阵型之间存在石头剪刀布关系。ES 的自对弈适应度（种群内互打分）完全不能反映模型对阵真实对手（ExampleAI）的能力，这是 ES 训练无法继续进步的根因。
