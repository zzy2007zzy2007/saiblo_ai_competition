# Baseline 模型加载和对战测试设计方案

## 1. 设计目标

验证所有 baseline 模型的加载正确性和对战功能的正常运行，包括**神经网络型模型**和**规则型 AI**。

## 2. 测试范围

- **模型加载测试**：验证每个 baseline 模型能够正确加载
- **对战功能测试**：验证模型能够与 BasicTowerAI 正常对战（每场对战2局）

## 3. 模型分类

| 模型类型 | 模型名称 | 特点 | 测试方式 |
|---------|---------|------|---------|
| **神经网络型** | ann_593, ann_v1 | 需要加载 model.pth，有 hidden_dim 参数 | 加载权重 → 对战 |
| **规则型** | gen99, gen199, sample | 基于规则评分选择动作，无需模型文件 | 直接实例化 → 对战 |

## 4. 测试用例矩阵

| 测试类型 | 测试用例 | 预期结果 |
|----------|----------|----------|
| **模型加载（神经网络）** | 验证模型文件存在 | 文件存在返回 True |
| **模型加载（神经网络）** | 验证网络类可导入 | 成功导入 NeuralNetworkAgent |
| **模型加载（神经网络）** | 验证 hidden_dim 参数 | 找到正确的 hidden_dim |
| **模型加载（神经网络）** | 验证权重形状匹配 | 所有权重形状匹配 |
| **模型加载（规则型）** | 验证 ai.py 存在 | 文件存在返回 True |
| **模型加载（规则型）** | 验证类可导入 | 成功导入 AI 类 |
| **对战功能** | 模型 vs BasicTowerAI | 能够完成2场对战，无异常 |

## 5. 代码实现方案

### 5.1 目录结构

```
unit_tests/baseline_load_and_battle/
├── __init__.py
├── main.py                    # 测试入口
├── model_registry.py          # 模型注册表
├── model_loader.py            # 模型加载器（仅神经网络）
├── battle_simulator.py        # 对战模拟器（支持两种类型）
└── baseline_load_and_battle_design.md
```

### 5.2 模型配置

```python
# 神经网络型模型配置
neural_network_config = {
    'name': 'ann_593',                    # 模型名称
    'type': 'neural_network',             # 模型类型
    'path': 'baselines/ann_593',          # 模型目录路径
    'model_file': 'model.pth',            # 模型权重文件
    'input_dim': 10144,                   # 输入维度
    'action_dim': 32,                     # 动作维度
    'hidden_dim_candidates': [256]         # 可能的hidden_dim值
}

# 规则型 AI 配置
rule_based_config = {
    'name': 'gen99',                      # AI 名称
    'type': 'rule_based',                 # AI 类型
    'path': 'baselines/gen99',            # AI 目录路径
    'class_name': 'AI',                   # 类名
    'module_file': 'ai.py'               # 模块文件
}
```

### 5.3 核心代码说明

#### 5.3.1 model_registry.py

- `load_default_models()`: 遍历默认模型列表，根据 `type` 字段区分模型类型
- 神经网络型：检查 `model.pth` 文件是否存在
- 规则型：检查 `ai.py` 文件是否存在

#### 5.3.2 model_loader.py

- 仅用于加载神经网络型模型
- 自动检测正确的 `hidden_dim` 参数
- 验证模型权重形状是否匹配

#### 5.3.3 battle_simulator.py

- `load_rule_based_agent()`: 加载规则型 AI 实例
- `BattleSimulator.battle()`: 通用对战函数，通过 `agent_type` 参数区分处理方式
  - `'neural_network'`: 使用神经网络推理选择动作
  - `'rule_based'`: 调用 `agent.choose_bundle()` 选择动作

#### 5.3.4 main.py

- 初始化模型注册表
- 根据模型类型选择加载方式（神经网络用 ModelLoader，规则型用 load_rule_based_agent）
- 遍历所有已加载模型执行对战测试

## 6. 规则型 AI 策略说明

### gen99 / gen199
- 评估所有候选动作（bundles[1:]）
- 选择 `bundle.score` 最高且操作数量最少的动作
- 代码位置: `baselines/gen99/ai.py`

### sample
- 仅评估前 8 个候选动作（bundles[1:8]）
- 选择 `bundle.score` 最高且操作数量最少的动作
- 代码位置: `baselines/sample/ai.py`

## 7. 测试执行

```bash
cd unit_tests/baseline_load_and_battle
python main.py
```

## 8. 预期结果

- 所有存在的模型能够正确加载
- 每个模型与 BasicTowerAI 对战2场
- 对战过程无异常崩溃
- 能够正确统计胜负结果

---

**文档版本**: v2.0
**更新日期**: 2026年5月
**更新内容**: 添加规则型 AI 支持