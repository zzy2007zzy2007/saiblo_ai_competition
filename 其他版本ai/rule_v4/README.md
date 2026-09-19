# 规则AI提交包

## 目录结构
```
rule_based_ai_submit/
├── main.py              # 统一入口文件
├── ai.py                # 规则AI核心代码
├── common.py            # 基础Agent类
├── protocol.py          # 通信协议
└── SDK/                 # 运行时依赖
    ├── backend/
    │   ├── __init__.py
    │   ├── core.py
    │   ├── engine.py
    │   ├── forecast.py
    │   ├── model.py
    │   ├── runtime.py
    │   └── state.py
    └── utils/
        ├── __init__.py
        ├── actions.py
        ├── actions_modifiable.py
        ├── constants.py
        ├── features.py
        ├── geometry.py
        └── turns.py
```

## 规则AI策略

### 核心策略
1. **闪电风暴优先**：如果闪电风暴可用，优先使用
2. **经济发展**：建造Basic塔并升级到生产塔
3. **动态调整**：根据闪电冷却和金币情况调整建塔权重

### 策略细节
- `k = 0.01`：指数衰减参数，控制建塔权重
- 闪电冷却35回合，费用90金币
- 优先升级生产塔（PRODUCER, PRODUCER_FAST等）

## 使用方法

### 本地测试
1. 将此目录复制到Ant-Game的AI目录
2. 使用本地对战工具测试

### 提交
1. 将整个 `rule_based_ai_submit` 目录内容打包为zip
2. 确保zip根目录包含main.py, ai.py等文件
3. 提交到比赛平台

## 性能
- 对战ExampleAgent胜率约80%
- 闪电风暴使用频率适中
- 经济发展良好

## 可调参数
- `ai.py` 中的 `k` 值：控制建塔权重衰减速度
- `LIGHTNING_COST = 90`：闪电风暴费用
