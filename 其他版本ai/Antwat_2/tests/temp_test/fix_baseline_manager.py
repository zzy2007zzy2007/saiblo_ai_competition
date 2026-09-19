#!/usr/bin/env python3
"""
修复 baseline_battle_manager.py 的加载逻辑
使用 baseline_agent 中的 ActorCriticNetwork
"""

import re

# 读取原始文件
with open('ppo/src/ppo_antwar/league/baseline_battle_manager.py', 'r') as f:
    content = f.read()

# 定义新的 load_baseline_agents 方法
new_method = '''    def load_baseline_agents(self) -> bool:
        self.baseline_agents = []
        loaded_count = 0

        logger.info("\\n=== 开始加载baseline智能体 ===")

        self.baseline_agents.append({"name": "BasicTowerAI", "agent": BasicTowerAI()})
        logger.info("✓ 成功加载 BasicTowerAI")
        loaded_count += 1

        ann_v1_dir = os.path.join(BASELINES_PATH, 'ann_v1')
        if os.path.exists(ann_v1_dir):
            try:
                ann_v1_agent = NeuralNetworkAgent(
                    board_dim=10144,
                    action_dim=32,
                    hidden_dim=256,
                    use_actor_critic=True
                )
                model_path = os.path.join(ann_v1_dir, 'model.pth')
                if os.path.exists(model_path):
                    if ann_v1_agent.load_model(model_path):
                        logger.info("✓ 成功加载 AnnV1 (ActorCritic, hidden_dim=256)")
                    else:
                        logger.warning("⚠ AnnV1 模型加载失败，使用新模型")
                else:
                    logger.warning(f"⚠ AnnV1 模型文件不存在: {model_path}")

                self.baseline_agents.append({"name": "AnnV1", "agent": ann_v1_agent})
                loaded_count += 1
            except Exception as e:
                logger.error(f"✗ 加载 AnnV1 失败: {e}")
                import traceback
                logger.error(f"详细错误: {traceback.format_exc()}")
        else:
            logger.warning("⚠ ann_v1 目录不存在")

        ann_593_dir = os.path.join(BASELINES_PATH, 'ann_593')
        if os.path.exists(ann_593_dir):
            try:
                ann_593_agent = NeuralNetworkAgent(
                    board_dim=10144,
                    action_dim=32,
                    hidden_dim=256,
                    use_actor_critic=True
                )
                model_path = os.path.join(ann_593_dir, 'model.pth')
                if os.path.exists(model_path):
                    if ann_593_agent.load_model(model_path):
                        logger.info("✓ 成功加载 Ann593 (ActorCritic, hidden_dim=256)")
                    else:
                        logger.warning("⚠ Ann593 模型加载失败，使用新模型")

                self.baseline_agents.append({"name": "Ann593", "agent": ann_593_agent})
                loaded_count += 1
            except Exception as e:
                logger.error(f"✗ 加载 Ann593 失败: {e}")
                import traceback
                logger.error(f"详细错误: {traceback.format_exc()}")
        else:
            logger.warning("⚠ ann_593 目录不存在")

        logger.info(f"=== Baseline智能体加载完成，成功加载 {loaded_count} 个模型 ===")
        return loaded_count > 0'''

# 使用正则表达式找到并替换 load_baseline_agents 方法
# 匹配从 "def load_baseline_agents" 到下一个方法定义之前的内容
pattern = r'    def load_baseline_agents\(self\) -> bool:.*?(?=\n    def |\nclass |\Z)'
content = re.sub(pattern, new_method, content, flags=re.DOTALL)

# 写回文件
with open('ppo/src/ppo_antwar/league/baseline_battle_manager.py', 'w') as f:
    f.write(content)

print("✓ baseline_battle_manager.py 已修复")
print("  - 使用 baseline_agent 中的 NeuralNetworkAgent")
print("  - 使用 ActorCritic 网络结构")
print("  - 正确的模型加载逻辑")
