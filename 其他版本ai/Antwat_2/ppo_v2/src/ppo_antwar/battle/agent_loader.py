from pathlib import Path
from typing import Any, Dict, List

from loguru import logger

from .rule_based_agents import BasicRandomAI, BasicTowerAI, MediumRuleAI


KNOWN_BUILTIN_AGENTS: Dict[str, type] = {
    "BasicRandomAI": BasicRandomAI,
    "BasicTowerAI": BasicTowerAI,
    "MediumRuleAI": MediumRuleAI,
}


class AgentLoader:
    """Agent加载器 - 管理内置和外部Agent"""

    _baselines_path: str = "baselines"

    # ── 内置 Agent ─────────────────────────────────────────────────────────

    @staticmethod
    def load_builtin(name: str, player_id: int = 0) -> Any:
        agent_class = KNOWN_BUILTIN_AGENTS.get(name)
        if agent_class is None:
            raise ValueError(
                f"Unknown built-in agent: {name}. Known agents: {list(KNOWN_BUILTIN_AGENTS.keys())}"
            )
        return agent_class(player_id=player_id)

    # ── 外部 Agent 管理 ────────────────────────────────────────────────────

    @staticmethod
    def set_baselines_path(path: str) -> None:
        AgentLoader._baselines_path = path

    @staticmethod
    def scan_external() -> List[str]:
        base = Path(AgentLoader._baselines_path)
        if not base.exists():
            return []
        agents = []
        for entry in sorted(base.iterdir()):
            if entry.is_dir() and (entry / "ai.py").exists():
                agents.append(entry.name)
        return agents

    @staticmethod
    def load_external(name: str, player_id: int = 0) -> Any:
        import importlib.util
        import os
        import sys

        agent_dir = Path(AgentLoader._baselines_path) / name
        module_path = agent_dir / "ai.py"
        if not module_path.exists():
            logger.error(f"External agent '{name}' not found at {module_path}")
            raise ValueError(f"External agent '{name}' not found")

        # ── 1. 清模块缓存（对齐 ppo_v1，防止不同 agent 的 common 模块交叉污染） ──
        backup_modules: Dict[str, Any] = {}
        for key in list(sys.modules.keys()):
            if key in ("ai", "common", "SDK", "AI") or key.startswith(("ai.", "common.", "SDK.", "AI.")):
                backup_modules[key] = sys.modules.pop(key)

        # ── 2. 保存原始环境 ──
        original_cwd = os.getcwd()
        original_sys_path = sys.path.copy()
        original_modules = set(sys.modules.keys())

        # ── 3. 设置 agent 目录 + SDK 路径 ──
        sdk_parent = str(Path(AgentLoader._baselines_path).parent / "Ant-Game")
        sys.path.insert(0, str(agent_dir))
        sys.path.insert(0, sdk_parent)
        os.chdir(str(agent_dir))

        try:
            spec = importlib.util.spec_from_file_location(f"external_agent_{name}", str(module_path))
            if spec is None or spec.loader is None:
                raise ImportError(f"Cannot load module from {module_path}")

            module = importlib.util.module_from_spec(spec)
            sys.modules[spec.name] = (
                module  # 注册到 sys.modules，dataclass(slots=True) 需要
            )
            spec.loader.exec_module(module)

            if hasattr(module, "create_agent"):
                try:
                    agent = module.create_agent(player_id=player_id)
                except TypeError:
                    agent = module.create_agent()
            elif hasattr(module, "AI"):
                try:
                    agent = module.AI(player_id=player_id)
                except TypeError:
                    agent = module.AI()
            else:
                raise ValueError(f"External agent '{name}' has neither create_agent() nor AI class")
        except Exception as e:
            logger.error(f"Failed to load external agent '{name}': {e}")
            raise
        finally:
            # ── 4. 恢复环境 ──
            os.chdir(original_cwd)
            sys.path = original_sys_path

            # 清除本次加载新增的模块
            for key in set(sys.modules.keys()) - original_modules:
                if key in sys.modules:
                    del sys.modules[key]

            # 恢复之前移除的原始模块
            for key, mod in backup_modules.items():
                sys.modules[key] = mod

        return agent

    # ── 统一加载 ────────────────────────────────────────────────────────────

    @staticmethod
    def load(name: str, player_id: int = 0) -> Any:
        """统一加载接口：内置优先，外部回退"""
        if name in KNOWN_BUILTIN_AGENTS:
            return AgentLoader.load_builtin(name, player_id)
        return AgentLoader.load_external(name, player_id)

    # ── 验证 ────────────────────────────────────────────────────────────────

    @staticmethod
    def validate_baselines(names: List[str]) -> None:
        external_agents = set(AgentLoader.scan_external())
        unknown = []
        for name in names:
            if name in KNOWN_BUILTIN_AGENTS:
                continue
            if name in external_agents:
                continue
            unknown.append(name)
        if unknown:
            raise ValueError(
                f"Unknown baseline agents: {unknown}. "
                f"Known builtin: {list(KNOWN_BUILTIN_AGENTS.keys())}, "
                f"known external: {sorted(external_agents)}"
            )

    # ── 序列化 ──────────────────────────────────────────────────────────────

    @staticmethod
    def serialize(agent: Any) -> bytes:
        import cloudpickle

        return cloudpickle.dumps(agent)

    @staticmethod
    def deserialize(data: bytes) -> Any:
        import cloudpickle

        return cloudpickle.loads(data)
