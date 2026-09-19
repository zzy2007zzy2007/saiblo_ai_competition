from __future__ import annotations
import os
from pathlib import Path
from typing import Optional
from loguru import logger


class PathConfigError(Exception):
    """路径配置异常"""
    pass


class PathConfig:
    """统一路径配置管理类 - 使用固定项目相对路径"""
    
    def __init__(self, base_dir: Optional[str] = None):
        # 1. 先定义项目根目录（所有路径的基准点）
        self.project_root = Path(__file__).resolve().parents[4]
        
        # 2. 再定义基础输出目录
        if base_dir:
            self.base_dir = Path(base_dir).resolve()
        else:
            # 固定输出目录：项目根目录 / outputs
            self.base_dir = self.project_root / "outputs"
        
        # 子目录（所有路径都基于 project_root 或 base_dir）
        self.training_dir = self.base_dir / "training"
        self.checkpoint_dir = self.base_dir / "checkpoint"
        self.battle_dir = self.base_dir / "battle"
        self.selfplay_dir = self.base_dir / "selfplay"
        self.system_dir = self.base_dir / "system"
        
        # SDK 和 Baseline 路径（基于 project_root）
        self._init_sdk_and_baselines_paths()
        
        # 确保目录存在
        self._ensure_directories()
    
    def _init_sdk_and_baselines_paths(self):
        """初始化 SDK 和 Baseline 路径（一次性验证，失败立即抛出异常）"""
        # 使用已定义的 project_root，避免重复计算
        
        # 1. SDK 路径：项目根目录 / Ant-Game / SDK
        self.sdk_dir = self._validate_path(
            self.project_root / "Ant-Game" / "SDK",
            "SDK目录"
        )
        
        # 2. Baseline 路径：项目根目录 / baselines
        self.baselines_dir = self._validate_path(
            self.project_root / "baselines",
            "Baseline目录"
        )
        
        logger.info(f"✓ SDK路径: {self.sdk_dir}")
        logger.info(f"✓ Baseline路径: {self.baselines_dir}")
    
    def _validate_path(self, path: Path, description: str) -> Path:
        """验证路径是否存在，不存在立即抛出异常"""
        resolved_path = path.resolve()
        if not resolved_path.exists():
            raise PathConfigError(
                f"{description}不存在：{resolved_path}"
            )
        return resolved_path
    
    def _ensure_directories(self):
        """确保所有输出目录存在"""
        for dir_path in [
            self.base_dir,
            self.training_dir,
            self.checkpoint_dir,
            self.battle_dir,
            self.selfplay_dir,
            self.system_dir,
        ]:
            dir_path.mkdir(parents=True, exist_ok=True)
    
    def __repr__(self):
        return f"PathConfig(base_dir={self.base_dir})"