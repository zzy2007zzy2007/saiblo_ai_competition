"""配置管理模块"""
import json
import os

class ConfigManager:
    """配置管理器类"""
    
    @staticmethod
    def load_config(config_file=None):
        """加载配置文件
        
        Args:
            config_file: 配置文件路径，None则使用默认配置
            
        Returns:
            dict: 配置字典
        """
        # 默认配置
        default_config = {
            "episodes": 50,
            "max_rounds": 500,
            "strategies": ["sample", "gen99", "ann_v1"],
            "pairs": [],
            "verbose": False,
            "export": None,
            "report": None
        }
        
        # 如果指定了配置文件，加载配置文件
        if config_file and os.path.exists(config_file):
            with open(config_file, "r", encoding="utf-8") as f:
                config = json.load(f)
            # 合并默认配置和文件配置
            for key, value in default_config.items():
                if key not in config:
                    config[key] = value
            return config
        else:
            return default_config
    
    @staticmethod
    def save_config(config, config_file):
        """保存配置到文件
        
        Args:
            config: 配置字典
            config_file: 配置文件路径
        """
        with open(config_file, "w", encoding="utf-8") as f:
            json.dump(config, f, indent=2, ensure_ascii=False)
        print(f"配置已保存到: {config_file}")
    
    @staticmethod
    def get_default_config():
        """获取默认配置
        
        Returns:
            dict: 默认配置
        """
        return {
            "episodes": 50,
            "max_rounds": 500,
            "strategies": ["sample", "gen99", "ann_v1"],
            "pairs": [],
            "verbose": False,
            "export": None,
            "report": None
        }
