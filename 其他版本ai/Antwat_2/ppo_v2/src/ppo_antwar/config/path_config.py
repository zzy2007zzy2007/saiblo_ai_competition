from pathlib import Path

_DEFAULT_OUTPUT_DIR = "outputs"
_SUBDIR_TRAINING = "training"
_SUBDIR_CHECKPOINT = "checkpoint"
_SUBDIR_BATTLE = "battle"
_SUBDIR_SELFPLAY = "selfplay"
_SUBDIR_SYSTEM = "system"
_SUBDIR_EVALUATION = "evaluation"


class PathConfigError(Exception):
    """路径配置错误"""


def _find_project_root() -> Path:
    """向上搜索项目根目录（包含 pyproject.toml 或 .git 的目录）"""
    current = Path(__file__).resolve().parent
    for parent in [current] + list(current.parents):
        if (parent / "pyproject.toml").exists() or (parent / ".git").exists():
            return parent
    # 降级方案：使用硬编码层级
    return Path(__file__).resolve().parents[4]


class PathConfig:
    """路径管理器 - 统一管理所有路径，基于项目根目录自动解析。

    采用快速失败策略：初始化时一次性验证所有必需路径，
    任何路径不存在时立即抛出 PathConfigError 异常。
    """

    def __init__(self, base_output_dir: str = ""):
        self._project_root = _find_project_root()

        if base_output_dir:
            self._base_output_dir = Path(base_output_dir)
        else:
            self._base_output_dir = self._project_root / _DEFAULT_OUTPUT_DIR

        self._sdk_path = self._project_root / "Ant-Game" / "SDK"
        self._baselines_path = self._project_root / "baselines"

        self._validate_paths()

    @property
    def project_root(self) -> Path:
        return self._project_root

    @property
    def sdk_path(self) -> Path:
        return self._sdk_path

    @property
    def baselines_path(self) -> Path:
        return self._baselines_path

    def get_training_dir(self, run_id: str) -> Path:
        path = self._base_output_dir / run_id / _SUBDIR_TRAINING
        path.mkdir(parents=True, exist_ok=True)
        return path

    def get_checkpoint_dir(self, run_id: str) -> Path:
        path = self._base_output_dir / run_id / _SUBDIR_CHECKPOINT
        path.mkdir(parents=True, exist_ok=True)
        return path

    def get_battle_dir(self, run_id: str) -> Path:
        path = self._base_output_dir / run_id / _SUBDIR_BATTLE
        path.mkdir(parents=True, exist_ok=True)
        return path

    def get_selfplay_dir(self, run_id: str) -> Path:
        path = self._base_output_dir / run_id / _SUBDIR_SELFPLAY
        path.mkdir(parents=True, exist_ok=True)
        return path

    def get_system_dir(self, run_id: str) -> Path:
        path = self._base_output_dir / run_id / _SUBDIR_SYSTEM
        path.mkdir(parents=True, exist_ok=True)
        return path

    def get_evaluation_dir(self, run_id: str) -> Path:
        path = self._base_output_dir / run_id / _SUBDIR_EVALUATION
        path.mkdir(parents=True, exist_ok=True)
        return path

    def get_run_dir(self, run_id: str) -> Path:
        path = self._base_output_dir / run_id
        path.mkdir(parents=True, exist_ok=True)
        return path

    def _validate_paths(self):
        if not self._base_output_dir.exists():
            self._base_output_dir.mkdir(parents=True, exist_ok=True)
        missing = []
        if not self._sdk_path.exists():
            missing.append(f"SDK path: {self._sdk_path}")
        if not self._baselines_path.exists():
            missing.append(f"Baselines path: {self._baselines_path}")
        if missing:
            raise PathConfigError(
                "Required paths not found:\n  " + "\n  ".join(missing)
            )
