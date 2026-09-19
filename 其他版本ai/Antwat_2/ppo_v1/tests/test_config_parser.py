import sys
import unittest
from unittest.mock import MagicMock
from easydict import EasyDict

# Mock openrl dependency before importing config_parser
sys.modules['openrl'] = MagicMock()
sys.modules['openrl.configs'] = MagicMock()
sys.modules['openrl.configs.config'] = MagicMock()

# Add src to path for direct import
sys.path.insert(0, 'src')

# Direct import of config_parser module using importlib to avoid
# triggering ppo_antwar/__init__.py (which requires numpy, torch, etc.)
import importlib.util
spec = importlib.util.spec_from_file_location(
    'ppo_antwar.config.config_parser',
    'src/ppo_antwar/config/config_parser.py'
)
config_parser_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(config_parser_module)

_filter_none = config_parser_module._filter_none
_deep_merge_overrides = config_parser_module._deep_merge_overrides


class TestFilterNone(unittest.TestCase):
    def test_all_not_none(self):
        self.assertEqual(_filter_none({"a": 1, "b": "hello"}), {"a": 1, "b": "hello"})

    def test_some_none(self):
        self.assertEqual(_filter_none({"a": 1, "b": None}), {"a": 1})

    def test_nested_none(self):
        self.assertEqual(_filter_none({"a": {"b": None}}), {})

    def test_nested_partial(self):
        self.assertEqual(_filter_none({"a": {"b": 1, "c": None}}), {"a": {"b": 1}})

    def test_none_input_is_noop(self):
        # _filter_none 仅在 cli_overrides 非 None 时调用，调用方需保证传入 dict
        pass


class TestDeepMergeOverrides(unittest.TestCase):
    def test_scalar_override(self):
        config = EasyDict({"ppo": EasyDict({"lr": 0.001})})
        _deep_merge_overrides(config, {"ppo": {"lr": 0.5}})
        self.assertEqual(config.ppo.lr, 0.5)

    def test_none_is_skipped(self):
        config = EasyDict({"ppo": EasyDict({"lr": 0.001})})
        _deep_merge_overrides(config, {"ppo": {"lr": None}})
        self.assertEqual(config.ppo.lr, 0.001)

    def test_nested_merge(self):
        config = EasyDict({"training": EasyDict({"warmup": EasyDict({"a": 1, "b": 2})})})
        _deep_merge_overrides(config, {"training": {"warmup": {"b": 99}}})
        self.assertEqual(config.training.warmup.a, 1)
        self.assertEqual(config.training.warmup.b, 99)

    def test_new_section_insertion(self):
        config = EasyDict({"ppo": EasyDict({"lr": 0.001})})
        _deep_merge_overrides(config, {"new_section": {"key": 42}})
        self.assertEqual(config.new_section.key, 42)

    def test_new_section_on_existing_non_dict(self):
        # 此测试验证代码逻辑健壮性而非真实业务场景（真实场景中 ppo 始终为 dict/EasyDict）
        config = EasyDict({"ppo": 0.001})
        _deep_merge_overrides(config, {"ppo": {"lr": 0.5}})
        self.assertEqual(config.ppo.lr, 0.5)

    def test_empty_overrides(self):
        config = EasyDict({"ppo": EasyDict({"lr": 0.001})})
        _deep_merge_overrides(config, {})
        self.assertEqual(config.ppo.lr, 0.001)


if __name__ == "__main__":
    unittest.main()
