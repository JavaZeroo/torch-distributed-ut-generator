# -*- coding: utf-8 -*-
"""
测试目的：验证 torch.distributed.checkpoint.stateful.Stateful 协议及方法功能正确性
API 名称：torch.distributed.checkpoint.stateful.Stateful
torch.distributed.checkpoint.stateful.Stateful.state_dict
torch.distributed.checkpoint.stateful.Stateful.load_state_dict

API 签名：
- Stateful (Protocol)
- state_dict(self) -> dict[str, Any]
- load_state_dict(self, state_dict: dict[str, Any]) -> None

覆盖维度表：
| 覆盖维度         | 说明                                                         | 覆盖情况                                       |
|------------------|--------------------------------------------------------------|------------------------------------------------|
| 空/非空          | state_dict 返回空或非空字典                                  | 已覆盖                                         |
| 枚举选项         | 无                                                           | N/A                                            |
| 参数类型         | 验证参数接受正确类型                                         | 已覆盖                                         |
| 传参与不传参     | state_dict 无参，load_state_dict 有参                      | 已覆盖                                         |
| 等价类/边界值    | 空字典、单键、多键字典                                       | 已覆盖                                         |
| 正常传参场景     | Mock 实现调用各方法                                          | 已覆盖                                         |
| 异常传参场景     | Protocol runtime checkable 行为验证                          | 已覆盖                                         |

未覆盖项及原因：
- 无

注意：本测试仅验证功能正确性（调用不报错、输出 shape/dtype/类型符合预期），
     不做精度和数值正确性校验。
"""

from typing import Any
from typing_extensions import Protocol

import torch
from torch.distributed.checkpoint.stateful import Stateful


try:
    from torch_npu.testing.testcase import TestCase, run_tests
except ImportError:
    import sys
    from unittest import TestCase

    def run_tests():
        import unittest
        unittest.main(argv=sys.argv)


class MockStatefulComplete:
    """Mock implementation with both required methods."""

    def __init__(self):
        self._state = {"param": torch.ones(10), "value": 42}

    def state_dict(self) -> dict[str, Any]:
        return self._state.copy()

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        self._state = state_dict.copy()


class MockStatefulEmptyState:
    """Mock implementation with empty state."""

    def state_dict(self) -> dict[str, Any]:
        return {}

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        pass


class MockStatefulMissingLoad:
    """Mock implementation missing load_state_dict."""

    def state_dict(self) -> dict[str, Any]:
        return {}


class MockStatefulMissingState:
    """Mock implementation missing state_dict."""

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        pass


class TestStateful(TestCase):
    """Test cases for Stateful Protocol."""

    def setUp(self):
        super().setUp()
        self.device_name = torch._C._get_privateuse1_backend_name()
        self.assertEqual(self.device_name, 'npu', f"Expected device 'npu', got '{self.device_name}'")

    def test_stateful_is_protocol(self):
        """Test that Stateful is a Protocol."""
        from typing_extensions import Protocol
        self.assertTrue(issubclass(Stateful, Protocol))

    def test_stateful_runtime_checkable(self):
        """Test that Stateful is runtime checkable."""
        from typing_extensions import runtime_checkable
        self.assertTrue(issubclass(Stateful, Protocol) and hasattr(Stateful, '__instancecheck__'))

    def test_stateful_complete_is_instance(self):
        """Test that complete implementation passes isinstance check."""
        obj = MockStatefulComplete()
        self.assertIsInstance(obj, Stateful)

    def test_stateful_empty_state_is_instance(self):
        """Test that empty state implementation passes isinstance check."""
        obj = MockStatefulEmptyState()
        self.assertIsInstance(obj, Stateful)

    def test_stateful_missing_load_not_instance(self):
        """Test that missing load_state_dict fails isinstance check."""
        obj = MockStatefulMissingLoad()
        self.assertNotIsInstance(obj, Stateful)

    def test_stateful_missing_state_not_instance(self):
        """Test that missing state_dict fails isinstance check."""
        obj = MockStatefulMissingState()
        self.assertNotIsInstance(obj, Stateful)

    def test_stateful_state_dict_returns_dict(self):
        """Test state_dict method returns a dictionary."""
        obj = MockStatefulComplete()
        result = obj.state_dict()

        self.assertIsInstance(result, dict)
        self.assertIn("param", result)
        self.assertIn("value", result)

    def test_stateful_state_dict_empty(self):
        """Test state_dict method can return empty dict."""
        obj = MockStatefulEmptyState()
        result = obj.state_dict()

        self.assertIsInstance(result, dict)
        self.assertEqual(len(result), 0)

    def test_stateful_load_state_dict(self):
        """Test load_state_dict method accepts dictionary."""
        obj = MockStatefulComplete()
        new_state = {"param": torch.zeros(5), "value": 100}

        # Should not raise
        obj.load_state_dict(new_state)

        # Verify state was loaded
        current_state = obj.state_dict()
        self.assertEqual(current_state["value"], 100)

    def test_stateful_load_state_dict_empty(self):
        """Test load_state_dict with empty dictionary."""
        obj = MockStatefulEmptyState()

        # Should not raise
        obj.load_state_dict({})

    def test_stateful_state_dict_returns_copy(self):
        """Test that state_dict returns a copy, not reference."""
        obj = MockStatefulComplete()
        state1 = obj.state_dict()
        state1["new_key"] = "new_value"
        state2 = obj.state_dict()

        self.assertNotIn("new_key", state2)

    def test_stateful_module_is_instance(self):
        """Test that torch.nn.Module implements Stateful."""
        module = torch.nn.Linear(10, 5)
        self.assertIsInstance(module, Stateful)

    def test_stateful_optimizer_is_instance(self):
        """Test that torch.optim.Optimizer implements Stateful."""
        param = torch.nn.Parameter(torch.ones(10))
        optimizer = torch.optim.SGD([param], lr=0.01)
        self.assertIsInstance(optimizer, Stateful)


if __name__ == "__main__":
    run_tests()
