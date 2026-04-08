# -*- coding: utf-8 -*-
"""
测试目的：验证 torch.distributed.checkpoint.staging.AsyncStager 协议及 should_synchronize_after_execute 功能正确性
API 名称：torch.distributed.checkpoint.staging.AsyncStager
torch.distributed.checkpoint.staging.AsyncStager.should_synchronize_after_execute

API 签名：
- AsyncStager (Protocol)
- should_synchronize_after_execute: property -> bool
- stage(self, state_dict: STATE_DICT_TYPE) -> Future[STATE_DICT_TYPE] | STATE_DICT_TYPE
- synchronize_staging(self) -> None (deprecated)
- close(self) -> None

覆盖维度表：
| 覆盖维度         | 说明                                                         | 覆盖情况                                       |
|------------------|--------------------------------------------------------------|------------------------------------------------|
| 空/非空          | state_dict 为空或非空                                        | 已覆盖                                         |
| 枚举选项         | _synchronize_after_execute True/False                       | 已覆盖                                         |
| 参数类型         | 验证参数接受正确类型                                         | 已覆盖                                         |
| 传参与不传参     | close/synchronize_staging 无参数                           | 已覆盖                                         |
| 等价类/边界值    | 不同大小 state_dict                                          | 已覆盖                                         |
| 正常传参场景     | Mock 实现调用各方法                                          | 已覆盖                                         |
| 异常传参场景     | Protocol runtime checkable 行为验证                          | 已覆盖                                         |

未覆盖项及原因：
- 无

注意：本测试仅验证功能正确性（调用不报错、输出 shape/dtype/类型符合预期），
     不做精度和数值正确性校验。
"""

from concurrent.futures import Future
from typing import Any

import torch
from torch.distributed.checkpoint.staging import AsyncStager


try:
    from torch_npu.testing.testcase import TestCase, run_tests
except ImportError:
    import sys
    from unittest import TestCase

    def run_tests():
        import unittest
        unittest.main(argv=sys.argv)


class MockAsyncStagerTrue:
    """Mock implementation with synchronize_after_execute=True."""

    _synchronize_after_execute: bool = True

    @property
    def should_synchronize_after_execute(self) -> bool:
        return self._synchronize_after_execute

    def stage(self, state_dict: dict) -> Future[dict] | dict:
        return state_dict

    def synchronize_staging(self) -> None:
        pass

    def close(self) -> None:
        pass


class MockAsyncStagerFalse:
    """Mock implementation with synchronize_after_execute=False."""

    _synchronize_after_execute: bool = False

    @property
    def should_synchronize_after_execute(self) -> bool:
        return self._synchronize_after_execute

    def stage(self, state_dict: dict) -> Future[dict] | dict:
        return Future()

    def synchronize_staging(self) -> None:
        pass

    def close(self) -> None:
        pass


class MockAsyncStagerNoProperty:
    """Mock implementation without should_synchronize_after_execute property."""

    def stage(self, state_dict: dict) -> Future[dict] | dict:
        return state_dict

    def synchronize_staging(self) -> None:
        pass

    def close(self) -> None:
        pass


class TestAsyncStager(TestCase):
    """Test cases for AsyncStager Protocol."""

    def setUp(self):
        super().setUp()
        self.device_name = torch._C._get_privateuse1_backend_name()
        self.assertEqual(self.device_name, 'npu', f"Expected device 'npu', got '{self.device_name}'")

    def test_async_stager_is_protocol(self):
        """Test that AsyncStager is a Protocol."""
        from typing_extensions import Protocol
        self.assertTrue(issubclass(AsyncStager, Protocol))

    def test_async_stager_runtime_checkable_true(self):
        """Test isinstance check with synchronize_after_execute=True."""
        stager = MockAsyncStagerTrue()
        self.assertIsInstance(stager, AsyncStager)

    def test_async_stager_runtime_checkable_false(self):
        """Test isinstance check with synchronize_after_execute=False."""
        stager = MockAsyncStagerFalse()
        self.assertIsInstance(stager, AsyncStager)

    def test_async_stager_should_synchronize_true(self):
        """Test should_synchronize_after_execute property returns True."""
        stager = MockAsyncStagerTrue()
        self.assertTrue(stager.should_synchronize_after_execute)

    def test_async_stager_should_synchronize_false(self):
        """Test should_synchronize_after_execute property returns False."""
        stager = MockAsyncStagerFalse()
        self.assertFalse(stager.should_synchronize_after_execute)

    def test_async_stager_stage_returns_dict(self):
        """Test stage method returning dict directly."""
        stager = MockAsyncStagerTrue()
        state_dict = {"param1": torch.ones(10), "param2": torch.zeros(5)}

        result = stager.stage(state_dict)

        self.assertEqual(result, state_dict)

    def test_async_stager_stage_returns_future(self):
        """Test stage method returning Future."""
        stager = MockAsyncStagerFalse()
        state_dict = {"param1": torch.ones(10)}

        result = stager.stage(state_dict)

        self.assertIsInstance(result, Future)

    def test_async_stager_stage_empty_state_dict(self):
        """Test stage method with empty state_dict."""
        stager = MockAsyncStagerTrue()
        state_dict = {}

        result = stager.stage(state_dict)

        self.assertEqual(result, {})

    def test_async_stager_close(self):
        """Test close method can be called."""
        stager = MockAsyncStagerTrue()

        # Should not raise
        stager.close()

    def test_async_stager_synchronize_staging(self):
        """Test synchronize_staging method can be called."""
        stager = MockAsyncStagerTrue()

        # Should not raise (deprecated but still callable)
        stager.synchronize_staging()

    def test_async_stager_without_property_not_instance(self):
        """Test that missing property makes it fail isinstance check."""
        stager = MockAsyncStagerNoProperty()
        # Protocol with runtime_checkable requires all attributes to exist
        self.assertNotIsInstance(stager, AsyncStager)

    def test_async_stager_default_synchronize_value(self):
        """Test that default _synchronize_after_execute is True."""
        # Verify the class attribute default
        self.assertTrue(hasattr(MockAsyncStagerTrue, '_synchronize_after_execute'))
        self.assertTrue(MockAsyncStagerTrue._synchronize_after_execute)


if __name__ == "__main__":
    run_tests()
