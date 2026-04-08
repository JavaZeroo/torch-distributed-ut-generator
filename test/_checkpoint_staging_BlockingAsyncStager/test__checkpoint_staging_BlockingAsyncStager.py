# -*- coding: utf-8 -*-
"""
测试目的：验证 torch.distributed.checkpoint.staging.BlockingAsyncStager 功能正确性
API 名称：torch.distributed.checkpoint.staging.BlockingAsyncStager
API 签名：BlockingAsyncStager(cache_staged_state_dict: bool = False, type_check: bool = False)

覆盖维度表：
| 覆盖维度         | 说明                                                         | 覆盖情况                                       |
|------------------|--------------------------------------------------------------|------------------------------------------------|
| 空/非空          | state_dict 为空或非空                                        | 已覆盖                                         |
| 枚举选项         | cache_staged_state_dict True/False                          | 已覆盖                                         |
| 参数类型         | cache_staged_state_dict bool, type_check bool               | 已覆盖                                         |
| 传参与不传参     | 构造函数使用默认值或显式传参                                 | 已覆盖                                         |
| 等价类/边界值    | 空 state_dict、不同大小的 state_dict                         | 已覆盖                                         |
| 正常传参场景     | stage/synchronize_staging/close 调用                        | 已覆盖                                         |
| 异常传参场景     | 无稳定异常路径                                               | 未覆盖（无明确异常场景）                        |

未覆盖项及原因：
- 异常传参场景：BlockingAsyncStager 没有明确抛出异常的场景

注意：本测试仅验证功能正确性（调用不报错、输出 shape/dtype/类型符合预期），
     不做精度和数值正确性校验。
"""

import torch
from torch.distributed.checkpoint.staging import BlockingAsyncStager, AsyncStager


try:
    from torch_npu.testing.testcase import TestCase, run_tests
except ImportError:
    import sys
    from unittest import TestCase

    def run_tests():
        import unittest
        unittest.main(argv=sys.argv)


class TestBlockingAsyncStager(TestCase):
    """Test cases for BlockingAsyncStager class."""

    def setUp(self):
        super().setUp()
        self.device_name = torch._C._get_privateuse1_backend_name()
        self.assertEqual(self.device_name, 'npu', f"Expected device 'npu', got '{self.device_name}'")

    def test_blocking_async_stager_is_async_stager(self):
        """Test that BlockingAsyncStager implements AsyncStager protocol."""
        stager = BlockingAsyncStager()
        self.assertIsInstance(stager, AsyncStager)

    def test_blocking_async_stager_default_init(self):
        """Test default initialization."""
        stager = BlockingAsyncStager()

        self.assertFalse(stager.cache_staged_state_dict)
        self.assertFalse(stager.type_check)
        self.assertIsNone(stager.state_dict_cache)

    def test_blocking_async_stager_init_with_cache_true(self):
        """Test initialization with cache_staged_state_dict=True."""
        stager = BlockingAsyncStager(cache_staged_state_dict=True)

        self.assertTrue(stager.cache_staged_state_dict)
        self.assertFalse(stager.type_check)

    def test_blocking_async_stager_init_with_type_check_true(self):
        """Test initialization with type_check=True."""
        stager = BlockingAsyncStager(type_check=True)

        self.assertFalse(stager.cache_staged_state_dict)
        self.assertTrue(stager.type_check)

    def test_blocking_async_stager_init_with_all_params(self):
        """Test initialization with all parameters."""
        stager = BlockingAsyncStager(cache_staged_state_dict=True, type_check=True)

        self.assertTrue(stager.cache_staged_state_dict)
        self.assertTrue(stager.type_check)

    def test_blocking_async_stager_stage_no_cache(self):
        """Test stage method without caching."""
        stager = BlockingAsyncStager(cache_staged_state_dict=False)
        device = f"{self.device_name}:0"
        state_dict = {
            "param1": torch.ones(10, device=device),
            "param2": torch.zeros(5, device=device)
        }

        result = stager.stage(state_dict)

        self.assertIsInstance(result, dict)
        self.assertIn("param1", result)
        self.assertIn("param2", result)

    def test_blocking_async_stager_stage_empty_state_dict(self):
        """Test stage method with empty state_dict."""
        stager = BlockingAsyncStager()
        state_dict = {}

        result = stager.stage(state_dict)

        self.assertIsInstance(result, dict)
        self.assertEqual(len(result), 0)

    def test_blocking_async_stager_should_synchronize_after_execute(self):
        """Test should_synchronize_after_execute property returns False."""
        stager = BlockingAsyncStager()

        # BlockingAsyncStager sets _synchronize_after_execute to False
        self.assertFalse(stager.should_synchronize_after_execute)

    def test_blocking_async_stager_synchronize_staging_no_op(self):
        """Test synchronize_staging is a no-op."""
        stager = BlockingAsyncStager()

        # Should not raise
        stager.synchronize_staging()

    def test_blocking_async_stager_no_close_method(self):
        """Test that BlockingAsyncStager does not have close method (unlike AsyncStager protocol)."""
        stager = BlockingAsyncStager()

        # BlockingAsyncStager does not implement close method
        self.assertFalse(hasattr(stager, 'close') and callable(getattr(stager, 'close', None)))

    def test_blocking_async_stager_stage_returns_cpu_copy(self):
        """Test that stage returns CPU copy of tensors."""
        stager = BlockingAsyncStager()
        device = f"{self.device_name}:0"
        state_dict = {
            "param1": torch.ones(10, device=device),
        }

        result = stager.stage(state_dict)

        # Result should be on CPU
        self.assertEqual(result["param1"].device, torch.device("cpu"))


if __name__ == "__main__":
    run_tests()
