# -*- coding: utf-8 -*-
"""
测试目的：验证 torch.distributed._functional_collectives.is_torchdynamo_compiling 接口功能正确性
API 名称：torch.distributed._functional_collectives.is_torchdynamo_compiling
API 签名：is_torchdynamo_compiling() -> bool

覆盖维度表：
| 覆盖维度         | 说明                                                         | 覆盖情况                                       |
|------------------|--------------------------------------------------------------|------------------------------------------------|
| 返回值类型       | 验证返回值为 bool 类型                                       | 已覆盖：test_return_type_bool                  |
| 编译状态检查     | 验证正常执行时返回 False；torch.compile 环境时返回 True     | 已覆盖：test_normal_execution_returns_false    |
| 无参数调用       | API 不接受参数的调用                                         | 已覆盖：test_no_arguments                      |

未覆盖项及原因：
- torch.compile 环境下的测试：需要在支持 torchdynamo 的环境中执行，单进程测试仅验证正常状态

注意：本测试仅验证功能正确性（调用不报错、返回值类型符合预期），
     不做精度和数值正确性校验。
"""

import unittest
import torch
import torch_npu  # noqa: F401 — registers NPU backend

try:
    from torch_npu.testing.testcase import TestCase, run_tests
except ImportError:
    import sys
    from unittest import TestCase

    def run_tests() -> None:
        unittest.main(argv=sys.argv)


class TestIsTorchdynamoCompiling(TestCase):
    """Test cases for torch.distributed._functional_collectives.is_torchdynamo_compiling."""

    def setUp(self):
        super().setUp()
        # Verify NPU backend is registered
        device_name = torch._C._get_privateuse1_backend_name()
        self.assertEqual(device_name, 'npu', f"Expected device 'npu', got '{device_name}'")

    def test_no_arguments(self):
        """is_torchdynamo_compiling should accept no arguments."""
        from torch.distributed._functional_collectives import is_torchdynamo_compiling
        # Should not raise
        result = is_torchdynamo_compiling()
        self.assertIsNotNone(result)

    def test_return_type_bool(self):
        """is_torchdynamo_compiling should return a boolean."""
        from torch.distributed._functional_collectives import is_torchdynamo_compiling
        result = is_torchdynamo_compiling()
        self.assertIsInstance(result, bool, f"Expected bool, got {type(result)}")

    def test_normal_execution_returns_false(self):
        """In normal execution (not under torch.compile), should return False."""
        from torch.distributed._functional_collectives import is_torchdynamo_compiling
        result = is_torchdynamo_compiling()
        self.assertFalse(result, "Expected False in normal execution context")

    def test_multiple_calls_consistent(self):
        """Multiple calls in same context should return same value."""
        from torch.distributed._functional_collectives import is_torchdynamo_compiling
        result1 = is_torchdynamo_compiling()
        result2 = is_torchdynamo_compiling()
        result3 = is_torchdynamo_compiling()
        self.assertEqual(result1, result2)
        self.assertEqual(result2, result3)


if __name__ == "__main__":
    run_tests()
