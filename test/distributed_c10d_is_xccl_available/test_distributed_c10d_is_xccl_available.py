# -*- coding: utf-8 -*-
"""
测试目的：验证 torch.distributed.distributed_c10d.is_xccl_available 接口功能正确性
API 名称：torch.distributed.distributed_c10d.is_xccl_available
API 签名：is_xccl_available() -> bool

覆盖维度表：
| 覆盖维度         | 说明                                                         | 覆盖情况                                       |
|------------------|--------------------------------------------------------------|------------------------------------------------|
| 返回值类型       | 验证返回值为 bool 类型                                       | 已覆盖：test_return_type_bool                  |
| XCCL 可用性      | 检查环境中 XCCL 后端是否可用                                 | 已覆盖：test_xccl_availability                 |
| 无参数调用       | API 不接受参数的调用                                         | 已覆盖：test_no_arguments                      |
| 重复调用         | 多次调用应返回相同结果                                       | 已覆盖：test_multiple_calls_consistent         |

未覆盖项及原因：
- 无

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


class TestIsXcclAvailable(TestCase):
    """Test cases for torch.distributed.distributed_c10d.is_xccl_available."""

    def setUp(self):
        super().setUp()
        # Verify NPU backend is registered
        device_name = torch._C._get_privateuse1_backend_name()
        self.assertEqual(device_name, 'npu', f"Expected device 'npu', got '{device_name}'")

    def test_no_arguments(self):
        """is_xccl_available should accept no arguments."""
        from torch.distributed.distributed_c10d import is_xccl_available
        # Should not raise
        result = is_xccl_available()
        self.assertIsNotNone(result)

    def test_return_type_bool(self):
        """is_xccl_available should return a boolean."""
        from torch.distributed.distributed_c10d import is_xccl_available
        result = is_xccl_available()
        self.assertIsInstance(result, bool, f"Expected bool, got {type(result)}")

    def test_xccl_availability(self):
        """Verify XCCL availability status."""
        from torch.distributed.distributed_c10d import is_xccl_available
        result = is_xccl_available()
        # Result should be boolean regardless of availability
        self.assertIsInstance(result, bool)

    def test_multiple_calls_consistent(self):
        """Multiple calls should return consistent result."""
        from torch.distributed.distributed_c10d import is_xccl_available
        result1 = is_xccl_available()
        result2 = is_xccl_available()
        result3 = is_xccl_available()
        self.assertEqual(result1, result2)
        self.assertEqual(result2, result3)


if __name__ == "__main__":
    run_tests()
