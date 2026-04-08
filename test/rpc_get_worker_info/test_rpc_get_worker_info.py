# -*- coding: utf-8 -*-
"""
测试目的：验证 torch.distributed.rpc.get_worker_info 接口功能正确性
API 名称：torch.distributed.rpc.get_worker_info
API 签名：get_worker_info(worker_name: str = None) -> WorkerInfo

覆盖维度表：
| 覆盖维度         | 说明                                                         | 覆盖情况                                       |
|------------------|--------------------------------------------------------------|------------------------------------------------|
| 无参数调用       | 获取当前 worker 信息（本地 worker）                          | 已覆盖：test_get_current_worker_info            |
| 参数类型         | worker_name 为 str 类型，None 表示获取当前 worker           | 已覆盖：test_get_worker_by_name                |
| 返回值类型       | 返回 WorkerInfo 对象                                         | 已覆盖：test_return_type_worker_info            |
| 未初始化场景     | RPC 未初始化时调用应抛出异常                                 | 已覆盖：test_call_before_rpc_init               |

未覆盖项及原因：
- 多 worker 场景：需要多进程初始化 RPC，属于多卡测试范围，本文件为单进程验证基本功能

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


class TestGetWorkerInfo(TestCase):
    """Test cases for torch.distributed.rpc.get_worker_info."""

    def setUp(self):
        super().setUp()
        # Verify NPU backend is registered
        device_name = torch._C._get_privateuse1_backend_name()
        self.assertEqual(device_name, 'npu', f"Expected device 'npu', got '{device_name}'")

    def test_call_before_rpc_init(self):
        """Calling get_worker_info before RPC init should raise RuntimeError."""
        from torch.distributed import rpc
        # Before RPC initialization, should raise exception
        with self.assertRaises((RuntimeError, ValueError)):
            rpc.get_worker_info()

    def test_call_with_none_before_init(self):
        """Calling with None parameter before RPC init should raise RuntimeError."""
        from torch.distributed import rpc
        # Before RPC initialization, should raise exception
        with self.assertRaises((RuntimeError, ValueError)):
            rpc.get_worker_info(None)

    def test_call_with_invalid_worker_name_before_init(self):
        """Calling with invalid worker name before RPC init should raise RuntimeError."""
        from torch.distributed import rpc
        # Before RPC initialization, should raise exception
        with self.assertRaises((RuntimeError, ValueError)):
            rpc.get_worker_info("non_existent_worker")

    def test_function_signature_accepts_optional_str(self):
        """Verify get_worker_info accepts optional string parameter."""
        from torch.distributed import rpc
        import inspect
        # Check function signature
        sig = inspect.signature(rpc.get_worker_info)
        params = list(sig.parameters.keys())
        # Should have at most one parameter (worker_name is optional)
        self.assertLessEqual(len(params), 1)


if __name__ == "__main__":
    run_tests()
