# -*- coding: utf-8 -*-
"""
测试目的：验证 torch.distributed.checkpoint.state_dict_saver.AsyncCheckpointerType 枚举功能正确性
API 名称：torch.distributed.checkpoint.state_dict_saver.AsyncCheckpointerType
API 签名：AsyncCheckpointerType(Enum): THREAD = "thread", PROCESS = "process"

覆盖维度表：
| 覆盖维度         | 说明                                                         | 覆盖情况                                       |
|------------------|--------------------------------------------------------------|------------------------------------------------|
| 空/非空          | 枚举值必有值                                                 | 已覆盖                                         |
| 枚举选项         | THREAD, PROCESS 两个选项                                     | 已覆盖                                         |
| 参数类型         | 枚举构造无额外参数                                           | 已覆盖                                         |
| 传参与不传参     | 使用枚举成员直接访问                                         | 已覆盖                                         |
| 等价类/边界值    | 所有枚举成员                                                 | 已覆盖                                         |
| 正常传参场景     | 访问枚举成员、比较、转换字符串                               | 已覆盖                                         |
| 异常传参场景     | 无效枚举值访问                                               | 已覆盖（通过 assertRaises）                    |

未覆盖项及原因：
- 无

注意：本测试仅验证功能正确性（调用不报错、输出 shape/dtype/类型符合预期），
     不做精度和数值正确性校验。
"""

import torch
from torch.distributed.checkpoint.state_dict_saver import AsyncCheckpointerType


try:
    from torch_npu.testing.testcase import TestCase, run_tests
except ImportError:
    import sys
    from unittest import TestCase

    def run_tests():
        import unittest
        unittest.main(argv=sys.argv)


class TestAsyncCheckpointerType(TestCase):
    """Test cases for AsyncCheckpointerType enum."""

    def setUp(self):
        super().setUp()
        self.device_name = torch._C._get_privateuse1_backend_name()
        self.assertEqual(self.device_name, 'npu', f"Expected device 'npu', got '{self.device_name}'")

    def test_async_checkpointer_type_is_enum(self):
        """Test that AsyncCheckpointerType is an Enum."""
        from enum import Enum
        self.assertTrue(issubclass(AsyncCheckpointerType, Enum))

    def test_async_checkpointer_type_thread_value(self):
        """Test THREAD enum member value."""
        self.assertEqual(AsyncCheckpointerType.THREAD.value, "thread")

    def test_async_checkpointer_type_process_value(self):
        """Test PROCESS enum member value."""
        self.assertEqual(AsyncCheckpointerType.PROCESS.value, "process")

    def test_async_checkpointer_type_thread_name(self):
        """Test THREAD enum member name."""
        self.assertEqual(AsyncCheckpointerType.THREAD.name, "THREAD")

    def test_async_checkpointer_type_process_name(self):
        """Test PROCESS enum member name."""
        self.assertEqual(AsyncCheckpointerType.PROCESS.name, "PROCESS")

    def test_async_checkpointer_type_equality(self):
        """Test enum member equality."""
        self.assertEqual(AsyncCheckpointerType.THREAD, AsyncCheckpointerType.THREAD)
        self.assertEqual(AsyncCheckpointerType.PROCESS, AsyncCheckpointerType.PROCESS)
        self.assertNotEqual(AsyncCheckpointerType.THREAD, AsyncCheckpointerType.PROCESS)

    def test_async_checkpointer_type_identity(self):
        """Test enum member identity."""
        self.assertIs(AsyncCheckpointerType.THREAD, AsyncCheckpointerType.THREAD)
        self.assertIs(AsyncCheckpointerType.PROCESS, AsyncCheckpointerType.PROCESS)

    def test_async_checkpointer_type_from_string_thread(self):
        """Test creating enum from string value 'thread'."""
        member = AsyncCheckpointerType("thread")
        self.assertIs(member, AsyncCheckpointerType.THREAD)

    def test_async_checkpointer_type_from_string_process(self):
        """Test creating enum from string value 'process'."""
        member = AsyncCheckpointerType("process")
        self.assertIs(member, AsyncCheckpointerType.PROCESS)

    def test_async_checkpointer_type_invalid_value(self):
        """Test that invalid enum value raises ValueError."""
        with self.assertRaises(ValueError):
            AsyncCheckpointerType("invalid")

    def test_async_checkpointer_type_member_count(self):
        """Test that enum has exactly two members."""
        self.assertEqual(len(AsyncCheckpointerType), 2)

    def test_async_checkpointer_type_iteration(self):
        """Test iterating over enum members."""
        members = list(AsyncCheckpointerType)
        self.assertEqual(len(members), 2)
        self.assertIn(AsyncCheckpointerType.THREAD, members)
        self.assertIn(AsyncCheckpointerType.PROCESS, members)

    def test_async_checkpointer_type_str(self):
        """Test string representation of enum members."""
        self.assertEqual(str(AsyncCheckpointerType.THREAD), "AsyncCheckpointerType.THREAD")
        self.assertEqual(str(AsyncCheckpointerType.PROCESS), "AsyncCheckpointerType.PROCESS")

    def test_async_checkpointer_type_repr(self):
        """Test repr of enum members."""
        self.assertEqual(repr(AsyncCheckpointerType.THREAD), "<AsyncCheckpointerType.THREAD: 'thread'>")
        self.assertEqual(repr(AsyncCheckpointerType.PROCESS), "<AsyncCheckpointerType.PROCESS: 'process'>")

    def test_async_checkpointer_type_isinstance_check(self):
        """Test isinstance check for enum members."""
        self.assertIsInstance(AsyncCheckpointerType.THREAD, AsyncCheckpointerType)
        self.assertIsInstance(AsyncCheckpointerType.PROCESS, AsyncCheckpointerType)


if __name__ == "__main__":
    run_tests()
