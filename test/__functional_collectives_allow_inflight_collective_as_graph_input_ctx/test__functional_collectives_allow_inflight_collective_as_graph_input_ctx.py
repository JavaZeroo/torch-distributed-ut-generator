# -*- coding: utf-8 -*-
"""
测试目的：验证 torch.distributed._functional_collectives.allow_inflight_collective_as_graph_input_ctx 接口功能正确性
API 名称：torch.distributed._functional_collectives.allow_inflight_collective_as_graph_input_ctx
API 签名：allow_inflight_collective_as_graph_input_ctx(value: bool = True) -> contextmanager

覆盖维度表：
| 覆盖维度         | 说明                                                         | 覆盖情况                                       |
|------------------|--------------------------------------------------------------|------------------------------------------------|
| 空/非空          | value=True（默认）vs value=False                             | 已覆盖：test_ctx_default / test_ctx_false      |
| 枚举选项         | value=True / value=False                                     | 已覆盖                                         |
| 参数类型         | value: bool                                                  | 已覆盖                                         |
| 传参与不传参     | 省略 value 使用默认 True vs 显式传入                         | 已覆盖                                         |
| 等价类/边界值    | 进入/退出上下文后状态恢复                                    | 已覆盖：test_ctx_restores_previous_value       |
| 正常传参场景     | 上下文管理器正常进入和退出                                   | 已覆盖：test_ctx_enters_and_exits              |
| 异常传参场景     | 上下文内抛出异常时状态仍能恢复                               | 已覆盖：test_ctx_restores_on_exception         |

未覆盖项及原因：
- 在 torch.compile 下实际触发 inflight collective 的端到端测试：依赖 CUDA/dynamo 编译路径，NPU 环境暂不覆盖

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

from torch.distributed._functional_collectives import (
    allow_inflight_collective_as_graph_input_ctx,
)


def _get_current_flag() -> bool:
    return torch._C._distributed_c10d._allow_inflight_collective_as_graph_input()


class TestAllowInflightCollectiveCtx(TestCase):
    """Test cases for allow_inflight_collective_as_graph_input_ctx."""

    def setUp(self):
        super().setUp()
        device_name = torch._C._get_privateuse1_backend_name()
        self.assertEqual(device_name, 'npu', f"Expected device 'npu', got '{device_name}'")
        # Record the initial state so we can verify it is restored
        self._initial = _get_current_flag()

    def tearDown(self):
        super().tearDown()
        # Restore initial state in case a test left the flag set
        torch._C._distributed_c10d._set_allow_inflight_collective_as_graph_input(
            self._initial
        )

    def test_ctx_enters_and_exits(self):
        """Context manager can be entered and exited without error."""
        with allow_inflight_collective_as_graph_input_ctx():
            pass  # should not raise

    def test_ctx_default_sets_true(self):
        """Default value=True sets the flag to True inside the context."""
        with allow_inflight_collective_as_graph_input_ctx():
            self.assertTrue(_get_current_flag())

    def test_ctx_false(self):
        """value=False sets the flag to False inside the context."""
        # First ensure the flag is True
        torch._C._distributed_c10d._set_allow_inflight_collective_as_graph_input(True)
        with allow_inflight_collective_as_graph_input_ctx(False):
            self.assertFalse(_get_current_flag())

    def test_ctx_explicit_true(self):
        """Explicit value=True sets the flag to True inside the context."""
        with allow_inflight_collective_as_graph_input_ctx(True):
            self.assertTrue(_get_current_flag())

    def test_ctx_restores_previous_value(self):
        """Flag is restored to its previous value after the context exits."""
        before = _get_current_flag()
        with allow_inflight_collective_as_graph_input_ctx(not before):
            pass  # flip
        self.assertEqual(_get_current_flag(), before)

    def test_ctx_restores_on_exception(self):
        """Flag is restored even when an exception is raised inside the context."""
        before = _get_current_flag()
        try:
            with allow_inflight_collective_as_graph_input_ctx(not before):
                raise RuntimeError("test error")
        except RuntimeError:
            pass
        self.assertEqual(_get_current_flag(), before)

    def test_ctx_nested(self):
        """Nested context managers restore state correctly (LIFO)."""
        before = _get_current_flag()
        with allow_inflight_collective_as_graph_input_ctx(True):
            with allow_inflight_collective_as_graph_input_ctx(False):
                self.assertFalse(_get_current_flag())
            self.assertTrue(_get_current_flag())
        self.assertEqual(_get_current_flag(), before)

    def test_ctx_is_generator(self):
        """allow_inflight_collective_as_graph_input_ctx is a context manager."""
        import contextlib
        ctx = allow_inflight_collective_as_graph_input_ctx()
        self.assertTrue(hasattr(ctx, '__enter__') and hasattr(ctx, '__exit__'))


if __name__ == "__main__":
    run_tests()
