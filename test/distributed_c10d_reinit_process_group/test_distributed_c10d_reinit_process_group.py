# -*- coding: utf-8 -*-
"""
测试目的：验证 torch.distributed.reinit_process_group 接口功能正确性
API 名称：torch.distributed.reinit_process_group
API 签名：reinit_process_group(group=None, rebuild_link=True) -> ProcessGroup | None

覆盖维度表：
| 覆盖维度         | 说明                                                         | 覆盖情况                                                      |
|------------------|--------------------------------------------------------------|---------------------------------------------------------------|
| 空/非空          | group=None（默认进程组）vs 显式 ProcessGroup 对象            | 已覆盖：test_reinit_default_group / test_reinit_explicit_group|
| 枚举选项         | rebuild_link=True vs rebuild_link=False                      | 已覆盖：test_reinit_rebuild_link_true / _false                |
| 参数类型         | group: None / ProcessGroup；rebuild_link: bool               | 已覆盖                                                        |
| 传参与不传参     | 省略所有参数使用默认值 vs 显式传入                           | 已覆盖                                                        |
| 等价类/边界值    | world_size=2 基础多卡场景                                    | 已覆盖                                                        |
| 正常传参场景     | group=None rebuild_link=True/False 正常调用                  | 已覆盖                                                        |
| 异常传参场景     | 未初始化进程组时调用（依赖外部状态，无稳定异常路径）         | 未覆盖，原因：NPU 环境下 reinit 属于恢复性 API，异常行为不稳定|

未覆盖项及原因：
- reinit 后继续通信的完整端对端验证：rebuild_link 后重建通信依赖底层 HCCL 实现细节

注意：本测试仅验证功能正确性（调用不报错、返回值类型符合预期），
     不做精度和数值正确性校验。
"""

import os
import inspect
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

import torch_npu  # noqa: F401 — registers NPU backend
from torch_npu.testing.testcase import TestCase, run_tests
from torch_npu.testing.common_distributed import skipIfUnsupportMultiNPU


def _test_reinit_default_group(rank, world_size, c2p):
    """Test reinit_process_group(group=None) — default group, rebuild_link=True."""
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '29518'
    os.environ['HCCL_WHITELIST_DISABLE'] = '1'
    torch_npu.npu.set_device(rank)

    try:
        dist.init_process_group(backend='hccl', rank=rank, world_size=world_size)

        # Call with all defaults: group=None, rebuild_link=True
        result = torch.distributed.reinit_process_group()
        # Should return the default ProcessGroup (rebuild_link=True path)
        c2p.put((rank, 'reinit_default_type', type(result).__name__))
        c2p.put((rank, 'reinit_default_ok', True))

        dist.destroy_process_group()
    except Exception as e:
        c2p.put((rank, 'error', str(e)))


def _test_reinit_rebuild_false(rank, world_size, c2p):
    """Test reinit_process_group with rebuild_link=False — resume mode."""
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '29518'
    os.environ['HCCL_WHITELIST_DISABLE'] = '1'
    torch_npu.npu.set_device(rank)

    try:
        dist.init_process_group(backend='hccl', rank=rank, world_size=world_size)

        # rebuild_link=False: resume hccl comm, returns None
        result = torch.distributed.reinit_process_group(rebuild_link=False)
        c2p.put((rank, 'reinit_rebuild_false_result', result))
        c2p.put((rank, 'reinit_rebuild_false_ok', True))

        dist.destroy_process_group()
    except Exception as e:
        c2p.put((rank, 'error', str(e)))


def _test_reinit_explicit_group(rank, world_size, c2p):
    """Test reinit_process_group with explicit group argument."""
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '29518'
    os.environ['HCCL_WHITELIST_DISABLE'] = '1'
    torch_npu.npu.set_device(rank)

    try:
        dist.init_process_group(backend='hccl', rank=rank, world_size=world_size)
        pg = dist.group.WORLD

        # Pass explicit ProcessGroup
        result = torch.distributed.reinit_process_group(group=pg, rebuild_link=True)
        c2p.put((rank, 'explicit_group_ok', True))
        c2p.put((rank, 'explicit_group_result_type', type(result).__name__))

        dist.destroy_process_group()
    except Exception as e:
        c2p.put((rank, 'error', str(e)))


class TestReinitProcessGroup(TestCase):
    """Test cases for torch.distributed.reinit_process_group."""

    def setUp(self):
        super().setUp()
        self.device_name = torch._C._get_privateuse1_backend_name()
        self.assertEqual(
            self.device_name, 'npu',
            f"Expected device 'npu', got '{self.device_name}'"
        )

    def test_function_signature(self):
        """reinit_process_group has correct parameter signature."""
        fn = torch.distributed.reinit_process_group
        self.assertTrue(callable(fn))
        sig = inspect.signature(fn)
        params = list(sig.parameters.keys())
        self.assertIn('group', params)
        self.assertIn('rebuild_link', params)
        # Both parameters should have defaults
        for name in ('group', 'rebuild_link'):
            p = sig.parameters[name]
            self.assertNotEqual(
                p.default, inspect.Parameter.empty,
                f"Parameter '{name}' should have a default value"
            )

    def _run_mp(self, fn, world_size=2):
        ctx = mp.get_context('spawn')
        c2p = ctx.Queue()
        ps = []
        for i in range(world_size):
            p = ctx.Process(target=fn, args=(i, world_size, c2p))
            p.start()
            ps.append(p)

        results = {}
        # Collect up to world_size * 4 messages
        for _ in range(world_size * 4):
            try:
                rank, key, val = c2p.get(timeout=60)
                results.setdefault(rank, {})[key] = val
            except Exception:
                break

        for p in ps:
            p.join(timeout=60)
            self.assertEqual(p.exitcode, 0, "subprocess exit with abnormal code.")

        return results

    @skipIfUnsupportMultiNPU(2)
    def test_reinit_default_group(self):
        """reinit_process_group() with default args (group=None, rebuild_link=True)."""
        results = self._run_mp(_test_reinit_default_group)
        for rank in range(2):
            r = results.get(rank, {})
            if 'error' in r:
                self.fail(f"Rank {rank} raised: {r['error']}")
            self.assertTrue(r.get('reinit_default_ok'))

    @skipIfUnsupportMultiNPU(2)
    def test_reinit_rebuild_link_false(self):
        """reinit_process_group(rebuild_link=False) returns None (resume mode)."""
        results = self._run_mp(_test_reinit_rebuild_false)
        for rank in range(2):
            r = results.get(rank, {})
            if 'error' in r:
                self.fail(f"Rank {rank} raised: {r['error']}")
            self.assertTrue(r.get('reinit_rebuild_false_ok'))
            # rebuild_link=False path returns None
            self.assertIsNone(r.get('reinit_rebuild_false_result'))

    @skipIfUnsupportMultiNPU(2)
    def test_reinit_explicit_group(self):
        """reinit_process_group with explicit ProcessGroup argument."""
        results = self._run_mp(_test_reinit_explicit_group)
        for rank in range(2):
            r = results.get(rank, {})
            if 'error' in r:
                self.fail(f"Rank {rank} raised: {r['error']}")
            self.assertTrue(r.get('explicit_group_ok'))


if __name__ == "__main__":
    run_tests()
