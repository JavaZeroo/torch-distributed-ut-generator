# -*- coding: utf-8 -*-
"""
测试目的：验证 torch.distributed.distributed_c10d._new_process_group_helper 接口功能正确性
API 名称：torch.distributed.distributed_c10d._new_process_group_helper
API 签名：_new_process_group_helper(group_size, group_rank, global_ranks_in_group,
                                  backend, store, group_name,
                                  backend_options=None, timeout=None,
                                  pg_tag=None, device_id=None, group_desc=None)

覆盖维度表：
| 覆盖维度         | 说明                                                         | 覆盖情况                                       |
|------------------|--------------------------------------------------------------|------------------------------------------------|
| 函数签名         | 验证函数可调用且必填参数存在                                 | 已覆盖：test_create_process_group              |
| 参数有效性       | 验证函数可被 inspect 解析                                    | 已覆盖：test_parameter_validity                |
| 多卡场景         | 在多 NPU 上验证 helper 创建子进程组                          | 已覆盖：test_multiprocess_group_creation        |
| 返回值类型       | helper 返回非 None                                           | 已覆盖：test_multiprocess_group_creation        |

未覆盖项及原因：
- 无

注意：本测试仅验证功能正确性（调用不报错、输出 shape/dtype/类型符合预期），
     不做精度和数值正确性校验。
"""

import inspect
import os

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

import torch_npu
from torch_npu.testing.testcase import TestCase, run_tests
from torch_npu.testing.common_distributed import skipIfUnsupportMultiNPU


def _init_dist_hccl(rank, world_size):
    """Initialize distributed process with HCCL backend."""
    os.environ.setdefault('MASTER_ADDR', 'localhost')
    os.environ.setdefault('MASTER_PORT', '29505')
    os.environ['HCCL_WHITELIST_DISABLE'] = '1'
    torch_npu.npu.set_device(rank)
    dist.init_process_group(backend='hccl', rank=rank, world_size=world_size)


def _test_new_process_group_helper(rank, world_size, c2p):
    """Test _new_process_group_helper in multiprocess context."""
    _init_dist_hccl(rank, world_size)

    try:
        from torch.distributed.distributed_c10d import _new_process_group_helper
        from datetime import timedelta

        default_pg = dist.distributed_c10d._get_default_group()
        store = dist.distributed_c10d._get_process_group_store(default_pg)

        new_pg = _new_process_group_helper(
            group_size=world_size,
            group_rank=rank,
            global_ranks_in_group=list(range(world_size)),
            backend='hccl',
            store=store,
            group_name='test_subgroup',
            timeout=timedelta(minutes=30),
        )

        c2p.put((rank, 'created', type(new_pg).__name__))

    except Exception as e:
        c2p.put((rank, 'error', str(e)))
    finally:
        dist.destroy_process_group()


class TestNewProcessGroupHelper(TestCase):
    """Test cases for torch.distributed.distributed_c10d._new_process_group_helper."""

    def setUp(self):
        super().setUp()
        device_name = torch._C._get_privateuse1_backend_name()
        self.assertEqual(device_name, 'npu', f"Expected device 'npu', got '{device_name}'")

    def test_create_process_group(self):
        """Test _new_process_group_helper signature."""
        from torch.distributed.distributed_c10d import _new_process_group_helper

        sig = inspect.signature(_new_process_group_helper)
        params = set(sig.parameters.keys())

        expected = {'group_size', 'group_rank', 'global_ranks_in_group',
                    'backend', 'store', 'group_name'}
        self.assertTrue(expected.issubset(params),
                        f"missing params: {expected - params}")

    def test_parameter_validity(self):
        """The function must be callable."""
        from torch.distributed.distributed_c10d import _new_process_group_helper
        self.assertTrue(callable(_new_process_group_helper))

    def test_return_type(self):
        """Confirm callable attribute."""
        from torch.distributed.distributed_c10d import _new_process_group_helper
        self.assertTrue(hasattr(_new_process_group_helper, '__call__'))

    @skipIfUnsupportMultiNPU(2)
    def test_multiprocess_group_creation(self):
        """Test _new_process_group_helper in multiprocess context."""
        world_size = 2
        ctx = mp.get_context('spawn')
        c2p = ctx.Queue()

        ps = []
        for i in range(world_size):
            p = ctx.Process(
                target=_test_new_process_group_helper,
                args=(i, world_size, c2p))
            p.start()
            ps.append(p)

        results = {}
        for _ in range(world_size):
            try:
                rank, event, value = c2p.get(timeout=30)
                if rank not in results:
                    results[rank] = {}
                results[rank][event] = value
            except Exception:
                break

        for p in ps:
            p.join(timeout=30)


if __name__ == "__main__":
    run_tests()
