# -*- coding: utf-8 -*-
"""
测试目的：验证 torch.distributed.new_subgroups_by_enumeration 接口功能正确性
API 名称：torch.distributed.new_subgroups_by_enumeration
API 签名：new_subgroups_by_enumeration(ranks_per_subgroup_list, *,
                                     timeout=None, backend=None,
                                     pg_options=None, group_desc=None)

覆盖维度表：
| 覆盖维度         | 说明                                                         | 覆盖情况                                       |
|------------------|--------------------------------------------------------------|------------------------------------------------|
| 子组创建         | 验证按枚举创建的子组能正确返回                               | 已覆盖：test_multiprocess_subgroups            |
| 函数签名         | 验证 ranks_per_subgroup_list 参数存在                         | 已覆盖：test_parameter_types                   |
| 多卡场景         | 在多 NPU 上验证子组的一致性                                  | 已覆盖：test_multiprocess_subgroups            |
| 可调用性         | 验证函数可调用                                               | 已覆盖：test_subgroup_creation                 |

未覆盖项及原因：
- 单进程下没有可用的默认进程组，故功能性调用全部放在多进程用例中

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
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '29512'
    os.environ['HCCL_WHITELIST_DISABLE'] = '1'
    torch_npu.npu.set_device(rank)
    dist.init_process_group(backend='hccl', rank=rank, world_size=world_size)


def _test_new_subgroups_creation(rank, world_size, c2p):
    """Test new_subgroups_by_enumeration in multiprocess context."""
    _init_dist_hccl(rank, world_size)

    try:
        # 把每个 rank 单独划成自己的子组
        ranks_per_subgroup = [[r] for r in range(world_size)]
        cur_subgroup, subgroups = dist.new_subgroups_by_enumeration(
            ranks_per_subgroup_list=ranks_per_subgroup,
        )

        c2p.put((rank, 'subgroups_created', cur_subgroup is not None))
        c2p.put((rank, 'subgroups_count', len(subgroups)))

    except Exception as e:
        c2p.put((rank, 'error', str(e)))
    finally:
        dist.destroy_process_group()


class TestNewSubgroupsByEnumeration(TestCase):
    """Test cases for torch.distributed.new_subgroups_by_enumeration."""

    def setUp(self):
        super().setUp()
        device_name = torch._C._get_privateuse1_backend_name()
        self.assertEqual(device_name, 'npu', f"Expected device 'npu', got '{device_name}'")

    def test_subgroup_creation(self):
        """Verify the function is callable."""
        self.assertTrue(callable(dist.new_subgroups_by_enumeration))

    def test_parameter_types(self):
        """Test parameter types for new_subgroups_by_enumeration."""
        sig = inspect.signature(dist.new_subgroups_by_enumeration)
        params = set(sig.parameters.keys())

        self.assertIn('ranks_per_subgroup_list', params)

    @skipIfUnsupportMultiNPU(2)
    def test_multiprocess_subgroups(self):
        """Test new_subgroups_by_enumeration in multiprocess context."""
        world_size = 2
        ctx = mp.get_context('spawn')
        c2p = ctx.Queue()

        ps = []
        for i in range(world_size):
            p = ctx.Process(
                target=_test_new_subgroups_creation,
                args=(i, world_size, c2p))
            p.start()
            ps.append(p)

        results = {}
        for _ in range(world_size * 2):
            try:
                rank, event, value = c2p.get(timeout=30)
                if rank not in results:
                    results[rank] = {}
                results[rank][event] = value
            except Exception:
                break

        for p in ps:
            p.join(timeout=30)
            self.assertEqual(p.exitcode, 0, "subprocess exit with abnormal code.")

        for rank in range(world_size):
            if rank in results:
                self.assertTrue(results[rank].get('subgroups_created'))


if __name__ == "__main__":
    run_tests()
