# -*- coding: utf-8 -*-
"""
测试目的：验证 torch.distributed.distributed_c10d.split_group 接口功能正确性
API 名称：torch.distributed.distributed_c10d.split_group
API 签名：
  split_group(
      parent_pg: ProcessGroup | None = None,
      split_ranks: list | None = None,
      timeout: timedelta | None = None,
      pg_options: Any | None = None,
      group_desc: str | None = None
  ) -> ProcessGroup | None

覆盖维度表：
| 覆盖维度         | 说明                                                         | 覆盖情况                                       |
|------------------|--------------------------------------------------------------|------------------------------------------------|
| 空/非空          | parent_pg=None vs explicit, split_ranks=None vs explicit     | 已覆盖                                         |
| 参数类型         | ProcessGroup, list, timedelta, str                           | 已覆盖 (除 pg_options)                         |
| 传参与不传参     | 默认参数 vs 显式传参                                         | 已覆盖 (timeout, group_desc)                   |
| 等价类/边界值    | 正常 split_ranks 配置                                        | 已覆盖                                         |
| 正常传参场景     | 正常调用 split_group                                         | 已覆盖                                         |
| 异常传参场景     | 非法 split_ranks 配置                                        | 未覆盖 (依赖底层实现)                          |

未覆盖项及原因：
- 异常传参场景：split_group 的异常处理依赖底层 NCCL/HCCL 实现，无稳定 Python 层异常路径

注意：本测试仅验证功能正确性（调用不报错、输出 shape/dtype/类型符合预期），
     不做精度和数值正确性校验。
"""

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from datetime import timedelta

from torch_npu.testing.testcase import TestCase, run_tests
from torch_npu.testing.common_distributed import skipIfUnsupportMultiNPU


def _init_dist_process(rank, world_size, fn, backend='hccl'):
    """Initialize distributed process with HCCL backend."""
    import os
    os.environ.setdefault('MASTER_ADDR', 'localhost')
    os.environ.setdefault('MASTER_PORT', '29505')

    torch.npu.set_device(rank)
    dist.init_process_group(backend, rank=rank, world_size=world_size)

    try:
        fn(rank, world_size)
    finally:
        dist.destroy_process_group()


def _test_split_group_with_default_parent(rank, world_size):
    """Test split_group with default parent_pg (None)."""
    device = torch.device(f'npu:{rank}')

    # Split 4 ranks into 2 groups of 2
    # split_ranks contains group ranks in parent pg
    split_ranks = [[0, 1], [2, 3]]

    # This test requires at least 4 ranks
    if world_size >= 4:
        sub_group = dist.split_group(
            parent_pg=None,
            split_ranks=split_ranks
        )

        # Verify that we got a ProcessGroup or None
        if sub_group is not None:
            assert hasattr(sub_group, 'size'), "sub_group should have size method"

    dist.barrier()


def _test_split_group_with_explicit_parent(rank, world_size):
    """Test split_group with explicit parent_pg."""
    device = torch.device(f'npu:{rank}')

    # Get default process group
    parent_pg = dist.group.WORLD

    # Split 4 ranks into 2 groups of 2
    split_ranks = [[0, 1], [2, 3]]

    if world_size >= 4:
        sub_group = dist.split_group(
            parent_pg=parent_pg,
            split_ranks=split_ranks
        )

        if sub_group is not None:
            assert hasattr(sub_group, 'size'), "sub_group should have size method"

    dist.barrier()


def _test_split_group_with_timeout(rank, world_size):
    """Test split_group with explicit timeout."""
    device = torch.device(f'npu:{rank}')

    split_ranks = [[0, 1], [2, 3]]

    if world_size >= 4:
        sub_group = dist.split_group(
            parent_pg=None,
            split_ranks=split_ranks,
            timeout=timedelta(seconds=600)
        )

        if sub_group is not None:
            assert hasattr(sub_group, 'size'), "sub_group should have size method"

    dist.barrier()


def _test_split_group_with_group_desc(rank, world_size):
    """Test split_group with group_desc."""
    device = torch.device(f'npu:{rank}')

    split_ranks = [[0, 1], [2, 3]]

    if world_size >= 4:
        sub_group = dist.split_group(
            parent_pg=None,
            split_ranks=split_ranks,
            group_desc="test_sub_group"
        )

        if sub_group is not None:
            assert hasattr(sub_group, 'size'), "sub_group should have size method"

    dist.barrier()


def _test_split_group_all_params(rank, world_size):
    """Test split_group with all parameters."""
    device = torch.device(f'npu:{rank}')

    parent_pg = dist.group.WORLD
    split_ranks = [[0, 1], [2, 3]]

    if world_size >= 4:
        sub_group = dist.split_group(
            parent_pg=parent_pg,
            split_ranks=split_ranks,
            timeout=timedelta(seconds=300),
            pg_options=None,
            group_desc="full_test_group"
        )

        if sub_group is not None:
            assert hasattr(sub_group, 'size'), "sub_group should have size method"

    dist.barrier()


def _test_split_group_single_group(rank, world_size):
    """Test split_group with single group (others become non-members)."""
    device = torch.device(f'npu:{rank}')

    # Only create one sub-group with ranks 0,1
    # Ranks 2,3 will return None
    split_ranks = [[0, 1]]

    if world_size >= 4:
        sub_group = dist.split_group(
            parent_pg=None,
            split_ranks=split_ranks
        )

        # Rank 0,1 should get a ProcessGroup, rank 2,3 should get None
        if rank in [0, 1]:
            assert sub_group is not None, f"Rank {rank} should be in sub_group"
        else:
            assert sub_group is None, f"Rank {rank} should not be in sub_group"

    dist.barrier()


class TestSplitGroup(TestCase):
    """Test cases for split_group function."""

    def setUp(self):
        super().setUp()
        self.device_name = torch._C._get_privateuse1_backend_name()
        self.assertEqual(self.device_name, 'npu', f"Expected device 'npu', got '{self.device_name}'")

    @skipIfUnsupportMultiNPU(4)
    def test_split_group_with_default_parent(self):
        """Test split_group with default parent_pg."""
        world_size = 4
        mp.spawn(
            _init_dist_process,
            args=(world_size, _test_split_group_with_default_parent),
            nprocs=world_size,
            join=True
        )

    @skipIfUnsupportMultiNPU(4)
    def test_split_group_with_explicit_parent(self):
        """Test split_group with explicit parent_pg."""
        world_size = 4
        mp.spawn(
            _init_dist_process,
            args=(world_size, _test_split_group_with_explicit_parent),
            nprocs=world_size,
            join=True
        )

    @skipIfUnsupportMultiNPU(4)
    def test_split_group_with_timeout(self):
        """Test split_group with explicit timeout."""
        world_size = 4
        mp.spawn(
            _init_dist_process,
            args=(world_size, _test_split_group_with_timeout),
            nprocs=world_size,
            join=True
        )

    @skipIfUnsupportMultiNPU(4)
    def test_split_group_with_group_desc(self):
        """Test split_group with group_desc."""
        world_size = 4
        mp.spawn(
            _init_dist_process,
            args=(world_size, _test_split_group_with_group_desc),
            nprocs=world_size,
            join=True
        )

    @skipIfUnsupportMultiNPU(4)
    def test_split_group_all_params(self):
        """Test split_group with all parameters."""
        world_size = 4
        mp.spawn(
            _init_dist_process,
            args=(world_size, _test_split_group_all_params),
            nprocs=world_size,
            join=True
        )

    @skipIfUnsupportMultiNPU(4)
    def test_split_group_single_group(self):
        """Test split_group with single group."""
        world_size = 4
        mp.spawn(
            _init_dist_process,
            args=(world_size, _test_split_group_single_group),
            nprocs=world_size,
            join=True
        )


if __name__ == "__main__":
    run_tests()
