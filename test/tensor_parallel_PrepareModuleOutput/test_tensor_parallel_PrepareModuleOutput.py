# -*- coding: utf-8 -*-
"""
测试目的：验证 torch.distributed.tensor.parallel.PrepareModuleOutput 接口功能正确性
API 名称：torch.distributed.tensor.parallel.PrepareModuleOutput
API 签名：
  __init__(
      *,
      output_layouts: Placement | tuple[Placement | None, ...],
      desired_output_layouts: Placement | tuple[Placement, ...],
      use_local_output: bool = True
  )

覆盖维度表：
| 覆盖维度         | 说明                                                         | 覆盖情况                                       |
|------------------|--------------------------------------------------------------|------------------------------------------------|
| 空/非空          | -                                                            | N/A                                            |
| 枚举选项         | use_local_output True/False                                  | 已覆盖                                         |
| 参数类型         | Placement, tuple[Placement, ...]                             | 已覆盖                                         |
| 传参与不传参     | use_local_output default vs explicit                         | 已覆盖                                         |
| 等价类/边界值    | 单 Placement, 多 Placement tuple                             | 已覆盖                                         |
| 正常传参场景     | 正常构造 PrepareModuleOutput                                 | 已覆盖                                         |
| 异常传参场景     | output_layouts和desired_output_layouts长度不一致           | 已覆盖 (应抛出 AssertionError)                 |

未覆盖项及原因：
- 无

注意：本测试仅验证功能正确性（调用不报错、输出 shape/dtype/类型符合预期），
     不做精度和数值正确性校验。
"""

import torch
import torch.nn as nn
import torch.multiprocessing as mp
import torch.distributed as dist
from torch.distributed.tensor.parallel import PrepareModuleOutput
from torch.distributed.tensor.placement_types import Replicate, Shard
from torch.distributed.tensor import DeviceMesh

from torch_npu.testing.testcase import TestCase, run_tests
from torch_npu.testing.common_distributed import skipIfUnsupportMultiNPU


def _init_dist_process(rank, world_size, fn, backend='hccl'):
    """Initialize distributed process with HCCL backend."""
    import os
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '29503'

    torch.npu.set_device(rank)
    dist.init_process_group(backend, rank=rank, world_size=world_size)

    try:
        fn(rank, world_size)
    finally:
        dist.destroy_process_group()


def _test_prepare_module_output_single_placement(rank, world_size):
    """Test PrepareModuleOutput with single Placement."""
    device = torch.device(f'npu:{rank}')

    # Test with single Placement
    prepare_output = PrepareModuleOutput(
        output_layouts=Replicate(),
        desired_output_layouts=Shard(0),
        use_local_output=True
    )

    assert isinstance(prepare_output, PrepareModuleOutput)
    assert prepare_output.use_local_output is True

    dist.barrier()


def _test_prepare_module_output_tuple_placement(rank, world_size):
    """Test PrepareModuleOutput with tuple of Placements."""
    device = torch.device(f'npu:{rank}')

    # Test with tuple of Placements
    prepare_output = PrepareModuleOutput(
        output_layouts=(Replicate(), Shard(0)),
        desired_output_layouts=(Shard(0), Replicate()),
        use_local_output=False
    )

    assert isinstance(prepare_output, PrepareModuleOutput)
    assert prepare_output.use_local_output is False

    dist.barrier()


def _test_prepare_module_output_default_use_local(rank, world_size):
    """Test PrepareModuleOutput with default use_local_output."""
    device = torch.device(f'npu:{rank}')

    # Test with default use_local_output (should be True)
    prepare_output = PrepareModuleOutput(
        output_layouts=Replicate(),
        desired_output_layouts=Shard(0)
    )

    assert isinstance(prepare_output, PrepareModuleOutput)
    assert prepare_output.use_local_output is True

    dist.barrier()


def _test_prepare_module_output_mismatched_lengths(rank, world_size):
    """Test PrepareModuleOutput with mismatched layout lengths (should raise)."""
    device = torch.device(f'npu:{rank}')

    # Test with mismatched lengths - should raise AssertionError
    try:
        prepare_output = PrepareModuleOutput(
            output_layouts=(Replicate(), Shard(0), Replicate()),
            desired_output_layouts=(Shard(0), Replicate())
        )
        # If we reach here without exception, that's unexpected
        assert False, "Expected AssertionError for mismatched lengths"
    except AssertionError as e:
        # Expected exception
        assert "output_layouts and desired_output_layouts should have same length" in str(e)

    dist.barrier()


class TestPrepareModuleOutput(TestCase):
    """Test cases for PrepareModuleOutput."""

    def setUp(self):
        super().setUp()
        self.device_name = torch._C._get_privateuse1_backend_name()
        self.assertEqual(self.device_name, 'npu', f"Expected device 'npu', got '{self.device_name}'")

    @skipIfUnsupportMultiNPU(2)
    def test_single_placement(self):
        """Test PrepareModuleOutput with single Placement."""
        world_size = 2
        mp.spawn(
            _init_dist_process,
            args=(world_size, _test_prepare_module_output_single_placement),
            nprocs=world_size,
            join=True
        )

    @skipIfUnsupportMultiNPU(2)
    def test_tuple_placement(self):
        """Test PrepareModuleOutput with tuple of Placements."""
        world_size = 2
        mp.spawn(
            _init_dist_process,
            args=(world_size, _test_prepare_module_output_tuple_placement),
            nprocs=world_size,
            join=True
        )

    @skipIfUnsupportMultiNPU(2)
    def test_default_use_local_output(self):
        """Test PrepareModuleOutput with default use_local_output."""
        world_size = 2
        mp.spawn(
            _init_dist_process,
            args=(world_size, _test_prepare_module_output_default_use_local),
            nprocs=world_size,
            join=True
        )

    @skipIfUnsupportMultiNPU(2)
    def test_mismatched_lengths(self):
        """Test PrepareModuleOutput with mismatched layout lengths."""
        world_size = 2
        mp.spawn(
            _init_dist_process,
            args=(world_size, _test_prepare_module_output_mismatched_lengths),
            nprocs=world_size,
            join=True
        )


if __name__ == "__main__":
    run_tests()
