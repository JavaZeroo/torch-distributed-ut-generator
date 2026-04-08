# -*- coding: utf-8 -*-
"""
测试目的：验证 torch.distributed.tensor.placement_types.Partial 接口功能正确性
API 名称：torch.distributed.tensor.placement_types.Partial
API 签名：
  __init__(reduce_op: str = "sum")

覆盖维度表：
| 覆盖维度         | 说明                                                         | 覆盖情况                                       |
|------------------|--------------------------------------------------------------|------------------------------------------------|
| 空/非空          | -                                                            | N/A                                            |
| 枚举选项         | reduce_op: "sum", "avg", "min", "max", "product"             | 已覆盖                                         |
| 参数类型         | str                                                          | 已覆盖                                         |
| 传参与不传参     | default "sum" vs explicit value                              | 已覆盖                                         |
| 等价类/边界值    | 合法 reduce_op 值                                            | 已覆盖                                         |
| 正常传参场景     | 正常构造 Partial                                             | 已覆盖                                         |
| 异常传参场景     | 非法 reduce_op 值                                            | 未覆盖 (行为未定义)                            |

未覆盖项及原因：
- 异常传参场景：Partial 对非法 reduce_op 的处理行为未明确文档化，未覆盖

注意：本测试仅验证功能正确性（调用不报错、输出 shape/dtype/类型符合预期），
     不做精度和数值正确性校验。
"""

import torch
import torch.multiprocessing as mp
import torch.distributed as dist
from torch.distributed.tensor.placement_types import Partial

from torch_npu.testing.testcase import TestCase, run_tests
from torch_npu.testing.common_distributed import skipIfUnsupportMultiNPU


# All supported reduce operations
LINEAR_REDUCE_OPS = ("sum", "avg")
ALL_REDUCE_OPS = ("sum", "avg", "min", "max", "product")


def _init_dist_process(rank, world_size, fn, backend='hccl'):
    """Initialize distributed process with HCCL backend."""
    import os
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '29504'

    torch.npu.set_device(rank)
    dist.init_process_group(backend, rank=rank, world_size=world_size)

    try:
        fn(rank, world_size)
    finally:
        dist.destroy_process_group()


def _test_partial_default(rank, world_size):
    """Test Partial with default reduce_op."""
    device = torch.device(f'npu:{rank}')

    # Test with default reduce_op (should be "sum")
    partial = Partial()

    assert isinstance(partial, Partial)
    assert partial.reduce_op == "sum"

    dist.barrier()


def _test_partial_sum(rank, world_size):
    """Test Partial with "sum" reduce_op."""
    device = torch.device(f'npu:{rank}')

    partial = Partial(reduce_op="sum")

    assert isinstance(partial, Partial)
    assert partial.reduce_op == "sum"

    dist.barrier()


def _test_partial_avg(rank, world_size):
    """Test Partial with "avg" reduce_op."""
    device = torch.device(f'npu:{rank}')

    partial = Partial(reduce_op="avg")

    assert isinstance(partial, Partial)
    assert partial.reduce_op == "avg"

    dist.barrier()


def _test_partial_min(rank, world_size):
    """Test Partial with "min" reduce_op."""
    device = torch.device(f'npu:{rank}')

    partial = Partial(reduce_op="min")

    assert isinstance(partial, Partial)
    assert partial.reduce_op == "min"

    dist.barrier()


def _test_partial_max(rank, world_size):
    """Test Partial with "max" reduce_op."""
    device = torch.device(f'npu:{rank}')

    partial = Partial(reduce_op="max")

    assert isinstance(partial, Partial)
    assert partial.reduce_op == "max"

    dist.barrier()


def _test_partial_product(rank, world_size):
    """Test Partial with "product" reduce_op."""
    device = torch.device(f'npu:{rank}')

    partial = Partial(reduce_op="product")

    assert isinstance(partial, Partial)
    assert partial.reduce_op == "product"

    dist.barrier()


def _test_partial_linear_ops(rank, world_size):
    """Test all linear reduce ops."""
    device = torch.device(f'npu:{rank}')

    for op in LINEAR_REDUCE_OPS:
        partial = Partial(reduce_op=op)
        assert isinstance(partial, Partial)
        assert partial.reduce_op == op
        assert op in Partial.LINEAR_REDUCE_OPS

    dist.barrier()


def _test_partial_all_ops(rank, world_size):
    """Test all supported reduce ops."""
    device = torch.device(f'npu:{rank}')

    for op in ALL_REDUCE_OPS:
        partial = Partial(reduce_op=op)
        assert isinstance(partial, Partial)
        assert partial.reduce_op == op
        assert op in Partial.ALL_REDUCE_OPS

    dist.barrier()


class TestPartial(TestCase):
    """Test cases for Partial placement type."""

    def setUp(self):
        super().setUp()
        self.device_name = torch._C._get_privateuse1_backend_name()
        self.assertEqual(self.device_name, 'npu', f"Expected device 'npu', got '{self.device_name}'")

    @skipIfUnsupportMultiNPU(2)
    def test_partial_default(self):
        """Test Partial with default reduce_op."""
        world_size = 2
        mp.spawn(
            _init_dist_process,
            args=(world_size, _test_partial_default),
            nprocs=world_size,
            join=True
        )

    @skipIfUnsupportMultiNPU(2)
    def test_partial_sum(self):
        """Test Partial with 'sum' reduce_op."""
        world_size = 2
        mp.spawn(
            _init_dist_process,
            args=(world_size, _test_partial_sum),
            nprocs=world_size,
            join=True
        )

    @skipIfUnsupportMultiNPU(2)
    def test_partial_avg(self):
        """Test Partial with 'avg' reduce_op."""
        world_size = 2
        mp.spawn(
            _init_dist_process,
            args=(world_size, _test_partial_avg),
            nprocs=world_size,
            join=True
        )

    @skipIfUnsupportMultiNPU(2)
    def test_partial_min(self):
        """Test Partial with 'min' reduce_op."""
        world_size = 2
        mp.spawn(
            _init_dist_process,
            args=(world_size, _test_partial_min),
            nprocs=world_size,
            join=True
        )

    @skipIfUnsupportMultiNPU(2)
    def test_partial_max(self):
        """Test Partial with 'max' reduce_op."""
        world_size = 2
        mp.spawn(
            _init_dist_process,
            args=(world_size, _test_partial_max),
            nprocs=world_size,
            join=True
        )

    @skipIfUnsupportMultiNPU(2)
    def test_partial_product(self):
        """Test Partial with 'product' reduce_op."""
        world_size = 2
        mp.spawn(
            _init_dist_process,
            args=(world_size, _test_partial_product),
            nprocs=world_size,
            join=True
        )

    @skipIfUnsupportMultiNPU(2)
    def test_partial_linear_ops(self):
        """Test all linear reduce ops."""
        world_size = 2
        mp.spawn(
            _init_dist_process,
            args=(world_size, _test_partial_linear_ops),
            nprocs=world_size,
            join=True
        )

    @skipIfUnsupportMultiNPU(2)
    def test_partial_all_ops(self):
        """Test all supported reduce ops."""
        world_size = 2
        mp.spawn(
            _init_dist_process,
            args=(world_size, _test_partial_all_ops),
            nprocs=world_size,
            join=True
        )


if __name__ == "__main__":
    run_tests()
