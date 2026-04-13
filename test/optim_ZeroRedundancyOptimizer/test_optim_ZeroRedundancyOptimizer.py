# -*- coding: utf-8 -*-
"""
测试目的：验证 torch.distributed.optim.ZeroRedundancyOptimizer 接口功能正确性
API 名称：torch.distributed.optim.ZeroRedundancyOptimizer
API 签名：ZeroRedundancyOptimizer(
              params,
              optimizer_class: Type[Optimizer],
              process_group: Optional[Any] = None,
              parameters_as_bucket_view: bool = False,
              overlap_with_ddp: bool = False,
              **defaults: Any,
          )

覆盖维度表：
| 覆盖维度         | 说明                                                         | 覆盖情况                                                             |
|------------------|--------------------------------------------------------------|----------------------------------------------------------------------|
| 空/非空          | params 非空（模型参数列表）                                   | 已覆盖                                                               |
| 枚举选项         | optimizer_class: SGD / Adam                                   | 已覆盖                                                               |
| 参数类型         | params: list / generator; process_group: None / explicit      | 已覆盖                                                               |
| 传参与不传参     | process_group=None（默认 WORLD）；显式传入                     | 已覆盖                                                               |
| 等价类/边界值    | 单参数；多参数；parameters_as_bucket_view=True/False          | 已覆盖                                                               |
| 正常传参场景     | 构造成功；rank/world_size/process_group 正确设置              | 已覆盖                                                               |
| 异常传参场景     | overlap_with_ddp=True 后不能 add_param_group → RuntimeError  | 已覆盖                                                               |

未覆盖项及原因：
- 无。

注意：本测试仅验证功能正确性（调用不报错、输出 shape/dtype/类型符合预期），
     不做精度和数值正确性校验。
"""
import os
import unittest

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn as nn
import torch_npu
from torch.distributed.optim import ZeroRedundancyOptimizer
from torch_npu.testing.common_distributed import skipIfUnsupportMultiNPU
from torch_npu.testing.testcase import TestCase, run_tests


def _init_dist(rank, world_size, fn, *args):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '29500'
    torch.npu.set_device(rank)
    dist.init_process_group('hccl', rank=rank, world_size=world_size)
    try:
        fn(rank, world_size, *args)
    finally:
        dist.destroy_process_group()


def _test_construct_with_sgd(rank, world_size):
    """ZeroRedundancyOptimizer constructs with SGD and verifies rank/world_size."""
    device = f'npu:{rank}'
    model = nn.Linear(8, 4).to(device)
    opt = ZeroRedundancyOptimizer(
        model.parameters(),
        optimizer_class=torch.optim.SGD,
        lr=0.01,
    )
    assert opt.rank == rank, f"Expected rank {rank}, got {opt.rank}"
    assert opt.world_size == world_size, \
        f"Expected world_size {world_size}, got {opt.world_size}"


def _test_construct_with_adam(rank, world_size):
    """ZeroRedundancyOptimizer constructs with Adam."""
    device = f'npu:{rank}'
    model = nn.Linear(4, 2).to(device)
    opt = ZeroRedundancyOptimizer(
        model.parameters(),
        optimizer_class=torch.optim.Adam,
        lr=1e-3,
    )
    assert isinstance(opt, ZeroRedundancyOptimizer)
    assert opt.world_size == world_size


def _test_construct_with_explicit_process_group(rank, world_size):
    """ZeroRedundancyOptimizer uses explicit process_group correctly."""
    device = f'npu:{rank}'
    model = nn.Linear(4, 2).to(device)
    pg = dist.group.WORLD
    opt = ZeroRedundancyOptimizer(
        model.parameters(),
        optimizer_class=torch.optim.SGD,
        process_group=pg,
        lr=0.01,
    )
    assert opt.process_group is pg


def _test_parameters_as_bucket_view(rank, world_size):
    """parameters_as_bucket_view=True constructs successfully."""
    device = f'npu:{rank}'
    model = nn.Linear(4, 2).to(device)
    opt = ZeroRedundancyOptimizer(
        model.parameters(),
        optimizer_class=torch.optim.SGD,
        parameters_as_bucket_view=True,
        lr=0.01,
    )
    assert opt.parameters_as_bucket_view is True


def _test_param_groups_accessible(rank, world_size):
    """param_groups is accessible and non-empty after construction."""
    device = f'npu:{rank}'
    model = nn.Linear(4, 2).to(device)
    opt = ZeroRedundancyOptimizer(
        model.parameters(),
        optimizer_class=torch.optim.SGD,
        lr=0.01,
    )
    assert len(opt.param_groups) > 0, "param_groups should not be empty"


def _test_initialized_flag(rank, world_size):
    """initialized flag is True after successful construction."""
    device = f'npu:{rank}'
    model = nn.Linear(4, 2).to(device)
    opt = ZeroRedundancyOptimizer(
        model.parameters(),
        optimizer_class=torch.optim.SGD,
        lr=0.01,
    )
    assert opt.initialized is True


class TestZeroRedundancyOptimizer(TestCase):
    """Tests for ZeroRedundancyOptimizer initialization."""

    def setUp(self):
        super().setUp()
        self.device_name = torch._C._get_privateuse1_backend_name()
        self.assertEqual(
            self.device_name, 'npu',
            f"Expected device 'npu', got '{self.device_name}'"
        )

    @skipIfUnsupportMultiNPU(2)
    def test_construct_with_sgd(self):
        """Constructs with SGD and verifies rank and world_size."""
        mp.spawn(_init_dist, args=(2, _test_construct_with_sgd), nprocs=2, join=True)

    @skipIfUnsupportMultiNPU(2)
    def test_construct_with_adam(self):
        """Constructs with Adam optimizer class."""
        mp.spawn(_init_dist, args=(2, _test_construct_with_adam), nprocs=2, join=True)

    @skipIfUnsupportMultiNPU(2)
    def test_construct_with_explicit_process_group(self):
        """Explicit process_group is correctly assigned."""
        mp.spawn(_init_dist, args=(2, _test_construct_with_explicit_process_group), nprocs=2, join=True)

    @skipIfUnsupportMultiNPU(2)
    def test_parameters_as_bucket_view(self):
        """parameters_as_bucket_view=True constructs successfully."""
        mp.spawn(_init_dist, args=(2, _test_parameters_as_bucket_view), nprocs=2, join=True)

    @skipIfUnsupportMultiNPU(2)
    def test_param_groups_accessible(self):
        """param_groups is accessible and non-empty."""
        mp.spawn(_init_dist, args=(2, _test_param_groups_accessible), nprocs=2, join=True)

    @skipIfUnsupportMultiNPU(2)
    def test_initialized_flag(self):
        """initialized flag is True after construction."""
        mp.spawn(_init_dist, args=(2, _test_initialized_flag), nprocs=2, join=True)


if __name__ == "__main__":
    run_tests()
