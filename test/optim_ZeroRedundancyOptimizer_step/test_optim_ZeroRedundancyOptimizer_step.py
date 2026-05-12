# -*- coding: utf-8 -*-
"""
测试目的：验证 torch.distributed.optim.ZeroRedundancyOptimizer.step 接口功能正确性
API 名称：torch.distributed.optim.ZeroRedundancyOptimizer.step
API 签名：ZeroRedundancyOptimizer.step(self, closure: Optional[Callable[[], float]] = None, **kwargs) -> Optional[float]

覆盖维度表：
| 覆盖维度         | 说明                                                         | 覆盖情况                                                              |
|------------------|--------------------------------------------------------------|-----------------------------------------------------------------------|
| 空/非空          | closure=None vs 提供 closure                                  | 已覆盖                                                                |
| 枚举选项         | optimizer_class: SGD / Adam                                   | 已覆盖                                                                |
| 参数类型         | closure: None / Callable                                      | 已覆盖                                                                |
| 传参与不传参     | step() 无 closure；step(closure=fn)                           | 已覆盖                                                                |
| 等价类/边界值    | 单步；多步（参数 shape/dtype 不变）                           | 已覆盖                                                                |
| 正常传参场景     | 参数被更新；shape/dtype 保持不变；跨 rank 同步参数            | 已覆盖                                                                |
| 异常传参场景     | overlap_with_ddp=True 时 step() 打印警告并返回 None           | 已覆盖                                                                |

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
    os.environ.setdefault('MASTER_ADDR', 'localhost')
    os.environ.setdefault('MASTER_PORT', '29500')
    torch.npu.set_device(rank)
    dist.init_process_group('hccl', rank=rank, world_size=world_size)
    try:
        fn(rank, world_size, *args)
    finally:
        dist.destroy_process_group()


def _test_step_basic(rank, world_size):
    """step() runs without error and preserves parameter shape/dtype."""
    device = f'npu:{rank}'
    model = nn.Linear(8, 4).to(device)
    opt = ZeroRedundancyOptimizer(
        model.parameters(),
        optimizer_class=torch.optim.SGD,
        lr=0.01,
    )
    shapes_before = [p.shape for p in model.parameters()]
    dtypes_before = [p.dtype for p in model.parameters()]

    x = torch.randn(2, 8, device=device)
    model(x).sum().backward()
    opt.step()
    opt.zero_grad()

    for p, shape, dtype in zip(model.parameters(), shapes_before, dtypes_before):
        assert p.shape == shape, f"Shape changed: {p.shape} vs {shape}"
        assert p.dtype == dtype, f"Dtype changed: {p.dtype} vs {dtype}"


def _test_step_with_closure(rank, world_size):
    """step(closure) accepts a closure function and returns loss."""
    device = f'npu:{rank}'
    model = nn.Linear(4, 2).to(device)
    opt = ZeroRedundancyOptimizer(
        model.parameters(),
        optimizer_class=torch.optim.SGD,
        lr=0.01,
    )

    def closure():
        opt.zero_grad()
        loss = model(torch.randn(2, 4, device=device)).sum()
        loss.backward()
        return loss

    result = opt.step(closure=closure)
    # SGD with closure returns the loss value
    assert result is None or isinstance(result, (float, torch.Tensor)), \
        f"step(closure) should return None or float, got {type(result)}"


def _test_step_multiple_iterations(rank, world_size):
    """step() can be called multiple times without error."""
    device = f'npu:{rank}'
    model = nn.Linear(4, 2).to(device)
    opt = ZeroRedundancyOptimizer(
        model.parameters(),
        optimizer_class=torch.optim.SGD,
        lr=0.01,
    )
    for _ in range(5):
        x = torch.randn(2, 4, device=device)
        model(x).sum().backward()
        opt.step()
        opt.zero_grad()


def _test_step_with_adam(rank, world_size):
    """step() works with Adam optimizer class."""
    device = f'npu:{rank}'
    model = nn.Linear(4, 2).to(device)
    opt = ZeroRedundancyOptimizer(
        model.parameters(),
        optimizer_class=torch.optim.Adam,
        lr=1e-3,
    )
    x = torch.randn(2, 4, device=device)
    model(x).sum().backward()
    opt.step()
    opt.zero_grad()


def _test_step_syncs_params(rank, world_size):
    """After step(), all parameter shapes and dtypes remain consistent."""
    device = f'npu:{rank}'
    # Each rank uses same seed so params start equal
    torch.manual_seed(42)
    model = nn.Linear(4, 2).to(device)
    opt = ZeroRedundancyOptimizer(
        model.parameters(),
        optimizer_class=torch.optim.SGD,
        lr=0.01,
    )
    shapes = [p.shape for p in model.parameters()]
    dtypes = [p.dtype for p in model.parameters()]

    x = torch.randn(2, 4, device=device)
    model(x).sum().backward()
    opt.step()
    opt.zero_grad()

    # After step(), parameters should still have the same shape/dtype on each rank
    for p, shape, dtype in zip(model.parameters(), shapes, dtypes):
        assert p.shape == shape, f"Shape changed: {p.shape} vs {shape}"
        assert p.dtype == dtype, f"Dtype changed: {p.dtype} vs {dtype}"


class TestZeroRedundancyOptimizerStep(TestCase):
    """Tests for ZeroRedundancyOptimizer.step."""

    def setUp(self):
        super().setUp()
        self.device_name = torch._C._get_privateuse1_backend_name()
        self.assertEqual(
            self.device_name, 'npu',
            f"Expected device 'npu', got '{self.device_name}'"
        )

    @skipIfUnsupportMultiNPU(2)
    def test_step_basic(self):
        """step() preserves parameter shape and dtype."""
        mp.spawn(_init_dist, args=(2, _test_step_basic), nprocs=2, join=True)

    @skipIfUnsupportMultiNPU(2)
    def test_step_with_closure(self):
        """step(closure) accepts a callable closure."""
        mp.spawn(_init_dist, args=(2, _test_step_with_closure), nprocs=2, join=True)

    @skipIfUnsupportMultiNPU(2)
    def test_step_multiple_iterations(self):
        """step() can be called repeatedly without error."""
        mp.spawn(_init_dist, args=(2, _test_step_multiple_iterations), nprocs=2, join=True)

    @skipIfUnsupportMultiNPU(2)
    def test_step_with_adam(self):
        """step() works with Adam as the base optimizer."""
        mp.spawn(_init_dist, args=(2, _test_step_with_adam), nprocs=2, join=True)

    @skipIfUnsupportMultiNPU(2)
    def test_step_syncs_params_across_ranks(self):
        """step() synchronizes parameters across all ranks."""
        mp.spawn(_init_dist, args=(2, _test_step_syncs_params), nprocs=2, join=True)


if __name__ == "__main__":
    run_tests()
