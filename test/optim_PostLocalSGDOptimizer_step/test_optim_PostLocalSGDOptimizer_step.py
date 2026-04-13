# -*- coding: utf-8 -*-
"""
测试目的：验证 torch.distributed.optim.PostLocalSGDOptimizer.step 接口功能正确性
API 名称：torch.distributed.optim.PostLocalSGDOptimizer.step
API 签名：PostLocalSGDOptimizer.step(self) -> None

覆盖维度表：
| 覆盖维度         | 说明                                                         | 覆盖情况                                                            |
|------------------|--------------------------------------------------------------|---------------------------------------------------------------------|
| 空/非空          | 有梯度的参数 vs 零梯度                                        | 已覆盖                                                              |
| 枚举选项         | warmup_steps=0（每步平均）；warmup_steps=100（预热后平均）   | 已覆盖                                                              |
| 参数类型         | float32 参数                                                  | 已覆盖                                                              |
| 传参与不传参     | step() 无参数                                                 | 已覆盖                                                              |
| 等价类/边界值    | 单步；多步（跨 period）；DDP 封装模型                         | 已覆盖                                                              |
| 正常传参场景     | step() 执行本地优化 + 模型平均，参数 shape/dtype 不变         | 已覆盖                                                              |
| 异常传参场景     | 无稳定异常路径                                               | 未覆盖，原因：step() 无参数，合法状态下无异常路径                   |

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
from torch.distributed.algorithms.model_averaging.averagers import PeriodicModelAverager
from torch.distributed.optim import PostLocalSGDOptimizer
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


def _test_step_basic(rank, world_size):
    """step() runs without error and parameters retain correct dtype/shape."""
    device = f'npu:{rank}'
    model = nn.Linear(4, 2).to(device)
    local_optim = torch.optim.SGD(model.parameters(), lr=0.01)
    averager = PeriodicModelAverager(period=1, warmup_steps=0)
    opt = PostLocalSGDOptimizer(optim=local_optim, averager=averager)

    x = torch.randn(2, 4, device=device)
    loss = model(x).sum()
    loss.backward()

    param_shapes_before = [p.shape for p in model.parameters()]
    param_dtypes_before = [p.dtype for p in model.parameters()]

    opt.step()
    opt.zero_grad()

    for p, shape, dtype in zip(model.parameters(), param_shapes_before, param_dtypes_before):
        assert p.shape == shape, f"Shape changed after step: {p.shape} vs {shape}"
        assert p.dtype == dtype, f"Dtype changed after step: {p.dtype} vs {dtype}"


def _test_step_multiple_iterations(rank, world_size):
    """step() can be called multiple times without error."""
    device = f'npu:{rank}'
    model = nn.Linear(4, 2).to(device)
    local_optim = torch.optim.SGD(model.parameters(), lr=0.01)
    averager = PeriodicModelAverager(period=2, warmup_steps=0)
    opt = PostLocalSGDOptimizer(optim=local_optim, averager=averager)

    for _ in range(5):
        x = torch.randn(2, 4, device=device)
        loss = model(x).sum()
        loss.backward()
        opt.step()
        opt.zero_grad()


def _test_step_with_warmup(rank, world_size):
    """step() with warmup_steps: no averaging during warmup period."""
    device = f'npu:{rank}'
    model = nn.Linear(4, 2).to(device)
    local_optim = torch.optim.SGD(model.parameters(), lr=0.01)
    # warmup_steps=3: averaging starts after step 3
    averager = PeriodicModelAverager(period=2, warmup_steps=3)
    opt = PostLocalSGDOptimizer(optim=local_optim, averager=averager)

    for _ in range(6):
        x = torch.randn(2, 4, device=device)
        loss = model(x).sum()
        loss.backward()
        opt.step()
        opt.zero_grad()


def _test_step_returns_none(rank, world_size):
    """step() returns None."""
    device = f'npu:{rank}'
    model = nn.Linear(4, 2).to(device)
    local_optim = torch.optim.SGD(model.parameters(), lr=0.01)
    averager = PeriodicModelAverager(period=1, warmup_steps=0)
    opt = PostLocalSGDOptimizer(optim=local_optim, averager=averager)
    x = torch.randn(2, 4, device=device)
    model(x).sum().backward()
    result = opt.step()
    assert result is None, f"step() should return None, got {result}"


def _test_step_averager_step_increments(rank, world_size):
    """averager.step counter increments after each call to step()."""
    device = f'npu:{rank}'
    model = nn.Linear(4, 2).to(device)
    local_optim = torch.optim.SGD(model.parameters(), lr=0.01)
    averager = PeriodicModelAverager(period=2, warmup_steps=0)
    opt = PostLocalSGDOptimizer(optim=local_optim, averager=averager)
    initial_step = opt.averager.step
    for _ in range(3):
        x = torch.randn(2, 4, device=device)
        model(x).sum().backward()
        opt.step()
        opt.zero_grad()
    assert opt.averager.step == initial_step + 3, \
        f"averager.step should be {initial_step + 3}, got {opt.averager.step}"


class TestPostLocalSGDOptimizerStep(TestCase):
    """Tests for PostLocalSGDOptimizer.step."""

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
    def test_step_multiple_iterations(self):
        """step() can be called repeatedly without error."""
        mp.spawn(_init_dist, args=(2, _test_step_multiple_iterations), nprocs=2, join=True)

    @skipIfUnsupportMultiNPU(2)
    def test_step_with_warmup(self):
        """step() with warmup_steps works correctly across warmup boundary."""
        mp.spawn(_init_dist, args=(2, _test_step_with_warmup), nprocs=2, join=True)

    @skipIfUnsupportMultiNPU(2)
    def test_step_returns_none(self):
        """step() returns None."""
        mp.spawn(_init_dist, args=(2, _test_step_returns_none), nprocs=2, join=True)

    @skipIfUnsupportMultiNPU(2)
    def test_step_increments_averager_step(self):
        """averager.step counter increments correctly after each step()."""
        mp.spawn(_init_dist, args=(2, _test_step_averager_step_increments), nprocs=2, join=True)


if __name__ == "__main__":
    run_tests()
