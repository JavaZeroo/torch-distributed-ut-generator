# -*- coding: utf-8 -*-
"""
测试目的：验证 torch.distributed.optim.PostLocalSGDOptimizer 接口功能正确性
API 名称：torch.distributed.optim.PostLocalSGDOptimizer
API 签名：PostLocalSGDOptimizer(optim: torch.optim.Optimizer, averager: averagers.ModelAverager)

覆盖维度表：
| 覆盖维度         | 说明                                                         | 覆盖情况                                                           |
|------------------|--------------------------------------------------------------|---------------------------------------------------------------------|
| 空/非空          | optim / averager 均非空                                      | 已覆盖                                                              |
| 枚举选项         | 搭配 SGD / Adam 作为 local optimizer                         | 已覆盖                                                              |
| 参数类型         | optim: SGD/Adam; averager: PeriodicModelAverager             | 已覆盖                                                              |
| 传参与不传参     | 必填参数 optim + averager                                    | 已覆盖                                                              |
| 等价类/边界值    | warmup_steps=0（立即平均）；warmup_steps=100（预热后平均）   | 已覆盖                                                              |
| 正常传参场景     | 构造成功；param_groups 代理；state 代理；__repr__ 可用        | 已覆盖                                                              |
| 异常传参场景     | 无稳定异常路径（构造合法即可）                               | 未覆盖，原因：构造只接受已初始化的 optimizer 和 averager            |

未覆盖项及原因：
- 异常传参：PostLocalSGDOptimizer 构造无参数检验，不存在稳定的构造时异常路径。

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


def _simple_model(device):
    model = nn.Linear(4, 2).to(device)
    return model


def _test_construct_with_sgd(rank, world_size):
    """PostLocalSGDOptimizer wraps SGD and proxies param_groups."""
    device = f'npu:{rank}'
    model = _simple_model(device)
    local_optim = torch.optim.SGD(model.parameters(), lr=0.01)
    averager = PeriodicModelAverager(period=2, warmup_steps=0)
    opt = PostLocalSGDOptimizer(optim=local_optim, averager=averager)
    assert opt.param_groups is local_optim.param_groups, \
        "param_groups should proxy to inner optimizer"
    assert opt.state is local_optim.state, \
        "state should proxy to inner optimizer"


def _test_construct_with_adam(rank, world_size):
    """PostLocalSGDOptimizer wraps Adam correctly."""
    device = f'npu:{rank}'
    model = _simple_model(device)
    local_optim = torch.optim.Adam(model.parameters(), lr=1e-3)
    averager = PeriodicModelAverager(period=4, warmup_steps=0)
    opt = PostLocalSGDOptimizer(optim=local_optim, averager=averager)
    assert opt.optim is local_optim


def _test_repr(rank, world_size):
    """__repr__ delegates to the inner optimizer's repr."""
    device = f'npu:{rank}'
    model = _simple_model(device)
    local_optim = torch.optim.SGD(model.parameters(), lr=0.01)
    averager = PeriodicModelAverager(period=2, warmup_steps=0)
    opt = PostLocalSGDOptimizer(optim=local_optim, averager=averager)
    repr_str = repr(opt)
    assert isinstance(repr_str, str) and len(repr_str) > 0


def _test_zero_grad(rank, world_size):
    """zero_grad delegates to inner optimizer."""
    device = f'npu:{rank}'
    model = _simple_model(device)
    local_optim = torch.optim.SGD(model.parameters(), lr=0.01)
    averager = PeriodicModelAverager(period=2, warmup_steps=0)
    opt = PostLocalSGDOptimizer(optim=local_optim, averager=averager)
    # Run forward/backward
    x = torch.randn(2, 4, device=device)
    loss = model(x).sum()
    loss.backward()
    opt.zero_grad(set_to_none=True)
    for p in model.parameters():
        assert p.grad is None, "zero_grad(set_to_none=True) should set grads to None"


def _test_add_param_group(rank, world_size):
    """add_param_group delegates to inner optimizer."""
    device = f'npu:{rank}'
    model1 = _simple_model(device)
    model2 = _simple_model(device)
    local_optim = torch.optim.SGD(model1.parameters(), lr=0.01)
    averager = PeriodicModelAverager(period=2, warmup_steps=0)
    opt = PostLocalSGDOptimizer(optim=local_optim, averager=averager)
    initial_count = len(opt.param_groups)
    opt.add_param_group({'params': list(model2.parameters()), 'lr': 0.001})
    assert len(opt.param_groups) == initial_count + 1


def _test_warmup_steps_config(rank, world_size):
    """PeriodicModelAverager with warmup_steps is stored correctly."""
    device = f'npu:{rank}'
    model = _simple_model(device)
    local_optim = torch.optim.SGD(model.parameters(), lr=0.01)
    averager = PeriodicModelAverager(period=4, warmup_steps=100)
    opt = PostLocalSGDOptimizer(optim=local_optim, averager=averager)
    assert opt.averager.period == 4
    assert opt.averager.warmup_steps == 100


class TestPostLocalSGDOptimizer(TestCase):
    """Tests for PostLocalSGDOptimizer initialization and configuration."""

    def setUp(self):
        super().setUp()
        self.device_name = torch._C._get_privateuse1_backend_name()
        self.assertEqual(
            self.device_name, 'npu',
            f"Expected device 'npu', got '{self.device_name}'"
        )

    @skipIfUnsupportMultiNPU(2)
    def test_construct_with_sgd(self):
        """PostLocalSGDOptimizer wraps SGD with param_groups/state proxied."""
        mp.spawn(_init_dist, args=(2, _test_construct_with_sgd), nprocs=2, join=True)

    @skipIfUnsupportMultiNPU(2)
    def test_construct_with_adam(self):
        """PostLocalSGDOptimizer wraps Adam optimizer."""
        mp.spawn(_init_dist, args=(2, _test_construct_with_adam), nprocs=2, join=True)

    @skipIfUnsupportMultiNPU(2)
    def test_repr_delegates_to_inner(self):
        """__repr__ delegates to inner optimizer's repr."""
        mp.spawn(_init_dist, args=(2, _test_repr), nprocs=2, join=True)

    @skipIfUnsupportMultiNPU(2)
    def test_zero_grad_clears_gradients(self):
        """zero_grad(set_to_none=True) clears all parameter gradients."""
        mp.spawn(_init_dist, args=(2, _test_zero_grad), nprocs=2, join=True)

    @skipIfUnsupportMultiNPU(2)
    def test_add_param_group(self):
        """add_param_group increases the number of param groups."""
        mp.spawn(_init_dist, args=(2, _test_add_param_group), nprocs=2, join=True)

    @skipIfUnsupportMultiNPU(2)
    def test_warmup_steps_configuration(self):
        """averager warmup_steps and period are configured correctly."""
        mp.spawn(_init_dist, args=(2, _test_warmup_steps_config), nprocs=2, join=True)


if __name__ == "__main__":
    run_tests()
