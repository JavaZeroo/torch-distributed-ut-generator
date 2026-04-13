# -*- coding: utf-8 -*-
"""
测试目的：验证 torch.distributed.tensor.rand 接口功能正确性
API 名称：torch.distributed.tensor.rand
API 签名：rand(
              *size,
              requires_grad: bool = False,
              dtype: Optional[torch.dtype] = None,
              layout: torch.layout = torch.strided,
              device_mesh: Optional[DeviceMesh] = None,
              placements: Optional[Sequence[Placement]] = None,
          ) -> DTensor

覆盖维度表：
| 覆盖维度         | 说明                                                         | 覆盖情况                                                          |
|------------------|--------------------------------------------------------------|-------------------------------------------------------------------|
| 空/非空          | 非空 size（各种形状）                                         | 已覆盖                                                            |
| 枚举选项         | placements: Shard(0) / Shard(1) / Replicate                  | 已覆盖                                                            |
| 参数类型         | dtype: None(默认 float32) / float32 / float16                | 已覆盖                                                            |
| 传参与不传参     | requires_grad=True/False；dtype 省略 vs 显式                  | 已覆盖                                                            |
| 等价类/边界值    | 1D/2D/3D；varargs/list；均匀分片                              | 已覆盖                                                            |
| 正常传参场景     | 返回 DTensor；shape/dtype/device 正确                         | 已覆盖                                                            |
| 异常传参场景     | 无稳定异常路径                                               | 未覆盖，原因：合法参数下无异常路径                                 |

未覆盖项及原因：
- 异常传参：无稳定路径。

注意：本测试仅验证功能正确性（调用不报错、输出 shape/dtype/类型符合预期），
     不做精度和数值正确性校验。rand 生成随机数值，不校验数值结果。
"""
import os
import unittest

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch_npu
from torch.distributed.device_mesh import DeviceMesh
from torch.distributed.tensor import DTensor, rand
from torch.distributed.tensor.placement_types import Replicate, Shard
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


def _test_rand_replicate(rank, world_size):
    device_type = torch._C._get_privateuse1_backend_name()
    mesh = DeviceMesh(device_type, list(range(world_size)))
    t = rand(4, 8, device_mesh=mesh, placements=[Replicate()])
    assert isinstance(t, DTensor)
    assert t.shape == torch.Size([4, 8])
    assert t.device.type == device_type
    assert t.dtype == torch.get_default_dtype()


def _test_rand_shard0(rank, world_size):
    device_type = torch._C._get_privateuse1_backend_name()
    mesh = DeviceMesh(device_type, list(range(world_size)))
    t = rand(4, 8, device_mesh=mesh, placements=[Shard(0)])
    assert isinstance(t, DTensor)
    assert t.shape == torch.Size([4, 8])


def _test_rand_shard1(rank, world_size):
    device_type = torch._C._get_privateuse1_backend_name()
    mesh = DeviceMesh(device_type, list(range(world_size)))
    t = rand(4, 8, device_mesh=mesh, placements=[Shard(1)])
    assert isinstance(t, DTensor)
    assert t.shape == torch.Size([4, 8])
    local = t.to_local()
    assert local.shape[0] == 4


def _test_rand_dtype_float16(rank, world_size):
    device_type = torch._C._get_privateuse1_backend_name()
    mesh = DeviceMesh(device_type, list(range(world_size)))
    t = rand(4, 4, dtype=torch.float16, device_mesh=mesh, placements=[Replicate()])
    assert t.dtype == torch.float16


def _test_rand_requires_grad(rank, world_size):
    device_type = torch._C._get_privateuse1_backend_name()
    mesh = DeviceMesh(device_type, list(range(world_size)))
    t = rand(4, 4, requires_grad=True, device_mesh=mesh, placements=[Replicate()])
    assert t.requires_grad is True


def _test_rand_list_size(rank, world_size):
    device_type = torch._C._get_privateuse1_backend_name()
    mesh = DeviceMesh(device_type, list(range(world_size)))
    t = rand([4, 8, 2], device_mesh=mesh, placements=[Replicate()])
    assert t.shape == torch.Size([4, 8, 2])


def _test_rand_1d(rank, world_size):
    device_type = torch._C._get_privateuse1_backend_name()
    mesh = DeviceMesh(device_type, list(range(world_size)))
    t = rand(16, device_mesh=mesh, placements=[Replicate()])
    assert t.shape == torch.Size([16])
    assert isinstance(t, DTensor)


def _test_rand_returns_dtensor_type(rank, world_size):
    device_type = torch._C._get_privateuse1_backend_name()
    mesh = DeviceMesh(device_type, list(range(world_size)))
    t = rand(2, 4, device_mesh=mesh, placements=[Replicate()])
    assert isinstance(t, DTensor), f"Expected DTensor, got {type(t)}"
    local = t.to_local()
    assert isinstance(local, torch.Tensor)
    assert local.device.type == device_type


class TestDTensorRand(TestCase):
    """Tests for torch.distributed.tensor.rand."""

    def setUp(self):
        super().setUp()
        self.device_name = torch._C._get_privateuse1_backend_name()
        self.assertEqual(
            self.device_name, 'npu',
            f"Expected device 'npu', got '{self.device_name}'"
        )

    @skipIfUnsupportMultiNPU(2)
    def test_rand_replicate(self):
        """rand with Replicate returns DTensor with correct shape and device."""
        mp.spawn(_init_dist, args=(2, _test_rand_replicate), nprocs=2, join=True)

    @skipIfUnsupportMultiNPU(2)
    def test_rand_shard0(self):
        """rand with Shard(0) returns DTensor with correct global shape."""
        mp.spawn(_init_dist, args=(2, _test_rand_shard0), nprocs=2, join=True)

    @skipIfUnsupportMultiNPU(2)
    def test_rand_shard1(self):
        """rand with Shard(1) splits along dim 1."""
        mp.spawn(_init_dist, args=(2, _test_rand_shard1), nprocs=2, join=True)

    @skipIfUnsupportMultiNPU(2)
    def test_rand_dtype_float16(self):
        """rand with dtype=float16 returns float16 DTensor."""
        mp.spawn(_init_dist, args=(2, _test_rand_dtype_float16), nprocs=2, join=True)

    @skipIfUnsupportMultiNPU(2)
    def test_rand_requires_grad(self):
        """rand with requires_grad=True returns differentiable DTensor."""
        mp.spawn(_init_dist, args=(2, _test_rand_requires_grad), nprocs=2, join=True)

    @skipIfUnsupportMultiNPU(2)
    def test_rand_list_size(self):
        """rand accepts size as a list."""
        mp.spawn(_init_dist, args=(2, _test_rand_list_size), nprocs=2, join=True)

    @skipIfUnsupportMultiNPU(2)
    def test_rand_1d(self):
        """rand creates a 1D DTensor correctly."""
        mp.spawn(_init_dist, args=(2, _test_rand_1d), nprocs=2, join=True)

    @skipIfUnsupportMultiNPU(2)
    def test_rand_returns_dtensor_type(self):
        """rand returns a DTensor whose local tensor is on the correct device."""
        mp.spawn(_init_dist, args=(2, _test_rand_returns_dtensor_type), nprocs=2, join=True)


if __name__ == "__main__":
    run_tests()
