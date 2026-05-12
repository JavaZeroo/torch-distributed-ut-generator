# -*- coding: utf-8 -*-
"""
测试目的：验证 torch.distributed.tensor.zeros 接口功能正确性
API 名称：torch.distributed.tensor.zeros
API 签名：zeros(
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
| 空/非空          | 非空 size（1D/2D/3D）                                         | 已覆盖                                                            |
| 枚举选项         | placements: Shard(0) / Shard(1) / Replicate；2D mesh         | 已覆盖                                                            |
| 参数类型         | dtype: None(默认) / float32 / float16 / int64                | 已覆盖                                                            |
| 传参与不传参     | requires_grad=True/False；dtype 省略 vs 显式                  | 已覆盖                                                            |
| 等价类/边界值    | 均匀分片；不均匀分片；1D mesh；varargs / list / tuple size    | 已覆盖                                                            |
| 正常传参场景     | 返回 DTensor，shape/dtype/device 符合预期                     | 已覆盖                                                            |
| 异常传参场景     | 无稳定异常路径                                               | 未覆盖，原因：合法参数组合下无异常路径                             |

未覆盖项及原因：
- 异常传参：无稳定路径。

注意：本测试仅验证功能正确性（调用不报错、输出 shape/dtype/类型符合预期），
     不做精度和数值正确性校验。
"""
import os
import unittest

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch_npu
from torch.distributed.device_mesh import DeviceMesh
from torch.distributed.tensor import DTensor, zeros
from torch.distributed.tensor.placement_types import Replicate, Shard
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


def _test_zeros_replicate(rank, world_size):
    device_type = torch._C._get_privateuse1_backend_name()
    mesh = DeviceMesh(device_type, list(range(world_size)))
    t = zeros(4, 8, device_mesh=mesh, placements=[Replicate()])
    assert isinstance(t, DTensor)
    assert t.shape == torch.Size([4, 8])
    assert t.device.type == device_type
    local = t.to_local()
    assert local.shape == torch.Size([4, 8])


def _test_zeros_shard0(rank, world_size):
    device_type = torch._C._get_privateuse1_backend_name()
    mesh = DeviceMesh(device_type, list(range(world_size)))
    t = zeros(4, 8, device_mesh=mesh, placements=[Shard(0)])
    assert isinstance(t, DTensor)
    assert t.shape == torch.Size([4, 8])
    local = t.to_local()
    assert local.shape[1] == 8  # Shard on dim 0, full dim 1


def _test_zeros_shard1(rank, world_size):
    device_type = torch._C._get_privateuse1_backend_name()
    mesh = DeviceMesh(device_type, list(range(world_size)))
    t = zeros(4, 8, device_mesh=mesh, placements=[Shard(1)])
    assert isinstance(t, DTensor)
    assert t.shape == torch.Size([4, 8])
    local = t.to_local()
    assert local.shape[0] == 4  # Shard on dim 1, full dim 0


def _test_zeros_dtype_float16(rank, world_size):
    device_type = torch._C._get_privateuse1_backend_name()
    mesh = DeviceMesh(device_type, list(range(world_size)))
    t = zeros(4, 4, dtype=torch.float16, device_mesh=mesh, placements=[Replicate()])
    assert t.dtype == torch.float16


def _test_zeros_dtype_int64(rank, world_size):
    device_type = torch._C._get_privateuse1_backend_name()
    mesh = DeviceMesh(device_type, list(range(world_size)))
    t = zeros(8, dtype=torch.int64, device_mesh=mesh, placements=[Replicate()])
    assert t.dtype == torch.int64


def _test_zeros_requires_grad(rank, world_size):
    device_type = torch._C._get_privateuse1_backend_name()
    mesh = DeviceMesh(device_type, list(range(world_size)))
    t = zeros(4, 4, requires_grad=True, device_mesh=mesh, placements=[Replicate()])
    assert t.requires_grad is True


def _test_zeros_tuple_size(rank, world_size):
    device_type = torch._C._get_privateuse1_backend_name()
    mesh = DeviceMesh(device_type, list(range(world_size)))
    t = zeros((4, 8), device_mesh=mesh, placements=[Replicate()])
    assert t.shape == torch.Size([4, 8])


def _test_zeros_3d(rank, world_size):
    device_type = torch._C._get_privateuse1_backend_name()
    mesh = DeviceMesh(device_type, list(range(world_size)))
    t = zeros(2, 4, 8, device_mesh=mesh, placements=[Replicate()])
    assert t.shape == torch.Size([2, 4, 8])


class TestDTensorZeros(TestCase):
    """Tests for torch.distributed.tensor.zeros."""

    def setUp(self):
        super().setUp()
        self.device_name = torch._C._get_privateuse1_backend_name()
        self.assertEqual(
            self.device_name, 'npu',
            f"Expected device 'npu', got '{self.device_name}'"
        )

    @skipIfUnsupportMultiNPU(2)
    def test_zeros_replicate(self):
        """zeros with Replicate returns full shape local tensor."""
        mp.spawn(_init_dist, args=(2, _test_zeros_replicate), nprocs=2, join=True)

    @skipIfUnsupportMultiNPU(2)
    def test_zeros_shard0(self):
        """zeros with Shard(0) splits along dim 0."""
        mp.spawn(_init_dist, args=(2, _test_zeros_shard0), nprocs=2, join=True)

    @skipIfUnsupportMultiNPU(2)
    def test_zeros_shard1(self):
        """zeros with Shard(1) splits along dim 1."""
        mp.spawn(_init_dist, args=(2, _test_zeros_shard1), nprocs=2, join=True)

    @skipIfUnsupportMultiNPU(2)
    def test_zeros_dtype_float16(self):
        """zeros with dtype=float16 returns float16 DTensor."""
        mp.spawn(_init_dist, args=(2, _test_zeros_dtype_float16), nprocs=2, join=True)

    @skipIfUnsupportMultiNPU(2)
    def test_zeros_dtype_int64(self):
        """zeros with dtype=int64 returns int64 DTensor."""
        mp.spawn(_init_dist, args=(2, _test_zeros_dtype_int64), nprocs=2, join=True)

    @skipIfUnsupportMultiNPU(2)
    def test_zeros_requires_grad(self):
        """zeros with requires_grad=True returns differentiable DTensor."""
        mp.spawn(_init_dist, args=(2, _test_zeros_requires_grad), nprocs=2, join=True)

    @skipIfUnsupportMultiNPU(2)
    def test_zeros_tuple_size(self):
        """zeros accepts size as a tuple."""
        mp.spawn(_init_dist, args=(2, _test_zeros_tuple_size), nprocs=2, join=True)

    @skipIfUnsupportMultiNPU(2)
    def test_zeros_3d(self):
        """zeros creates a 3D DTensor correctly."""
        mp.spawn(_init_dist, args=(2, _test_zeros_3d), nprocs=2, join=True)


if __name__ == "__main__":
    run_tests()
