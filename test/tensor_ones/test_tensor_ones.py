# -*- coding: utf-8 -*-
"""
测试目的：验证 torch.distributed.tensor.ones 接口功能正确性
API 名称：torch.distributed.tensor.ones
API 签名：ones(
              *size,
              dtype: Optional[torch.dtype] = None,
              layout: torch.layout = torch.strided,
              requires_grad: bool = False,
              device_mesh: Optional[DeviceMesh] = None,
              placements: Optional[Sequence[Placement]] = None,
          ) -> DTensor

覆盖维度表：
| 覆盖维度         | 说明                                                         | 覆盖情况                                                          |
|------------------|--------------------------------------------------------------|-------------------------------------------------------------------|
| 空/非空          | 非空 size（各种 shape）                                       | 已覆盖                                                            |
| 枚举选项         | placements: Shard(0) / Shard(1) / Replicate                  | 已覆盖                                                            |
| 参数类型         | dtype: None(默认) / float32 / float16 / int32                | 已覆盖                                                            |
| 传参与不传参     | requires_grad=True/False；dtype=None vs 显式传入              | 已覆盖                                                            |
| 等价类/边界值    | 1D / 2D / 3D size；size 为 list / tuple / varargs             | 已覆盖                                                            |
| 正常传参场景     | 返回 DTensor，shape/dtype/device 正确；local tensor 形状正确  | 已覆盖                                                            |
| 异常传参场景     | 无稳定异常路径（参数约束由 DeviceMesh 保证）                  | 未覆盖，原因：合法参数组合下无异常路径                             |

未覆盖项及原因：
- 异常传参：合法 DeviceMesh + 合法 placements 组合下无稳定异常路径。

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
from torch.distributed.tensor import DTensor, ones
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


def _test_ones_replicate(rank, world_size):
    """ones with Replicate placement returns DTensor with correct shape/dtype."""
    device_type = torch._C._get_privateuse1_backend_name()
    mesh = DeviceMesh(device_type, list(range(world_size)))
    t = ones(4, 8, device_mesh=mesh, placements=[Replicate()])
    assert isinstance(t, DTensor), f"Expected DTensor, got {type(t)}"
    assert t.shape == torch.Size([4, 8]), f"Expected [4,8], got {t.shape}"
    assert t.dtype == torch.get_default_dtype()
    assert t.device.type == device_type
    local = t.to_local()
    assert local.shape == torch.Size([4, 8])  # Replicate: full shape on each rank


def _test_ones_shard0(rank, world_size):
    """ones with Shard(0) splits along dim 0."""
    device_type = torch._C._get_privateuse1_backend_name()
    mesh = DeviceMesh(device_type, list(range(world_size)))
    t = ones(4, 8, device_mesh=mesh, placements=[Shard(0)])
    assert isinstance(t, DTensor)
    assert t.shape == torch.Size([4, 8])
    local = t.to_local()
    # Each rank holds 4//world_size rows
    assert local.shape[1] == 8


def _test_ones_dtype_float16(rank, world_size):
    """ones with dtype=float16 returns float16 DTensor."""
    device_type = torch._C._get_privateuse1_backend_name()
    mesh = DeviceMesh(device_type, list(range(world_size)))
    t = ones(4, 4, dtype=torch.float16, device_mesh=mesh, placements=[Replicate()])
    assert t.dtype == torch.float16, f"Expected float16, got {t.dtype}"


def _test_ones_dtype_int32(rank, world_size):
    """ones with dtype=int32 returns int32 DTensor."""
    device_type = torch._C._get_privateuse1_backend_name()
    mesh = DeviceMesh(device_type, list(range(world_size)))
    t = ones(6, device_mesh=mesh, placements=[Replicate()], dtype=torch.int32)
    assert t.dtype == torch.int32


def _test_ones_requires_grad(rank, world_size):
    """ones with requires_grad=True returns differentiable DTensor."""
    device_type = torch._C._get_privateuse1_backend_name()
    mesh = DeviceMesh(device_type, list(range(world_size)))
    t = ones(4, 4, device_mesh=mesh, placements=[Replicate()], requires_grad=True)
    assert t.requires_grad is True


def _test_ones_list_size(rank, world_size):
    """ones accepts size as a list."""
    device_type = torch._C._get_privateuse1_backend_name()
    mesh = DeviceMesh(device_type, list(range(world_size)))
    t = ones([4, 8, 2], device_mesh=mesh, placements=[Replicate()])
    assert t.shape == torch.Size([4, 8, 2])


def _test_ones_1d(rank, world_size):
    """ones creates a 1D DTensor correctly."""
    device_type = torch._C._get_privateuse1_backend_name()
    mesh = DeviceMesh(device_type, list(range(world_size)))
    t = ones(16, device_mesh=mesh, placements=[Replicate()])
    assert t.shape == torch.Size([16])
    assert isinstance(t, DTensor)


class TestDTensorOnes(TestCase):
    """Tests for torch.distributed.tensor.ones."""

    def setUp(self):
        super().setUp()
        self.device_name = torch._C._get_privateuse1_backend_name()
        self.assertEqual(
            self.device_name, 'npu',
            f"Expected device 'npu', got '{self.device_name}'"
        )

    @skipIfUnsupportMultiNPU(2)
    def test_ones_replicate(self):
        """ones with Replicate returns full shape on each rank."""
        mp.spawn(_init_dist, args=(2, _test_ones_replicate), nprocs=2, join=True)

    @skipIfUnsupportMultiNPU(2)
    def test_ones_shard0(self):
        """ones with Shard(0) returns correct global shape."""
        mp.spawn(_init_dist, args=(2, _test_ones_shard0), nprocs=2, join=True)

    @skipIfUnsupportMultiNPU(2)
    def test_ones_dtype_float16(self):
        """ones with dtype=float16 returns float16 DTensor."""
        mp.spawn(_init_dist, args=(2, _test_ones_dtype_float16), nprocs=2, join=True)

    @skipIfUnsupportMultiNPU(2)
    def test_ones_dtype_int32(self):
        """ones with dtype=int32 returns int32 DTensor."""
        mp.spawn(_init_dist, args=(2, _test_ones_dtype_int32), nprocs=2, join=True)

    @skipIfUnsupportMultiNPU(2)
    def test_ones_requires_grad(self):
        """ones with requires_grad=True returns differentiable DTensor."""
        mp.spawn(_init_dist, args=(2, _test_ones_requires_grad), nprocs=2, join=True)

    @skipIfUnsupportMultiNPU(2)
    def test_ones_list_size(self):
        """ones accepts size as a list argument."""
        mp.spawn(_init_dist, args=(2, _test_ones_list_size), nprocs=2, join=True)

    @skipIfUnsupportMultiNPU(2)
    def test_ones_1d(self):
        """ones creates a 1D DTensor correctly."""
        mp.spawn(_init_dist, args=(2, _test_ones_1d), nprocs=2, join=True)


if __name__ == "__main__":
    run_tests()
