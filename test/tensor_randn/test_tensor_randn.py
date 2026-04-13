# -*- coding: utf-8 -*-
"""
测试目的：验证 torch.distributed.tensor.randn 接口功能正确性
API 名称：torch.distributed.tensor.randn
API 签名：randn(
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
| 参数类型         | dtype: None(默认) / float32 / float16                        | 已覆盖                                                            |
| 传参与不传参     | requires_grad=True/False；dtype 省略 vs 显式                  | 已覆盖                                                            |
| 等价类/边界值    | 1D/2D/3D；varargs/list；均匀分片                              | 已覆盖                                                            |
| 正常传参场景     | 返回 DTensor；shape/dtype/device 正确                         | 已覆盖                                                            |
| 异常传参场景     | 无稳定异常路径                                               | 未覆盖，原因：合法参数下无异常路径                                 |

未覆盖项及原因：
- 异常传参：无稳定路径。

注意：本测试仅验证功能正确性（调用不报错、输出 shape/dtype/类型符合预期），
     不做精度和数值正确性校验。randn 生成随机数值，不校验数值结果。
"""
import os
import unittest

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch_npu
from torch.distributed.device_mesh import DeviceMesh
from torch.distributed.tensor import DTensor, randn
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


def _test_randn_replicate(rank, world_size):
    device_type = torch._C._get_privateuse1_backend_name()
    mesh = DeviceMesh(device_type, list(range(world_size)))
    t = randn(4, 8, device_mesh=mesh, placements=[Replicate()])
    assert isinstance(t, DTensor)
    assert t.shape == torch.Size([4, 8])
    assert t.device.type == device_type


def _test_randn_shard0(rank, world_size):
    device_type = torch._C._get_privateuse1_backend_name()
    mesh = DeviceMesh(device_type, list(range(world_size)))
    t = randn(4, 8, device_mesh=mesh, placements=[Shard(0)])
    assert isinstance(t, DTensor)
    assert t.shape == torch.Size([4, 8])


def _test_randn_shard1(rank, world_size):
    device_type = torch._C._get_privateuse1_backend_name()
    mesh = DeviceMesh(device_type, list(range(world_size)))
    t = randn(4, 8, device_mesh=mesh, placements=[Shard(1)])
    assert isinstance(t, DTensor)
    assert t.shape == torch.Size([4, 8])
    local = t.to_local()
    assert local.shape[0] == 4


def _test_randn_dtype_float16(rank, world_size):
    device_type = torch._C._get_privateuse1_backend_name()
    mesh = DeviceMesh(device_type, list(range(world_size)))
    t = randn(4, 4, dtype=torch.float16, device_mesh=mesh, placements=[Replicate()])
    assert t.dtype == torch.float16


def _test_randn_requires_grad(rank, world_size):
    device_type = torch._C._get_privateuse1_backend_name()
    mesh = DeviceMesh(device_type, list(range(world_size)))
    t = randn(4, 4, requires_grad=True, device_mesh=mesh, placements=[Replicate()])
    assert t.requires_grad is True


def _test_randn_list_size(rank, world_size):
    device_type = torch._C._get_privateuse1_backend_name()
    mesh = DeviceMesh(device_type, list(range(world_size)))
    t = randn([4, 8, 2], device_mesh=mesh, placements=[Replicate()])
    assert t.shape == torch.Size([4, 8, 2])


def _test_randn_1d(rank, world_size):
    device_type = torch._C._get_privateuse1_backend_name()
    mesh = DeviceMesh(device_type, list(range(world_size)))
    t = randn(16, device_mesh=mesh, placements=[Replicate()])
    assert t.shape == torch.Size([16])
    assert isinstance(t, DTensor)


def _test_randn_local_tensor_on_npu(rank, world_size):
    """Local tensor of randn DTensor is on the correct NPU device."""
    device_type = torch._C._get_privateuse1_backend_name()
    mesh = DeviceMesh(device_type, list(range(world_size)))
    t = randn(4, 4, device_mesh=mesh, placements=[Replicate()])
    local = t.to_local()
    assert local.device.type == device_type


class TestDTensorRandn(TestCase):
    """Tests for torch.distributed.tensor.randn."""

    def setUp(self):
        super().setUp()
        self.device_name = torch._C._get_privateuse1_backend_name()
        self.assertEqual(
            self.device_name, 'npu',
            f"Expected device 'npu', got '{self.device_name}'"
        )

    @skipIfUnsupportMultiNPU(2)
    def test_randn_replicate(self):
        """randn with Replicate returns DTensor on NPU with correct shape."""
        mp.spawn(_init_dist, args=(2, _test_randn_replicate), nprocs=2, join=True)

    @skipIfUnsupportMultiNPU(2)
    def test_randn_shard0(self):
        """randn with Shard(0) returns DTensor with correct global shape."""
        mp.spawn(_init_dist, args=(2, _test_randn_shard0), nprocs=2, join=True)

    @skipIfUnsupportMultiNPU(2)
    def test_randn_shard1(self):
        """randn with Shard(1) splits along dim 1."""
        mp.spawn(_init_dist, args=(2, _test_randn_shard1), nprocs=2, join=True)

    @skipIfUnsupportMultiNPU(2)
    def test_randn_dtype_float16(self):
        """randn with dtype=float16 returns float16 DTensor."""
        mp.spawn(_init_dist, args=(2, _test_randn_dtype_float16), nprocs=2, join=True)

    @skipIfUnsupportMultiNPU(2)
    def test_randn_requires_grad(self):
        """randn with requires_grad=True returns differentiable DTensor."""
        mp.spawn(_init_dist, args=(2, _test_randn_requires_grad), nprocs=2, join=True)

    @skipIfUnsupportMultiNPU(2)
    def test_randn_list_size(self):
        """randn accepts size as a list."""
        mp.spawn(_init_dist, args=(2, _test_randn_list_size), nprocs=2, join=True)

    @skipIfUnsupportMultiNPU(2)
    def test_randn_1d(self):
        """randn creates a 1D DTensor correctly."""
        mp.spawn(_init_dist, args=(2, _test_randn_1d), nprocs=2, join=True)

    @skipIfUnsupportMultiNPU(2)
    def test_randn_local_tensor_on_npu(self):
        """Local tensor of randn DTensor is on the correct NPU device."""
        mp.spawn(_init_dist, args=(2, _test_randn_local_tensor_on_npu), nprocs=2, join=True)


if __name__ == "__main__":
    run_tests()
