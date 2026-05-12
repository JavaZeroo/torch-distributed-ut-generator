# -*- coding: utf-8 -*-
"""
测试目的：验证 torch.distributed.tensor.full 接口功能正确性
API 名称：torch.distributed.tensor.full
API 签名：full(
              size,
              fill_value,
              *,
              dtype: Optional[torch.dtype] = None,
              layout: torch.layout = torch.strided,
              requires_grad: bool = False,
              device_mesh: Optional[DeviceMesh] = None,
              placements: Optional[Sequence[Placement]] = None,
          ) -> DTensor

覆盖维度表：
| 覆盖维度         | 说明                                                         | 覆盖情况                                                          |
|------------------|--------------------------------------------------------------|-------------------------------------------------------------------|
| 空/非空          | fill_value: 整数、浮点数、负数、0                             | 已覆盖                                                            |
| 枚举选项         | placements: Shard(0) / Replicate                              | 已覆盖                                                            |
| 参数类型         | fill_value: int / float；dtype: None / float32 / float16      | 已覆盖                                                            |
| 传参与不传参     | requires_grad=True/False；dtype 省略 vs 显式                  | 已覆盖                                                            |
| 等价类/边界值    | fill_value=0；fill_value=负数；fill_value=大整数              | 已覆盖                                                            |
| 正常传参场景     | 返回 DTensor，shape/dtype 正确，fill_value 被正确填充         | 已覆盖                                                            |
| 异常传参场景     | 无稳定异常路径                                               | 未覆盖，原因：合法参数下无异常路径                                 |

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
from torch.distributed.tensor import DTensor, full
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


def _test_full_replicate_float(rank, world_size):
    device_type = torch._C._get_privateuse1_backend_name()
    mesh = DeviceMesh(device_type, list(range(world_size)))
    t = full([4, 8], 3.14, device_mesh=mesh, placements=[Replicate()])
    assert isinstance(t, DTensor)
    assert t.shape == torch.Size([4, 8])
    assert t.device.type == device_type


def _test_full_replicate_int(rank, world_size):
    device_type = torch._C._get_privateuse1_backend_name()
    mesh = DeviceMesh(device_type, list(range(world_size)))
    t = full([6], 42, dtype=torch.int32, device_mesh=mesh, placements=[Replicate()])
    assert isinstance(t, DTensor)
    assert t.dtype == torch.int32
    assert t.shape == torch.Size([6])


def _test_full_shard0(rank, world_size):
    device_type = torch._C._get_privateuse1_backend_name()
    mesh = DeviceMesh(device_type, list(range(world_size)))
    t = full([4, 8], 1.0, device_mesh=mesh, placements=[Shard(0)])
    assert isinstance(t, DTensor)
    assert t.shape == torch.Size([4, 8])


def _test_full_fill_value_zero(rank, world_size):
    device_type = torch._C._get_privateuse1_backend_name()
    mesh = DeviceMesh(device_type, list(range(world_size)))
    t = full([4, 4], 0, device_mesh=mesh, placements=[Replicate()])
    assert isinstance(t, DTensor)


def _test_full_fill_value_negative(rank, world_size):
    device_type = torch._C._get_privateuse1_backend_name()
    mesh = DeviceMesh(device_type, list(range(world_size)))
    t = full([3, 3], -7.5, device_mesh=mesh, placements=[Replicate()])
    assert isinstance(t, DTensor)
    assert t.shape == torch.Size([3, 3])


def _test_full_dtype_float16(rank, world_size):
    device_type = torch._C._get_privateuse1_backend_name()
    mesh = DeviceMesh(device_type, list(range(world_size)))
    t = full([4, 4], 1.0, dtype=torch.float16, device_mesh=mesh, placements=[Replicate()])
    assert t.dtype == torch.float16


def _test_full_requires_grad(rank, world_size):
    device_type = torch._C._get_privateuse1_backend_name()
    mesh = DeviceMesh(device_type, list(range(world_size)))
    t = full([4, 4], 1.0, requires_grad=True, device_mesh=mesh, placements=[Replicate()])
    assert t.requires_grad is True


def _test_full_3d(rank, world_size):
    device_type = torch._C._get_privateuse1_backend_name()
    mesh = DeviceMesh(device_type, list(range(world_size)))
    t = full([2, 3, 4], 5, device_mesh=mesh, placements=[Replicate()])
    assert t.shape == torch.Size([2, 3, 4])


class TestDTensorFull(TestCase):
    """Tests for torch.distributed.tensor.full."""

    def setUp(self):
        super().setUp()
        self.device_name = torch._C._get_privateuse1_backend_name()
        self.assertEqual(
            self.device_name, 'npu',
            f"Expected device 'npu', got '{self.device_name}'"
        )

    @skipIfUnsupportMultiNPU(2)
    def test_full_replicate_float(self):
        """full with float fill_value and Replicate placement."""
        mp.spawn(_init_dist, args=(2, _test_full_replicate_float), nprocs=2, join=True)

    @skipIfUnsupportMultiNPU(2)
    def test_full_replicate_int(self):
        """full with int fill_value and int32 dtype."""
        mp.spawn(_init_dist, args=(2, _test_full_replicate_int), nprocs=2, join=True)

    @skipIfUnsupportMultiNPU(2)
    def test_full_shard0(self):
        """full with Shard(0) placement returns correct global shape."""
        mp.spawn(_init_dist, args=(2, _test_full_shard0), nprocs=2, join=True)

    @skipIfUnsupportMultiNPU(2)
    def test_full_fill_value_zero(self):
        """full with fill_value=0 creates zero-filled DTensor."""
        mp.spawn(_init_dist, args=(2, _test_full_fill_value_zero), nprocs=2, join=True)

    @skipIfUnsupportMultiNPU(2)
    def test_full_fill_value_negative(self):
        """full with negative fill_value creates correctly shaped DTensor."""
        mp.spawn(_init_dist, args=(2, _test_full_fill_value_negative), nprocs=2, join=True)

    @skipIfUnsupportMultiNPU(2)
    def test_full_dtype_float16(self):
        """full with dtype=float16 returns float16 DTensor."""
        mp.spawn(_init_dist, args=(2, _test_full_dtype_float16), nprocs=2, join=True)

    @skipIfUnsupportMultiNPU(2)
    def test_full_requires_grad(self):
        """full with requires_grad=True returns differentiable DTensor."""
        mp.spawn(_init_dist, args=(2, _test_full_requires_grad), nprocs=2, join=True)

    @skipIfUnsupportMultiNPU(2)
    def test_full_3d(self):
        """full creates a 3D DTensor correctly."""
        mp.spawn(_init_dist, args=(2, _test_full_3d), nprocs=2, join=True)


if __name__ == "__main__":
    run_tests()
