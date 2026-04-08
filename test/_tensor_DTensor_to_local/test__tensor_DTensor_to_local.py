# -*- coding: utf-8 -*-
"""
测试目的：验证 torch.distributed.tensor.DTensor.to_local 接口功能正确性
API 名称：torch.distributed.tensor.DTensor.to_local
API 签名：to_local(*, grad_placements: Sequence[Placement] | None = None) -> torch.Tensor

覆盖维度表：
| 覆盖维度         | 说明                                                         | 覆盖情况                                                      |
|------------------|--------------------------------------------------------------|---------------------------------------------------------------|
| 空/非空          | grad_placements 传 None vs 显式 Placement 列表               | 已覆盖：test_to_local_no_grad_placements / test_to_local_with_grad_placements |
| 枚举选项         | Replicate、Shard(0)、Shard(1) 放置策略                       | 已覆盖：test_to_local_replicate / test_to_local_shard         |
| 参数类型         | grad_placements 为 None / list / tuple                       | 已覆盖                                                        |
| 传参与不传参     | 省略 grad_placements 使用默认值 vs 显式传入                  | 已覆盖                                                        |
| 等价类/边界值    | 1D、2D 张量，不同 dtype (float32, float16, int32)            | 已覆盖：test_to_local_dtypes / test_to_local_1d              |
| 正常传参场景     | Replicate/Shard 下调用 to_local                              | 已覆盖                                                        |
| 异常传参场景     | 无稳定可断言的异常路径（API 自身不做类型强校验）             | 未覆盖，原因：API 容错较好，异常路径依赖内部 autograd 状态    |

未覆盖项及原因：
- 异常路径（如 grad_placements 类型错误）：API 在部分情况下静默处理，无稳定异常

注意：本测试仅验证功能正确性（调用不报错、输出 shape/dtype/类型符合预期），
     不做精度和数值正确性校验。
"""

import os
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.distributed.tensor import DTensor
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.tensor.placement_types import Replicate, Shard

import torch_npu  # noqa: F401 — registers NPU backend
from torch_npu.testing.testcase import TestCase, run_tests
from torch_npu.testing.common_distributed import skipIfUnsupportMultiNPU


def _init_dist_hccl(rank, world_size, port='29516'):
    os.environ['MASTER_ADDR'] = '127.0.0.1'
    os.environ['MASTER_PORT'] = port
    os.environ['HCCL_WHITELIST_DISABLE'] = '1'
    torch_npu.npu.set_device(rank)
    dist.init_process_group(backend='hccl', world_size=world_size, rank=rank)


def _test_to_local_no_grad_placements(rank, world_size):
    """to_local without grad_placements on Replicate DTensor."""
    _init_dist_hccl(rank, world_size)
    try:
        mesh = init_device_mesh('npu', (world_size,))
        local = torch.randn(4, 4, device=f'npu:{rank}')
        dt = DTensor.from_local(local, mesh, [Replicate()])

        result = dt.to_local()
        assert isinstance(result, torch.Tensor), f"Expected Tensor, got {type(result)}"
        assert result.shape == (4, 4), f"Unexpected shape: {result.shape}"
        assert 'npu' in str(result.device), f"Expected npu device, got {result.device}"
    finally:
        dist.destroy_process_group()


def _test_to_local_with_grad_placements(rank, world_size):
    """to_local with explicit grad_placements on Replicate DTensor."""
    _init_dist_hccl(rank, world_size)
    try:
        mesh = init_device_mesh('npu', (world_size,))
        local = torch.randn(4, 4, device=f'npu:{rank}', requires_grad=True)
        dt = DTensor.from_local(local, mesh, [Replicate()])

        # grad_placements as list
        result = dt.to_local(grad_placements=[Replicate()])
        assert isinstance(result, torch.Tensor), f"Expected Tensor, got {type(result)}"
        assert result.shape == (4, 4), f"Unexpected shape: {result.shape}"
    finally:
        dist.destroy_process_group()


def _test_to_local_shard_dim0(rank, world_size):
    """to_local on Shard(0) DTensor returns correct local shard shape."""
    _init_dist_hccl(rank, world_size)
    try:
        mesh = init_device_mesh('npu', (world_size,))
        # Each rank contributes a [2, 4] local tensor, global shape is [4, 4]
        local = torch.randn(2, 4, device=f'npu:{rank}')
        dt = DTensor.from_local(local, mesh, [Shard(0)])

        result = dt.to_local()
        assert isinstance(result, torch.Tensor), f"Expected Tensor, got {type(result)}"
        assert result.shape == (2, 4), f"Unexpected shape: {result.shape}"
    finally:
        dist.destroy_process_group()


def _test_to_local_shard_dim1(rank, world_size):
    """to_local on Shard(1) DTensor returns correct local shard shape."""
    _init_dist_hccl(rank, world_size)
    try:
        mesh = init_device_mesh('npu', (world_size,))
        local = torch.randn(4, 2, device=f'npu:{rank}')
        dt = DTensor.from_local(local, mesh, [Shard(1)])

        result = dt.to_local()
        assert isinstance(result, torch.Tensor), f"Expected Tensor, got {type(result)}"
        assert result.shape == (4, 2), f"Unexpected shape: {result.shape}"
    finally:
        dist.destroy_process_group()


def _test_to_local_dtypes(rank, world_size):
    """to_local preserves dtype for float32, float16, int32."""
    _init_dist_hccl(rank, world_size)
    try:
        mesh = init_device_mesh('npu', (world_size,))
        for dtype in [torch.float32, torch.float16, torch.int32]:
            local = torch.ones(4, 4, dtype=dtype, device=f'npu:{rank}')
            dt = DTensor.from_local(local, mesh, [Replicate()])
            result = dt.to_local()
            assert result.dtype == dtype, f"dtype mismatch: expected {dtype}, got {result.dtype}"
    finally:
        dist.destroy_process_group()


def _test_to_local_1d(rank, world_size):
    """to_local on 1-D sharded DTensor."""
    _init_dist_hccl(rank, world_size)
    try:
        mesh = init_device_mesh('npu', (world_size,))
        local = torch.arange(8, dtype=torch.float32, device=f'npu:{rank}')
        dt = DTensor.from_local(local, mesh, [Replicate()])
        result = dt.to_local()
        assert result.shape == (8,), f"Unexpected shape: {result.shape}"
        assert result.dtype == torch.float32
    finally:
        dist.destroy_process_group()


def _test_to_local_requires_grad(rank, world_size):
    """to_local is differentiable; returned tensor tracks grad when DTensor does."""
    _init_dist_hccl(rank, world_size)
    try:
        mesh = init_device_mesh('npu', (world_size,))
        local = torch.randn(4, 4, device=f'npu:{rank}', requires_grad=True)
        dt = DTensor.from_local(local, mesh, [Replicate()])

        result = dt.to_local()
        # When DTensor requires_grad, local tensor should also require grad
        assert result.requires_grad, "Expected requires_grad=True on local tensor"
    finally:
        dist.destroy_process_group()


def _test_to_local_no_grad(rank, world_size):
    """to_local under torch.no_grad() returns _local_tensor directly."""
    _init_dist_hccl(rank, world_size)
    try:
        mesh = init_device_mesh('npu', (world_size,))
        local = torch.randn(4, 4, device=f'npu:{rank}')
        dt = DTensor.from_local(local, mesh, [Replicate()])

        with torch.no_grad():
            result = dt.to_local()
        assert isinstance(result, torch.Tensor)
        assert result.shape == (4, 4)
    finally:
        dist.destroy_process_group()


def _test_to_local_grad_placements_tuple(rank, world_size):
    """to_local accepts grad_placements as a tuple."""
    _init_dist_hccl(rank, world_size)
    try:
        mesh = init_device_mesh('npu', (world_size,))
        local = torch.randn(4, 4, device=f'npu:{rank}', requires_grad=True)
        dt = DTensor.from_local(local, mesh, [Replicate()])

        # grad_placements passed as tuple (not just list)
        result = dt.to_local(grad_placements=(Replicate(),))
        assert isinstance(result, torch.Tensor)
        assert result.shape == (4, 4)
    finally:
        dist.destroy_process_group()


# ---------------------------------------------------------------------------
# TestCase
# ---------------------------------------------------------------------------

class TestDTensorToLocal(TestCase):
    """Test cases for DTensor.to_local on NPU with HCCL backend."""

    def setUp(self):
        super().setUp()
        self.device_name = torch._C._get_privateuse1_backend_name()
        self.assertEqual(
            self.device_name, 'npu',
            f"Expected device 'npu', got '{self.device_name}'"
        )

    def _run(self, fn, world_size=2):
        mp.spawn(fn, args=(world_size,), nprocs=world_size, join=True)

    @skipIfUnsupportMultiNPU(2)
    def test_to_local_no_grad_placements(self):
        """to_local without grad_placements on Replicate DTensor."""
        self._run(_test_to_local_no_grad_placements)

    @skipIfUnsupportMultiNPU(2)
    def test_to_local_with_grad_placements(self):
        """to_local with explicit grad_placements list."""
        self._run(_test_to_local_with_grad_placements)

    @skipIfUnsupportMultiNPU(2)
    def test_to_local_shard_dim0(self):
        """to_local on Shard(0) returns correct local shard."""
        self._run(_test_to_local_shard_dim0)

    @skipIfUnsupportMultiNPU(2)
    def test_to_local_shard_dim1(self):
        """to_local on Shard(1) returns correct local shard."""
        self._run(_test_to_local_shard_dim1)

    @skipIfUnsupportMultiNPU(2)
    def test_to_local_dtypes(self):
        """to_local preserves dtype for float32, float16, int32."""
        self._run(_test_to_local_dtypes)

    @skipIfUnsupportMultiNPU(2)
    def test_to_local_1d(self):
        """to_local on 1-D DTensor returns correct shape."""
        self._run(_test_to_local_1d)

    @skipIfUnsupportMultiNPU(2)
    def test_to_local_requires_grad(self):
        """to_local returns grad-tracking tensor when DTensor requires_grad."""
        self._run(_test_to_local_requires_grad)

    @skipIfUnsupportMultiNPU(2)
    def test_to_local_no_grad(self):
        """to_local under torch.no_grad() returns valid local tensor."""
        self._run(_test_to_local_no_grad)

    @skipIfUnsupportMultiNPU(2)
    def test_to_local_grad_placements_tuple(self):
        """to_local accepts grad_placements as tuple."""
        self._run(_test_to_local_grad_placements_tuple)


if __name__ == "__main__":
    run_tests()
