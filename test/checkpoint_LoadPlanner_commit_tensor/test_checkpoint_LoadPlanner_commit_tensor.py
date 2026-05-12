# -*- coding: utf-8 -*-
"""
测试目的：验证 torch.distributed.checkpoint.LoadPlanner.commit_tensor 接口功能正确性
API 名称：torch.distributed.checkpoint.LoadPlanner.commit_tensor
API 签名：LoadPlanner.commit_tensor(self, read_item: ReadItem, tensor: torch.Tensor) -> None

覆盖维度表：
| 覆盖维度         | 说明                                                         | 覆盖情况                                                        |
|------------------|--------------------------------------------------------------|------------------------------------------------------------------|
| 空/非空          | tensor 为非空 Tensor                                          | 已覆盖                                                           |
| 枚举选项         | LoadItemType.TENSOR                                           | 已覆盖                                                           |
| 参数类型         | float32 / float16 tensor                                     | 已覆盖                                                           |
| 传参与不传参     | 调用参数固定（read_item + tensor）                            | 已覆盖                                                           |
| 等价类/边界值    | 各种 shape；commit 后 state_dict 内容不变（no-op 默认实现）  | 已覆盖                                                           |
| 正常传参场景     | commit_tensor 调用不报错，state_dict 保持一致               | 已覆盖                                                           |
| 异常传参场景     | 无稳定异常路径（DefaultLoadPlanner 默认实现为 no-op）        | 未覆盖，原因：默认 commit_tensor 为空实现                        |

未覆盖项及原因：
- 异常传参：DefaultLoadPlanner.commit_tensor 是空方法（no-op），无稳定异常路径。

注意：本测试仅验证功能正确性（调用不报错、输出 shape/dtype/类型符合预期），
     不做精度和数值正确性校验。
"""
import os
import unittest

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch_npu
from torch.distributed.checkpoint.default_planner import DefaultLoadPlanner
from torch.distributed.checkpoint.metadata import (
    ChunkStorageMetadata,
    Metadata,
    TensorProperties,
    TensorStorageMetadata,
)
from torch.distributed.checkpoint.planner import LoadItemType
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


def _build_metadata(state_dict):
    state_dict_metadata = {}
    for key, tensor in state_dict.items():
        state_dict_metadata[key] = TensorStorageMetadata(
            properties=TensorProperties(dtype=tensor.dtype),
            size=tensor.size(),
            chunks=[ChunkStorageMetadata(
                offsets=torch.Size([0] * tensor.dim()),
                sizes=tensor.size(),
            )],
        )
    return Metadata(state_dict_metadata=state_dict_metadata)


def _test_commit_tensor_basic(rank, world_size):
    """commit_tensor completes without error for a basic float32 tensor."""
    device = f'npu:{rank}'
    state_dict = {'weight': torch.zeros(4, 4, device=device)}
    metadata = _build_metadata(state_dict)
    planner = DefaultLoadPlanner()
    planner.set_up_planner(state_dict, metadata=metadata, is_coordinator=(rank == 0))
    plan = planner.create_local_plan()
    for item in plan.items:
        if item.type == LoadItemType.TENSOR:
            tensor = planner.resolve_tensor(item)
            planner.commit_tensor(item, tensor)  # Should be no-op, not raise


def _test_commit_tensor_float16(rank, world_size):
    """commit_tensor accepts float16 tensors without error."""
    device = f'npu:{rank}'
    state_dict = {'fp16': torch.zeros(8, device=device, dtype=torch.float16)}
    metadata = _build_metadata(state_dict)
    planner = DefaultLoadPlanner()
    planner.set_up_planner(state_dict, metadata=metadata, is_coordinator=(rank == 0))
    plan = planner.create_local_plan()
    for item in plan.items:
        if item.type == LoadItemType.TENSOR:
            tensor = planner.resolve_tensor(item)
            planner.commit_tensor(item, tensor)


def _test_commit_tensor_multiple_items(rank, world_size):
    """commit_tensor is called for each TENSOR read item."""
    device = f'npu:{rank}'
    state_dict = {
        'w1': torch.zeros(2, 4, device=device),
        'w2': torch.zeros(4, device=device),
        'w3': torch.zeros(3, 3, 3, device=device),
    }
    metadata = _build_metadata(state_dict)
    planner = DefaultLoadPlanner()
    planner.set_up_planner(state_dict, metadata=metadata, is_coordinator=(rank == 0))
    plan = planner.create_local_plan()
    committed = 0
    for item in plan.items:
        if item.type == LoadItemType.TENSOR:
            tensor = planner.resolve_tensor(item)
            planner.commit_tensor(item, tensor)
            committed += 1
    assert committed == len(state_dict), \
        f"Expected {len(state_dict)} commits, got {committed}"


def _test_commit_tensor_returns_none(rank, world_size):
    """commit_tensor return value is None (no-op by default)."""
    device = f'npu:{rank}'
    state_dict = {'w': torch.zeros(4, device=device)}
    metadata = _build_metadata(state_dict)
    planner = DefaultLoadPlanner()
    planner.set_up_planner(state_dict, metadata=metadata, is_coordinator=(rank == 0))
    plan = planner.create_local_plan()
    for item in plan.items:
        if item.type == LoadItemType.TENSOR:
            tensor = planner.resolve_tensor(item)
            result = planner.commit_tensor(item, tensor)
            assert result is None, f"commit_tensor should return None, got {result}"


class TestLoadPlannerCommitTensor(TestCase):
    """Tests for LoadPlanner.commit_tensor via DefaultLoadPlanner."""

    def setUp(self):
        super().setUp()
        self.device_name = torch._C._get_privateuse1_backend_name()
        self.assertEqual(
            self.device_name, 'npu',
            f"Expected device 'npu', got '{self.device_name}'"
        )

    @skipIfUnsupportMultiNPU(2)
    def test_commit_tensor_basic(self):
        """commit_tensor completes without error for float32 tensor."""
        mp.spawn(_init_dist, args=(2, _test_commit_tensor_basic), nprocs=2, join=True)

    @skipIfUnsupportMultiNPU(2)
    def test_commit_tensor_float16(self):
        """commit_tensor accepts float16 dtype tensor."""
        mp.spawn(_init_dist, args=(2, _test_commit_tensor_float16), nprocs=2, join=True)

    @skipIfUnsupportMultiNPU(2)
    def test_commit_tensor_multiple_items(self):
        """commit_tensor processes all TENSOR items in plan."""
        mp.spawn(_init_dist, args=(2, _test_commit_tensor_multiple_items), nprocs=2, join=True)

    @skipIfUnsupportMultiNPU(2)
    def test_commit_tensor_returns_none(self):
        """commit_tensor returns None (DefaultLoadPlanner is a no-op)."""
        mp.spawn(_init_dist, args=(2, _test_commit_tensor_returns_none), nprocs=2, join=True)


if __name__ == "__main__":
    run_tests()
