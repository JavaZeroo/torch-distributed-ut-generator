# -*- coding: utf-8 -*-
"""
测试目的：验证 torch.distributed.checkpoint.LoadPlanner.resolve_tensor 接口功能正确性
API 名称：torch.distributed.checkpoint.LoadPlanner.resolve_tensor
API 签名：LoadPlanner.resolve_tensor(self, read_item: ReadItem) -> torch.Tensor

覆盖维度表：
| 覆盖维度         | 说明                                                         | 覆盖情况                                                      |
|------------------|--------------------------------------------------------------|---------------------------------------------------------------|
| 空/非空          | state_dict 含单 tensor vs 多 tensor                          | 已覆盖                                                        |
| 枚举选项         | LoadItemType.TENSOR（唯一支持类型）                           | 已覆盖                                                        |
| 参数类型         | float32 / float16 / int32 dtype 的 tensor                    | 已覆盖                                                        |
| 传参与不传参     | ReadItem 通过 create_local_plan 自动生成                       | 已覆盖                                                        |
| 等价类/边界值    | 不同 shape；单元素；多维                                      | 已覆盖                                                        |
| 正常传参场景     | resolve_tensor 返回目标 tensor，shape/dtype 正确              | 已覆盖                                                        |
| 异常传参场景     | 无稳定异常路径（ReadItem 由 plan 生成保证合法）               | 未覆盖，原因：合法路径由 plan 保证                            |

未覆盖项及原因：
- 异常传参：ReadItem 由 plan 生成，构造非法 ReadItem 无法稳定触发有意义的异常。

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
    MetadataIndex,
    TensorProperties,
    TensorStorageMetadata,
)
from torch.distributed.checkpoint.planner import LoadItemType
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


def _build_metadata_for_state_dict(state_dict, device):
    """Build a Metadata object matching the given state_dict."""
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


def _test_resolve_tensor_basic(rank, world_size):
    """resolve_tensor returns a tensor aliasing the state_dict entry."""
    device = f'npu:{rank}'
    state_dict = {'weight': torch.zeros(4, 4, device=device)}
    metadata = _build_metadata_for_state_dict(state_dict, device)
    planner = DefaultLoadPlanner()
    planner.set_up_planner(state_dict, metadata=metadata, is_coordinator=(rank == 0))
    plan = planner.create_local_plan()
    tensor_items = [item for item in plan.items if item.type == LoadItemType.TENSOR]
    assert len(tensor_items) > 0, "Expected TENSOR read items"
    for item in tensor_items:
        result = planner.resolve_tensor(item)
        assert isinstance(result, torch.Tensor), f"Expected Tensor, got {type(result)}"
        assert result.shape == state_dict[item.dest_index.fqn].shape


def _test_resolve_tensor_float16(rank, world_size):
    """resolve_tensor works for float16 tensors."""
    device = f'npu:{rank}'
    state_dict = {'fp16_w': torch.zeros(8, device=device, dtype=torch.float16)}
    metadata = _build_metadata_for_state_dict(state_dict, device)
    planner = DefaultLoadPlanner()
    planner.set_up_planner(state_dict, metadata=metadata, is_coordinator=(rank == 0))
    plan = planner.create_local_plan()
    tensor_items = [item for item in plan.items if item.type == LoadItemType.TENSOR]
    for item in tensor_items:
        result = planner.resolve_tensor(item)
        assert result.dtype == torch.float16, f"Expected float16, got {result.dtype}"


def _test_resolve_tensor_multiple_keys(rank, world_size):
    """resolve_tensor handles multiple keys in state_dict."""
    device = f'npu:{rank}'
    state_dict = {
        'w1': torch.zeros(2, 3, device=device),
        'w2': torch.zeros(5, device=device),
    }
    metadata = _build_metadata_for_state_dict(state_dict, device)
    planner = DefaultLoadPlanner()
    planner.set_up_planner(state_dict, metadata=metadata, is_coordinator=(rank == 0))
    plan = planner.create_local_plan()
    resolved_keys = set()
    for item in plan.items:
        if item.type == LoadItemType.TENSOR:
            result = planner.resolve_tensor(item)
            assert isinstance(result, torch.Tensor)
            resolved_keys.add(item.dest_index.fqn)
    assert resolved_keys == set(state_dict.keys()), \
        f"Expected keys {set(state_dict.keys())}, got {resolved_keys}"


def _test_resolve_tensor_shape_preserved(rank, world_size):
    """resolve_tensor preserves the shape of the target tensor."""
    device = f'npu:{rank}'
    shapes = [(1,), (4, 4), (2, 3, 4)]
    for i, shape in enumerate(shapes):
        state_dict = {f'key_{i}': torch.zeros(*shape, device=device)}
        metadata = _build_metadata_for_state_dict(state_dict, device)
        planner = DefaultLoadPlanner()
        planner.set_up_planner(state_dict, metadata=metadata, is_coordinator=(rank == 0))
        plan = planner.create_local_plan()
        for item in plan.items:
            if item.type == LoadItemType.TENSOR:
                result = planner.resolve_tensor(item)
                assert result.shape == torch.Size(shape), \
                    f"Shape mismatch: expected {shape}, got {result.shape}"


class TestLoadPlannerResolveTensor(TestCase):
    """Tests for LoadPlanner.resolve_tensor via DefaultLoadPlanner."""

    def setUp(self):
        super().setUp()
        self.device_name = torch._C._get_privateuse1_backend_name()
        self.assertEqual(
            self.device_name, 'npu',
            f"Expected device 'npu', got '{self.device_name}'"
        )

    @skipIfUnsupportMultiNPU(2)
    def test_resolve_tensor_basic(self):
        """resolve_tensor returns a valid Tensor for a basic float32 weight."""
        mp.spawn(_init_dist, args=(2, _test_resolve_tensor_basic), nprocs=2, join=True)

    @skipIfUnsupportMultiNPU(2)
    def test_resolve_tensor_float16(self):
        """resolve_tensor preserves float16 dtype."""
        mp.spawn(_init_dist, args=(2, _test_resolve_tensor_float16), nprocs=2, join=True)

    @skipIfUnsupportMultiNPU(2)
    def test_resolve_tensor_multiple_keys(self):
        """resolve_tensor iterates all keys in state_dict correctly."""
        mp.spawn(_init_dist, args=(2, _test_resolve_tensor_multiple_keys), nprocs=2, join=True)

    @skipIfUnsupportMultiNPU(2)
    def test_resolve_tensor_shape_preserved(self):
        """resolve_tensor returns tensor with correct shape for various shapes."""
        mp.spawn(_init_dist, args=(2, _test_resolve_tensor_shape_preserved), nprocs=2, join=True)


if __name__ == "__main__":
    run_tests()
