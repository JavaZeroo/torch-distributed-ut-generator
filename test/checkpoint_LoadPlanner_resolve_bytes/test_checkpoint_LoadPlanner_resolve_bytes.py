# -*- coding: utf-8 -*-
"""
测试目的：验证 torch.distributed.checkpoint.LoadPlanner.resolve_bytes 接口功能正确性
API 名称：torch.distributed.checkpoint.LoadPlanner.resolve_bytes
API 签名：LoadPlanner.resolve_bytes(self, read_item: ReadItem) -> io.BytesIO

覆盖维度表：
| 覆盖维度         | 说明                                                         | 覆盖情况                                                            |
|------------------|--------------------------------------------------------------|---------------------------------------------------------------------|
| 空/非空          | state_dict 含字节对象（dict/str）                            | 已覆盖                                                              |
| 枚举选项         | LoadItemType.BYTE_IO                                         | 已覆盖                                                              |
| 参数类型         | 序列化 dict / int 存入 BytesIO                               | 已覆盖                                                              |
| 传参与不传参     | 无额外参数                                                   | 已覆盖                                                              |
| 等价类/边界值    | 单 key bytes state dict；多 key                              | 已覆盖                                                              |
| 正常传参场景     | resolve_bytes 返回 BytesIO 对象                              | 已覆盖                                                              |
| 异常传参场景     | 基类抛 NotImplementedError；DefaultLoadPlanner 有具体实现    | 已覆盖：基类抛 NotImplementedError                                  |

未覆盖项及原因：
- 无。

注意：本测试仅验证功能正确性（调用不报错、输出 shape/dtype/类型符合预期），
     不做精度和数值正确性校验。
"""
import io
import os
import unittest

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch_npu
from torch.distributed.checkpoint.default_planner import DefaultLoadPlanner
from torch.distributed.checkpoint.metadata import (
    BytesStorageMetadata,
    Metadata,
)
from torch.distributed.checkpoint.planner import LoadItemType, LoadPlanner
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


def _test_resolve_bytes_default_raises_not_implemented(rank, world_size):
    """DefaultLoadPlanner.resolve_bytes raises NotImplementedError (not overridden)."""
    state_dict = {'config': {"lr": 0.01, "steps": 100}}
    metadata = Metadata(state_dict_metadata={'config': BytesStorageMetadata()})
    planner = DefaultLoadPlanner()
    planner.set_up_planner(state_dict, metadata=metadata, is_coordinator=(rank == 0))
    plan = planner.create_local_plan()
    byte_items = [item for item in plan.items if item.type == LoadItemType.BYTE_IO]
    assert len(byte_items) > 0, "Expected BYTE_IO items for bytes state_dict value"
    for item in byte_items:
        try:
            planner.resolve_bytes(item)
            raise AssertionError("Expected NotImplementedError")
        except NotImplementedError:
            pass  # Expected: DefaultLoadPlanner does not implement resolve_bytes


def _test_resolve_bytes_custom_subclass_returns_bytesio(rank, world_size):
    """A subclass overriding resolve_bytes can return BytesIO for BYTE_IO items."""

    class _BytesLoadPlanner(DefaultLoadPlanner):
        """Custom planner that implements resolve_bytes."""
        def resolve_bytes(self, read_item):
            # Return a BytesIO backed by the serialized state_dict value
            obj = self.state_dict.get(read_item.dest_index.fqn)
            buf = io.BytesIO()
            torch.save(obj, buf)
            buf.seek(0)
            return buf

    state_dict = {
        'opt_state': {"step": 5},
        'scheduler': {"base_lr": 0.1},
    }
    metadata = Metadata(state_dict_metadata={
        'opt_state': BytesStorageMetadata(),
        'scheduler': BytesStorageMetadata(),
    })
    planner = _BytesLoadPlanner()
    planner.set_up_planner(state_dict, metadata=metadata, is_coordinator=(rank == 0))
    plan = planner.create_local_plan()
    resolved = 0
    for item in plan.items:
        if item.type == LoadItemType.BYTE_IO:
            result = planner.resolve_bytes(item)
            assert isinstance(result, io.BytesIO), \
                f"Expected BytesIO, got {type(result)}"
            resolved += 1
    assert resolved == 2, f"Expected 2 BytesIO results, got {resolved}"


def _test_base_planner_resolve_bytes_raises(rank, world_size):
    """LoadPlanner base class resolve_bytes raises NotImplementedError."""
    # Verify the base class raises, subclasses must override
    import abc

    class _MinimalLoadPlanner(LoadPlanner):
        def set_up_planner(self, state_dict, metadata=None, is_coordinator=False):
            pass
        def create_local_plan(self):
            pass
        def create_global_plan(self, global_plan):
            pass
        def finish_plan(self, central_plan):
            pass
        def load_bytes(self, read_item, value):
            pass
        def resolve_tensor(self, read_item):
            pass
        def commit_tensor(self, read_item, tensor):
            pass

    planner = _MinimalLoadPlanner()
    try:
        planner.resolve_bytes(None)
        raise AssertionError("Expected NotImplementedError")
    except NotImplementedError:
        pass  # Expected per base class contract


class TestLoadPlannerResolveBytes(TestCase):
    """Tests for LoadPlanner.resolve_bytes via DefaultLoadPlanner."""

    def setUp(self):
        super().setUp()
        self.device_name = torch._C._get_privateuse1_backend_name()
        self.assertEqual(
            self.device_name, 'npu',
            f"Expected device 'npu', got '{self.device_name}'"
        )

    @skipIfUnsupportMultiNPU(2)
    def test_resolve_bytes_default_raises_not_implemented(self):
        """DefaultLoadPlanner.resolve_bytes raises NotImplementedError."""
        mp.spawn(_init_dist, args=(2, _test_resolve_bytes_default_raises_not_implemented), nprocs=2, join=True)

    @skipIfUnsupportMultiNPU(2)
    def test_resolve_bytes_custom_subclass_returns_bytesio(self):
        """Custom subclass overriding resolve_bytes returns BytesIO for BYTE_IO items."""
        mp.spawn(_init_dist, args=(2, _test_resolve_bytes_custom_subclass_returns_bytesio), nprocs=2, join=True)

    @skipIfUnsupportMultiNPU(2)
    def test_base_planner_resolve_bytes_not_implemented(self):
        """Base LoadPlanner.resolve_bytes raises NotImplementedError."""
        mp.spawn(_init_dist, args=(2, _test_base_planner_resolve_bytes_raises), nprocs=2, join=True)


if __name__ == "__main__":
    run_tests()
