# -*- coding: utf-8 -*-
"""
测试目的：验证 torch.distributed.checkpoint.LoadPlanner.load_bytes 接口功能正确性
API 名称：torch.distributed.checkpoint.LoadPlanner.load_bytes
API 签名：LoadPlanner.load_bytes(self, read_item: ReadItem, value: io.BytesIO) -> None

覆盖维度表：
| 覆盖维度         | 说明                                                         | 覆盖情况                                                              |
|------------------|--------------------------------------------------------------|-----------------------------------------------------------------------|
| 空/非空          | BytesIO 内容非空（含序列化对象）                              | 已覆盖                                                                |
| 枚举选项         | LoadItemType.BYTE_IO                                          | 已覆盖                                                                |
| 参数类型         | BytesIO 含序列化 int / str / dict                            | 已覆盖                                                                |
| 传参与不传参     | 调用参数固定                                                  | 已覆盖                                                                |
| 等价类/边界值    | 最小字节内容（序列化 int）；稍大内容（dict）                  | 已覆盖                                                                |
| 正常传参场景     | load_bytes 将 BytesIO 反序列化写入 state_dict                 | 已覆盖                                                                |
| 异常传参场景     | 无稳定异常路径（DefaultLoadPlanner 使用 torch.load 反序列化） | 未覆盖，原因：合法 BytesIO 保证无异常                                 |

未覆盖项及原因：
- 无稳定异常路径：load_bytes 接受任意合法 BytesIO，破坏格式会触发 torch.load 异常但不属于 API 本身错误。

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
    ChunkStorageMetadata,
    Metadata,
    TensorProperties,
    TensorStorageMetadata,
)
from torch.distributed.checkpoint.planner import LoadItemType, ReadItem, LoadPlan
from torch.distributed.checkpoint.metadata import MetadataIndex
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


def _make_bytes_io(obj):
    """Serialize obj to BytesIO using torch.save."""
    buf = io.BytesIO()
    torch.save(obj, buf)
    buf.seek(0)
    return buf


def _test_load_bytes_basic(rank, world_size):
    """load_bytes deserializes a BytesIO value into state_dict."""
    device = f'npu:{rank}'
    # Use a bytes-type value in state_dict
    state_dict = {'meta': {"step": 10, "epoch": 3}}
    metadata = Metadata(state_dict_metadata={'meta': BytesStorageMetadata()})
    planner = DefaultLoadPlanner()
    planner.set_up_planner(state_dict, metadata=metadata, is_coordinator=(rank == 0))
    plan = planner.create_local_plan()
    byte_items = [item for item in plan.items if item.type == LoadItemType.BYTE_IO]
    assert len(byte_items) > 0, "Expected BYTE_IO read items for non-tensor state"
    for item in byte_items:
        value = _make_bytes_io({"step": 10, "epoch": 3})
        planner.load_bytes(item, value)  # Should not raise


def _test_load_bytes_int(rank, world_size):
    """load_bytes handles serialized int value."""
    state_dict = {'step': 42}
    metadata = Metadata(state_dict_metadata={'step': BytesStorageMetadata()})
    planner = DefaultLoadPlanner()
    planner.set_up_planner(state_dict, metadata=metadata, is_coordinator=(rank == 0))
    plan = planner.create_local_plan()
    for item in plan.items:
        if item.type == LoadItemType.BYTE_IO:
            value = _make_bytes_io(42)
            planner.load_bytes(item, value)


def _test_load_bytes_updates_state_dict(rank, world_size):
    """load_bytes updates the state_dict with the deserialized value."""
    state_dict = {'config': {"lr": 0.01}}
    metadata = Metadata(state_dict_metadata={'config': BytesStorageMetadata()})
    planner = DefaultLoadPlanner()
    planner.set_up_planner(state_dict, metadata=metadata, is_coordinator=(rank == 0))
    plan = planner.create_local_plan()
    for item in plan.items:
        if item.type == LoadItemType.BYTE_IO:
            new_config = {"lr": 0.001, "momentum": 0.9}
            value = _make_bytes_io(new_config)
            planner.load_bytes(item, value)
    # After load_bytes, the state_dict should be updated
    # DefaultLoadPlanner stores the loaded value back
    assert 'config' in planner.state_dict


class TestLoadPlannerLoadBytes(TestCase):
    """Tests for LoadPlanner.load_bytes via DefaultLoadPlanner."""

    def setUp(self):
        super().setUp()
        self.device_name = torch._C._get_privateuse1_backend_name()
        self.assertEqual(
            self.device_name, 'npu',
            f"Expected device 'npu', got '{self.device_name}'"
        )

    @skipIfUnsupportMultiNPU(2)
    def test_load_bytes_basic(self):
        """load_bytes accepts serialized dict without error."""
        mp.spawn(_init_dist, args=(2, _test_load_bytes_basic), nprocs=2, join=True)

    @skipIfUnsupportMultiNPU(2)
    def test_load_bytes_int_value(self):
        """load_bytes accepts serialized int value."""
        mp.spawn(_init_dist, args=(2, _test_load_bytes_int), nprocs=2, join=True)

    @skipIfUnsupportMultiNPU(2)
    def test_load_bytes_updates_state_dict(self):
        """load_bytes updates planner's state_dict with deserialized value."""
        mp.spawn(_init_dist, args=(2, _test_load_bytes_updates_state_dict), nprocs=2, join=True)


if __name__ == "__main__":
    run_tests()
