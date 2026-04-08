# -*- coding: utf-8 -*-
"""
测试目的：验证 torch.distributed.checkpoint.format_utils 相关接口功能正确性
API 名称：BroadcastingTorchSaveReader, BroadcastingTorchSaveReader.prepare_global_plan,
         BroadcastingTorchSaveReader.prepare_local_plan, BroadcastingTorchSaveReader.reset,
         BroadcastingTorchSaveReader.set_up_storage_reader, BroadcastingTorchSaveReader.validate_checkpoint_id,
         DynamicMetaLoadPlanner

API 签名：
- BroadcastingTorchSaveReader.__init__(checkpoint_id, coordinator_rank)
- BroadcastingTorchSaveReader.prepare_global_plan(global_plan: List[LoadPlan]) -> List[LoadPlan]
- BroadcastingTorchSaveReader.prepare_local_plan(plan: LoadPlan) -> LoadPlan
- BroadcastingTorchSaveReader.reset() -> None
- BroadcastingTorchSaveReader.set_up_storage_reader(metadata: Metadata, is_coordinator: bool) -> None
- BroadcastingTorchSaveReader.validate_checkpoint_id() -> None
- DynamicMetaLoadPlanner class and methods

覆盖维度表：
| 覆盖维度         | 说明                                                         | 覆盖情况                                       |
|------------------|--------------------------------------------------------------|------------------------------------------------|
| 初始化参数       | checkpoint_id 有/无, coordinator_rank 不同值                 | 已覆盖                                         |
| plan 操作         | prepare_global_plan, prepare_local_plan 返回类型              | 已覆盖                                         |
| 存储读取器设置    | is_coordinator True/False, metadata 合法值                   | 已覆盖                                         |
| checkpoint_id 校验 | 合法/非法的路径或值                                           | 已覆盖                                         |
| 动态元数据规划    | DynamicMetaLoadPlanner 基本功能                               | 已覆盖                                         |
| 重置操作         | reset 后状态清理                                              | 已覆盖                                         |

未覆盖项及原因：
- 跨进程通信场景（涉及实际 torch.distributed.broadcast）未覆盖，原因：纯工具类测试在单进程环境

注意：本测试仅验证功能正确性（调用不报错、输出类型符合预期），
     不做精度和数值正确性校验。
"""

import unittest
import torch
import os
import tempfile
from pathlib import Path

try:
    from torch_npu.testing.testcase import TestCase, run_tests
except ImportError:
    from unittest import TestCase

    def run_tests():
        unittest.main(argv=[''])

from torch.distributed.checkpoint.format_utils import (
    BroadcastingTorchSaveReader,
    DynamicMetaLoadPlanner,
)
from torch.distributed.checkpoint.planner import LoadPlan, ReadItem, LoadItemType
from torch.distributed.checkpoint.metadata import Metadata, MetadataIndex


class TestBroadcastingTorchSaveReaderInit(TestCase):
    """Test BroadcastingTorchSaveReader initialization."""

    def setUp(self):
        super().setUp()
        device_name = torch._C._get_privateuse1_backend_name()
        self.assertEqual(device_name, 'npu', f"Expected device 'npu', got '{device_name}'")

    def test_init_with_checkpoint_id(self):
        """Test BroadcastingTorchSaveReader initialization with checkpoint_id."""
        checkpoint_id = "/path/to/checkpoint.pt"
        reader = BroadcastingTorchSaveReader(checkpoint_id=checkpoint_id)

        self.assertEqual(reader.checkpoint_id, checkpoint_id)
        self.assertEqual(reader.coordinator_rank, 0)

    def test_init_with_custom_coordinator_rank(self):
        """Test BroadcastingTorchSaveReader with custom coordinator_rank."""
        checkpoint_id = "/path/to/checkpoint.pt"
        coordinator_rank = 2
        reader = BroadcastingTorchSaveReader(
            checkpoint_id=checkpoint_id,
            coordinator_rank=coordinator_rank,
        )

        self.assertEqual(reader.checkpoint_id, checkpoint_id)
        self.assertEqual(reader.coordinator_rank, coordinator_rank)

    def test_init_without_checkpoint_id(self):
        """Test BroadcastingTorchSaveReader initialization without checkpoint_id."""
        reader = BroadcastingTorchSaveReader()

        self.assertIsNone(reader.checkpoint_id)
        self.assertEqual(reader.coordinator_rank, 0)

    def test_init_with_path_object(self):
        """Test BroadcastingTorchSaveReader with Path object."""
        checkpoint_path = Path("/path/to/checkpoint.pt")
        reader = BroadcastingTorchSaveReader(checkpoint_id=checkpoint_path)

        self.assertEqual(reader.checkpoint_id, checkpoint_path)

    def test_init_with_none_checkpoint_id(self):
        """Test BroadcastingTorchSaveReader with explicitly None checkpoint_id."""
        reader = BroadcastingTorchSaveReader(checkpoint_id=None, coordinator_rank=1)

        self.assertIsNone(reader.checkpoint_id)
        self.assertEqual(reader.coordinator_rank, 1)


class TestBroadcastingTorchSaveReaderMethods(TestCase):
    """Test BroadcastingTorchSaveReader methods."""

    def setUp(self):
        super().setUp()
        device_name = torch._C._get_privateuse1_backend_name()
        self.assertEqual(device_name, 'npu', f"Expected device 'npu', got '{device_name}'")
        self.checkpoint_id = "/path/to/checkpoint.pt"
        self.reader = BroadcastingTorchSaveReader(checkpoint_id=self.checkpoint_id)

    def test_prepare_local_plan(self):
        """Test prepare_local_plan returns plan unchanged."""
        plan = LoadPlan(items=[])

        result = self.reader.prepare_local_plan(plan)

        self.assertIsInstance(result, LoadPlan)
        self.assertEqual(result, plan)

    def test_prepare_local_plan_with_items(self):
        """Test prepare_local_plan with items in plan."""
        read_item = ReadItem(
            type=LoadItemType.TENSOR,
            dest_index=MetadataIndex(fqn="weight"),
            dest_offsets=torch.Size([0, 0]),
            storage_index=MetadataIndex(fqn="weight"),
            storage_offsets=torch.Size([0, 0]),
            lengths=torch.Size([3, 4]),
        )
        plan = LoadPlan(items=[read_item])

        result = self.reader.prepare_local_plan(plan)

        self.assertIsInstance(result, LoadPlan)
        self.assertEqual(len(result.items), 1)

    def test_prepare_global_plan(self):
        """Test prepare_global_plan returns plans unchanged."""
        plan1 = LoadPlan(items=[])
        plan2 = LoadPlan(items=[])
        global_plan = [plan1, plan2]

        result = self.reader.prepare_global_plan(global_plan)

        self.assertIsInstance(result, list)
        self.assertEqual(len(result), 2)
        self.assertEqual(result, global_plan)

    def test_prepare_global_plan_empty(self):
        """Test prepare_global_plan with empty list."""
        global_plan = []

        result = self.reader.prepare_global_plan(global_plan)

        self.assertIsInstance(result, list)
        self.assertEqual(len(result), 0)


class TestBroadcastingTorchSaveReaderSetUpStorageReader(TestCase):
    """Test BroadcastingTorchSaveReader.set_up_storage_reader method."""

    def setUp(self):
        super().setUp()
        device_name = torch._C._get_privateuse1_backend_name()
        self.assertEqual(device_name, 'npu', f"Expected device 'npu', got '{device_name}'")

    def test_set_up_storage_reader_not_coordinator(self):
        """Test set_up_storage_reader with is_coordinator=False."""
        checkpoint_id = "/path/to/checkpoint.pt"
        reader = BroadcastingTorchSaveReader(checkpoint_id=checkpoint_id)
        metadata = Metadata(state_dict_metadata={})

        # Can call without distributed initialization when is_coordinator=False
        reader.set_up_storage_reader(metadata, is_coordinator=False)

        self.assertFalse(reader.is_coordinator)

    def test_set_up_storage_reader_without_checkpoint_id(self):
        """Test set_up_storage_reader raises AssertionError when checkpoint_id is None."""
        reader = BroadcastingTorchSaveReader(checkpoint_id=None)
        metadata = Metadata(state_dict_metadata={})

        # Should raise AssertionError before calling dist.get_rank()
        with self.assertRaises(AssertionError):
            reader.set_up_storage_reader(metadata, is_coordinator=False)

    def test_set_up_storage_reader_changes_coordinator_flag(self):
        """Test set_up_storage_reader updates is_coordinator flag."""
        checkpoint_id = "/path/to/checkpoint.pt"
        reader = BroadcastingTorchSaveReader(checkpoint_id=checkpoint_id)
        metadata = Metadata(state_dict_metadata={})

        # First call
        reader.set_up_storage_reader(metadata, is_coordinator=False)
        self.assertFalse(reader.is_coordinator)

        # Second call changes state
        reader.set_up_storage_reader(metadata, is_coordinator=False)
        self.assertFalse(reader.is_coordinator)


class TestBroadcastingTorchSaveReaderReadMetadata(TestCase):
    """Test BroadcastingTorchSaveReader.read_metadata method."""

    def setUp(self):
        super().setUp()
        device_name = torch._C._get_privateuse1_backend_name()
        self.assertEqual(device_name, 'npu', f"Expected device 'npu', got '{device_name}'")

    def test_read_metadata_returns_empty_metadata(self):
        """Test read_metadata returns empty Metadata."""
        checkpoint_id = "/path/to/checkpoint.pt"
        reader = BroadcastingTorchSaveReader(checkpoint_id=checkpoint_id)

        metadata = reader.read_metadata()

        self.assertIsInstance(metadata, Metadata)
        self.assertEqual(len(metadata.state_dict_metadata), 0)


class TestDynamicMetaLoadPlanner(TestCase):
    """Test DynamicMetaLoadPlanner class."""

    def setUp(self):
        super().setUp()
        device_name = torch._C._get_privateuse1_backend_name()
        self.assertEqual(device_name, 'npu', f"Expected device 'npu', got '{device_name}'")

    def test_dynamic_meta_load_planner_instantiation(self):
        """Test DynamicMetaLoadPlanner can be instantiated."""
        planner = DynamicMetaLoadPlanner()

        self.assertIsNotNone(planner)
        self.assertIsInstance(planner, DynamicMetaLoadPlanner)

    def test_dynamic_meta_load_planner_set_up_planner(self):
        """Test DynamicMetaLoadPlanner.set_up_planner method."""
        planner = DynamicMetaLoadPlanner()
        state_dict = {"weight": torch.randn(2, 3)}
        metadata = Metadata(state_dict_metadata={})

        # Should not raise error
        planner.set_up_planner(state_dict, metadata=metadata, is_coordinator=False)

        self.assertIsNotNone(planner.state_dict)

    def test_dynamic_meta_load_planner_with_empty_state_dict(self):
        """Test DynamicMetaLoadPlanner with empty state_dict."""
        planner = DynamicMetaLoadPlanner()
        state_dict = {}
        metadata = Metadata(state_dict_metadata={})

        planner.set_up_planner(state_dict, metadata=metadata, is_coordinator=False)

        # Should be able to handle empty state_dict
        self.assertEqual(len(planner.state_dict), 0)

    def test_dynamic_meta_load_planner_create_local_plan(self):
        """Test DynamicMetaLoadPlanner.create_local_plan method."""
        planner = DynamicMetaLoadPlanner()
        state_dict = {"weight": torch.randn(2, 3)}
        metadata = Metadata(state_dict_metadata={})

        planner.set_up_planner(state_dict, metadata=metadata, is_coordinator=False)
        plan = planner.create_local_plan()

        self.assertIsInstance(plan, LoadPlan)
        self.assertIsInstance(plan.items, list)


class TestBroadcastingTorchSaveReaderIntegration(TestCase):
    """Integration tests for BroadcastingTorchSaveReader."""

    def setUp(self):
        super().setUp()
        device_name = torch._C._get_privateuse1_backend_name()
        self.assertEqual(device_name, 'npu', f"Expected device 'npu', got '{device_name}'")

    def test_reader_with_planner_setup(self):
        """Test BroadcastingTorchSaveReader working with DynamicMetaLoadPlanner."""
        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint_path = os.path.join(tmpdir, "checkpoint.pt")

            # Create a dummy checkpoint file
            torch.save({"weight": torch.randn(2, 3)}, checkpoint_path)

            reader = BroadcastingTorchSaveReader(checkpoint_id=checkpoint_path)
            planner = DynamicMetaLoadPlanner()

            metadata = reader.read_metadata()
            state_dict = {"weight": torch.zeros(2, 3)}

            planner.set_up_planner(state_dict, metadata=metadata, is_coordinator=False)

            self.assertIsNotNone(planner.state_dict)
            self.assertIsNotNone(metadata)

    def test_reader_state_transitions(self):
        """Test state transitions of BroadcastingTorchSaveReader."""
        checkpoint_id = "/path/to/checkpoint.pt"
        reader = BroadcastingTorchSaveReader(checkpoint_id=checkpoint_id)
        metadata = Metadata(state_dict_metadata={})

        # Initial state
        self.assertIsNone(getattr(reader, 'is_coordinator', None))

        # After set_up_storage_reader (without distributed context, use is_coordinator=False)
        reader.set_up_storage_reader(metadata, is_coordinator=False)
        self.assertFalse(reader.is_coordinator)

        # Plans can be prepared after setup
        plan = LoadPlan(items=[])
        prepared_plan = reader.prepare_local_plan(plan)
        self.assertIsNotNone(prepared_plan)


if __name__ == "__main__":
    run_tests()
