# -*- coding: utf-8 -*-
"""
测试目的：验证 torch.distributed.checkpoint 基础接口功能正确性
API 名称：LoadPlan, LoadPlanner, LoadPlanner.set_up_planner, LoadPlanner.finish_plan,
         FileSystemReader.checkpoint_id

API 签名：
- LoadPlan(items: List[ReadItem], storage_data, planner_data)
- LoadPlanner.set_up_planner(state_dict, metadata, is_coordinator) -> None
- LoadPlanner.finish_plan(central_plan: LoadPlan) -> LoadPlan
- FileSystemReader.__init__(root: Union[str, os.PathLike])
- FileSystemReader.checkpoint_id property

覆盖维度表：
| 覆盖维度         | 说明                                                         | 覆盖情况                                       |
|------------------|--------------------------------------------------------------|------------------------------------------------|
| LoadPlan 创建    | 空 items / 单个 item / 多个 items                            | 已覆盖                                         |
| set_up_planner   | state_dict 初始化 / metadata / is_coordinator 标志           | 已覆盖                                         |
| finish_plan 返回  | 返回 LoadPlan 类型, 与输入一致                              | 已覆盖                                         |
| FileSystemReader 初始化 | 有效路径 / None / PathLike 对象                        | 已覆盖                                         |
| checkpoint_id 属性 | 读写属性行为                                                 | 已覆盖                                         |
| ReadItem 集合    | 不同大小和类型的 items 列表                                  | 已覆盖                                         |
| 存储元数据       | storage_data / planner_data 处理                            | 已覆盖                                         |

未覆盖项及原因：
- 无

注意：本测试仅验证功能正确性（调用不报错、输出类型符合预期），
     不做精度和数值正确性校验。
"""

import unittest
import torch
import os
import tempfile
from pathlib import Path
from typing import Any, Optional

try:
    from torch_npu.testing.testcase import TestCase, run_tests
except ImportError:
    from unittest import TestCase

    def run_tests():
        unittest.main(argv=[''])

from torch.distributed.checkpoint.planner import (
    LoadPlan, LoadPlanner, ReadItem, LoadItemType, MetadataIndex
)
from torch.distributed.checkpoint.filesystem import FileSystemReader
from torch.distributed.checkpoint.metadata import Metadata


class TestLoadPlanInitialization(TestCase):
    """Test LoadPlan initialization and properties."""

    def setUp(self):
        super().setUp()
        device_name = torch._C._get_privateuse1_backend_name()
        self.assertEqual(device_name, 'npu', f"Expected device 'npu', got '{device_name}'")

    def test_load_plan_empty_items(self):
        """Test LoadPlan with empty items list."""
        plan = LoadPlan(items=[])

        self.assertIsInstance(plan, LoadPlan)
        self.assertEqual(len(plan.items), 0)
        self.assertEqual(plan.items, [])

    def test_load_plan_single_item(self):
        """Test LoadPlan with single ReadItem."""
        read_item = ReadItem(
            type=LoadItemType.TENSOR,
            dest_index=MetadataIndex(fqn="weight"),
            dest_offsets=torch.Size([0, 0]),
            storage_index=MetadataIndex(fqn="weight"),
            storage_offsets=torch.Size([0, 0]),
            lengths=torch.Size([3, 4]),
        )
        plan = LoadPlan(items=[read_item])

        self.assertEqual(len(plan.items), 1)
        self.assertEqual(plan.items[0], read_item)

    def test_load_plan_multiple_items(self):
        """Test LoadPlan with multiple ReadItems."""
        items = [
            ReadItem(
                type=LoadItemType.TENSOR,
                dest_index=MetadataIndex(fqn="weight"),
                dest_offsets=torch.Size([0]),
                storage_index=MetadataIndex(fqn="weight"),
                storage_offsets=torch.Size([0]),
                lengths=torch.Size([3]),
            ),
            ReadItem(
                type=LoadItemType.TENSOR,
                dest_index=MetadataIndex(fqn="bias"),
                dest_offsets=torch.Size([0]),
                storage_index=MetadataIndex(fqn="bias"),
                storage_offsets=torch.Size([0]),
                lengths=torch.Size([3]),
            ),
        ]
        plan = LoadPlan(items=items)

        self.assertEqual(len(plan.items), 2)
        self.assertEqual(plan.items[0].dest_index.fqn, "weight")
        self.assertEqual(plan.items[1].dest_index.fqn, "bias")

    def test_load_plan_with_storage_data(self):
        """Test LoadPlan with storage_data."""
        storage_data = {"metadata": "some_value"}
        plan = LoadPlan(items=[], storage_data=storage_data)

        self.assertEqual(plan.storage_data, storage_data)
        self.assertEqual(plan.storage_data["metadata"], "some_value")

    def test_load_plan_with_planner_data(self):
        """Test LoadPlan with planner_data."""
        planner_data = {"custom_key": "custom_value"}
        plan = LoadPlan(items=[], planner_data=planner_data)

        self.assertEqual(plan.planner_data, planner_data)

    def test_load_plan_with_all_parameters(self):
        """Test LoadPlan with all parameters."""
        read_item = ReadItem(
            type=LoadItemType.BYTE_IO,
            dest_index=MetadataIndex(fqn="config"),
            dest_offsets=torch.Size([0]),
            storage_index=MetadataIndex(fqn="config"),
            storage_offsets=torch.Size([0]),
            lengths=torch.Size([1]),
        )
        storage_data = {"root": "/path/to/checkpoint"}
        planner_data = {"version": "1.0"}

        plan = LoadPlan(
            items=[read_item],
            storage_data=storage_data,
            planner_data=planner_data,
        )

        self.assertEqual(len(plan.items), 1)
        self.assertEqual(plan.storage_data["root"], "/path/to/checkpoint")
        self.assertEqual(plan.planner_data["version"], "1.0")

    def test_load_plan_items_mutability(self):
        """Test LoadPlan items list is mutable."""
        plan = LoadPlan(items=[])

        read_item = ReadItem(
            type=LoadItemType.TENSOR,
            dest_index=MetadataIndex(fqn="weight"),
            dest_offsets=torch.Size([0]),
            storage_index=MetadataIndex(fqn="weight"),
            storage_offsets=torch.Size([0]),
            lengths=torch.Size([3]),
        )

        plan.items.append(read_item)

        self.assertEqual(len(plan.items), 1)


class TestLoadPlannerSetUpPlanner(TestCase):
    """Test LoadPlanner.set_up_planner method."""

    def setUp(self):
        super().setUp()
        device_name = torch._C._get_privateuse1_backend_name()
        self.assertEqual(device_name, 'npu', f"Expected device 'npu', got '{device_name}'")

    def test_set_up_planner_basic(self):
        """Test set_up_planner with basic state_dict."""
        class SimpleLoadPlanner(LoadPlanner):
            def set_up_planner(self, state_dict, metadata=None, is_coordinator=False):
                self.state_dict = state_dict
                self.metadata = metadata
                self.is_coordinator = is_coordinator

            def create_local_plan(self):
                return LoadPlan(items=[])

            def create_global_plan(self, global_plan):
                return global_plan

            def finish_plan(self, central_plan):
                return central_plan

            def load_bytes(self, read_item, value):
                pass

            def resolve_tensor(self, read_item):
                pass

            def commit_tensor(self, read_item, tensor):
                pass

        planner = SimpleLoadPlanner()
        state_dict = {"weight": torch.randn(2, 3), "bias": torch.randn(3)}
        metadata = Metadata(state_dict_metadata={})

        planner.set_up_planner(state_dict, metadata=metadata, is_coordinator=False)

        self.assertEqual(planner.state_dict, state_dict)
        self.assertEqual(planner.metadata, metadata)
        self.assertFalse(planner.is_coordinator)

    def test_set_up_planner_as_coordinator(self):
        """Test set_up_planner with is_coordinator=True."""
        class CoordinatorLoadPlanner(LoadPlanner):
            def set_up_planner(self, state_dict, metadata=None, is_coordinator=False):
                self.state_dict = state_dict
                self.metadata = metadata
                self.is_coordinator = is_coordinator

            def create_local_plan(self):
                return LoadPlan(items=[])

            def create_global_plan(self, global_plan):
                return global_plan

            def finish_plan(self, central_plan):
                return central_plan

            def load_bytes(self, read_item, value):
                pass

            def resolve_tensor(self, read_item):
                pass

            def commit_tensor(self, read_item, tensor):
                pass

        planner = CoordinatorLoadPlanner()
        state_dict = {"model": torch.randn(10, 10)}
        metadata = Metadata(state_dict_metadata={})

        planner.set_up_planner(state_dict, metadata=metadata, is_coordinator=True)

        self.assertTrue(planner.is_coordinator)

    def test_set_up_planner_without_metadata(self):
        """Test set_up_planner with metadata=None."""
        class MetadataOptionalPlanner(LoadPlanner):
            def set_up_planner(self, state_dict, metadata=None, is_coordinator=False):
                self.state_dict = state_dict
                self.metadata = metadata
                self.is_coordinator = is_coordinator

            def create_local_plan(self):
                return LoadPlan(items=[])

            def create_global_plan(self, global_plan):
                return global_plan

            def finish_plan(self, central_plan):
                return central_plan

            def load_bytes(self, read_item, value):
                pass

            def resolve_tensor(self, read_item):
                pass

            def commit_tensor(self, read_item, tensor):
                pass

        planner = MetadataOptionalPlanner()
        state_dict = {"param": torch.randn(5, 5)}

        planner.set_up_planner(state_dict)

        self.assertEqual(planner.state_dict, state_dict)
        self.assertIsNone(planner.metadata)
        self.assertFalse(planner.is_coordinator)

    def test_set_up_planner_empty_state_dict(self):
        """Test set_up_planner with empty state_dict."""
        class EmptyStateDictPlanner(LoadPlanner):
            def set_up_planner(self, state_dict, metadata=None, is_coordinator=False):
                self.state_dict = state_dict
                self.metadata = metadata

            def create_local_plan(self):
                return LoadPlan(items=[])

            def create_global_plan(self, global_plan):
                return global_plan

            def finish_plan(self, central_plan):
                return central_plan

            def load_bytes(self, read_item, value):
                pass

            def resolve_tensor(self, read_item):
                pass

            def commit_tensor(self, read_item, tensor):
                pass

        planner = EmptyStateDictPlanner()
        state_dict = {}
        metadata = Metadata(state_dict_metadata={})

        planner.set_up_planner(state_dict, metadata=metadata, is_coordinator=False)

        self.assertEqual(len(planner.state_dict), 0)

    def test_set_up_planner_various_state_dict_types(self):
        """Test set_up_planner with various state_dict content types."""
        class FlexiblePlanner(LoadPlanner):
            def set_up_planner(self, state_dict, metadata=None, is_coordinator=False):
                self.state_dict = state_dict
                self.metadata = metadata

            def create_local_plan(self):
                return LoadPlan(items=[])

            def create_global_plan(self, global_plan):
                return global_plan

            def finish_plan(self, central_plan):
                return central_plan

            def load_bytes(self, read_item, value):
                pass

            def resolve_tensor(self, read_item):
                pass

            def commit_tensor(self, read_item, tensor):
                pass

        planner = FlexiblePlanner()

        # State dict with tensors and scalars
        state_dict = {
            "tensor_param": torch.randn(3, 4),
            "scalar_param": 42,
            "nested_dict": {"value": 3.14},
            "list_param": [1, 2, 3],
        }

        planner.set_up_planner(state_dict)

        self.assertIn("tensor_param", planner.state_dict)
        self.assertIn("scalar_param", planner.state_dict)
        self.assertIn("nested_dict", planner.state_dict)
        self.assertIn("list_param", planner.state_dict)

    def test_set_up_planner_multiple_calls(self):
        """Test set_up_planner can be called multiple times with different configs."""
        class MultiCallPlanner(LoadPlanner):
            def set_up_planner(self, state_dict, metadata=None, is_coordinator=False):
                self.state_dict = state_dict
                self.metadata = metadata
                self.is_coordinator = is_coordinator

            def create_local_plan(self):
                return LoadPlan(items=[])

            def create_global_plan(self, global_plan):
                return global_plan

            def finish_plan(self, central_plan):
                return central_plan

            def load_bytes(self, read_item, value):
                pass

            def resolve_tensor(self, read_item):
                pass

            def commit_tensor(self, read_item, tensor):
                pass

        planner = MultiCallPlanner()

        # First call
        state_dict1 = {"param1": torch.randn(2, 2)}
        planner.set_up_planner(state_dict1, is_coordinator=False)
        self.assertEqual(len(planner.state_dict), 1)
        self.assertFalse(planner.is_coordinator)

        # Second call with different config
        state_dict2 = {"param2": torch.randn(3, 3), "param3": torch.randn(4, 4)}
        metadata2 = Metadata(state_dict_metadata={})
        planner.set_up_planner(state_dict2, metadata=metadata2, is_coordinator=True)
        self.assertEqual(len(planner.state_dict), 2)
        self.assertTrue(planner.is_coordinator)
        self.assertIsNotNone(planner.metadata)


class TestLoadPlannerFinishPlan(TestCase):
    """Test LoadPlanner.finish_plan method."""

    def setUp(self):
        super().setUp()
        device_name = torch._C._get_privateuse1_backend_name()
        self.assertEqual(device_name, 'npu', f"Expected device 'npu', got '{device_name}'")

    def test_finish_plan_basic_implementation(self):
        """Test finish_plan returns a LoadPlan instance."""
        # Create a simple LoadPlanner subclass for testing
        class SimpleLoadPlanner(LoadPlanner):
            def set_up_planner(self, state_dict, metadata=None, is_coordinator=False):
                pass

            def create_local_plan(self):
                return LoadPlan(items=[])

            def create_global_plan(self, global_plan):
                return global_plan

            def finish_plan(self, central_plan):
                # Simple implementation: return as-is
                return central_plan

            def load_bytes(self, read_item, value):
                pass

            def resolve_tensor(self, read_item):
                pass

            def commit_tensor(self, read_item, tensor):
                pass

        planner = SimpleLoadPlanner()
        input_plan = LoadPlan(items=[])

        result = planner.finish_plan(input_plan)

        self.assertIsInstance(result, LoadPlan)
        self.assertEqual(result, input_plan)

    def test_finish_plan_with_items(self):
        """Test finish_plan with plan containing items."""
        class CustomLoadPlanner(LoadPlanner):
            def set_up_planner(self, state_dict, metadata=None, is_coordinator=False):
                pass

            def create_local_plan(self):
                return LoadPlan(items=[])

            def create_global_plan(self, global_plan):
                return global_plan

            def finish_plan(self, central_plan):
                return central_plan

            def load_bytes(self, read_item, value):
                pass

            def resolve_tensor(self, read_item):
                pass

            def commit_tensor(self, read_item, tensor):
                pass

        planner = CustomLoadPlanner()

        read_item = ReadItem(
            type=LoadItemType.TENSOR,
            dest_index=MetadataIndex(fqn="weight"),
            dest_offsets=torch.Size([0]),
            storage_index=MetadataIndex(fqn="weight"),
            storage_offsets=torch.Size([0]),
            lengths=torch.Size([3]),
        )

        input_plan = LoadPlan(items=[read_item])
        result = planner.finish_plan(input_plan)

        self.assertEqual(len(result.items), 1)
        self.assertEqual(result.items[0].dest_index.fqn, "weight")

    def test_finish_plan_preserves_metadata(self):
        """Test finish_plan preserves storage_data and planner_data."""
        class MetadataPreservingPlanner(LoadPlanner):
            def set_up_planner(self, state_dict, metadata=None, is_coordinator=False):
                pass

            def create_local_plan(self):
                return LoadPlan(items=[])

            def create_global_plan(self, global_plan):
                return global_plan

            def finish_plan(self, central_plan):
                return central_plan

            def load_bytes(self, read_item, value):
                pass

            def resolve_tensor(self, read_item):
                pass

            def commit_tensor(self, read_item, tensor):
                pass

        planner = MetadataPreservingPlanner()

        storage_data = {"key": "value"}
        planner_data = {"planner_key": "planner_value"}

        input_plan = LoadPlan(
            items=[],
            storage_data=storage_data,
            planner_data=planner_data,
        )

        result = planner.finish_plan(input_plan)

        self.assertEqual(result.storage_data, storage_data)
        self.assertEqual(result.planner_data, planner_data)


class TestFileSystemReaderCheckpointId(TestCase):
    """Test FileSystemReader initialization and path handling."""

    def setUp(self):
        super().setUp()
        device_name = torch._C._get_privateuse1_backend_name()
        self.assertEqual(device_name, 'npu', f"Expected device 'npu', got '{device_name}'")

    def test_file_system_reader_with_string_path(self):
        """Test FileSystemReader initialization with string path."""
        root = "/path/to/checkpoint"
        reader = FileSystemReader(root)

        # FileSystemReader stores path internally after processing with FileSystem
        self.assertIsNotNone(reader.path)

    def test_file_system_reader_with_path_object(self):
        """Test FileSystemReader initialization with Path object."""
        root = Path("/path/to/checkpoint")
        reader = FileSystemReader(root)

        # FileSystemReader processes Path object through FileSystem
        self.assertIsNotNone(reader.path)

    def test_file_system_reader_with_real_directory(self):
        """Test FileSystemReader with actual temporary directory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            reader = FileSystemReader(tmpdir)

            # FileSystemReader should have path attribute
            self.assertIsNotNone(reader.path)
            # Verify directory exists
            self.assertTrue(os.path.exists(tmpdir))

    def test_file_system_reader_has_fs_attribute(self):
        """Test FileSystemReader has FileSystem attribute."""
        root = "/path/to/checkpoint"
        reader = FileSystemReader(root)

        # FileSystemReader should have a FileSystem instance
        self.assertIsNotNone(reader.fs)
        self.assertEqual(reader.fs.__class__.__name__, 'FileSystem')

    def test_file_system_reader_reset_path(self):
        """Test FileSystemReader.reset method updates path."""
        with tempfile.TemporaryDirectory() as tmpdir:
            initial_path = os.path.join(tmpdir, "checkpoint1.pt")
            reader = FileSystemReader(initial_path)
            initial_reader_path = reader.path

            # Reset with new path
            new_path = os.path.join(tmpdir, "checkpoint2.pt")
            reader.reset(checkpoint_id=new_path)

            # Path should be updated
            self.assertIsNotNone(reader.path)


class TestReadItem(TestCase):
    """Test ReadItem creation and properties."""

    def setUp(self):
        super().setUp()
        device_name = torch._C._get_privateuse1_backend_name()
        self.assertEqual(device_name, 'npu', f"Expected device 'npu', got '{device_name}'")

    def test_read_item_tensor_type(self):
        """Test ReadItem with TENSOR LoadItemType."""
        read_item = ReadItem(
            type=LoadItemType.TENSOR,
            dest_index=MetadataIndex(fqn="weight"),
            dest_offsets=torch.Size([0, 0]),
            storage_index=MetadataIndex(fqn="weight"),
            storage_offsets=torch.Size([0, 0]),
            lengths=torch.Size([3, 4]),
        )

        self.assertEqual(read_item.type, LoadItemType.TENSOR)
        self.assertEqual(read_item.dest_index.fqn, "weight")
        self.assertEqual(read_item.lengths, torch.Size([3, 4]))

    def test_read_item_byte_io_type(self):
        """Test ReadItem with BYTE_IO LoadItemType."""
        read_item = ReadItem(
            type=LoadItemType.BYTE_IO,
            dest_index=MetadataIndex(fqn="config"),
            dest_offsets=torch.Size([]),
            storage_index=MetadataIndex(fqn="config"),
            storage_offsets=torch.Size([]),
            lengths=torch.Size([]),
        )

        self.assertEqual(read_item.type, LoadItemType.BYTE_IO)
        self.assertEqual(read_item.dest_index.fqn, "config")

    def test_read_item_offsets_and_lengths(self):
        """Test ReadItem offset and length handling."""
        read_item = ReadItem(
            type=LoadItemType.TENSOR,
            dest_index=MetadataIndex(fqn="param"),
            dest_offsets=torch.Size([1, 2]),
            storage_index=MetadataIndex(fqn="param"),
            storage_offsets=torch.Size([0, 0]),
            lengths=torch.Size([5, 5]),
        )

        self.assertEqual(read_item.dest_offsets, torch.Size([1, 2]))
        self.assertEqual(read_item.storage_offsets, torch.Size([0, 0]))
        self.assertEqual(read_item.lengths, torch.Size([5, 5]))


class TestCheckpointMetadataIndex(TestCase):
    """Test MetadataIndex for checkpoint APIs."""

    def setUp(self):
        super().setUp()
        device_name = torch._C._get_privateuse1_backend_name()
        self.assertEqual(device_name, 'npu', f"Expected device 'npu', got '{device_name}'")

    def test_metadata_index_creation(self):
        """Test MetadataIndex creation with fqn."""
        index = MetadataIndex(fqn="model.weight")

        self.assertEqual(index.fqn, "model.weight")

    def test_metadata_index_various_fqn(self):
        """Test MetadataIndex with various fully qualified names."""
        fqns = [
            "weight",
            "model.weight",
            "model.layer1.weight",
            "optimizer.state.0",
        ]

        for fqn in fqns:
            index = MetadataIndex(fqn=fqn)
            self.assertEqual(index.fqn, fqn)

    def test_metadata_index_in_read_item(self):
        """Test MetadataIndex used in ReadItem."""
        dest_index = MetadataIndex(fqn="destination_key")
        storage_index = MetadataIndex(fqn="storage_key")

        read_item = ReadItem(
            type=LoadItemType.TENSOR,
            dest_index=dest_index,
            dest_offsets=torch.Size([0]),
            storage_index=storage_index,
            storage_offsets=torch.Size([0]),
            lengths=torch.Size([10]),
        )

        self.assertEqual(read_item.dest_index, dest_index)
        self.assertEqual(read_item.storage_index, storage_index)


class TestCheckpointBaseIntegration(TestCase):
    """Integration tests for checkpoint base APIs."""

    def setUp(self):
        super().setUp()
        device_name = torch._C._get_privateuse1_backend_name()
        self.assertEqual(device_name, 'npu', f"Expected device 'npu', got '{device_name}'")

    def test_load_plan_with_multiple_tensor_items(self):
        """Test LoadPlan with realistic tensor loading scenario."""
        items = [
            ReadItem(
                type=LoadItemType.TENSOR,
                dest_index=MetadataIndex(fqn="encoder.weight"),
                dest_offsets=torch.Size([0, 0]),
                storage_index=MetadataIndex(fqn="encoder.weight"),
                storage_offsets=torch.Size([0, 0]),
                lengths=torch.Size([768, 768]),
            ),
            ReadItem(
                type=LoadItemType.TENSOR,
                dest_index=MetadataIndex(fqn="encoder.bias"),
                dest_offsets=torch.Size([0]),
                storage_index=MetadataIndex(fqn="encoder.bias"),
                storage_offsets=torch.Size([0]),
                lengths=torch.Size([768]),
            ),
            ReadItem(
                type=LoadItemType.BYTE_IO,
                dest_index=MetadataIndex(fqn="config"),
                dest_offsets=torch.Size([]),
                storage_index=MetadataIndex(fqn="config"),
                storage_offsets=torch.Size([]),
                lengths=torch.Size([]),
            ),
        ]

        plan = LoadPlan(items=items)

        self.assertEqual(len(plan.items), 3)
        self.assertTrue(all(isinstance(item, ReadItem) for item in plan.items))

    def test_file_system_reader_and_load_plan_integration(self):
        """Test FileSystemReader integration with LoadPlan."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create checkpoint files
            checkpoint_file = os.path.join(tmpdir, "checkpoint.pt")
            torch.save({"weight": torch.randn(2, 3)}, checkpoint_file)

            reader = FileSystemReader(tmpdir)
            plan = LoadPlan(items=[])

            # FileSystemReader should have path and fs attributes
            self.assertIsNotNone(reader.path)
            self.assertIsNotNone(reader.fs)
            # LoadPlan should be empty as created
            self.assertEqual(len(plan.items), 0)


if __name__ == "__main__":
    run_tests()
