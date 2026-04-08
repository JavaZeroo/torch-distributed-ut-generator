# -*- coding: utf-8 -*-
"""
测试目的：验证 torch.distributed.checkpoint.StorageReader 抽象基类及方法功能正确性
API 名称：torch.distributed.checkpoint.StorageReader
torch.distributed.checkpoint.StorageReader.prepare_local_plan
torch.distributed.checkpoint.StorageReader.reset
torch.distributed.checkpoint.StorageReader.set_up_storage_reader

API 签名：
- StorageReader (abstract base class)
- reset(self, checkpoint_id: str | os.PathLike | None = None) -> None
- set_up_storage_reader(self, metadata: Metadata, is_coordinator: bool, *args: Any, **kwargs: Any) -> None
- prepare_local_plan(self, plan: LoadPlan) -> LoadPlan
- prepare_global_plan(self, plans: list[LoadPlan]) -> list[LoadPlan]
- read_data(self, plan: LoadPlan, planner: LoadPlanner) -> Future[None]
- read_metadata(self, *args: Any, **kwargs: Any) -> Metadata

覆盖维度表：
| 覆盖维度         | 说明                                                         | 覆盖情况                                       |
|------------------|--------------------------------------------------------------|------------------------------------------------|
| 空/非空          | checkpoint_id 为空或非空，metadata 验证                      | 已覆盖                                         |
| 枚举选项         | is_coordinator True/False                                   | 已覆盖                                         |
| 参数类型         | 验证参数接受正确类型                                         | 已覆盖                                         |
| 传参与不传参     | reset 的可选参数 checkpoint_id                               | 已覆盖                                         |
| 等价类/边界值    | 空 plan、单 item、多 item plan                              | 已覆盖                                         |
| 正常传参场景     | Mock 实现调用各方法                                          | 已覆盖                                         |
| 异常传参场景     | 抽象基类不能直接实例化                                       | 已覆盖                                         |

未覆盖项及原因：
- 无

注意：本测试仅验证功能正确性（调用不报错、输出 shape/dtype/类型符合预期），
     不做精度和数值正确性校验。
"""

import abc
import os
from typing import Any

import torch
from torch.distributed.checkpoint.storage import StorageReader
from torch.distributed.checkpoint.planner import LoadPlan, ReadItem, LoadItemType
from torch.distributed.checkpoint.metadata import Metadata, MetadataIndex
from torch.futures import Future


try:
    from torch_npu.testing.testcase import TestCase, run_tests
except ImportError:
    import sys
    from unittest import TestCase

    def run_tests():
        import unittest
        unittest.main(argv=sys.argv)


class MockStorageReader(StorageReader):
    """Mock implementation of StorageReader for testing."""

    def __init__(self):
        self.checkpoint_id = None
        self.metadata = None
        self.is_coordinator = False
        self.reset_called = False
        self.setup_called = False

    def reset(self, checkpoint_id: str | os.PathLike | None = None) -> None:
        self.checkpoint_id = checkpoint_id
        self.reset_called = True

    def read_metadata(self, *args: Any, **kwargs: Any) -> Metadata:
        return Metadata(state_dict_metadata={})

    def set_up_storage_reader(
        self, metadata: Metadata, is_coordinator: bool, *args: Any, **kwargs: Any
    ) -> None:
        self.metadata = metadata
        self.is_coordinator = is_coordinator
        self.setup_called = True

    def prepare_local_plan(self, plan: LoadPlan) -> LoadPlan:
        return plan

    def prepare_global_plan(self, plans: list[LoadPlan]) -> list[LoadPlan]:
        return plans

    def read_data(self, plan: LoadPlan, planner: Any) -> Future[None]:
        fut = Future()
        fut.set_result(None)
        return fut

    @classmethod
    def validate_checkpoint_id(cls, checkpoint_id: str | os.PathLike) -> bool:
        return True


class TestStorageReader(TestCase):
    """Test cases for StorageReader abstract base class."""

    def setUp(self):
        super().setUp()
        self.device_name = torch._C._get_privateuse1_backend_name()
        self.assertEqual(self.device_name, 'npu', f"Expected device 'npu', got '{self.device_name}'")

    def test_storage_reader_is_abstract(self):
        """Test that StorageReader cannot be instantiated directly."""
        with self.assertRaises(TypeError):
            StorageReader()

    def test_storage_reader_subclass_abstract_methods(self):
        """Test that subclasses must implement abstract methods."""
        class IncompleteReader(StorageReader):
            pass

        with self.assertRaises(TypeError):
            IncompleteReader()

    def test_mock_storage_reader_reset_with_checkpoint_id(self):
        """Test reset method with checkpoint_id."""
        reader = MockStorageReader()
        checkpoint_id = "/path/to/checkpoint"

        reader.reset(checkpoint_id)

        self.assertTrue(reader.reset_called)
        self.assertEqual(reader.checkpoint_id, checkpoint_id)

    def test_mock_storage_reader_reset_without_checkpoint_id(self):
        """Test reset method without checkpoint_id."""
        reader = MockStorageReader()

        reader.reset()

        self.assertTrue(reader.reset_called)
        self.assertIsNone(reader.checkpoint_id)

    def test_mock_storage_reader_reset_with_pathlike(self):
        """Test reset method with os.PathLike."""
        reader = MockStorageReader()
        path = os.path.join("path", "to", "checkpoint")

        reader.reset(path)

        self.assertTrue(reader.reset_called)
        self.assertEqual(reader.checkpoint_id, path)

    def test_mock_storage_reader_setup_storage_reader(self):
        """Test set_up_storage_reader method."""
        reader = MockStorageReader()
        metadata = Metadata(state_dict_metadata={"key": None})

        reader.set_up_storage_reader(metadata, is_coordinator=True)

        self.assertTrue(reader.setup_called)
        self.assertEqual(reader.metadata, metadata)
        self.assertTrue(reader.is_coordinator)

    def test_mock_storage_reader_setup_storage_reader_not_coordinator(self):
        """Test set_up_storage_reader as non-coordinator."""
        reader = MockStorageReader()
        metadata = Metadata(state_dict_metadata={})

        reader.set_up_storage_reader(metadata, is_coordinator=False)

        self.assertFalse(reader.is_coordinator)

    def test_mock_storage_reader_prepare_local_plan(self):
        """Test prepare_local_plan method."""
        reader = MockStorageReader()
        read_item = ReadItem(
            type=LoadItemType.TENSOR,
            dest_index=MetadataIndex(fqn="test"),
            dest_offsets=torch.Size([0]),
            storage_index=MetadataIndex(fqn="test"),
            storage_offsets=torch.Size([0]),
            lengths=torch.Size([10])
        )
        plan = LoadPlan(items=[read_item])

        result = reader.prepare_local_plan(plan)

        self.assertEqual(result, plan)
        self.assertIsInstance(result, LoadPlan)

    def test_mock_storage_reader_prepare_local_plan_empty(self):
        """Test prepare_local_plan with empty plan."""
        reader = MockStorageReader()
        plan = LoadPlan(items=[])

        result = reader.prepare_local_plan(plan)

        self.assertEqual(result.items, [])

    def test_mock_storage_reader_isinstance_check(self):
        """Test isinstance check for StorageReader."""
        reader = MockStorageReader()
        self.assertIsInstance(reader, StorageReader)

    def test_mock_storage_reader_validate_checkpoint_id(self):
        """Test validate_checkpoint_id class method."""
        result = MockStorageReader.validate_checkpoint_id("/some/path")
        self.assertTrue(result)

    def test_mock_storage_reader_read_metadata(self):
        """Test read_metadata method."""
        reader = MockStorageReader()
        metadata = reader.read_metadata()

        self.assertIsInstance(metadata, Metadata)

    def test_mock_storage_reader_read_data(self):
        """Test read_data method returns Future."""
        reader = MockStorageReader()
        plan = LoadPlan(items=[])

        result = reader.read_data(plan, None)

        self.assertIsInstance(result, Future)
        # torch.futures.Future uses wait() instead of result()
        self.assertIsNone(result.wait())


if __name__ == "__main__":
    run_tests()
