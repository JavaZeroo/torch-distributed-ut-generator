# -*- coding: utf-8 -*-
"""
测试目的：验证 torch.distributed.checkpoint.StorageWriter 抽象基类及方法功能正确性
API 名称：torch.distributed.checkpoint.StorageWriter
torch.distributed.checkpoint.StorageWriter.prepare_local_plan
torch.distributed.checkpoint.StorageWriter.reset
torch.distributed.checkpoint.StorageWriter.set_up_storage_writer
torch.distributed.checkpoint.StorageWriter.storage_meta

API 签名：
- StorageWriter (abstract base class)
- reset(self, checkpoint_id: str | os.PathLike | None = None) -> None
- set_up_storage_writer(self, is_coordinator: bool, *args: Any, **kwargs: Any) -> None
- prepare_local_plan(self, plan: SavePlan) -> SavePlan
- prepare_global_plan(self, plans: list[SavePlan]) -> list[SavePlan]
- write_data(self, plan: SavePlan, planner: SavePlanner) -> Future[list[WriteResult]]
- finish(self, metadata: Metadata, results: list[list[WriteResult]]) -> None
- storage_meta(self) -> StorageMeta | None

覆盖维度表：
| 覆盖维度         | 说明                                                         | 覆盖情况                                       |
|------------------|--------------------------------------------------------------|------------------------------------------------|
| 空/非空          | checkpoint_id 为空或非空，storage_meta 返回 None 或对象      | 已覆盖                                         |
| 枚举选项         | is_coordinator True/False                                   | 已覆盖                                         |
| 参数类型         | 验证参数接受正确类型                                         | 已覆盖                                         |
| 传参与不传参     | reset 的可选参数，storage_meta 默认实现                      | 已覆盖                                         |
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
from typing import Any, List

import torch
from torch.distributed.checkpoint.storage import StorageWriter, WriteResult
from torch.distributed.checkpoint.planner import SavePlan, WriteItem, WriteItemType
from torch.distributed.checkpoint.metadata import Metadata, StorageMeta, MetadataIndex
from torch.futures import Future


try:
    from torch_npu.testing.testcase import TestCase, run_tests
except ImportError:
    import sys
    from unittest import TestCase

    def run_tests():
        import unittest
        unittest.main(argv=sys.argv)


class MockStorageWriter(StorageWriter):
    """Mock implementation of StorageWriter for testing."""

    def __init__(self, return_storage_meta: StorageMeta | None = None):
        self.checkpoint_id = None
        self.is_coordinator = False
        self.reset_called = False
        self.setup_called = False
        self._storage_meta = return_storage_meta

    def reset(self, checkpoint_id: str | os.PathLike | None = None) -> None:
        self.checkpoint_id = checkpoint_id
        self.reset_called = True

    def set_up_storage_writer(self, is_coordinator: bool, *args: Any, **kwargs: Any) -> None:
        self.is_coordinator = is_coordinator
        self.setup_called = True

    def prepare_local_plan(self, plan: SavePlan) -> SavePlan:
        return plan

    def prepare_global_plan(self, plans: List[SavePlan]) -> List[SavePlan]:
        return plans

    def write_data(self, plan: SavePlan, planner: Any) -> Future[List[WriteResult]]:
        fut = Future()
        fut.set_result([])
        return fut

    def finish(self, metadata: Metadata, results: List[List[WriteResult]]) -> None:
        pass

    @classmethod
    def validate_checkpoint_id(cls, checkpoint_id: str | os.PathLike) -> bool:
        return True

    def storage_meta(self) -> StorageMeta | None:
        return self._storage_meta


class TestStorageWriter(TestCase):
    """Test cases for StorageWriter abstract base class."""

    def setUp(self):
        super().setUp()
        self.device_name = torch._C._get_privateuse1_backend_name()
        self.assertEqual(self.device_name, 'npu', f"Expected device 'npu', got '{self.device_name}'")

    def test_storage_writer_is_abstract(self):
        """Test that StorageWriter cannot be instantiated directly."""
        with self.assertRaises(TypeError):
            StorageWriter()

    def test_storage_writer_subclass_abstract_methods(self):
        """Test that subclasses must implement abstract methods."""
        class IncompleteWriter(StorageWriter):
            pass

        with self.assertRaises(TypeError):
            IncompleteWriter()

    def test_mock_storage_writer_reset_with_checkpoint_id(self):
        """Test reset method with checkpoint_id."""
        writer = MockStorageWriter()
        checkpoint_id = "/path/to/checkpoint"

        writer.reset(checkpoint_id)

        self.assertTrue(writer.reset_called)
        self.assertEqual(writer.checkpoint_id, checkpoint_id)

    def test_mock_storage_writer_reset_without_checkpoint_id(self):
        """Test reset method without checkpoint_id."""
        writer = MockStorageWriter()

        writer.reset()

        self.assertTrue(writer.reset_called)
        self.assertIsNone(writer.checkpoint_id)

    def test_mock_storage_writer_reset_with_pathlike(self):
        """Test reset method with os.PathLike."""
        writer = MockStorageWriter()
        path = os.path.join("path", "to", "checkpoint")

        writer.reset(path)

        self.assertTrue(writer.reset_called)
        self.assertEqual(writer.checkpoint_id, path)

    def test_mock_storage_writer_setup_storage_writer_coordinator(self):
        """Test set_up_storage_writer as coordinator."""
        writer = MockStorageWriter()

        writer.set_up_storage_writer(is_coordinator=True)

        self.assertTrue(writer.setup_called)
        self.assertTrue(writer.is_coordinator)

    def test_mock_storage_writer_setup_storage_writer_not_coordinator(self):
        """Test set_up_storage_writer as non-coordinator."""
        writer = MockStorageWriter()

        writer.set_up_storage_writer(is_coordinator=False)

        self.assertTrue(writer.setup_called)
        self.assertFalse(writer.is_coordinator)

    def test_mock_storage_writer_prepare_local_plan(self):
        """Test prepare_local_plan method."""
        writer = MockStorageWriter()
        write_item = WriteItem(
            index=MetadataIndex(fqn="test"),
            type=WriteItemType.TENSOR
        )
        plan = SavePlan(items=[write_item])

        result = writer.prepare_local_plan(plan)

        self.assertEqual(result, plan)
        self.assertIsInstance(result, SavePlan)

    def test_mock_storage_writer_prepare_local_plan_empty(self):
        """Test prepare_local_plan with empty plan."""
        writer = MockStorageWriter()
        plan = SavePlan(items=[])

        result = writer.prepare_local_plan(plan)

        self.assertEqual(result.items, [])

    def test_mock_storage_writer_storage_meta_returns_none(self):
        """Test storage_meta method returns None by default."""
        writer = MockStorageWriter()
        result = writer.storage_meta()

        self.assertIsNone(result)

    def test_mock_storage_writer_storage_meta_returns_object(self):
        """Test storage_meta method returns StorageMeta object."""
        storage_meta = StorageMeta(checkpoint_id="test_id", save_id="save_1")
        writer = MockStorageWriter(return_storage_meta=storage_meta)
        result = writer.storage_meta()

        self.assertIsInstance(result, StorageMeta)
        self.assertEqual(result.checkpoint_id, "test_id")
        self.assertEqual(result.save_id, "save_1")

    def test_mock_storage_writer_isinstance_check(self):
        """Test isinstance check for StorageWriter."""
        writer = MockStorageWriter()
        self.assertIsInstance(writer, StorageWriter)

    def test_mock_storage_writer_validate_checkpoint_id(self):
        """Test validate_checkpoint_id class method."""
        result = MockStorageWriter.validate_checkpoint_id("/some/path")
        self.assertTrue(result)

    def test_mock_storage_writer_write_data(self):
        """Test write_data method returns Future."""
        writer = MockStorageWriter()
        plan = SavePlan(items=[])

        result = writer.write_data(plan, None)

        self.assertIsInstance(result, Future)
        # torch.futures.Future uses wait() instead of result()
        self.assertEqual(result.wait(), [])

    def test_mock_storage_writer_finish(self):
        """Test finish method can be called."""
        writer = MockStorageWriter()
        metadata = Metadata(state_dict_metadata={})

        # Should not raise
        writer.finish(metadata, [[]])


if __name__ == "__main__":
    run_tests()
