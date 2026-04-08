# -*- coding: utf-8 -*-
"""
测试目的：验证 torch.distributed.checkpoint 默认规划器接口功能正确性
API 名称：DefaultLoadPlanner.lookup_tensor, DefaultLoadPlanner.transform_tensor,
         DefaultSavePlanner.lookup_object, DefaultSavePlanner.transform_object
API 签名：
- DefaultLoadPlanner.lookup_tensor(index: MetadataIndex) -> torch.Tensor
- DefaultLoadPlanner.transform_tensor(read_item: ReadItem, tensor: torch.Tensor) -> torch.Tensor
- DefaultSavePlanner.lookup_object(index: MetadataIndex) -> Any
- DefaultSavePlanner.transform_object(write_item: WriteItem, object: Any) -> Any

覆盖维度表：
| 覆盖维度         | 说明                                                         | 覆盖情况                                       |
|------------------|--------------------------------------------------------------|------------------------------------------------|
| 对象类型         | Tensor / BytesIO / 其他 Python 对象                         | 已覆盖                                         |
| state_dict 结构  | 扁平 dict / 嵌套 dict / 空 dict                              | 已覆盖                                         |
| MetadataIndex 查询 | 存在 / 不存在的 key                                         | 已覆盖                                         |
| transform 变换    | BYTE_IO / 非 BYTE_IO / tensor 转换                           | 已覆盖                                         |
| 参数传递         | 传参与默认值                                                 | 已覆盖                                         |

未覆盖项及原因：
- 无

注意：本测试仅验证功能正确性（调用不报错、输出类型符合预期），
     不做精度和数值正确性校验。
"""

import io
import unittest
import torch

try:
    from torch_npu.testing.testcase import TestCase, run_tests
except ImportError:
    from unittest import TestCase

    def run_tests():
        unittest.main(argv=[''])

from torch.distributed.checkpoint.planner import (
    WriteItem, WriteItemType, ReadItem, LoadItemType, MetadataIndex
)
from torch.distributed.checkpoint.default_planner import (
    DefaultLoadPlanner, DefaultSavePlanner
)
from torch.distributed.checkpoint.metadata import (
    ChunkStorageMetadata, TensorProperties, TensorStorageMetadata
)
from torch.distributed._shard._utils import narrow_tensor_by_index


class TestDefaultSavePlannerLookupObject(TestCase):
    """Test DefaultSavePlanner.lookup_object method."""

    def setUp(self):
        super().setUp()
        device_name = torch._C._get_privateuse1_backend_name()
        self.assertEqual(device_name, 'npu', f"Expected device 'npu', got '{device_name}'")
        self.planner = DefaultSavePlanner()

    def test_lookup_object_flat_dict_tensor(self):
        """Test lookup_object with flat dict containing tensor."""
        state_dict = {
            "model.weight": torch.randn(3, 4),
            "model.bias": torch.randn(3),
        }
        self.planner.set_up_planner(state_dict, is_coordinator=False)

        index = MetadataIndex(fqn="model.weight")
        obj = self.planner.lookup_object(index)

        self.assertIsInstance(obj, torch.Tensor)
        self.assertEqual(obj.shape, torch.Size([3, 4]))

    def test_lookup_object_flat_dict_scalar(self):
        """Test lookup_object with flat dict containing scalar."""
        state_dict = {
            "step": 100,
            "learning_rate": 0.001,
        }
        self.planner.set_up_planner(state_dict, is_coordinator=False)

        index = MetadataIndex(fqn="step")
        obj = self.planner.lookup_object(index)

        self.assertEqual(obj, 100)

    def test_lookup_object_nested_dict(self):
        """Test lookup_object with nested dict enabled."""
        state_dict = {
            "model": {
                "weight": torch.randn(2, 3),
                "bias": torch.randn(2),
            },
            "optimizer": {
                "state": [1, 2, 3],
            }
        }
        # flatten_state_dict=True (default)
        self.planner.set_up_planner(state_dict, is_coordinator=False)

        # After flattening, nested keys become "model.weight", "model.bias", etc.
        index = MetadataIndex(fqn="model.weight")
        obj = self.planner.lookup_object(index)

        self.assertIsInstance(obj, torch.Tensor)

    def test_lookup_object_empty_dict(self):
        """Test lookup_object with empty state_dict."""
        state_dict = {}
        self.planner.set_up_planner(state_dict, is_coordinator=False)

        # Trying to lookup non-existent key should raise ValueError
        index = MetadataIndex(fqn="nonexistent")
        with self.assertRaises(ValueError):
            self.planner.lookup_object(index)

    def test_lookup_object_none_value(self):
        """Test lookup_object with None value in state_dict."""
        state_dict = {
            "value": None,
        }
        self.planner.set_up_planner(state_dict, is_coordinator=False)

        index = MetadataIndex(fqn="value")
        obj = self.planner.lookup_object(index)

        self.assertIsNone(obj)


class TestDefaultSavePlannerTransformObject(TestCase):
    """Test DefaultSavePlanner.transform_object method."""

    def setUp(self):
        super().setUp()
        device_name = torch._C._get_privateuse1_backend_name()
        self.assertEqual(device_name, 'npu', f"Expected device 'npu', got '{device_name}'")
        self.planner = DefaultSavePlanner()

    def test_transform_object_tensor_type(self):
        """Test transform_object with TENSOR WriteItemType."""
        state_dict = {"weight": torch.randn(2, 3)}
        self.planner.set_up_planner(state_dict, is_coordinator=False)

        tensor = torch.randn(2, 3)
        write_item = WriteItem(
            index=MetadataIndex(fqn="weight"),
            type=WriteItemType.TENSOR,
        )

        result = self.planner.transform_object(write_item, tensor)

        self.assertIsInstance(result, torch.Tensor)
        self.assertEqual(result.shape, tensor.shape)

    def test_transform_object_byte_io_type(self):
        """Test transform_object with BYTE_IO WriteItemType."""
        state_dict = {"config": {"lr": 0.001}}
        self.planner.set_up_planner(state_dict, is_coordinator=False)

        config_obj = {"lr": 0.001}
        write_item = WriteItem(
            index=MetadataIndex(fqn="config"),
            type=WriteItemType.BYTE_IO,
        )

        result = self.planner.transform_object(write_item, config_obj)

        self.assertIsInstance(result, io.BytesIO)
        # Verify serialized content
        result.seek(0)
        loaded = torch.load(result, weights_only=False)
        self.assertEqual(loaded["lr"], 0.001)

    def test_transform_object_shard_type_tensor(self):
        """Test transform_object with SHARD WriteItemType and tensor."""
        state_dict = {"shard": torch.randn(4, 4)}
        self.planner.set_up_planner(state_dict, is_coordinator=False)

        tensor = torch.randn(4, 4)
        write_item = WriteItem(
            index=MetadataIndex(fqn="shard"),
            type=WriteItemType.SHARD,
        )

        result = self.planner.transform_object(write_item, tensor)

        self.assertIsInstance(result, torch.Tensor)


class TestDefaultLoadPlannerLookupTensor(TestCase):
    """Test DefaultLoadPlanner.lookup_tensor method."""

    def setUp(self):
        super().setUp()
        device_name = torch._C._get_privateuse1_backend_name()
        self.assertEqual(device_name, 'npu', f"Expected device 'npu', got '{device_name}'")

    def test_lookup_tensor_flat_dict(self):
        """Test lookup_tensor with flat state_dict."""
        state_dict = {
            "weight": torch.randn(3, 4),
            "bias": torch.randn(3),
        }
        planner = DefaultLoadPlanner()
        planner.set_up_planner(state_dict)

        index = MetadataIndex(fqn="weight")
        tensor = planner.lookup_tensor(index)

        self.assertIsInstance(tensor, torch.Tensor)
        self.assertEqual(tensor.shape, torch.Size([3, 4]))

    def test_lookup_tensor_nested_dict_with_flatten(self):
        """Test lookup_tensor with nested dict (flatten_state_dict=True)."""
        state_dict = {
            "model": {
                "weight": torch.randn(2, 3),
            }
        }
        planner = DefaultLoadPlanner(flatten_state_dict=True)
        planner.set_up_planner(state_dict)

        # After flattening, key becomes "model.weight"
        index = MetadataIndex(fqn="model.weight")
        tensor = planner.lookup_tensor(index)

        self.assertIsInstance(tensor, torch.Tensor)
        self.assertEqual(tensor.shape, torch.Size([2, 3]))

    def test_lookup_tensor_nonexistent_key(self):
        """Test lookup_tensor with non-existent key."""
        state_dict = {"weight": torch.randn(2, 3)}
        planner = DefaultLoadPlanner()
        planner.set_up_planner(state_dict)

        index = MetadataIndex(fqn="nonexistent")
        with self.assertRaises(ValueError):
            planner.lookup_tensor(index)

    def test_lookup_tensor_different_dtypes(self):
        """Test lookup_tensor with different tensor dtypes."""
        state_dict = {
            "float32_tensor": torch.randn(2, 3, dtype=torch.float32),
            "int64_tensor": torch.tensor([1, 2, 3], dtype=torch.int64),
        }
        planner = DefaultLoadPlanner()
        planner.set_up_planner(state_dict)

        # Lookup float32 tensor
        float_tensor = planner.lookup_tensor(MetadataIndex(fqn="float32_tensor"))
        self.assertEqual(float_tensor.dtype, torch.float32)

        # Lookup int64 tensor
        int_tensor = planner.lookup_tensor(MetadataIndex(fqn="int64_tensor"))
        self.assertEqual(int_tensor.dtype, torch.int64)


class TestDefaultLoadPlannerTransformTensor(TestCase):
    """Test DefaultLoadPlanner.transform_tensor method."""

    def setUp(self):
        super().setUp()
        device_name = torch._C._get_privateuse1_backend_name()
        self.assertEqual(device_name, 'npu', f"Expected device 'npu', got '{device_name}'")

    def test_transform_tensor_identity(self):
        """Test transform_tensor with zero offsets and full size."""
        state_dict = {"weight": torch.randn(3, 4)}
        planner = DefaultLoadPlanner()
        planner.set_up_planner(state_dict)

        tensor = torch.randn(3, 4)
        read_item = ReadItem(
            type=LoadItemType.TENSOR,
            dest_index=MetadataIndex(fqn="weight"),
            dest_offsets=torch.Size([0, 0]),
            storage_index=MetadataIndex(fqn="weight"),
            storage_offsets=torch.Size([0, 0]),
            lengths=torch.Size([3, 4]),
        )

        result = planner.transform_tensor(read_item, tensor)

        self.assertIsInstance(result, torch.Tensor)
        self.assertEqual(result.shape, torch.Size([3, 4]))

    def test_transform_tensor_with_offsets(self):
        """Test transform_tensor with dest_offsets."""
        state_dict = {"weight": torch.randn(5, 5)}
        planner = DefaultLoadPlanner()
        planner.set_up_planner(state_dict)

        tensor = torch.randn(5, 5)
        read_item = ReadItem(
            type=LoadItemType.TENSOR,
            dest_index=MetadataIndex(fqn="weight"),
            dest_offsets=torch.Size([1, 1]),
            storage_index=MetadataIndex(fqn="weight"),
            storage_offsets=torch.Size([0, 0]),
            lengths=torch.Size([3, 3]),
        )

        result = planner.transform_tensor(read_item, tensor)

        self.assertIsInstance(result, torch.Tensor)
        # Result should be narrowed to [3, 3]
        self.assertEqual(result.shape, torch.Size([3, 3]))

    def test_transform_tensor_various_shapes(self):
        """Test transform_tensor with various tensor shapes."""
        shapes = [
            torch.Size([10]),
            torch.Size([5, 5]),
            torch.Size([2, 3, 4]),
            torch.Size([1, 1, 1, 1]),
        ]

        for shape in shapes:
            state_dict = {"tensor": torch.randn(*shape)}
            planner = DefaultLoadPlanner()
            planner.set_up_planner(state_dict)

            tensor = torch.randn(*shape)
            read_item = ReadItem(
                type=LoadItemType.TENSOR,
                dest_index=MetadataIndex(fqn="tensor"),
                dest_offsets=torch.Size([0] * len(shape)),
                storage_index=MetadataIndex(fqn="tensor"),
                storage_offsets=torch.Size([0] * len(shape)),
                lengths=shape,
            )

            result = planner.transform_tensor(read_item, tensor)
            self.assertEqual(result.shape, shape)


class TestPlannerIntegration(TestCase):
    """Integration tests for SavePlanner and LoadPlanner."""

    def setUp(self):
        super().setUp()
        device_name = torch._C._get_privateuse1_backend_name()
        self.assertEqual(device_name, 'npu', f"Expected device 'npu', got '{device_name}'")

    def test_save_load_planner_roundtrip(self):
        """Test SavePlanner and LoadPlanner with same state_dict."""
        original_dict = {
            "param1": torch.randn(2, 3),
            "param2": torch.randn(4),
        }

        # Setup SavePlanner
        save_planner = DefaultSavePlanner()
        save_planner.set_up_planner(original_dict, is_coordinator=False)

        # Setup LoadPlanner
        copy_dict = {
            "param1": torch.zeros(2, 3),
            "param2": torch.zeros(4),
        }
        load_planner = DefaultLoadPlanner()
        load_planner.set_up_planner(copy_dict)

        # Verify both planners can access the tensors
        param1_saved = save_planner.lookup_object(MetadataIndex(fqn="param1"))
        param1_load = load_planner.lookup_tensor(MetadataIndex(fqn="param1"))

        self.assertEqual(param1_saved.shape, param1_load.shape)

    def test_planner_with_no_flatten(self):
        """Test planner with flatten_state_dict=False."""
        state_dict = {
            "weight": torch.randn(2, 3),
            "bias": torch.randn(3),
        }

        load_planner = DefaultLoadPlanner(flatten_state_dict=False)
        load_planner.set_up_planner(state_dict)

        # With flatten_state_dict=False, can directly access flat keys
        index = MetadataIndex(fqn="weight")
        tensor = load_planner.lookup_tensor(index)

        self.assertIsInstance(tensor, torch.Tensor)
        self.assertEqual(tensor.shape, torch.Size([2, 3]))


if __name__ == "__main__":
    run_tests()
