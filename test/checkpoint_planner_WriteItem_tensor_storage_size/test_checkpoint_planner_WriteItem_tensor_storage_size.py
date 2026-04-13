# -*- coding: utf-8 -*-
"""
测试目的：验证 torch.distributed.checkpoint.planner.WriteItem.tensor_storage_size 接口功能正确性
API 名称：torch.distributed.checkpoint.planner.WriteItem.tensor_storage_size
API 签名：WriteItem.tensor_storage_size(self) -> Optional[int]

覆盖维度表：
| 覆盖维度         | 说明                                                         | 覆盖情况                                                  |
|------------------|--------------------------------------------------------------|-----------------------------------------------------------|
| 空/非空          | tensor_data 为 None vs 非 None                               | 已覆盖：None 返回 None；非 None 返回字节数               |
| 枚举选项         | WriteItemType: TENSOR / SHARD / BYTE_IO                      | 已覆盖：TENSOR、SHARD（均带 tensor_data）；BYTE_IO 无数据 |
| 参数类型         | 各 dtype: float32, float16, int32, int64, bool               | 已覆盖                                                    |
| 传参与不传参     | 无额外参数，方法无参                                          | 已覆盖                                                    |
| 等价类/边界值    | 空 shape (1,)，大 shape，多维 shape                           | 已覆盖                                                    |
| 正常传参场景     | 各 dtype 计算字节数                                           | 已覆盖                                                    |
| 异常传参场景     | 无稳定异常路径（构造时已约束类型）                            | 未覆盖，原因：dataclass frozen=True 构造约束              |

未覆盖项及原因：
- 异常传参场景：WriteItem 是 frozen dataclass，类型检查在构造时，无稳定运行时异常路径。

注意：本测试仅验证功能正确性（调用不报错、输出 shape/dtype/类型符合预期），
     不做精度和数值正确性校验。
"""
import unittest

import torch
import torch_npu
from torch.distributed.checkpoint.metadata import (
    ChunkStorageMetadata,
    MetadataIndex,
    TensorProperties,
)
from torch.distributed.checkpoint.planner import (
    TensorWriteData,
    WriteItem,
    WriteItemType,
)
from torch_npu.testing.testcase import TestCase, run_tests


def _make_tensor_write_item(fqn, dtype, size):
    """Helper to build a WriteItem with TensorWriteData for a given dtype/size."""
    index = MetadataIndex(fqn=fqn, offset=torch.Size([0] * len(size)))
    chunk = ChunkStorageMetadata(
        offsets=torch.Size([0] * len(size)),
        sizes=torch.Size(size),
    )
    props = TensorProperties(dtype=dtype)
    tensor_data = TensorWriteData(chunk=chunk, properties=props, size=torch.Size(size))
    return WriteItem(index=index, type=WriteItemType.TENSOR, tensor_data=tensor_data)


class TestWriteItemTensorStorageSize(TestCase):
    """Tests for WriteItem.tensor_storage_size — pure Python dataclass method."""

    def setUp(self):
        super().setUp()
        self.device_name = torch._C._get_privateuse1_backend_name()
        self.assertEqual(
            self.device_name, 'npu',
            f"Expected device 'npu', got '{self.device_name}'"
        )

    def test_returns_none_for_bytes_write_item(self):
        """BYTE_IO WriteItem with no tensor_data returns None."""
        index = MetadataIndex(fqn="key")
        item = WriteItem(
            index=index,
            type=WriteItemType.BYTE_IO,
            tensor_data=None,
        )
        self.assertIsNone(item.tensor_storage_size())

    def test_returns_none_when_tensor_data_is_none(self):
        """WriteItem with tensor_data=None returns None regardless of type."""
        index = MetadataIndex(fqn="key")
        item = WriteItem(index=index, type=WriteItemType.TENSOR, tensor_data=None)
        self.assertIsNone(item.tensor_storage_size())

    def test_float32_storage_size(self):
        """float32 tensor: 4 bytes per element."""
        item = _make_tensor_write_item("w", torch.float32, [4, 8])
        result = item.tensor_storage_size()
        self.assertIsNotNone(result)
        self.assertIsInstance(result, int)
        self.assertEqual(result, 4 * 8 * 4)  # 4*8 elements * 4 bytes

    def test_float16_storage_size(self):
        """float16 tensor: 2 bytes per element."""
        item = _make_tensor_write_item("w", torch.float16, [3, 5])
        self.assertEqual(item.tensor_storage_size(), 3 * 5 * 2)

    def test_int32_storage_size(self):
        """int32 tensor: 4 bytes per element."""
        item = _make_tensor_write_item("w", torch.int32, [2, 2, 2])
        self.assertEqual(item.tensor_storage_size(), 2 * 2 * 2 * 4)

    def test_int64_storage_size(self):
        """int64 tensor: 8 bytes per element."""
        item = _make_tensor_write_item("w", torch.int64, [10])
        self.assertEqual(item.tensor_storage_size(), 10 * 8)

    def test_bool_storage_size(self):
        """bool tensor: 1 byte per element."""
        item = _make_tensor_write_item("w", torch.bool, [16])
        self.assertEqual(item.tensor_storage_size(), 16 * 1)

    def test_single_element_tensor(self):
        """Scalar-like tensor with shape (1,): size = element_size."""
        item = _make_tensor_write_item("scalar", torch.float32, [1])
        self.assertEqual(item.tensor_storage_size(), 4)

    def test_large_tensor_size(self):
        """Large tensor shape computes correctly."""
        item = _make_tensor_write_item("large", torch.float32, [1024, 1024])
        self.assertEqual(item.tensor_storage_size(), 1024 * 1024 * 4)

    def test_shard_write_item_storage_size(self):
        """SHARD type WriteItem with tensor_data returns correct size."""
        index = MetadataIndex(fqn="shard_key", offset=torch.Size([0, 0]))
        chunk = ChunkStorageMetadata(
            offsets=torch.Size([0, 0]),
            sizes=torch.Size([8, 4]),
        )
        props = TensorProperties(dtype=torch.float16)
        tensor_data = TensorWriteData(
            chunk=chunk, properties=props, size=torch.Size([8, 4])
        )
        item = WriteItem(index=index, type=WriteItemType.SHARD, tensor_data=tensor_data)
        self.assertEqual(item.tensor_storage_size(), 8 * 4 * 2)

    def test_return_type_is_int(self):
        """Return type must be int when tensor_data is present."""
        item = _make_tensor_write_item("w", torch.float32, [2, 3])
        result = item.tensor_storage_size()
        self.assertIsInstance(result, int)

    def test_multidimensional_tensor(self):
        """Multi-dimensional tensor: product of all dims * element_size."""
        item = _make_tensor_write_item("nd", torch.float32, [2, 3, 4, 5])
        self.assertEqual(item.tensor_storage_size(), 2 * 3 * 4 * 5 * 4)


if __name__ == "__main__":
    run_tests()
