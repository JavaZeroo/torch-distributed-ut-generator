# -*- coding: utf-8 -*-
"""
测试目的：验证 torch.distributed.checkpoint.ReadItem 数据类功能正确性
API 名称：torch.distributed.checkpoint.ReadItem
API 签名：ReadItem(type: LoadItemType, dest_index: MetadataIndex, dest_offsets: torch.Size, storage_index: MetadataIndex, storage_offsets: torch.Size, lengths: torch.Size)

覆盖维度表：
| 覆盖维度         | 说明                                                         | 覆盖情况                                       |
|------------------|--------------------------------------------------------------|------------------------------------------------|
| 空/非空          | 验证各字段为空或非空时的构造                                  | 已覆盖                                         |
| 枚举选项         | LoadItemType.TENSOR 和 BYTE_IO 类型                         | 已覆盖                                         |
| 参数类型         | 验证各字段接受正确类型                                       | 已覆盖                                         |
| 传参与不传参     | ReadItem 为 dataclass，所有参数必填                          | 已覆盖（全必填场景）                            |
| 等价类/边界值    | 不同 sizes 和 offsets 组合                                   | 已覆盖                                         |
| 正常传参场景     | 构造不同配置的 ReadItem 对象                                 | 已覆盖                                         |
| 异常传参场景     | 类型检查由 dataclass 自动处理                                | 未覆盖，Python dataclass 不做运行时类型检查    |

未覆盖项及原因：
- 异常传参场景：Python dataclass 不做运行时类型检查，类型错误在静态检查阶段发现

注意：本测试仅验证功能正确性（调用不报错、输出 shape/dtype/类型符合预期），
     不做精度和数值正确性校验。
"""

import torch
from torch.distributed.checkpoint.planner import ReadItem, LoadItemType
from torch.distributed.checkpoint.metadata import MetadataIndex


try:
    from torch_npu.testing.testcase import TestCase, run_tests
except ImportError:
    import sys
    from unittest import TestCase

    def run_tests():
        import unittest
        unittest.main(argv=sys.argv)


class TestReadItem(TestCase):
    """Test cases for ReadItem dataclass."""

    def setUp(self):
        super().setUp()
        self.device_name = torch._C._get_privateuse1_backend_name()
        self.assertEqual(self.device_name, 'npu', f"Expected device 'npu', got '{self.device_name}'")

    def test_read_item_tensor_type(self):
        """Test constructing ReadItem with TENSOR type."""
        dest_index = MetadataIndex(fqn="tensor1", offset=torch.Size([0, 0]))
        storage_index = MetadataIndex(fqn="tensor1", offset=torch.Size([0, 0]))

        read_item = ReadItem(
            type=LoadItemType.TENSOR,
            dest_index=dest_index,
            dest_offsets=torch.Size([0, 0]),
            storage_index=storage_index,
            storage_offsets=torch.Size([0, 0]),
            lengths=torch.Size([10, 10])
        )

        self.assertEqual(read_item.type, LoadItemType.TENSOR)
        self.assertEqual(read_item.dest_index.fqn, "tensor1")
        self.assertEqual(read_item.lengths, torch.Size([10, 10]))

    def test_read_item_byte_io_type(self):
        """Test constructing ReadItem with BYTE_IO type."""
        dest_index = MetadataIndex(fqn="bytes1")
        storage_index = MetadataIndex(fqn="bytes1")

        read_item = ReadItem(
            type=LoadItemType.BYTE_IO,
            dest_index=dest_index,
            dest_offsets=torch.Size([0]),
            storage_index=storage_index,
            storage_offsets=torch.Size([0]),
            lengths=torch.Size([100])
        )

        self.assertEqual(read_item.type, LoadItemType.BYTE_IO)
        self.assertEqual(read_item.dest_index.fqn, "bytes1")

    def test_read_item_with_offsets(self):
        """Test ReadItem with non-zero offsets."""
        dest_index = MetadataIndex(fqn="sharded_tensor", offset=torch.Size([5, 5]))
        storage_index = MetadataIndex(fqn="sharded_tensor", offset=torch.Size([0, 0]), index=0)

        read_item = ReadItem(
            type=LoadItemType.TENSOR,
            dest_index=dest_index,
            dest_offsets=torch.Size([5, 5]),
            storage_index=storage_index,
            storage_offsets=torch.Size([0, 0]),
            lengths=torch.Size([5, 5])
        )

        self.assertEqual(read_item.dest_offsets, torch.Size([5, 5]))
        self.assertEqual(read_item.storage_offsets, torch.Size([0, 0]))
        self.assertEqual(read_item.storage_index.index, 0)

    def test_read_item_frozen_dataclass(self):
        """Test that ReadItem is frozen (immutable)."""
        dest_index = MetadataIndex(fqn="tensor1")
        storage_index = MetadataIndex(fqn="tensor1")

        read_item = ReadItem(
            type=LoadItemType.TENSOR,
            dest_index=dest_index,
            dest_offsets=torch.Size([0]),
            storage_index=storage_index,
            storage_offsets=torch.Size([0]),
            lengths=torch.Size([10])
        )

        # Verify dataclass is frozen
        with self.assertRaises((AttributeError, TypeError)):
            read_item.type = LoadItemType.BYTE_IO

    def test_read_item_equality(self):
        """Test ReadItem equality comparison."""
        dest_index1 = MetadataIndex(fqn="tensor1")
        storage_index1 = MetadataIndex(fqn="tensor1")

        read_item1 = ReadItem(
            type=LoadItemType.TENSOR,
            dest_index=dest_index1,
            dest_offsets=torch.Size([0]),
            storage_index=storage_index1,
            storage_offsets=torch.Size([0]),
            lengths=torch.Size([10])
        )

        dest_index2 = MetadataIndex(fqn="tensor1")
        storage_index2 = MetadataIndex(fqn="tensor1")

        read_item2 = ReadItem(
            type=LoadItemType.TENSOR,
            dest_index=dest_index2,
            dest_offsets=torch.Size([0]),
            storage_index=storage_index2,
            storage_offsets=torch.Size([0]),
            lengths=torch.Size([10])
        )

        self.assertEqual(read_item1, read_item2)

    def test_read_item_inequality(self):
        """Test ReadItem inequality with different values."""
        dest_index1 = MetadataIndex(fqn="tensor1")
        storage_index1 = MetadataIndex(fqn="tensor1")

        read_item1 = ReadItem(
            type=LoadItemType.TENSOR,
            dest_index=dest_index1,
            dest_offsets=torch.Size([0]),
            storage_index=storage_index1,
            storage_offsets=torch.Size([0]),
            lengths=torch.Size([10])
        )

        dest_index2 = MetadataIndex(fqn="tensor2")
        storage_index2 = MetadataIndex(fqn="tensor2")

        read_item2 = ReadItem(
            type=LoadItemType.TENSOR,
            dest_index=dest_index2,
            dest_offsets=torch.Size([0]),
            storage_index=storage_index2,
            storage_offsets=torch.Size([0]),
            lengths=torch.Size([20])
        )

        self.assertNotEqual(read_item1, read_item2)


if __name__ == "__main__":
    run_tests()
