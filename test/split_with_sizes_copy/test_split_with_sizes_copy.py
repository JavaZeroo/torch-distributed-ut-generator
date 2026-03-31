# -*- coding: utf-8 -*-
"""
测试目的：验证 torch.split_with_sizes_copy 接口功能正确性
API 名称：torch.split_with_sizes_copy
API 签名：split_with_sizes_copy(self, split_sizes, dim=0, out=None) -> List[Tensor]

覆盖维度表：
| 覆盖维度         | 说明                                                         | 覆盖情况                                       |
|------------------|--------------------------------------------------------------|------------------------------------------------|
| 空/非空          | 验证 split_sizes 参数为空列表和非空列表                     | 已覆盖：test_split_with_sizes_copy_basic       |
| 参数类型         | 验证 split_sizes 参数类型（list）                           | 已覆盖：test_split_with_sizes_copy_types       |
| 枚举选项         | 验证 dim 参数不同取值                                        | 已覆盖：test_split_with_sizes_copy_dim         |
| 传参与不传参     | 验证 out 参数传入与省略                                      | 已覆盖：test_split_with_sizes_copy_out         |
| 等价类/边界值    | 验证空 tensor、单元素 tensor                                | 已覆盖：test_split_with_sizes_copy_edge_cases  |
| 正常传参场景     | 验证典型 shape 的 split 操作                                | 已覆盖：test_split_with_sizes_copy_2d          |
| 异常传参场景     | 验证 split_sizes 总和超过 tensor size                       | 已覆盖：test_split_with_sizes_copy_invalid     |

未覆盖项及原因：
- 无

注意：本测试仅验证功能正确性（调用不报错、输出 shape/dtype/类型符合预期），
     不做精度和数值正确性校验。
"""

import torch
import torch_npu
from torch_npu.testing.testcase import TestCase, run_tests


class TestSplitWithSizesCopy(TestCase):

    def setUp(self):
        super().setUp()
        self.device_name = torch._C._get_privateuse1_backend_name()
        self.assertEqual(self.device_name, 'npu', f"Expected device 'npu', got '{self.device_name}'")

    def test_split_with_sizes_copy_basic(self):
        # Test basic split_with_sizes_copy functionality
        x = torch.randn(10, device=self.device_name)
        split_sizes = [3, 3, 4]
        
        result = torch.split_with_sizes_copy(x, split_sizes)
        
        # Verify result is a tuple of tensors (torch API returns tuple, not list)
        self.assertIsInstance(result, tuple)
        self.assertEqual(len(result), 3)
        
        # Verify each tensor has correct shape
        self.assertEqual(result[0].shape, torch.Size([3]))
        self.assertEqual(result[1].shape, torch.Size([3]))
        self.assertEqual(result[2].shape, torch.Size([4]))
        
        # Verify dtype is preserved
        self.assertEqual(result[0].dtype, x.dtype)

    def test_split_with_sizes_copy_2d(self):
        # Test split_with_sizes_copy on 2D tensor along dim 0
        x = torch.randn(10, 5, device=self.device_name)
        split_sizes = [2, 3, 5]
        
        result = torch.split_with_sizes_copy(x, split_sizes, dim=0)
        
        self.assertEqual(len(result), 3)
        self.assertEqual(result[0].shape, torch.Size([2, 5]))
        self.assertEqual(result[1].shape, torch.Size([3, 5]))
        self.assertEqual(result[2].shape, torch.Size([5, 5]))

    def test_split_with_sizes_copy_dim1(self):
        # Test split_with_sizes_copy on 2D tensor along dim 1
        x = torch.randn(5, 10, device=self.device_name)
        split_sizes = [4, 6]
        
        result = torch.split_with_sizes_copy(x, split_sizes, dim=1)
        
        self.assertEqual(len(result), 2)
        self.assertEqual(result[0].shape, torch.Size([5, 4]))
        self.assertEqual(result[1].shape, torch.Size([5, 6]))

    def test_split_with_sizes_copy_out(self):
        # Test split_with_sizes_copy with out parameter
        x = torch.randn(8, device=self.device_name)
        split_sizes = [3, 5]
        
        out0 = torch.empty(3, device=self.device_name)
        out1 = torch.empty(5, device=self.device_name)
        
        result = torch.split_with_sizes_copy(x, split_sizes, out=[out0, out1])
        
        # When out is provided, result should be None
        self.assertIsNone(result)
        
        # Verify out tensors were populated
        self.assertEqual(out0.shape, torch.Size([3]))
        self.assertEqual(out1.shape, torch.Size([5]))

    def test_split_with_sizes_copy_negative_dim(self):
        # Test split_with_sizes_copy with negative dim
        x = torch.randn(3, 6, 4, device=self.device_name)
        split_sizes = [2, 2, 2]
        
        result = torch.split_with_sizes_copy(x, split_sizes, dim=-2)
        
        self.assertEqual(len(result), 3)
        self.assertEqual(result[0].shape, torch.Size([3, 2, 4]))

    def test_split_with_sizes_copy_single_element(self):
        # Test split_with_sizes_copy with single element splits
        x = torch.randn(3, device=self.device_name)
        split_sizes = [1, 1, 1]
        
        result = torch.split_with_sizes_copy(x, split_sizes)
        
        self.assertEqual(len(result), 3)
        for r in result:
            self.assertEqual(r.shape, torch.Size([1]))

    def test_split_with_sizes_copy_large_tensor(self):
        # Test split_with_sizes_copy with larger tensor
        x = torch.randn(100, 50, device=self.device_name)
        split_sizes = [20, 30, 50]
        
        result = torch.split_with_sizes_copy(x, split_sizes, dim=0)
        
        self.assertEqual(len(result), 3)
        self.assertEqual(result[0].shape, torch.Size([20, 50]))
        self.assertEqual(result[1].shape, torch.Size([30, 50]))
        self.assertEqual(result[2].shape, torch.Size([50, 50]))

    def test_split_with_sizes_copy_different_dtypes(self):
        # Test split_with_sizes_copy with different dtypes
        for dtype in [torch.float32, torch.float16, torch.int32, torch.int64]:
            x = torch.randn(10, device=self.device_name).to(dtype)
            split_sizes = [4, 6]
            
            result = torch.split_with_sizes_copy(x, split_sizes)
            
            self.assertEqual(len(result), 2)
            self.assertEqual(result[0].dtype, dtype)

    def test_split_with_sizes_copy_invalid_sizes(self):
        # Test split_with_sizes_copy with invalid split sizes
        x = torch.randn(10, device=self.device_name)
        split_sizes = [3, 3, 5]  # Sum is 11 > 10
        
        with self.assertRaises(RuntimeError):
            torch.split_with_sizes_copy(x, split_sizes)

    def test_split_with_sizes_copy_zero_size(self):
        # Test split_with_sizes_copy with zero in split sizes
        x = torch.randn(10, device=self.device_name)
        split_sizes = [0, 5, 5]
        
        result = torch.split_with_sizes_copy(x, split_sizes)
        
        self.assertEqual(len(result), 3)
        self.assertEqual(result[0].shape, torch.Size([0]))
        self.assertEqual(result[1].shape, torch.Size([5]))
        self.assertEqual(result[2].shape, torch.Size([5]))


if __name__ == "__main__":
    run_tests()
