# -*- coding: utf-8 -*-
"""
测试目的：验证 torch.distributed.tensor._dtensor_spec.TensorMeta 接口功能正确性
API 名称：torch.distributed.tensor._dtensor_spec.TensorMeta
API 签名：TensorMeta(shape, stride, dtype) 数据结构

覆盖维度表：
| 覆盖维度         | 说明                                                         | 覆盖情况                                       |
|------------------|--------------------------------------------------------------|------------------------------------------------|
| 创建/初始化      | 验证 TensorMeta 对象创建                                     | 已覆盖：test_tensor_meta_creation              |
| 属性访问         | 验证 shape、stride、dtype 属性                               | 已覆盖：test_tensor_meta_attributes            |
| 参数类型         | 验证不同 shape、stride、dtype 组合                           | 已覆盖：test_tensor_meta_different_types       |
| 等价类/边界值    | 验证空 shape、不同维度数                                     | 已覆盖：test_tensor_meta_edge_cases            |
| 正常传参场景     | 创建典型的 TensorMeta 实例                                   | 已覆盖：test_tensor_meta_basic                 |

未覆盖项及原因：
- 无

注意：本测试仅验证功能正确性（调用不报错、输出 shape/dtype/类型符合预期），
     不做精度和数值正确性校验。
"""

import torch
import torch_npu
from torch.distributed.tensor._dtensor_spec import TensorMeta
from torch_npu.testing.testcase import TestCase, run_tests


class TestTensorMeta(TestCase):

    def setUp(self):
        super().setUp()
        self.device_name = torch._C._get_privateuse1_backend_name()
        self.assertEqual(self.device_name, 'npu', f"Expected device 'npu', got '{self.device_name}'")

    def test_tensor_meta_creation(self):
        # Test TensorMeta creation
        shape = torch.Size([4, 4])
        stride = (4, 1)
        dtype = torch.float32
        
        meta = TensorMeta(shape, stride, dtype)
        
        self.assertIsNotNone(meta)
        self.assertIsInstance(meta, TensorMeta)

    def test_tensor_meta_attributes(self):
        # Test TensorMeta attributes
        shape = torch.Size([3, 5, 7])
        stride = (35, 7, 1)
        dtype = torch.float64
        
        meta = TensorMeta(shape, stride, dtype)
        
        self.assertEqual(meta.shape, shape)
        self.assertEqual(meta.stride, stride)
        self.assertEqual(meta.dtype, dtype)

    def test_tensor_meta_different_shapes(self):
        # Test TensorMeta with different shapes
        test_cases = [
            (torch.Size([10]), (1,)),
            (torch.Size([3, 4]), (4, 1)),
            (torch.Size([2, 3, 4]), (12, 4, 1)),
            (torch.Size([1, 1, 1]), (1, 1, 1)),
        ]
        
        for shape, stride in test_cases:
            meta = TensorMeta(shape, stride, torch.float32)
            self.assertEqual(meta.shape, shape)
            self.assertEqual(meta.stride, stride)

    def test_tensor_meta_different_dtypes(self):
        # Test TensorMeta with different dtypes
        shape = torch.Size([4, 4])
        stride = (4, 1)
        
        dtypes = [
            torch.float32,
            torch.float64,
            torch.int32,
            torch.int64,
            torch.float16,
        ]
        
        for dtype in dtypes:
            meta = TensorMeta(shape, stride, dtype)
            self.assertEqual(meta.dtype, dtype)

    def test_tensor_meta_empty_shape(self):
        # Test TensorMeta with scalar-like shape
        meta = TensorMeta(torch.Size([]), (), torch.float32)
        
        self.assertEqual(meta.shape, torch.Size([]))
        self.assertEqual(meta.stride, ())

    def test_tensor_meta_large_shape(self):
        # Test TensorMeta with large shape
        shape = torch.Size([100, 100, 100])
        stride = (10000, 100, 1)
        
        meta = TensorMeta(shape, stride, torch.float32)
        
        self.assertEqual(meta.shape, shape)
        self.assertEqual(meta.stride, stride)

    def test_tensor_meta_repr(self):
        # Test TensorMeta string representation
        shape = torch.Size([2, 3])
        stride = (3, 1)
        dtype = torch.float32
        
        meta = TensorMeta(shape, stride, dtype)
        repr_str = repr(meta)
        
        self.assertIn('TensorMeta', repr_str)
        self.assertIn(str(shape), repr_str)


if __name__ == "__main__":
    run_tests()
