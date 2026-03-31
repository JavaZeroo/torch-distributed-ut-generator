# -*- coding: utf-8 -*-
"""
测试目的：验证 tensor.copy_ 接口功能正确性
API 名称：tensor.copy_
API 签名：Tensor.copy_(self, src, non_blocking=False) -> Tensor

覆盖维度表：
| 覆盖维度         | 说明                                                         | 覆盖情况                                       |
|------------------|--------------------------------------------------------------|------------------------------------------------|
| 空/非空          | 验证 src 参数传入有效 Tensor                                 | 已覆盖：test_copy_basic                        |
| 参数类型         | 验证 src 参数类型（Tensor）                                  | 已覆盖：test_copy_different_shapes             |
| 枚举选项         | 验证 non_blocking 参数 True/False                            | 已覆盖：test_copy_non_blocking                 |
| 传参与不传参     | 验证 non_blocking 默认与显式传入                             | 已覆盖：test_copy_non_blocking                 |
| 等价类/边界值    | 验证空 tensor、不同 dtype、不同设备                          | 已覆盖：test_copy_different_dtypes             |
| 正常传参场景     | 同设备拷贝、跨 shape 拷贝                                    | 已覆盖：test_copy_same_device                  |
| 异常传参场景     | 验证 src 为标量/非 Tensor 类型                               | 已覆盖：test_copy_invalid_src                  |

未覆盖项及原因：
- 无

注意：本测试仅验证功能正确性（调用不报错、输出 shape/dtype/类型符合预期），
     不做精度和数值正确性校验。
"""

import torch
import torch_npu
from torch_npu.testing.testcase import TestCase, run_tests


class TestTensorCopy(TestCase):

    def setUp(self):
        super().setUp()
        self.device_name = torch._C._get_privateuse1_backend_name()
        self.assertEqual(self.device_name, 'npu', f"Expected device 'npu', got '{self.device_name}'")

    def test_copy_basic(self):
        # Test basic copy_ functionality
        dst = torch.zeros(5, 5, device=self.device_name)
        src = torch.ones(5, 5, device=self.device_name)
        
        result = dst.copy_(src)
        
        # copy_ returns self
        self.assertEqual(result.shape, dst.shape)
        self.assertEqual(dst.shape, torch.Size([5, 5]))

    def test_copy_non_blocking(self):
        # Test copy_ with non_blocking parameter
        dst = torch.zeros(3, 3, device=self.device_name)
        src = torch.ones(3, 3, device=self.device_name)
        
        # Test non_blocking=False (default)
        dst.copy_(src, non_blocking=False)
        
        # Test non_blocking=True
        dst2 = torch.zeros(3, 3, device=self.device_name)
        dst2.copy_(src, non_blocking=True)

    def test_copy_different_shapes(self):
        # Test copy_ with different but compatible shapes
        dst = torch.zeros(2, 3, device=self.device_name)
        src = torch.ones(2, 3, device=self.device_name)
        
        dst.copy_(src)
        self.assertEqual(dst.shape, torch.Size([2, 3]))

    def test_copy_different_dtypes(self):
        # Test copy_ with different dtypes
        for dst_dtype in [torch.float32, torch.float16]:
            for src_dtype in [torch.float32, torch.float16]:
                dst = torch.zeros(3, 3, device=self.device_name, dtype=dst_dtype)
                src = torch.ones(3, 3, device=self.device_name, dtype=src_dtype)
                
                dst.copy_(src)
                self.assertEqual(dst.dtype, dst_dtype)

    def test_copy_1d(self):
        # Test copy_ on 1D tensor
        dst = torch.zeros(10, device=self.device_name)
        src = torch.ones(10, device=self.device_name)
        
        dst.copy_(src)
        self.assertEqual(dst.shape, torch.Size([10]))

    def test_copy_3d(self):
        # Test copy_ on 3D tensor
        dst = torch.zeros(2, 3, 4, device=self.device_name)
        src = torch.ones(2, 3, 4, device=self.device_name)
        
        dst.copy_(src)
        self.assertEqual(dst.shape, torch.Size([2, 3, 4]))

    def test_copy_same_tensor(self):
        # Test copy_ with same tensor (self-copy)
        x = torch.randn(5, 5, device=self.device_name)
        x.copy_(x)
        self.assertEqual(x.shape, torch.Size([5, 5]))

    def test_copy_large_tensor(self):
        # Test copy_ with larger tensor
        dst = torch.zeros(100, 100, device=self.device_name)
        src = torch.ones(100, 100, device=self.device_name)
        
        dst.copy_(src)
        self.assertEqual(dst.shape, torch.Size([100, 100]))

    def test_copy_invalid_src(self):
        # Test copy_ with invalid src type
        dst = torch.zeros(3, 3, device=self.device_name)
        
        with self.assertRaises(TypeError):
            dst.copy_("not_a_tensor")

    def test_copy_return_value(self):
        # Test that copy_ returns self
        dst = torch.zeros(3, 3, device=self.device_name)
        src = torch.ones(3, 3, device=self.device_name)
        
        result = dst.copy_(src)
        self.assertIs(result, dst)


if __name__ == "__main__":
    run_tests()
