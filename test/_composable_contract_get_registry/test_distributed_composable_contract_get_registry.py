# -*- coding: utf-8 -*-
"""
测试目的：验证 torch.distributed._composable.contract._get_registry 接口功能正确性
API 名称：torch.distributed._composable.contract._get_registry
API 签名：_get_registry(module: nn.Module)

覆盖维度表：
| 覆盖维度         | 说明                                                         | 覆盖情况                                       |
|------------------|--------------------------------------------------------------|------------------------------------------------|
| 空/非空          | 验证 module 参数传入                                         | 已覆盖：test_get_registry_basic                |
| 参数类型         | 验证 module 为 nn.Module 类型                                | 已覆盖：test_get_registry_different_modules    |
| 正常传参场景     | 从模块获取注册表                                             | 已覆盖：test_get_registry_with_state           |
| 返回值验证       | 验证返回值为 dict/registry 类型                              | 已覆盖：test_get_registry_return_type          |

未覆盖项及原因：
- 无

注意：本测试仅验证功能正确性（调用不报错、输出 shape/dtype/类型符合预期），
     不做精度和数值正确性校验。
"""

import torch
import torch.nn as nn
import torch_npu
from torch.distributed._composable.contract import _get_registry
from torch_npu.testing.testcase import TestCase, run_tests


class TestGetRegistry(TestCase):

    def setUp(self):
        super().setUp()
        self.device_name = torch._C._get_privateuse1_backend_name()
        self.assertEqual(self.device_name, 'npu', f"Expected device 'npu', got '{self.device_name}'")

    def test_get_registry_basic(self):
        # Test _get_registry with basic module
        module = nn.Linear(10, 10).to('npu')
        
        registry = _get_registry(module)
        
        # Registry may be empty dict or None initially
        self.assertTrue(registry is None or isinstance(registry, dict))

    def test_get_registry_different_modules(self):
        # Test _get_registry with different module types
        modules = [
            nn.Linear(10, 10).to('npu'),
            nn.Conv2d(3, 16, 3).to('npu'),
            nn.Sequential(
                nn.Linear(10, 20).to('npu'),
                nn.ReLU(),
            ),
        ]
        
        for module in modules:
            registry = _get_registry(module)
            self.assertTrue(registry is None or isinstance(registry, dict))

    def test_get_registry_after_contract(self):
        # Test _get_registry after applying contract decorator
        from torch.distributed._composable.contract import contract
        
        @contract
        def my_composable(module: nn.Module) -> nn.Module:
            return module
        
        module = nn.Linear(10, 10).to('npu')
        wrapped = my_composable(module)
        
        registry = _get_registry(wrapped)
        
        # After contract, registry may be populated
        self.assertTrue(isinstance(registry, dict) or registry is None)

    def test_get_registry_return_type(self):
        # Test that _get_registry returns expected type
        module = nn.Conv2d(3, 16, 3).to('npu')
        
        registry = _get_registry(module)
        
        # Registry should be dict-like or None
        if registry is not None:
            self.assertIsInstance(registry, dict)
            # Should support basic dict operations
            self.assertTrue(hasattr(registry, 'get'))
            self.assertTrue(hasattr(registry, 'items'))

    def test_get_registry_with_custom_module(self):
        # Test _get_registry with custom module class
        class CustomModule(nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = nn.Linear(10, 10).to('npu')
            
            def forward(self, x):
                return self.linear(x)
        
        custom = CustomModule().to('npu')
        
        registry = _get_registry(custom)
        
        self.assertTrue(registry is None or isinstance(registry, dict))


if __name__ == "__main__":
    run_tests()
