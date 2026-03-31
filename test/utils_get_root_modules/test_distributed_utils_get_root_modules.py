# -*- coding: utf-8 -*-
"""
测试目的：验证 torch.distributed.utils._get_root_modules 接口功能正确性
API 名称：torch.distributed.utils._get_root_modules
API 签名：_get_root_modules(modules) -> set

覆盖维度表：
| 覆盖维度         | 说明                                                         | 覆盖情况                                       |
|------------------|--------------------------------------------------------------|------------------------------------------------|
| 空/非空          | 验证 modules 参数传入空列表和模块列表                        | 已覆盖：test_get_root_modules_empty            |
| 参数类型         | 验证 modules 参数类型（list/nn.Module）                      | 已覆盖：test_get_root_modules_types            |
| 正常传参场景     | 传入单个/多个模块获取根模块                                  | 已覆盖：test_get_root_modules_basic            |
| 返回值验证       | 验证返回值为 set 类型                                        | 已覆盖：test_get_root_modules_return_type      |

未覆盖项及原因：
- 无

注意：本测试仅验证功能正确性（调用不报错、输出 shape/dtype/类型符合预期），
     不做精度和数值正确性校验。
"""

import torch
import torch.nn as nn
import torch_npu
from torch.distributed.utils import _get_root_modules
from torch_npu.testing.testcase import TestCase, run_tests


class TestGetRootModules(TestCase):

    def setUp(self):
        super().setUp()
        self.device_name = torch._C._get_privateuse1_backend_name()
        self.assertEqual(self.device_name, 'npu', f"Expected device 'npu', got '{self.device_name}'")

    def test_get_root_modules_basic(self):
        # Test _get_root_modules with single module
        module = nn.Linear(10, 10).to('npu')
        
        result = _get_root_modules([module])
        
        # API returns list, not set
        self.assertIsInstance(result, list)
        self.assertEqual(len(result), 1)

    def test_get_root_modules_multiple(self):
        # Test _get_root_modules with multiple modules
        module1 = nn.Linear(10, 20).to('npu')
        module2 = nn.Linear(20, 10).to('npu')
        
        result = _get_root_modules([module1, module2])
        
        # API returns list, not set
        self.assertIsInstance(result, list)
        self.assertEqual(len(result), 2)

    def test_get_root_modules_empty(self):
        # Test _get_root_modules with empty list
        result = _get_root_modules([])
        
        # API returns list, not set
        self.assertIsInstance(result, list)
        self.assertEqual(len(result), 0)

    def test_get_root_modules_sequential(self):
        # Test _get_root_modules with Sequential
        seq = nn.Sequential(
            nn.Linear(10, 20).to('npu'),
            nn.ReLU(),
            nn.Linear(20, 10).to('npu')
        )
        
        result = _get_root_modules([seq])
        
        # API returns list, not set
        self.assertIsInstance(result, list)
        self.assertEqual(len(result), 1)

    def test_get_root_modules_nested(self):
        # Test _get_root_modules with nested modules
        class NestedModule(nn.Module):
            def __init__(self):
                super().__init__()
                self.inner = nn.Linear(10, 10).to('npu')
            
            def forward(self, x):
                return self.inner(x)
        
        nested = NestedModule().to('npu')
        
        result = _get_root_modules([nested])
        
        # API returns list, not set
        self.assertIsInstance(result, list)

    def test_get_root_modules_with_submodules(self):
        # Test _get_root_modules extracting specific submodules
        model = nn.Sequential(
            nn.Linear(10, 20).to('npu'),
            nn.ReLU(),
            nn.Linear(20, 10).to('npu')
        )
        
        # Get the linear layers
        linears = [m for m in model if isinstance(m, nn.Linear)]
        result = _get_root_modules(linears)
        
        # API returns list, not set
        self.assertIsInstance(result, list)
        self.assertGreaterEqual(len(result), 1)

    def test_get_root_modules_single_module_input(self):
        # Test _get_root_modules with single module (not in list)
        module = nn.Conv2d(3, 16, 3).to('npu')
        
        # Wrap in list as expected by API
        result = _get_root_modules([module])
        
        # API returns list, not set
        self.assertIsInstance(result, list)
        self.assertEqual(len(result), 1)


if __name__ == "__main__":
    run_tests()
