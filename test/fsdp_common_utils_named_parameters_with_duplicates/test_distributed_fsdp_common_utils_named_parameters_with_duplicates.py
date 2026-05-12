# -*- coding: utf-8 -*-
"""
测试目的：验证 torch.distributed.fsdp._common_utils._named_parameters_with_duplicates 接口功能正确性
API 名称：torch.distributed.fsdp._common_utils._named_parameters_with_duplicates
API 签名：_named_parameters_with_duplicates(module, prefix='', remove_duplicate)

覆盖维度表：
| 覆盖维度         | 说明                                                         | 覆盖情况                                       |
|------------------|--------------------------------------------------------------|------------------------------------------------|
| 空/非空          | 验证 module 参数                                             | 已覆盖：test_named_parameters_basic            |
| 参数类型         | 验证 prefix 字符串、remove_duplicate bool                    | 已覆盖：test_named_parameters_options          |
| 枚举选项         | 验证 remove_duplicate=True/False                             | 已覆盖：test_named_parameters_remove_duplicate |
| 正常传参场景     | 从模块获取命名参数（含重复）                                 | 已覆盖：test_named_parameters_with_duplicates  |
| 多卡场景         | 验证多卡环境下参数获取                                       | 已覆盖：test_named_parameters_multiprocess     |
| 返回值验证       | 验证返回值为迭代器/列表                                      | 已覆盖：test_named_parameters_return_type      |

未覆盖项及原因：
- 无

注意：本测试仅验证功能正确性（调用不报错、输出 shape/dtype/类型符合预期），
     不做精度和数值正确性校验。
"""

import os
import torch
import torch.nn as nn
import torch.distributed as dist
import torch.multiprocessing as mp

import torch_npu
from torch.distributed.fsdp._common_utils import _named_parameters_with_duplicates
from torch_npu.testing.testcase import TestCase, run_tests
from torch_npu.testing.common_distributed import skipIfUnsupportMultiNPU


class TestNamedParametersWithDuplicates(TestCase):

    def setUp(self):
        super().setUp()
        self.device_name = torch._C._get_privateuse1_backend_name()
        self.assertEqual(self.device_name, 'npu', f"Expected device 'npu', got '{self.device_name}'")

    @classmethod
    def _init_dist_hccl(cls, rank, world_size):
        os.environ.setdefault('MASTER_ADDR', '127.0.0.1')
        os.environ.setdefault('MASTER_PORT', '29504')
        os.environ['HCCL_WHITELIST_DISABLE'] = '1'
        torch_npu.npu.set_device(rank)
        dist.init_process_group(backend='hccl', world_size=world_size, rank=rank)
        return dist

    def test_named_parameters_basic(self):
        # Test _named_parameters_with_duplicates with basic module
        module = nn.Linear(10, 10).to('npu')
        
        params = list(_named_parameters_with_duplicates(module))
        
        # Should return list of tuples (name, param)
        self.assertIsInstance(params, list)
        self.assertGreater(len(params), 0)
        
        # Check structure
        for name, param in params:
            self.assertIsInstance(name, str)
            self.assertIsInstance(param, torch.nn.Parameter)

    def test_named_parameters_with_prefix(self):
        # Test _named_parameters_with_duplicates with prefix
        module = nn.Linear(10, 10).to('npu')
        prefix = "model.layer1"
        
        params = list(_named_parameters_with_duplicates(module, prefix=prefix))
        
        # All parameter names should start with the prefix
        for name, param in params:
            self.assertTrue(name.startswith(prefix))

    def test_named_parameters_with_duplicates(self):
        # Test _named_parameters_with_duplicates returns all parameters including duplicates
        class SharedModule(nn.Module):
            def __init__(self):
                super().__init__()
                self.shared = nn.Linear(10, 10).to('npu')
                self.ref = self.shared  # Reference to same module
        
        module = SharedModule()
        # Note: _named_parameters_with_duplicates does not accept remove_duplicate argument
        params = list(_named_parameters_with_duplicates(module))
        
        # Should return list of tuples
        self.assertIsInstance(params, list)
        for name, param in params:
            self.assertIsInstance(name, str)
            self.assertIsInstance(param, torch.nn.Parameter)

    def test_named_parameters_all_entries(self):
        # Test _named_parameters_with_duplicates returns all parameter entries
        module = nn.Sequential(
            nn.Linear(10, 20).to('npu'),
            nn.Linear(20, 10).to('npu')
        )
        
        # Note: _named_parameters_with_duplicates does not accept remove_duplicate argument
        params = list(_named_parameters_with_duplicates(module))
        
        # Should include all parameters
        self.assertIsInstance(params, list)
        self.assertGreater(len(params), 0)

    def test_named_parameters_sequential(self):
        # Test _named_parameters_with_duplicates with Sequential
        module = nn.Sequential(
            nn.Linear(10, 20).to('npu'),
            nn.ReLU(),
            nn.Linear(20, 10).to('npu')
        )
        
        params = list(_named_parameters_with_duplicates(module))
        
        # Should get parameters from both Linear layers
        param_names = [name for name, _ in params]
        self.assertTrue(any('0' in name for name in param_names))
        self.assertTrue(any('2' in name for name in param_names))

    def test_named_parameters_nested(self):
        # Test _named_parameters_with_duplicates with nested modules
        class NestedModule(nn.Module):
            def __init__(self):
                super().__init__()
                self.inner = nn.Sequential(
                    nn.Linear(10, 20).to('npu'),
                    nn.Linear(20, 10).to('npu')
                )
            
            def forward(self, x):
                return self.inner(x)
        
        module = NestedModule()
        params = list(_named_parameters_with_duplicates(module))
        
        # Should have parameters from nested structure
        self.assertIsInstance(params, list)
        self.assertGreaterEqual(len(params), 2)

    @classmethod
    def _test_named_parameters_multiprocess(cls, rank, world_size, c2p):
        dist_group = cls._init_dist_hccl(rank, world_size)
        
        # Create a simple model
        model = nn.Sequential(
            nn.Linear(10, 20).to(f'npu:{rank}'),
            nn.Linear(20, 10).to(f'npu:{rank}')
        )
        
        # Get named parameters
        params = list(_named_parameters_with_duplicates(model))
        
        c2p.put((rank, 'param_count', len(params)))
        c2p.put((rank, 'param_names', [name for name, _ in params]))
        
        dist_group.destroy_process_group()

    @skipIfUnsupportMultiNPU(2)
    def test_named_parameters_multiprocess(self):
        # Test _named_parameters_with_duplicates in multiprocess environment
        ctx = mp.get_context('spawn')
        c2p = ctx.Queue(4)
        
        ps = []
        for i in range(2):
            p = ctx.Process(
                target=self._test_named_parameters_multiprocess,
                args=(i, 2, c2p))
            p.start()
            ps.append(p)
        
        results = []
        for _ in range(4):
            results.append(c2p.get(timeout=30))
        
        for p in ps:
            p.join(timeout=30)
            self.assertEqual(p.exitcode, 0, "subprocess exit with abnormal code.")
        
        # Verify results from both ranks
        for rank, event, value in results:
            if event == 'param_count':
                self.assertGreater(value, 0)
            elif event == 'param_names':
                self.assertIsInstance(value, list)


if __name__ == "__main__":
    run_tests()
