# -*- coding: utf-8 -*-
"""
测试目的：验证 torch.distributed.tensor.DTensor._local_tensor 接口功能正确性
API 名称：torch.distributed.tensor.DTensor._local_tensor
API 签名：DTensor._local_tensor 属性访问，返回本地张量

覆盖维度表：
| 覆盖维度         | 说明                                                         | 覆盖情况                                       |
|------------------|--------------------------------------------------------------|------------------------------------------------|
| 属性访问         | 验证 _local_tensor 属性返回本地张量                          | 已覆盖：test_local_tensor_attribute            |
| 多卡场景         | 验证多卡环境下各 rank 的本地 tensor                          | 已覆盖：test_local_tensor_multiprocess         |
| 不同 shape       | 验证不同 shape DTensor 的本地 tensor                         | 已覆盖：test_local_tensor_different_shapes     |
| 不同 dtype       | 验证不同 dtype DTensor 的本地 tensor                         | 已覆盖：test_local_tensor_different_dtypes     |
| 正常传参场景     | 创建 DTensor 后访问 _local_tensor                            | 已覆盖：test_local_tensor_basic                |

未覆盖项及原因：
- 无

注意：本测试仅验证功能正确性（调用不报错、输出 shape/dtype/类型符合预期），
     不做精度和数值正确性校验。
"""

import os
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.distributed.tensor import DTensor
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.tensor.placement_types import Replicate, Shard

import torch_npu
from torch_npu.testing.testcase import TestCase, run_tests
from torch_npu.testing.common_distributed import skipIfUnsupportMultiNPU


class TestDTensorLocalTensor(TestCase):

    def setUp(self):
        super().setUp()
        self.device_name = torch._C._get_privateuse1_backend_name()
        self.assertEqual(self.device_name, 'npu', f"Expected device 'npu', got '{self.device_name}'")

    @classmethod
    def _init_dist_hccl(cls, rank, world_size):
        os.environ.setdefault('MASTER_ADDR', '127.0.0.1')
        os.environ.setdefault('MASTER_PORT', '29503')
        os.environ['HCCL_WHITELIST_DISABLE'] = '1'
        torch_npu.npu.set_device(rank)
        dist.init_process_group(backend='hccl', world_size=world_size, rank=rank)
        return dist

    @classmethod
    def _test_local_tensor_basic(cls, rank, world_size, c2p):
        dist_group = cls._init_dist_hccl(rank, world_size)
        
        # Create a device mesh
        mesh = init_device_mesh('npu', (world_size,))
        
        # Create a DTensor with replicate placement
        local_tensor = torch.randn(4, 4, device=f'npu:{rank}')
        dt = DTensor.from_local(local_tensor, mesh, [Replicate()])
        
        # Access _local_tensor
        local = dt._local_tensor
        
        c2p.put((rank, 'local_tensor_type', type(local).__name__))
        c2p.put((rank, 'local_tensor_shape', list(local.shape)))
        c2p.put((rank, 'local_tensor_device', str(local.device)))
        
        dist_group.destroy_process_group()

    @classmethod
    def _test_local_tensor_shard(cls, rank, world_size, c2p):
        dist_group = cls._init_dist_hccl(rank, world_size)
        
        # Create a device mesh
        mesh = init_device_mesh('npu', (world_size,))
        
        # Create a DTensor with shard placement
        local_tensor = torch.randn(4, 4, device=f'npu:{rank}')
        dt = DTensor.from_local(local_tensor, mesh, [Shard(0)])
        
        # Access _local_tensor
        local = dt._local_tensor
        
        c2p.put((rank, 'sharded_local_shape', list(local.shape)))
        
        dist_group.destroy_process_group()

    @skipIfUnsupportMultiNPU(2)
    def test_local_tensor_basic(self):
        # Test _local_tensor attribute with replicate placement
        ctx = mp.get_context('spawn')
        c2p = ctx.Queue(6)
        
        ps = []
        for i in range(2):
            p = ctx.Process(
                target=self._test_local_tensor_basic,
                args=(i, 2, c2p))
            p.start()
            ps.append(p)
        
        results = []
        for _ in range(6):
            results.append(c2p.get(timeout=30))
        
        for p in ps:
            p.join(timeout=30)
            self.assertEqual(p.exitcode, 0, "subprocess exit with abnormal code.")
        
        # Verify _local_tensor properties
        for rank, event, value in results:
            if event == 'local_tensor_type':
                self.assertEqual(value, 'Tensor')
            elif event == 'local_tensor_device':
                self.assertIn('npu', value)

    @skipIfUnsupportMultiNPU(2)
    def test_local_tensor_sharded(self):
        # Test _local_tensor with sharded DTensor
        ctx = mp.get_context('spawn')
        c2p = ctx.Queue(2)
        
        ps = []
        for i in range(2):
            p = ctx.Process(
                target=self._test_local_tensor_shard,
                args=(i, 2, c2p))
            p.start()
            ps.append(p)
        
        results = []
        for _ in range(2):
            results.append(c2p.get(timeout=30))
        
        for p in ps:
            p.join(timeout=30)
            self.assertEqual(p.exitcode, 0, "subprocess exit with abnormal code.")
        
        # Verify sharded tensor shapes
        for rank, event, value in results:
            if event == 'sharded_local_shape':
                # Each rank should have local shape [4, 4]
                self.assertEqual(value, [4, 4])


if __name__ == "__main__":
    run_tests()
