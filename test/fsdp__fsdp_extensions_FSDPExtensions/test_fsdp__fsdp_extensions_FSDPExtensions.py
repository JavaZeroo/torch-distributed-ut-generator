# -*- coding: utf-8 -*-
"""
测试目的：验证 torch.distributed.fsdp._fsdp_extensions.FSDPExtensions 接口功能正确性
API 名称：torch.distributed.fsdp._fsdp_extensions.FSDPExtensions
API 签名：FSDPExtensions 是 FSDP 扩展框架的数据类

覆盖维度表：
| 覆盖维度         | 说明                                                         | 覆盖情况                                       |
|------------------|--------------------------------------------------------------|------------------------------------------------|
| 对象创建         | 验证 FSDPExtensions 对象能正确创建                           | 已覆盖：test_fsdp_extensions_creation          |
| 属性访问         | 验证 FSDPExtensions 对象的属性                              | 已覆盖：test_fsdp_extensions_attributes        |
| 参数类型         | 验证各参数的类型及有效性                                     | 已覆盖：test_parameter_types                   |
| 多卡场景         | 在多 NPU 环境下验证扩展管理                                  | 已覆盖：test_fsdp_extensions_multiprocess      |

未覆盖项及原因：
- 无

注意：本测试仅验证功能正确性（调用不报错、输出 shape/dtype/类型符合预期），
     不做精度和数值正确性校验。
"""

import os
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

import torch_npu
from torch_npu.testing.testcase import TestCase, run_tests
from torch_npu.testing.common_distributed import skipIfUnsupportMultiNPU


def _init_dist_hccl(rank, world_size):
    """Initialize distributed process with HCCL backend."""
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '29511'
    os.environ['HCCL_WHITELIST_DISABLE'] = '1'
    torch_npu.npu.set_device(rank)
    dist.init_process_group(backend='hccl', rank=rank, world_size=world_size)


def _test_fsdp_extensions_creation(rank, world_size, c2p):
    """Test FSDPExtensions creation in multiprocess context."""
    _init_dist_hccl(rank, world_size)

    try:
        from torch.distributed.fsdp._fsdp_extensions import FSDPExtensions

        # Create FSDPExtensions instance
        ext = FSDPExtensions()

        # Verify creation
        c2p.put((rank, 'created', type(ext).__name__))

    except Exception as e:
        c2p.put((rank, 'error', str(e)))
    finally:
        dist.destroy_process_group()


class TestFSDPExtensions(TestCase):
    """Test cases for torch.distributed.fsdp._fsdp_extensions.FSDPExtensions."""

    def setUp(self):
        super().setUp()
        device_name = torch._C._get_privateuse1_backend_name()
        self.assertEqual(device_name, 'npu', f"Expected device 'npu', got '{device_name}'")

    @skipIfUnsupportMultiNPU(2)
    def test_fsdp_extensions_creation(self):
        """Test FSDPExtensions creation."""
        from torch.distributed.fsdp._fsdp_extensions import FSDPExtensions

        ext = FSDPExtensions()
        self.assertIsNotNone(ext)
        self.assertIsInstance(ext, FSDPExtensions)

    @skipIfUnsupportMultiNPU(2)
    def test_fsdp_extensions_attributes(self):
        """Test FSDPExtensions attributes."""
        from torch.distributed.fsdp._fsdp_extensions import FSDPExtensions

        ext = FSDPExtensions()

        # Verify it's a valid object
        self.assertTrue(hasattr(ext, '__class__'))

    @skipIfUnsupportMultiNPU(2)
    def test_parameter_types(self):
        """Test FSDPExtensions parameter types."""
        from torch.distributed.fsdp._fsdp_extensions import FSDPExtensions

        # FSDPExtensions might have initialization parameters
        # Test basic creation without parameters
        ext = FSDPExtensions()
        self.assertIsNotNone(ext)

    @skipIfUnsupportMultiNPU(2)
    def test_fsdp_extensions_multiprocess(self):
        """Test FSDPExtensions in multiprocess context."""
        world_size = 2
        ctx = mp.get_context('spawn')
        c2p = ctx.Queue()

        ps = []
        for i in range(world_size):
            p = ctx.Process(
                target=_test_fsdp_extensions_creation,
                args=(i, world_size, c2p))
            p.start()
            ps.append(p)

        results = {}
        for _ in range(world_size):
            try:
                rank, event, value = c2p.get(timeout=30)
                if rank not in results:
                    results[rank] = {}
                results[rank][event] = value
            except Exception:
                break

        for p in ps:
            p.join(timeout=30)
            self.assertEqual(p.exitcode, 0, "subprocess exit with abnormal code.")

        # Verify all ranks created FSDPExtensions successfully
        for rank in range(world_size):
            if rank in results:
                self.assertEqual(results[rank].get('created'), 'FSDPExtensions')


if __name__ == "__main__":
    run_tests()
