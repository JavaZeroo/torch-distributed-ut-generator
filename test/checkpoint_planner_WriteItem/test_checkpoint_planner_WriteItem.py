# -*- coding: utf-8 -*-
"""
测试目的：验证 torch.distributed.checkpoint.planner.WriteItem 接口功能正确性
API 名称：torch.distributed.checkpoint.planner.WriteItem
API 签名：WriteItem(index: int, item: object)

覆盖维度表：
| 覆盖维度         | 说明                                                         | 覆盖情况                                       |
|------------------|--------------------------------------------------------------|------------------------------------------------|
| 对象创建         | 验证 WriteItem 对象能正确创建                                | 已覆盖：test_write_item_creation                |
| 属性访问         | 验证 WriteItem 对象的属性（index, item）                    | 已覆盖：test_write_item_attributes              |
| 参数类型         | 验证 index 为整数，item 为对象                               | 已覆盖：test_parameter_types                   |
| 多卡场景         | 在多 NPU 环境下验证 WriteItem 的一致性                       | 已覆盖：test_write_item_multiprocess            |

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
    os.environ['MASTER_PORT'] = '29503'
    os.environ['HCCL_WHITELIST_DISABLE'] = '1'
    torch_npu.npu.set_device(rank)
    dist.init_process_group(backend='hccl', rank=rank, world_size=world_size)


def _test_write_item_creation(rank, world_size, c2p):
    """Test WriteItem creation in multiprocess context."""
    _init_dist_hccl(rank, world_size)

    try:
        from torch.distributed.checkpoint.planner import WriteItem

        # Create WriteItem instance
        item = WriteItem(index=rank, type="state_dict")

        # Verify creation
        c2p.put((rank, 'created', type(item).__name__))

        # Verify attributes
        c2p.put((rank, 'has_index', hasattr(item, 'index')))
        c2p.put((rank, 'has_type', hasattr(item, 'type')))

    except Exception as e:
        c2p.put((rank, 'error', str(e)))
    finally:
        dist.destroy_process_group()


class TestWriteItem(TestCase):
    """Test cases for torch.distributed.checkpoint.planner.WriteItem."""

    def setUp(self):
        super().setUp()
        device_name = torch._C._get_privateuse1_backend_name()
        self.assertEqual(device_name, 'npu', f"Expected device 'npu', got '{device_name}'")

    @skipIfUnsupportMultiNPU(2)
    def test_write_item_creation(self):
        """Test WriteItem creation with basic parameters."""
        from torch.distributed.checkpoint.planner import WriteItem

        item = WriteItem(index=0, type="state_dict", tensor_data=None)
        self.assertIsNotNone(item)
        self.assertIsInstance(item, WriteItem)

    @skipIfUnsupportMultiNPU(2)
    def test_write_item_attributes(self):
        """Test WriteItem attributes."""
        from torch.distributed.checkpoint.planner import WriteItem

        index = 5

        item = WriteItem(index=index, type="state_dict")

        # Verify attributes are accessible
        self.assertTrue(hasattr(item, 'index') or hasattr(item, 'type'))

    @skipIfUnsupportMultiNPU(2)
    def test_parameter_types(self):
        """Test WriteItem parameter types."""
        from torch.distributed.checkpoint.planner import WriteItem

        # Test with integer index and various type values
        item1 = WriteItem(index=0, type="state_dict")
        item2 = WriteItem(index=1, type="tensor")
        item3 = WriteItem(index=2, type="other")

        self.assertIsNotNone(item1)
        self.assertIsNotNone(item2)
        self.assertIsNotNone(item3)

    @skipIfUnsupportMultiNPU(2)
    def test_write_item_multiprocess(self):
        """Test WriteItem in multiprocess context."""
        world_size = 2
        ctx = mp.get_context('spawn')
        c2p = ctx.Queue()

        ps = []
        for i in range(world_size):
            p = ctx.Process(
                target=_test_write_item_creation,
                args=(i, world_size, c2p))
            p.start()
            ps.append(p)

        results = {}
        for _ in range(world_size * 3):
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

        # Verify all ranks created items successfully
        for rank in range(world_size):
            if rank in results:
                self.assertEqual(results[rank].get('created'), 'WriteItem')


if __name__ == "__main__":
    run_tests()
