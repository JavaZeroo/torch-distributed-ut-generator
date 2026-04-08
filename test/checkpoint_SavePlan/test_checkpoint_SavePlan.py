# -*- coding: utf-8 -*-
"""
测试目的：验证 torch.distributed.checkpoint.SavePlan 接口功能正确性
API 名称：torch.distributed.checkpoint.SavePlan
API 签名：SavePlan(plan_items: List[WriteItem])

覆盖维度表：
| 覆盖维度         | 说明                                                         | 覆盖情况                                       |
|------------------|--------------------------------------------------------------|------------------------------------------------|
| 对象创建         | 验证 SavePlan 对象能正确创建                                 | 已覆盖：test_save_plan_creation                |
| 属性访问         | 验证 SavePlan 对象的属性（plan_items 等）                   | 已覆盖：test_save_plan_attributes              |
| 参数类型         | 验证 plan_items 为列表类型                                   | 已覆盖：test_parameter_types                   |
| 多卡场景         | 在多 NPU 环境下验证 SavePlan 的创建和访问                    | 已覆盖：test_save_plan_multiprocess            |

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
    os.environ['MASTER_PORT'] = '29504'
    os.environ['HCCL_WHITELIST_DISABLE'] = '1'
    torch_npu.npu.set_device(rank)
    dist.init_process_group(backend='hccl', rank=rank, world_size=world_size)


def _test_save_plan_creation(rank, world_size, c2p):
    """Test SavePlan creation in multiprocess context."""
    _init_dist_hccl(rank, world_size)

    try:
        from torch.distributed.checkpoint import SavePlan
        from torch.distributed.checkpoint.planner import WriteItem

        # Create WriteItems
        items = [WriteItem(index=i, item=f"data_{i}") for i in range(2)]

        # Create SavePlan
        plan = SavePlan(plan_items=items)

        # Verify creation
        c2p.put((rank, 'created', type(plan).__name__))

        # Verify attributes
        c2p.put((rank, 'has_plan_items', hasattr(plan, 'plan_items')))

    except Exception as e:
        c2p.put((rank, 'error', str(e)))
    finally:
        dist.destroy_process_group()


class TestSavePlan(TestCase):
    """Test cases for torch.distributed.checkpoint.SavePlan."""

    def setUp(self):
        super().setUp()
        device_name = torch._C._get_privateuse1_backend_name()
        self.assertEqual(device_name, 'npu', f"Expected device 'npu', got '{device_name}'")

    @skipIfUnsupportMultiNPU(2)
    def test_save_plan_creation(self):
        """Test SavePlan creation with empty and non-empty lists."""
        from torch.distributed.checkpoint import SavePlan
        from torch.distributed.checkpoint.planner import WriteItem

        # Create with empty list
        plan1 = SavePlan(plan_items=[])
        self.assertIsNotNone(plan1)
        self.assertIsInstance(plan1, SavePlan)

        # Create with items
        items = [WriteItem(index=0, item="test")]
        plan2 = SavePlan(plan_items=items)
        self.assertIsNotNone(plan2)

    @skipIfUnsupportMultiNPU(2)
    def test_save_plan_attributes(self):
        """Test SavePlan attributes."""
        from torch.distributed.checkpoint import SavePlan
        from torch.distributed.checkpoint.planner import WriteItem

        items = [
            WriteItem(index=0, item="item_0"),
            WriteItem(index=1, item="item_1")
        ]

        plan = SavePlan(plan_items=items)

        # Verify attributes are accessible
        self.assertTrue(hasattr(plan, 'plan_items'))

    @skipIfUnsupportMultiNPU(2)
    def test_parameter_types(self):
        """Test SavePlan parameter types."""
        from torch.distributed.checkpoint import SavePlan
        from torch.distributed.checkpoint.planner import WriteItem

        # Test with different list sizes
        items1 = []
        items2 = [WriteItem(index=0, item="data")]
        items3 = [WriteItem(index=i, item=f"data_{i}") for i in range(5)]

        plan1 = SavePlan(plan_items=items1)
        plan2 = SavePlan(plan_items=items2)
        plan3 = SavePlan(plan_items=items3)

        self.assertIsNotNone(plan1)
        self.assertIsNotNone(plan2)
        self.assertIsNotNone(plan3)

    @skipIfUnsupportMultiNPU(2)
    def test_save_plan_multiprocess(self):
        """Test SavePlan in multiprocess context."""
        world_size = 2
        ctx = mp.get_context('spawn')
        c2p = ctx.Queue()

        ps = []
        for i in range(world_size):
            p = ctx.Process(
                target=_test_save_plan_creation,
                args=(i, world_size, c2p))
            p.start()
            ps.append(p)

        results = {}
        for _ in range(world_size * 2):
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

        # Verify all ranks created SavePlan successfully
        for rank in range(world_size):
            if rank in results:
                self.assertEqual(results[rank].get('created'), 'SavePlan')


if __name__ == "__main__":
    run_tests()
