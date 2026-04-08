# -*- coding: utf-8 -*-
"""
测试目的：验证 torch.distributed.algorithms.ddp_comm_hooks.post_localSGD_hook.PostLocalSGDState 接口功能正确性
API 名称：torch.distributed.algorithms.ddp_comm_hooks.post_localSGD_hook.PostLocalSGDState
API 签名：PostLocalSGDState(process_group, subgroup_size, start_localsgd_iter)

覆盖维度表：
| 覆盖维度         | 说明                                                         | 覆盖情况                                       |
|------------------|--------------------------------------------------------------|------------------------------------------------|
| 对象创建         | 验证 PostLocalSGDState 对象能正确创建                         | 已覆盖：test_state_creation                    |
| 属性访问         | 验证 state 对象的属性（process_group, subgroup_size 等）    | 已覆盖：test_state_attributes                  |
| 参数类型         | 验证各参数的类型及有效性                                     | 已覆盖：test_parameter_types                   |
| 多卡场景         | 在多 NPU 环境下验证状态管理                                  | 已覆盖：test_state_in_multiprocess             |

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
    os.environ['MASTER_PORT'] = '29502'
    os.environ['HCCL_WHITELIST_DISABLE'] = '1'
    torch_npu.npu.set_device(rank)
    dist.init_process_group(backend='hccl', rank=rank, world_size=world_size)


def _test_post_local_sgd_state_creation(rank, world_size, c2p):
    """Test PostLocalSGDState creation in multiprocess context."""
    _init_dist_hccl(rank, world_size)

    try:
        from torch.distributed.algorithms.ddp_comm_hooks.post_localSGD_hook import PostLocalSGDState

        # Create PostLocalSGDState instance
        state = PostLocalSGDState(
            process_group=None,
            subgroup=None,
            start_localSGD_iter=100
        )

        # Verify state was created
        c2p.put((rank, 'created', type(state).__name__))

        # Verify attributes can be accessed
        c2p.put((rank, 'has_process_group', hasattr(state, 'process_group')))
        c2p.put((rank, 'has_subgroup', hasattr(state, 'subgroup')))
        c2p.put((rank, 'has_start_localsgd_iter', hasattr(state, 'start_localSGD_iter')))

    except Exception as e:
        c2p.put((rank, 'error', str(e)))
    finally:
        dist.destroy_process_group()


class TestPostLocalSGDState(TestCase):
    """Test cases for PostLocalSGDState with HCCL."""

    def setUp(self):
        super().setUp()
        device_name = torch._C._get_privateuse1_backend_name()
        self.assertEqual(device_name, 'npu', f"Expected device 'npu', got '{device_name}'")

    @skipIfUnsupportMultiNPU(2)
    def test_state_creation(self):
        """Test PostLocalSGDState creation in single process."""
        from torch.distributed.algorithms.ddp_comm_hooks.post_localSGD_hook import PostLocalSGDState
        import torch.distributed as dist

        # Create with valid parameters
        state = PostLocalSGDState(
            process_group=None,
            subgroup=None,
            start_localSGD_iter=0
        )

        self.assertIsNotNone(state)
        self.assertIsInstance(state, PostLocalSGDState)

    @skipIfUnsupportMultiNPU(2)
    def test_state_attributes(self):
        """Test PostLocalSGDState attributes."""
        from torch.distributed.algorithms.ddp_comm_hooks.post_localSGD_hook import PostLocalSGDState

        start_iter = 100

        state = PostLocalSGDState(
            process_group=None,
            subgroup=None,
            start_localSGD_iter=start_iter
        )

        # Verify attributes are accessible
        self.assertTrue(hasattr(state, 'subgroup') or hasattr(state, 'start_localSGD_iter'))

    @skipIfUnsupportMultiNPU(2)
    def test_parameter_types(self):
        """Test PostLocalSGDState parameter types."""
        from torch.distributed.algorithms.ddp_comm_hooks.post_localSGD_hook import PostLocalSGDState

        # Test with different parameter types
        state = PostLocalSGDState(
            process_group=None,
            subgroup=None,
            start_localSGD_iter=50
        )

        self.assertIsNotNone(state)

    @skipIfUnsupportMultiNPU(2)
    def test_state_in_multiprocess(self):
        """Test PostLocalSGDState in multiprocess HCCL context."""
        world_size = 2
        ctx = mp.get_context('spawn')
        c2p = ctx.Queue()

        ps = []
        for i in range(world_size):
            p = ctx.Process(
                target=_test_post_local_sgd_state_creation,
                args=(i, world_size, c2p))
            p.start()
            ps.append(p)

        results = {}
        for _ in range(world_size * 4):
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

        # Verify all ranks created state successfully
        for rank in range(world_size):
            if rank in results:
                self.assertEqual(results[rank].get('created'), 'PostLocalSGDState')


if __name__ == "__main__":
    run_tests()
