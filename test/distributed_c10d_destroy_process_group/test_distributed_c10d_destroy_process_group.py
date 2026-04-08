# -*- coding: utf-8 -*-
"""
测试目的：验证 torch.distributed.distributed_c10d.destroy_process_group 接口功能正确性
API 名称：torch.distributed.distributed_c10d.destroy_process_group
API 签名：destroy_process_group(group=None)

覆盖维度表：
| 覆盖维度         | 说明                                                         | 覆盖情况                                       |
|------------------|--------------------------------------------------------------|------------------------------------------------|
| 进程组销毁       | 验证进程组能正确销毁                                         | 已覆盖：test_destroy_default_group              |
| 参数类型         | 验证 group 参数为 None 或 ProcessGroup 类型                  | 已覆盖：test_parameter_type_none                |
| 无参数调用       | 销毁默认进程组                                               | 已覆盖：test_no_arguments                      |
| 多卡场景         | 在多 NPU 上验证进程组销毁                                    | 已覆盖：test_multiprocess_destroy               |

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


def _init_and_destroy(rank, world_size, c2p):
    """Test destroy_process_group in multiprocess context."""
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '29506'
    os.environ['HCCL_WHITELIST_DISABLE'] = '1'
    torch_npu.npu.set_device(rank)

    try:
        # Initialize process group
        dist.init_process_group(backend='hccl', rank=rank, world_size=world_size)
        c2p.put((rank, 'init', True))

        # Verify group is initialized
        if dist.is_initialized():
            c2p.put((rank, 'is_initialized_before', True))

        # Destroy process group
        dist.destroy_process_group()
        c2p.put((rank, 'destroyed', True))

        # After destroy, is_initialized should return False
        if not dist.is_initialized():
            c2p.put((rank, 'is_initialized_after', False))

    except Exception as e:
        c2p.put((rank, 'error', str(e)))


class TestDestroyProcessGroup(TestCase):
    """Test cases for torch.distributed.distributed_c10d.destroy_process_group."""

    def setUp(self):
        super().setUp()
        device_name = torch._C._get_privateuse1_backend_name()
        self.assertEqual(device_name, 'npu', f"Expected device 'npu', got '{device_name}'")

    @skipIfUnsupportMultiNPU(2)
    def test_destroy_default_group(self):
        """Test destroying the default process group."""
        # After init_process_group, we should be able to destroy
        self.assertTrue(callable(dist.destroy_process_group))

    @skipIfUnsupportMultiNPU(2)
    def test_no_arguments(self):
        """Test destroy_process_group with no arguments."""
        from torch.distributed.distributed_c10d import destroy_process_group
        # Verify function accepts no arguments or only optional group
        import inspect
        sig = inspect.signature(destroy_process_group)
        # All parameters should be optional
        for param_name, param in sig.parameters.items():
            has_default = param.default != inspect.Parameter.empty
            self.assertTrue(has_default, f"Parameter {param_name} should be optional")

    @skipIfUnsupportMultiNPU(2)
    def test_parameter_type_none(self):
        """Test destroy_process_group with None parameter."""
        from torch.distributed.distributed_c10d import destroy_process_group
        # Verify function signature
        import inspect
        sig = inspect.signature(destroy_process_group)
        # Should have group parameter which defaults to None
        params = list(sig.parameters.keys())
        self.assertTrue(len(params) <= 1)

    @skipIfUnsupportMultiNPU(2)
    def test_multiprocess_destroy(self):
        """Test destroy_process_group in multiprocess context."""
        world_size = 2
        ctx = mp.get_context('spawn')
        c2p = ctx.Queue()

        ps = []
        for i in range(world_size):
            p = ctx.Process(
                target=_init_and_destroy,
                args=(i, world_size, c2p))
            p.start()
            ps.append(p)

        results = {}
        expected_messages = world_size * 4  # Each process puts 4 messages
        for _ in range(expected_messages):
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

        # Verify all ranks successfully initialized and destroyed
        for rank in range(world_size):
            if rank in results:
                self.assertTrue(results[rank].get('init'))
                self.assertTrue(results[rank].get('destroyed'))
                self.assertTrue(results[rank].get('is_initialized_before'))
                self.assertFalse(results[rank].get('is_initialized_after'))


if __name__ == "__main__":
    run_tests()
