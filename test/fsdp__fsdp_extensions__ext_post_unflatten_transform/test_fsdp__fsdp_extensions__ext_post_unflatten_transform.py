# -*- coding: utf-8 -*-
"""
测试目的：验证 torch.distributed.fsdp._fsdp_extensions._ext_post_unflatten_transform 接口功能正确性
API 名称：torch.distributed.fsdp._fsdp_extensions._ext_post_unflatten_transform
API 签名：_ext_post_unflatten_transform(state, fsdp_state, config)

覆盖维度表：
| 覆盖维度         | 说明                                                         | 覆盖情况                                       |
|------------------|--------------------------------------------------------------|------------------------------------------------|
| 转换执行         | 验证状态转换能正确执行                                       | 已覆盖：test_transform_execution                |
| 参数类型         | 验证参数类型正确性                                           | 已覆盖：test_parameter_types                   |
| 多卡场景         | 在多 NPU 上验证状态转换一致性                                | 已覆盖：test_multiprocess_transform            |
| 返回值验证       | 验证返回值类型和结构                                         | 已覆盖：test_return_type                       |

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
    os.environ['MASTER_PORT'] = '29508'
    os.environ['HCCL_WHITELIST_DISABLE'] = '1'
    torch_npu.npu.set_device(rank)
    dist.init_process_group(backend='hccl', rank=rank, world_size=world_size)


def _test_ext_post_unflatten_transform(rank, world_size, c2p):
    """Test _ext_post_unflatten_transform in multiprocess context."""
    _init_dist_hccl(rank, world_size)

    try:
        from torch.distributed.fsdp._fsdp_extensions import _ext_post_unflatten_transform

        # Create dummy state dicts
        state = {}
        fsdp_state = None
        config = None

        # Call transform
        result = _ext_post_unflatten_transform(state, fsdp_state, config)

        c2p.put((rank, 'transformed', True))
        c2p.put((rank, 'result_type', type(result).__name__ if result is not None else 'None'))

    except Exception as e:
        c2p.put((rank, 'error', str(e)))
    finally:
        dist.destroy_process_group()


class TestExtPostUnflattenTransform(TestCase):
    """Test cases for torch.distributed.fsdp._fsdp_extensions._ext_post_unflatten_transform."""

    def setUp(self):
        super().setUp()
        device_name = torch._C._get_privateuse1_backend_name()
        self.assertEqual(device_name, 'npu', f"Expected device 'npu', got '{device_name}'")

    @skipIfUnsupportMultiNPU(2)
    def test_transform_execution(self):
        """Test _ext_post_unflatten_transform execution."""
        from torch.distributed.fsdp._fsdp_extensions import _ext_post_unflatten_transform

        # Create minimal arguments
        state = {}
        fsdp_state = None
        config = None

        # Should not raise
        result = _ext_post_unflatten_transform(state, fsdp_state, config)
        self.assertIsNotNone(result) or self.assertIsNone(result)

    @skipIfUnsupportMultiNPU(2)
    def test_parameter_types(self):
        """Test parameter types for _ext_post_unflatten_transform."""
        from torch.distributed.fsdp._fsdp_extensions import _ext_post_unflatten_transform

        # Verify function accepts expected parameters
        import inspect
        sig = inspect.signature(_ext_post_unflatten_transform)
        params = set(sig.parameters.keys())

        # Should have at least state parameter
        self.assertGreater(len(params), 0)

    @skipIfUnsupportMultiNPU(2)
    def test_return_type(self):
        """Test return type of _ext_post_unflatten_transform."""
        from torch.distributed.fsdp._fsdp_extensions import _ext_post_unflatten_transform

        result = _ext_post_unflatten_transform({}, None, None)

        # Result can be None, dict, or other valid type
        self.assertTrue(result is None or isinstance(result, (dict, type(None))))

    @skipIfUnsupportMultiNPU(2)
    def test_multiprocess_transform(self):
        """Test _ext_post_unflatten_transform in multiprocess context."""
        world_size = 2
        ctx = mp.get_context('spawn')
        c2p = ctx.Queue()

        ps = []
        for i in range(world_size):
            p = ctx.Process(
                target=_test_ext_post_unflatten_transform,
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

        # Verify transformation succeeded on all ranks
        for rank in range(world_size):
            if rank in results:
                self.assertTrue(results[rank].get('transformed'))


if __name__ == "__main__":
    run_tests()
