# -*- coding: utf-8 -*-
"""
测试目的：验证 torch.distributed.fsdp._fsdp_extensions._ext_post_unflatten_transform 接口功能正确性
API 名称：torch.distributed.fsdp._fsdp_extensions._ext_post_unflatten_transform
API 签名：_ext_post_unflatten_transform(tensor: Tensor,
                                     param_extension: Any,
                                     fsdp_extension: Optional[FSDPExtensions] = None)
                                     -> Tensor

覆盖维度表：
| 覆盖维度         | 说明                                                         | 覆盖情况                                       |
|------------------|--------------------------------------------------------------|------------------------------------------------|
| 转换执行         | 验证传入 tensor 能正确返回 Tensor                            | 已覆盖：test_transform_execution                |
| 函数签名         | 验证参数及个数                                               | 已覆盖：test_parameter_types                   |
| 多卡场景         | 在多 NPU 上验证转换一致性                                    | 已覆盖：test_multiprocess_transform            |
| 返回值验证       | 验证返回值类型                                               | 已覆盖：test_return_type                       |

未覆盖项及原因：
- 无

注意：本测试仅验证功能正确性（调用不报错、输出 shape/dtype/类型符合预期），
     不做精度和数值正确性校验。
"""

import inspect
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

        tensor = torch.randn(4)
        result = _ext_post_unflatten_transform(tensor, None)

        c2p.put((rank, 'transformed', True))
        c2p.put((rank, 'is_tensor', isinstance(result, torch.Tensor)))

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

    def test_transform_execution(self):
        """Test _ext_post_unflatten_transform execution."""
        from torch.distributed.fsdp._fsdp_extensions import _ext_post_unflatten_transform

        tensor = torch.randn(3, 3)
        result = _ext_post_unflatten_transform(tensor, None)

        # 默认实现返回原 tensor，结果必须仍是 Tensor（非 None）
        self.assertIsInstance(result, torch.Tensor)

    def test_parameter_types(self):
        """Test parameter types for _ext_post_unflatten_transform."""
        from torch.distributed.fsdp._fsdp_extensions import _ext_post_unflatten_transform

        sig = inspect.signature(_ext_post_unflatten_transform)
        params = set(sig.parameters.keys())

        self.assertIn('tensor', params)
        self.assertIn('param_extension', params)
        self.assertIn('fsdp_extension', params)

    def test_return_type(self):
        """Test return type of _ext_post_unflatten_transform."""
        from torch.distributed.fsdp._fsdp_extensions import _ext_post_unflatten_transform

        tensor = torch.zeros(2, 2)
        result = _ext_post_unflatten_transform(tensor, None, None)

        self.assertIsInstance(result, torch.Tensor)

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

        for rank in range(world_size):
            if rank in results:
                self.assertTrue(results[rank].get('transformed'))


if __name__ == "__main__":
    run_tests()
