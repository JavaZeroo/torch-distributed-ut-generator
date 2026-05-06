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
| 默认 passthrough | param_extension/fsdp_extension 任一为 None 时直接返回 tensor | 已覆盖：test_transform_execution                |
| extension 路径   | 同时传入非 None 的 fsdp_extension 与 param_extension         | 已覆盖：test_with_extension_path               |
| 函数签名         | 验证参数及个数                                               | 已覆盖：test_parameter_types                   |
| 多卡场景         | 在多 NPU 上验证 extension 路径转换一致性                     | 已覆盖：test_multiprocess_transform            |
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


def _make_marker_extension():
    """A concrete FSDPExtensions whose post_unflatten_transform leaves a marker."""
    from torch.distributed.fsdp._fsdp_extensions import FSDPExtensions

    class _MarkerFSDPExtensions(FSDPExtensions):
        def pre_flatten_transform(self, tensor):
            return tensor, None

        def post_unflatten_transform(self, tensor, param_extension):
            # 用 +param_extension 留下可观测痕迹，证明此分支被走到
            return tensor + param_extension

        def chunk_tensor(self, tensor, rank, world_size,
                         num_devices_per_node, pg, device=None):
            return tensor

        def chunk_dtensor(self, tensor, rank, device_mesh):
            return tensor

        def pre_load_state_dict_transform(self, tensor):
            return tensor, []

        def all_gather_dtensor(self, tensor, parent_mesh):
            return tensor

    return _MarkerFSDPExtensions()


def _init_dist_hccl(rank, world_size):
    """Initialize distributed process with HCCL backend."""
    os.environ.setdefault('MASTER_ADDR', 'localhost')
    os.environ.setdefault('MASTER_PORT', '29508')
    os.environ['HCCL_WHITELIST_DISABLE'] = '1'
    torch_npu.npu.set_device(rank)
    dist.init_process_group(backend='hccl', rank=rank, world_size=world_size)


def _test_ext_post_unflatten_transform(rank, world_size, c2p):
    """Test _ext_post_unflatten_transform extension path in multiprocess context."""
    try:
        _init_dist_hccl(rank, world_size)
        from torch.distributed.fsdp._fsdp_extensions import _ext_post_unflatten_transform

        tensor = torch.zeros(4)
        ext = _make_marker_extension()
        # extension 路径：fsdp_extension 与 param_extension 同时非 None
        result = _ext_post_unflatten_transform(tensor, 1.0, ext)

        c2p.put((rank, 'transformed', True))
        c2p.put((rank, 'is_tensor', isinstance(result, torch.Tensor)))
        c2p.put((rank, 'marker_applied', bool(torch.allclose(result, torch.ones(4)))))

    except Exception as e:
        c2p.put((rank, 'error', str(e)))
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


class TestExtPostUnflattenTransform(TestCase):
    """Test cases for torch.distributed.fsdp._fsdp_extensions._ext_post_unflatten_transform."""

    def setUp(self):
        super().setUp()
        device_name = torch._C._get_privateuse1_backend_name()
        self.assertEqual(device_name, 'npu', f"Expected device 'npu', got '{device_name}'")

    def test_transform_execution(self):
        """Default passthrough: param_extension/fsdp_extension 任一 None 时直接返回原 tensor。"""
        from torch.distributed.fsdp._fsdp_extensions import _ext_post_unflatten_transform

        tensor = torch.randn(3, 3)
        result = _ext_post_unflatten_transform(tensor, None)

        self.assertIsInstance(result, torch.Tensor)
        self.assertTrue(torch.equal(result, tensor))

    def test_with_extension_path(self):
        """Extension 路径：fsdp_extension 与 param_extension 同时非 None 时调用 extension。"""
        from torch.distributed.fsdp._fsdp_extensions import _ext_post_unflatten_transform

        tensor = torch.zeros(2, 2)
        ext = _make_marker_extension()
        result = _ext_post_unflatten_transform(tensor, 2.0, ext)

        self.assertIsInstance(result, torch.Tensor)
        # marker 实现把 param_extension 加到 tensor 上，验证此分支真的被走到
        self.assertTrue(torch.allclose(result, torch.full((2, 2), 2.0)))

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

        for rank in range(world_size):
            if rank in results:
                self.assertTrue(results[rank].get('transformed'))
                self.assertTrue(results[rank].get('is_tensor'))
                self.assertTrue(results[rank].get('marker_applied'))


if __name__ == "__main__":
    run_tests()
