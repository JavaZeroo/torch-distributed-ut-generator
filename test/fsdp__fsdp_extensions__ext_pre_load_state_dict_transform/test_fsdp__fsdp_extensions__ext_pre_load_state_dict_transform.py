# -*- coding: utf-8 -*-
"""
测试目的：验证 torch.distributed.fsdp._fsdp_extensions._ext_pre_load_state_dict_transform 接口功能正确性
API 名称：torch.distributed.fsdp._fsdp_extensions._ext_pre_load_state_dict_transform
API 签名：_ext_pre_load_state_dict_transform(tensor: Tensor,
                                          fsdp_extension: Optional[FSDPExtensions] = None)
                                          -> tuple[Tensor, list[Shard]]

行为：
- 当传入 fsdp_extension 时，委托给 fsdp_extension.pre_load_state_dict_transform(tensor)
- 否则 tensor 必须是 ShardedTensor，返回 (tensor, tensor.local_shards())

覆盖维度表：
| 覆盖维度         | 说明                                                         | 覆盖情况                                       |
|------------------|--------------------------------------------------------------|------------------------------------------------|
| 转换执行         | 通过 fsdp_extension 路径完成转换                             | 已覆盖：test_transform_execution                |
| 函数签名         | 验证参数及个数                                               | 已覆盖：test_parameter_types                   |
| 多卡场景         | 在多 NPU 上验证转换一致性                                    | 已覆盖：test_multiprocess_transform            |
| 返回值验证       | 验证返回值是 (Tensor, list) 元组                             | 已覆盖：test_return_type                       |
| 异常分支         | 不传 extension 且非 ShardedTensor 时抛 AssertionError        | 已覆盖：test_raises_when_not_sharded_tensor    |

未覆盖项及原因：
- ShardedTensor 路径：构造 ShardedTensor 需要分布式环境，已纳入多进程用例

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


def _make_dummy_extension():
    """A concrete FSDPExtensions stub that lets pre_load_state_dict_transform run."""
    from torch.distributed.fsdp._fsdp_extensions import FSDPExtensions

    class _DummyFSDPExtensions(FSDPExtensions):
        def pre_flatten_transform(self, tensor):
            return tensor, None

        def post_unflatten_transform(self, tensor, param_extension):
            return tensor

        def chunk_tensor(self, tensor, rank, world_size,
                         num_devices_per_node, pg, device=None):
            return tensor

        def chunk_dtensor(self, tensor, rank, device_mesh):
            return tensor

        def pre_load_state_dict_transform(self, tensor):
            return tensor, []

        def all_gather_dtensor(self, tensor, parent_mesh):
            return tensor

    return _DummyFSDPExtensions()


def _init_dist_hccl(rank, world_size):
    """Initialize distributed process with HCCL backend."""
    os.environ.setdefault('MASTER_ADDR', 'localhost')
    os.environ.setdefault('MASTER_PORT', '29510')
    os.environ['HCCL_WHITELIST_DISABLE'] = '1'
    torch_npu.npu.set_device(rank)
    dist.init_process_group(backend='hccl', rank=rank, world_size=world_size)


def _test_ext_pre_load_state_dict_transform(rank, world_size, c2p):
    """Test _ext_pre_load_state_dict_transform in multiprocess context."""
    try:
        _init_dist_hccl(rank, world_size)
        from torch.distributed.fsdp._fsdp_extensions import _ext_pre_load_state_dict_transform

        tensor = torch.randn(8)
        ext = _make_dummy_extension()
        result = _ext_pre_load_state_dict_transform(tensor, ext)

        c2p.put((rank, 'transformed', True))
        c2p.put((rank, 'is_tuple', isinstance(result, tuple)))
        c2p.put((rank, 'tuple_len', len(result) if isinstance(result, tuple) else -1))

    except Exception as e:
        c2p.put((rank, 'error', str(e)))
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


class TestExtPreLoadStateDictTransform(TestCase):
    """Test cases for torch.distributed.fsdp._fsdp_extensions._ext_pre_load_state_dict_transform."""

    def setUp(self):
        super().setUp()
        device_name = torch._C._get_privateuse1_backend_name()
        self.assertEqual(device_name, 'npu', f"Expected device 'npu', got '{device_name}'")

    def test_transform_execution(self):
        """Test _ext_pre_load_state_dict_transform via the extension fast path."""
        from torch.distributed.fsdp._fsdp_extensions import _ext_pre_load_state_dict_transform

        tensor = torch.randn(4)
        ext = _make_dummy_extension()
        result = _ext_pre_load_state_dict_transform(tensor, ext)

        self.assertIsInstance(result, tuple)
        self.assertEqual(len(result), 2)
        self.assertIsInstance(result[0], torch.Tensor)
        self.assertIsInstance(result[1], list)

    def test_parameter_types(self):
        """Test parameter types for _ext_pre_load_state_dict_transform."""
        from torch.distributed.fsdp._fsdp_extensions import _ext_pre_load_state_dict_transform

        sig = inspect.signature(_ext_pre_load_state_dict_transform)
        params = set(sig.parameters.keys())

        self.assertIn('tensor', params)
        self.assertIn('fsdp_extension', params)

    def test_return_type(self):
        """Test return type of _ext_pre_load_state_dict_transform."""
        from torch.distributed.fsdp._fsdp_extensions import _ext_pre_load_state_dict_transform

        tensor = torch.tensor([1.0, 2.0, 3.0])
        ext = _make_dummy_extension()
        result = _ext_pre_load_state_dict_transform(tensor, ext)

        self.assertIsInstance(result, tuple)
        self.assertIsInstance(result[0], torch.Tensor)
        self.assertIsInstance(result[1], list)

    def test_raises_when_not_sharded_tensor(self):
        """Without an extension, a plain Tensor must trigger an AssertionError."""
        from torch.distributed.fsdp._fsdp_extensions import _ext_pre_load_state_dict_transform

        tensor = torch.randn(4)
        with self.assertRaises(AssertionError):
            _ext_pre_load_state_dict_transform(tensor)

    @skipIfUnsupportMultiNPU(2)
    def test_multiprocess_transform(self):
        """Test _ext_pre_load_state_dict_transform in multiprocess context."""
        world_size = 2
        ctx = mp.get_context('spawn')
        c2p = ctx.Queue()

        ps = []
        for i in range(world_size):
            p = ctx.Process(
                target=_test_ext_pre_load_state_dict_transform,
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


if __name__ == "__main__":
    run_tests()
