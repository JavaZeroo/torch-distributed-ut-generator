# -*- coding: utf-8 -*-
"""
测试目的：验证 torch.distributed.fsdp._fsdp_extensions.FSDPExtensions 接口功能正确性
API 名称：torch.distributed.fsdp._fsdp_extensions.FSDPExtensions
API 签名：FSDPExtensions 是 FSDP 扩展框架的抽象基类（ABC），
         必须实现以下抽象方法：
           pre_flatten_transform / post_unflatten_transform
           chunk_tensor / chunk_dtensor
           pre_load_state_dict_transform / all_gather_dtensor

覆盖维度表：
| 覆盖维度         | 说明                                                         | 覆盖情况                                       |
|------------------|--------------------------------------------------------------|------------------------------------------------|
| 抽象类约束       | 验证 FSDPExtensions 不能被直接实例化                         | 已覆盖：test_abstract_class_cannot_instantiate |
| 子类化创建       | 通过继承并实现抽象方法实例化                                 | 已覆盖：test_fsdp_extensions_creation          |
| 抽象方法集合     | 验证抽象方法集合与文档一致                                   | 已覆盖：test_fsdp_extensions_attributes        |
| 多卡场景         | 在多 NPU 环境下验证扩展子类的可创建性                        | 已覆盖：test_fsdp_extensions_multiprocess      |

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


def _make_concrete_extension_cls():
    """Build a concrete subclass that satisfies all abstract methods."""
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

    return _DummyFSDPExtensions


def _init_dist_hccl(rank, world_size):
    """Initialize distributed process with HCCL backend."""
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '29511'
    os.environ['HCCL_WHITELIST_DISABLE'] = '1'
    torch_npu.npu.set_device(rank)
    dist.init_process_group(backend='hccl', rank=rank, world_size=world_size)


def _test_fsdp_extensions_creation(rank, world_size, c2p):
    """Test FSDPExtensions subclass creation in multiprocess context."""
    _init_dist_hccl(rank, world_size)

    try:
        cls = _make_concrete_extension_cls()
        ext = cls()

        c2p.put((rank, 'created', cls.__name__))

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

    def test_abstract_class_cannot_instantiate(self):
        """Direct instantiation of the abstract base must raise TypeError."""
        from torch.distributed.fsdp._fsdp_extensions import FSDPExtensions
        with self.assertRaises(TypeError):
            FSDPExtensions()

    def test_fsdp_extensions_creation(self):
        """Test FSDPExtensions subclass creation."""
        from torch.distributed.fsdp._fsdp_extensions import FSDPExtensions
        cls = _make_concrete_extension_cls()
        ext = cls()
        self.assertIsNotNone(ext)
        self.assertIsInstance(ext, FSDPExtensions)

    def test_fsdp_extensions_attributes(self):
        """Test FSDPExtensions abstract method set."""
        from torch.distributed.fsdp._fsdp_extensions import FSDPExtensions

        expected_abstracts = {
            'pre_flatten_transform', 'post_unflatten_transform',
            'chunk_tensor', 'chunk_dtensor',
            'pre_load_state_dict_transform', 'all_gather_dtensor',
        }
        self.assertTrue(expected_abstracts.issubset(FSDPExtensions.__abstractmethods__))

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

        for rank in range(world_size):
            if rank in results:
                self.assertEqual(results[rank].get('created'), '_DummyFSDPExtensions')


if __name__ == "__main__":
    run_tests()
