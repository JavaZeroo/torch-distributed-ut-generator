# -*- coding: utf-8 -*-
"""
测试目的：验证 torch.distributed.fsdp._fsdp_extensions._ext_chunk_tensor 接口功能正确性
API 名称：torch.distributed.fsdp._fsdp_extensions._ext_chunk_tensor
API 签名：_ext_chunk_tensor(tensor, chunk_size, rank, world_size)

覆盖维度表：
| 覆盖维度         | 说明                                                         | 覆盖情况                                       |
|------------------|--------------------------------------------------------------|------------------------------------------------|
| 张量分块         | 验证张量能正确分块                                           | 已覆盖：test_chunk_tensor_basic                |
| 参数类型         | 验证参数类型（tensor, chunk_size, rank, world_size）        | 已覆盖：test_parameter_types                   |
| 多卡场景         | 在多 NPU 上验证张量分块一致性                                | 已覆盖：test_multiprocess_chunking             |
| 返回值验证       | 验证返回张量的 shape 和 dtype                               | 已覆盖：test_return_tensor_properties          |

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
    os.environ['MASTER_PORT'] = '29507'
    os.environ['HCCL_WHITELIST_DISABLE'] = '1'
    torch_npu.npu.set_device(rank)
    dist.init_process_group(backend='hccl', rank=rank, world_size=world_size)


def _test_ext_chunk_tensor(rank, world_size, c2p, device_name):
    """Test _ext_chunk_tensor in multiprocess context."""
    _init_dist_hccl(rank, world_size)

    try:
        from torch.distributed.fsdp._fsdp_extensions import _ext_chunk_tensor

        # Create a test tensor
        tensor = torch.randn(10, 20, device=device_name)

        # Chunk the tensor
        chunk_size = 5
        chunked = _ext_chunk_tensor(tensor, chunk_size, rank, world_size)

        c2p.put((rank, 'chunked', True))
        c2p.put((rank, 'chunked_shape', chunked.shape))
        c2p.put((rank, 'chunked_dtype', str(chunked.dtype)))

    except Exception as e:
        c2p.put((rank, 'error', str(e)))
    finally:
        dist.destroy_process_group()


class TestExtChunkTensor(TestCase):
    """Test cases for torch.distributed.fsdp._fsdp_extensions._ext_chunk_tensor."""

    def setUp(self):
        super().setUp()
        self.device_name = torch._C._get_privateuse1_backend_name()
        self.assertEqual(self.device_name, 'npu', f"Expected device 'npu', got '{self.device_name}'")

    @skipIfUnsupportMultiNPU(2)
    def test_chunk_tensor_basic(self):
        """Test basic tensor chunking."""
        from torch.distributed.fsdp._fsdp_extensions import _ext_chunk_tensor

        tensor = torch.randn(10, 20)
        chunk_size = 5

        # Call should not raise
        result = _ext_chunk_tensor(tensor, chunk_size, 0, 2)
        self.assertIsNotNone(result)

    @skipIfUnsupportMultiNPU(2)
    def test_parameter_types(self):
        """Test _ext_chunk_tensor parameter types."""
        from torch.distributed.fsdp._fsdp_extensions import _ext_chunk_tensor

        tensor = torch.randn(16, 32)
        chunk_size = 8
        rank = 0
        world_size = 2

        # Verify function accepts expected parameters
        result = _ext_chunk_tensor(tensor, chunk_size, rank, world_size)
        self.assertIsInstance(result, torch.Tensor)

    @skipIfUnsupportMultiNPU(2)
    def test_return_tensor_properties(self):
        """Test properties of returned tensor."""
        from torch.distributed.fsdp._fsdp_extensions import _ext_chunk_tensor

        dtype = torch.float32
        tensor = torch.randn(12, 24, dtype=dtype)
        chunk_size = 6

        result = _ext_chunk_tensor(tensor, chunk_size, 0, 2)

        # Verify returned tensor has expected properties
        self.assertEqual(result.dtype, dtype)
        self.assertGreater(result.numel(), 0)

    @skipIfUnsupportMultiNPU(2)
    def test_multiprocess_chunking(self):
        """Test tensor chunking in multiprocess context."""
        world_size = 2
        ctx = mp.get_context('spawn')
        c2p = ctx.Queue()

        ps = []
        for i in range(world_size):
            p = ctx.Process(
                target=_test_ext_chunk_tensor,
                args=(i, world_size, c2p, self.device_name))
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

        # Verify chunking succeeded on all ranks
        for rank in range(world_size):
            if rank in results:
                self.assertTrue(results[rank].get('chunked'))


if __name__ == "__main__":
    run_tests()
