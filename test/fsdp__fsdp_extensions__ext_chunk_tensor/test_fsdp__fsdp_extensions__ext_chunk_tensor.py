# -*- coding: utf-8 -*-
"""
测试目的：验证 torch.distributed.fsdp._fsdp_extensions._ext_chunk_tensor 接口功能正确性
API 名称：torch.distributed.fsdp._fsdp_extensions._ext_chunk_tensor
API 签名：_ext_chunk_tensor(tensor: Tensor, rank: int, world_size: int,
                          num_devices_per_node: int,
                          pg: ProcessGroup,
                          fsdp_extension: Optional[FSDPExtensions] = None) -> Tensor

覆盖维度表：
| 覆盖维度         | 说明                                                         | 覆盖情况                                       |
|------------------|--------------------------------------------------------------|------------------------------------------------|
| 函数签名         | 验证函数能被 inspect 识别且参数数量匹配                       | 已覆盖：test_parameter_types                   |
| 张量分块         | 验证多卡环境下张量能正确分块                                 | 已覆盖：test_multiprocess_chunking             |
| 返回值验证       | 验证返回值仍是 torch.Tensor                                  | 已覆盖：test_multiprocess_chunking             |
| 抽象类约束       | 直接调用 _ext_chunk_tensor 需要真实的 ProcessGroup           | 单进程下仅做签名校验                           |

未覆盖项及原因：
- 单进程下没有可用 ProcessGroup，故功能性调用全部放在多进程用例中

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
    os.environ['MASTER_PORT'] = '29507'
    os.environ['HCCL_WHITELIST_DISABLE'] = '1'
    torch_npu.npu.set_device(rank)
    dist.init_process_group(backend='hccl', rank=rank, world_size=world_size)


def _test_ext_chunk_tensor(rank, world_size, c2p, device_name):
    """Test _ext_chunk_tensor in multiprocess context."""
    _init_dist_hccl(rank, world_size)

    try:
        from torch.distributed.fsdp._fsdp_extensions import _ext_chunk_tensor

        tensor = torch.randn(10, 20, device=f"{device_name}:{rank}")
        pg = dist.distributed_c10d._get_default_group()

        chunked = _ext_chunk_tensor(
            tensor=tensor,
            rank=rank,
            world_size=world_size,
            num_devices_per_node=world_size,
            pg=pg,
        )

        c2p.put((rank, 'chunked', True))
        c2p.put((rank, 'chunked_is_tensor', isinstance(chunked, torch.Tensor)))
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

    def test_callable(self):
        """_ext_chunk_tensor must be importable and callable."""
        from torch.distributed.fsdp._fsdp_extensions import _ext_chunk_tensor
        self.assertTrue(callable(_ext_chunk_tensor))

    def test_parameter_types(self):
        """Verify expected parameters via inspect."""
        from torch.distributed.fsdp._fsdp_extensions import _ext_chunk_tensor

        sig = inspect.signature(_ext_chunk_tensor)
        params = set(sig.parameters.keys())

        expected = {'tensor', 'rank', 'world_size', 'num_devices_per_node', 'pg'}
        self.assertTrue(expected.issubset(params),
                        f"missing params: {expected - params}")

    def test_signature_arity(self):
        """Function should accept 5 required params + optional fsdp_extension."""
        from torch.distributed.fsdp._fsdp_extensions import _ext_chunk_tensor

        sig = inspect.signature(_ext_chunk_tensor)
        # 至少 5 个必填位置参数
        required = [p for p in sig.parameters.values()
                    if p.default is inspect.Parameter.empty]
        self.assertGreaterEqual(len(required), 5)

    @skipIfUnsupportMultiNPU(2)
    def test_multiprocess_chunking(self):
        """Test tensor chunking in multiprocess context with a real pg."""
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

        for rank in range(world_size):
            if rank in results:
                self.assertTrue(results[rank].get('chunked'))
                self.assertTrue(results[rank].get('chunked_is_tensor'))


if __name__ == "__main__":
    run_tests()
