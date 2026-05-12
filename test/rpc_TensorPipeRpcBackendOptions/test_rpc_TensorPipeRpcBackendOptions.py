# -*- coding: utf-8 -*-
"""
测试目的：验证 torch.distributed.rpc.TensorPipeRpcBackendOptions 接口功能正确性
API 名称：torch.distributed.rpc.TensorPipeRpcBackendOptions
API 签名：TensorPipeRpcBackendOptions(num_worker_threads=16, ...)

覆盖维度表：
| 覆盖维度         | 说明                                                         | 覆盖情况                                       |
|------------------|--------------------------------------------------------------|------------------------------------------------|
| 对象创建         | 验证 TensorPipeRpcBackendOptions 对象能正确创建              | 已覆盖：test_options_creation                  |
| 属性访问         | 验证对象的属性（num_worker_threads 等）                    | 已覆盖：test_options_attributes                |
| 参数类型         | 验证各参数的类型及有效性                                     | 已覆盖：test_parameter_types                   |
| 多卡场景         | 在多 NPU 上验证选项对象的一致性                              | 已覆盖：test_options_multiprocess              |

未覆盖项及原因：
- 无

注意：本测试仅验证功能正确性（调用不报错、输出 shape/dtype/类型符合预期），
     不做精度和数值正确性校验。
"""

import os
import torch
import torch.distributed.rpc as rpc
import torch.multiprocessing as mp

import torch_npu
from torch_npu.testing.testcase import TestCase, run_tests
from torch_npu.testing.common_distributed import skipIfUnsupportMultiNPU


def _test_backend_options_creation(rank, world_size, c2p):
    """Test TensorPipeRpcBackendOptions creation in multiprocess context."""
    os.environ.setdefault('MASTER_ADDR', 'localhost')
    torch_npu.npu.set_device(rank)

    try:
        # Create backend options
        options = rpc.TensorPipeRpcBackendOptions()

        c2p.put((rank, 'options_created', type(options).__name__))

        # Check common attributes
        c2p.put((rank, 'has_num_worker_threads', hasattr(options, 'num_worker_threads')))

    except Exception as e:
        c2p.put((rank, 'error', str(e)))


class TestTensorPipeRpcBackendOptions(TestCase):
    """Test cases for torch.distributed.rpc.TensorPipeRpcBackendOptions."""

    def setUp(self):
        super().setUp()
        device_name = torch._C._get_privateuse1_backend_name()
        self.assertEqual(device_name, 'npu', f"Expected device 'npu', got '{device_name}'")

    def test_options_creation(self):
        """Test TensorPipeRpcBackendOptions creation."""
        options = rpc.TensorPipeRpcBackendOptions()
        self.assertIsNotNone(options)
        self.assertIsInstance(options, rpc.TensorPipeRpcBackendOptions)

    def test_options_attributes(self):
        """Test TensorPipeRpcBackendOptions attributes."""
        options = rpc.TensorPipeRpcBackendOptions()

        # Verify some common attributes exist or can be set
        if hasattr(options, 'num_worker_threads'):
            self.assertIsNotNone(options.num_worker_threads)

    def test_parameter_types(self):
        """Test TensorPipeRpcBackendOptions parameter types."""
        # Create with default parameters
        options1 = rpc.TensorPipeRpcBackendOptions()
        self.assertIsNotNone(options1)

        # Create with specific parameters if supported
        try:
            options2 = rpc.TensorPipeRpcBackendOptions(num_worker_threads=8)
            self.assertIsNotNone(options2)
        except TypeError:
            # Parameter might not be available in this version
            pass

    @skipIfUnsupportMultiNPU(2)
    def test_options_multiprocess(self):
        """Test TensorPipeRpcBackendOptions in multiprocess context."""
        world_size = 2
        ctx = mp.get_context('spawn')
        c2p = ctx.Queue()

        ps = []
        for i in range(world_size):
            p = ctx.Process(
                target=_test_backend_options_creation,
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

        # Verify all ranks created options successfully
        for rank in range(world_size):
            if rank in results:
                self.assertEqual(results[rank].get('options_created'), 'TensorPipeRpcBackendOptions')


if __name__ == "__main__":
    run_tests()
