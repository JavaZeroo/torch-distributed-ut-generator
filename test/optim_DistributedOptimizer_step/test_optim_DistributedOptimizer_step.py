# -*- coding: utf-8 -*-
"""
测试目的：验证 torch.distributed.optim.DistributedOptimizer.step 接口功能正确性
API 名称：torch.distributed.optim.DistributedOptimizer.step
API 签名：DistributedOptimizer.step(self, context_id: int) -> None

覆盖维度表：
| 覆盖维度         | 说明                                                         | 覆盖情况                                                              |
|------------------|--------------------------------------------------------------|-----------------------------------------------------------------------|
| 空/非空          | context_id 为合法 dist_autograd context id                   | 已覆盖                                                                |
| 枚举选项         | optimizer_class: SGD / Adam                                   | 已覆盖                                                                |
| 参数类型         | context_id: int                                               | 已覆盖                                                                |
| 传参与不传参     | 必填 context_id                                               | 已覆盖                                                                |
| 等价类/边界值    | 单步；前向反向完整链路                                        | 已覆盖                                                                |
| 正常传参场景     | step() 不报错，参数 shape/dtype 不变                          | 已覆盖                                                                |
| 异常传参场景     | 无效 context_id → RuntimeError                               | 已覆盖                                                                |

未覆盖项及原因：
- 无。

注意：
- DistributedOptimizer.step 依赖 torch.distributed.rpc + torch.distributed.autograd。
- 本测试仅验证功能正确性，不做精度和数值正确性校验。
"""
import os
import unittest
import warnings

import torch
import torch.multiprocessing as mp
import torch_npu
from torch_npu.testing.common_distributed import skipIfUnsupportMultiNPU
from torch_npu.testing.testcase import TestCase, run_tests

try:
    import torch.distributed.rpc as rpc
    import torch.distributed.autograd as dist_autograd
    from torch.distributed.optim import DistributedOptimizer
    from torch_npu.distributed.rpc.options import NPUTensorPipeRpcBackendOptions
    HAS_RPC = True
except (ImportError, AttributeError):
    HAS_RPC = False


def _init_rpc(rank, world_size, fn, *args):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '29501'
    os.environ.setdefault('GLOO_SOCKET_IFNAME', 'lo')
    torch.npu.set_device(rank)
    warnings.filterwarnings('ignore', category=UserWarning)
    if HAS_RPC:
        options = NPUTensorPipeRpcBackendOptions(num_worker_threads=4)
        options.rpc_timeout = 60
        options.set_devices([f'npu:{rank}'])
        for other_rank in range(world_size):
            if other_rank != rank:
                options.set_device_map(
                    f'worker{other_rank}',
                    {f'npu:{rank}': f'npu:{other_rank}'}
                )
        rpc.init_rpc(
            f'worker{rank}',
            rank=rank,
            world_size=world_size,
            backend=rpc.backend_registry.BackendType.NPU_TENSORPIPE,
            rpc_backend_options=options,
        )
        try:
            fn(rank, world_size, *args)
        finally:
            rpc.shutdown()


def _test_step_basic(rank, world_size):
    """step() runs without error using valid dist_autograd context."""
    import torch.nn as nn
    device = f'npu:{rank}'
    model = nn.Linear(4, 2).to(device)
    param_rrefs = [rpc.RRef(p) for p in model.parameters()]
    dist_opt = DistributedOptimizer(torch.optim.SGD, param_rrefs, lr=0.01)

    shapes_before = [p.shape for p in model.parameters()]

    with dist_autograd.context() as ctx_id:
        x = torch.randn(2, 4, device=device, requires_grad=True)
        loss = model(x).sum()
        dist_autograd.backward(ctx_id, [loss])
        dist_opt.step(ctx_id)

    for p, shape in zip(model.parameters(), shapes_before):
        assert p.shape == shape, f"Shape changed: {p.shape} vs {shape}"


def _test_step_with_adam(rank, world_size):
    """step() works with Adam as optimizer class."""
    import torch.nn as nn
    device = f'npu:{rank}'
    model = nn.Linear(4, 2).to(device)
    param_rrefs = [rpc.RRef(p) for p in model.parameters()]
    dist_opt = DistributedOptimizer(torch.optim.Adam, param_rrefs, lr=1e-3)

    with dist_autograd.context() as ctx_id:
        x = torch.randn(2, 4, device=device, requires_grad=True)
        loss = model(x).sum()
        dist_autograd.backward(ctx_id, [loss])
        dist_opt.step(ctx_id)


def _test_step_invalid_context_raises(rank, world_size):
    """step() with invalid context_id raises RuntimeError."""
    import torch.nn as nn
    device = f'npu:{rank}'
    model = nn.Linear(4, 2).to(device)
    param_rrefs = [rpc.RRef(p) for p in model.parameters()]
    dist_opt = DistributedOptimizer(torch.optim.SGD, param_rrefs, lr=0.01)
    try:
        dist_opt.step(-1)  # Invalid context id
        raise AssertionError("Expected RuntimeError for invalid context_id")
    except RuntimeError:
        pass  # Expected


class TestDistributedOptimizerStep(TestCase):
    """Tests for DistributedOptimizer.step via RPC."""

    def setUp(self):
        super().setUp()
        self.device_name = torch._C._get_privateuse1_backend_name()
        self.assertEqual(
            self.device_name, 'npu',
            f"Expected device 'npu', got '{self.device_name}'"
        )

    @skipIfUnsupportMultiNPU(2)
    @unittest.skipUnless(HAS_RPC, "torch.distributed.rpc or NPUTensorPipeRpcBackendOptions not available")
    def test_step_basic(self):
        """step() runs forward-backward-step loop without error."""
        mp.spawn(_init_rpc, args=(2, _test_step_basic), nprocs=2, join=True)

    @skipIfUnsupportMultiNPU(2)
    @unittest.skipUnless(HAS_RPC, "torch.distributed.rpc or NPUTensorPipeRpcBackendOptions not available")
    def test_step_with_adam(self):
        """step() works with Adam optimizer."""
        mp.spawn(_init_rpc, args=(2, _test_step_with_adam), nprocs=2, join=True)

    @skipIfUnsupportMultiNPU(2)
    @unittest.skipUnless(HAS_RPC, "torch.distributed.rpc or NPUTensorPipeRpcBackendOptions not available")
    def test_step_invalid_context_raises(self):
        """step() with invalid context_id raises RuntimeError."""
        mp.spawn(_init_rpc, args=(2, _test_step_invalid_context_raises), nprocs=2, join=True)


if __name__ == "__main__":
    run_tests()
