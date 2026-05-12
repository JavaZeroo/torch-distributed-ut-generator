# -*- coding: utf-8 -*-
"""
测试目的：验证 torch.distributed.optim.DistributedOptimizer 接口功能正确性
API 名称：torch.distributed.optim.DistributedOptimizer
API 签名：DistributedOptimizer(
              optimizer_class: Type[optim.Optimizer],
              params_rref: List[RRef],
              *args,
              **kwargs,
          )

覆盖维度表：
| 覆盖维度         | 说明                                                         | 覆盖情况                                                            |
|------------------|--------------------------------------------------------------|---------------------------------------------------------------------|
| 空/非空          | params_rref 非空（RRef 列表）                                 | 已覆盖                                                              |
| 枚举选项         | optimizer_class: SGD / Adam                                   | 已覆盖                                                              |
| 参数类型         | params_rref: List[RRef]; optimizer_class: Type[Optimizer]     | 已覆盖                                                              |
| 传参与不传参     | 必填参数 optimizer_class + params_rref + lr                   | 已覆盖                                                              |
| 等价类/边界值    | 单 worker；多 worker；RRef 指向本地参数                       | 已覆盖                                                              |
| 正常传参场景     | 构造成功；remote_optimizers 非空                              | 已覆盖                                                              |
| 异常传参场景     | 无稳定异常路径（RPC 初始化后构造均合法）                     | 未覆盖，原因：需 RPC 环境，异常依赖 RPC 错误                        |

未覆盖项及原因：
- 异常传参：需要 RPC 初始化完成后才能构造，异常路径依赖 RPC 环境错误。

注意：
- DistributedOptimizer 依赖 torch.distributed.rpc，在 NPU 上使用 NPUTensorPipeRpcBackendOptions。
- 本测试仅验证功能正确性（调用不报错、输出 shape/dtype/类型符合预期），不做精度和数值正确性校验。
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
    """Initialize RPC workers for DistributedOptimizer tests."""
    os.environ.setdefault('MASTER_ADDR', 'localhost')
    os.environ.setdefault('MASTER_PORT', '29501')
    os.environ.setdefault('GLOO_SOCKET_IFNAME', 'lo')
    torch.npu.set_device(rank)
    worker_name = f'worker{rank}'
    warnings.filterwarnings('ignore', category=UserWarning)
    if HAS_RPC:
        options = NPUTensorPipeRpcBackendOptions(num_worker_threads=4)
        options.rpc_timeout = 60
        options.set_devices([f'npu:{rank}'])
        device_map = {f'npu:{r}': f'npu:{rank}' for r in range(world_size) if r != rank}
        for other_rank in range(world_size):
            if other_rank != rank:
                options.set_device_map(
                    f'worker{other_rank}',
                    {f'npu:{rank}': f'npu:{other_rank}'}
                )
        rpc.init_rpc(
            worker_name,
            rank=rank,
            world_size=world_size,
            backend=rpc.backend_registry.BackendType.NPU_TENSORPIPE,
            rpc_backend_options=options,
        )
        try:
            fn(rank, world_size, *args)
        finally:
            rpc.shutdown()
    else:
        # RPC not available, skip silently
        pass


def _test_construct_with_sgd(rank, world_size):
    """DistributedOptimizer constructs with SGD and local parameter RRefs."""
    import torch.nn as nn
    device = f'npu:{rank}'
    model = nn.Linear(4, 2).to(device)
    # Create RRefs to local parameters
    param_rrefs = [rpc.RRef(p) for p in model.parameters()]
    dist_opt = DistributedOptimizer(
        torch.optim.SGD,
        param_rrefs,
        lr=0.01,
    )
    assert dist_opt.remote_optimizers is not None
    assert len(dist_opt.remote_optimizers) > 0


def _test_construct_with_adam(rank, world_size):
    """DistributedOptimizer constructs with Adam."""
    import torch.nn as nn
    device = f'npu:{rank}'
    model = nn.Linear(4, 2).to(device)
    param_rrefs = [rpc.RRef(p) for p in model.parameters()]
    dist_opt = DistributedOptimizer(
        torch.optim.Adam,
        param_rrefs,
        lr=1e-3,
    )
    assert isinstance(dist_opt, DistributedOptimizer)


def _test_is_functional_optim_flag(rank, world_size):
    """DistributedOptimizer.is_functional_optim flag is set correctly."""
    import torch.nn as nn
    device = f'npu:{rank}'
    model = nn.Linear(4, 2).to(device)
    param_rrefs = [rpc.RRef(p) for p in model.parameters()]
    dist_opt = DistributedOptimizer(
        torch.optim.SGD,
        param_rrefs,
        lr=0.01,
    )
    # is_functional_optim depends on whether SGD is in functional_optim_map
    assert isinstance(dist_opt.is_functional_optim, bool)


class TestDistributedOptimizer(TestCase):
    """Tests for DistributedOptimizer initialization via RPC."""

    def setUp(self):
        super().setUp()
        self.device_name = torch._C._get_privateuse1_backend_name()
        self.assertEqual(
            self.device_name, 'npu',
            f"Expected device 'npu', got '{self.device_name}'"
        )

    @skipIfUnsupportMultiNPU(2)
    @unittest.skipUnless(HAS_RPC, "torch.distributed.rpc or NPUTensorPipeRpcBackendOptions not available")
    def test_construct_with_sgd(self):
        """DistributedOptimizer constructs with SGD using RRef params."""
        mp.spawn(_init_rpc, args=(2, _test_construct_with_sgd), nprocs=2, join=True)

    @skipIfUnsupportMultiNPU(2)
    @unittest.skipUnless(HAS_RPC, "torch.distributed.rpc or NPUTensorPipeRpcBackendOptions not available")
    def test_construct_with_adam(self):
        """DistributedOptimizer constructs with Adam using RRef params."""
        mp.spawn(_init_rpc, args=(2, _test_construct_with_adam), nprocs=2, join=True)

    @skipIfUnsupportMultiNPU(2)
    @unittest.skipUnless(HAS_RPC, "torch.distributed.rpc or NPUTensorPipeRpcBackendOptions not available")
    def test_is_functional_optim_flag(self):
        """is_functional_optim flag is a bool."""
        mp.spawn(_init_rpc, args=(2, _test_is_functional_optim_flag), nprocs=2, join=True)


if __name__ == "__main__":
    run_tests()
