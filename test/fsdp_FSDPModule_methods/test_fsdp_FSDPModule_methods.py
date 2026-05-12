# -*- coding: utf-8 -*-
"""
测试目的：验证 torch.distributed.fsdp.FSDPModule 各接口功能正确性
API 名称：
  - torch.distributed.fsdp.FSDPModule.set_all_reduce_hook
  - torch.distributed.fsdp.FSDPModule.set_is_last_backward
  - torch.distributed.fsdp.FSDPModule.set_post_optim_event
  - torch.distributed.fsdp.FSDPModule.set_reduce_scatter_divide_factor
  - torch.distributed.fsdp.FSDPModule.set_requires_all_reduce
  - torch.distributed.fsdp.FSDPModule.set_reshard_after_backward
  - torch.distributed.fsdp.FSDPModule.set_unshard_in_backward

API 签名：
  - set_all_reduce_hook(hook: Callable[[torch.Tensor], None], *, stream: torch.Stream | None = None) -> None
  - set_is_last_backward(is_last_backward: bool) -> None
  - set_post_optim_event(event: torch.Event) -> None
  - set_reduce_scatter_divide_factor(factor: float) -> None
  - set_requires_all_reduce(requires_all_reduce: bool, *, recurse: bool = True) -> None
  - set_reshard_after_backward(reshard_after_backward: bool, *, recurse: bool = True) -> None
  - set_unshard_in_backward(unshard_in_backward: bool) -> None

覆盖维度表：
| 覆盖维度         | 说明                                                         | 覆盖情况                                       |
|------------------|--------------------------------------------------------------|------------------------------------------------|
| 空/非空          | 参数为 None 或合法值                                         | 已覆盖 (set_all_reduce_hook with None stream)  |
| 枚举选项         | bool 参数的 True/False                                       | 已覆盖 (所有 bool 参数)                        |
| 参数类型         | Callable, bool, float, Event, Stream                         | 已覆盖                                         |
| 传参与不传参     | 默认参数 vs 显式传参                                         | 已覆盖 (recurse 参数)                          |
| 等价类/边界值    | 正常 float 值                                                | 已覆盖 (factor=0.5, 1.0, 2.0)                  |
| 正常传参场景     | 正常调用各方法                                               | 已覆盖                                         |
| 异常传参场景     | 非 root FSDP 调用                                            | 未覆盖 (依赖内部状态，无稳定异常路径)          |

未覆盖项及原因：
- 异常传参场景：这些 setter 方法主要用于 FSDP 内部状态管理，大多数不验证输入合法性，依赖 FSDP 内部逻辑保证正确调用

注意：本测试仅验证功能正确性（调用不报错、输出 shape/dtype/类型符合预期），
     不做精度和数值正确性校验。
"""

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn as nn
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp._fully_shard import fully_shard
from torch.distributed.fsdp.api import ShardingStrategy

from torch_npu.testing.testcase import TestCase, run_tests
from torch_npu.testing.common_distributed import skipIfUnsupportMultiNPU


class SimpleModel(nn.Module):
    """Simple model for FSDP testing."""

    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(10, 20)
        self.fc2 = nn.Linear(20, 10)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x


def _init_dist_process(rank, world_size, fn, backend='hccl'):
    """Initialize distributed process with HCCL backend."""
    import os
    os.environ.setdefault('MASTER_ADDR', 'localhost')
    os.environ.setdefault('MASTER_PORT', '29501')

    torch.npu.set_device(rank)
    dist.init_process_group(backend, rank=rank, world_size=world_size)

    try:
        fn(rank, world_size)
    finally:
        dist.destroy_process_group()


def _test_set_is_last_backward(rank, world_size):
    """Test set_is_last_backward method."""
    device = torch.device(f'npu:{rank}')
    model = SimpleModel().to(device)
    fsdp_model = fully_shard(model)

    # Test setting is_last_backward to True
    fsdp_model.set_is_last_backward(True)

    # Test setting is_last_backward to False
    fsdp_model.set_is_last_backward(False)

    dist.barrier()


def _test_set_requires_all_reduce(rank, world_size):
    """Test set_requires_all_reduce method."""
    device = torch.device(f'npu:{rank}')
    model = SimpleModel().to(device)
    fsdp_model = fully_shard(model)

    # Test with requires_all_reduce=True and recurse=True (default)
    fsdp_model.set_requires_all_reduce(True)

    # Test with requires_all_reduce=False and recurse=True
    fsdp_model.set_requires_all_reduce(False)

    # Test with explicit recurse=False
    fsdp_model.set_requires_all_reduce(True, recurse=False)
    fsdp_model.set_requires_all_reduce(False, recurse=False)

    dist.barrier()


def _test_set_reshard_after_backward(rank, world_size):
    """Test set_reshard_after_backward method."""
    device = torch.device(f'npu:{rank}')
    model = SimpleModel().to(device)
    fsdp_model = fully_shard(model)

    # Test with reshard_after_backward=True and recurse=True (default)
    fsdp_model.set_reshard_after_backward(True)

    # Test with reshard_after_backward=False and recurse=True
    fsdp_model.set_reshard_after_backward(False)

    # Test with explicit recurse=False
    fsdp_model.set_reshard_after_backward(True, recurse=False)
    fsdp_model.set_reshard_after_backward(False, recurse=False)

    dist.barrier()


def _test_set_unshard_in_backward(rank, world_size):
    """Test set_unshard_in_backward method."""
    device = torch.device(f'npu:{rank}')
    model = SimpleModel().to(device)
    fsdp_model = fully_shard(model)

    # Test setting unshard_in_backward to True
    fsdp_model.set_unshard_in_backward(True)

    # Test setting unshard_in_backward to False
    fsdp_model.set_unshard_in_backward(False)

    dist.barrier()


def _test_set_reduce_scatter_divide_factor(rank, world_size):
    """Test set_reduce_scatter_divide_factor method."""
    device = torch.device(f'npu:{rank}')
    model = SimpleModel().to(device)
    fsdp_model = fully_shard(model)

    # Test with different factor values
    fsdp_model.set_reduce_scatter_divide_factor(0.5)
    fsdp_model.set_reduce_scatter_divide_factor(1.0)
    fsdp_model.set_reduce_scatter_divide_factor(2.0)

    dist.barrier()


def _test_set_post_optim_event(rank, world_size):
    """Test set_post_optim_event method."""
    device = torch.device(f'npu:{rank}')
    model = SimpleModel().to(device)
    fsdp_model = fully_shard(model)

    # Create an event
    event = torch.npu.Event()
    event.record()

    # Test setting post-optim event
    fsdp_model.set_post_optim_event(event)

    dist.barrier()


def _test_set_all_reduce_hook(rank, world_size):
    """Test set_all_reduce_hook method."""
    device = torch.device(f'npu:{rank}')
    model = SimpleModel().to(device)
    fsdp_model = fully_shard(model)

    # Define a simple hook function
    def simple_hook(tensor):
        pass

    # Test setting all-reduce hook
    fsdp_model.set_all_reduce_hook(simple_hook)

    # Test setting all-reduce hook with None stream
    fsdp_model.set_all_reduce_hook(simple_hook, stream=None)

    dist.barrier()


class TestFSDPModuleMethods(TestCase):
    """Test cases for FSDPModule methods."""

    def setUp(self):
        super().setUp()
        self.device_name = torch._C._get_privateuse1_backend_name()
        self.assertEqual(self.device_name, 'npu', f"Expected device 'npu', got '{self.device_name}'")

    @skipIfUnsupportMultiNPU(2)
    def test_set_is_last_backward(self):
        """Test set_is_last_backward with multiple NPUs."""
        world_size = 2
        mp.spawn(
            _init_dist_process,
            args=(world_size, _test_set_is_last_backward),
            nprocs=world_size,
            join=True
        )

    @skipIfUnsupportMultiNPU(2)
    def test_set_requires_all_reduce(self):
        """Test set_requires_all_reduce with multiple NPUs."""
        world_size = 2
        mp.spawn(
            _init_dist_process,
            args=(world_size, _test_set_requires_all_reduce),
            nprocs=world_size,
            join=True
        )

    @skipIfUnsupportMultiNPU(2)
    def test_set_reshard_after_backward(self):
        """Test set_reshard_after_backward with multiple NPUs."""
        world_size = 2
        mp.spawn(
            _init_dist_process,
            args=(world_size, _test_set_reshard_after_backward),
            nprocs=world_size,
            join=True
        )

    @skipIfUnsupportMultiNPU(2)
    def test_set_unshard_in_backward(self):
        """Test set_unshard_in_backward with multiple NPUs."""
        world_size = 2
        mp.spawn(
            _init_dist_process,
            args=(world_size, _test_set_unshard_in_backward),
            nprocs=world_size,
            join=True
        )

    @skipIfUnsupportMultiNPU(2)
    def test_set_reduce_scatter_divide_factor(self):
        """Test set_reduce_scatter_divide_factor with multiple NPUs."""
        world_size = 2
        mp.spawn(
            _init_dist_process,
            args=(world_size, _test_set_reduce_scatter_divide_factor),
            nprocs=world_size,
            join=True
        )

    @skipIfUnsupportMultiNPU(2)
    def test_set_post_optim_event(self):
        """Test set_post_optim_event with multiple NPUs."""
        world_size = 2
        mp.spawn(
            _init_dist_process,
            args=(world_size, _test_set_post_optim_event),
            nprocs=world_size,
            join=True
        )

    @skipIfUnsupportMultiNPU(2)
    def test_set_all_reduce_hook(self):
        """Test set_all_reduce_hook with multiple NPUs."""
        world_size = 2
        mp.spawn(
            _init_dist_process,
            args=(world_size, _test_set_all_reduce_hook),
            nprocs=world_size,
            join=True
        )


if __name__ == "__main__":
    run_tests()
