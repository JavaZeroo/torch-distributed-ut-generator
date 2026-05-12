# -*- coding: utf-8 -*-
"""
测试目的：验证 torch.distributed.fsdp.FSDPModule 及其关键方法的接口功能正确性
API 名称：
  - torch.distributed.fsdp.FSDPModule
  - torch.distributed.fsdp.FSDPModule.reshard
  - torch.distributed.fsdp.FSDPModule.set_modules_to_backward_prefetch
  - torch.distributed.fsdp.FSDPModule.set_modules_to_forward_prefetch
  - torch.distributed.fsdp.FSDPModule.set_requires_gradient_sync

API 签名：
  - class FSDPModule  (mixin，由 fully_shard() 注入)
  - reshard() -> None
  - set_modules_to_backward_prefetch(modules: list[FSDPModule]) -> None
  - set_modules_to_forward_prefetch(modules: list[FSDPModule]) -> None
  - set_requires_gradient_sync(requires_gradient_sync: bool, *, recurse: bool = True) -> None

覆盖维度表：
| 覆盖维度         | 说明                                                         | 覆盖情况                                       |
|------------------|--------------------------------------------------------------|------------------------------------------------|
| 空/非空          | set_*prefetch 传空列表 vs 非空列表                           | 已覆盖：test_set_forward_prefetch_empty_list   |
| 枚举选项         | set_requires_gradient_sync 的 True/False                     | 已覆盖：test_set_requires_gradient_sync        |
| 参数类型         | list[FSDPModule], bool                                       | 已覆盖                                         |
| 传参与不传参     | recurse 参数显式传入 vs 使用默认值                           | 已覆盖：test_set_requires_gradient_sync        |
| 等价类/边界值    | 空列表、单元素列表                                           | 已覆盖                                         |
| 正常传参场景     | reshard、forward/backward prefetch、gradient sync 正常调用   | 已覆盖                                         |
| 异常传参场景     | set_*prefetch 传入非 FSDPModule 抛 AssertionError            | 已覆盖：test_set_forward_prefetch_invalid_type |

未覆盖项及原因：
- FSDPModule 直接实例化：FSDPModule 是 mixin，无法单独实例化，必须通过 fully_shard() 使用
- reshard 后的内存行为验证：依赖内部实现细节，无公开 API 可断言

注意：本测试仅验证功能正确性（调用不报错、输出 shape/dtype/类型符合预期），
     不做精度和数值正确性校验。
"""

import os
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn as nn
from torch.distributed.fsdp._fully_shard import fully_shard
from torch.distributed.fsdp._fully_shard._fully_shard import FSDPModule

import torch_npu  # noqa: F401 — registers NPU backend
from torch_npu.testing.testcase import TestCase, run_tests
from torch_npu.testing.common_distributed import skipIfUnsupportMultiNPU


class SimpleModel(nn.Module):
    """Simple two-layer MLP for FSDP testing."""

    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(16, 32)
        self.fc2 = nn.Linear(32, 16)

    def forward(self, x):
        return self.fc2(torch.relu(self.fc1(x)))


class TwoLayerModel(nn.Module):
    """Two-submodule model for prefetch testing."""

    def __init__(self):
        super().__init__()
        self.layer1 = SimpleModel()
        self.layer2 = SimpleModel()

    def forward(self, x):
        return self.layer2(self.layer1(x))


def _init_dist_process(rank, world_size, fn, backend='hccl'):
    """Initialize distributed process with HCCL backend."""
    os.environ.setdefault('MASTER_ADDR', 'localhost')
    os.environ.setdefault('MASTER_PORT', '29515')
    os.environ['HCCL_WHITELIST_DISABLE'] = '1'
    torch_npu.npu.set_device(rank)
    dist.init_process_group(backend, rank=rank, world_size=world_size)
    try:
        fn(rank, world_size)
    finally:
        dist.destroy_process_group()


# ---------------------------------------------------------------------------
# Per-rank worker functions
# ---------------------------------------------------------------------------

def _test_fsdp_module_isinstance(rank, world_size):
    """Verify that fully_shard produces an FSDPModule instance."""
    device = torch.device(f'npu:{rank}')
    model = SimpleModel().to(device)
    fsdp_model = fully_shard(model)
    assert isinstance(fsdp_model, FSDPModule), (
        f"Expected FSDPModule, got {type(fsdp_model)}"
    )
    dist.barrier()


def _test_reshard(rank, world_size):
    """Test reshard() can be called without error after unshard."""
    device = torch.device(f'npu:{rank}')
    model = SimpleModel().to(device)
    fsdp_model = fully_shard(model)

    # unshard then immediately reshard
    fsdp_model.unshard()
    fsdp_model.reshard()

    # reshard on already-resharded module should also be safe
    fsdp_model.reshard()

    dist.barrier()


def _test_set_requires_gradient_sync(rank, world_size):
    """Test set_requires_gradient_sync with all parameter combinations."""
    device = torch.device(f'npu:{rank}')
    model = SimpleModel().to(device)
    fsdp_model = fully_shard(model)

    # True with default recurse=True
    fsdp_model.set_requires_gradient_sync(True)
    # False with default recurse=True
    fsdp_model.set_requires_gradient_sync(False)
    # True with explicit recurse=True
    fsdp_model.set_requires_gradient_sync(True, recurse=True)
    # False with explicit recurse=False (only current module)
    fsdp_model.set_requires_gradient_sync(False, recurse=False)
    # Restore to True
    fsdp_model.set_requires_gradient_sync(True, recurse=False)

    dist.barrier()


def _test_set_requires_gradient_sync_nested(rank, world_size):
    """Test set_requires_gradient_sync on nested FSDP modules."""
    device = torch.device(f'npu:{rank}')
    model = TwoLayerModel().to(device)
    # shard each submodule individually, then the root
    fully_shard(model.layer1)
    fully_shard(model.layer2)
    fsdp_model = fully_shard(model)

    # recurse=True should propagate to all submodules
    fsdp_model.set_requires_gradient_sync(False, recurse=True)
    fsdp_model.set_requires_gradient_sync(True, recurse=True)

    dist.barrier()


def _test_set_modules_to_forward_prefetch(rank, world_size):
    """Test set_modules_to_forward_prefetch with empty and non-empty lists."""
    device = torch.device(f'npu:{rank}')
    model = TwoLayerModel().to(device)
    fully_shard(model.layer1)
    fully_shard(model.layer2)
    fsdp_model = fully_shard(model)

    # Empty list — clears prefetch state
    fsdp_model.set_modules_to_forward_prefetch([])

    # Singleton list — explicit forward prefetch of layer2
    fsdp_model.set_modules_to_forward_prefetch([model.layer2])

    # Reset to empty
    fsdp_model.set_modules_to_forward_prefetch([])

    dist.barrier()


def _test_set_modules_to_forward_prefetch_invalid(rank, world_size):
    """Test set_modules_to_forward_prefetch raises on non-FSDPModule input."""
    device = torch.device(f'npu:{rank}')
    model = TwoLayerModel().to(device)
    fully_shard(model.layer1)
    fully_shard(model.layer2)
    fsdp_model = fully_shard(model)

    # Pass a plain nn.Module (not an FSDPModule) — should raise AssertionError
    plain = nn.Linear(4, 4)
    raised = False
    try:
        fsdp_model.set_modules_to_forward_prefetch([plain])
    except (AssertionError, Exception):
        raised = True
    assert raised, "Expected an exception for non-FSDPModule input"

    dist.barrier()


def _test_set_modules_to_backward_prefetch(rank, world_size):
    """Test set_modules_to_backward_prefetch with empty and non-empty lists."""
    device = torch.device(f'npu:{rank}')
    model = TwoLayerModel().to(device)
    fully_shard(model.layer1)
    fully_shard(model.layer2)
    fsdp_model = fully_shard(model)

    # Empty list — clears backward prefetch state
    fsdp_model.set_modules_to_backward_prefetch([])

    # Singleton list — explicit backward prefetch of layer1
    fsdp_model.set_modules_to_backward_prefetch([model.layer1])

    # Reset to empty
    fsdp_model.set_modules_to_backward_prefetch([])

    dist.barrier()


def _test_set_modules_to_backward_prefetch_invalid(rank, world_size):
    """Test set_modules_to_backward_prefetch raises on non-FSDPModule input."""
    device = torch.device(f'npu:{rank}')
    model = TwoLayerModel().to(device)
    fully_shard(model.layer1)
    fully_shard(model.layer2)
    fsdp_model = fully_shard(model)

    plain = nn.Linear(4, 4)
    raised = False
    try:
        fsdp_model.set_modules_to_backward_prefetch([plain])
    except (AssertionError, Exception):
        raised = True
    assert raised, "Expected an exception for non-FSDPModule input"

    dist.barrier()


def _test_forward_with_gradient_sync_disabled(rank, world_size):
    """Verify a forward/backward pass works when gradient sync is disabled."""
    device = torch.device(f'npu:{rank}')
    model = SimpleModel().to(device)
    fsdp_model = fully_shard(model)

    fsdp_model.set_requires_gradient_sync(False)
    inp = torch.randn(4, 16, device=device)
    out = fsdp_model(inp)
    assert out.shape == (4, 16), f"Unexpected output shape: {out.shape}"
    out.sum().backward()

    # Re-enable sync
    fsdp_model.set_requires_gradient_sync(True)
    dist.barrier()


# ---------------------------------------------------------------------------
# TestCase
# ---------------------------------------------------------------------------

class TestFSDPModule(TestCase):
    """Test cases for FSDPModule and its reshard/prefetch/gradient-sync methods."""

    def setUp(self):
        super().setUp()
        self.device_name = torch._C._get_privateuse1_backend_name()
        self.assertEqual(
            self.device_name, 'npu',
            f"Expected device 'npu', got '{self.device_name}'"
        )

    def _run(self, fn, world_size=2):
        mp.spawn(
            _init_dist_process,
            args=(world_size, fn),
            nprocs=world_size,
            join=True,
        )

    @skipIfUnsupportMultiNPU(2)
    def test_fsdp_module_isinstance(self):
        """fully_shard() returns an FSDPModule instance."""
        self._run(_test_fsdp_module_isinstance)

    @skipIfUnsupportMultiNPU(2)
    def test_reshard(self):
        """reshard() can be called after unshard without error."""
        self._run(_test_reshard)

    @skipIfUnsupportMultiNPU(2)
    def test_set_requires_gradient_sync(self):
        """set_requires_gradient_sync with True/False and recurse True/False."""
        self._run(_test_set_requires_gradient_sync)

    @skipIfUnsupportMultiNPU(2)
    def test_set_requires_gradient_sync_nested(self):
        """set_requires_gradient_sync propagates to nested FSDP submodules."""
        self._run(_test_set_requires_gradient_sync_nested)

    @skipIfUnsupportMultiNPU(2)
    def test_set_modules_to_forward_prefetch(self):
        """set_modules_to_forward_prefetch with empty and singleton lists."""
        self._run(_test_set_modules_to_forward_prefetch)

    @skipIfUnsupportMultiNPU(2)
    def test_set_modules_to_forward_prefetch_invalid_type(self):
        """set_modules_to_forward_prefetch raises for non-FSDPModule input."""
        self._run(_test_set_modules_to_forward_prefetch_invalid)

    @skipIfUnsupportMultiNPU(2)
    def test_set_modules_to_backward_prefetch(self):
        """set_modules_to_backward_prefetch with empty and singleton lists."""
        self._run(_test_set_modules_to_backward_prefetch)

    @skipIfUnsupportMultiNPU(2)
    def test_set_modules_to_backward_prefetch_invalid_type(self):
        """set_modules_to_backward_prefetch raises for non-FSDPModule input."""
        self._run(_test_set_modules_to_backward_prefetch_invalid)

    @skipIfUnsupportMultiNPU(2)
    def test_forward_with_gradient_sync_disabled(self):
        """Forward/backward pass succeeds when gradient sync is disabled."""
        self._run(_test_forward_with_gradient_sync_disabled)


if __name__ == "__main__":
    run_tests()
