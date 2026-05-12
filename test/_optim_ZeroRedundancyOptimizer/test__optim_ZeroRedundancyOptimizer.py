# -*- coding: utf-8 -*-
"""
测试目的：验证 torch.distributed.optim.ZeroRedundancyOptimizer 各接口功能正确性
API 名称：
  - torch.distributed.optim.ZeroRedundancyOptimizer.add_param_group
  - torch.distributed.optim.ZeroRedundancyOptimizer.join_device
  - torch.distributed.optim.ZeroRedundancyOptimizer.join_hook
  - torch.distributed.optim.ZeroRedundancyOptimizer.join_process_group
  - torch.distributed.optim.ZeroRedundancyOptimizer.load_state_dict
  - torch.distributed.optim.ZeroRedundancyOptimizer.state_dict
API 签名：
  - add_param_group(self, param_group: dict) -> None
  - join_device  -> torch.device  (property)
  - join_hook(self, **kwargs) -> JoinHook
  - join_process_group  -> ProcessGroup  (property)
  - load_state_dict(self, state_dict: dict) -> None
  - state_dict(self) -> dict  (requires prior consolidate_state_dict call)

覆盖维度表：
| 覆盖维度         | 说明                                                         | 覆盖情况                                       |
|------------------|--------------------------------------------------------------|------------------------------------------------|
| 空/非空          | add_param_group 传新参数组 vs 已有参数组                     | 已覆盖：test_add_param_group                   |
| 枚举选项         | join_device 对应 NPU 设备类型                                | 已覆盖：test_join_device                       |
| 参数类型         | param_group: dict；state_dict: dict                          | 已覆盖                                         |
| 传参与不传参     | join_hook 无额外 kwargs                                      | 已覆盖：test_join_hook                         |
| 等价类/边界值    | world_size=2；lr 不同值的 add_param_group                    | 已覆盖                                         |
| 正常传参场景     | consolidate_state_dict + state_dict + load_state_dict 完整流程 | 已覆盖：test_state_dict_roundtrip            |
| 异常传参场景     | state_dict 无 consolidate 前调用抛 RuntimeError              | 已覆盖：test_state_dict_without_consolidate_raises |

未覆盖项及原因：
- overlap_with_ddp=True 场景：需要 DDP 集成，超出单文件范围

注意：本测试仅验证功能正确性（调用不报错、返回值类型符合预期），
     不做精度和数值正确性校验。
"""

import os
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn as nn

import torch_npu  # noqa: F401 — registers NPU backend
from torch_npu.testing.testcase import TestCase, run_tests
from torch_npu.testing.common_distributed import skipIfUnsupportMultiNPU

from torch.distributed.optim import ZeroRedundancyOptimizer


def _init_dist_process(rank, world_size, fn, backend='hccl'):
    os.environ.setdefault('MASTER_ADDR', 'localhost')
    os.environ.setdefault('MASTER_PORT', '29520')
    os.environ['HCCL_WHITELIST_DISABLE'] = '1'
    torch_npu.npu.set_device(rank)
    dist.init_process_group(backend, rank=rank, world_size=world_size)
    try:
        fn(rank, world_size)
    finally:
        dist.destroy_process_group()


def _make_zero(rank, params, lr=0.01):
    return ZeroRedundancyOptimizer(params, optimizer_class=torch.optim.SGD, lr=lr)


# ------------------------------------------------------------------
# Per-rank worker functions
# ------------------------------------------------------------------

def _test_add_param_group(rank, world_size):
    """add_param_group adds a new parameter group to ZeRO."""
    device = torch.device(f'npu:{rank}')
    model = nn.Linear(8, 4).to(device)
    zero = _make_zero(rank, list(model.parameters()))

    # Number of param_groups before add
    n_before = len(zero.param_groups)

    # Add a new param group with different lr
    extra_fc = nn.Linear(4, 2).to(device)
    zero.add_param_group({'params': list(extra_fc.parameters()), 'lr': 0.001})

    assert len(zero.param_groups) == n_before + 1, (
        f"Expected {n_before + 1} param groups, got {len(zero.param_groups)}"
    )
    dist.barrier()


def _test_add_param_group_lr_preserved(rank, world_size):
    """add_param_group preserves the lr of the new group."""
    device = torch.device(f'npu:{rank}')
    model = nn.Linear(8, 4).to(device)
    zero = _make_zero(rank, list(model.parameters()))

    extra_fc = nn.Linear(4, 2).to(device)
    zero.add_param_group({'params': list(extra_fc.parameters()), 'lr': 0.005})

    added = zero.param_groups[-1]
    assert added['lr'] == 0.005, f"Expected lr=0.005, got {added['lr']}"
    dist.barrier()


def _test_join_device(rank, world_size):
    """join_device returns a torch.device on NPU."""
    device = torch.device(f'npu:{rank}')
    model = nn.Linear(8, 4).to(device)
    zero = _make_zero(rank, list(model.parameters()))

    jd = zero.join_device
    assert isinstance(jd, torch.device), f"Expected torch.device, got {type(jd)}"
    assert jd.type == 'npu', f"Expected npu device type, got {jd.type}"
    dist.barrier()


def _test_join_hook(rank, world_size):
    """join_hook returns a JoinHook instance."""
    device = torch.device(f'npu:{rank}')
    model = nn.Linear(8, 4).to(device)
    zero = _make_zero(rank, list(model.parameters()))

    hook = zero.join_hook()
    assert hook is not None, "join_hook should not return None"
    # JoinHook must have main_hook and post_hook methods
    assert hasattr(hook, 'main_hook'), "JoinHook must have main_hook"
    assert hasattr(hook, 'post_hook'), "JoinHook must have post_hook"
    dist.barrier()


def _test_join_process_group(rank, world_size):
    """join_process_group returns the process group used by ZeRO."""
    device = torch.device(f'npu:{rank}')
    model = nn.Linear(8, 4).to(device)
    zero = _make_zero(rank, list(model.parameters()))

    pg = zero.join_process_group
    assert pg is not None, "join_process_group should not be None"
    dist.barrier()


def _test_state_dict_without_consolidate_raises(rank, world_size):
    """state_dict raises RuntimeError if consolidate_state_dict not called first."""
    device = torch.device(f'npu:{rank}')
    model = nn.Linear(8, 4).to(device)
    zero = _make_zero(rank, list(model.parameters()))

    raised = False
    try:
        zero.state_dict()
    except RuntimeError:
        raised = True
    assert raised, "Expected RuntimeError when consolidate_state_dict not called"
    dist.barrier()


def _test_state_dict_roundtrip(rank, world_size):
    """consolidate_state_dict → state_dict → load_state_dict round-trip."""
    device = torch.device(f'npu:{rank}')
    model = nn.Linear(8, 4).to(device)
    zero = _make_zero(rank, list(model.parameters()))

    # Perform one optimizer step to populate state
    inp = torch.randn(2, 8, device=device)
    loss = model(inp).sum()
    loss.backward()
    zero.step()
    zero.zero_grad()

    # Consolidate to rank 0 and get state_dict
    zero.consolidate_state_dict(to=0)
    if rank == 0:
        sd = zero.state_dict()
        assert isinstance(sd, dict), f"Expected dict, got {type(sd)}"
        assert 'state' in sd, "state_dict should contain 'state'"
        assert 'param_groups' in sd, "state_dict should contain 'param_groups'"

        # load_state_dict should succeed
        zero.load_state_dict(sd)

    dist.barrier()


def _test_load_state_dict_restores_lr(rank, world_size):
    """load_state_dict restores param group lr."""
    device = torch.device(f'npu:{rank}')
    model = nn.Linear(8, 4).to(device)
    zero = _make_zero(rank, list(model.parameters()), lr=0.01)

    inp = torch.randn(2, 8, device=device)
    loss = model(inp).sum()
    loss.backward()
    zero.step()
    zero.zero_grad()

    original_lr = zero.param_groups[0]['lr']

    zero.consolidate_state_dict(to=0)
    if rank == 0:
        sd = zero.state_dict()
        # Mutate lr, then restore via load_state_dict
        zero.param_groups[0]['lr'] = 0.99
        zero.load_state_dict(sd)
        assert zero.param_groups[0]['lr'] == original_lr, (
            f"Expected lr={original_lr}, got {zero.param_groups[0]['lr']}"
        )

    dist.barrier()


# ------------------------------------------------------------------
# TestCase
# ------------------------------------------------------------------

class TestZeroRedundancyOptimizer(TestCase):
    """Test cases for ZeroRedundancyOptimizer methods."""

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
    def test_add_param_group(self):
        """add_param_group increases param_groups count."""
        self._run(_test_add_param_group)

    @skipIfUnsupportMultiNPU(2)
    def test_add_param_group_lr_preserved(self):
        """add_param_group preserves the lr of the new group."""
        self._run(_test_add_param_group_lr_preserved)

    @skipIfUnsupportMultiNPU(2)
    def test_join_device(self):
        """join_device returns torch.device on npu."""
        self._run(_test_join_device)

    @skipIfUnsupportMultiNPU(2)
    def test_join_hook(self):
        """join_hook returns a valid JoinHook with main_hook and post_hook."""
        self._run(_test_join_hook)

    @skipIfUnsupportMultiNPU(2)
    def test_join_process_group(self):
        """join_process_group returns the process group."""
        self._run(_test_join_process_group)

    @skipIfUnsupportMultiNPU(2)
    def test_state_dict_without_consolidate_raises(self):
        """state_dict raises RuntimeError without prior consolidate_state_dict."""
        self._run(_test_state_dict_without_consolidate_raises)

    @skipIfUnsupportMultiNPU(2)
    def test_state_dict_roundtrip(self):
        """consolidate_state_dict + state_dict + load_state_dict round-trip."""
        self._run(_test_state_dict_roundtrip)

    @skipIfUnsupportMultiNPU(2)
    def test_load_state_dict_restores_lr(self):
        """load_state_dict restores param group lr after mutation."""
        self._run(_test_load_state_dict_restores_lr)


if __name__ == "__main__":
    run_tests()
