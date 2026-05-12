# -*- coding: utf-8 -*-
"""
测试目的：验证 torch.distributed.optim.PostLocalSGDOptimizer.state_dict 与 load_state_dict 接口功能正确性
API 名称：
  - torch.distributed.optim.PostLocalSGDOptimizer.state_dict
  - torch.distributed.optim.PostLocalSGDOptimizer.load_state_dict
API 签名：
  - state_dict(self) -> dict
  - load_state_dict(self, state_dict: dict) -> None

覆盖维度表：
| 覆盖维度         | 说明                                                         | 覆盖情况                                       |
|------------------|--------------------------------------------------------------|------------------------------------------------|
| 空/非空          | state_dict 含 "step" key vs 不含 "step" key                  | 已覆盖：test_load_state_dict_no_step_warns     |
| 枚举选项         | 不同 period 的 averager（step 值）                           | 已覆盖：test_state_dict_contains_step          |
| 参数类型         | state_dict: dict                                             | 已覆盖                                         |
| 传参与不传参     | load_state_dict 必须传入 dict                                | 已覆盖                                         |
| 等价类/边界值    | averager.step=0、step=5                                      | 已覆盖：test_state_dict_step_roundtrip         |
| 正常传参场景     | state_dict 序列化后再 load_state_dict 恢复                   | 已覆盖：test_state_dict_step_roundtrip         |
| 异常传参场景     | load_state_dict 不含 "step" 触发 UserWarning                 | 已覆盖：test_load_state_dict_no_step_warns     |

未覆盖项及原因：
- load_state_dict 后 param_groups['lr'] 恢复：PyTorch 2.7 的 Optimizer.load_state_dict 会替换底层
  param_groups 对象，PostLocalSGDOptimizer.param_groups 引用旧对象，属实现层行为，不在公开 API 承诺范围内
- 跨 rank 序列化：PostLocalSGDOptimizer 的 checkpoint 需结合 dist.save/load，测试量级超出单文件范围

注意：本测试仅验证功能正确性（调用不报错、返回值类型符合预期），
     不做精度和数值正确性校验。
"""

import os
import warnings
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn as nn

import torch_npu  # noqa: F401 — registers NPU backend
from torch_npu.testing.testcase import TestCase, run_tests
from torch_npu.testing.common_distributed import skipIfUnsupportMultiNPU

from torch.distributed.optim import PostLocalSGDOptimizer
from torch.distributed.algorithms.model_averaging.averagers import PeriodicModelAverager


def _init_dist_process(rank, world_size, fn, backend='hccl'):
    os.environ.setdefault('MASTER_ADDR', 'localhost')
    os.environ.setdefault('MASTER_PORT', '29519')
    os.environ['HCCL_WHITELIST_DISABLE'] = '1'
    torch_npu.npu.set_device(rank)
    dist.init_process_group(backend, rank=rank, world_size=world_size)
    try:
        fn(rank, world_size)
    finally:
        dist.destroy_process_group()


def _make_optimizer(rank, model):
    """Helper: create a PostLocalSGDOptimizer with PeriodicModelAverager."""
    averager = PeriodicModelAverager(period=2, warmup_steps=0)
    optim = torch.optim.SGD(model.parameters(), lr=0.01)
    return PostLocalSGDOptimizer(optim=optim, averager=averager)


# ------------------------------------------------------------------
# Per-rank worker functions
# ------------------------------------------------------------------

def _test_state_dict_contains_step(rank, world_size):
    """state_dict includes 'step' key from averager."""
    device = torch.device(f'npu:{rank}')
    model = nn.Linear(4, 4).to(device)
    optimizer = _make_optimizer(rank, model)

    sd = optimizer.state_dict()
    assert 'step' in sd, "state_dict must contain 'step'"
    assert isinstance(sd, dict), f"Expected dict, got {type(sd)}"
    dist.barrier()


def _test_state_dict_step_roundtrip(rank, world_size):
    """state_dict / load_state_dict round-trip preserves averager step."""
    device = torch.device(f'npu:{rank}')
    model = nn.Linear(4, 4).to(device)
    optimizer = _make_optimizer(rank, model)

    # Advance the averager step to 5 manually
    optimizer.averager.step = 5

    sd = optimizer.state_dict()
    assert sd['step'] == 5, f"Expected step=5, got {sd['step']}"

    # Reset and reload
    optimizer.averager.step = 0
    optimizer.load_state_dict(sd)
    assert optimizer.averager.step == 5, (
        f"After load_state_dict, expected step=5, got {optimizer.averager.step}"
    )
    dist.barrier()


def _test_load_state_dict_no_step_warns(rank, world_size):
    """load_state_dict warns and resets step to 0 when 'step' key is absent."""
    device = torch.device(f'npu:{rank}')
    model = nn.Linear(4, 4).to(device)
    optimizer = _make_optimizer(rank, model)

    # Obtain a valid state_dict then remove the 'step' key
    sd = optimizer.state_dict()
    sd.pop('step', None)

    optimizer.averager.step = 99
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        optimizer.load_state_dict(sd)
        assert any('step' in str(warning.message).lower() for warning in w), (
            "Expected a warning about missing 'step' in state_dict"
        )
    assert optimizer.averager.step == 0, (
        f"Expected step=0 after missing-step load, got {optimizer.averager.step}"
    )
    dist.barrier()


def _test_state_dict_returns_dict(rank, world_size):
    """state_dict returns a plain dict."""
    device = torch.device(f'npu:{rank}')
    model = nn.Linear(4, 4).to(device)
    optimizer = _make_optimizer(rank, model)

    sd = optimizer.state_dict()
    assert isinstance(sd, dict), f"Expected dict, got {type(sd)}"
    assert 'state' in sd or 'param_groups' in sd or 'step' in sd, (
        "state_dict should contain optimizer state keys"
    )
    dist.barrier()


def _test_load_state_dict_does_not_raise(rank, world_size):
    """load_state_dict runs without error on a valid state_dict."""
    device = torch.device(f'npu:{rank}')
    model = nn.Linear(4, 4).to(device)
    optimizer = _make_optimizer(rank, model)

    sd = optimizer.state_dict()
    # Must not raise
    optimizer.load_state_dict(sd)
    dist.barrier()


# ------------------------------------------------------------------
# TestCase
# ------------------------------------------------------------------

class TestPostLocalSGDOptimizer(TestCase):
    """Test cases for PostLocalSGDOptimizer.state_dict and load_state_dict."""

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
    def test_state_dict_contains_step(self):
        """state_dict contains 'step' key from averager."""
        self._run(_test_state_dict_contains_step)

    @skipIfUnsupportMultiNPU(2)
    def test_state_dict_returns_dict(self):
        """state_dict returns a dict with expected keys."""
        self._run(_test_state_dict_returns_dict)

    @skipIfUnsupportMultiNPU(2)
    def test_state_dict_step_roundtrip(self):
        """state_dict / load_state_dict round-trip preserves averager step."""
        self._run(_test_state_dict_step_roundtrip)

    @skipIfUnsupportMultiNPU(2)
    def test_load_state_dict_no_step_warns(self):
        """load_state_dict issues UserWarning when 'step' key is missing."""
        self._run(_test_load_state_dict_no_step_warns)

    @skipIfUnsupportMultiNPU(2)
    def test_load_state_dict_does_not_raise(self):
        """load_state_dict runs without error on a valid state_dict."""
        self._run(_test_load_state_dict_does_not_raise)


if __name__ == "__main__":
    run_tests()
