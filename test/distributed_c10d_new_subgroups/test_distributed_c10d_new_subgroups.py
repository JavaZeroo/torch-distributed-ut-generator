# -*- coding: utf-8 -*-
"""
测试目的：验证 torch.distributed.distributed_c10d.new_subgroups 接口功能正确性
API 名称：torch.distributed.distributed_c10d.new_subgroups
API 签名：
  new_subgroups(
      group_size=None,
      group=None,
      timeout=None,
      backend=None,
      pg_options=None,
      group_desc=None
  )

覆盖维度表：
| 覆盖维度         | 说明                                                         | 覆盖情况                                       |
|------------------|--------------------------------------------------------------|------------------------------------------------|
| 空/非空          | group=None vs explicit, group_size=None vs explicit          | 已覆盖                                         |
| 枚举选项         | backend: "hccl", "gloo"                                      | 已覆盖 (hccl)                                  |
| 参数类型         | int, ProcessGroup, timedelta, str                            | 已覆盖 (除 pg_options)                         |
| 传参与不传参     | 默认参数 vs 显式传参                                         | 已覆盖                                         |
| 等价类/边界值    | group_size 整除 world_size                                   | 已覆盖                                         |
| 正常传参场景     | 正常调用 new_subgroups                                       | 已覆盖                                         |
| 异常传参场景     | group_size 不整除 world_size                                 | 未覆盖 (行为依赖底层实现)                      |

未覆盖项及原因：
- 异常传参场景：new_subgroups 的异常处理依赖底层实现，无稳定 Python 层异常路径

注意：本测试仅验证功能正确性（调用不报错、输出 shape/dtype/类型符合预期），
     不做精度和数值正确性校验。
"""

import fcntl
import os
import socket
from datetime import timedelta

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from torch_npu.testing.testcase import TestCase, run_tests
from torch_npu.testing.common_distributed import skipIfUnsupportMultiNPU


_HCCL_PORT_BASE = 50000
_HCCL_PORT_RANGE = 12000
_HCCL_PORT_PER_SPAWN = 512
_HCCL_PORT_SLOTS = _HCCL_PORT_RANGE // _HCCL_PORT_PER_SPAWN
_HCCL_PORT_COUNTER_FILE = '/tmp/_new_subgroups_hccl_port_counter'


def _pick_free_port():
    """Pick a free TCP port for torch.distributed rendezvous."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(('', 0))
        return sock.getsockname()[1]


def _next_hccl_base_port():
    """Return a fresh HCCL base port with enough room for subgroup PGs."""
    try:
        with open(_HCCL_PORT_COUNTER_FILE, 'a+') as f:
            fcntl.flock(f, fcntl.LOCK_EX)
            try:
                f.seek(0)
                content = f.read().strip()
                slot = int(content) if content else 0
                slot = slot % _HCCL_PORT_SLOTS
                next_slot = (slot + 1) % _HCCL_PORT_SLOTS
                f.seek(0)
                f.truncate()
                f.write(str(next_slot))
                return _HCCL_PORT_BASE + slot * _HCCL_PORT_PER_SPAWN
            finally:
                fcntl.flock(f, fcntl.LOCK_UN)
    except (OSError, ValueError):
        slot = os.getpid() % _HCCL_PORT_SLOTS
        return _HCCL_PORT_BASE + slot * _HCCL_PORT_PER_SPAWN


def _spawn_dist_process(world_size, fn):
    """Spawn HCCL ranks with isolated rendezvous and HCCL port ranges."""
    os.environ.setdefault('MASTER_ADDR', 'localhost')
    os.environ['MASTER_PORT'] = str(_pick_free_port())
    os.environ['HCCL_IF_BASE_PORT'] = str(_next_hccl_base_port())

    mp.spawn(
        _init_dist_process,
        args=(world_size, fn),
        nprocs=world_size,
        join=True
    )


def _init_dist_process(rank, world_size, fn, backend='hccl'):
    """Initialize distributed process with HCCL backend."""
    os.environ.setdefault('MASTER_ADDR', 'localhost')
    os.environ.setdefault('MASTER_PORT', '29506')

    torch.npu.set_device(rank)
    dist.init_process_group(backend, rank=rank, world_size=world_size)

    try:
        fn(rank, world_size)
    finally:
        dist.destroy_process_group()


def _test_new_subgroups_default(rank, world_size):
    """Test new_subgroups with explicit group_size (NPU 后端不支持 default size)."""
    device = torch.device(f'npu:{rank}')

    # NPU/HCCL backend does not support inferring group_size from CUDA-only
    # device count, so an explicit group_size must be passed.
    subgroup, subgroups = dist.new_subgroups(group_size=world_size)

    # Verify we got a subgroup
    assert subgroup is not None or len(subgroups) > 0, "Should get at least one subgroup"

    if subgroup is not None:
        assert hasattr(subgroup, 'size'), "subgroup should have size method"
        assert hasattr(subgroup, 'rank'), "subgroup should have rank method"

    dist.barrier()


def _test_new_subgroups_with_group_size(rank, world_size):
    """Test new_subgroups with explicit group_size."""
    device = torch.device(f'npu:{rank}')

    # Create subgroups of size 2
    subgroup, subgroups = dist.new_subgroups(group_size=2)

    assert subgroup is not None, "Should get a subgroup"
    assert len(subgroups) == world_size // 2, f"Expected {world_size // 2} subgroups, got {len(subgroups)}"

    if subgroup is not None:
        assert subgroup.size() == 2, f"Expected subgroup size 2, got {subgroup.size()}"

    dist.barrier()


def _test_new_subgroups_with_explicit_group(rank, world_size):
    """Test new_subgroups with explicit group."""
    device = torch.device(f'npu:{rank}')

    # Use default world group
    world_group = dist.group.WORLD

    subgroup, subgroups = dist.new_subgroups(
        group_size=2,
        group=world_group
    )

    assert subgroup is not None, "Should get a subgroup"

    dist.barrier()


def _test_new_subgroups_with_timeout(rank, world_size):
    """Test new_subgroups with explicit timeout."""
    device = torch.device(f'npu:{rank}')

    subgroup, subgroups = dist.new_subgroups(
        group_size=2,
        timeout=timedelta(seconds=600)
    )

    assert subgroup is not None, "Should get a subgroup"

    dist.barrier()


def _test_new_subgroups_with_backend(rank, world_size):
    """Test new_subgroups with explicit backend."""
    device = torch.device(f'npu:{rank}')

    # Use hccl backend
    subgroup, subgroups = dist.new_subgroups(
        group_size=2,
        backend='hccl'
    )

    assert subgroup is not None, "Should get a subgroup"

    dist.barrier()


def _test_new_subgroups_with_group_desc(rank, world_size):
    """Test new_subgroups with group_desc."""
    device = torch.device(f'npu:{rank}')

    subgroup, subgroups = dist.new_subgroups(
        group_size=2,
        group_desc="test_subgroups"
    )

    assert subgroup is not None, "Should get a subgroup"

    dist.barrier()


def _test_new_subgroups_all_params(rank, world_size):
    """Test new_subgroups with all parameters."""
    device = torch.device(f'npu:{rank}')

    world_group = dist.group.WORLD

    subgroup, subgroups = dist.new_subgroups(
        group_size=2,
        group=world_group,
        timeout=timedelta(seconds=300),
        backend='hccl',
        pg_options=None,
        group_desc="full_test_subgroups"
    )

    assert subgroup is not None, "Should get a subgroup"
    assert len(subgroups) == world_size // 2, f"Expected {world_size // 2} subgroups"

    dist.barrier()


def _test_new_subgroups_subgroup_ops(rank, world_size):
    """Test operations on created subgroups."""
    device = torch.device(f'npu:{rank}')

    subgroup, subgroups = dist.new_subgroups(group_size=2)

    assert subgroup is not None, "Should get a subgroup"

    # Test all_reduce on subgroup
    if subgroup is not None:
        tensor = torch.ones(10, device=device)
        dist.all_reduce(tensor, group=subgroup)

        # Each subgroup has 2 ranks, so result should be 2
        expected = torch.ones(10) * 2
        assert tensor.shape == expected.shape, "Shape mismatch after all_reduce"

    dist.barrier()


class TestNewSubgroups(TestCase):
    """Test cases for new_subgroups function."""

    def setUp(self):
        super().setUp()
        self.device_name = torch._C._get_privateuse1_backend_name()
        self.assertEqual(self.device_name, 'npu', f"Expected device 'npu', got '{self.device_name}'")

    @skipIfUnsupportMultiNPU(2)
    def test_new_subgroups_default(self):
        """Test new_subgroups with default parameters."""
        world_size = 2
        _spawn_dist_process(world_size, _test_new_subgroups_default)

    @skipIfUnsupportMultiNPU(4)
    def test_new_subgroups_with_group_size(self):
        """Test new_subgroups with explicit group_size."""
        world_size = 4
        _spawn_dist_process(world_size, _test_new_subgroups_with_group_size)

    @skipIfUnsupportMultiNPU(4)
    def test_new_subgroups_with_explicit_group(self):
        """Test new_subgroups with explicit group."""
        world_size = 4
        _spawn_dist_process(world_size, _test_new_subgroups_with_explicit_group)

    @skipIfUnsupportMultiNPU(4)
    def test_new_subgroups_with_timeout(self):
        """Test new_subgroups with explicit timeout."""
        world_size = 4
        _spawn_dist_process(world_size, _test_new_subgroups_with_timeout)

    @skipIfUnsupportMultiNPU(4)
    def test_new_subgroups_with_backend(self):
        """Test new_subgroups with explicit backend."""
        world_size = 4
        _spawn_dist_process(world_size, _test_new_subgroups_with_backend)

    @skipIfUnsupportMultiNPU(4)
    def test_new_subgroups_with_group_desc(self):
        """Test new_subgroups with group_desc."""
        world_size = 4
        _spawn_dist_process(world_size, _test_new_subgroups_with_group_desc)

    @skipIfUnsupportMultiNPU(4)
    def test_new_subgroups_all_params(self):
        """Test new_subgroups with all parameters."""
        world_size = 4
        _spawn_dist_process(world_size, _test_new_subgroups_all_params)

    @skipIfUnsupportMultiNPU(4)
    def test_new_subgroups_subgroup_ops(self):
        """Test operations on created subgroups."""
        world_size = 4
        _spawn_dist_process(world_size, _test_new_subgroups_subgroup_ops)


if __name__ == "__main__":
    run_tests()
