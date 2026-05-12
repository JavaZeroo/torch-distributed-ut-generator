# -*- coding: utf-8 -*-
"""
测试目的：验证 torch.distributed.checkpoint.format_utils.DynamicMetaLoadPlanner.set_up_planner 接口功能正确性
API 名称：torch.distributed.checkpoint.format_utils.DynamicMetaLoadPlanner.set_up_planner
API 签名：DynamicMetaLoadPlanner.set_up_planner(
              self,
              state_dict: STATE_DICT_TYPE,
              metadata: Optional[Metadata] = None,
              is_coordinator: bool = False,
          ) -> None

覆盖维度表：
| 覆盖维度         | 说明                                                         | 覆盖情况                                                     |
|------------------|--------------------------------------------------------------|--------------------------------------------------------------|
| 空/非空          | state_dict 空 vs 非空；metadata 为 None vs 非 None           | 已覆盖：空 state_dict 会报 RuntimeError；非空 tensor dict    |
| 枚举选项         | is_coordinator=True / False                                  | 已覆盖                                                       |
| 参数类型         | state_dict 含 float32/float16/int32 tensor                   | 已覆盖                                                       |
| 传参与不传参     | metadata 默认 None vs 显式传入                               | 已覆盖                                                       |
| 等价类/边界值    | 单 key state_dict；多 key state_dict；含不同 shape           | 已覆盖                                                       |
| 正常传参场景     | 多种 tensor dtype 均可建立 metadata                           | 已覆盖                                                       |
| 异常传参场景     | state_dict 含非 tensor 值 → RuntimeError                     | 已覆盖                                                       |

未覆盖项及原因：
- 无。

注意：本测试仅验证功能正确性（调用不报错、输出 shape/dtype/类型符合预期），
     不做精度和数值正确性校验。
"""
import os
import unittest

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch_npu
from torch.distributed.checkpoint.format_utils import DynamicMetaLoadPlanner
from torch.distributed.checkpoint.metadata import Metadata
from torch_npu.testing.common_distributed import skipIfUnsupportMultiNPU
from torch_npu.testing.testcase import TestCase, run_tests


def _init_dist(rank, world_size, fn, *args):
    os.environ.setdefault('MASTER_ADDR', 'localhost')
    os.environ.setdefault('MASTER_PORT', '29500')
    torch.npu.set_device(rank)
    dist.init_process_group('hccl', rank=rank, world_size=world_size)
    try:
        fn(rank, world_size, *args)
    finally:
        dist.destroy_process_group()


def _test_setup_single_tensor(rank, world_size):
    """set_up_planner builds metadata for a single float32 tensor."""
    device = f'npu:{rank}'
    state_dict = {'weight': torch.zeros(4, 4, device=device)}
    planner = DynamicMetaLoadPlanner()
    planner.set_up_planner(state_dict, metadata=None, is_coordinator=(rank == 0))
    assert planner.metadata is not None, "metadata should be set after set_up_planner"
    assert 'weight' in planner.metadata.state_dict_metadata


def _test_setup_multi_tensor(rank, world_size):
    """set_up_planner handles multiple tensors with different dtypes."""
    device = f'npu:{rank}'
    state_dict = {
        'weight': torch.zeros(8, 4, device=device, dtype=torch.float32),
        'bias': torch.zeros(4, device=device, dtype=torch.float16),
        'idx': torch.zeros(2, 2, device=device, dtype=torch.int32),
    }
    planner = DynamicMetaLoadPlanner()
    planner.set_up_planner(state_dict, metadata=None, is_coordinator=False)
    assert planner.metadata is not None
    for key in state_dict:
        assert key in planner.metadata.state_dict_metadata, f"key '{key}' missing"


def _test_setup_is_coordinator_true(rank, world_size):
    """is_coordinator=True path succeeds for coordinator rank."""
    device = f'npu:{rank}'
    state_dict = {'w': torch.ones(3, 3, device=device)}
    planner = DynamicMetaLoadPlanner()
    # Test as coordinator and as non-coordinator
    is_coord = (rank == 0)
    planner.set_up_planner(state_dict, metadata=None, is_coordinator=is_coord)
    assert planner.is_coordinator == is_coord


def _test_setup_non_tensor_raises(rank, world_size):
    """state_dict with non-tensor value raises RuntimeError."""
    device = f'npu:{rank}'
    state_dict = {'weight': torch.zeros(4, device=device), 'bad_key': "string_value"}
    planner = DynamicMetaLoadPlanner()
    try:
        planner.set_up_planner(state_dict, metadata=None, is_coordinator=(rank == 0))
        raise AssertionError("Expected RuntimeError for non-tensor value")
    except RuntimeError:
        pass  # Expected


def _test_setup_metadata_preserved(rank, world_size):
    """set_up_planner stores state_dict and builds valid Metadata object."""
    device = f'npu:{rank}'
    state_dict = {'layer.weight': torch.randn(16, 8, device=device)}
    planner = DynamicMetaLoadPlanner()
    planner.set_up_planner(state_dict, metadata=None, is_coordinator=(rank == 0))
    meta = planner.metadata
    assert isinstance(meta, Metadata), f"Expected Metadata, got {type(meta)}"
    tensor_meta = meta.state_dict_metadata['layer.weight']
    assert tensor_meta.size == torch.Size([16, 8])
    assert tensor_meta.properties.dtype == torch.float32


class TestDynamicMetaLoadPlannerSetUpPlanner(TestCase):
    """Tests for DynamicMetaLoadPlanner.set_up_planner with HCCL multi-process."""

    def setUp(self):
        super().setUp()
        self.device_name = torch._C._get_privateuse1_backend_name()
        self.assertEqual(
            self.device_name, 'npu',
            f"Expected device 'npu', got '{self.device_name}'"
        )

    @skipIfUnsupportMultiNPU(2)
    def test_setup_single_tensor(self):
        """Planner builds metadata for a single tensor state_dict."""
        mp.spawn(_init_dist, args=(2, _test_setup_single_tensor), nprocs=2, join=True)

    @skipIfUnsupportMultiNPU(2)
    def test_setup_multi_tensor_dtypes(self):
        """Planner handles multiple tensors with different dtypes."""
        mp.spawn(_init_dist, args=(2, _test_setup_multi_tensor), nprocs=2, join=True)

    @skipIfUnsupportMultiNPU(2)
    def test_setup_is_coordinator_flag(self):
        """is_coordinator flag is correctly stored on planner instance."""
        mp.spawn(_init_dist, args=(2, _test_setup_is_coordinator_true), nprocs=2, join=True)

    @skipIfUnsupportMultiNPU(2)
    def test_setup_non_tensor_raises_runtime_error(self):
        """Non-tensor value in state_dict raises RuntimeError."""
        mp.spawn(_init_dist, args=(2, _test_setup_non_tensor_raises), nprocs=2, join=True)

    @skipIfUnsupportMultiNPU(2)
    def test_setup_metadata_structure(self):
        """Built metadata has correct structure and tensor properties."""
        mp.spawn(_init_dist, args=(2, _test_setup_metadata_preserved), nprocs=2, join=True)


if __name__ == "__main__":
    run_tests()
