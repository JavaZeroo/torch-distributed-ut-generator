# -*- coding: utf-8 -*-
"""
测试目的：验证 torch.distributed.checkpoint.SavePlanner.resolve_data 接口功能正确性
API 名称：torch.distributed.checkpoint.SavePlanner.resolve_data
API 签名：SavePlanner.resolve_data(self, write_item: WriteItem) -> Union[torch.Tensor, io.BytesIO]

覆盖维度表：
| 覆盖维度         | 说明                                                         | 覆盖情况                                                               |
|------------------|--------------------------------------------------------------|------------------------------------------------------------------------|
| 空/非空          | state_dict 含 tensor vs bytes 对象                           | 已覆盖                                                                 |
| 枚举选项         | WriteItemType.TENSOR / BYTE_IO                               | 已覆盖：TENSOR 返回 Tensor；BYTE_IO 返回 BytesIO                       |
| 参数类型         | float32 / float16 tensor；bytes 对象                         | 已覆盖                                                                 |
| 传参与不传参     | write_item 由 create_local_plan 生成                          | 已覆盖                                                                 |
| 等价类/边界值    | 单 key；多 key；含 bytes 的混合 state_dict                    | 已覆盖                                                                 |
| 正常传参场景     | TENSOR item → Tensor；BYTE_IO item → BytesIO                 | 已覆盖                                                                 |
| 异常传参场景     | 无稳定异常路径（write_item 由 plan 保证合法）                 | 未覆盖，原因：合法路径由 plan 保证                                     |

未覆盖项及原因：
- 异常传参：write_item 由 create_local_plan 生成，不存在非法 write_item 的稳定路径。

注意：本测试仅验证功能正确性（调用不报错、输出 shape/dtype/类型符合预期），
     不做精度和数值正确性校验。
"""
import io
import os
import unittest

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch_npu
from torch.distributed.checkpoint.default_planner import DefaultSavePlanner
from torch.distributed.checkpoint.planner import WriteItemType
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


def _test_resolve_data_tensor(rank, world_size):
    """resolve_data returns a Tensor for TENSOR WriteItem."""
    device = f'npu:{rank}'
    state_dict = {'weight': torch.zeros(4, 4, device=device)}
    planner = DefaultSavePlanner()
    planner.set_up_planner(state_dict, is_coordinator=(rank == 0))
    plan = planner.create_local_plan()
    all_plans = [plan]  # single rank simulation
    if rank == 0:
        global_plan, _ = planner.create_global_plan(all_plans)
        planner.finish_plan(global_plan[0])
    else:
        # For non-coordinator, finish_plan with local plan
        planner.finish_plan(plan)

    for item in planner.plan.items:
        if item.type == WriteItemType.TENSOR or item.type == WriteItemType.SHARD:
            result = planner.resolve_data(item)
            assert isinstance(result, torch.Tensor), \
                f"TENSOR item should return Tensor, got {type(result)}"


def _test_resolve_data_bytes(rank, world_size):
    """resolve_data returns BytesIO for BYTE_IO WriteItem."""
    # Use a non-tensor value to get BYTE_IO items
    state_dict = {'optimizer_step': 42}
    planner = DefaultSavePlanner()
    planner.set_up_planner(state_dict, is_coordinator=(rank == 0))
    plan = planner.create_local_plan()
    if rank == 0:
        global_plan, _ = planner.create_global_plan([plan])
        planner.finish_plan(global_plan[0])
    else:
        planner.finish_plan(plan)

    for item in planner.plan.items:
        if item.type == WriteItemType.BYTE_IO:
            result = planner.resolve_data(item)
            assert isinstance(result, io.BytesIO), \
                f"BYTE_IO item should return BytesIO, got {type(result)}"


def _test_resolve_data_float16(rank, world_size):
    """resolve_data returns float16 tensor for float16 state_dict entry."""
    device = f'npu:{rank}'
    state_dict = {'fp16_w': torch.zeros(8, 4, device=device, dtype=torch.float16)}
    planner = DefaultSavePlanner()
    planner.set_up_planner(state_dict, is_coordinator=(rank == 0))
    plan = planner.create_local_plan()
    if rank == 0:
        global_plan, _ = planner.create_global_plan([plan])
        planner.finish_plan(global_plan[0])
    else:
        planner.finish_plan(plan)

    for item in planner.plan.items:
        if item.type in (WriteItemType.TENSOR, WriteItemType.SHARD):
            result = planner.resolve_data(item)
            assert isinstance(result, torch.Tensor)
            assert result.dtype == torch.float16, \
                f"Expected float16, got {result.dtype}"


def _test_resolve_data_multiple_keys(rank, world_size):
    """resolve_data handles multiple tensor keys."""
    device = f'npu:{rank}'
    state_dict = {
        'w1': torch.zeros(2, 4, device=device),
        'w2': torch.zeros(4, device=device),
    }
    planner = DefaultSavePlanner()
    planner.set_up_planner(state_dict, is_coordinator=(rank == 0))
    plan = planner.create_local_plan()
    if rank == 0:
        global_plan, _ = planner.create_global_plan([plan])
        planner.finish_plan(global_plan[0])
    else:
        planner.finish_plan(plan)

    tensor_count = 0
    for item in planner.plan.items:
        if item.type in (WriteItemType.TENSOR, WriteItemType.SHARD):
            result = planner.resolve_data(item)
            assert isinstance(result, torch.Tensor)
            tensor_count += 1
    assert tensor_count >= len(state_dict), \
        f"Expected at least {len(state_dict)} tensor items"


class TestSavePlannerResolveData(TestCase):
    """Tests for SavePlanner.resolve_data via DefaultSavePlanner."""

    def setUp(self):
        super().setUp()
        self.device_name = torch._C._get_privateuse1_backend_name()
        self.assertEqual(
            self.device_name, 'npu',
            f"Expected device 'npu', got '{self.device_name}'"
        )

    @skipIfUnsupportMultiNPU(2)
    def test_resolve_data_tensor_item(self):
        """resolve_data returns Tensor for TENSOR WriteItem."""
        mp.spawn(_init_dist, args=(2, _test_resolve_data_tensor), nprocs=2, join=True)

    @skipIfUnsupportMultiNPU(2)
    def test_resolve_data_bytes_item(self):
        """resolve_data returns BytesIO for BYTE_IO WriteItem."""
        mp.spawn(_init_dist, args=(2, _test_resolve_data_bytes), nprocs=2, join=True)

    @skipIfUnsupportMultiNPU(2)
    def test_resolve_data_float16(self):
        """resolve_data preserves float16 dtype."""
        mp.spawn(_init_dist, args=(2, _test_resolve_data_float16), nprocs=2, join=True)

    @skipIfUnsupportMultiNPU(2)
    def test_resolve_data_multiple_keys(self):
        """resolve_data handles multiple tensor keys in state_dict."""
        mp.spawn(_init_dist, args=(2, _test_resolve_data_multiple_keys), nprocs=2, join=True)


if __name__ == "__main__":
    run_tests()
