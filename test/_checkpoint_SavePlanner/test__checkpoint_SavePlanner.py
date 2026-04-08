# -*- coding: utf-8 -*-
"""
测试目的：验证 torch.distributed.checkpoint.SavePlanner 抽象基类及方法功能正确性
API 名称：torch.distributed.checkpoint.SavePlanner
torch.distributed.checkpoint.SavePlanner.finish_plan
torch.distributed.checkpoint.SavePlanner.set_up_planner

API 签名：
- SavePlanner (abstract base class)
- finish_plan(self, new_plan: SavePlan) -> SavePlan
- set_up_planner(self, state_dict: STATE_DICT_TYPE, storage_meta: StorageMeta | None = None, is_coordinator: bool = False) -> None

覆盖维度表：
| 覆盖维度         | 说明                                                         | 覆盖情况                                       |
|------------------|--------------------------------------------------------------|------------------------------------------------|
| 空/非空          | state_dict 为空或非空，storage_meta 为空或非空              | 已覆盖                                         |
| 枚举选项         | is_coordinator True/False                                   | 已覆盖                                         |
| 参数类型         | 验证参数接受正确类型                                         | 已覆盖                                         |
| 传参与不传参     | set_up_planner 的可选参数 storage_meta/is_coordinator        | 已覆盖                                         |
| 等价类/边界值    | 空 state_dict、单元素、多元素 state_dict                    | 已覆盖                                         |
| 正常传参场景     | Mock 实现调用各方法                                          | 已覆盖                                         |
| 异常传参场景     | 抽象基类不能直接实例化                                       | 已覆盖                                         |

未覆盖项及原因：
- 无

注意：本测试仅验证功能正确性（调用不报错、输出 shape/dtype/类型符合预期），
     不做精度和数值正确性校验。
"""

import abc
import io
from typing import Any

import torch
from torch.distributed.checkpoint.planner import SavePlanner, SavePlan, WriteItem, WriteItemType
from torch.distributed.checkpoint.metadata import Metadata, StorageMeta, MetadataIndex


try:
    from torch_npu.testing.testcase import TestCase, run_tests
except ImportError:
    import sys
    from unittest import TestCase

    def run_tests():
        import unittest
        unittest.main(argv=sys.argv)


class MockSavePlanner(SavePlanner):
    """Mock implementation of SavePlanner for testing."""

    def __init__(self):
        self.state_dict = None
        self.storage_meta = None
        self.is_coordinator = False
        self.setup_called = False
        self.finish_plan_called = False

    def set_up_planner(
        self,
        state_dict: dict,
        storage_meta: StorageMeta | None = None,
        is_coordinator: bool = False,
    ) -> None:
        self.state_dict = state_dict
        self.storage_meta = storage_meta
        self.is_coordinator = is_coordinator
        self.setup_called = True

    def create_local_plan(self) -> SavePlan:
        return SavePlan(items=[])

    def create_global_plan(
        self, all_plans: list[SavePlan]
    ) -> tuple[list[SavePlan], Metadata]:
        metadata = Metadata(state_dict_metadata={})
        return all_plans, metadata

    def finish_plan(self, new_plan: SavePlan) -> SavePlan:
        self.finish_plan_called = True
        return new_plan

    def resolve_data(self, write_item: WriteItem) -> torch.Tensor | io.BytesIO:
        if write_item.type == WriteItemType.BYTE_IO:
            return io.BytesIO(b"mock data")
        return torch.zeros(10)


class TestSavePlanner(TestCase):
    """Test cases for SavePlanner abstract base class."""

    def setUp(self):
        super().setUp()
        self.device_name = torch._C._get_privateuse1_backend_name()
        self.assertEqual(self.device_name, 'npu', f"Expected device 'npu', got '{self.device_name}'")

    def test_save_planner_is_abstract(self):
        """Test that SavePlanner cannot be instantiated directly."""
        with self.assertRaises(TypeError):
            SavePlanner()

    def test_save_planner_subclass_abstract_methods(self):
        """Test that subclasses must implement abstract methods."""
        class IncompletePlanner(SavePlanner):
            pass

        with self.assertRaises(TypeError):
            IncompletePlanner()

    def test_mock_save_planner_setup_with_all_params(self):
        """Test set_up_planner with all parameters provided."""
        planner = MockSavePlanner()
        state_dict = {"param1": torch.ones(10), "param2": torch.zeros(5)}
        storage_meta = StorageMeta(checkpoint_id="test_ckpt")

        planner.set_up_planner(state_dict, storage_meta, is_coordinator=True)

        self.assertTrue(planner.setup_called)
        self.assertEqual(planner.state_dict, state_dict)
        self.assertEqual(planner.storage_meta, storage_meta)
        self.assertTrue(planner.is_coordinator)

    def test_mock_save_planner_setup_with_defaults(self):
        """Test set_up_planner with default parameters."""
        planner = MockSavePlanner()
        state_dict = {"param1": torch.ones(10)}

        planner.set_up_planner(state_dict)

        self.assertTrue(planner.setup_called)
        self.assertEqual(planner.state_dict, state_dict)
        self.assertIsNone(planner.storage_meta)
        self.assertFalse(planner.is_coordinator)

    def test_mock_save_planner_setup_empty_state_dict(self):
        """Test set_up_planner with empty state_dict."""
        planner = MockSavePlanner()
        state_dict = {}

        planner.set_up_planner(state_dict, is_coordinator=False)

        self.assertTrue(planner.setup_called)
        self.assertEqual(planner.state_dict, {})
        self.assertFalse(planner.is_coordinator)

    def test_mock_save_planner_finish_plan(self):
        """Test finish_plan method."""
        planner = MockSavePlanner()
        write_item = WriteItem(
            index=MetadataIndex(fqn="test"),
            type=WriteItemType.TENSOR
        )
        plan = SavePlan(items=[write_item])

        result = planner.finish_plan(plan)

        self.assertTrue(planner.finish_plan_called)
        self.assertEqual(result, plan)

    def test_mock_save_planner_finish_plan_empty(self):
        """Test finish_plan with empty plan."""
        planner = MockSavePlanner()
        plan = SavePlan(items=[])

        result = planner.finish_plan(plan)

        self.assertTrue(planner.finish_plan_called)
        self.assertEqual(result.items, [])

    def test_save_planner_isinstance_check(self):
        """Test isinstance check for SavePlanner."""
        planner = MockSavePlanner()
        self.assertIsInstance(planner, SavePlanner)

    def test_save_planner_abstract_method_resolution(self):
        """Test that all abstract methods are implemented in MockSavePlanner."""
        planner = MockSavePlanner()

        # Should not raise TypeError
        self.assertIsNotNone(planner)

        # Verify all abstract methods are callable
        state_dict = {"param": torch.ones(5)}
        planner.set_up_planner(state_dict)

        local_plan = planner.create_local_plan()
        self.assertIsInstance(local_plan, SavePlan)

        all_plans = [local_plan]
        global_plans, metadata = planner.create_global_plan(all_plans)
        self.assertIsInstance(global_plans, list)
        self.assertIsInstance(metadata, Metadata)

        final_plan = planner.finish_plan(local_plan)
        self.assertIsInstance(final_plan, SavePlan)


if __name__ == "__main__":
    run_tests()
