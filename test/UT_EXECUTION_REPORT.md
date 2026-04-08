# 分布式类 API 功能测试报告

## 概述

本报告记录为 torch.distributed 相关 API 生成的功能测试用例。

## 测试 API 列表

本次共为 **18个分布式类 API** 生成功能测试用例：

### FSDPModule 方法（7个）
| API 名称 | 路径 | 测试方法数 |
|---------|------|-----------|
| torch.distributed.fsdp.FSDPModule.set_all_reduce_hook | test/fsdp_FSDPModule_methods/ | 1 |
| torch.distributed.fsdp.FSDPModule.set_is_last_backward | test/fsdp_FSDPModule_methods/ | 1 |
| torch.distributed.fsdp.FSDPModule.set_post_optim_event | test/fsdp_FSDPModule_methods/ | 1 |
| torch.distributed.fsdp.FSDPModule.set_reduce_scatter_divide_factor | test/fsdp_FSDPModule_methods/ | 1 |
| torch.distributed.fsdp.FSDPModule.set_requires_all_reduce | test/fsdp_FSDPModule_methods/ | 1 |
| torch.distributed.fsdp.FSDPModule.set_reshard_after_backward | test/fsdp_FSDPModule_methods/ | 1 |
| torch.distributed.fsdp.FSDPModule.set_unshard_in_backward | test/fsdp_FSDPModule_methods/ | 1 |

### FullyShardedDataParallel 方法（10个）
| API 名称 | 路径 | 测试方法数 |
|---------|------|-----------|
| torch.distributed.fsdp.FullyShardedDataParallel.apply | test/fsdp_FullyShardedDataParallel_methods/ | 1 |
| torch.distributed.fsdp.FullyShardedDataParallel.check_is_root | test/fsdp_FullyShardedDataParallel_methods/ | 1 |
| torch.distributed.fsdp.FullyShardedDataParallel.flatten_sharded_optim_state_dict | test/fsdp_FullyShardedDataParallel_methods/ | 1 |
| torch.distributed.fsdp.FullyShardedDataParallel.forward | test/fsdp_FullyShardedDataParallel_methods/ | 1 |
| torch.distributed.fsdp.FullyShardedDataParallel.module | test/fsdp_FullyShardedDataParallel_methods/ | 1 |
| torch.distributed.fsdp.FullyShardedDataParallel.named_buffers | test/fsdp_FullyShardedDataParallel_methods/ | 1 |
| torch.distributed.fsdp.FullyShardedDataParallel.named_parameters | test/fsdp_FullyShardedDataParallel_methods/ | 1 |
| torch.distributed.fsdp.FullyShardedDataParallel.no_sync | test/fsdp_FullyShardedDataParallel_methods/ | 1 |
| torch.distributed.fsdp.FullyShardedDataParallel.register_comm_hook | test/fsdp_FullyShardedDataParallel_methods/ | 1 |
| torch.distributed.fsdp.FullyShardedDataParallel.sharded_optim_state_dict | test/fsdp_FullyShardedDataParallel_methods/ | 1 |

### 其他分布式类 API（4个）
| API 名称 | 路径 | 测试方法数 |
|---------|------|-----------|
| torch.distributed.tensor.parallel.PrepareModuleOutput | test/tensor_parallel_PrepareModuleOutput/ | 4 |
| torch.distributed.tensor.placement_types.Partial | test/tensor_placement_types_Partial/ | 8 |
| torch.distributed.distributed_c10d.split_group | test/distributed_c10d_split_group/ | 6 |
| torch.distributed.distributed_c10d.new_subgroups | test/distributed_c10d_new_subgroups/ | 8 |

## 测试文件详情

### 1. test/fsdp_FSDPModule_methods/test_fsdp_FSDPModule_methods.py

**测试 API**: 
- FSDPModule.set_all_reduce_hook
- FSDPModule.set_is_last_backward
- FSDPModule.set_post_optim_event
- FSDPModule.set_reduce_scatter_divide_factor
- FSDPModule.set_requires_all_reduce
- FSDPModule.set_reshard_after_backward
- FSDPModule.set_unshard_in_backward

**测试方法**:
| 测试方法 | 卡数要求 | 说明 |
|---------|---------|------|
| test_set_is_last_backward | 2 | 测试 is_last_backward 设置为 True/False |
| test_set_requires_all_reduce | 2 | 测试 requires_all_reduce 及 recurse 参数 |
| test_set_reshard_after_backward | 2 | 测试 reshard_after_backward 及 recurse 参数 |
| test_set_unshard_in_backward | 2 | 测试 unshard_in_backward 设置 |
| test_set_reduce_scatter_divide_factor | 2 | 测试 factor 参数 (0.5, 1.0, 2.0) |
| test_set_post_optim_event | 2 | 测试 post_optim_event 设置 |
| test_set_all_reduce_hook | 2 | 测试 all_reduce hook 及 stream 参数 |

### 2. test/fsdp_FullyShardedDataParallel_methods/test_fsdp_FullyShardedDataParallel_methods.py

**测试 API**:
- FullyShardedDataParallel.apply
- FullyShardedDataParallel.check_is_root
- FullyShardedDataParallel.flatten_sharded_optim_state_dict
- FullyShardedDataParallel.forward
- FullyShardedDataParallel.module
- FullyShardedDataParallel.named_buffers
- FullyShardedDataParallel.named_parameters
- FullyShardedDataParallel.no_sync
- FullyShardedDataParallel.register_comm_hook
- FullyShardedDataParallel.sharded_optim_state_dict

**测试方法**:
| 测试方法 | 卡数要求 | 说明 |
|---------|---------|------|
| test_apply | 2 | 测试 apply 方法调用及返回值 |
| test_check_is_root | 2 | 测试 check_is_root 返回值类型 |
| test_forward | 2 | 测试 forward 输出 shape 和 device |
| test_module_property | 2 | 测试 module property 返回类型 |
| test_named_buffers | 2 | 测试 named_buffers 迭代器 |
| test_named_parameters | 2 | 测试 named_parameters 迭代器 |
| test_no_sync | 2 | 测试 no_sync 上下文管理器 |
| test_register_comm_hook | 2 | 测试 register_comm_hook 方法 |
| test_sharded_optim_state_dict | 2 | 测试 sharded_optim_state_dict 静态方法 |
| test_flatten_sharded_optim_state_dict | 2 | 测试 flatten_sharded_optim_state_dict 静态方法 |

### 3. test/tensor_parallel_PrepareModuleOutput/test_tensor_parallel_PrepareModuleOutput.py

**测试 API**: torch.distributed.tensor.parallel.PrepareModuleOutput

**测试方法**:
| 测试方法 | 卡数要求 | 说明 |
|---------|---------|------|
| test_single_placement | 2 | 测试单 Placement 构造 |
| test_tuple_placement | 2 | 测试 tuple of Placements 构造 |
| test_default_use_local_output | 2 | 测试默认 use_local_output 值 |
| test_mismatched_lengths | 2 | 测试长度不匹配异常场景 |

### 4. test/tensor_placement_types_Partial/test_tensor_placement_types_Partial.py

**测试 API**: torch.distributed.tensor.placement_types.Partial

**测试方法**:
| 测试方法 | 卡数要求 | 说明 |
|---------|---------|------|
| test_partial_default | 2 | 测试默认 reduce_op ("sum") |
| test_partial_sum | 2 | 测试 reduce_op="sum" |
| test_partial_avg | 2 | 测试 reduce_op="avg" |
| test_partial_min | 2 | 测试 reduce_op="min" |
| test_partial_max | 2 | 测试 reduce_op="max" |
| test_partial_product | 2 | 测试 reduce_op="product" |
| test_partial_linear_ops | 2 | 测试所有 LINEAR_REDUCE_OPS |
| test_partial_all_ops | 2 | 测试所有 ALL_REDUCE_OPS |

### 5. test/distributed_c10d_split_group/test_distributed_c10d_split_group.py

**测试 API**: torch.distributed.distributed_c10d.split_group

**测试方法**:
| 测试方法 | 卡数要求 | 说明 |
|---------|---------|------|
| test_split_group_with_default_parent | 4 | 测试默认 parent_pg=None |
| test_split_group_with_explicit_parent | 4 | 测试显式 parent_pg |
| test_split_group_with_timeout | 4 | 测试 timeout 参数 |
| test_split_group_with_group_desc | 4 | 测试 group_desc 参数 |
| test_split_group_all_params | 4 | 测试所有参数组合 |
| test_split_group_single_group | 4 | 测试单 group 场景 |

### 6. test/distributed_c10d_new_subgroups/test_distributed_c10d_new_subgroups.py

**测试 API**: torch.distributed.distributed_c10d.new_subgroups

**测试方法**:
| 测试方法 | 卡数要求 | 说明 |
|---------|---------|------|
| test_new_subgroups_default | 2 | 测试默认参数 |
| test_new_subgroups_with_group_size | 4 | 测试 group_size 参数 |
| test_new_subgroups_with_explicit_group | 4 | 测试 group 参数 |
| test_new_subgroups_with_timeout | 4 | 测试 timeout 参数 |
| test_new_subgroups_with_backend | 4 | 测试 backend 参数 |
| test_new_subgroups_with_group_desc | 4 | 测试 group_desc 参数 |
| test_new_subgroups_all_params | 4 | 测试所有参数组合 |
| test_new_subgroups_subgroup_ops | 4 | 测试子 group 上的 all_reduce 操作 |

## 测试框架说明

### 通用测试框架

所有测试文件遵循 ascend_pytorch/test 风格：
- 使用 **unittest** 框架（**禁止 pytest**）
- 测试类继承 `TestCase`
- 设备检查放在 `setUp` 中，使用 `self.assertEqual(device_name, 'npu')` 检查
- 多卡测试使用 `@skipIfUnsupportMultiNPU(n)` 装饰器
- 使用 `torch.multiprocessing.spawn` 创建多进程
- 后端使用 `hccl`（通过 transfer_to_npu 自动映射）

### 测试执行命令示例

```bash
# 执行 FSDPModule 方法测试
python test/fsdp_FSDPModule_methods/test_fsdp_FSDPModule_methods.py

# 执行 FullyShardedDataParallel 方法测试
python test/fsdp_FullyShardedDataParallel_methods/test_fsdp_FullyShardedDataParallel_methods.py

# 执行 PrepareModuleOutput 测试
python test/tensor_parallel_PrepareModuleOutput/test_tensor_parallel_PrepareModuleOutput.py

# 执行 Partial 测试
python test/tensor_placement_types_Partial/test_tensor_placement_types_Partial.py

# 执行 split_group 测试
python test/distributed_c10d_split_group/test_distributed_c10d_split_group.py

# 执行 new_subgroups 测试
python test/distributed_c10d_new_subgroups/test_distributed_c10d_new_subgroups.py
```

## 未覆盖项说明

| API | 未覆盖场景 | 原因 |
|-----|-----------|------|
| FSDPModule 各 setter 方法 | 异常传参场景 | 这些 setter 主要用于 FSDP 内部状态管理，大多数不验证输入合法性，依赖 FSDP 内部逻辑保证正确调用 |
| FullyShardedDataParallel.no_sync | 非 root FSDP 调用 | 依赖特定嵌套 FSDP 结构才能触发异常 |
| PrepareModuleOutput | 非法 reduce_op 值 | Partial 对非法 reduce_op 的处理行为未明确文档化 |
| split_group/new_subgroups | 非法参数组合 | 异常处理依赖底层 NCCL/HCCL 实现，无稳定 Python 层异常路径 |

## 测试统计

| 类别 | 数量 |
|-----|------|
| 测试 API 总数 | 18 |
| 测试文件数 | 6 |
| 测试方法总数 | 44 |
| 需 2 卡测试方法 | 34 |
| 需 4 卡测试方法 | 10 |

## 文件列表

本次改动新增/修改的文件：

```
test/fsdp_FSDPModule_methods/test_fsdp_FSDPModule_methods.py
test/fsdp_FullyShardedDataParallel_methods/test_fsdp_FullyShardedDataParallel_methods.py
test/tensor_parallel_PrepareModuleOutput/test_tensor_parallel_PrepareModuleOutput.py
test/tensor_placement_types_Partial/test_tensor_placement_types_Partial.py
test/distributed_c10d_split_group/test_distributed_c10d_split_group.py
test/distributed_c10d_new_subgroups/test_distributed_c10d_new_subgroups.py
test/UT_EXECUTION_REPORT.md
```

---

报告生成时间: 2026-04-08
