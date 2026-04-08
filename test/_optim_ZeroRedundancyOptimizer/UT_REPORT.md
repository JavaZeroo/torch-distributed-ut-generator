# UT Report — torch.distributed.optim.ZeroRedundancyOptimizer

## 执行命令
```bash
python test/_optim_ZeroRedundancyOptimizer/test__optim_ZeroRedundancyOptimizer.py
```

## 环境摘要
| 项目 | 值 |
|------|----|
| Python | 3.11.14 |
| PyTorch | 2.7.1+cpu |
| torch_npu | 2.7.1.post2 |
| NPU 设备 | Ascend910B3 × 8 |
| 测试后端 | hccl |

## 测试结果
| 测试方法 | 结果 | 说明 |
|----------|------|------|
| test_add_param_group | PASS | param_groups 数量增加 1 |
| test_add_param_group_lr_preserved | PASS | 新增 param group 的 lr 保持 |
| test_join_device | PASS | join_device 返回 npu 设备 |
| test_join_hook | PASS | join_hook 返回含 main_hook/post_hook 的 JoinHook |
| test_join_process_group | PASS | join_process_group 返回非空进程组 |
| test_state_dict_without_consolidate_raises | PASS | 未 consolidate 时调用 state_dict 抛 RuntimeError |
| test_state_dict_roundtrip | PASS | consolidate + state_dict + load_state_dict 完整流程 |
| test_load_state_dict_restores_lr | PASS | load_state_dict 后 optim 内 lr 恢复正确 |

## 统计
- 通过: 8
- 跳过: 0
- 失败: 0

## 跳过用例分析
| 测试方法 | 跳过条件 | 跳过原因 | 合理性评估 |
|----------|----------|----------|------------|
| （无跳过） | — | — | — |

> 所有 8 个测试均带 `@skipIfUnsupportMultiNPU(2)`，环境 8 卡，全部执行。

## 本次改动文件列表
- `test/_optim_ZeroRedundancyOptimizer/test__optim_ZeroRedundancyOptimizer.py`（新建）
- `test/_optim_ZeroRedundancyOptimizer/UT_REPORT.md`（新建）
