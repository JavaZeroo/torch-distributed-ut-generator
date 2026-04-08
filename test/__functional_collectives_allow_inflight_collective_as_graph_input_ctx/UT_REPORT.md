# UT Report — torch.distributed._functional_collectives.allow_inflight_collective_as_graph_input_ctx

## 执行命令
```bash
python test/__functional_collectives_allow_inflight_collective_as_graph_input_ctx/test__functional_collectives_allow_inflight_collective_as_graph_input_ctx.py
```

## 环境摘要
| 项目 | 值 |
|------|----|
| Python | 3.11.14 |
| PyTorch | 2.7.1+cpu |
| torch_npu | 2.7.1.post2 |
| NPU 设备 | Ascend910B3 × 8 |

## 测试结果
| 测试方法 | 结果 | 说明 |
|----------|------|------|
| test_ctx_enters_and_exits | PASS | 上下文正常进入和退出 |
| test_ctx_default_sets_true | PASS | 默认 value=True 在上下文内设置标志位为 True |
| test_ctx_false | PASS | value=False 在上下文内设置标志位为 False |
| test_ctx_explicit_true | PASS | 显式 value=True |
| test_ctx_restores_previous_value | PASS | 退出后标志位恢复为原始值 |
| test_ctx_restores_on_exception | PASS | 异常时标志位仍能恢复 |
| test_ctx_nested | PASS | 嵌套上下文 LIFO 恢复正确 |
| test_ctx_is_generator | PASS | 返回对象具有 __enter__/__exit__ |

## 统计
- 通过: 8
- 跳过: 0
- 失败: 0

## 跳过用例分析
无跳过用例（均为单进程测试）。

## 本次改动文件列表
- `test/__functional_collectives_allow_inflight_collective_as_graph_input_ctx/test__functional_collectives_allow_inflight_collective_as_graph_input_ctx.py`（新建）
- `test/__functional_collectives_allow_inflight_collective_as_graph_input_ctx/UT_REPORT.md`（新建）
