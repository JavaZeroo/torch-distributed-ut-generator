# UT Report — torch.distributed.fsdp.FSDPModule

## 执行命令
```bash
python test/_fsdp_FSDPModule/test__fsdp_FSDPModule.py
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
| test_fsdp_module_isinstance | PASS | fully_shard() 返回 FSDPModule 实例 |
| test_reshard | PASS | unshard 后调用 reshard() 不报错 |
| test_set_requires_gradient_sync | PASS | True/False + recurse True/False 全覆盖 |
| test_set_requires_gradient_sync_nested | PASS | 嵌套子模块递归传播 |
| test_set_modules_to_forward_prefetch | PASS | 空列表和 singleton 列表均正常 |
| test_set_modules_to_forward_prefetch_invalid_type | PASS | 非 FSDPModule 输入抛异常 |
| test_set_modules_to_backward_prefetch | PASS | 空列表和 singleton 列表均正常 |
| test_set_modules_to_backward_prefetch_invalid_type | PASS | 非 FSDPModule 输入抛异常 |
| test_forward_with_gradient_sync_disabled | PASS | gradient sync 禁用时 forward/backward 正常 |

## 统计
- 通过: 9
- 跳过: 0
- 失败: 0

## 跳过用例分析
| 测试方法 | 跳过条件 | 跳过原因 | 合理性评估 |
|----------|----------|----------|------------|
| （无跳过） | — | — | — |

> 所有 9 个用例均带 `@skipIfUnsupportMultiNPU(2)`，环境有 8 卡，全部执行。

## 失败栈摘要
无

## 本次改动文件列表
- `test/_fsdp_FSDPModule/test__fsdp_FSDPModule.py`（新建）
- `test/_fsdp_FSDPModule/UT_REPORT.md`（新建）
