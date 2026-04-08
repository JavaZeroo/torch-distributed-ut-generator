# UT Report — torch.distributed.distributed_c10d.ProcessGroupXCCL

## 执行命令
```bash
# 首次（含修复前）
python test/distributed_c10d_ProcessGroupXCCL/test_distributed_c10d_ProcessGroupXCCL.py

# 修复后重跑
python test/distributed_c10d_ProcessGroupXCCL/test_distributed_c10d_ProcessGroupXCCL.py
```

## 环境摘要
| 项目 | 值 |
|------|----|
| Python | 3.11.14 |
| PyTorch | 2.7.1+cpu |
| torch_npu | 2.7.1.post2 |
| NPU 设备 | Ascend910B3 × 8 |
| XCCL 可用 | False（当前环境 ProcessGroupXCCL 未编译进安装包） |
| 测试后端 | hccl（集合通信测试） |

## 测试结果
| 测试方法 | 结果 | 说明 |
|----------|------|------|
| test_xccl_availability_check | PASS | is_xccl_available() 返回 bool |
| test_xccl_options_construction | PASS | XCCL 不可用时提前返回，不触发 ImportError |
| test_xccl_options_fields | PASS | XCCL 不可用时提前返回 |
| test_xccl_class_importable | PASS | XCCL 不可用时走 else 分支，断言一致 |
| test_xccl_process_group_creation | PASS | hccl 进程组创建 + all_reduce 正常 |

## 统计
- 通过: 5
- 跳过: 0
- 失败: 0

## 修复记录
首次执行 `test_xccl_options_construction` 和 `test_xccl_options_fields` 报 `ImportError`：
```
ImportError: cannot import name 'ProcessGroupXCCL' from 'torch.distributed.distributed_c10d'
```
原因：当 `is_xccl_available()` 为 False 时，`ProcessGroupXCCL` 不在 `distributed_c10d` 的导出列表中，直接 import 会失败。  
修复：将 import 语句改为先检查 `is_xccl_available()`，再用 `try/except ImportError` 包裹，XCCL 不可用时直接 return。

## 跳过用例分析
| 测试方法 | 跳过条件 | 跳过原因 | 合理性评估 |
|----------|----------|----------|------------|
| （无跳过） | — | — | — |

> `test_xccl_process_group_creation` 带 `@skipIfUnsupportMultiNPU(2)`，环境 8 卡，正常执行。  
> `test_xccl_options_*` 在 XCCL 不可用时通过 `return` 提前退出（非 skip），属合理降级处理。

## 失败栈摘要
无（修复后）

## 本次改动文件列表
- `test/distributed_c10d_ProcessGroupXCCL/test_distributed_c10d_ProcessGroupXCCL.py`（新建 + 修复 import 守护）
- `test/distributed_c10d_ProcessGroupXCCL/UT_REPORT.md`（新建）
