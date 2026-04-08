# UT Report — torch.distributed.reinit_process_group

## 执行命令
```bash
python test/distributed_c10d_reinit_process_group/test_distributed_c10d_reinit_process_group.py
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
| test_function_signature | PASS | 签名含 group、rebuild_link，均有默认值 |
| test_reinit_default_group | PASS | group=None rebuild_link=True，返回 ProcessGroup |
| test_reinit_rebuild_link_false | PASS | rebuild_link=False 返回 None（resume 路径） |
| test_reinit_explicit_group | PASS | 显式传入 WORLD ProcessGroup 正常调用 |

## 统计
- 通过: 4
- 跳过: 0
- 失败: 0

## 跳过用例分析
| 测试方法 | 跳过条件 | 跳过原因 | 合理性评估 |
|----------|----------|----------|------------|
| （无跳过） | — | — | — |

> `test_reinit_default_group`、`test_reinit_rebuild_link_false`、`test_reinit_explicit_group` 均带 `@skipIfUnsupportMultiNPU(2)`，环境 8 卡，全部执行。  
> `test_function_signature` 为单进程签名检查，无需多卡。

## 失败栈摘要
无

## 本次改动文件列表
- `test/distributed_c10d_reinit_process_group/test_distributed_c10d_reinit_process_group.py`（新建）
- `test/distributed_c10d_reinit_process_group/UT_REPORT.md`（新建）
