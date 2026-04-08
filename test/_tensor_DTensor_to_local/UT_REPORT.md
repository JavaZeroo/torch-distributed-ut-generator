# UT Report — torch.distributed.tensor.DTensor.to_local

## 执行命令
```bash
python test/_tensor_DTensor_to_local/test__tensor_DTensor_to_local.py
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
| test_to_local_no_grad_placements | PASS | Replicate DTensor，不传 grad_placements |
| test_to_local_with_grad_placements | PASS | 显式传 grad_placements=[Replicate()] |
| test_to_local_shard_dim0 | PASS | Shard(0) 本地 shard shape 正确 |
| test_to_local_shard_dim1 | PASS | Shard(1) 本地 shard shape 正确 |
| test_to_local_dtypes | PASS | float32 / float16 / int32 dtype 保持不变 |
| test_to_local_1d | PASS | 1-D DTensor 返回正确 shape |
| test_to_local_requires_grad | PASS | DTensor requires_grad 时返回张量也 requires_grad |
| test_to_local_no_grad | PASS | torch.no_grad() 下返回合法本地张量 |
| test_to_local_grad_placements_tuple | PASS | grad_placements 以 tuple 传入正常处理 |

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
- `test/_tensor_DTensor_to_local/test__tensor_DTensor_to_local.py`（新建）
- `test/_tensor_DTensor_to_local/UT_REPORT.md`（新建）
