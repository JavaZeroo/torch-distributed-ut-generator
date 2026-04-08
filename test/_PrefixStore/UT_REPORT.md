# UT Report — torch.distributed.PrefixStore

## 执行命令
```bash
python test/_PrefixStore/test__PrefixStore.py
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
| test_init_basic | PASS | 基础构造 |
| test_init_empty_prefix | PASS | 空字符串前缀合法 |
| test_init_special_chars | PASS | 含 / 等特殊字符前缀 |
| test_init_no_store_raises | PASS | 缺少 store 参数抛 TypeError |
| test_nested_prefix_store | PASS | 嵌套 PrefixStore 构造 |
| test_underlying_is_filestore | PASS | underlying_store 返回 FileStore 实例 |
| test_underlying_store_type | PASS | 嵌套时 underlying_store 为 PrefixStore |
| test_underlying_store_identity | PASS | underlying_store.path 与底层 FileStore 一致 |
| test_kv_isolation_by_prefix | PASS | 不同前缀 KV 隔离 |

## 统计
- 通过: 9
- 跳过: 0
- 失败: 0

## 跳过用例分析
无跳过用例（均为单进程测试）。

## 本次改动文件列表
- `test/_PrefixStore/test__PrefixStore.py`（新建）
- `test/_PrefixStore/UT_REPORT.md`（新建）
