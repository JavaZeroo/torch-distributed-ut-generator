# UT Report — torch.distributed.FileStore

## 执行命令
```bash
python test/_FileStore/test__FileStore.py
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
| test_init_basic | PASS | 基础路径创建 |
| test_init_world_size_default | PASS | 省略 world_size 使用默认 -1 |
| test_init_world_size_variants | PASS | world_size=-1/1/4 均正常 |
| test_init_explicit_world_size_positive | PASS | 显式 world_size=2 |
| test_init_empty_filename_raises | PASS | 空 file_name 抛异常 |
| test_path_matches_file_name | PASS | path 与构造时传入的路径一致 |
| test_path_type_is_str | PASS | path 返回 str 类型 |
| test_path_nonempty | PASS | path 非空 |
| test_multiple_stores_different_paths | PASS | 不同实例 path 互不相同 |
| test_store_basic_kv_operations | PASS | set/get 基础 KV 操作 |

## 统计
- 通过: 10
- 跳过: 0
- 失败: 0

## 跳过用例分析
无跳过用例（均为单进程测试）。

## 本次改动文件列表
- `test/_FileStore/test__FileStore.py`（新建）
- `test/_FileStore/UT_REPORT.md`（新建）
