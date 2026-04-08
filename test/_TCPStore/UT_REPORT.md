# UT Report — torch.distributed.TCPStore

## 执行命令
```bash
python test/_TCPStore/test__TCPStore.py
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
| test_init_localhost | PASS | localhost 主机名构造 |
| test_init_ipv4 | PASS | 127.0.0.1 主机名构造 |
| test_init_world_size_none | PASS | world_size=None 合法 |
| test_init_with_timeout | PASS | 显式 timeout 参数 |
| test_libuv_variants | PASS | use_libuv=True/False 均正常 |
| test_invalid_port_raises | PASS | 负端口抛异常 |
| test_host_returns_str | PASS | host 返回 str |
| test_host_value | PASS | host 与构造时一致 |
| test_port_returns_int | PASS | port 返回 int |
| test_port_value | PASS | port 与构造时一致 |
| test_libuvbackend_returns_bool | PASS | libuvBackend 返回 bool |
| test_libuvbackend_true_when_use_libuv | PASS | use_libuv=True → libuvBackend=True |
| test_libuvbackend_false_when_not_libuv | PASS | use_libuv=False → libuvBackend=False |

## 统计
- 通过: 13
- 跳过: 0
- 失败: 0

## 跳过用例分析
无跳过用例（均为单进程 master-only 测试）。

## 本次改动文件列表
- `test/_TCPStore/test__TCPStore.py`（新建）
- `test/_TCPStore/UT_REPORT.md`（新建）
