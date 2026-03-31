# torch_npu API UT 执行报告

## 执行信息

| 项目 | 内容 |
|------|------|
| 执行时间 | 2026-03-31 |
| 执行命令 | `/usr/bin/python -m pytest test/ -v` |
| Python 版本 | 3.11.9 |
| PyTorch 版本 | 2.7.1+cpu |
| torch_npu 可用性 | True |
| NPU 设备数量 | 8 |

## 测试结果汇总

| API 名称 | 测试文件 | 测试数 | 通过 | 失败 | 状态 |
|----------|----------|--------|------|------|------|
| torch.cuda.Stream.wait_stream | cuda_Stream_wait_stream | 6 | 6 | 0 | ✅ |
| torch.distributed.Work | distributed_Work | 2 | 2 | 0 | ✅ |
| torch.distributed.Work.wait | distributed_Work_wait | 3 | 3 | 0 | ✅ |
| torch.split_with_sizes_copy | split_with_sizes_copy | 10 | 10 | 0 | ✅ |
| tensor.copy_ | tensor_copy_ | 10 | 10 | 0 | ✅ |
| torch.distributed.tensor.DTensor._local_tensor | distributed_tensor_DTensor_local_tensor | 2 | 2 | 0 | ✅ |
| torch.distributed.tensor._dtensor_spec.TensorMeta | distributed_tensor_dtensor_spec_TensorMeta | 7 | 7 | 0 | ✅ |
| torch.distributed._composable_state._insert_module_state | distributed_composable_state_insert_module_state | 5 | 5 | 0 | ✅ |
| torch.distributed._composable.contract | distributed_composable_contract | 5 | 5 | 0 | ✅ |
| torch.distributed.utils._get_root_modules | distributed_utils_get_root_modules | 7 | 7 | 0 | ✅ |
| torch.distributed.device_mesh._get_device_handle | distributed_device_mesh_get_device_handle | 6 | 6 | 0 | ✅ |
| torch.distributed._composable.contract._get_registry | distributed_composable_contract_get_registry | 5 | 5 | 0 | ✅ |
| torch.distributed.fsdp._common_utils._named_parameters_with_duplicates | distributed_fsdp_common_utils_named_parameters_with_duplicates | 7 | 7 | 0 | ✅ |

**统计**: 通过: 75 | 失败: 0 | 跳过: 0 | **总计: 75 (100%)**

## 结论

本次为 13 个 torch_npu API 生成功能 UT 用例，全部测试通过。UT 文件位于 `test/` 目录下，遵循 ascend_pytorch 测试风格（unittest + TestCase）。
