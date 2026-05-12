# UT 批量执行报告

**生成时间**：2026-05-06 17:07:43
**并发 workers**：1  **单文件超时**：300s
**总文件数**：79  **累计耗时**：4400.6s

## 汇总

| 状态 | 数量 |
|------|------|
| ✅ PASS    | 78 |
| ❌ FAIL    | 1 |
| ⏱  TIMEOUT | 0 |
| 💥 ERROR   | 0 |
| **合计**   | **79** |

## 全量结果

| 状态 | 用时(s) | 通过 | 失败 | 跳过 | 文件 |
|------|---------|------|------|------|------|
| ✅ PASS | 49.3 | 2 | 0 | 0 | `test/Work/test_distributed_Work.py` |
| ✅ PASS | 66.6 | 3 | 0 | 0 | `test/Work_wait/test_distributed_Work_wait.py` |
| ✅ PASS | 14.3 | 9 | 0 | 1 | `test/_FileStore/test__FileStore.py` |
| ✅ PASS | 15.3 | 9 | 0 | 0 | `test/_PrefixStore/test__PrefixStore.py` |
| ✅ PASS | 16.3 | 13 | 0 | 0 | `test/_TCPStore/test__TCPStore.py` |
| ✅ PASS | 15.3 | 8 | 0 | 0 | `test/__functional_collectives_allow_inflight_collective_as_graph_input_ctx/test__functional_collectives_allow_inflight_collective_as_graph_input_ctx.py` |
| ✅ PASS | 15.4 | 6 | 0 | 0 | `test/_checkpoint_ReadItem/test__checkpoint_ReadItem.py` |
| ✅ PASS | 15.3 | 9 | 0 | 0 | `test/_checkpoint_SavePlanner/test__checkpoint_SavePlanner.py` |
| ✅ PASS | 15.0 | 13 | 0 | 0 | `test/_checkpoint_StorageReader/test__checkpoint_StorageReader.py` |
| ✅ PASS | 16.2 | 15 | 0 | 0 | `test/_checkpoint_StorageWriter/test__checkpoint_StorageWriter.py` |
| ✅ PASS | 15.4 | 12 | 0 | 0 | `test/_checkpoint_staging_AsyncStager/test__checkpoint_staging_AsyncStager.py` |
| ✅ PASS | 17.6 | 11 | 0 | 0 | `test/_checkpoint_staging_BlockingAsyncStager/test__checkpoint_staging_BlockingAsyncStager.py` |
| ✅ PASS | 15.3 | 15 | 0 | 0 | `test/_checkpoint_state_dict_saver_AsyncCheckpointerType/test__checkpoint_state_dict_saver_AsyncCheckpointerType.py` |
| ✅ PASS | 15.3 | 13 | 0 | 0 | `test/_checkpoint_stateful_Stateful/test__checkpoint_stateful_Stateful.py` |
| ✅ PASS | 17.2 | 5 | 0 | 0 | `test/_composable_contract/test_distributed_composable_contract.py` |
| ✅ PASS | 16.6 | 5 | 0 | 0 | `test/_composable_contract_get_registry/test_distributed_composable_contract_get_registry.py` |
| ✅ PASS | 16.8 | 5 | 0 | 0 | `test/_composable_state_insert_module_state/test_distributed_composable_state_insert_module_state.py` |
| ✅ PASS | 17.6 | 16 | 0 | 0 | `test/_foreach_copy_/test__foreach_copy_.py` |
| ✅ PASS | 170.1 | 9 | 0 | 0 | `test/_fsdp_FSDPModule/test__fsdp_FSDPModule.py` |
| ✅ PASS | 15.6 | 4 | 0 | 0 | `test/_functional_collectives_is_torchdynamo_compiling/test__functional_collectives_is_torchdynamo_compiling.py` |
| ✅ PASS | 15.0 | 12 | 0 | 0 | `test/_logging_warning_once/test__logging_warning_once.py` |
| ✅ PASS | 103.2 | 5 | 0 | 0 | `test/_optim_PostLocalSGDOptimizer/test__optim_PostLocalSGDOptimizer.py` |
| ✅ PASS | 162.9 | 8 | 0 | 0 | `test/_optim_ZeroRedundancyOptimizer/test__optim_ZeroRedundancyOptimizer.py` |
| ✅ PASS | 166.9 | 9 | 0 | 0 | `test/_tensor_DTensor_to_local/test__tensor_DTensor_to_local.py` |
| ✅ PASS | 32.4 | 4 | 0 | 0 | `test/algorithms_ddp_comm_hooks_post_localSGD_hook_PostLocalSGDState/test_algorithms_ddp_comm_hooks_post_localSGD_hook_PostLocalSGDState.py` |
| ✅ PASS | 17.0 | 23 | 0 | 0 | `test/autograd_graph__MultiHandle/test_autograd_graph__MultiHandle.py` |
| ✅ PASS | 78.7 | 4 | 0 | 0 | `test/checkpoint_LoadPlanner_commit_tensor/test_checkpoint_LoadPlanner_commit_tensor.py` |
| ✅ PASS | 59.9 | 3 | 0 | 0 | `test/checkpoint_LoadPlanner_load_bytes/test_checkpoint_LoadPlanner_load_bytes.py` |
| ✅ PASS | 59.5 | 3 | 0 | 0 | `test/checkpoint_LoadPlanner_resolve_bytes/test_checkpoint_LoadPlanner_resolve_bytes.py` |
| ✅ PASS | 79.7 | 4 | 0 | 0 | `test/checkpoint_LoadPlanner_resolve_tensor/test_checkpoint_LoadPlanner_resolve_tensor.py` |
| ✅ PASS | 30.0 | 4 | 0 | 0 | `test/checkpoint_SavePlan/test_checkpoint_SavePlan.py` |
| ✅ PASS | 80.3 | 4 | 0 | 0 | `test/checkpoint_SavePlanner_resolve_data/test_checkpoint_SavePlanner_resolve_data.py` |
| ✅ PASS | 16.0 | 29 | 0 | 0 | `test/checkpoint_base/test_checkpoint_base.py` |
| ✅ PASS | 15.5 | 17 | 0 | 0 | `test/checkpoint_default_planners/test_checkpoint_default_planners.py` |
| ✅ PASS | 15.2 | 19 | 0 | 0 | `test/checkpoint_format_utils/test_checkpoint_format_utils.py` |
| ✅ PASS | 97.3 | 5 | 0 | 0 | `test/checkpoint_format_utils_DynamicMetaLoadPlanner_set_up_planner/test_checkpoint_format_utils_DynamicMetaLoadPlanner_set_up_planner.py` |
| ✅ PASS | 30.5 | 4 | 0 | 0 | `test/checkpoint_planner_WriteItem/test_checkpoint_planner_WriteItem.py` |
| ✅ PASS | 15.4 | 12 | 0 | 0 | `test/checkpoint_planner_WriteItem_tensor_storage_size/test_checkpoint_planner_WriteItem_tensor_storage_size.py` |
| ✅ PASS | 16.7 | 6 | 0 | 0 | `test/cuda_Stream_wait_stream/test_cuda_Stream_wait_stream.py` |
| ✅ PASS | 15.3 | 6 | 0 | 0 | `test/device_mesh_get_device_handle/test_distributed_device_mesh_get_device_handle.py` |
| ✅ PASS | 34.0 | 5 | 0 | 0 | `test/distributed_c10d_ProcessGroupXCCL/test_distributed_c10d_ProcessGroupXCCL.py` |
| ✅ PASS | 31.2 | 4 | 0 | 0 | `test/distributed_c10d__new_process_group_helper/test_distributed_c10d__new_process_group_helper.py` |
| ✅ PASS | 29.3 | 4 | 0 | 0 | `test/distributed_c10d_destroy_process_group/test_distributed_c10d_destroy_process_group.py` |
| ✅ PASS | 15.1 | 4 | 0 | 0 | `test/distributed_c10d_is_xccl_available/test_distributed_c10d_is_xccl_available.py` |
| ✅ PASS | 32.4 | 1 | 0 | 7 | `test/distributed_c10d_new_subgroups/test_distributed_c10d_new_subgroups.py` |
| ✅ PASS | 227.1 | 4 | 0 | 0 | `test/distributed_c10d_reinit_process_group/test_distributed_c10d_reinit_process_group.py` |
| ✅ PASS | 14.7 | 0 | 0 | 6 | `test/distributed_c10d_split_group/test_distributed_c10d_split_group.py` |
| ✅ PASS | 132.7 | 7 | 0 | 0 | `test/fsdp_FSDPModule_methods/test_fsdp_FSDPModule_methods.py` |
| ✅ PASS | 186.2 | 10 | 0 | 0 | `test/fsdp_FullyShardedDataParallel_methods/test_fsdp_FullyShardedDataParallel_methods.py` |
| ✅ PASS | 30.4 | 4 | 0 | 0 | `test/fsdp__fsdp_extensions_FSDPExtensions/test_fsdp__fsdp_extensions_FSDPExtensions.py` |
| ✅ PASS | 31.6 | 4 | 0 | 0 | `test/fsdp__fsdp_extensions__ext_chunk_tensor/test_fsdp__fsdp_extensions__ext_chunk_tensor.py` |
| ✅ PASS | 29.2 | 5 | 0 | 0 | `test/fsdp__fsdp_extensions__ext_post_unflatten_transform/test_fsdp__fsdp_extensions__ext_post_unflatten_transform.py` |
| ✅ PASS | 30.1 | 5 | 0 | 0 | `test/fsdp__fsdp_extensions__ext_pre_flatten_transform/test_fsdp__fsdp_extensions__ext_pre_flatten_transform.py` |
| ✅ PASS | 30.1 | 5 | 0 | 0 | `test/fsdp__fsdp_extensions__ext_pre_load_state_dict_transform/test_fsdp__fsdp_extensions__ext_pre_load_state_dict_transform.py` |
| ✅ PASS | 32.5 | 7 | 0 | 0 | `test/fsdp_common_utils_named_parameters_with_duplicates/test_distributed_fsdp_common_utils_named_parameters_with_duplicates.py` |
| ✅ PASS | 30.6 | 3 | 0 | 0 | `test/new_subgroups_by_enumeration/test_new_subgroups_by_enumeration.py` |
| ✅ PASS | 67.3 | 3 | 0 | 0 | `test/optim_DistributedOptimizer/test_optim_DistributedOptimizer.py` |
| ✅ PASS | 67.3 | 3 | 0 | 0 | `test/optim_DistributedOptimizer_step/test_optim_DistributedOptimizer_step.py` |
| ✅ PASS | 112.2 | 6 | 0 | 0 | `test/optim_PostLocalSGDOptimizer/test_optim_PostLocalSGDOptimizer.py` |
| ✅ PASS | 104.6 | 5 | 0 | 0 | `test/optim_PostLocalSGDOptimizer_step/test_optim_PostLocalSGDOptimizer_step.py` |
| ✅ PASS | 113.4 | 6 | 0 | 0 | `test/optim_ZeroRedundancyOptimizer/test_optim_ZeroRedundancyOptimizer.py` |
| ✅ PASS | 101.2 | 5 | 0 | 0 | `test/optim_ZeroRedundancyOptimizer_step/test_optim_ZeroRedundancyOptimizer_step.py` |
| ✅ PASS | 55.9 | 4 | 0 | 0 | `test/rendezvous/test_rendezvous.py` |
| ✅ PASS | 29.0 | 4 | 0 | 0 | `test/rpc_TensorPipeRpcBackendOptions/test_rpc_TensorPipeRpcBackendOptions.py` |
| ✅ PASS | 14.8 | 4 | 0 | 0 | `test/rpc_get_worker_info/test_rpc_get_worker_info.py` |
| ✅ PASS | 54.7 | 4 | 0 | 0 | `test/rpc_init_rpc/test_rpc_init_rpc.py` |
| ✅ PASS | 15.1 | 4 | 0 | 0 | `test/rpc_shutdown/test_rpc_shutdown.py` |
| ✅ PASS | 16.2 | 18 | 0 | 0 | `test/split_with_sizes_copy/test_split_with_sizes_copy.py` |
| ✅ PASS | 47.0 | 2 | 0 | 0 | `test/tensor_DTensor_local_tensor/test_distributed_tensor_DTensor_local_tensor.py` |
| ✅ PASS | 17.3 | 18 | 0 | 0 | `test/tensor_copy_/test_tensor_copy_.py` |
| ✅ PASS | 15.6 | 7 | 0 | 0 | `test/tensor_dtensor_spec_TensorMeta/test_distributed_tensor_dtensor_spec_TensorMeta.py` |
| ✅ PASS | 147.9 | 8 | 0 | 0 | `test/tensor_full/test_tensor_full.py` |
| ✅ PASS | 128.4 | 7 | 0 | 0 | `test/tensor_ones/test_tensor_ones.py` |
| ✅ PASS | 87.0 | 4 | 0 | 0 | `test/tensor_parallel_PrepareModuleOutput/test_tensor_parallel_PrepareModuleOutput.py` |
| ❌ FAIL | 156.6 | 7 | 1 | 0 | `test/tensor_placement_types_Partial/test_tensor_placement_types_Partial.py` |
| ✅ PASS | 155.9 | 8 | 0 | 0 | `test/tensor_rand/test_tensor_rand.py` |
| ✅ PASS | 151.3 | 8 | 0 | 0 | `test/tensor_randn/test_tensor_randn.py` |
| ✅ PASS | 142.7 | 8 | 0 | 0 | `test/tensor_zeros/test_tensor_zeros.py` |
| ✅ PASS | 17.6 | 7 | 0 | 0 | `test/utils_get_root_modules/test_distributed_utils_get_root_modules.py` |

---

## 失败 / 超时 / 错误 详情

### ❌ test/tensor_placement_types_Partial/test_tensor_placement_types_Partial.py

- **状态**：FAIL
- **耗时**：156.6s
- **通过/失败/跳过**：7/1/0

```
【失败用例】
FAILED test/tensor_placement_types_Partial/test_tensor_placement_types_Partial.py::TestPartial::test_partial_max

【错误详情】
test/tensor_placement_types_Partial/test_tensor_placement_types_Partial.py:210: in test_partial_max
    mp.spawn(
/usr/local/python3.11.14/lib/python3.11/site-packages/torch/multiprocessing/spawn.py:340: in spawn
    return start_processes(fn, args, nprocs, join, daemon, start_method="spawn")
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
/usr/local/python3.11.14/lib/python3.11/site-packages/torch/multiprocessing/spawn.py:296: in start_processes
    while not context.join():
              ^^^^^^^^^^^^^^
/usr/local/python3.11.14/lib/python3.11/site-packages/torch/multiprocessing/spawn.py:215: in join
    raise ProcessRaisedException(msg, error_index, failed_process.pid)
E   torch.multiprocessing.spawn.ProcessRaisedException: 
E   
E   -- Process 0 terminate
---
```
