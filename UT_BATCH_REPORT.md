# UT 批量执行报告

**生成时间**：2026-04-28 17:50:10
**并发 workers**：1  **单文件超时**：200s
**总文件数**：79  **累计耗时**：5235.1s

## 汇总

| 状态 | 数量 |
|------|------|
| ✅ PASS    | 51 |
| ❌ FAIL    | 25 |
| ⏱  TIMEOUT | 3 |
| 💥 ERROR   | 0 |
| **合计**   | **79** |

## 全量结果

| 状态 | 用时(s) | 通过 | 失败 | 跳过 | 文件 |
|------|---------|------|------|------|------|
| ❌ FAIL | 80.1 | 0 | 2 | 0 | `test/Work/test_distributed_Work.py` |
| ❌ FAIL | 105.6 | 0 | 3 | 0 | `test/Work_wait/test_distributed_Work_wait.py` |
| ⏱ TIMEOUT | 200.4 | 0 | 0 | 0 | `test/_FileStore/test__FileStore.py` |
| ✅ PASS | 15.5 | 9 | 0 | 0 | `test/_PrefixStore/test__PrefixStore.py` |
| ✅ PASS | 16.3 | 13 | 0 | 0 | `test/_TCPStore/test__TCPStore.py` |
| ✅ PASS | 16.0 | 8 | 0 | 0 | `test/__functional_collectives_allow_inflight_collective_as_graph_input_ctx/test__functional_collectives_allow_inflight_collective_as_graph_input_ctx.py` |
| ✅ PASS | 15.2 | 6 | 0 | 0 | `test/_checkpoint_ReadItem/test__checkpoint_ReadItem.py` |
| ✅ PASS | 15.6 | 9 | 0 | 0 | `test/_checkpoint_SavePlanner/test__checkpoint_SavePlanner.py` |
| ✅ PASS | 17.9 | 13 | 0 | 0 | `test/_checkpoint_StorageReader/test__checkpoint_StorageReader.py` |
| ✅ PASS | 15.4 | 15 | 0 | 0 | `test/_checkpoint_StorageWriter/test__checkpoint_StorageWriter.py` |
| ✅ PASS | 15.8 | 12 | 0 | 0 | `test/_checkpoint_staging_AsyncStager/test__checkpoint_staging_AsyncStager.py` |
| ✅ PASS | 17.8 | 11 | 0 | 0 | `test/_checkpoint_staging_BlockingAsyncStager/test__checkpoint_staging_BlockingAsyncStager.py` |
| ✅ PASS | 15.6 | 15 | 0 | 0 | `test/_checkpoint_state_dict_saver_AsyncCheckpointerType/test__checkpoint_state_dict_saver_AsyncCheckpointerType.py` |
| ✅ PASS | 15.6 | 13 | 0 | 0 | `test/_checkpoint_stateful_Stateful/test__checkpoint_stateful_Stateful.py` |
| ✅ PASS | 16.8 | 5 | 0 | 0 | `test/_composable_contract/test_distributed_composable_contract.py` |
| ✅ PASS | 18.3 | 5 | 0 | 0 | `test/_composable_contract_get_registry/test_distributed_composable_contract_get_registry.py` |
| ✅ PASS | 16.8 | 5 | 0 | 0 | `test/_composable_state_insert_module_state/test_distributed_composable_state_insert_module_state.py` |
| ✅ PASS | 17.8 | 16 | 0 | 0 | `test/_foreach_copy_/test__foreach_copy_.py` |
| ❌ FAIL | 190.3 | 0 | 9 | 0 | `test/_fsdp_FSDPModule/test__fsdp_FSDPModule.py` |
| ✅ PASS | 16.4 | 4 | 0 | 0 | `test/_functional_collectives_is_torchdynamo_compiling/test__functional_collectives_is_torchdynamo_compiling.py` |
| ✅ PASS | 15.5 | 12 | 0 | 0 | `test/_logging_warning_once/test__logging_warning_once.py` |
| ❌ FAIL | 115.3 | 0 | 5 | 0 | `test/_optim_PostLocalSGDOptimizer/test__optim_PostLocalSGDOptimizer.py` |
| ❌ FAIL | 162.8 | 0 | 8 | 0 | `test/_optim_ZeroRedundancyOptimizer/test__optim_ZeroRedundancyOptimizer.py` |
| ✅ PASS | 191.2 | 9 | 0 | 0 | `test/_tensor_DTensor_to_local/test__tensor_DTensor_to_local.py` |
| ✅ PASS | 30.5 | 4 | 0 | 0 | `test/algorithms_ddp_comm_hooks_post_localSGD_hook_PostLocalSGDState/test_algorithms_ddp_comm_hooks_post_localSGD_hook_PostLocalSGDState.py` |
| ✅ PASS | 16.6 | 23 | 0 | 0 | `test/autograd_graph__MultiHandle/test_autograd_graph__MultiHandle.py` |
| ✅ PASS | 79.9 | 4 | 0 | 0 | `test/checkpoint_LoadPlanner_commit_tensor/test_checkpoint_LoadPlanner_commit_tensor.py` |
| ✅ PASS | 61.0 | 3 | 0 | 0 | `test/checkpoint_LoadPlanner_load_bytes/test_checkpoint_LoadPlanner_load_bytes.py` |
| ✅ PASS | 61.2 | 3 | 0 | 0 | `test/checkpoint_LoadPlanner_resolve_bytes/test_checkpoint_LoadPlanner_resolve_bytes.py` |
| ✅ PASS | 79.8 | 4 | 0 | 0 | `test/checkpoint_LoadPlanner_resolve_tensor/test_checkpoint_LoadPlanner_resolve_tensor.py` |
| ❌ FAIL | 56.2 | 0 | 4 | 0 | `test/checkpoint_SavePlan/test_checkpoint_SavePlan.py` |
| ✅ PASS | 78.1 | 4 | 0 | 0 | `test/checkpoint_SavePlanner_resolve_data/test_checkpoint_SavePlanner_resolve_data.py` |
| ✅ PASS | 16.1 | 29 | 0 | 0 | `test/checkpoint_base/test_checkpoint_base.py` |
| ✅ PASS | 15.8 | 17 | 0 | 0 | `test/checkpoint_default_planners/test_checkpoint_default_planners.py` |
| ✅ PASS | 16.3 | 19 | 0 | 0 | `test/checkpoint_format_utils/test_checkpoint_format_utils.py` |
| ✅ PASS | 101.5 | 5 | 0 | 0 | `test/checkpoint_format_utils_DynamicMetaLoadPlanner_set_up_planner/test_checkpoint_format_utils_DynamicMetaLoadPlanner_set_up_planner.py` |
| ✅ PASS | 30.3 | 4 | 0 | 0 | `test/checkpoint_planner_WriteItem/test_checkpoint_planner_WriteItem.py` |
| ✅ PASS | 16.3 | 12 | 0 | 0 | `test/checkpoint_planner_WriteItem_tensor_storage_size/test_checkpoint_planner_WriteItem_tensor_storage_size.py` |
| ✅ PASS | 16.9 | 6 | 0 | 0 | `test/cuda_Stream_wait_stream/test_cuda_Stream_wait_stream.py` |
| ✅ PASS | 15.0 | 6 | 0 | 0 | `test/device_mesh_get_device_handle/test_distributed_device_mesh_get_device_handle.py` |
| ❌ FAIL | 87.0 | 4 | 1 | 0 | `test/distributed_c10d_ProcessGroupXCCL/test_distributed_c10d_ProcessGroupXCCL.py` |
| ❌ FAIL | 29.7 | 3 | 1 | 0 | `test/distributed_c10d__new_process_group_helper/test_distributed_c10d__new_process_group_helper.py` |
| ✅ PASS | 30.4 | 4 | 0 | 0 | `test/distributed_c10d_destroy_process_group/test_distributed_c10d_destroy_process_group.py` |
| ✅ PASS | 15.6 | 4 | 0 | 0 | `test/distributed_c10d_is_xccl_available/test_distributed_c10d_is_xccl_available.py` |
| ❌ FAIL | 157.0 | 0 | 8 | 0 | `test/distributed_c10d_new_subgroups/test_distributed_c10d_new_subgroups.py` |
| ⏱ TIMEOUT | 200.4 | 0 | 0 | 0 | `test/distributed_c10d_reinit_process_group/test_distributed_c10d_reinit_process_group.py` |
| ❌ FAIL | 107.2 | 0 | 6 | 0 | `test/distributed_c10d_split_group/test_distributed_c10d_split_group.py` |
| ❌ FAIL | 145.9 | 0 | 7 | 0 | `test/fsdp_FSDPModule_methods/test_fsdp_FSDPModule_methods.py` |
| ❌ FAIL | 176.1 | 0 | 10 | 0 | `test/fsdp_FullyShardedDataParallel_methods/test_fsdp_FullyShardedDataParallel_methods.py` |
| ❌ FAIL | 29.5 | 0 | 4 | 0 | `test/fsdp__fsdp_extensions_FSDPExtensions/test_fsdp__fsdp_extensions_FSDPExtensions.py` |
| ❌ FAIL | 56.3 | 0 | 4 | 0 | `test/fsdp__fsdp_extensions__ext_chunk_tensor/test_fsdp__fsdp_extensions__ext_chunk_tensor.py` |
| ❌ FAIL | 30.1 | 3 | 1 | 0 | `test/fsdp__fsdp_extensions__ext_post_unflatten_transform/test_fsdp__fsdp_extensions__ext_post_unflatten_transform.py` |
| ❌ FAIL | 56.8 | 1 | 3 | 0 | `test/fsdp__fsdp_extensions__ext_pre_flatten_transform/test_fsdp__fsdp_extensions__ext_pre_flatten_transform.py` |
| ❌ FAIL | 57.0 | 1 | 3 | 0 | `test/fsdp__fsdp_extensions__ext_pre_load_state_dict_transform/test_fsdp__fsdp_extensions__ext_pre_load_state_dict_transform.py` |
| ✅ PASS | 37.5 | 7 | 0 | 0 | `test/fsdp_common_utils_named_parameters_with_duplicates/test_distributed_fsdp_common_utils_named_parameters_with_duplicates.py` |
| ❌ FAIL | 55.9 | 1 | 3 | 0 | `test/new_subgroups_by_enumeration/test_new_subgroups_by_enumeration.py` |
| ✅ PASS | 68.5 | 3 | 0 | 0 | `test/optim_DistributedOptimizer/test_optim_DistributedOptimizer.py` |
| ✅ PASS | 115.2 | 3 | 0 | 0 | `test/optim_DistributedOptimizer_step/test_optim_DistributedOptimizer_step.py` |
| ⏱ TIMEOUT | 200.3 | 0 | 0 | 0 | `test/optim_PostLocalSGDOptimizer/test_optim_PostLocalSGDOptimizer.py` |
| ❌ FAIL | 100.5 | 0 | 5 | 0 | `test/optim_PostLocalSGDOptimizer_step/test_optim_PostLocalSGDOptimizer_step.py` |
| ✅ PASS | 113.0 | 6 | 0 | 0 | `test/optim_ZeroRedundancyOptimizer/test_optim_ZeroRedundancyOptimizer.py` |
| ❌ FAIL | 100.5 | 0 | 5 | 0 | `test/optim_ZeroRedundancyOptimizer_step/test_optim_ZeroRedundancyOptimizer_step.py` |
| ✅ PASS | 55.6 | 4 | 0 | 0 | `test/rendezvous/test_rendezvous.py` |
| ✅ PASS | 29.4 | 4 | 0 | 0 | `test/rpc_TensorPipeRpcBackendOptions/test_rpc_TensorPipeRpcBackendOptions.py` |
| ✅ PASS | 14.8 | 4 | 0 | 0 | `test/rpc_get_worker_info/test_rpc_get_worker_info.py` |
| ✅ PASS | 56.1 | 4 | 0 | 0 | `test/rpc_init_rpc/test_rpc_init_rpc.py` |
| ✅ PASS | 14.9 | 4 | 0 | 0 | `test/rpc_shutdown/test_rpc_shutdown.py` |
| ❌ FAIL | 10.2 | 0 | 0 | 0 | `test/split_with_sizes_copy/test_split_with_sizes_copy.py` |
| ✅ PASS | 46.9 | 2 | 0 | 0 | `test/tensor_DTensor_local_tensor/test_distributed_tensor_DTensor_local_tensor.py` |
| ✅ PASS | 16.6 | 18 | 0 | 0 | `test/tensor_copy_/test_tensor_copy_.py` |
| ✅ PASS | 15.3 | 7 | 0 | 0 | `test/tensor_dtensor_spec_TensorMeta/test_distributed_tensor_dtensor_spec_TensorMeta.py` |
| ✅ PASS | 149.5 | 8 | 0 | 0 | `test/tensor_full/test_tensor_full.py` |
| ✅ PASS | 133.2 | 7 | 0 | 0 | `test/tensor_ones/test_tensor_ones.py` |
| ❌ FAIL | 80.7 | 0 | 4 | 0 | `test/tensor_parallel_PrepareModuleOutput/test_tensor_parallel_PrepareModuleOutput.py` |
| ❌ FAIL | 145.3 | 0 | 8 | 0 | `test/tensor_placement_types_Partial/test_tensor_placement_types_Partial.py` |
| ❌ FAIL | 152.1 | 0 | 8 | 0 | `test/tensor_rand/test_tensor_rand.py` |
| ❌ FAIL | 156.0 | 0 | 8 | 0 | `test/tensor_randn/test_tensor_randn.py` |
| ✅ PASS | 156.5 | 8 | 0 | 0 | `test/tensor_zeros/test_tensor_zeros.py` |
| ✅ PASS | 16.5 | 7 | 0 | 0 | `test/utils_get_root_modules/test_distributed_utils_get_root_modules.py` |

---

## 失败 / 超时 / 错误 详情

### ❌ test/Work/test_distributed_Work.py

- **状态**：FAIL
- **耗时**：80.1s
- **通过/失败/跳过**：0/2/0

```
【失败用例】
FAILED test/Work/test_distributed_Work.py::TestDistributedWork::test_work_creation_from_isend
FAILED test/Work/test_distributed_Work.py::TestDistributedWork::test_work_is_completed

【错误详情】
test/Work/test_distributed_Work.py:92: in test_work_creation_from_isend
    results.append(c2p.get(timeout=30))
                   ^^^^^^^^^^^^^^^^^^^
/usr/local/python3.11.14/lib/python3.11/multiprocessing/queues.py:114: in get
    raise Empty
E   _queue.Empty
----------------------------- Captured stderr call -----------------------------
Process SpawnProcess-1:
Traceback (most recent call last):
  File "/usr/local/python3.11.14/lib/python3.11/multiprocessing/process.py", line 314, in _bootstrap
    self.run()
  File "/usr/local/python3.11.14/lib/python3.11/multiprocessing/process.py", line 108, in run
    self._target(*self._args, **self._kwargs)
  File "/home/l00913161/projects/torch-distributed-ut-generator/test/Work/test_distributed_Work.py", line 60, in _test_work_creation_and_wait

---
test/Work/test_distributed_Work.py:121: in test_work_is_completed
    results.append(c2p.get(timeout=30))
                   ^^^^^^^^^^^^^^^^^^^
/usr/local/python3.11.14/lib/python3.11/multiprocessing/queues.py:114: in get
    raise Empty
E   _queue.Empty
----------------------------- Captured stderr call -----------------------------
Process SpawnProcess-3:
Traceback (most recent call last):
  File "/usr/local/python3.11.14/lib/python3.11/multiprocessing/process.py", line 314, in _bootstrap
    self.run()
  File "/usr/local/python3.11.14/lib/python3.11/multiprocessing/process.py", line 108, in run
    self._target(*self._args, **self._kwargs)
  File "/home/l00913161/projects/torch-distributed-ut-generator/test/Work/test_distributed_Work.py", line 60, in _test_work_creation_and_wait
    wo
---
```

### ❌ test/Work_wait/test_distributed_Work_wait.py

- **状态**：FAIL
- **耗时**：105.6s
- **通过/失败/跳过**：0/3/0

```
【失败用例】
FAILED test/Work_wait/test_distributed_Work_wait.py::TestDistributedWorkWait::test_wait_completion
FAILED test/Work_wait/test_distributed_Work_wait.py::TestDistributedWorkWait::test_wait_default_timeout
FAILED test/Work_wait/test_distributed_Work_wait.py::TestDistributedWorkWait::test_wait_with_timeout

【错误详情】
test/Work_wait/test_distributed_Work_wait.py:161: in test_wait_completion
    results.append(c2p.get(timeout=30))
                   ^^^^^^^^^^^^^^^^^^^
/usr/local/python3.11.14/lib/python3.11/multiprocessing/queues.py:114: in get
    raise Empty
E   _queue.Empty
----------------------------- Captured stderr call -----------------------------
Process SpawnProcess-1:
Traceback (most recent call last):
  File "/usr/local/python3.11.14/lib/python3.11/multiprocessing/process.py", line 314, in _bootstrap
    self.run()
  File "/usr/local/python3.11.14/lib/python3.11/multiprocessing/process.py", line 108, in run
    self._target(*self._args, **self._kwargs)
  File "/home/l00913161/projects/torch-distributed-ut-generator/test/Work_wait/test_distributed_Work_wait.py", line 180, in _test_wait_compl
---
test/Work_wait/test_distributed_Work_wait.py:107: in test_wait_default_timeout
    results.append(c2p.get(timeout=30))
                   ^^^^^^^^^^^^^^^^^^^
/usr/local/python3.11.14/lib/python3.11/multiprocessing/queues.py:114: in get
    raise Empty
E   _queue.Empty
----------------------------- Captured stderr call -----------------------------
Process SpawnProcess-3:
Traceback (most recent call last):
  File "/usr/local/python3.11.14/lib/python3.11/multiprocessing/process.py", line 314, in _bootstrap
    self.run()
  File "/usr/local/python3.11.14/lib/python3.11/multiprocessing/process.py", line 108, in run
    self._target(*self._args, **self._kwargs)
  File "/home/l00913161/projects/torch-distributed-ut-generator/test/Work_wait/test_distributed_Work_wait.py", line 57, in _test_wait_d
---
test/Work_wait/test_distributed_Work_wait.py:134: in test_wait_with_timeout
    results.append(c2p.get(timeout=30))
                   ^^^^^^^^^^^^^^^^^^^
/usr/local/python3.11.14/lib/python3.11/multiprocessing/queues.py:114: in get
    raise Empty
E   _queue.Empty
----------------------------- Captured stderr call -----------------------------
Process SpawnProcess-5:
Traceback (most recent call last):
  File "/usr/local/python3.11.14/lib/python3.11/multiprocessing/process.py", line 314, in _bootstrap
    self.run()
  File "/usr/local/python3.11.14/lib/python3.11/multiprocessing/process.py", line 108, in run
    self._target(*self._args, **self._kwargs)
  File "/home/l00913161/projects/torch-distributed-ut-generator/test/Work_wait/test_distributed_Work_wait.py", line 77, in _test_wait_with
---
```

### ⏱ test/_FileStore/test__FileStore.py

- **状态**：TIMEOUT
- **耗时**：200.4s
- **通过/失败/跳过**：0/0/0

```
超时（>200s），进程已终止
```

### ❌ test/_fsdp_FSDPModule/test__fsdp_FSDPModule.py

- **状态**：FAIL
- **耗时**：190.3s
- **通过/失败/跳过**：0/9/0

```
【失败用例】
FAILED test/_fsdp_FSDPModule/test__fsdp_FSDPModule.py::TestFSDPModule::test_forward_with_gradient_sync_disabled
FAILED test/_fsdp_FSDPModule/test__fsdp_FSDPModule.py::TestFSDPModule::test_fsdp_module_isinstance
FAILED test/_fsdp_FSDPModule/test__fsdp_FSDPModule.py::TestFSDPModule::test_reshard
FAILED test/_fsdp_FSDPModule/test__fsdp_FSDPModule.py::TestFSDPModule::test_set_modules_to_backward_prefetch
FAILED test/_fsdp_FSDPModule/test__fsdp_FSDPModule.py::TestFSDPModule::test_set_modules_to_backward_prefetch_invalid_type
FAILED test/_fsdp_FSDPModule/test__fsdp_FSDPModule.py::TestFSDPModule::test_set_modules_to_forward_prefetch
FAILED test/_fsdp_FSDPModule/test__fsdp_FSDPModule.py::TestFSDPModule::test_set_modules_to_forward_prefetch_invalid_type
FAILED test/_fsdp_FSDPModule/test__fsdp_FSDPModule.py::TestFSDPModule::test_set_requires_gradient_sync
FAILED test/_fsdp_FSDPModule/test__fsdp_FSDPModule.py::TestFSDPModule::test_set_requires_gradient_sync_nested

【错误详情】
test/_fsdp_FSDPModule/test__fsdp_FSDPModule.py:316: in test_forward_with_gradient_sync_disabled
    self._run(_test_forward_with_gradient_sync_disabled)
test/_fsdp_FSDPModule/test__fsdp_FSDPModule.py:266: in _run
    mp.spawn(
/usr/local/python3.11.14/lib/python3.11/site-packages/torch/multiprocessing/spawn.py:340: in spawn
    return start_processes(fn, args, nprocs, join, daemon, start_method="spawn")
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
/usr/local/python3.11.14/lib/python3.11/site-packages/torch/multiprocessing/spawn.py:296: in start_processes
    while not context.join():
              ^^^^^^^^^^^^^^
/usr/local/python3.11.14/lib/python3.11/site-packages/torch/multiprocessing/spawn.py:215: in join
    raise ProcessRaisedException(msg, error_in
---
test/_fsdp_FSDPModule/test__fsdp_FSDPModule.py:276: in test_fsdp_module_isinstance
    self._run(_test_fsdp_module_isinstance)
test/_fsdp_FSDPModule/test__fsdp_FSDPModule.py:266: in _run
    mp.spawn(
/usr/local/python3.11.14/lib/python3.11/site-packages/torch/multiprocessing/spawn.py:340: in spawn
    return start_processes(fn, args, nprocs, join, daemon, start_method="spawn")
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
/usr/local/python3.11.14/lib/python3.11/site-packages/torch/multiprocessing/spawn.py:296: in start_processes
    while not context.join():
              ^^^^^^^^^^^^^^
/usr/local/python3.11.14/lib/python3.11/site-packages/torch/multiprocessing/spawn.py:215: in join
    raise ProcessRaisedException(msg, error_index, failed_process.pid)
E
---
test/_fsdp_FSDPModule/test__fsdp_FSDPModule.py:281: in test_reshard
    self._run(_test_reshard)
test/_fsdp_FSDPModule/test__fsdp_FSDPModule.py:266: in _run
    mp.spawn(
/usr/local/python3.11.14/lib/python3.11/site-packages/torch/multiprocessing/spawn.py:340: in spawn
    return start_processes(fn, args, nprocs, join, daemon, start_method="spawn")
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
/usr/local/python3.11.14/lib/python3.11/site-packages/torch/multiprocessing/spawn.py:296: in start_processes
    while not context.join():
              ^^^^^^^^^^^^^^
/usr/local/python3.11.14/lib/python3.11/site-packages/torch/multiprocessing/spawn.py:215: in join
    raise ProcessRaisedException(msg, error_index, failed_process.pid)
E   torch.multiprocessing.spawn
---
test/_fsdp_FSDPModule/test__fsdp_FSDPModule.py:306: in test_set_modules_to_backward_prefetch
    self._run(_test_set_modules_to_backward_prefetch)
test/_fsdp_FSDPModule/test__fsdp_FSDPModule.py:266: in _run
    mp.spawn(
/usr/local/python3.11.14/lib/python3.11/site-packages/torch/multiprocessing/spawn.py:340: in spawn
    return start_processes(fn, args, nprocs, join, daemon, start_method="spawn")
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
/usr/local/python3.11.14/lib/python3.11/site-packages/torch/multiprocessing/spawn.py:296: in start_processes
    while not context.join():
              ^^^^^^^^^^^^^^
/usr/local/python3.11.14/lib/python3.11/site-packages/torch/multiprocessing/spawn.py:215: in join
    raise ProcessRaisedException(msg, error_index, f
---
test/_fsdp_FSDPModule/test__fsdp_FSDPModule.py:311: in test_set_modules_to_backward_prefetch_invalid_type
    self._run(_test_set_modules_to_backward_prefetch_invalid)
test/_fsdp_FSDPModule/test__fsdp_FSDPModule.py:266: in _run
    mp.spawn(
/usr/local/python3.11.14/lib/python3.11/site-packages/torch/multiprocessing/spawn.py:340: in spawn
    return start_processes(fn, args, nprocs, join, daemon, start_method="spawn")
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
/usr/local/python3.11.14/lib/python3.11/site-packages/torch/multiprocessing/spawn.py:296: in start_processes
    while not context.join():
              ^^^^^^^^^^^^^^
/usr/local/python3.11.14/lib/python3.11/site-packages/torch/multiprocessing/spawn.py:215: in join
    raise ProcessRaisedExcep
... (截断，共 8227 字符)
```

### ❌ test/_optim_PostLocalSGDOptimizer/test__optim_PostLocalSGDOptimizer.py

- **状态**：FAIL
- **耗时**：115.3s
- **通过/失败/跳过**：0/5/0

```
【失败用例】
FAILED test/_optim_PostLocalSGDOptimizer/test__optim_PostLocalSGDOptimizer.py::TestPostLocalSGDOptimizer::test_load_state_dict_does_not_raise
FAILED test/_optim_PostLocalSGDOptimizer/test__optim_PostLocalSGDOptimizer.py::TestPostLocalSGDOptimizer::test_load_state_dict_no_step_warns
FAILED test/_optim_PostLocalSGDOptimizer/test__optim_PostLocalSGDOptimizer.py::TestPostLocalSGDOptimizer::test_state_dict_contains_step
FAILED test/_optim_PostLocalSGDOptimizer/test__optim_PostLocalSGDOptimizer.py::TestPostLocalSGDOptimizer::test_state_dict_returns_dict
FAILED test/_optim_PostLocalSGDOptimizer/test__optim_PostLocalSGDOptimizer.py::TestPostLocalSGDOptimizer::test_state_dict_step_roundtrip

【错误详情】
test/_optim_PostLocalSGDOptimizer/test__optim_PostLocalSGDOptimizer.py:197: in test_load_state_dict_does_not_raise
    self._run(_test_load_state_dict_does_not_raise)
test/_optim_PostLocalSGDOptimizer/test__optim_PostLocalSGDOptimizer.py:167: in _run
    mp.spawn(
/usr/local/python3.11.14/lib/python3.11/site-packages/torch/multiprocessing/spawn.py:340: in spawn
    return start_processes(fn, args, nprocs, join, daemon, start_method="spawn")
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
/usr/local/python3.11.14/lib/python3.11/site-packages/torch/multiprocessing/spawn.py:296: in start_processes
    while not context.join():
              ^^^^^^^^^^^^^^
/usr/local/python3.11.14/lib/python3.11/site-packages/torch/multiprocessing/spawn.py:215: in join
    rais
---
test/_optim_PostLocalSGDOptimizer/test__optim_PostLocalSGDOptimizer.py:192: in test_load_state_dict_no_step_warns
    self._run(_test_load_state_dict_no_step_warns)
test/_optim_PostLocalSGDOptimizer/test__optim_PostLocalSGDOptimizer.py:167: in _run
    mp.spawn(
/usr/local/python3.11.14/lib/python3.11/site-packages/torch/multiprocessing/spawn.py:340: in spawn
    return start_processes(fn, args, nprocs, join, daemon, start_method="spawn")
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
/usr/local/python3.11.14/lib/python3.11/site-packages/torch/multiprocessing/spawn.py:296: in start_processes
    while not context.join():
              ^^^^^^^^^^^^^^
/usr/local/python3.11.14/lib/python3.11/site-packages/torch/multiprocessing/spawn.py:215: in join
    raise 
---
test/_optim_PostLocalSGDOptimizer/test__optim_PostLocalSGDOptimizer.py:177: in test_state_dict_contains_step
    self._run(_test_state_dict_contains_step)
test/_optim_PostLocalSGDOptimizer/test__optim_PostLocalSGDOptimizer.py:167: in _run
    mp.spawn(
/usr/local/python3.11.14/lib/python3.11/site-packages/torch/multiprocessing/spawn.py:340: in spawn
    return start_processes(fn, args, nprocs, join, daemon, start_method="spawn")
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
/usr/local/python3.11.14/lib/python3.11/site-packages/torch/multiprocessing/spawn.py:296: in start_processes
    while not context.join():
              ^^^^^^^^^^^^^^
/usr/local/python3.11.14/lib/python3.11/site-packages/torch/multiprocessing/spawn.py:215: in join
    raise ProcessRai
---
test/_optim_PostLocalSGDOptimizer/test__optim_PostLocalSGDOptimizer.py:182: in test_state_dict_returns_dict
    self._run(_test_state_dict_returns_dict)
test/_optim_PostLocalSGDOptimizer/test__optim_PostLocalSGDOptimizer.py:167: in _run
    mp.spawn(
/usr/local/python3.11.14/lib/python3.11/site-packages/torch/multiprocessing/spawn.py:340: in spawn
    return start_processes(fn, args, nprocs, join, daemon, start_method="spawn")
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
/usr/local/python3.11.14/lib/python3.11/site-packages/torch/multiprocessing/spawn.py:296: in start_processes
    while not context.join():
              ^^^^^^^^^^^^^^
/usr/local/python3.11.14/lib/python3.11/site-packages/torch/multiprocessing/spawn.py:215: in join
    raise ProcessRaise
---
test/_optim_PostLocalSGDOptimizer/test__optim_PostLocalSGDOptimizer.py:187: in test_state_dict_step_roundtrip
    self._run(_test_state_dict_step_roundtrip)
test/_optim_PostLocalSGDOptimizer/test__optim_PostLocalSGDOptimizer.py:167: in _run
    mp.spawn(
/usr/local/python3.11.14/lib/python3.11/site-packages/torch/multiprocessing/spawn.py:340: in spawn
    return start_processes(fn, args, nprocs, join, daemon, start_method="spawn")
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
/usr/local/python3.11.14/lib/python3.11/site-packages/torch/multiprocessing/spawn.py:296: in start_processes
    while not context.join():
              ^^^^^^^^^^^^^^
/usr/local/python3.11.14/lib/python3.11/site-packages/torch/multiprocessing/spawn.py:215: in join
    raise ProcessR
---
```

### ❌ test/_optim_ZeroRedundancyOptimizer/test__optim_ZeroRedundancyOptimizer.py

- **状态**：FAIL
- **耗时**：162.8s
- **通过/失败/跳过**：0/8/0

```
【失败用例】
FAILED test/_optim_ZeroRedundancyOptimizer/test__optim_ZeroRedundancyOptimizer.py::TestZeroRedundancyOptimizer::test_add_param_group
FAILED test/_optim_ZeroRedundancyOptimizer/test__optim_ZeroRedundancyOptimizer.py::TestZeroRedundancyOptimizer::test_add_param_group_lr_preserved
FAILED test/_optim_ZeroRedundancyOptimizer/test__optim_ZeroRedundancyOptimizer.py::TestZeroRedundancyOptimizer::test_join_device
FAILED test/_optim_ZeroRedundancyOptimizer/test__optim_ZeroRedundancyOptimizer.py::TestZeroRedundancyOptimizer::test_join_hook
FAILED test/_optim_ZeroRedundancyOptimizer/test__optim_ZeroRedundancyOptimizer.py::TestZeroRedundancyOptimizer::test_join_process_group
FAILED test/_optim_ZeroRedundancyOptimizer/test__optim_ZeroRedundancyOptimizer.py::TestZeroRedundancyOptimizer::test_load_state_dict_restores_lr
FAILED test/_optim_ZeroRedundancyOptimizer/test__optim_ZeroRedundancyOptimizer.py::TestZeroRedundancyOptimizer::test_state_dict_roundtrip
FAILED test/_optim_ZeroRedundancyOptimizer/test__optim_ZeroRedundancyOptimizer.py::TestZeroRedundancyOptimizer::test_state_dict_without_consolidate_raises

【错误详情】
test/_optim_ZeroRedundancyOptimizer/test__optim_ZeroRedundancyOptimizer.py:235: in test_add_param_group
    self._run(_test_add_param_group)
test/_optim_ZeroRedundancyOptimizer/test__optim_ZeroRedundancyOptimizer.py:225: in _run
    mp.spawn(
/usr/local/python3.11.14/lib/python3.11/site-packages/torch/multiprocessing/spawn.py:340: in spawn
    return start_processes(fn, args, nprocs, join, daemon, start_method="spawn")
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
/usr/local/python3.11.14/lib/python3.11/site-packages/torch/multiprocessing/spawn.py:296: in start_processes
    while not context.join():
              ^^^^^^^^^^^^^^
/usr/local/python3.11.14/lib/python3.11/site-packages/torch/multiprocessing/spawn.py:215: in join
    raise ProcessRaisedExcepti
---
test/_optim_ZeroRedundancyOptimizer/test__optim_ZeroRedundancyOptimizer.py:240: in test_add_param_group_lr_preserved
    self._run(_test_add_param_group_lr_preserved)
test/_optim_ZeroRedundancyOptimizer/test__optim_ZeroRedundancyOptimizer.py:225: in _run
    mp.spawn(
/usr/local/python3.11.14/lib/python3.11/site-packages/torch/multiprocessing/spawn.py:340: in spawn
    return start_processes(fn, args, nprocs, join, daemon, start_method="spawn")
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
/usr/local/python3.11.14/lib/python3.11/site-packages/torch/multiprocessing/spawn.py:296: in start_processes
    while not context.join():
              ^^^^^^^^^^^^^^
/usr/local/python3.11.14/lib/python3.11/site-packages/torch/multiprocessing/spawn.py:215: in join
    
---
test/_optim_ZeroRedundancyOptimizer/test__optim_ZeroRedundancyOptimizer.py:245: in test_join_device
    self._run(_test_join_device)
test/_optim_ZeroRedundancyOptimizer/test__optim_ZeroRedundancyOptimizer.py:225: in _run
    mp.spawn(
/usr/local/python3.11.14/lib/python3.11/site-packages/torch/multiprocessing/spawn.py:340: in spawn
    return start_processes(fn, args, nprocs, join, daemon, start_method="spawn")
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
/usr/local/python3.11.14/lib/python3.11/site-packages/torch/multiprocessing/spawn.py:296: in start_processes
    while not context.join():
              ^^^^^^^^^^^^^^
/usr/local/python3.11.14/lib/python3.11/site-packages/torch/multiprocessing/spawn.py:215: in join
    raise ProcessRaisedException(msg, 
---
test/_optim_ZeroRedundancyOptimizer/test__optim_ZeroRedundancyOptimizer.py:250: in test_join_hook
    self._run(_test_join_hook)
test/_optim_ZeroRedundancyOptimizer/test__optim_ZeroRedundancyOptimizer.py:225: in _run
    mp.spawn(
/usr/local/python3.11.14/lib/python3.11/site-packages/torch/multiprocessing/spawn.py:340: in spawn
    return start_processes(fn, args, nprocs, join, daemon, start_method="spawn")
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
/usr/local/python3.11.14/lib/python3.11/site-packages/torch/multiprocessing/spawn.py:296: in start_processes
    while not context.join():
              ^^^^^^^^^^^^^^
/usr/local/python3.11.14/lib/python3.11/site-packages/torch/multiprocessing/spawn.py:215: in join
    raise ProcessRaisedException(msg, erro
---
test/_optim_ZeroRedundancyOptimizer/test__optim_ZeroRedundancyOptimizer.py:255: in test_join_process_group
    self._run(_test_join_process_group)
test/_optim_ZeroRedundancyOptimizer/test__optim_ZeroRedundancyOptimizer.py:225: in _run
    mp.spawn(
/usr/local/python3.11.14/lib/python3.11/site-packages/torch/multiprocessing/spawn.py:340: in spawn
    return start_processes(fn, args, nprocs, join, daemon, start_method="spawn")
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
/usr/local/python3.11.14/lib/python3.11/site-packages/torch/multiprocessing/spawn.py:296: in start_processes
    while not context.join():
       
... (截断，共 6758 字符)
```

### ❌ test/checkpoint_SavePlan/test_checkpoint_SavePlan.py

- **状态**：FAIL
- **耗时**：56.2s
- **通过/失败/跳过**：0/4/0

```
【失败用例】
FAILED test/checkpoint_SavePlan/test_checkpoint_SavePlan.py::TestSavePlan::test_parameter_types
FAILED test/checkpoint_SavePlan/test_checkpoint_SavePlan.py::TestSavePlan::test_save_plan_attributes
FAILED test/checkpoint_SavePlan/test_checkpoint_SavePlan.py::TestSavePlan::test_save_plan_creation
FAILED test/checkpoint_SavePlan/test_checkpoint_SavePlan.py::TestSavePlan::test_save_plan_multiprocess

【错误详情】
test/checkpoint_SavePlan/test_checkpoint_SavePlan.py:115: in test_parameter_types
    items2 = [WriteItem(index=0, item="data")]
              ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
E   TypeError: WriteItem.__init__() got an unexpected keyword argument 'item'
---
test/checkpoint_SavePlan/test_checkpoint_SavePlan.py:98: in test_save_plan_attributes
    WriteItem(index=0, item="item_0"),
    ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
E   TypeError: WriteItem.__init__() got an unexpected keyword argument 'item'
---
test/checkpoint_SavePlan/test_checkpoint_SavePlan.py:82: in test_save_plan_creation
    plan1 = SavePlan(plan_items=[])
            ^^^^^^^^^^^^^^^^^^^^^^^
E   TypeError: SavePlan.__init__() got an unexpected keyword argument 'plan_items'
---
test/checkpoint_SavePlan/test_checkpoint_SavePlan.py:158: in test_save_plan_multiprocess
    self.assertEqual(results[rank].get('created'), 'SavePlan')
/usr/local/python3.11.14/lib/python3.11/site-packages/torch_npu/testing/testcase.py:382: in assertEqual
    _assertEqual(x, y, prec=prec, message=message, allow_inf=allow_inf, exact_dtype=exact_dtype)
/usr/local/python3.11.14/lib/python3.11/site-packages/torch_npu/testing/testcase.py:380: in _assertEqual
    super(TestCase, self).assertEqual(x, y, message)
E   AssertionError: None != 'SavePlan' :
=========================== short test summary info ============================
FAILED test/checkpoint_SavePlan/test_checkpoint_SavePlan.py::TestSavePlan::test_parameter_types
FAILED test/checkpoint_SavePlan/test_checkpoint_SavePlan.py::TestSavePl
---
```

### ❌ test/distributed_c10d_ProcessGroupXCCL/test_distributed_c10d_ProcessGroupXCCL.py

- **状态**：FAIL
- **耗时**：87.0s
- **通过/失败/跳过**：4/1/0

```
【失败用例】
FAILED test/distributed_c10d_ProcessGroupXCCL/test_distributed_c10d_ProcessGroupXCCL.py::TestProcessGroupXCCL::test_xccl_process_group_creation

【错误详情】
test/distributed_c10d_ProcessGroupXCCL/test_distributed_c10d_ProcessGroupXCCL.py:152: in test_xccl_process_group_creation
    self.fail(f"Rank {rank} raised: {r['error']}")
E   AssertionError: Rank 0 raised: create_config:build/CMakeFiles/torch_npu.dir/compiler_depend.ts:130 HCCL function error: hcclCommInitRootInfoConfig(numRanks, &rootInfo, rank, config, &(comm->hcclComm_)), error code is 7
E   [ERROR] 2026-04-28-16:56:53 (PID:3354790, Device:0, RankID:-1) ERR02200 DIST call hccl api failed.
E   [PID: 3354790] 2026-04-28-16:56:52.897.463 Communication_Error_Bind_IP_Port(EJ0003): Failed to bind the IP port. Reason: The IP address and port have been bound already.
=========================== short test summary info ============================
FAILED test/distributed_c10d_ProcessGroupXCCL/
---
```

### ❌ test/distributed_c10d__new_process_group_helper/test_distributed_c10d__new_process_group_helper.py

- **状态**：FAIL
- **耗时**：29.7s
- **通过/失败/跳过**：3/1/0

```
【失败用例】
FAILED test/distributed_c10d__new_process_group_helper/test_distributed_c10d__new_process_group_helper.py::TestNewProcessGroupHelper::test_create_process_group

【错误详情】
test/distributed_c10d__new_process_group_helper/test_distributed_c10d__new_process_group_helper.py:92: in test_create_process_group
    self.assertTrue(expected.issubset(params))
E   AssertionError: False is not true
=========================== short test summary info ============================
FAILED test/distributed_c10d__new_process_group_helper/test_distributed_c10d__new_process_group_helper.py::TestNewProcessGroupHelper::test_create_process_group
========================= 1 failed, 3 passed in 24.67s =========================

---
```

### ❌ test/distributed_c10d_new_subgroups/test_distributed_c10d_new_subgroups.py

- **状态**：FAIL
- **耗时**：157.0s
- **通过/失败/跳过**：0/8/0

```
【失败用例】
FAILED test/distributed_c10d_new_subgroups/test_distributed_c10d_new_subgroups.py::TestNewSubgroups::test_new_subgroups_all_params
FAILED test/distributed_c10d_new_subgroups/test_distributed_c10d_new_subgroups.py::TestNewSubgroups::test_new_subgroups_default
FAILED test/distributed_c10d_new_subgroups/test_distributed_c10d_new_subgroups.py::TestNewSubgroups::test_new_subgroups_subgroup_ops
FAILED test/distributed_c10d_new_subgroups/test_distributed_c10d_new_subgroups.py::TestNewSubgroups::test_new_subgroups_with_backend
FAILED test/distributed_c10d_new_subgroups/test_distributed_c10d_new_subgroups.py::TestNewSubgroups::test_new_subgroups_with_explicit_group
FAILED test/distributed_c10d_new_subgroups/test_distributed_c10d_new_subgroups.py::TestNewSubgroups::test_new_subgroups_with_group_desc
FAILED test/distributed_c10d_new_subgroups/test_distributed_c10d_new_subgroups.py::TestNewSubgroups::test_new_subgroups_with_group_size
FAILED test/distributed_c10d_new_subgroups/test_distributed_c10d_new_subgroups.py::TestNewSubgroups::test_new_subgroups_with_timeout

【错误详情】
test/distributed_c10d_new_subgroups/test_distributed_c10d_new_subgroups.py:270: in test_new_subgroups_all_params
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
E   -- Proces
---
test/distributed_c10d_new_subgroups/test_distributed_c10d_new_subgroups.py:204: in test_new_subgroups_default
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
E   -- Process 1
---
test/distributed_c10d_new_subgroups/test_distributed_c10d_new_subgroups.py:281: in test_new_subgroups_subgroup_ops
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
E   -- Proc
---
test/distributed_c10d_new_subgroups/test_distributed_c10d_new_subgroups.py:248: in test_new_subgroups_with_backend
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
E   -- Proc
---
test/distributed_c10d_new_subgroups/test_distributed_c10d_new_subgroups.py:226: in test_new_subgroups_with_explicit_group
    mp.spawn(
/usr/local/python3.11.14/lib/python3.11/site-packages/torch/multiprocessing/spawn.py:340: in spawn
    return start_processes(fn, args, nprocs, join, daemon, start_method="spawn")
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
/usr/local/python3.11.14/lib/python3.11/site-packages/torch/multiprocessing/spawn.py:296: in start_processes
    while not context.join():
              ^^^^^^^^^^^^^^
/usr/local/python3.11.14/lib/python3.11/site-packages/torch/multiprocessing/spawn.py:215: in join
    raise ProcessRaisedException
... (截断，共 7524 字符)
```

### ⏱ test/distributed_c10d_reinit_process_group/test_distributed_c10d_reinit_process_group.py

- **状态**：TIMEOUT
- **耗时**：200.4s
- **通过/失败/跳过**：0/0/0

```
超时（>200s），进程已终止
```

### ❌ test/distributed_c10d_split_group/test_distributed_c10d_split_group.py

- **状态**：FAIL
- **耗时**：107.2s
- **通过/失败/跳过**：0/6/0

```
【失败用例】
FAILED test/distributed_c10d_split_group/test_distributed_c10d_split_group.py::TestSplitGroup::test_split_group_all_params
FAILED test/distributed_c10d_split_group/test_distributed_c10d_split_group.py::TestSplitGroup::test_split_group_single_group
FAILED test/distributed_c10d_split_group/test_distributed_c10d_split_group.py::TestSplitGroup::test_split_group_with_default_parent
FAILED test/distributed_c10d_split_group/test_distributed_c10d_split_group.py::TestSplitGroup::test_split_group_with_explicit_parent
FAILED test/distributed_c10d_split_group/test_distributed_c10d_split_group.py::TestSplitGroup::test_split_group_with_group_desc
FAILED test/distributed_c10d_split_group/test_distributed_c10d_split_group.py::TestSplitGroup::test_split_group_with_timeout

【错误详情】
test/distributed_c10d_split_group/test_distributed_c10d_split_group.py:238: in test_split_group_all_params
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
E   -- Process 0 te
---
test/distributed_c10d_split_group/test_distributed_c10d_split_group.py:249: in test_split_group_single_group
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
E   -- Process 1 
---
test/distributed_c10d_split_group/test_distributed_c10d_split_group.py:194: in test_split_group_with_default_parent
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
E   -- Pro
---
test/distributed_c10d_split_group/test_distributed_c10d_split_group.py:205: in test_split_group_with_explicit_parent
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
E   -- Pr
---
test/distributed_c10d_split_group/test_distributed_c10d_split_group.py:227: in test_split_group_with_group_desc
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
E   -- Process
---
test/distributed_c10d_split_group/test_distributed_c10d_split_group.py:216: in test_split_group_with_timeout
    mp.spawn(
/usr/local/python3.11.14/lib/python3.11/site-packages/torch/multiproces
... (截断，共 5610 字符)
```

### ❌ test/fsdp_FSDPModule_methods/test_fsdp_FSDPModule_methods.py

- **状态**：FAIL
- **耗时**：145.9s
- **通过/失败/跳过**：0/7/0

```
【失败用例】
FAILED test/fsdp_FSDPModule_methods/test_fsdp_FSDPModule_methods.py::TestFSDPModuleMethods::test_set_all_reduce_hook
FAILED test/fsdp_FSDPModule_methods/test_fsdp_FSDPModule_methods.py::TestFSDPModuleMethods::test_set_is_last_backward
FAILED test/fsdp_FSDPModule_methods/test_fsdp_FSDPModule_methods.py::TestFSDPModuleMethods::test_set_post_optim_event
FAILED test/fsdp_FSDPModule_methods/test_fsdp_FSDPModule_methods.py::TestFSDPModuleMethods::test_set_reduce_scatter_divide_factor
FAILED test/fsdp_FSDPModule_methods/test_fsdp_FSDPModule_methods.py::TestFSDPModuleMethods::test_set_requires_all_reduce
FAILED test/fsdp_FSDPModule_methods/test_fsdp_FSDPModule_methods.py::TestFSDPModuleMethods::test_set_reshard_after_backward
FAILED test/fsdp_FSDPModule_methods/test_fsdp_FSDPModule_methods.py::TestFSDPModuleMethods::test_set_unshard_in_backward

【错误详情】
test/fsdp_FSDPModule_methods/test_fsdp_FSDPModule_methods.py:276: in test_set_all_reduce_hook
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
E   -- Process 0 terminated with
---
test/fsdp_FSDPModule_methods/test_fsdp_FSDPModule_methods.py:210: in test_set_is_last_backward
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
E   -- Process 0 terminated wit
---
test/fsdp_FSDPModule_methods/test_fsdp_FSDPModule_methods.py:265: in test_set_post_optim_event
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
E   -- Process 1 terminated wit
---
test/fsdp_FSDPModule_methods/test_fsdp_FSDPModule_methods.py:254: in test_set_reduce_scatter_divide_factor
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
E   -- Process 1 te
---
test/fsdp_FSDPModule_methods/test_fsdp_FSDPModule_methods.py:221: in test_set_requires_all_reduce
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
E   -- Process 0 terminated 
---
test/fsdp_FSDPModule_methods/test_fsdp_FSDPModule_methods.py:232: in test_set_reshard_after_backward
    mp.spa
... (截断，共 6498 字符)
```

### ❌ test/fsdp_FullyShardedDataParallel_methods/test_fsdp_FullyShardedDataParallel_methods.py

- **状态**：FAIL
- **耗时**：176.1s
- **通过/失败/跳过**：0/10/0

```
【失败用例】
FAILED test/fsdp_FullyShardedDataParallel_methods/test_fsdp_FullyShardedDataParallel_methods.py::TestFullyShardedDataParallelMethods::test_apply
FAILED test/fsdp_FullyShardedDataParallel_methods/test_fsdp_FullyShardedDataParallel_methods.py::TestFullyShardedDataParallelMethods::test_check_is_root
FAILED test/fsdp_FullyShardedDataParallel_methods/test_fsdp_FullyShardedDataParallel_methods.py::TestFullyShardedDataParallelMethods::test_flatten_sharded_optim_state_dict
FAILED test/fsdp_FullyShardedDataParallel_methods/test_fsdp_FullyShardedDataParallel_methods.py::TestFullyShardedDataParallelMethods::test_forward
FAILED test/fsdp_FullyShardedDataParallel_methods/test_fsdp_FullyShardedDataParallel_methods.py::TestFullyShardedDataParallelMethods::test_module_property
FAILED test/fsdp_FullyShardedDataParallel_methods/test_fsdp_FullyShardedDataParallel_methods.py::TestFullyShardedDataParallelMethods::test_named_buffers
FAILED test/fsdp_FullyShardedDataParallel_methods/test_fsdp_FullyShardedDataParallel_methods.py::TestFullyShardedDataParallelMethods::test_named_parameters
FAILED test/fsdp_FullyShardedDataParallel_methods/test_fsdp_FullyShardedDataParallel_methods.py::TestFullyShardedDataParallelMethods::test_no_sync
FAILED test/fsdp_FullyShardedDataParallel_methods/test_fsdp_FullyShardedDataParallel_methods.py::TestFullyShardedDataParallelMethods::test_register_comm_hook
FAILED test/fsdp_FullyShardedDataParallel_methods/test_fsdp_FullyShardedDataParallel_methods.py::TestFullyShardedDataParallelMethods::test_sharded_optim_state_dict

【错误详情】
test/fsdp_FullyShardedDataParallel_methods/test_fsdp_FullyShardedDataParallel_methods.py:277: in test_apply
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
E   -- Process 1 t
---
test/fsdp_FullyShardedDataParallel_methods/test_fsdp_FullyShardedDataParallel_methods.py:288: in test_check_is_root
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
E   -- Pro
---
test/fsdp_FullyShardedDataParallel_methods/test_fsdp_FullyShardedDataParallel_methods.py:299: in test_forward
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
E   -- Process 0
---
test/fsdp_FullyShardedDataParallel_methods/test_fsdp_FullyShardedDataParallel_methods.py:310: in test_module_property
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
E   -- P
---
test/fsdp_FullyShardedDataParallel_methods/test_fsdp_FullyShardedDataParallel_methods.py:321: in test_named_buffers
    mp.spawn(
/usr/local/python3.11.14/lib/python3.11/site-packages/torch/multiprocessing/spawn.py:
... (截断，共 8809 字符)
```

### ❌ test/fsdp__fsdp_extensions_FSDPExtensions/test_fsdp__fsdp_extensions_FSDPExtensions.py

- **状态**：FAIL
- **耗时**：29.5s
- **通过/失败/跳过**：0/4/0

```
【失败用例】
FAILED test/fsdp__fsdp_extensions_FSDPExtensions/test_fsdp__fsdp_extensions_FSDPExtensions.py::TestFSDPExtensions::test_fsdp_extensions_attributes
FAILED test/fsdp__fsdp_extensions_FSDPExtensions/test_fsdp__fsdp_extensions_FSDPExtensions.py::TestFSDPExtensions::test_fsdp_extensions_creation
FAILED test/fsdp__fsdp_extensions_FSDPExtensions/test_fsdp__fsdp_extensions_FSDPExtensions.py::TestFSDPExtensions::test_fsdp_extensions_multiprocess
FAILED test/fsdp__fsdp_extensions_FSDPExtensions/test_fsdp__fsdp_extensions_FSDPExtensions.py::TestFSDPExtensions::test_parameter_types

【错误详情】
test/fsdp__fsdp_extensions_FSDPExtensions/test_fsdp__fsdp_extensions_FSDPExtensions.py:82: in test_fsdp_extensions_attributes
    ext = FSDPExtensions()
          ^^^^^^^^^^^^^^^^
E   TypeError: Can't instantiate abstract class FSDPExtensions with abstract methods all_gather_dtensor, chunk_dtensor, chunk_tensor, post_unflatten_transform, pre_flatten_transform, pre_load_state_dict_transform
---
test/fsdp__fsdp_extensions_FSDPExtensions/test_fsdp__fsdp_extensions_FSDPExtensions.py:73: in test_fsdp_extensions_creation
    ext = FSDPExtensions()
          ^^^^^^^^^^^^^^^^
E   TypeError: Can't instantiate abstract class FSDPExtensions with abstract methods all_gather_dtensor, chunk_dtensor, chunk_tensor, post_unflatten_transform, pre_flatten_transform, pre_load_state_dict_transform
---
test/fsdp__fsdp_extensions_FSDPExtensions/test_fsdp__fsdp_extensions_FSDPExtensions.py:129: in test_fsdp_extensions_multiprocess
    self.assertEqual(results[rank].get('created'), 'FSDPExtensions')
/usr/local/python3.11.14/lib/python3.11/site-packages/torch_npu/testing/testcase.py:382: in assertEqual
    _assertEqual(x, y, prec=prec, message=message, allow_inf=allow_inf, exact_dtype=exact_dtype)
/usr/local/python3.11.14/lib/python3.11/site-packages/torch_npu/testing/testcase.py:380: in _assertEqual
    super(TestCase, self).assertEqual(x, y, message)
E   AssertionError: None != 'FSDPExtensions' :
---
test/fsdp__fsdp_extensions_FSDPExtensions/test_fsdp__fsdp_extensions_FSDPExtensions.py:94: in test_parameter_types
    ext = FSDPExtensions()
          ^^^^^^^^^^^^^^^^
E   TypeError: Can't instantiate abstract class FSDPExtensions with abstract methods all_gather_dtensor, chunk_dtensor, chunk_tensor, post_unflatten_transform, pre_flatten_transform, pre_load_state_dict_transform
=========================== short test summary info ============================
FAILED test/fsdp__fsdp_extensions_FSDPExtensions/test_fsdp__fsdp_extensions_FSDPExtensions.py::TestFSDPExtensions::test_fsdp_extensions_attributes
FAILED test/fsdp__fsdp_extensions_FSDPExtensions/test_fsdp__fsdp_extensions_FSDPExtensions.py::TestFSDPExtensions::test_fsdp_extensions_creation
FAILED test/fsdp__fsdp_extensions_FSDPExtensi
---
```

### ❌ test/fsdp__fsdp_extensions__ext_chunk_tensor/test_fsdp__fsdp_extensions__ext_chunk_tensor.py

- **状态**：FAIL
- **耗时**：56.3s
- **通过/失败/跳过**：0/4/0

```
【失败用例】
FAILED test/fsdp__fsdp_extensions__ext_chunk_tensor/test_fsdp__fsdp_extensions__ext_chunk_tensor.py::TestExtChunkTensor::test_chunk_tensor_basic
FAILED test/fsdp__fsdp_extensions__ext_chunk_tensor/test_fsdp__fsdp_extensions__ext_chunk_tensor.py::TestExtChunkTensor::test_multiprocess_chunking
FAILED test/fsdp__fsdp_extensions__ext_chunk_tensor/test_fsdp__fsdp_extensions__ext_chunk_tensor.py::TestExtChunkTensor::test_parameter_types
FAILED test/fsdp__fsdp_extensions__ext_chunk_tensor/test_fsdp__fsdp_extensions__ext_chunk_tensor.py::TestExtChunkTensor::test_return_tensor_properties

【错误详情】
test/fsdp__fsdp_extensions__ext_chunk_tensor/test_fsdp__fsdp_extensions__ext_chunk_tensor.py:82: in test_chunk_tensor_basic
    result = _ext_chunk_tensor(tensor, chunk_size, 0, 2)
             ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
E   TypeError: _ext_chunk_tensor() missing 1 required positional argument: 'pg'
---
test/fsdp__fsdp_extensions__ext_chunk_tensor/test_fsdp__fsdp_extensions__ext_chunk_tensor.py:146: in test_multiprocess_chunking
    self.assertTrue(results[rank].get('chunked'))
E   AssertionError: None is not true
---
test/fsdp__fsdp_extensions__ext_chunk_tensor/test_fsdp__fsdp_extensions__ext_chunk_tensor.py:96: in test_parameter_types
    result = _ext_chunk_tensor(tensor, chunk_size, rank, world_size)
             ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
E   TypeError: _ext_chunk_tensor() missing 1 required positional argument: 'pg'
---
test/fsdp__fsdp_extensions__ext_chunk_tensor/test_fsdp__fsdp_extensions__ext_chunk_tensor.py:108: in test_return_tensor_properties
    result = _ext_chunk_tensor(tensor, chunk_size, 0, 2)
             ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
E   TypeError: _ext_chunk_tensor() missing 1 required positional argument: 'pg'
=========================== short test summary info ============================
FAILED test/fsdp__fsdp_extensions__ext_chunk_tensor/test_fsdp__fsdp_extensions__ext_chunk_tensor.py::TestExtChunkTensor::test_chunk_tensor_basic
FAILED test/fsdp__fsdp_extensions__ext_chunk_tensor/test_fsdp__fsdp_extensions__ext_chunk_tensor.py::TestExtChunkTensor::test_multiprocess_chunking
FAILED test/fsdp__fsdp_extensions__ext_chunk_tensor/test_fsdp__fsdp_extensions__ext_chunk_tensor.py::
---
```

### ❌ test/fsdp__fsdp_extensions__ext_post_unflatten_transform/test_fsdp__fsdp_extensions__ext_post_unflatten_transform.py

- **状态**：FAIL
- **耗时**：30.1s
- **通过/失败/跳过**：3/1/0

```
【失败用例】
FAILED test/fsdp__fsdp_extensions__ext_post_unflatten_transform/test_fsdp__fsdp_extensions__ext_post_unflatten_transform.py::TestExtPostUnflattenTransform::test_transform_execution

【错误详情】
test/fsdp__fsdp_extensions__ext_post_unflatten_transform/test_fsdp__fsdp_extensions__ext_post_unflatten_transform.py:85: in test_transform_execution
    self.assertIsNotNone(result) or self.assertIsNone(result)
                                    ^^^^^^^^^^^^^^^^^^^^^^^^^
E   AssertionError: {} is not None
=========================== short test summary info ============================
FAILED test/fsdp__fsdp_extensions__ext_post_unflatten_transform/test_fsdp__fsdp_extensions__ext_post_unflatten_transform.py::TestExtPostUnflattenTransform::test_transform_execution
========================= 1 failed, 3 passed in 24.90s =========================

---
```

### ❌ test/fsdp__fsdp_extensions__ext_pre_flatten_transform/test_fsdp__fsdp_extensions__ext_pre_flatten_transform.py

- **状态**：FAIL
- **耗时**：56.8s
- **通过/失败/跳过**：1/3/0

```
【失败用例】
FAILED test/fsdp__fsdp_extensions__ext_pre_flatten_transform/test_fsdp__fsdp_extensions__ext_pre_flatten_transform.py::TestExtPreFlattenTransform::test_multiprocess_transform
FAILED test/fsdp__fsdp_extensions__ext_pre_flatten_transform/test_fsdp__fsdp_extensions__ext_pre_flatten_transform.py::TestExtPreFlattenTransform::test_return_type
FAILED test/fsdp__fsdp_extensions__ext_pre_flatten_transform/test_fsdp__fsdp_extensions__ext_pre_flatten_transform.py::TestExtPreFlattenTransform::test_transform_execution

【错误详情】
test/fsdp__fsdp_extensions__ext_pre_flatten_transform/test_fsdp__fsdp_extensions__ext_pre_flatten_transform.py:143: in test_multiprocess_transform
    self.assertTrue(results[rank].get('transformed'))
E   AssertionError: None is not true
---
test/fsdp__fsdp_extensions__ext_pre_flatten_transform/test_fsdp__fsdp_extensions__ext_pre_flatten_transform.py:106: in test_return_type
    result = _ext_pre_flatten_transform(module, None, None)
             ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
E   TypeError: _ext_pre_flatten_transform() takes from 1 to 2 positional arguments but 3 were given
---
test/fsdp__fsdp_extensions__ext_pre_flatten_transform/test_fsdp__fsdp_extensions__ext_pre_flatten_transform.py:83: in test_transform_execution
    result = _ext_pre_flatten_transform(module, fsdp_state, config)
             ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
E   TypeError: _ext_pre_flatten_transform() takes from 1 to 2 positional arguments but 3 were given
=========================== short test summary info ============================
FAILED test/fsdp__fsdp_extensions__ext_pre_flatten_transform/test_fsdp__fsdp_extensions__ext_pre_flatten_transform.py::TestExtPreFlattenTransform::test_multiprocess_transform
FAILED test/fsdp__fsdp_extensions__ext_pre_flatten_transform/test_fsdp__fsdp_extensions__ext_pre_flatten_transform.py::TestExtPreFlattenTransform::test_return_type
F
---
```

### ❌ test/fsdp__fsdp_extensions__ext_pre_load_state_dict_transform/test_fsdp__fsdp_extensions__ext_pre_load_state_dict_transform.py

- **状态**：FAIL
- **耗时**：57.0s
- **通过/失败/跳过**：1/3/0

```
【失败用例】
FAILED test/fsdp__fsdp_extensions__ext_pre_load_state_dict_transform/test_fsdp__fsdp_extensions__ext_pre_load_state_dict_transform.py::TestExtPreLoadStateDictTransform::test_multiprocess_transform
FAILED test/fsdp__fsdp_extensions__ext_pre_load_state_dict_transform/test_fsdp__fsdp_extensions__ext_pre_load_state_dict_transform.py::TestExtPreLoadStateDictTransform::test_return_type
FAILED test/fsdp__fsdp_extensions__ext_pre_load_state_dict_transform/test_fsdp__fsdp_extensions__ext_pre_load_state_dict_transform.py::TestExtPreLoadStateDictTransform::test_transform_execution

【错误详情】
test/fsdp__fsdp_extensions__ext_pre_load_state_dict_transform/test_fsdp__fsdp_extensions__ext_pre_load_state_dict_transform.py:143: in test_multiprocess_transform
    self.assertTrue(results[rank].get('transformed'))
E   AssertionError: None is not true
---
test/fsdp__fsdp_extensions__ext_pre_load_state_dict_transform/test_fsdp__fsdp_extensions__ext_pre_load_state_dict_transform.py:106: in test_return_type
    result = _ext_pre_load_state_dict_transform(state_dict, None, None)
             ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
E   TypeError: _ext_pre_load_state_dict_transform() takes from 1 to 2 positional arguments but 3 were given
---
test/fsdp__fsdp_extensions__ext_pre_load_state_dict_transform/test_fsdp__fsdp_extensions__ext_pre_load_state_dict_transform.py:83: in test_transform_execution
    result = _ext_pre_load_state_dict_transform(state_dict, fsdp_state, config)
             ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
E   TypeError: _ext_pre_load_state_dict_transform() takes from 1 to 2 positional arguments but 3 were given
=========================== short test summary info ============================
FAILED test/fsdp__fsdp_extensions__ext_pre_load_state_dict_transform/test_fsdp__fsdp_extensions__ext_pre_load_state_dict_transform.py::TestExtPreLoadStateDictTransform::test_multiprocess_transform
FAILED test/fsdp__fsdp_extensions__ext_pre_load_state_dict_transform/test_fsdp__fsdp_extensions
---
```

### ❌ test/new_subgroups_by_enumeration/test_new_subgroups_by_enumeration.py

- **状态**：FAIL
- **耗时**：55.9s
- **通过/失败/跳过**：1/3/0

```
【失败用例】
FAILED test/new_subgroups_by_enumeration/test_new_subgroups_by_enumeration.py::TestNewSubgroupsByEnumeration::test_multiprocess_subgroups
FAILED test/new_subgroups_by_enumeration/test_new_subgroups_by_enumeration.py::TestNewSubgroupsByEnumeration::test_parameter_types
FAILED test/new_subgroups_by_enumeration/test_new_subgroups_by_enumeration.py::TestNewSubgroupsByEnumeration::test_return_type

【错误详情】
test/new_subgroups_by_enumeration/test_new_subgroups_by_enumeration.py:121: in test_multiprocess_subgroups
    self.assertTrue(results[rank].get('subgroups_created'))
E   AssertionError: None is not true
---
test/new_subgroups_by_enumeration/test_new_subgroups_by_enumeration.py:80: in test_parameter_types
    self.assertIn('group_size', params)
E   AssertionError: 'group_size' not found in {'ranks_per_subgroup_list', 'pg_options', 'timeout', 'backend', 'group_desc'}
---
test/new_subgroups_by_enumeration/test_new_subgroups_by_enumeration.py:86: in test_return_type
    result = dist.new_subgroups_by_enumeration(group_size=1)
             ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
E   TypeError: new_subgroups_by_enumeration() got an unexpected keyword argument 'group_size'
=========================== short test summary info ============================
FAILED test/new_subgroups_by_enumeration/test_new_subgroups_by_enumeration.py::TestNewSubgroupsByEnumeration::test_multiprocess_subgroups
FAILED test/new_subgroups_by_enumeration/test_new_subgroups_by_enumeration.py::TestNewSubgroupsByEnumeration::test_parameter_types
FAILED test/new_subgroups_by_enumeration/test_new_subgroups_by_enumeration.py::TestNewSubgroupsByEnumeration::test_return_type
============
---
```

### ⏱ test/optim_PostLocalSGDOptimizer/test_optim_PostLocalSGDOptimizer.py

- **状态**：TIMEOUT
- **耗时**：200.3s
- **通过/失败/跳过**：0/0/0

```
超时（>200s），进程已终止
```

### ❌ test/optim_PostLocalSGDOptimizer_step/test_optim_PostLocalSGDOptimizer_step.py

- **状态**：FAIL
- **耗时**：100.5s
- **通过/失败/跳过**：0/5/0

```
【失败用例】
FAILED test/optim_PostLocalSGDOptimizer_step/test_optim_PostLocalSGDOptimizer_step.py::TestPostLocalSGDOptimizerStep::test_step_basic
FAILED test/optim_PostLocalSGDOptimizer_step/test_optim_PostLocalSGDOptimizer_step.py::TestPostLocalSGDOptimizerStep::test_step_increments_averager_step
FAILED test/optim_PostLocalSGDOptimizer_step/test_optim_PostLocalSGDOptimizer_step.py::TestPostLocalSGDOptimizerStep::test_step_multiple_iterations
FAILED test/optim_PostLocalSGDOptimizer_step/test_optim_PostLocalSGDOptimizer_step.py::TestPostLocalSGDOptimizerStep::test_step_returns_none
FAILED test/optim_PostLocalSGDOptimizer_step/test_optim_PostLocalSGDOptimizer_step.py::TestPostLocalSGDOptimizerStep::test_step_with_warmup

【错误详情】
test/optim_PostLocalSGDOptimizer_step/test_optim_PostLocalSGDOptimizer_step.py:149: in test_step_basic
    mp.spawn(_init_dist, args=(2, _test_step_basic), nprocs=2, join=True)
/usr/local/python3.11.14/lib/python3.11/site-packages/torch/multiprocessing/spawn.py:340: in spawn
    return start_processes(fn, args, nprocs, join, daemon, start_method="spawn")
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
/usr/local/python3.11.14/lib/python3.11/site-packages/torch/multiprocessing/spawn.py:296: in start_processes
    while not context.join():
              ^^^^^^^^^^^^^^
/usr/local/python3.11.14/lib/python3.11/site-packages/torch/multiprocessing/spawn.py:215: in join
    raise ProcessRaisedException(msg, error_index, failed_process.pid)
E   torch.multiprocessing
---
test/optim_PostLocalSGDOptimizer_step/test_optim_PostLocalSGDOptimizer_step.py:169: in test_step_increments_averager_step
    mp.spawn(_init_dist, args=(2, _test_step_averager_step_increments), nprocs=2, join=True)
/usr/local/python3.11.14/lib/python3.11/site-packages/torch/multiprocessing/spawn.py:340: in spawn
    return start_processes(fn, args, nprocs, join, daemon, start_method="spawn")
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
/usr/local/python3.11.14/lib/python3.11/site-packages/torch/multiprocessing/spawn.py:296: in start_processes
    while not context.join():
              ^^^^^^^^^^^^^^
/usr/local/python3.11.14/lib/python3.11/site-packages/torch/multiprocessing/spawn.py:215: in join
    raise ProcessRaisedException(msg, error_index, failed_
---
test/optim_PostLocalSGDOptimizer_step/test_optim_PostLocalSGDOptimizer_step.py:154: in test_step_multiple_iterations
    mp.spawn(_init_dist, args=(2, _test_step_multiple_iterations), nprocs=2, join=True)
/usr/local/python3.11.14/lib/python3.11/site-packages/torch/multiprocessing/spawn.py:340: in spawn
    return start_processes(fn, args, nprocs, join, daemon, start_method="spawn")
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
/usr/local/python3.11.14/lib/python3.11/site-packages/torch/multiprocessing/spawn.py:296: in start_processes
    while not context.join():
              ^^^^^^^^^^^^^^
/usr/local/python3.11.14/lib/python3.11/site-packages/torch/multiprocessing/spawn.py:215: in join
    raise ProcessRaisedException(msg, error_index, failed_process.pi
---
test/optim_PostLocalSGDOptimizer_step/test_optim_PostLocalSGDOptimizer_step.py:164: in test_step_returns_none
    mp.spawn(_init_dist, args=(2, _test_step_returns_none), nprocs=2, join=True)
/usr/local/python3.11.14/lib/python3.11/site-packages/torch/multiprocessing/spawn.py:340: in spawn
    return start_processes(fn, args, nprocs, join, daemon, start_method="spawn")
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
/usr/local/python3.11.14/lib/python3.11/site-packages/torch/multiprocessing/spawn.py:296: in start_processes
    while not context.join():
              ^^^^^^^^^^^^^^
/usr/local/python3.11.14/lib/python3.11/site-packages/torch/multiprocessing/spawn.py:215: in join
    raise ProcessRaisedException(msg, error_index, failed_process.pid)
E   torch.m
---
test/optim_PostLocalSGDOptimizer_step/test_optim_PostLocalSGDOptimizer_step.py:159: in test_step_with_warmup
    mp.spawn(_init_dist, args=(2, _test_step_with_warmup), nprocs=2, join=True)
/usr/local/python3.11.14/lib/python3.11/site-packages/torch/multiprocessing/spawn.py:340: in spawn
    return start_processes(fn, args, nprocs, join, daemon, start_method="spawn")
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
/usr/local/python3.11.14/lib/python3.11/site-packages/torch/multiprocessing/spawn.py:296: in start_processes
    while not context.join():
              ^^^^^^^^^^^^^^
/usr/local/python3.11.14/lib/python3.11/site-packages/torch/multiprocessing/spawn.py:215: in join
    raise ProcessRaisedException(msg, error_index, failed_process.pid)
E   torch.mul
---
```

### ❌ test/optim_ZeroRedundancyOptimizer_step/test_optim_ZeroRedundancyOptimizer_step.py

- **状态**：FAIL
- **耗时**：100.5s
- **通过/失败/跳过**：0/5/0

```
【失败用例】
FAILED test/optim_ZeroRedundancyOptimizer_step/test_optim_ZeroRedundancyOptimizer_step.py::TestZeroRedundancyOptimizerStep::test_step_basic
FAILED test/optim_ZeroRedundancyOptimizer_step/test_optim_ZeroRedundancyOptimizer_step.py::TestZeroRedundancyOptimizerStep::test_step_multiple_iterations
FAILED test/optim_ZeroRedundancyOptimizer_step/test_optim_ZeroRedundancyOptimizer_step.py::TestZeroRedundancyOptimizerStep::test_step_syncs_params_across_ranks
FAILED test/optim_ZeroRedundancyOptimizer_step/test_optim_ZeroRedundancyOptimizer_step.py::TestZeroRedundancyOptimizerStep::test_step_with_adam
FAILED test/optim_ZeroRedundancyOptimizer_step/test_optim_ZeroRedundancyOptimizer_step.py::TestZeroRedundancyOptimizerStep::test_step_with_closure

【错误详情】
test/optim_ZeroRedundancyOptimizer_step/test_optim_ZeroRedundancyOptimizer_step.py:162: in test_step_basic
    mp.spawn(_init_dist, args=(2, _test_step_basic), nprocs=2, join=True)
/usr/local/python3.11.14/lib/python3.11/site-packages/torch/multiprocessing/spawn.py:340: in spawn
    return start_processes(fn, args, nprocs, join, daemon, start_method="spawn")
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
/usr/local/python3.11.14/lib/python3.11/site-packages/torch/multiprocessing/spawn.py:296: in start_processes
    while not context.join():
              ^^^^^^^^^^^^^^
/usr/local/python3.11.14/lib/python3.11/site-packages/torch/multiprocessing/spawn.py:215: in join
    raise ProcessRaisedException(msg, error_index, failed_process.pid)
E   torch.multiproces
---
test/optim_ZeroRedundancyOptimizer_step/test_optim_ZeroRedundancyOptimizer_step.py:172: in test_step_multiple_iterations
    mp.spawn(_init_dist, args=(2, _test_step_multiple_iterations), nprocs=2, join=True)
/usr/local/python3.11.14/lib/python3.11/site-packages/torch/multiprocessing/spawn.py:340: in spawn
    return start_processes(fn, args, nprocs, join, daemon, start_method="spawn")
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
/usr/local/python3.11.14/lib/python3.11/site-packages/torch/multiprocessing/spawn.py:296: in start_processes
    while not context.join():
              ^^^^^^^^^^^^^^
/usr/local/python3.11.14/lib/python3.11/site-packages/torch/multiprocessing/spawn.py:215: in join
    raise ProcessRaisedException(msg, error_index, failed_proces
---
test/optim_ZeroRedundancyOptimizer_step/test_optim_ZeroRedundancyOptimizer_step.py:182: in test_step_syncs_params_across_ranks
    mp.spawn(_init_dist, args=(2, _test_step_syncs_params), nprocs=2, join=True)
/usr/local/python3.11.14/lib/python3.11/site-packages/torch/multiprocessing/spawn.py:340: in spawn
    return start_processes(fn, args, nprocs, join, daemon, start_method="spawn")
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
/usr/local/python3.11.14/lib/python3.11/site-packages/torch/multiprocessing/spawn.py:296: in start_processes
    while not context.join():
              ^^^^^^^^^^^^^^
/usr/local/python3.11.14/lib/python3.11/site-packages/torch/multiprocessing/spawn.py:215: in join
    raise ProcessRaisedException(msg, error_index, failed_process
---
test/optim_ZeroRedundancyOptimizer_step/test_optim_ZeroRedundancyOptimizer_step.py:177: in test_step_with_adam
    mp.spawn(_init_dist, args=(2, _test_step_with_adam), nprocs=2, join=True)
/usr/local/python3.11.14/lib/python3.11/site-packages/torch/multiprocessing/spawn.py:340: in spawn
    return start_processes(fn, args, nprocs, join, daemon, start_method="spawn")
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
/usr/local/python3.11.14/lib/python3.11/site-packages/torch/multiprocessing/spawn.py:296: in start_processes
    while not context.join():
              ^^^^^^^^^^^^^^
/usr/local/python3.11.14/lib/python3.11/site-packages/torch/multiprocessing/spawn.py:215: in join
    raise ProcessRaisedException(msg, error_index, failed_process.pid)
E   torch.mul
---
test/optim_ZeroRedundancyOptimizer_step/test_optim_ZeroRedundancyOptimizer_step.py:167: in test_step_with_closure
    mp.spawn(_init_dist, args=(2, _test_step_with_closure), nprocs=2, join=True)
/usr/local/python3.11.14/lib/python3.11/site-packages/torch/multiprocessing/spawn.py:340: in spawn
    return start_processes(fn, args, nprocs, join, daemon, start_method="spawn")
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
/usr/local/python3.11.14/lib/python3.11/site-packages/torch/multiprocessing/spawn.py:296: in start_processes
    while not context.join():
              ^^^^^^^^^^^^^^
/usr/local/python3.11.14/lib/python3.11/site-packages/torch/multiprocessing/spawn.py:215: in join
    raise ProcessRaisedException(msg, error_index, failed_process.pid)
E   tor
---
```

### ❌ test/split_with_sizes_copy/test_split_with_sizes_copy.py

- **状态**：FAIL
- **耗时**：10.2s
- **通过/失败/跳过**：0/0/0

### ❌ test/tensor_parallel_PrepareModuleOutput/test_tensor_parallel_PrepareModuleOutput.py

- **状态**：FAIL
- **耗时**：80.7s
- **通过/失败/跳过**：0/4/0

```
【失败用例】
FAILED test/tensor_parallel_PrepareModuleOutput/test_tensor_parallel_PrepareModuleOutput.py::TestPrepareModuleOutput::test_default_use_local_output
FAILED test/tensor_parallel_PrepareModuleOutput/test_tensor_parallel_PrepareModuleOutput.py::TestPrepareModuleOutput::test_mismatched_lengths
FAILED test/tensor_parallel_PrepareModuleOutput/test_tensor_parallel_PrepareModuleOutput.py::TestPrepareModuleOutput::test_single_placement
FAILED test/tensor_parallel_PrepareModuleOutput/test_tensor_parallel_PrepareModuleOutput.py::TestPrepareModuleOutput::test_tuple_placement

【错误详情】
test/tensor_parallel_PrepareModuleOutput/test_tensor_parallel_PrepareModuleOutput.py:161: in test_default_use_local_output
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
E  
---
test/tensor_parallel_PrepareModuleOutput/test_tensor_parallel_PrepareModuleOutput.py:172: in test_mismatched_lengths
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
E   -- Pr
---
test/tensor_parallel_PrepareModuleOutput/test_tensor_parallel_PrepareModuleOutput.py:139: in test_single_placement
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
E   -- Proc
---
test/tensor_parallel_PrepareModuleOutput/test_tensor_parallel_PrepareModuleOutput.py:150: in test_tuple_placement
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
E   -- Proce
---
```

### ❌ test/tensor_placement_types_Partial/test_tensor_placement_types_Partial.py

- **状态**：FAIL
- **耗时**：145.3s
- **通过/失败/跳过**：0/8/0

```
【失败用例】
FAILED test/tensor_placement_types_Partial/test_tensor_placement_types_Partial.py::TestPartial::test_partial_all_ops
FAILED test/tensor_placement_types_Partial/test_tensor_placement_types_Partial.py::TestPartial::test_partial_avg
FAILED test/tensor_placement_types_Partial/test_tensor_placement_types_Partial.py::TestPartial::test_partial_default
FAILED test/tensor_placement_types_Partial/test_tensor_placement_types_Partial.py::TestPartial::test_partial_linear_ops
FAILED test/tensor_placement_types_Partial/test_tensor_placement_types_Partial.py::TestPartial::test_partial_max
FAILED test/tensor_placement_types_Partial/test_tensor_placement_types_Partial.py::TestPartial::test_partial_min
FAILED test/tensor_placement_types_Partial/test_tensor_placement_types_Partial.py::TestPartial::test_partial_product
FAILED test/tensor_placement_types_Partial/test_tensor_placement_types_Partial.py::TestPartial::test_partial_sum

【错误详情】
test/tensor_placement_types_Partial/test_tensor_placement_types_Partial.py:243: in test_partial_all_ops
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
E   -- Process 0 termi
---
test/tensor_placement_types_Partial/test_tensor_placement_types_Partial.py:188: in test_partial_avg
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
E   -- Process 1 terminate
---
test/tensor_placement_types_Partial/test_tensor_placement_types_Partial.py:166: in test_partial_default
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
E   -- Process 0 termi
---
test/tensor_placement_types_Partial/test_tensor_placement_types_Partial.py:232: in test_partial_linear_ops
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
E   -- Process 1 te
---
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
E   -- Process 1 terminate
---
test/tensor_placement_types_Partial/t
... (截断，共 7377 字符)
```

### ❌ test/tensor_rand/test_tensor_rand.py

- **状态**：FAIL
- **耗时**：152.1s
- **通过/失败/跳过**：0/8/0

```
【失败用例】
FAILED test/tensor_rand/test_tensor_rand.py::TestDTensorRand::test_rand_1d - ...
FAILED test/tensor_rand/test_tensor_rand.py::TestDTensorRand::test_rand_dtype_float16
FAILED test/tensor_rand/test_tensor_rand.py::TestDTensorRand::test_rand_list_size
FAILED test/tensor_rand/test_tensor_rand.py::TestDTensorRand::test_rand_replicate
FAILED test/tensor_rand/test_tensor_rand.py::TestDTensorRand::test_rand_requires_grad
FAILED test/tensor_rand/test_tensor_rand.py::TestDTensorRand::test_rand_returns_dtensor_type
FAILED test/tensor_rand/test_tensor_rand.py::TestDTensorRand::test_rand_shard0
FAILED test/tensor_rand/test_tensor_rand.py::TestDTensorRand::test_rand_shard1

【错误详情】
test/tensor_rand/test_tensor_rand.py:167: in test_rand_1d
    mp.spawn(_init_dist, args=(2, _test_rand_1d), nprocs=2, join=True)
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
E   -- Proc
---
test/tensor_rand/test_tensor_rand.py:152: in test_rand_dtype_float16
    mp.spawn(_init_dist, args=(2, _test_rand_dtype_float16), nprocs=2, join=True)
/usr/local/python3.11.14/lib/python3.11/site-packages/torch/multiprocessing/spawn.py:340: in spawn
    return start_processes(fn, args, nprocs, join, daemon, start_method="spawn")
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
/usr/local/python3.11.14/lib/python3.11/site-packages/torch/multiprocessing/spawn.py:296: in start_processes
    while not context.join():
              ^^^^^^^^^^^^^^
/usr/local/python3.11.14/lib/python3.11/site-packages/torch/multiprocessing/spawn.py:215: in join
    raise ProcessRaisedException(msg, error_index, failed_process.pid)
E   torch.multiprocessing.spawn.ProcessRaisedExcept
---
test/tensor_rand/test_tensor_rand.py:162: in test_rand_list_size
    mp.spawn(_init_dist, args=(2, _test_rand_list_size), nprocs=2, join=True)
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
---
test/tensor_rand/test_tensor_rand.py:137: in test_rand_replicate
    mp.spawn(_init_dist, args=(2, _test_rand_replicate), nprocs=2, join=True)
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
---
test/tensor_rand/test_tensor_rand.py:157: in test_rand_requires_grad
    mp.spawn(_init_dist, args=(2, _test_rand_requires_grad), nprocs=2, join=True)
/usr/local/python3.11.14/lib/python3.11/site-packages/torch/multiprocessing/spawn.py:340: in spawn
    return start_processes(fn, args, nprocs, join, daemon, start_method="spawn")
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
/usr/local/python3.11.14/lib/python3.11/site-packages/torch/multiprocessing/spawn.py:296: in start_processes
    while not context.join():
              ^^^^^^^^^^^^^^
/usr/local/python3.11.14/lib/python3.11/site-packages/torch/multiprocessing/spawn.py:215: in join
    raise ProcessRaisedException(msg, error_index, failed_process.pid)
E   torch.multiprocessing.spawn.ProcessRaisedExcept
---
test/tensor_rand/test_tensor_rand.py:172: in test_rand_returns_dtensor_type
    mp.spawn(_init_dist, args=(2, _test_rand_returns_dtensor_type), nprocs=2, join=True)
/usr/local/python3.11.14/lib/python3.11/site-packages/torch/multiprocessing/spawn.py:340: in spawn
    return start_processes(f
... (截断，共 7122 字符)
```

### ❌ test/tensor_randn/test_tensor_randn.py

- **状态**：FAIL
- **耗时**：156.0s
- **通过/失败/跳过**：0/8/0

```
【失败用例】
FAILED test/tensor_randn/test_tensor_randn.py::TestDTensorRandn::test_randn_1d
FAILED test/tensor_randn/test_tensor_randn.py::TestDTensorRandn::test_randn_dtype_float16
FAILED test/tensor_randn/test_tensor_randn.py::TestDTensorRandn::test_randn_list_size
FAILED test/tensor_randn/test_tensor_randn.py::TestDTensorRandn::test_randn_local_tensor_on_npu
FAILED test/tensor_randn/test_tensor_randn.py::TestDTensorRandn::test_randn_replicate
FAILED test/tensor_randn/test_tensor_randn.py::TestDTensorRandn::test_randn_requires_grad
FAILED test/tensor_randn/test_tensor_randn.py::TestDTensorRandn::test_randn_shard0
FAILED test/tensor_randn/test_tensor_randn.py::TestDTensorRandn::test_randn_shard1

【错误详情】
test/tensor_randn/test_tensor_randn.py:165: in test_randn_1d
    mp.spawn(_init_dist, args=(2, _test_randn_1d), nprocs=2, join=True)
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
E   -- 
---
test/tensor_randn/test_tensor_randn.py:150: in test_randn_dtype_float16
    mp.spawn(_init_dist, args=(2, _test_randn_dtype_float16), nprocs=2, join=True)
/usr/local/python3.11.14/lib/python3.11/site-packages/torch/multiprocessing/spawn.py:340: in spawn
    return start_processes(fn, args, nprocs, join, daemon, start_method="spawn")
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
/usr/local/python3.11.14/lib/python3.11/site-packages/torch/multiprocessing/spawn.py:296: in start_processes
    while not context.join():
              ^^^^^^^^^^^^^^
/usr/local/python3.11.14/lib/python3.11/site-packages/torch/multiprocessing/spawn.py:215: in join
    raise ProcessRaisedException(msg, error_index, failed_process.pid)
E   torch.multiprocessing.spawn.ProcessRaisedEx
---
test/tensor_randn/test_tensor_randn.py:160: in test_randn_list_size
    mp.spawn(_init_dist, args=(2, _test_randn_list_size), nprocs=2, join=True)
/usr/local/python3.11.14/lib/python3.11/site-packages/torch/multiprocessing/spawn.py:340: in spawn
    return start_processes(fn, args, nprocs, join, daemon, start_method="spawn")
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
/usr/local/python3.11.14/lib/python3.11/site-packages/torch/multiprocessing/spawn.py:296: in start_processes
    while not context.join():
              ^^^^^^^^^^^^^^
/usr/local/python3.11.14/lib/python3.11/site-packages/torch/multiprocessing/spawn.py:215: in join
    raise ProcessRaisedException(msg, error_index, failed_process.pid)
E   torch.multiprocessing.spawn.ProcessRaisedException:
---
test/tensor_randn/test_tensor_randn.py:170: in test_randn_local_tensor_on_npu
    mp.spawn(_init_dist, args=(2, _test_randn_local_tensor_on_npu), nprocs=2, join=True)
/usr/local/python3.11.14/lib/python3.11/site-packages/torch/multiprocessing/spawn.py:340: in spawn
    return start_processes(fn, args, nprocs, join, daemon, start_method="spawn")
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
/usr/local/python3.11.14/lib/python3.11/site-packages/torch/multiprocessing/spawn.py:296: in start_processes
    while not context.join():
              ^^^^^^^^^^^^^^
/usr/local/python3.11.14/lib/python3.11/site-packages/torch/multiprocessing/spawn.py:215: in join
    raise ProcessRaisedException(msg, error_index, failed_process.pid)
E   torch.multiprocessing.spawn.Pro
---
test/tensor_randn/test_tensor_randn.py:135: in test_randn_replicate
    mp.spawn(_init_dist, args=(2, _test_randn_replicate), nprocs=2, join=True)
/usr/local/python3.11.14/lib/python3.11/site-packages/torch/multiprocessing/spawn.py:340: in spawn
    return start_processes(fn, args, nprocs, join, daemon, start_method="spawn")
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
/usr/local/python3.11.14/lib/python3.11/site-packages/torch/multiprocessing/spawn.py:296: in start_processes
    while not context.join():
              ^^^^^^^^^^^^^^
/usr/local/python3.11.14/lib/python3.11/site-packages/torch/multiprocessing/spawn.py:215: in join
    raise ProcessRaisedException(msg, error_index, failed_process.pid)
E   torch.multiprocessing.spawn.ProcessRaisedException:
---
test/tensor_randn/test_tensor_randn.py:155: in test_randn_requires_grad
    mp.spawn(_init_dist, args=(2, _test_randn_requires_grad), nprocs=2, join=True)
/usr/local/python3.11.14/lib/python3.11/site-packages/torch/multiprocessing/spawn.py:340: in spawn
    return st
... (截断，共 7147 字符)
```
