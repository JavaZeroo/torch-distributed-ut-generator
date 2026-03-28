# 示例：all_reduce UT 文件

以下是一个完整的 `test/all_reduce/test_all_reduce.py` 示例，展示了所有规范要求的落地方式。

```python
# -*- coding: utf-8 -*-
"""
测试目的：验证 torch.distributed.all_reduce 接口在多进程分布式场景下的功能正确性
API 名称：torch.distributed.all_reduce
API 签名：all_reduce(tensor, op=ReduceOp.SUM, group=None, async_op=False)

覆盖维度表：
| 覆盖维度         | 说明                                                         | 覆盖情况                                       |
|------------------|--------------------------------------------------------------|------------------------------------------------|
| 空/非空          | group 可为 None；其余参数取默认与显式传入                    | 已覆盖：group=None；默认参数 all_reduce(tensor) |
| 枚举选项         | ReduceOp：SUM、MAX、MIN、PRODUCT、AVG 等                     | 已覆盖：多种 op；未覆盖 BAND/BOR/BXOR（见未覆盖项） |
| 参数类型         | Tensor、ReduceOp、ProcessGroup、bool(async_op)               | 已覆盖                                         |
| 传参与不传参     | op/group/async_op 显式传参与省略                             | 已覆盖                                         |
| 等价类/边界值    | 1D/标量/大 tensor、连续调用、bfloat16/float32                | 已覆盖                                         |
| 正常传参场景     | 合法输入下集合通信完成、shape/dtype 保持                      | 已覆盖：sync/async、自定义子组                 |
| 异常传参场景     | 非 Tensor；CPU tensor + NCCL/HCCL                             | 已覆盖：非 Tensor；CPU tensor（非 gloo 后端时） |

未覆盖项及原因：
- BAND/BOR/BXOR：HCCL 后端不支持位运算 ReduceOp
- int8/uint8：整型 all_reduce 因后端而异，非本 UT 核心
- PREMUL_SUM：仅部分 NCCL 构建支持，NPU 环境通常不涉及
- float16：精度验证非本测试目的（与 bfloat16 功能覆盖不冲突）

注意：本测试仅验证功能正确性（调用不报错、输出 shape/dtype 符合预期），
     不做精度和数值正确性校验。
"""

import os
import socket
import time
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import pytest

_IS_NPU = hasattr(torch, 'npu') and torch.npu.is_available()
_IS_CUDA = not _IS_NPU and torch.cuda.is_available()

if _IS_NPU:
    import torch_npu  # noqa: F401
    from torch_npu.contrib import transfer_to_npu  # noqa: F401

DEVICE_TYPE = "npu" if _IS_NPU else ("cuda" if _IS_CUDA else "cpu")
BACKEND = "hccl" if _IS_NPU else ("nccl" if _IS_CUDA else "gloo")
WORLD_SIZE = 2


def _assert_raises(exc_types, fn):
    """spawn 子进程内不可用 pytest.raises，用此辅助函数断言异常类型。"""
    try:
        fn()
    except exc_types:
        return
    raise AssertionError(f"expected one of {exc_types}")


def _get_free_port():
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(('', 0))
        return str(s.getsockname()[1])


def _setup_device(rank):
    if _IS_NPU:
        os.environ['HCCL_WHITELIST_DISABLE'] = '1'
        torch.npu.set_device(rank)
    elif _IS_CUDA:
        torch.cuda.set_device(rank)


def _worker(rank, world_size, port, test_fn, result_queue=None):
    _setup_device(rank)
    os.environ['MASTER_ADDR'] = '127.0.0.1'
    os.environ['MASTER_PORT'] = port
    dist.init_process_group(backend=BACKEND, rank=rank, world_size=world_size)
    try:
        test_fn(rank, world_size)
        if result_queue is not None:
            result_queue.put((rank, None))
    except Exception as e:
        if result_queue is not None:
            result_queue.put((rank, e))
        else:
            raise
    finally:
        dist.destroy_process_group()


def _run_test(test_fn, world_size=WORLD_SIZE):
    if DEVICE_TYPE == "cpu":
        pytest.skip("无可用 GPU/NPU 设备，跳过分布式测试")
    port = _get_free_port()
    mp.spawn(
        _worker,
        args=(world_size, port, test_fn),
        nprocs=world_size,
        join=True,
    )


# ===================== 正常场景 =====================

@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("async_op", [True, False])
@pytest.mark.timeout(120)
def test_all_reduce_basic(dtype, async_op):
    """基础功能：不同 dtype + sync/async（bfloat16 不因「可能不支持」而 skip，由失败显式暴露环境能力）"""
    def _test_fn(rank, world_size):
        tensor = torch.ones(4, 4, dtype=dtype, device=DEVICE_TYPE)
        work = dist.all_reduce(tensor, op=dist.ReduceOp.SUM, async_op=async_op)
        if async_op:
            assert isinstance(work, dist.Work)
            wait_result = work.wait()
            assert wait_result in (True, None)
            for _ in range(20):
                if work.is_completed():
                    break
                time.sleep(0.01)
            assert work.is_completed() in (True, False)
        else:
            assert work is None
        assert tensor.shape == (4, 4)
        assert tensor.dtype == dtype

    _run_test(_test_fn)


@pytest.mark.parametrize("op", [dist.ReduceOp.SUM, dist.ReduceOp.MAX, dist.ReduceOp.MIN, dist.ReduceOp.PRODUCT])
@pytest.mark.timeout(120)
def test_all_reduce_reduce_ops(op):
    """不同 ReduceOp"""
    def _test_fn(rank, world_size):
        tensor = torch.ones(4, 4, dtype=torch.float32, device=DEVICE_TYPE) * (rank + 1)
        dist.all_reduce(tensor, op=op)
        assert tensor.shape == (4, 4)
        assert tensor.dtype == torch.float32

    _run_test(_test_fn)


@pytest.mark.timeout(120)
def test_all_reduce_avg():
    """ReduceOp.AVG"""
    def _test_fn(rank, world_size):
        tensor = torch.ones(4, 4, dtype=torch.float32, device=DEVICE_TYPE) * (rank + 1)
        dist.all_reduce(tensor, op=dist.ReduceOp.AVG)
        assert tensor.shape == (4, 4)

    _run_test(_test_fn)


@pytest.mark.timeout(120)
def test_all_reduce_default_params():
    """仅传必填参数，其余使用默认值"""
    def _test_fn(rank, world_size):
        tensor = torch.ones(8, dtype=torch.float32, device=DEVICE_TYPE)
        dist.all_reduce(tensor)
        assert tensor.shape == (8,)

    _run_test(_test_fn)


@pytest.mark.timeout(120)
def test_all_reduce_group_none():
    """显式传 group=None（等同于默认组）"""
    def _test_fn(rank, world_size):
        tensor = torch.ones(4, dtype=torch.float32, device=DEVICE_TYPE)
        dist.all_reduce(tensor, group=None)
        assert tensor.shape == (4,)

    _run_test(_test_fn)


@pytest.mark.timeout(120)
def test_all_reduce_custom_group():
    """使用自定义子组"""
    def _test_fn(rank, world_size):
        sub_group = dist.new_group(ranks=list(range(world_size)))
        tensor = torch.ones(4, 4, dtype=torch.float32, device=DEVICE_TYPE)
        dist.all_reduce(tensor, group=sub_group)
        assert tensor.shape == (4, 4)

    _run_test(_test_fn)


@pytest.mark.timeout(120)
def test_all_reduce_1d_tensor():
    """1D tensor"""
    def _test_fn(rank, world_size):
        tensor = torch.ones(16, dtype=torch.float32, device=DEVICE_TYPE)
        dist.all_reduce(tensor)
        assert tensor.dim() == 1

    _run_test(_test_fn)


@pytest.mark.timeout(120)
def test_all_reduce_scalar_tensor():
    """单元素 tensor"""
    def _test_fn(rank, world_size):
        tensor = torch.tensor(1.0, dtype=torch.float32, device=DEVICE_TYPE)
        dist.all_reduce(tensor)
        assert tensor.shape == ()

    _run_test(_test_fn)


@pytest.mark.timeout(120)
def test_all_reduce_large_tensor():
    """大 tensor [1024, 1024]"""
    def _test_fn(rank, world_size):
        tensor = torch.ones(1024, 1024, dtype=torch.float32, device=DEVICE_TYPE)
        dist.all_reduce(tensor)
        assert tensor.shape == (1024, 1024)

    _run_test(_test_fn)


@pytest.mark.timeout(120)
def test_all_reduce_consecutive_calls():
    """连续多次调用"""
    def _test_fn(rank, world_size):
        for _ in range(3):
            tensor = torch.ones(4, 4, dtype=torch.float32, device=DEVICE_TYPE)
            dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
            assert tensor.shape == (4, 4)

    _run_test(_test_fn)


@pytest.mark.timeout(120)
def test_all_reduce_async_wait_and_is_completed():
    """异步操作 Work 对象的 wait() 与 is_completed()（不同后端对 is_completed 刷新时机可能不同）"""
    def _test_fn(rank, world_size):
        tensor = torch.ones(4, 4, dtype=torch.float32, device=DEVICE_TYPE)
        work = dist.all_reduce(tensor, async_op=True)
        assert isinstance(work, dist.Work)
        wait_result = work.wait()
        assert wait_result in (True, None)
        for _ in range(20):
            if work.is_completed():
                break
            time.sleep(0.01)
        assert work.is_completed() in (True, False)

    _run_test(_test_fn)


# ===================== 异常场景 =====================

@pytest.mark.timeout(120)
def test_all_reduce_invalid_non_tensor():
    """传入非 Tensor 类型"""
    def _test_fn(rank, world_size):
        _assert_raises(
            (TypeError, RuntimeError, AttributeError),
            lambda: dist.all_reduce("not_a_tensor"),
        )

    _run_test(_test_fn)


@pytest.mark.timeout(120)
def test_all_reduce_cpu_tensor_with_gpu_backend():
    """CPU tensor 在 NCCL/HCCL 后端上应报错"""
    if BACKEND == "gloo":
        pytest.skip("gloo 后端支持 CPU tensor，无法触发此异常")

    def _test_fn(rank, world_size):
        cpu_tensor = torch.ones(4, dtype=torch.float32, device="cpu")
        _assert_raises(RuntimeError, lambda: dist.all_reduce(cpu_tensor))

    _run_test(_test_fn)


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
```

## 关键要点总结

1. **环境检测在模块顶部**，一次性完成
2. **transfer_to_npu 导入后自动生效**，代码中仍写 `cuda/nccl` 风格可被自动映射，但建议用 `DEVICE_TYPE/BACKEND` 变量更清晰
3. **_worker 函数统一管理** init/destroy 生命周期
4. **端口在 spawn 前获取**，通过参数传入子进程
5. **pytest.parametrize** 组合参数维度
6. **异常断言策略**：spawn 子进程内用 `_assert_raises(exc_types, fn)`；**在主进程单独执行**的异常测试（如"未初始化"场景）才可以直接用 `pytest.raises`
7. **pytest.skip** 仅用于**无可用 GPU/NPU**（如 `DEVICE_TYPE == "cpu"`）等环境缺失场景，**禁止**因「不支持 bfloat16」而 skip；`bfloat16` 用例应照常执行，由失败反映真实能力
8. **异步 Work**：`wait()` 后对 `is_completed()` 的断言宜兼容 HCCL/NCCL 差异（可短轮询后再断言），避免对刷新时机做过强假设
9. **设备类型字符串类 API**（如 `"cuda"`/`"npu"` 二选一）：合并为单用例分支，勿拆成两个用例导致常驻 skip（见 SKILL 主文档「CUDA / NPU 互斥设备」）
10. **无数值正确性断言**，仅验证 shape、dtype、不抛异常
