# -*- coding: utf-8 -*-
"""pytest 全局 fixture：每个测试方法前给 HCCL/MASTER 选用全新的端口段。

背景：
- 测试用例都是 `mp.spawn` 起子进程跑分布式集合通信。
- 上一次测试 `dist.destroy_process_group()` 之后，HCCL 内部通信端口
  仍处于占用/TIME_WAIT 状态一段时间（典型 60s）。
- 本机的 NPU 0~5 长期被其他用户进程占用，只有 NPU 6 / 7 空闲；
  在被占用的 NPU 上 HCCL 会触发 EJ0003 "Failed to bind the IP port"。
- run_tests.py 会一文件一个新 pytest 进程，每个 pytest 进程内的计数器
  又都从 0 开始，于是相邻 pytest 进程的第一个测试方法都会试图绑定
  同一个 HCCL_IF_BASE_PORT（旧版固定 50000），上一个进程刚释放的端口
  仍在 TIME_WAIT 时就会触发 EJ0003。

解决方法：
1. 用 ASCEND_RT_VISIBLE_DEVICES 把可见 NPU 限定到本机空闲的 6, 7，
   torch_npu.npu.set_device(0/1) 实际映射到物理 6/7。
2. 每个 pytest session 启动时通过 /tmp 下的持久计数器（带 flock）领取
   一段独立的 HCCL_IF_BASE_PORT 偏移，确保和最近退出的兄弟 pytest 进程
   端口段不冲突。
3. 在每个 test 方法之前：
   - 用 socket bind(0) 让内核挑一个空闲 TCP 端口写入 MASTER_PORT。
   - HCCL_IF_BASE_PORT 在 session 段内顺次偏移，每个方法 64 个端口。
4. mp.spawn(start_method='spawn') 会把父进程 env 完整拷给子进程。
5. 各 test 文件里的 `_init_dist_hccl` 已改成 setdefault，会保留这里设的值。
"""

import fcntl
import os
import socket

import pytest


# HCCL 端口区间设计 ----------------------------------------------------------
_HCCL_PORT_BASE = 50000               # 起始端口
_HCCL_PORT_RANGE = 12000              # 总跨度：50000-61999
_HCCL_PORT_PER_METHOD = 64            # 每个测试方法预留 64 个端口
_HCCL_METHODS_PER_SESSION = 16        # 每个 pytest session 预留 16 个方法的额度
_HCCL_BUDGET_PER_SESSION = (
    _HCCL_PORT_PER_METHOD * _HCCL_METHODS_PER_SESSION  # 1024 ports / session
)
_HCCL_SESSION_SLOTS = _HCCL_PORT_RANGE // _HCCL_BUDGET_PER_SESSION  # 11 slots

# 跨 pytest 进程持久化的 session 计数器
_PERSISTENT_COUNTER_FILE = os.path.join('/tmp', '_pytest_hccl_session_counter')


def _next_session_slot() -> int:
    """以 fcntl 互斥地推进 /tmp 下的持久计数器，返回本 session 的槽位编号。

    返回 [0, _HCCL_SESSION_SLOTS) 内的整数。每个 pytest 进程领到的槽位
    与上一个不同；wrap 周期为 _HCCL_SESSION_SLOTS 个 session（≈11 个，
    远长于 TIME_WAIT 60s 的回收时间），所以不会撞到仍在 TIME_WAIT 的端口。
    """
    try:
        with open(_PERSISTENT_COUNTER_FILE, 'a+') as f:
            fcntl.flock(f, fcntl.LOCK_EX)
            try:
                f.seek(0)
                content = f.read().strip()
                cur = int(content) if content else 0
                cur = cur % _HCCL_SESSION_SLOTS
                nxt = (cur + 1) % _HCCL_SESSION_SLOTS
                f.seek(0)
                f.truncate()
                f.write(str(nxt))
                return cur
            finally:
                fcntl.flock(f, fcntl.LOCK_UN)
    except (OSError, ValueError):
        # 兜底：用 PID 派生槽位
        return os.getpid() % _HCCL_SESSION_SLOTS


_SESSION_SLOT = _next_session_slot()
_SESSION_HCCL_BASE = _HCCL_PORT_BASE + _SESSION_SLOT * _HCCL_BUDGET_PER_SESSION
_method_counter = {'value': 0}

# NPU 可见设备由 run_tests.py 在 spawn 子进程前注入到 env，
# 这里不再设置，避免在 torch_npu 已 import 后才生效导致 device_count 错乱。


def _pick_free_port() -> int:
    """让内核选一个空闲端口（bind to 0），立即关闭 socket 后返回该端口号。"""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(('', 0))
        return s.getsockname()[1]


def _next_hccl_base_port() -> int:
    """返回本 session 内下一个测试方法的 HCCL_IF_BASE_PORT。

    超出 _HCCL_METHODS_PER_SESSION 时回卷到本 session 段内的起点；
    因为是同一进程内回卷，端口已被本 session 自己释放并经过若干秒，
    通常已脱离 TIME_WAIT。
    """
    cur = _method_counter['value']
    _method_counter['value'] += 1
    cur = cur % _HCCL_METHODS_PER_SESSION
    return _SESSION_HCCL_BASE + cur * _HCCL_PORT_PER_METHOD


@pytest.fixture(autouse=True)
def _fresh_distributed_ports_per_test():
    """每个测试方法运行前刷新分布式相关端口环境变量。"""
    os.environ['MASTER_ADDR'] = os.environ.get('MASTER_ADDR', 'localhost')
    os.environ['MASTER_PORT'] = str(_pick_free_port())
    os.environ['HCCL_IF_BASE_PORT'] = str(_next_hccl_base_port())
    yield
