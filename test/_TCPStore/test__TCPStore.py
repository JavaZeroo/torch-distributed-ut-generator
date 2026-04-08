# -*- coding: utf-8 -*-
"""
测试目的：验证 torch.distributed.TCPStore.__init__、host、port、libuvBackend 接口功能正确性
API 名称：
  - torch.distributed.TCPStore.__init__
  - torch.distributed.TCPStore.host
  - torch.distributed.TCPStore.port
  - torch.distributed.TCPStore.libuvBackend
API 签名：
  - TCPStore.__init__(self, host_name: str, port: int, world_size: Optional[int] = None,
                      is_master: bool = False, timeout: timedelta = timedelta(seconds=300),
                      wait_for_workers: bool = True, multi_tenant: bool = False,
                      master_listen_fd: Optional[int] = None, use_libuv: bool = True) -> None
  - TCPStore.host        -> str  (property)
  - TCPStore.port        -> int  (property)
  - TCPStore.libuvBackend -> bool (property)

覆盖维度表：
| 覆盖维度         | 说明                                                         | 覆盖情况                                       |
|------------------|--------------------------------------------------------------|------------------------------------------------|
| 空/非空          | host_name 为 "localhost"、"127.0.0.1"                        | 已覆盖：test_init_localhost / test_init_ipv4   |
| 枚举选项         | use_libuv=True vs False；is_master=True                      | 已覆盖：test_libuv_variants                    |
| 参数类型         | host_name: str；port: int；world_size: Optional[int]         | 已覆盖                                         |
| 传参与不传参     | 省略 world_size / timeout 使用默认值 vs 显式传入             | 已覆盖                                         |
| 等价类/边界值    | world_size=None、world_size=1；timeout 不同值                | 已覆盖                                         |
| 正常传参场景     | 创建 master TCPStore 并读取 host/port/libuvBackend 属性      | 已覆盖                                         |
| 异常传参场景     | 无效端口（负数）                                             | 已覆盖：test_invalid_port_raises               |

未覆盖项及原因：
- 多进程 client/server 模式：需要多进程协调，基本属性测试已用 is_master=True + wait_for_workers=False 覆盖

注意：本测试仅验证功能正确性（调用不报错、返回值类型符合预期），
     不做精度和数值正确性校验。
"""

import datetime
import socket
import unittest
import torch
import torch_npu  # noqa: F401 — registers NPU backend

try:
    from torch_npu.testing.testcase import TestCase, run_tests
except ImportError:
    import sys
    from unittest import TestCase

    def run_tests() -> None:
        unittest.main(argv=sys.argv)

from torch.distributed import TCPStore


def _find_free_port():
    """Return a free TCP port on localhost."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(('', 0))
        return s.getsockname()[1]


class TestTCPStore(TestCase):
    """Test cases for torch.distributed.TCPStore.__init__, host, port, libuvBackend."""

    def setUp(self):
        super().setUp()
        device_name = torch._C._get_privateuse1_backend_name()
        self.assertEqual(device_name, 'npu', f"Expected device 'npu', got '{device_name}'")

    # ------------------------------------------------------------------
    # __init__ tests (master-only, wait_for_workers=False to avoid blocking)
    # ------------------------------------------------------------------

    def test_init_localhost(self):
        """TCPStore can be created as master on localhost."""
        port = _find_free_port()
        store = TCPStore(
            host_name="localhost",
            port=port,
            world_size=1,
            is_master=True,
            wait_for_workers=False,
        )
        self.assertIsNotNone(store)

    def test_init_ipv4(self):
        """TCPStore accepts 127.0.0.1 as host_name."""
        port = _find_free_port()
        store = TCPStore(
            host_name="127.0.0.1",
            port=port,
            world_size=1,
            is_master=True,
            wait_for_workers=False,
        )
        self.assertIsNotNone(store)

    def test_init_world_size_none(self):
        """TCPStore with world_size=None."""
        port = _find_free_port()
        store = TCPStore(
            host_name="localhost",
            port=port,
            world_size=None,
            is_master=True,
            wait_for_workers=False,
        )
        self.assertIsNotNone(store)

    def test_init_with_timeout(self):
        """TCPStore with explicit timeout parameter."""
        port = _find_free_port()
        store = TCPStore(
            host_name="localhost",
            port=port,
            world_size=1,
            is_master=True,
            timeout=datetime.timedelta(seconds=60),
            wait_for_workers=False,
        )
        self.assertIsNotNone(store)

    def test_libuv_variants(self):
        """TCPStore with use_libuv=True and use_libuv=False."""
        for use_libuv in (True, False):
            port = _find_free_port()
            store = TCPStore(
                host_name="localhost",
                port=port,
                world_size=1,
                is_master=True,
                wait_for_workers=False,
                use_libuv=use_libuv,
            )
            self.assertIsNotNone(store, f"Failed with use_libuv={use_libuv}")

    def test_invalid_port_raises(self):
        """TCPStore with an invalid negative port raises an exception."""
        raised = False
        try:
            TCPStore(
                host_name="localhost",
                port=-1,
                world_size=1,
                is_master=True,
                wait_for_workers=False,
            )
        except Exception:
            raised = True
        self.assertTrue(raised, "Expected exception for port=-1")

    # ------------------------------------------------------------------
    # host property
    # ------------------------------------------------------------------

    def test_host_returns_str(self):
        """TCPStore.host returns a str."""
        port = _find_free_port()
        store = TCPStore("localhost", port, 1, is_master=True, wait_for_workers=False)
        self.assertIsInstance(store.host, str)

    def test_host_value(self):
        """TCPStore.host matches the host_name passed at construction."""
        port = _find_free_port()
        store = TCPStore("localhost", port, 1, is_master=True, wait_for_workers=False)
        self.assertEqual(store.host, "localhost")

    # ------------------------------------------------------------------
    # port property
    # ------------------------------------------------------------------

    def test_port_returns_int(self):
        """TCPStore.port returns an int."""
        port = _find_free_port()
        store = TCPStore("localhost", port, 1, is_master=True, wait_for_workers=False)
        self.assertIsInstance(store.port, int)

    def test_port_value(self):
        """TCPStore.port matches the port passed at construction."""
        port = _find_free_port()
        store = TCPStore("localhost", port, 1, is_master=True, wait_for_workers=False)
        self.assertEqual(store.port, port)

    # ------------------------------------------------------------------
    # libuvBackend property
    # ------------------------------------------------------------------

    def test_libuvbackend_returns_bool(self):
        """TCPStore.libuvBackend returns a bool."""
        port = _find_free_port()
        store = TCPStore("localhost", port, 1, is_master=True, wait_for_workers=False)
        self.assertIsInstance(store.libuvBackend, bool)

    def test_libuvbackend_true_when_use_libuv(self):
        """libuvBackend is True when use_libuv=True at construction."""
        port = _find_free_port()
        store = TCPStore(
            "localhost", port, 1,
            is_master=True, wait_for_workers=False, use_libuv=True
        )
        self.assertTrue(store.libuvBackend)

    def test_libuvbackend_false_when_not_libuv(self):
        """libuvBackend is False when use_libuv=False at construction."""
        port = _find_free_port()
        store = TCPStore(
            "localhost", port, 1,
            is_master=True, wait_for_workers=False, use_libuv=False
        )
        self.assertFalse(store.libuvBackend)


if __name__ == "__main__":
    run_tests()
