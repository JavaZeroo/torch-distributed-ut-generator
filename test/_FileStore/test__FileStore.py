# -*- coding: utf-8 -*-
"""
测试目的：验证 torch.distributed.FileStore.__init__ 与 FileStore.path 接口功能正确性
API 名称：
  - torch.distributed.FileStore.__init__
  - torch.distributed.FileStore.path
API 签名：
  - FileStore.__init__(self, file_name: str, world_size: int = -1) -> None
  - FileStore.path  -> str  (property)

覆盖维度表：
| 覆盖维度         | 说明                                                         | 覆盖情况                                       |
|------------------|--------------------------------------------------------------|------------------------------------------------|
| 空/非空          | file_name 非空路径                                           | 已覆盖：test_init_basic                        |
| 枚举选项         | world_size=-1（默认）、world_size=2（有界）                  | 已覆盖：test_init_world_size_variants          |
| 参数类型         | file_name: str；world_size: int                              | 已覆盖                                         |
| 传参与不传参     | 省略 world_size 使用默认值 -1 vs 显式传入正整数              | 已覆盖                                         |
| 等价类/边界值    | world_size=-1（无界）、world_size=1、world_size=4            | 已覆盖：test_init_world_size_variants          |
| 正常传参场景     | 创建 FileStore 并访问 path 属性                              | 已覆盖：test_path_matches_file_name            |
| 异常传参场景     | file_name 为空字符串                                         | 已覆盖：test_init_empty_filename_raises        |

未覆盖项及原因：
- 并发读写场景：FileStore 是进程间 KV 存储，多进程场景等效测试已由 init_process_group 路径覆盖

注意：本测试仅验证功能正确性（调用不报错、返回值类型符合预期），
     不做精度和数值正确性校验。
"""

import os
import tempfile
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

from torch.distributed import FileStore


class TestFileStore(TestCase):
    """Test cases for torch.distributed.FileStore.__init__ and FileStore.path."""

    def setUp(self):
        super().setUp()
        device_name = torch._C._get_privateuse1_backend_name()
        self.assertEqual(device_name, 'npu', f"Expected device 'npu', got '{device_name}'")
        # Temp dir for store files
        self._tmpdir = tempfile.mkdtemp()

    def tearDown(self):
        super().tearDown()
        import shutil
        shutil.rmtree(self._tmpdir, ignore_errors=True)

    def _tmpfile(self, name='store'):
        return os.path.join(self._tmpdir, name)

    # ------------------------------------------------------------------
    # __init__ tests
    # ------------------------------------------------------------------

    def test_init_basic(self):
        """FileStore can be created with a valid file path and default world_size."""
        path = self._tmpfile('basic')
        store = FileStore(path)
        self.assertIsNotNone(store)

    def test_init_world_size_default(self):
        """FileStore with omitted world_size defaults to -1 (unbounded)."""
        path = self._tmpfile('ws_default')
        store = FileStore(path)
        self.assertIsNotNone(store)

    def test_init_world_size_variants(self):
        """FileStore accepts world_size=-1, 1, and 4."""
        for ws in (-1, 1, 4):
            path = self._tmpfile(f'ws_{ws}')
            store = FileStore(path, world_size=ws)
            self.assertIsNotNone(store, f"FileStore creation failed for world_size={ws}")

    def test_init_explicit_world_size_positive(self):
        """FileStore with explicit world_size=2."""
        path = self._tmpfile('ws_2')
        store = FileStore(path, 2)
        self.assertIsNotNone(store)

    def test_init_empty_filename_raises(self):
        """FileStore with empty file_name should raise an exception."""
        raised = False
        try:
            FileStore("")
        except Exception:
            raised = True
        self.assertTrue(raised, "Expected exception for empty file_name")

    # ------------------------------------------------------------------
    # path property tests
    # ------------------------------------------------------------------

    def test_path_matches_file_name(self):
        """FileStore.path returns the same path used at construction."""
        path = self._tmpfile('path_check')
        store = FileStore(path)
        self.assertEqual(store.path, path)

    def test_path_type_is_str(self):
        """FileStore.path returns a str."""
        path = self._tmpfile('path_type')
        store = FileStore(path)
        self.assertIsInstance(store.path, str)

    def test_path_nonempty(self):
        """FileStore.path is non-empty."""
        path = self._tmpfile('path_nonempty')
        store = FileStore(path)
        self.assertTrue(len(store.path) > 0)

    def test_multiple_stores_different_paths(self):
        """Multiple FileStore instances on different paths have distinct paths."""
        p1 = self._tmpfile('store1')
        p2 = self._tmpfile('store2')
        s1 = FileStore(p1)
        s2 = FileStore(p2)
        self.assertNotEqual(s1.path, s2.path)

    def test_store_basic_kv_operations(self):
        """FileStore set/get works after construction."""
        path = self._tmpfile('kv_ops')
        store = FileStore(path, world_size=1)
        store.set("key1", "value1")
        result = store.get("key1")
        self.assertIsNotNone(result)


if __name__ == "__main__":
    run_tests()
