# -*- coding: utf-8 -*-
"""
测试目的：验证 torch.distributed.PrefixStore.__init__ 与 PrefixStore.underlying_store 接口功能正确性
API 名称：
  - torch.distributed.PrefixStore.__init__
  - torch.distributed.PrefixStore.underlying_store
API 签名：
  - PrefixStore.__init__(self, prefix: str, store: Store) -> None
  - PrefixStore.underlying_store  -> Store  (property)

覆盖维度表：
| 覆盖维度         | 说明                                                         | 覆盖情况                                       |
|------------------|--------------------------------------------------------------|------------------------------------------------|
| 空/非空          | prefix 空字符串 vs 非空字符串                                | 已覆盖：test_init_empty_prefix / test_init_basic |
| 枚举选项         | 底层 Store 为 FileStore / PrefixStore（嵌套）                | 已覆盖：test_underlying_is_filestore / test_nested_prefix_store |
| 参数类型         | prefix: str；store: Store (FileStore / PrefixStore)          | 已覆盖                                         |
| 传参与不传参     | 两个参数均为必选参数                                         | 已覆盖                                         |
| 等价类/边界值    | 空前缀、含特殊字符前缀、多级嵌套 PrefixStore                 | 已覆盖：test_init_special_chars / test_nested_prefix_store |
| 正常传参场景     | 创建 PrefixStore 并通过 underlying_store 获取底层存储        | 已覆盖                                         |
| 异常传参场景     | store 参数缺失（TypeError）                                  | 已覆盖：test_init_no_store_raises              |

未覆盖项及原因：
- 无

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

from torch.distributed import FileStore, PrefixStore


class TestPrefixStore(TestCase):
    """Test cases for torch.distributed.PrefixStore.__init__ and underlying_store."""

    def setUp(self):
        super().setUp()
        device_name = torch._C._get_privateuse1_backend_name()
        self.assertEqual(device_name, 'npu', f"Expected device 'npu', got '{device_name}'")
        self._tmpdir = tempfile.mkdtemp()

    def tearDown(self):
        super().tearDown()
        import shutil
        shutil.rmtree(self._tmpdir, ignore_errors=True)

    def _make_filestore(self, name='base'):
        path = os.path.join(self._tmpdir, name)
        return FileStore(path, world_size=1)

    # ------------------------------------------------------------------
    # __init__ tests
    # ------------------------------------------------------------------

    def test_init_basic(self):
        """PrefixStore can be created with a valid prefix and FileStore."""
        base = self._make_filestore('basic')
        ps = PrefixStore("prefix", base)
        self.assertIsNotNone(ps)

    def test_init_empty_prefix(self):
        """PrefixStore accepts empty string as prefix."""
        base = self._make_filestore('empty_pfx')
        ps = PrefixStore("", base)
        self.assertIsNotNone(ps)

    def test_init_special_chars(self):
        """PrefixStore accepts prefix with special characters."""
        base = self._make_filestore('special')
        ps = PrefixStore("/rank_0/layer_1", base)
        self.assertIsNotNone(ps)

    def test_init_no_store_raises(self):
        """PrefixStore without store argument raises TypeError."""
        raised = False
        try:
            PrefixStore("prefix")
        except TypeError:
            raised = True
        self.assertTrue(raised, "Expected TypeError when store argument is missing")

    def test_nested_prefix_store(self):
        """PrefixStore can wrap another PrefixStore (nested)."""
        base = self._make_filestore('nested')
        ps1 = PrefixStore("outer", base)
        ps2 = PrefixStore("inner", ps1)
        self.assertIsNotNone(ps2)

    # ------------------------------------------------------------------
    # underlying_store property tests
    # ------------------------------------------------------------------

    def test_underlying_is_filestore(self):
        """underlying_store returns the wrapped FileStore instance."""
        base = self._make_filestore('underlying')
        ps = PrefixStore("pfx", base)
        underlying = ps.underlying_store
        self.assertIsNotNone(underlying)
        self.assertIsInstance(underlying, FileStore)

    def test_underlying_store_type(self):
        """underlying_store type check for nested PrefixStore."""
        base = self._make_filestore('nested_type')
        ps1 = PrefixStore("outer", base)
        ps2 = PrefixStore("inner", ps1)
        # ps2's underlying should be ps1, which is a PrefixStore
        self.assertIsInstance(ps2.underlying_store, PrefixStore)

    def test_underlying_store_identity(self):
        """underlying_store is the same object passed at construction."""
        base = self._make_filestore('identity')
        ps = PrefixStore("id_test", base)
        # The path of the underlying FileStore should be the same
        self.assertIsInstance(ps.underlying_store, FileStore)
        self.assertEqual(ps.underlying_store.path, base.path)

    def test_kv_isolation_by_prefix(self):
        """Two PrefixStores with different prefixes do not share keys."""
        base = self._make_filestore('kv_iso')
        ps1 = PrefixStore("ns1", base)
        ps2 = PrefixStore("ns2", base)
        ps1.set("key", "val1")
        ps2.set("key", "val2")
        self.assertEqual(ps1.get("key"), b"val1")
        self.assertEqual(ps2.get("key"), b"val2")


if __name__ == "__main__":
    run_tests()
