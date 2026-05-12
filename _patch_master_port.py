#!/usr/bin/env python3
"""一次性脚本：把 test/ 下所有测试文件里硬编码的 MASTER_PORT 改成 setdefault。

匹配模式：
    os.environ['MASTER_PORT'] = '<digits>'
    os.environ["MASTER_PORT"] = "<digits>"

替换为：
    os.environ.setdefault('MASTER_PORT', '<digits>')

同时把 MASTER_ADDR 也改成 setdefault，保持一致风格（不影响现有逻辑）。
"""
import re
from pathlib import Path

ROOT = Path(__file__).parent / 'test'

PORT_RE = re.compile(r"os\.environ\[(['\"])MASTER_PORT\1\]\s*=\s*(['\"])(\d+)\2")
ADDR_RE = re.compile(r"os\.environ\[(['\"])MASTER_ADDR\1\]\s*=\s*(['\"])([^'\"]+)\2")

modified = []
for f in ROOT.rglob('test_*.py'):
    text = f.read_text(encoding='utf-8')
    new = PORT_RE.sub(r"os.environ.setdefault('MASTER_PORT', '\3')", text)
    new = ADDR_RE.sub(r"os.environ.setdefault('MASTER_ADDR', '\3')", new)
    if new != text:
        f.write_text(new, encoding='utf-8')
        modified.append(str(f.relative_to(ROOT.parent)))

print(f"修改了 {len(modified)} 个文件")
for m in modified:
    print(f"  {m}")
