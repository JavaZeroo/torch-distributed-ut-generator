#!/usr/bin/env python3
"""
批量运行 test/ 下的所有 UT，通过则跳过记录，失败则记录详细错误信息。
用法：
    python run_tests.py [--workers N] [--timeout T] [--output report.md]
"""

import argparse
import os
import re
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path


@dataclass
class TestResult:
    file: str           # 相对路径
    passed: int = 0
    failed: int = 0
    skipped: int = 0
    errors: int = 0
    duration: float = 0.0
    status: str = "UNKNOWN"   # PASS / FAIL / ERROR / TIMEOUT
    failure_detail: str = ""  # 失败时的错误摘要


def run_one(test_file: Path, root: Path, timeout: int) -> TestResult:
    rel = str(test_file.relative_to(root))
    result = TestResult(file=rel)

    # 把 NPU 限定到 4,5 传给子进程（一段时间内观察这两张卡稳定空闲）。
    # 必须在子进程 import torch_npu 之前注入，所以走 env 注入。
    # 如其他用户占用了 4 / 5，可外部覆盖：
    #   ASCEND_RT_VISIBLE_DEVICES=2,3 python run_tests.py
    child_env = os.environ.copy()
    child_env.setdefault('ASCEND_RT_VISIBLE_DEVICES', '4,5')

    t0 = time.monotonic()
    try:
        proc = subprocess.run(
            [
                sys.executable, "-m", "pytest",
                str(test_file),
                "-v",
                "--tb=short",
                "--no-header",
                "-q",
            ],
            capture_output=True,
            text=True,
            timeout=timeout,
            cwd=str(root),
            env=child_env,
        )
        elapsed = time.monotonic() - t0
        result.duration = elapsed

        stdout = proc.stdout
        stderr = proc.stderr
        combined = stdout + "\n" + stderr

        # 解析统计行，例如 "3 passed, 2 failed, 1 skipped"
        stat_match = re.search(
            r"(\d+) passed|(\d+) failed|(\d+) error|(\d+) skipped",
            combined,
        )
        # 更健壮地逐个提取
        def extract(pattern):
            m = re.search(pattern, combined)
            return int(m.group(1)) if m else 0

        result.passed  = extract(r"(\d+) passed")
        result.failed  = extract(r"(\d+) failed")
        result.errors  = extract(r"(\d+) error")
        result.skipped = extract(r"(\d+) skipped")

        if proc.returncode == 0:
            result.status = "PASS"
        elif result.failed > 0 or result.errors > 0:
            result.status = "FAIL"
            # 提取 FAILED 行 + 每个 FAILED block 的简短错误
            result.failure_detail = _extract_failure_detail(combined)
        else:
            # 非零退出但没解析到 failed（可能是收集错误）
            result.status = "ERROR"
            result.failure_detail = _truncate(combined, 3000)

    except subprocess.TimeoutExpired:
        result.duration = time.monotonic() - t0
        result.status = "TIMEOUT"
        result.failure_detail = f"超时（>{timeout}s），进程已终止"
    except Exception as exc:
        result.duration = time.monotonic() - t0
        result.status = "ERROR"
        result.failure_detail = str(exc)

    return result


def _extract_failure_detail(text: str) -> str:
    """从 pytest 输出中提取失败摘要（FAILED 行 + short traceback 块）。"""
    lines = text.splitlines()
    sections = []

    # 1. 收集 "FAILED xxx::yyy - ..." 行
    failed_lines = [l for l in lines if l.startswith("FAILED")]
    if failed_lines:
        sections.append("【失败用例】")
        sections.extend(failed_lines)

    # 2. 提取每个 short traceback 段落（以 "FAILED" 或 "_ test_" 开头的块）
    in_block = False
    block_lines = []
    capture_blocks = []
    for line in lines:
        if re.match(r"^_{5,}", line):   # 分隔线 "_____..."
            if block_lines:
                capture_blocks.append("\n".join(block_lines))
            block_lines = []
            in_block = True
        elif in_block:
            block_lines.append(line)
    if block_lines:
        capture_blocks.append("\n".join(block_lines))

    if capture_blocks:
        sections.append("\n【错误详情】")
        for blk in capture_blocks[:10]:      # 最多 10 个块，避免太长
            sections.append(blk[:800])       # 每块限 800 字符
            sections.append("---")

    return _truncate("\n".join(sections), 5000)


def _truncate(text: str, limit: int) -> str:
    if len(text) <= limit:
        return text
    return text[:limit] + f"\n... (截断，共 {len(text)} 字符)"


def find_test_files(test_dir: Path):
    return sorted(test_dir.rglob("test_*.py"))


def print_progress(idx, total, result: TestResult):
    icon = {"PASS": "✅", "FAIL": "❌", "TIMEOUT": "⏱ ", "ERROR": "💥"}.get(result.status, "❓")
    print(
        f"[{idx:3}/{total}] {icon} {result.status:<7} "
        f"{result.duration:6.1f}s  {result.file}",
        flush=True,
    )


def write_report(results: list[TestResult], output: Path, workers: int, timeout: int):
    total = len(results)
    n_pass    = sum(1 for r in results if r.status == "PASS")
    n_fail    = sum(1 for r in results if r.status == "FAIL")
    n_timeout = sum(1 for r in results if r.status == "TIMEOUT")
    n_error   = sum(1 for r in results if r.status == "ERROR")
    total_duration = sum(r.duration for r in results)

    lines = []
    lines.append("# UT 批量执行报告")
    lines.append(f"\n**生成时间**：{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append(f"**并发 workers**：{workers}  **单文件超时**：{timeout}s")
    lines.append(f"**总文件数**：{total}  **累计耗时**：{total_duration:.1f}s\n")

    lines.append("## 汇总\n")
    lines.append("| 状态 | 数量 |")
    lines.append("|------|------|")
    lines.append(f"| ✅ PASS    | {n_pass} |")
    lines.append(f"| ❌ FAIL    | {n_fail} |")
    lines.append(f"| ⏱  TIMEOUT | {n_timeout} |")
    lines.append(f"| 💥 ERROR   | {n_error} |")
    lines.append(f"| **合计**   | **{total}** |")

    lines.append("\n## 全量结果\n")
    lines.append("| 状态 | 用时(s) | 通过 | 失败 | 跳过 | 文件 |")
    lines.append("|------|---------|------|------|------|------|")
    for r in results:
        icon = {"PASS": "✅", "FAIL": "❌", "TIMEOUT": "⏱", "ERROR": "💥"}.get(r.status, "❓")
        lines.append(
            f"| {icon} {r.status} | {r.duration:.1f} | {r.passed} | {r.failed} | {r.skipped} | `{r.file}` |"
        )

    # 只有非 PASS 的才展开详情
    problem_results = [r for r in results if r.status != "PASS"]
    if problem_results:
        lines.append("\n---\n")
        lines.append("## 失败 / 超时 / 错误 详情\n")
        for r in problem_results:
            icon = {"FAIL": "❌", "TIMEOUT": "⏱", "ERROR": "💥"}.get(r.status, "❓")
            lines.append(f"### {icon} {r.file}\n")
            lines.append(f"- **状态**：{r.status}")
            lines.append(f"- **耗时**：{r.duration:.1f}s")
            lines.append(f"- **通过/失败/跳过**：{r.passed}/{r.failed}/{r.skipped}\n")
            if r.failure_detail:
                lines.append("```")
                lines.append(r.failure_detail)
                lines.append("```\n")

    output.write_text("\n".join(lines), encoding="utf-8")
    print(f"\n报告已写入：{output}")


def main():
    parser = argparse.ArgumentParser(description="批量运行 test/ 目录下的 UT")
    parser.add_argument("--workers", type=int, default=1,
                        help="并发 worker 数（默认 1；分布式测试建议 1 避免端口冲突）")
    parser.add_argument("--timeout", type=int, default=300,
                        help="单个测试文件超时秒数（默认 300s）")
    parser.add_argument("--output", default="UT_BATCH_REPORT.md",
                        help="报告输出路径（默认 UT_BATCH_REPORT.md）")
    parser.add_argument("--filter", default="",
                        help="文件名关键字过滤，只运行匹配的文件（可选）")
    args = parser.parse_args()

    root = Path(__file__).parent
    test_dir = root / "test"

    test_files = find_test_files(test_dir)
    if args.filter:
        test_files = [f for f in test_files if args.filter in f.name]

    total = len(test_files)
    print(f"发现 {total} 个测试文件，workers={args.workers}，timeout={args.timeout}s\n")

    results_map: dict[str, TestResult] = {}
    completed_count = 0

    if args.workers == 1:
        for i, tf in enumerate(test_files, 1):
            r = run_one(tf, root, args.timeout)
            results_map[str(tf)] = r
            completed_count += 1
            print_progress(completed_count, total, r)
    else:
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            future_to_file = {pool.submit(run_one, tf, root, args.timeout): tf for tf in test_files}
            for future in as_completed(future_to_file):
                r = future.result()
                results_map[str(future_to_file[future])] = r
                completed_count += 1
                print_progress(completed_count, total, r)

    # 按原始顺序输出
    ordered = [results_map[str(tf)] for tf in test_files]

    n_pass    = sum(1 for r in ordered if r.status == "PASS")
    n_fail    = sum(1 for r in ordered if r.status != "PASS")
    print(f"\n完成：{n_pass} PASS，{n_fail} 非 PASS（含 FAIL/TIMEOUT/ERROR）")

    write_report(ordered, root / args.output, args.workers, args.timeout)


if __name__ == "__main__":
    main()
