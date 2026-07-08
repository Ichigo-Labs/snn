#!/usr/bin/env python3
"""Minimal gcov gate for the host C implementation.

Usage: python3 scripts/coverage.py <build-dir> [required-percent]
"""
from __future__ import annotations

import pathlib
import re
import subprocess
import sys


# Library translation units whose line coverage is gated. Kernel device code in
# snn_cuda.cu is not measurable by gcov; only its host-side control flow is.
SOURCE_BASENAMES = ("snn.c", "snn_cuda.cu", "snn_cuda_stub.c")


def main() -> int:
    if len(sys.argv) < 2:
        print("usage: coverage.py <build-dir> [required-percent]", file=sys.stderr)
        return 2
    build = pathlib.Path(sys.argv[1]).resolve()
    required = float(sys.argv[2]) if len(sys.argv) > 2 else 100.0
    gcno_files = [
        p
        for p in build.rglob("*.gcno")
        if "CMakeFiles/snn_cpu.dir" in p.as_posix()
        or "CMakeFiles/snn_cuda_obj.dir" in p.as_posix()
        or "CMakeFiles/snn.dir/src/snn_cuda_stub.c" in p.as_posix()
    ]
    if not gcno_files:
        print(f"no .gcno files found under {build}", file=sys.stderr)
        return 1
    # gcov -n prints a "File '<path>'" line followed by "Lines executed:P% of N".
    file_pat = re.compile(r"File '([^']+)'")
    lines_pat = re.compile(r"Lines executed:([0-9.]+)% of (\d+)")
    total_lines = 0
    weighted_percent_sum = 0.0
    reports: list[tuple[str, float, int]] = []
    for gcno in gcno_files:
        # The object directory is where gcov writes/looks for the matching .gcda.
        proc = subprocess.run(
            ["gcov", "-n", "-b", "-c", str(gcno)],
            cwd=gcno.parent,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            check=False,
        )
        if proc.returncode != 0:
            print(proc.stdout, file=sys.stderr)
            return proc.returncode
        # Walk the gcov -n report, keeping only our own library sources so that
        # toolchain/system headers pulled into the .cu unit are ignored.
        out_lines = proc.stdout.splitlines()
        current = None
        for line in out_lines:
            fm = file_pat.search(line)
            if fm:
                current = fm.group(1)
                continue
            lm = lines_pat.search(line)
            if lm and current is not None:
                base = pathlib.PurePath(current).name
                if base in SOURCE_BASENAMES:
                    pct = float(lm.group(1))
                    line_count = int(lm.group(2))
                    total_lines += line_count
                    weighted_percent_sum += pct * line_count
                    reports.append((current, pct, line_count))
                current = None
    if total_lines == 0:
        print("gcov reported zero instrumented lines", file=sys.stderr)
        return 1
    overall = weighted_percent_sum / total_lines
    for path, pct, lines in reports:
        print(f"{pct:6.2f}% {lines:5d} lines {path}")
    print(f"overall line coverage: {overall:.2f}%")
    if overall + 1e-9 < required:
        print(f"coverage below required {required:.2f}%", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
