#!/usr/bin/env python3
"""Stage C -- compare a run in progress (or finished) against the reference trajectory.

The coupled run appends one row per timestep to output_PV.csv. This compares the rows
produced so far, in order, against reference/output_PV.csv:

  * byte-identical prefix  -> the trajectory is the reference trajectory, and no further
    numerical argument is needed for the rows compared;
  * otherwise              -> reports the first differing row and the largest absolute
    and relative deviation per column, so a near-miss is distinguishable from a diverged
    solve.

Usage:  scripts/verify_trajectory.py <run output_PV.csv>
Exit 0 when the compared prefix is byte-identical, 1 otherwise.
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REF = ROOT / "reference" / "output_PV.csv"


def main(argv):
    if len(argv) != 2:
        print(__doc__)
        return 2
    run = Path(argv[1])
    if not run.exists():
        print(f"no such file: {run}")
        return 2

    ref_lines = REF.read_text().splitlines()
    run_lines = run.read_text().splitlines()
    # The run appends while we read, so the last line can be a half-written row. Drop any
    # trailing line that does not have the full column count rather than calling it a
    # divergence.
    ncol = len(ref_lines[0].split(","))
    while len(run_lines) > 1 and len(run_lines[-1].split(",")) != ncol:
        run_lines.pop()
    n = min(len(ref_lines), len(run_lines))
    print(f"reference rows: {len(ref_lines) - 1}   run rows: {len(run_lines) - 1}   comparing: {n - 1}")

    if run_lines[0] != ref_lines[0]:
        print("FAIL: header differs")
        print(f"  ref {ref_lines[0]}")
        print(f"  run {run_lines[0]}")
        return 1

    first_diff = None
    for i in range(1, n):
        if run_lines[i] != ref_lines[i]:
            first_diff = i
            break

    if first_diff is None:
        pct = 100.0 * (n - 1) / max(len(ref_lines) - 1, 1)
        print(f"OK: first {n - 1} rows byte-identical to the reference ({pct:.1f}% of the run).")
        if len(run_lines) >= len(ref_lines):
            print("Run is complete and matches the reference trajectory in full.")
        return 0

    cols = ref_lines[0].split(",")
    bad = run_lines[first_diff].split(",")
    where = f"t={bad[1]} ms" if len(bad) == ncol else f"malformed row, {len(bad)} of {ncol} columns"
    print(f"FAIL: first difference at data row {first_diff} ({where})")
    print(f"  ref {ref_lines[first_diff]}")
    print(f"  run {run_lines[first_diff]}")
    worst = []
    for i in range(1, n):
        if run_lines[i] == ref_lines[i]:
            continue
        a = ref_lines[i].split(",")
        b = run_lines[i].split(",")
        for j, name in enumerate(cols):
            try:
                x, y = float(a[j]), float(b[j])
            except (ValueError, IndexError):
                continue
            d = abs(x - y)
            if d:
                worst.append((d, d / abs(x) if x else float("inf"), name, i))
    if worst:
        worst.sort(reverse=True)
        print("  largest absolute deviations:")
        seen = set()
        for d, rel, name, row in worst:
            if name in seen:
                continue
            seen.add(name)
            print(f"    {name:<8} abs={d:.6e}  rel={rel:.3e}  (row {row})")
            if len(seen) >= 6:
                break
    return 1


if __name__ == "__main__":
    sys.exit(main(sys.argv))
