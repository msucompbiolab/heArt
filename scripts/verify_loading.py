#!/usr/bin/env python3
"""Stage C (early) -- compare the loading ramp against the reference run's log.

output_PV.csv only starts filling after the ~1 h volume-controlled loading ramp, so the
first hour of a run is otherwise unverifiable. The ramp is not silent, though: the solver
logs the unloaded cavity volumes and then a progress line per ramp step. Those numbers are
a trajectory, and reference/slurm_17137762_1.out recorded the reference one.

Compares, in order:
  1. V_LV_unload / V_RV_unload -- full precision, emitted ~2 minutes in. These come out of
     the unloaded reference mesh before any ramp step, so a mismatch here means the mesh or
     the passive material differs, not the solver.
  2. the `fe/loading progress` sequence -- every key=value pair on every ramp line, compared
     positionally at the precision the log prints.

A log holds TWO ramps: the np=1 JIT pre-warm and then the 16-rank run. Their MPI
reduction orders differ, so their unloaded volumes differ in the last digits
(161.6423698975919 at np=1 vs ...921 at np=16). Everything is therefore compared
POSITIONALLY and truncated to what the run has produced so far, which lines the pre-warm
up against the pre-warm and the run against the run, and lets this be run repeatedly
against a job in flight. Do not compare logs from different rank counts.

Usage:  scripts/verify_loading.py <run log>
Exit 0 when every compared value agrees, 1 otherwise.
"""
import math
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REF = ROOT / "reference" / "slurm_17137762_1.out"

KV = re.compile(r"(\w+)=(-?[\d.eE+]+)")
UNLOAD = re.compile(r"(V_[LR]V_unload)\s*=\s*(-?[\d.eE+]+)")


def unload_volumes(text):
    """Every occurrence, in order. A log holds one set per ramp (pre-warm, then the run),
    and they legitimately differ in the last digits, so these are compared positionally --
    never last-against-last, which would grade a run still in its pre-warm against the
    reference's 16-rank numbers and call a correct run a failure."""
    return [(m.group(1), m.group(2)) for m in UNLOAD.finditer(text)]


def ramp_steps(text):
    steps = []
    for line in text.splitlines():
        if "fe/loading" in line and "progress" in line:
            steps.append(dict(KV.findall(line)))
    return steps


def main(argv):
    if len(argv) != 2:
        print(__doc__)
        return 2
    run_path = Path(argv[1])
    if not run_path.exists():
        print(f"no such file: {run_path}")
        return 2
    ref_text = REF.read_text(errors="ignore")
    run_text = run_path.read_text(errors="ignore")

    failures = 0

    ref_u, run_u = unload_volumes(ref_text), unload_volumes(run_text)
    if not run_u:
        print("[WAIT] unloaded cavity volumes not logged yet")
    else:
        if len(run_u) > len(ref_u):
            print(f"[FAIL] run logged {len(run_u)} unloaded volumes, reference only {len(ref_u)}")
            failures += 1
        for i, (k, want) in enumerate(ref_u[: len(run_u)]):
            gk, got = run_u[i]
            ok = (gk, got) == (k, want)
            ramp = "pre-warm" if i < 2 else "run"
            print(f"[{'OK ' if ok else 'FAIL'}] {ramp:<8} {gk} = {got}  (reference {want})")
            if not ok and gk == k:
                # Quantify: a last-bit reduction-order difference and a diverged solve both
                # print as FAIL, and they mean completely different things.
                try:
                    a, b = float(got), float(want)
                    d = abs(a - b)
                    ulps = d / math.ulp(b) if math.ulp(b) else float("inf")
                    rel = d / abs(b) if b else float("inf")
                    print(f"         abs={d:.4e}  rel={rel:.3e}  ULPs={ulps:.1f}")
                except ValueError:
                    pass
            failures += not ok
        if len(run_u) < len(ref_u):
            print(f"[WAIT] {len(ref_u) - len(run_u)} more unloaded volume(s) still to come")

    ref_s, run_s = ramp_steps(ref_text), ramp_steps(run_text)
    n = min(len(ref_s), len(run_s))
    print(f"\nramp steps -- reference {len(ref_s)}, run {len(run_s)}, comparing {n}")
    for i in range(n):
        diffs = [
            f"{k}: run={run_s[i].get(k)} ref={v}"
            for k, v in ref_s[i].items()
            if run_s[i].get(k) != v
        ]
        if diffs:
            print(f"[FAIL] ramp step {i + 1}: " + "; ".join(diffs))
            failures += 1
    if n and not failures:
        pct = 100.0 * n / max(len(ref_s), 1)
        print(f"[OK ] all {n} ramp steps identical to the reference ({pct:.0f}% of the ramp)")

    if failures:
        print(f"\nFAILED: {failures} mismatch(es) -- the run is not on the reference trajectory.")
        return 1
    if not run_u and not n:
        print("\nNothing to compare yet; the run has not reached the loading ramp.")
        return 1
    print("\nLoading ramp matches the reference.")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
