#!/usr/bin/env python3
"""Stage A -- prove this checkout is the code and input set that produced the run.

Checks, in order:
  1. code_provenance()  -- md5 over the nine trajectory-defining sources declared in
     src/sim_protocols/fch_restart.py::_CODE_FILES, against the value the run recorded
     in its restart_manifest.json. Any byte change in FE, 0D, material or config code
     changes this hash, so a match means the numerical trajectory is defined by
     identical code.
  2. mesh md5           -- against restart_manifest.json::mesh_md5 (skipped when the
     mesh has not been fetched; see scripts/fetch_mesh.sh).
  3. input sha256       -- every run input and reference artifact against the entries
     in reference/MANIFEST.sha256 written by the original run.

Exits non-zero on the first failure. Needs only the standard library.
"""
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "reference" / "restart_manifest.json"
SHA_MANIFEST = ROOT / "reference" / "MANIFEST.sha256"

# Mirrors src/sim_protocols/fch_restart.py::_CODE_FILES. Kept here as a literal so this
# check does not depend on importing the package (which needs FEniCS).
CODE_FILES = (
    "src/sim_protocols/run_light.py",
    "src/sim_protocols/circBiV.py",
    "src/sim_protocols/fch_restart.py",
    "src/mechanics/MEmodel3.py",
    "src/mechanics/forms_MRC2.py",
    "src/mechanics/activeforms_MRC2.py",
    "src/mechanics/spring_bc_forms.py",
    "orchestrate/fch_baseline_config.py",
    "calibration/params/canonical.py",
)

# repo path -> path as recorded in reference/MANIFEST.sha256
SHA_CHECKS = {
    "inputs/anchor_v2b_sv300pv50_rad0.7_csa1.5_params.json":
        "./circuit/anchor_v2b_sv300pv50_rad0.7_csa1.5_params.json",
    "reference/arms.json": "./arms.json",
    "reference/ed_state_manifest.json": "./ed_state_manifest.json",
    "inputs/unload_fitted_passive.json": "./reference/unload_fitted_passive.json",
    "reference/output_PV.csv": "./output_PV.csv",
    "reference/baseline_summary.csv": "./baseline_summary.csv",
    "reference/restart_manifest.json": "./restart_manifest.json",
    "reference/reference_check.json": "./reference/reference_check.json",
    "reference/snapshots.json": "./snapshots/snapshots.json",
}


def code_provenance() -> str:
    h = hashlib.md5()
    for rel in sorted(CODE_FILES):
        p = ROOT / rel
        h.update(rel.encode())
        h.update(b"=")
        h.update(p.read_bytes() if p.exists() else b"<absent>")
        h.update(b"\n")
    return h.hexdigest()


def file_digest(path: Path, algo: str) -> str:
    h = hashlib.new(algo)
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> int:
    manifest = json.loads(MANIFEST.read_text())
    failures = []

    want = manifest["code_provenance"]
    got = code_provenance()
    ok = got == want
    print(f"[{'OK ' if ok else 'FAIL'}] code_provenance  {got}  (run recorded {want})")
    if not ok:
        failures.append("code_provenance")

    mesh = ROOT / "data/mesh/fch_clregion_whole.hdf5"
    if mesh.exists():
        want = manifest["mesh_md5"]
        got = file_digest(mesh, "md5")
        ok = got == want
        print(f"[{'OK ' if ok else 'FAIL'}] mesh md5         {got}  (run recorded {want})")
        if not ok:
            failures.append("mesh_md5")
    else:
        print("[SKIP] mesh md5         data/mesh not fetched -- run scripts/fetch_mesh.sh")

    want_by_key = {}
    for line in SHA_MANIFEST.read_text().splitlines():
        if line.strip():
            digest, _, key = line.partition("  ")
            want_by_key[key.strip()] = digest
    for rel, key in sorted(SHA_CHECKS.items()):
        got = file_digest(ROOT / rel, "sha256")
        want = want_by_key.get(key, "<absent from MANIFEST>")
        ok = got == want
        print(f"[{'OK ' if ok else 'FAIL'}] sha256           {rel}")
        if not ok:
            failures.append(rel)
            print(f"        want {want}\n        got  {got}")

    if failures:
        print(f"\nFAILED: {len(failures)} check(s): {', '.join(failures)}")
        return 1
    print("\nAll provenance checks passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
