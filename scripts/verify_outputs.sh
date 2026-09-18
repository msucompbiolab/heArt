#!/usr/bin/env bash
# Stage D -- check a COMPLETED run against reference/MANIFEST.sha256, the checksum list
# the original run wrote over its own outputs.
#
#   ./scripts/verify_outputs.sh <run case directory>
#
# The manifest also covers inputs that the run directory does not contain (arms.json,
# circuit params, the ed manifest); those are verified separately by
# scripts/verify_provenance.py and are reported here as "not produced by the run".
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CASE_DIR="${1:?usage: verify_outputs.sh <run case directory>}"
MANIFEST="${ROOT}/reference/MANIFEST.sha256"

PRODUCED='^\./(output_PV\.csv|output_strain\.csv|baseline_summary\.csv|restart_manifest\.json|snapshots/.*)$'
# The *_viz.xdmf files are XML indexes over the HDF5 payloads: they name dataset shapes,
# never values, and are byte-identical between runs that agree on nothing numerically
# (verified against a different arm of the same campaign -- all 15 matched, all 35
# substantive artifacts differed). They are counted separately so a partial match cannot
# read as partial success.
INDEX_ONLY='_viz\.xdmf$'
pass=0; fail=0; absent=0; idx=0
cd "${CASE_DIR}"
while read -r want path; do
  [[ "${path}" =~ ${PRODUCED} ]] || continue
  if [ ! -f "${path}" ]; then
    echo "[MISSING] ${path}"; absent=$((absent+1)); continue
  fi
  got="$(sha256sum "${path}" | awk '{print $1}')"
  if [ "${got}" = "${want}" ]; then
    if [[ "${path}" =~ ${INDEX_ONLY} ]]; then idx=$((idx+1)); else pass=$((pass+1)); fi
  else fail=$((fail+1)); echo "[DIFFER ] ${path}"; echo "          want ${want}"; echo "          got  ${got}"; fi
done < <(sed 's/  \./ ./' "${MANIFEST}")

echo
echo "numerical artifacts matched: ${pass}"
echo "xdmf indexes matched:        ${idx}   (carry no values; not evidence on their own)"
echo "differing: ${fail}   missing: ${absent}"
[ "${fail}" -eq 0 ] && [ "${absent}" -eq 0 ]
