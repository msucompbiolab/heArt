#!/usr/bin/env bash
# Fetch the two mesh files the run reads. They are ~42 MB each and are NOT tracked in
# git; data/mesh/ is gitignored.
#
# Provenance of this mesh: Strocchi case 03 (Zenodo 3890034) ED segmentation, unloaded
# with `python -m unloading fch` at Cparam_lv/rv/la/ra = 287.091/287.0/135.022/108.126 Pa
# and EDP 8/6/4/4 mmHg. The resulting unloaded reference is the directory named
# `unload_rvwall_c287`. reference/reference_check.json records the unloaded and loaded
# cavity volumes; reference/restart_manifest.json records the mesh md5 this script checks.
#
#   SRC=<dir containing fch_clregion_whole.hdf5 and fch_clregion_refine.hdf5> ./fetch_mesh.sh
#
# SRC may be a local path or an rsync/scp remote of the form host:/path.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DEST="${ROOT}/data/mesh"
SRC="${SRC:-hpcc-msu:/mnt/gs21/scratch/ziaeirad/fch_coupled/case03/unload_rvwall_c287}"
# mesh_md5 out of the run's own restart manifest, without needing a python interpreter
WANT_MD5="$(sed -n 's/.*"mesh_md5"[[:space:]]*:[[:space:]]*"\([0-9a-f]*\)".*/\1/p' \
            "${ROOT}/reference/restart_manifest.json")"
[ -n "${WANT_MD5}" ] || { echo "[fetch_mesh] ERROR: no mesh_md5 in restart_manifest.json" >&2; exit 1; }

mkdir -p "${DEST}"
for f in fch_clregion_whole.hdf5 fch_clregion_refine.hdf5; do
  if [ -f "${DEST}/${f}" ]; then
    echo "[fetch_mesh] have ${f}"
  else
    echo "[fetch_mesh] ${SRC}/${f} -> ${DEST}/${f}"
    if [[ "${SRC}" == *:* ]]; then scp -q "${SRC}/${f}" "${DEST}/${f}"
    else cp -p "${SRC}/${f}" "${DEST}/${f}"; fi
  fi
done
# unload_fitted_passive.json must sit beside the mesh: the run reads the matched-pair
# passive stiffness from <mesh-dir>/unload_fitted_passive.json (preload path, NO_PRELOAD=0).
cp -f "${ROOT}/inputs/unload_fitted_passive.json" "${DEST}/unload_fitted_passive.json"

GOT_MD5="$(md5sum "${DEST}/fch_clregion_whole.hdf5" | awk '{print $1}')"
if [ "${GOT_MD5}" != "${WANT_MD5}" ]; then
  echo "[fetch_mesh] ERROR: mesh md5 ${GOT_MD5} != recorded ${WANT_MD5}" >&2; exit 1
fi
echo "[fetch_mesh] OK  mesh md5 ${GOT_MD5} matches the run's restart_manifest."
