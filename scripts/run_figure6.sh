#!/usr/bin/env bash
# The Figure-6 example: one coupled 3D-0D four-chamber run, arm
#   vc4-l01.28-lv480-sv300pv50-rad0.7-csa1.5-unlc287
# Every flag below is the value the original run resolved (see reference/slurm_17137762_1.out
# lines 2-4 and reference/restart_manifest.json). Nothing here is a default to be tuned --
# changing any of it changes the trajectory and the run stops being this example.
#
#   RANKS=16 OUT=/path/to/output ./scripts/run_figure6.sh
#
# Wall clock at 16 ranks on the original hardware: ~1 h loading ramp + ~4 h 10 m coupled
# cycles. Produces 4 complete beats (stop_iter=3 runs cycles 0..3).
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RANKS="${RANKS:-16}"
OUT="${OUT:?set OUT=<output directory>}"
CASE_ID="fch_case03_vc4-l01.28-lv480-sv300pv50-rad0.7-csa1.5-unlc287"

# demo/fch_baseline.py imports the repo as `heArt_legacy` and puts its own grandparent on
# sys.path. Give it an import root that satisfies that without editing a single source byte.
IMPORT_ROOT="${ROOT}/.import_root"
mkdir -p "${IMPORT_ROOT}"
ln -sfn "${ROOT}" "${IMPORT_ROOT}/heArt_legacy"
export PYTHONPATH="${IMPORT_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"

export FCH_MESH_SCALE="${FCH_MESH_SCALE:-0.1}"   # mm -> cm; a unit conversion, not a lever
export HDF5_USE_FILE_LOCKING=FALSE HDF5_DISABLE_VERSION_CHECK=2
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 BLIS_NUM_THREADS=1

MESH_DIR="${ROOT}/data/mesh/"
for f in fch_clregion_whole.hdf5 fch_clregion_refine.hdf5 unload_fitted_passive.json; do
  [ -f "${MESH_DIR}${f}" ] || { echo "missing ${MESH_DIR}${f} -- run scripts/fetch_mesh.sh" >&2; exit 3; }
done
python3 -c "import dolfin; assert dolfin.__version__.startswith('2019'), dolfin.__version__"
mkdir -p "${OUT}"

ARGS=(
  --case_ID "${CASE_ID}"
  --mesh-dir "${MESH_DIR}"
  --output-root "${OUT}"
  --params "${ROOT}/inputs/anchor_v2b_sv300pv50_rad0.7_csa1.5_params.json"
  --coupling volctrl4
  --event-snapshots "${ROOT}/inputs/events.json" --event-snapshot-first-cycle 1
  --edp-mmhg 8.0 --edp-rv-mmhg 6.0 --edp-la-mmhg 4.0 --edp-ra-mmhg 4.0
  --viscous-eta 250 --dt-systole 0.5 --dt-relax 1.0 --dt-filling 2.0
  --backoff-min-factor 0.03 --newton-linesearch --write-step 40
  --tmax-lv 480000 --tmax-rv 60000 --tmax-la 55000 --tmax-ra 55000
  --active-l0 1.28
)
# NOTE: no --no-preload and no --ed-manifest. The mesh is the UNLOADED reference, so
# volctrl4 ramps the four cavities to the EDP targets first (the run logged no_preload=0).

cd "${ROOT}"
# np=1 pre-warm compiles the UFL forms once; without it the 16-rank fan-out races on the
# dijitso cache. Identical flags, one cycle, one solve.
echo "[run] JIT pre-warm (np=1)"
python3 demo/fch_baseline.py "${ARGS[@]}" --stop-iter 1 --single-solve-timing
echo "[run] coupled cycles (${RANKS} ranks, stop_iter=3 -> 4 complete beats)"
mpirun -np "${RANKS}" python3 demo/fch_baseline.py "${ARGS[@]}" --stop-iter 3
echo "[run] done -> ${OUT}/${CASE_ID}"
