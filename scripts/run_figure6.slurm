#!/bin/bash --login
#SBATCH --job-name=fch_fig6
#SBATCH --nodes=1
#SBATCH --ntasks=16
#SBATCH --cpus-per-task=1
#SBATCH --hint=nomultithread
#SBATCH --mem=120G
#SBATCH --time=24:00:00
#SBATCH --partition=general-long,general-short
#SBATCH --constraint=acm
# The original run requested exactly these resources. --constraint=acm pins the CPU
# microarchitecture: floating-point results depend on it (FMA and vector-width differences
# change the last bits), so a bitwise comparison against reference/MANIFEST.sha256 is only
# meaningful on the same feature. Drop the constraint and expect a trajectory that agrees
# numerically but not byte-for-byte.
#
# Submit:  OUT=<dir> ENV_ACTIVATE=<path to activate script> sbatch scripts/run_figure6.slurm
#
# ENV_ACTIVATE is sourced as-is, so anything IT needs must be exported at submit time too
# (sbatch forwards the submitting environment). A site whose activate script is a wrapper
# around a shared read-only env typically needs to be told where that env lives. At MSU:
#
#   OUT=<dir> \
#   ENV_ACTIVATE=/mnt/research/compbiolab/heart_revision/activate_fenics_mamba.sh \
#   HEART_LEGACY_ENV_ROOT=/mnt/research/compbiolab/HeartX/runtime/legacy-dolfin \
#   sbatch scripts/run_figure6.slurm
#
# Note the cache variables below are exported BEFORE sourcing: the shared activate script
# defaults them into its own (read-only) prefix otherwise, and every rank then fights over
# one JIT cache.
set -euo pipefail
ROOT="${SLURM_SUBMIT_DIR:-$(pwd)}"
export TMPDIR="/tmp/${USER}_${SLURM_JOB_ID:-$$}"; mkdir -p "$TMPDIR"
export XDG_CACHE_HOME="${TMPDIR}/xdg"
export INSTANT_CACHE_DIR="${TMPDIR}/instant" DIJITSO_CACHE_DIR="${TMPDIR}/dijitso"
mkdir -p "$XDG_CACHE_HOME" "$INSTANT_CACHE_DIR" "$DIJITSO_CACHE_DIR"
trap '[ -d "$TMPDIR" ] && rm -rf "$TMPDIR" || true' EXIT
set +u; source "${ENV_ACTIVATE:?set ENV_ACTIVATE=<activate script for the FEniCS 2019 env>}"; set -u
RANKS="${SLURM_NTASKS:-16}" OUT="${OUT:?set OUT}" exec "${ROOT}/scripts/run_figure6.sh"
