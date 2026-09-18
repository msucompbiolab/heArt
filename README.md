# A coupled 3D–0D four-chamber cardiac simulation

A complete, self-contained example of one finite-element heart simulation: a four-chamber
mechanics model, solved in FEniCS, coupled to a closed-loop lumped-parameter circulation.
It runs one specific case — arm `vc4-l01.28-lv480-sv300pv50-rad0.7-csa1.5-unlc287` — and
ships the inputs it reads, the results it originally produced, and scripts that check a new
run against them.

The package is the import closure of that single run and nothing more, so everything you
open is code the example actually executes.

## Quick start

```bash
# legacy FEniCS/DOLFIN (not DOLFINx)
micromamba create -p ./.venv -f env/environment.yml && micromamba activate ./.venv

# the mesh is fetched rather than tracked
SRC=<dir with fch_clregion_{whole,refine}.hdf5> ./scripts/fetch_mesh.sh

# confirm the checkout before spending compute on it
./scripts/check_closure.py
./scripts/verify_provenance.py

# run
RANKS=16 OUT=$PWD/outputs ./scripts/run_figure6.sh
#   under SLURM: OUT=<dir> ENV_ACTIVATE=<activate script> sbatch scripts/run_figure6.slurm
```

## Reading the code

Forty-four modules, most of them small; the mechanics model is by far the largest.
This is the path through them.

**Start at [`demo/fch_baseline.py`](demo/fch_baseline.py).** It is the whole driver, and
short. `build_parser()` declares every knob the simulation has; `main()` does three things —
assembles two dictionaries, optionally stamps a restart manifest, and calls the solver:

```python
IODet, SimDet = build_fch_baseline_config(...)   # what to run
run_BiV_ClosedLoop(IODet=IODet, SimDet=SimDet)   # run it
```

`IODet` is paths, `SimDet` is physics. Between them they are the entire specification, and
almost every module downstream just reads `SimDet`.

**Configuration: [`orchestrate/fch_baseline_config.py`](orchestrate/fch_baseline_config.py).**
Builds those two dictionaries by layering the canonical parameter groups in
[`calibration/params/canonical.py`](calibration/params/canonical.py) over the locked
calibration set in `configs/`, then the per-run overrides from the command line. If you want
to know where a value came from, this is the file that decides.

**The solver: [`src/sim_protocols/run_light.py`](src/sim_protocols/run_light.py).** One long
function, `run_BiV_ClosedLoop`, in the order it executes:

| | what happens |
|---|---|
| `EPmodel(...)` | electrophysiology model is constructed; this case uses homogeneous activation, so it does not drive spatial activation |
| `MEmodel(...)` | the mechanics: mesh, function spaces, material laws, boundary conditions |
| `exportfiles(...)` | opens the output streams — note it **empties the output directory** first |
| `ramp_volumes_to_pressures(...)` | the mesh is an *unloaded* reference, so the four cavities are inflated to their end-diastolic pressure targets before any heartbeat |
| `CLmodel_biv(...)` | the 0D circulation, initialised from the loaded cavity volumes |
| the time loop | alternates 0D and 3D: the circulation proposes cavity volumes, the FE solve returns the pressures that produce them |
| `write_cycle_summary_metrics(...)` | per-cycle summary at the end |

**The mechanics: [`src/mechanics/MEmodel3.py`](src/mechanics/MEmodel3.py).** The largest
file. It assembles a mixed function space — displacement, hydrostatic pressure, and one
Lagrange multiplier per cavity — and the multipliers *are* the cavity pressures. Passive
stress comes from [`forms_MRC2.py`](src/mechanics/forms_MRC2.py) (Guccione, in
[`GuccionePas.py`](src/mechanics/GuccionePas.py)), active tension from
[`activeforms_MRC2.py`](src/mechanics/activeforms_MRC2.py) (a time-varying length–tension
twitch, in [`BurkhoffTimevarying3.py`](src/mechanics/BurkhoffTimevarying3.py)), and
pericardial support from [`spring_bc_forms.py`](src/mechanics/spring_bc_forms.py).

**The circulation: [`src/sim_protocols/circBiV.py`](src/sim_protocols/circBiV.py).** Pure
Python, no FEniCS, readable end to end. Five vascular compartments and four valves;
`UpdateLVV()` advances it one step. `snapshot_valves()`/`restore_valves()` exist because the
coupled solve can back off its timestep and must be able to rewind the 0D state with it.

**How the two are coupled.** This case uses volume-prescribed coupling (`volctrl4`): the 0D
model's target volumes become constraints on the FE problem and the cavity pressures fall
out as the multipliers, so there is one nonlinear solve per step and no pressure root-find.
The alternative pressure-partitioned scheme is still in the code, but this case does not
select it.

**Outputs.** [`fch_snapshots.py`](src/sim_protocols/fch_snapshots.py) writes a full FE state
at named cardiac events (valve openings and closures, peak pressure, atrial contraction);
`output_PV.csv` gets one row per timestep; `Data.h5` accumulates the displacement field.

## Repository layout

```
demo/fch_baseline.py        entry point
orchestrate/                configuration assembly
calibration/params/         canonical parameter groups
src/sim_protocols/          coupling loop, 0D circulation, volume ramp,
                            event snapshots, restart, cycle metrics
src/mechanics/              mechanics model, material laws, Robin springs
src/utils/                  mesh I/O, Newton solver, logging
src/ep/                     electrophysiology model
heartlog/                   structured logging
configs/                    the locked calibration set (refine1/sample_0003)
inputs/                     what the run reads
reference/                  what the original run recorded, with its checksums
scripts/                    fetch, run, verify
env/environment.yml         the solver environment
data/mesh/                  fetched, not tracked
```

## Where the values are

Parameters and results are not restated in prose anywhere in this repository; each lives in
one file, and the verification scripts print what they compare.

| | |
|---|---|
| the run's resolved settings | `reference/restart_manifest.json` |
| the flags as invoked | `scripts/run_figure6.sh` |
| the arm definition | `reference/arms.json` |
| passive stiffness | `inputs/unload_fitted_passive.json` |
| the 0D circuit | `inputs/anchor_v2b_sv300pv50_rad0.7_csa1.5_params.json` |
| cardiac event times | `inputs/events.json` |
| the original run's output | `reference/output_PV.csv`, `reference/baseline_summary.csv` |
| checksums for all of it | `reference/MANIFEST.sha256` |
| the original job log | `reference/slurm_17137762_1.out` |

## Checking a run

Five scripts, roughly in the order you would use them.

**[`check_closure.py`](scripts/check_closure.py)** — walks the import graph from the entry
point and asserts the package is exactly the closure: every file reached, every import
resolving, except four modules deliberately absent (below). Useful after any edit.

**[`verify_provenance.py`](scripts/verify_provenance.py)** — `fch_restart.py` declares nine
source files whose contents can change the numerical trajectory and hashes them into
`code_provenance`. This compares that hash, the mesh checksum, and every input and reference
artifact against what the original run recorded. Seconds, no solver needed — run it before
committing compute.

**[`verify_loading.py`](scripts/verify_loading.py)** — the inflation ramp takes a while and
writes no CSV rows, so this compares the ramp itself against the original job log. It works
against a job still running and reports the magnitude of any mismatch, which distinguishes a
last-bit difference from a diverged solve.

**[`verify_trajectory.py`](scripts/verify_trajectory.py)** — compares `output_PV.csv` row by
row against the reference, tolerating a half-written trailing row so it can be run mid-flight.

**[`verify_outputs.sh`](scripts/verify_outputs.sh)** — checksums the finished outputs against
`reference/MANIFEST.sha256`. It counts XDMF index files separately from the artifacts that
carry values, because index files are identical between runs that agree on nothing.

`VALIDATION.md` records how far this has been carried, including where reproduction is exact
and where it is only exact to floating-point round-off.

### A note on VTK output

The run writes four PVD files, but they are single-frame dumps of the mesh and its region
markers, written once at startup — no displacement, no time axis. The solution never goes to
VTK: it goes to `Data.h5`, to XDMF event snapshots, and to CSV. If you want the deformation
in a viewer, read the event snapshots, not the PVD files.

## Scope

Four modules present upstream are absent here, because the configuration cannot reach them:
`holzapfelogden` and `GuccioneAct` (this case selects the Guccione passive and time-varying
active laws), `rigid_support`, and `transverse_config` (cross-fibre activation). Their import
statements remain inside untaken branches and will raise `ImportError` if a modified
configuration reaches them — this package runs one case rather than serving as a general
solver.

The nine files named in `fch_restart.py::_CODE_FILES` are byte-identical to the original,
paths included, and are not tidied even where they contain code this case never runs. They
are the input to `code_provenance`, so any edit breaks the link between this checkout and
the recorded result.

`src/mechanics/lv_spring_strain.py` and `orchestrate/lv_baseline_config.py` read as
left-ventricle leftovers but are genuine dependencies of the four-chamber strain emitter.

## Practical notes

* Legacy **FEniCS/DOLFIN 2019.1.0**, not DOLFINx. The run scripts assert this on startup.
* `exportfiles` empties its output directory when constructed. Do not point `OUT` at a
  directory holding anything you want to keep.
* `FCH_MESH_SCALE` converts the mesh from millimetres to centimetres — a unit conversion,
  not a tunable.
* `demo/fch_baseline.py` imports the repository as `heArt_legacy`; `run_figure6.sh` provides
  that through a symlinked import root rather than by editing any source file.
* Floating-point results depend on the CPU microarchitecture, so `run_figure6.slurm` pins
  the one the original run used. On different hardware expect agreement to round-off rather
  than byte-for-byte.
