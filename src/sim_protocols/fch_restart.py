"""FCH-contained checkpoint/restart for the four-chamber forward run.

The shared ``exportfiles`` HDF5 stream (``Data.h5``) is a VISUALIZATION output: it is
opened ``"w"`` (truncated) on every construction, INCLUDING restart, and before the
driver's restart-read -- so it cannot hold a survivable checkpoint. This module gives
``run_light`` a SEPARATE, self-contained restart checkpoint that the shared layer never
touches, so geometry + cavity-pressure/Lagrange-multiplier state restore without the
truncation/indexing bugs. It does not change exportfiles, Data.h5, or the LV pipeline.

Two files live beside ``Data.h5`` in the run's output dir:
  * ``restart_state.h5``  -- the full mixed mechanics state ``w_me`` (mixed displacement +
    hydrostatic pressure + cavity Lagrange multipliers), INDEXED by checkpoint count and
    appended (never truncated on restart).
  * ``restart_state.csv`` -- ONE row per checkpoint with the matching 0D/CL + time state
    (tstep/t/cycle, chamber + vascular volumes, chamber pressures). Written together with
    each ``w_me`` index so FE and CL state at restore are from the SAME tstep -- no
    write-step cadence mismatch, hence no discontinuity (this EXCEEDS the LV semantics,
    which restore FE from the last write-step but CL from the last PV row).

Restart is allowed ONLY for a same-parameter continuation. ``restart_manifest.json`` records
the restart-DEFINING inputs (mesh + unloaded ref identity, per-chamber material, loading
EDP targets, dt + solver knobs, viscous eta, backoff, stop_iter, atrial convention, and the
trajectory-defining code provenance). :func:`validate_restart_manifest` REFUSES the restart
(raises) if any of them changed -- those cases must rerun from a self-consistent initial
state. Forward-only outputs (writeStep, output paths) are excluded.
"""

import csv
import hashlib
import json
import os
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import dolfin as df

CKPT_H5 = "restart_state.h5"
CKPT_CSV = "restart_state.csv"
MANIFEST = "restart_manifest.json"

# CL/time scalars persisted per checkpoint (the 0D state run_light needs to resume).
_CL_FIELDS = (
    "tstep", "t", "cycle",
    "V_LV", "V_RV", "V_LA", "V_RA",
    "V_sa", "V_ad", "V_sv", "V_pa", "V_pv",
    "P_LV", "P_RV", "P_LA", "P_RA",
)

# Trajectory-defining source files: ANY change refuses a restart (conservative + correct --
# a code edit may alter the numerical trajectory, so the checkpoint is no longer continuable).
_CODE_FILES = (
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


def _repo_root() -> Path:
    # src/sim_protocols/fch_restart.py -> repo root is parents[2]
    return Path(__file__).resolve().parents[2]


def code_provenance() -> str:
    """md5 over the trajectory-defining source files (sorted, missing files noted). Changes
    iff any FE/0D/material/config code that could alter the trajectory changed."""
    h = hashlib.md5()
    root = _repo_root()
    for rel in sorted(_CODE_FILES):
        p = root / rel
        h.update(rel.encode())
        h.update(b"=")
        h.update(p.read_bytes() if p.exists() else b"<absent>")
        h.update(b"\n")
    return h.hexdigest()


def _file_md5(path) -> str:
    """md5 of a file's bytes ('' when absent) -- for fingerprinting trace/axis inputs."""
    if not path:
        return ""
    p = Path(str(path))
    if not p.exists():
        return ""
    h = hashlib.md5()
    with open(p, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def restart_fingerprint(IODet: dict, SimDet: dict) -> Dict[str, Any]:
    """The restart-DEFINING inputs. Two runs with equal fingerprints have the same numerical
    trajectory and are continuable; any difference must rerun fresh. Forward-only output knobs
    (writeStep, output paths) are deliberately excluded."""
    g = SimDet.get("GiccioneParams", {})
    pp = g.get("Passive params", {})
    md = str(IODet.get("directory_me", ""))
    mesh_md5 = ""
    basename = SimDet.get("mesh_basename", IODet.get("casename", "fch_clregion"))
    mesh_file = Path(md) / f"{basename}.hdf5"
    if mesh_file.exists():
        h = hashlib.md5()
        with open(mesh_file, "rb") as fh:
            for chunk in iter(lambda: fh.read(1 << 20), b""):
                h.update(chunk)
        mesh_md5 = h.hexdigest()
    fp = {
        "schema": 1,
        "mesh_dir": md,
        "mesh_basename": basename,
        "mesh_md5": mesh_md5,
        "mesh_scale_fch": SimDet.get("mesh_scale_fch"),
        # per-chamber material (the 8 dims as applied to the run)
        "Cparam_lv": SimDet.get("Cparam_lv"), "Cparam_rv": SimDet.get("Cparam_rv"),
        "Cparam_la": SimDet.get("Cparam_la"), "Cparam_ra": SimDet.get("Cparam_ra"),
        "Tmax_lv": SimDet.get("Tmax_lv"), "Tmax_rv": SimDet.get("Tmax_rv"),
        "Tmax_la": SimDet.get("Tmax_la"), "Tmax_ra": SimDet.get("Tmax_ra"),
        "active_l0_um": g.get("Active params", {}).get("l0"),
        # loading targets
        "EDP": SimDet.get("EDP"), "nLoadSteps": SimDet.get("nLoadSteps"),
        "P_LV": SimDet.get("P_LV"), "P_RV": SimDet.get("P_RV"),
        "P_LA": SimDet.get("P_LA"), "P_RA": SimDet.get("P_RA"),
        "fch_atria_unstressed_ed": SimDet.get("fch_atria_unstressed_ed"),
        # time-step + solver knobs that alter the trajectory
        "dt": SimDet.get("dt"), "HeartBeatLength": SimDet.get("HeartBeatLength"),
        "dt_systole": SimDet.get("dt_systole"), "dt_relax": SimDet.get("dt_relax"),
        "dt_filling": SimDet.get("dt_filling"), "dt_diastasis": SimDet.get("dt_diastasis"),
        "dt_relax_n_tau": SimDet.get("dt_relax_n_tau"),
        "passive_viscous": bool(SimDet.get("_passive_viscous_")),
        "eta": pp.get("eta", 0.0),
        "backoff_min_factor": SimDet.get("backoff_min_factor"),
        "stop_iter": SimDet.get("closedloopparam", {}).get("stop_iter"),
        # coupling scheme + reference convention (fch_coupled campaign): a run under
        # volume-prescribed coupling, an open-loop trace, a no-preload ED reference, a
        # reference-pressure offset or a cross-fibre active law is a DIFFERENT trajectory.
        "fch_coupling_mode": SimDet.get("fch_coupling_mode", "partitioned_pressure"),
        "fch_swept_volctrl": SimDet.get("fch_swept_volctrl"),
        "fch_no_preload": bool(SimDet.get("fch_no_preload", False)),
        "fch_reference_pressure_pa": SimDet.get("fch_reference_pressure_pa"),
        "fch_openloop_trace_md5": _file_md5(SimDet.get("fch_openloop_trace")),
        "fch_ed_volumes_ml": SimDet.get("fch_ed_volumes_ml"),
        "transverse_active_fraction": SimDet.get("transverse_active_fraction", 0.0),
        "transverse_active_structure": SimDet.get("transverse_active_structure"),
        "transverse_active_inplane_angle_md5": _file_md5(
            SimDet.get("transverse_active_inplane_angle")),
        "active_strain": bool(SimDet.get("active_strain", False)),
        "active_strain_gamma": {ch: SimDet.get("active_strain_gamma_%s" % ch)
                                for ch in ("lv", "rv", "la", "ra")},
        # closed-loop circuit (vascular R/C + valves drive the trajectory)
        "closedloop_md5": hashlib.md5(
            json.dumps({k: v for k, v in sorted(SimDet.get("closedloopparam", {}).items())
                        if k != "stop_iter"}, sort_keys=True, default=str).encode()
        ).hexdigest(),
        "code_provenance": code_provenance(),
    }
    return fp


def write_restart_manifest(output_dir: os.PathLike, IODet: dict, SimDet: dict) -> Path:
    """Stamp the restart manifest for a FRESH run (rank-0 caller only)."""
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    p = out / MANIFEST
    p.write_text(json.dumps(restart_fingerprint(IODet, SimDet), indent=2, default=str,
                            sort_keys=True) + "\n", encoding="utf-8")
    return p


def validate_restart_manifest(output_dir: os.PathLike, IODet: dict, SimDet: dict) -> None:
    """Refuse (raise) a restart whose checkpoint was produced with DIFFERENT restart-defining
    inputs -- those must rerun fresh. Names the changed fields. Missing manifest => refuse."""
    out = Path(output_dir)
    p = out / MANIFEST
    if not p.exists():
        raise RuntimeError(
            f"[fch-restart] REFUSED: no restart manifest at {p}. Cannot verify the checkpoint "
            f"is a same-parameter continuation -- rerun fresh (without --restart).")
    stored = json.loads(p.read_text(encoding="utf-8"))
    current = restart_fingerprint(IODet, SimDet)
    diffs = []
    for k in sorted(set(stored) | set(current)):
        if str(stored.get(k)) != str(current.get(k)):
            diffs.append(f"  {k}: checkpoint={stored.get(k)!r} now={current.get(k)!r}")
    if diffs:
        raise RuntimeError(
            "[fch-restart] REFUSED: restart-defining inputs changed since the checkpoint "
            "(a tuning/material/mesh/code change alters the numerical trajectory; rerun fresh "
            "from a self-consistent initial state). Changed:\n" + "\n".join(diffs))


class FCHCheckpoint:
    """Manages the dedicated FCH restart checkpoint (``restart_state.h5`` + ``restart_state.csv``).

    Fresh run: opens ``restart_state.h5`` ``"w"`` (the prior file was already removed by
    exportfiles.removeAllfiles on a fresh run) and truncates the CSV. Restart: opens ``"a"``
    (preserve + append) so a further restart is possible. ``restore`` reads the LAST mixed
    state + its matching CL/time row (same tstep). Rank-0 owns the CSV.
    """

    def __init__(self, output_dir: os.PathLike, comm, isrestart: bool):
        self.dir = Path(output_dir)
        self.comm = comm
        self.isrestart = bool(isrestart)
        self.h5_path = str(self.dir / CKPT_H5)
        self.csv_path = str(self.dir / CKPT_CSV)
        self.chk = 0
        self._rank0 = (df.MPI.rank(comm) == 0)
        if self.isrestart:
            if not os.path.exists(self.h5_path):
                raise RuntimeError(
                    f"[fch-restart] isrestart set but no checkpoint at {self.h5_path}; "
                    f"run a fresh case first.")
            self.hdf = df.HDF5File(comm, self.h5_path, "a")  # read/write, NO truncate
        else:
            self.hdf = df.HDF5File(comm, self.h5_path, "w")
            if self._rank0:
                with open(self.csv_path, "w", newline="") as fh:
                    csv.writer(fh).writerow(["chk", *_CL_FIELDS])

    def restore(self, w_me) -> Dict[str, float]:
        """Read the latest mixed FE state into ``w_me`` and return the matching CL/time dict.
        Sets the next checkpoint index to continue. Raises if the checkpoint is empty.

        Each checkpoint is a DISTINCT dataset ``ME/w_me_<i>`` (not a DOLFIN time-series --
        the series API needs a 'count' attribute and conflates write-time vs read-index);
        existence is probed with ``has_dataset`` (clean, no error spam)."""
        widx = 0
        while self.hdf.has_dataset("ME/w_me_%d" % widx):
            widx += 1
        if widx == 0:
            raise RuntimeError(
                f"[fch-restart] checkpoint {self.h5_path} has no ME/w_me state to resume from.")
        self.hdf.read(w_me, "ME/w_me_%d" % (widx - 1))
        self.chk = widx
        # CL/time row matching the last FE checkpoint (same tstep -> consistent).
        cl: Dict[str, float] = {}
        if os.path.exists(self.csv_path):
            with open(self.csv_path, "r", newline="") as fh:
                rows = list(csv.DictReader(fh))
            # prefer the row whose chk == widx-1; else the last row
            match = [r for r in rows if str(r.get("chk")) == str(widx - 1)]
            row = match[-1] if match else (rows[-1] if rows else {})
            for k in _CL_FIELDS:
                if k in row and row[k] != "":
                    cl[k] = float(row[k])
        return cl

    def write(self, w_me, cl: Dict[str, float]) -> None:
        """Append one checkpoint: the mixed FE state + the matching CL/time row (same tstep)."""
        self.hdf.write(w_me, "ME/w_me_%d" % self.chk)
        try:
            self.hdf.flush()
        except Exception:
            pass
        if self._rank0:
            with open(self.csv_path, "a", newline="") as fh:
                csv.writer(fh).writerow([self.chk, *[cl.get(k, "") for k in _CL_FIELDS]])
        self.chk += 1

    def close(self) -> None:
        try:
            self.hdf.close()
        except Exception:
            pass


def truncate_pv_to_tstep(csv_path: os.PathLike, tstep: float) -> None:
    """Drop PV rows AFTER the restart tstep so the appended continuation is a single
    continuous trace (rank-0 caller only). No-op if the file/column is absent."""
    p = Path(csv_path)
    if not p.exists():
        return
    with open(p, "r", newline="") as fh:
        reader = csv.reader(fh)
        rows = list(reader)
    if not rows:
        return
    header = rows[0]
    try:
        ti = header.index("tstep")
    except ValueError:
        return
    kept = [header]
    for r in rows[1:]:
        try:
            if float(r[ti]) <= float(tstep) + 1e-9:
                kept.append(r)
        except (ValueError, IndexError):
            kept.append(r)
    with open(p, "w", newline="") as fh:
        csv.writer(fh).writerows(kept)
