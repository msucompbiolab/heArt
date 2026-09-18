"""Per-step FOUR-CHAMBER strain emitter (``output_strain.csv``) for ``run_light``.

LEGACY FEniCS ONLY. The four-chamber analogue of ``strain_emit.StrainEmitter`` (LV+aorta), built
on the per-chamber kernels that already exist in ``src/mechanics/fch_spring_strain.py`` --
``chamber_geometry`` (endo-surface centroid / long axis / cylindrical (eCC,eRR,eLL) basis per
chamber, computed ONCE on the reference mesh), ``chamber_strains`` (volume-averaged natural
strain E = 1/2(1 - 1/C_e) over the chamber's matid region), ``annular_descent`` (MAPSE/TAPSE),
``chamber_twist``, ``surface_excursion`` -- so a coupled run and the spring-tuning probes report
the SAME definitions. Nothing is reimplemented here.

Two reference frames are written side by side, exactly as the LV+aorta campaign reports its
emitter convention beside the study-matched one:
  * ``*_unl``  -- referred to the mesh (the solver's reference; the manuscript's stated
                  definition, lambda_i = sqrt(e.C.e), eps = 1/2(1 - 1/lambda^2));
  * ``*_ED``   -- referred to the end-diastolic state captured when the driver defines
                  ``u_me_ED`` (F_rel = F F_ED^-1). Under the no-preload policy the reference IS
                  the ED state, so the two coincide until the loop's own ED drifts; both are
                  still written so a reader never has to know which policy produced the file.

Cadence: every ``strain_write_step`` coupled steps (default = ``writeStep``); each row costs
~20 assemblies over the whole mesh, which is why it is not written every timestep. Rank 0 owns
the file; every assemble is collective, so ``write_row`` must be called on ALL ranks.
"""
from __future__ import annotations

import csv
import os
from typing import Any, Dict, List, Optional

import dolfin as df

from ..mechanics import fch_spring_strain as K

CHAMBERS = ("LV", "RV", "LA", "RA")
MMHG_PER_PA = 0.0075


class FchStrainEmitter:
    def __init__(self, ME, out_dir: str, *, length_factor_mm: float, log=None,
                 filename: str = "output_strain.csv"):
        self.ME = ME
        self.comm = ME.mesh_me.mpi_comm()
        self.is_root = df.MPI.rank(self.comm) == 0
        self.length_factor_mm = float(length_factor_mm)   # mesh units -> mm (0.1 cm mesh: 10)
        self.log = log
        self.geom = K.chamber_geometry(ME)
        self.F_ED = None
        self.path = os.path.join(out_dir, filename)
        self._fh = None
        self._writer = None
        self.columns: List[str] = ["tstep", "t", "cycle"]
        for ch in CHAMBERS:
            for ref in ("unl", "ED"):
                self.columns += ["%s_Ell_%s" % (ch, ref), "%s_Ecc_%s" % (ch, ref),
                                 "%s_Err_%s" % (ch, ref)]
        self.columns += ["LV_MAPSE_mm", "RV_TAPSE_mm", "LV_twist_deg", "RV_twist_deg",
                         "LA_excursion_mm", "RA_excursion_mm",
                         "P_LV_mmhg", "P_RV_mmhg", "P_LA_mmhg", "P_RA_mmhg",
                         "V_LV_ml", "V_RV_ml", "V_LA_ml", "V_RA_ml"]
        if self.is_root:
            os.makedirs(out_dir, exist_ok=True)
            self._fh = open(self.path, "w", newline="")
            self._writer = csv.DictWriter(self._fh, fieldnames=self.columns)
            self._writer.writeheader()
        if log is not None:
            log.info("init", "four-chamber strain emitter ON", path=self.path,
                     **{("axis_%s" % ch.lower()): "endo-PCA" for ch in CHAMBERS})

    def capture_ed(self) -> None:
        """Freeze the current deformation gradient as the ED reference (F_ED)."""
        # A COPY of the current F as a Function on the DG0 tensor space so it does not track u.
        F = self.ME.get_deformation_gradient()
        T = df.TensorFunctionSpace(self.ME.mesh_me, "DG", 0)
        self.F_ED = df.project(F, T, solver_type="cg", preconditioner_type="jacobi")

    def row(self, tstep: float, t: float, cycle: int) -> Dict[str, Any]:
        ME = self.ME
        r: Dict[str, Any] = {"tstep": tstep, "t": t, "cycle": int(cycle)}
        for ch in CHAMBERS:
            g = self.geom[ch]
            gls, gcs, grs = K.chamber_strains(ME, g, F_ref=None)
            r["%s_Ell_unl" % ch], r["%s_Ecc_unl" % ch], r["%s_Err_unl" % ch] = gls, gcs, grs
            if self.F_ED is not None:
                gls, gcs, grs = K.chamber_strains(ME, g, F_ref=self.F_ED)
            r["%s_Ell_ED" % ch], r["%s_Ecc_ED" % ch], r["%s_Err_ED" % ch] = gls, gcs, grs
        u = ME.get_displacement()
        lf = self.length_factor_mm
        r["LV_MAPSE_mm"] = K.annular_descent(ME, u, self.geom["LV"], lf)
        r["RV_TAPSE_mm"] = K.annular_descent(ME, u, self.geom["RV"], lf)
        r["LV_twist_deg"] = K.chamber_twist(ME, u, self.geom["LV"])
        r["RV_twist_deg"] = K.chamber_twist(ME, u, self.geom["RV"])
        r["LA_excursion_mm"] = K.surface_excursion(ME, u, self.geom["LA"]["endoid"], lf)
        r["RA_excursion_mm"] = K.surface_excursion(ME, u, self.geom["RA"]["endoid"], lf)
        r["P_LV_mmhg"] = ME.get_lv_pressure() * MMHG_PER_PA
        r["P_RV_mmhg"] = ME.get_rv_pressure() * MMHG_PER_PA
        r["P_LA_mmhg"] = ME.get_la_pressure() * MMHG_PER_PA
        r["P_RA_mmhg"] = ME.get_ra_pressure() * MMHG_PER_PA
        r["V_LV_ml"] = ME.get_lv_volume()
        r["V_RV_ml"] = ME.get_rv_volume()
        r["V_LA_ml"] = ME.get_la_volume()
        r["V_RA_ml"] = ME.get_ra_volume()
        return r

    def write_row(self, tstep: float, t: float, cycle: int, pressure_offsets_pa: Optional[Dict[str, float]] = None) -> None:
        """Collective. ``pressure_offsets_pa`` adds the no-preload reference offsets to the
        reported pressures so the CSV matches ``output_PV.csv``."""
        r = self.row(tstep, t, cycle)
        if pressure_offsets_pa:
            for ch in ("lv", "rv", "la", "ra"):
                r["P_%s_mmhg" % ch.upper()] += float(pressure_offsets_pa.get(ch, 0.0)) * MMHG_PER_PA
        if self.is_root and self._writer is not None:
            self._writer.writerow({k: (float(v) if isinstance(v, (int, float)) else v)
                                   for k, v in r.items()})
            self._fh.flush()

    def close(self) -> None:
        if self._fh is not None:
            self._fh.close()
            self._fh = None
