import csv
import math
import os
import time
import warnings
from collections import deque

import numpy as np
import dolfin as df
from ffc.quadrature.deprecation import QuadratureRepresentationDeprecationWarning
from mpi4py import MPI as pyMPI

warnings.simplefilter("ignore", QuadratureRepresentationDeprecationWarning)
# Suppress verbose DOLFIN subset iteration notices that clutter stdout in profiling runs.
df.set_log_level(df.LogLevel.ERROR)

from ..ep.EPmodel import EPmodel
from ..mechanics.MEmodel3 import MEmodel, fch_active_tmax_factor
from ..utils.log_mpi import get_logger
from ..utils.oops_objects_MRC2 import State_Variables, exportfiles, printout
from ..utils.periodicity_logging import (
    append_periodicity_rows,
    write_periodicity_metrics_json,
)
from .circBiV import CLmodel as CLmodel_biv
from .fch_openloop import VolumeTrace
from .fch_restart import FCHCheckpoint, truncate_pv_to_tstep
from .fch_snapshots import EventSnapshotWriter, load_event_times
from .fch_strain_emit import FchStrainEmitter
from .run_light_metrics import (
    evaluate_full_waveform_periodicity,
    evaluate_full_waveform_periodicity_from_buffers,
    evaluate_pvloop_periodicity,
    evaluate_pvloop_periodicity_from_buffers,
    write_cycle_summary_metrics,
)


def run_BiV_ClosedLoop(IODet, SimDet):
    if "fiber_fspace_deg" in SimDet:
        deg = SimDet["fiber_fspace_deg"]
    else:
        deg = 4
    flags = ["-O3", "-ffast-math", "-march=native"]
    # Quadrature elements are used heavily; keep compiler settings explicit/reproducible.
    df.parameters["form_compiler"]["representation"] = "uflacs"
    df.parameters["form_compiler"]["quadrature_degree"] = deg

    casename = IODet["casename"]
    directory_me = IODet["directory_me"]
    directory_ep = IODet["directory_ep"]
    outputfolder = IODet["outputfolder"]
    folderName = os.path.join(IODet["folderName"], IODet["caseID"])
    output_dir = os.path.join(outputfolder, folderName)
    case_id = str(IODet.get("caseID", casename))
    delTat = SimDet["dt"]
    matid_dataset = SimDet.get("matid_dataset", IODet.get("matid_dataset", "matid1"))
    SimDet.setdefault("matid_dataset", matid_dataset)

    # --- Read EP data from HDF5 files ---
    mesh_ep = df.Mesh()
    comm_common = mesh_ep.mpi_comm()

    meshfilename_ep = directory_ep + casename + "_refine.hdf5"
    f = df.HDF5File(comm_common, meshfilename_ep, "r")
    f.read(mesh_ep, casename, False)
    # EP mesh scale: the FCH refine mesh is an exact copy of the ME mesh, so EP uses
    # the SAME scale as ME -- FCH_MESH_SCALE env first, then SimDet["mesh_scale_fch"]
    # (mirrors the ME loader, oops_objects_MRC2.fch_mesh). The legacy hardcoded 6.5e-2
    # left EP and ME at different physical sizes: harmless under HomogenousActivation
    # (phi is not spatially mapped to ME) but WRONG for spatial EP->ME activation.
    # Override with mesh_scale_ep_fch only if a distinct EP scale is genuinely needed.
    _ep_scale_env = os.environ.get("FCH_MESH_SCALE")
    try:
        _ep_scale = float(
            SimDet.get(
                "mesh_scale_ep_fch",
                _ep_scale_env if _ep_scale_env is not None
                else SimDet.get("mesh_scale_fch", 6.5e-2),
            )
        )
    except (TypeError, ValueError):
        _ep_scale = 6.5e-2
    _homog_act = SimDet.get("GiccioneParams", {}).get("HomogenousActivation", False)
    if abs(_ep_scale - 6.5e-2) > 1e-12 and not _homog_act:
        printout(
            f"WARNING: FCH EP mesh scaled by {_ep_scale} (legacy hardcode was 6.5e-2) "
            f"with HomogenousActivation OFF -- spatial EP->ME activation requires the "
            f"EP and ME geometries to match; verify the EP refine-mesh scale.",
            comm_common,
        )
    mesh_ep.scale(_ep_scale)

    df.File(os.path.join(output_dir, "mesh_ep.pvd")) << mesh_ep

    facetboundaries_ep = df.MeshFunction("size_t", mesh_ep, 2)
    f.read(facetboundaries_ep, casename + "/" + "facetboundaries")

    matid_ep = df.MeshFunction("size_t", mesh_ep, mesh_ep.topology().dim())
    AHAid_ep = df.MeshFunction("size_t", mesh_ep, mesh_ep.topology().dim())

    matid_ep_dataset = f"{casename}/{matid_dataset}"
    if f.has_dataset(matid_ep_dataset):
        if SimDet.get("function_matid"):
            # Optional: derive region ids procedurally by sampling a DG0 "matid" function.
            VQuadelem = df.FiniteElement(
                "DG", mesh_ep.ufl_cell(), degree=0, quad_scheme="default"
            )
            matid_FS = df.FunctionSpace(mesh_ep, VQuadelem)
            matid_func = dolfin.df.Function(matid_FS)
            for cell in df.cells(mesh_ep):
                matid_ep[cell.index()] = round(matid_func(cell.midpoint()))
        else:
            f.read(matid_ep, matid_ep_dataset)
    else:
        # Default to a single material region if the dataset is absent.
        matid_ep.set_all(0)

    df.File(os.path.join(output_dir, "matid_ep.pvd")) << matid_ep

    if f.has_dataset(casename + "/" + "AHAid"):
        f.read(AHAid_ep, casename + "/" + "AHAid")
    else:
        AHAid_ep.set_all(0)

    deg_ep = 4

    if "fiber_fspace" in SimDet and "fiber_fspace_deg" in SimDet:
        # Allow overriding the fiber field space for compatibility with different preprocessors.
        VQuadelem_ep = df.VectorElement(
            SimDet["fiber_fspace"],
            mesh_ep.ufl_cell(),
            degree=SimDet["fiber_fspace_deg"],
            quad_scheme="default",
        )
        VQuadelem_ep._quad_scheme = "default"
    else:
        VQuadelem_ep = df.VectorElement(
            "Quadrature", mesh_ep.ufl_cell(), degree=deg_ep, quad_scheme="default"
        )
        VQuadelem_ep._quad_scheme = "default"

    fiberFS_ep = df.FunctionSpace(mesh_ep, VQuadelem_ep)

    f0_ep = df.Function(fiberFS_ep)
    s0_ep = df.Function(fiberFS_ep)
    n0_ep = df.Function(fiberFS_ep)

    if SimDet["DTI_EP"] is True:
        f.read(f0_ep, casename + "/" + "eF_DTI")
        f.read(s0_ep, casename + "/" + "eS_DTI")
        f.read(n0_ep, casename + "/" + "eN_DTI")
    else:
        # Use the SAME fiber datasets as the mechanics model (SimDet["fiber_datasets"];
        # default = whole-heart eF_rb/eS_rb/eN_rb). The EP _refine mesh is an exact copy
        # of the ME mesh, so the identical whole-heart fibers are present here. They are
        # DG0-stored, so read into a DG0 space and assign into the EP fiber space (mirrors
        # oops _load_fiber_field; DG0->Quadrature assign is supported in legacy dolfin).
        _ep_fiber_names = SimDet.get("fiber_datasets") or {"f0": "eF", "s0": "eS", "n0": "eN"}
        _ep_storage_fs = df.VectorFunctionSpace(mesh_ep, "DG", 0)
        for _tgt, _key, _dflt in ((f0_ep, "f0", "eF"), (s0_ep, "s0", "eS"), (n0_ep, "n0", "eN")):
            _tmp = df.Function(_ep_storage_fs)
            f.read(_tmp, casename + "/" + _ep_fiber_names.get(_key, _dflt))
            _tgt.assign(_tmp)

    f.close()

    comm_ep = mesh_ep.mpi_comm()

    # --- Define state variables ---
    # Shared time/cycle bookkeeping used by both EP and mechanics.
    state_obj = State_Variables(comm_ep, SimDet)
    state_obj.dt.dt = delTat

    EPparams = {
        "EPmesh": mesh_ep,
        "deg": 4,
        "matid": matid_ep,
        "facetboundaries": facetboundaries_ep,
        "f0": f0_ep,
        "s0": s0_ep,
        "n0": n0_ep,
        "state_obj": state_obj,
        "d_iso": SimDet["d_iso"],
        "d_ani_factor": SimDet["d_ani_factor"],
        "AHAid": AHAid_ep,
        "matid": matid_ep,
    }

    if "ploc" in SimDet:
        EPparams.update({"ploc": SimDet["ploc"]})
    if "Ischemia" in SimDet:
        EPparams.update({"Ischemia": SimDet["Ischemia"]})
    if "pacing_timing" in SimDet:
        EPparams.update({"pacing_timing": SimDet["pacing_timing"]})

    # Define EP model and solver
    EPmodel_ = EPmodel(EPparams)
    EpiBCid_ep = EPmodel_.MarkStimulus()

    # --- Initialize mechanics mesh ---

    mesh_me = df.Mesh()
    mesh_me_params = {
        "directory": directory_me,
        "casename": casename,
        "fibre_quad_degree": 4,
        "outputfolder": outputfolder,
        "foldername": folderName + "/",
        "state_obj": state_obj,
        "common_communicator": comm_common,
        "MEmesh": mesh_me,
        "matid_dataset": matid_dataset,
    }

    MEmodel_ = MEmodel(mesh_me_params, SimDet)
    solver_elas = MEmodel_.Solver()
    comm_me = MEmodel_.mesh_me.mpi_comm()

    # Ensure output directory exists once (rank 0) before any file writes
    try:
        if pyMPI.COMM_WORLD.rank == 0:
            os.makedirs(output_dir, exist_ok=True)
    except Exception:
        pass

    # Set up export class
    export = exportfiles(comm_me, comm_ep, IODet, SimDet)

    export.exportVTKobj("facetboundaries_ep.pvd", facetboundaries_ep)
    export.exportVTKobj("EpiBCid_ep.pvd", EpiBCid_ep)

    # Get Unloaded volumes
    V_LV_unload = MEmodel_.get_lv_volume()
    V_RV_unload = MEmodel_.get_rv_volume()

    printout("V_LV_unload = " + str(V_LV_unload), comm_me)
    printout("V_RV_unload = " + str(V_RV_unload), comm_me)

    nloadstep = SimDet["nLoadSteps"]

    # --- Coupling scheme + reference convention (fch_coupled campaign) -----------------
    # fch_coupling_mode:
    #   "partitioned_pressure" (default, byte-identical legacy): the 4-D cavity-pressure
    #       root-find `Rp_fch` below.
    #   "volctrl4": the four 0D target volumes are PRESCRIBED to the monolithic swept-cavity
    #       Lagrange constraints (SimDet["fch_swept_volctrl"]="lv,rv,la,ra"); ONE nonlinear
    #       solve per step, each cavity pressure read off its multiplier. The volume-prescribed
    #       branch passes through the ejection limit point at which pressure control folds
    #       (the HFpEF LV+aorta closed loop, run_waorta `lvw_swept_volctrl`).
    #   "openloop_volume": as volctrl4, but the chamber volumes are READ from an established
    #       0D PV trace (SimDet["fch_openloop_trace"]) instead of advancing the circulation;
    #       FE pressures are reported, never fed back (PRELIMINARY, one-way coupling).
    # fch_no_preload: the reference mesh IS the end-diastolic state -- skip the inflation
    #       loading phase (u = 0 at ED), assert the FE cavity volumes against the ED-state
    #       manifest (SimDet["fch_ed_volumes_ml"]), and treat the ED geometry as loaded at
    #       per-chamber EDP via a uniform cavity-pressure OFFSET:
    # fch_reference_pressure_pa: {lv,rv,la,ra} -> reported pressure = FE load + offset (the
    #       FE load being the prescribed-pressure Constant, or the swept multiplier).
    _coupling_mode = str(SimDet.get("fch_coupling_mode", "partitioned_pressure"))
    if _coupling_mode not in ("partitioned_pressure", "volctrl4", "openloop_volume"):
        raise ValueError("fch_coupling_mode=%r; expected partitioned_pressure | volctrl4 | "
                         "openloop_volume" % _coupling_mode)
    _volctrl = _coupling_mode in ("volctrl4", "openloop_volume")
    _openloop = _coupling_mode == "openloop_volume"
    _no_preload = bool(SimDet.get("fch_no_preload", False))
    _ref_p = {ch: float((SimDet.get("fch_reference_pressure_pa") or {}).get(ch, 0.0) or 0.0)
              for ch in ("lv", "rv", "la", "ra")}

    def _report_p(ch, p_fe):
        """Cavity pressure as the circulation/CSV sees it: FE load + reference offset."""
        return float(p_fe) + _ref_p[ch]

    def _fe_p(ch, p_reported):
        """The FE load that realises a reported cavity pressure."""
        return float(p_reported) - _ref_p[ch]

    _swept_vols = None
    _vol_tol = float(SimDet.get("fch_volctrl_vol_tol_ml", 1.0e-3))
    if _volctrl:
        _chs = sorted(MEmodel_.fch_swept_chambers())
        if _chs != ["la", "lv", "ra", "rv"]:
            raise RuntimeError(
                "fch_coupling_mode=%s needs SimDet['fch_swept_volctrl']='lv,rv,la,ra' (all four "
                "cavities under monolithic volume control); the model has %r" % (_coupling_mode, _chs))
        if SimDet.get("auto_rigid_support"):
            raise RuntimeError("fch_coupling_mode=%s + auto_rigid_support is not wired (the LVW "
                               "volume-control path refuses it for the same reason)" % _coupling_mode)
        if not SimDet.get("fch_fe", True):
            raise RuntimeError("fch_coupling_mode=%s requires fch_fe coupling (all four chambers FE)"
                               % _coupling_mode)
        _swept_vols = MEmodel_.FCHsweptCavityvols
        printout("fch coupling: VOLUME-prescribed four-chamber coupling (%s): the 0D target "
                 "volumes go to four monolithic swept-cavity constraints, one nonlinear solve "
                 "per step, cavity pressures = the multipliers (no pressure root-find)"
                 % _coupling_mode, comm_me, stage="init", level="INFO",
                 coupling_mode=_coupling_mode, vol_tol_ml=_vol_tol)
    _trace = None
    if _openloop:
        _trace_path = SimDet.get("fch_openloop_trace")
        if not _trace_path:
            raise ValueError("fch_coupling_mode=openloop_volume requires SimDet['fch_openloop_trace']")
        _trace = VolumeTrace(str(_trace_path), float(state_obj.BCL))
        _td = _trace.describe()
        printout("WARNING: OPEN-LOOP volume driving -- the 0D circulation is NOT advanced; chamber "
                 "volumes are replayed from %s (last complete cycle, %d rows) and the FE pressures "
                 "are reported but never fed back. Every output of this run is PRELIMINARY "
                 "(one-way coupling), not a coupled result." % (_td["path"], _td["cycle_rows"]),
                 comm_me, stage="init", level="WARN",
                 **{("v_%s_start_ml" % ch.lower()): round(v, 3)
                    for ch, v in _td["cycle_start_volumes_ml"].items()})
    if _no_preload:
        printout("WARNING: NO-PRELOAD policy (ed_reference_no_preload): the reference mesh is "
                 "taken AS the end-diastolic state (u = 0 at ED, no inflation); the ED geometry is "
                 "treated as loaded at per-chamber EDP through a uniform cavity-pressure offset. "
                 "This is an in-vivo-reference simplification (residual stress lumped into the "
                 "offset), not a recovered zero-pressure reference.",
                 comm_me, stage="loading", level="WARN",
                 **{("p_ref_%s_mmhg" % ch): round(v * 0.0075, 3) for ch, v in _ref_p.items()})
    elif any(abs(v) > 0.0 for v in _ref_p.values()):
        raise ValueError("fch_reference_pressure_pa is only meaningful under fch_no_preload "
                         "(an inflated reference already carries its EDP)")

    # Quasi-static pressurization to the requested EDP to establish a loaded reference state.
    MEmodel_.LVCavityvol.assign(MEmodel_.get_lv_volume())
    MEmodel_.LVCavitypres.assign(0.0)
    MEmodel_.RVCavitypres.assign(0.0)
    MEmodel_.AortaCavitypres.assign(0.0)

    export.hdf.write(MEmodel_.mesh_me, "ME/mesh")
    export.hdf.write(EPmodel_.mesh_ep, "EP/mesh")

    # --- Dump input file ---
    export.dump_input_file()

    EDP = SimDet.get("EDP", 12.0)
    single_solve_timing = bool(SimDet.get("single_solve_timing", False))

    it = 0

    def get_mpi_comm(comm_me):
        """Convert a DOLFIN MpiComm to an mpi4py df.MPI.Comm, or pass through if already one."""
        try:
            return comm_me.tompi4py()
        except AttributeError:
            return comm_me

    # --- Setup MPI communicator ---
    comm = get_mpi_comm(comm_me)

    if not SimDet.get("isrestart") and _no_preload:
        # NO PRELOAD: the mesh is the ED state. Assert the FE cavity volumes against the ED-state
        # manifest (the units/scale consistency check -- a mismatch is a marking or unit defect
        # and is never rescaled away), then leave u = 0 with the cavity loads at zero: the
        # reported pressures carry the reference offset.
        _v_fe = {"lv": MEmodel_.get_lv_volume(), "rv": MEmodel_.get_rv_volume(),
                 "la": MEmodel_.get_la_volume(), "ra": MEmodel_.get_ra_volume()}
        _v_ed = SimDet.get("fch_ed_volumes_ml") or {}
        _rel_tol = float(SimDet.get("fch_ed_volume_rel_tol", 1.0e-6))
        _bad = []
        for ch, v in _v_fe.items():
            if ch in _v_ed and _v_ed[ch] is not None:
                ref = float(_v_ed[ch])
                if ref <= 0.0 or abs(v - ref) > _rel_tol * abs(ref):
                    _bad.append("%s: FE %.6f mL vs manifest %.6f mL" % (ch, v, ref))
        if _bad:
            raise RuntimeError(
                "no-preload ED-volume identity FAILED (mesh units/scale or manifest mismatch; "
                "never rescaled away): " + "; ".join(_bad))
        if not _v_ed:
            printout("WARNING: no-preload run without an ED-state manifest (fch_ed_volumes_ml): the "
                     "FE cavity volumes at u=0 are NOT being checked against a measured record.",
                     comm_me, stage="loading", level="WARN")
        for _c in (MEmodel_.LVCavitypres, MEmodel_.RVCavitypres,
                   MEmodel_.LACavitypres, MEmodel_.RACavitypres):
            _c.assign(0.0)
        export.writePV(MEmodel_, 0)
        printout("no-preload ED state adopted as the loaded configuration", comm_me,
                 stage="loading", level="INFO",
                 **{("v_%s_ml" % ch): round(v, 4) for ch, v in _v_fe.items()},
                 **{("p_%s_mmhg" % ch): round(_ref_p[ch] * 0.0075, 3) for ch in _v_fe})
    elif not SimDet.get("isrestart") and _volctrl:
        # UNLOADED reference under volume control (rung U): the cavity loads are the Lagrange
        # multipliers, so the pressure-inflation loop cannot load the model. March the four
        # prescribed volumes from the solved zero state until every multiplier reaches its
        # per-chamber EDP (fch_coupled: EDP_lv from SimDet["EDP"], the others from P_RV/P_LA/P_RA
        # scaled exactly as the pressure path does), freezing each chamber at its target.
        from .volctrl_ramp import VolumeRampError, ramp_volumes_to_pressures, volctrl_edp_targets_pa
        _base_pa = EDP / 0.0075
        # LV lands on EDP; the others on EDP * P_ch/P_LV -- exactly the state the lockstep pressure
        # loading loop ends in (P_* are loading INCREMENTS, not targets: targeting them directly
        # put the LV at 6.15 instead of 8 mmHg on the anchored set, forward v5 2026-09-10).
        _edp_pa = volctrl_edp_targets_pa(EDP, float(SimDet.get("P_LV", _base_pa)),
                                         float(SimDet.get("P_RV", _base_pa)),
                                         float(SimDet.get("P_LA", _base_pa / 5.0)),
                                         float(SimDet.get("P_RA", _base_pa / 2.0)))
        _getv = {"lv": MEmodel_.get_lv_volume, "rv": MEmodel_.get_rv_volume,
                 "la": MEmodel_.get_la_volume, "ra": MEmodel_.get_ra_volume}
        _v0 = {c: float(f()) for c, f in _getv.items()}

        def _solve_at(vols):
            for c, v in vols.items():
                _swept_vols[c].assign(float(v))
            solver_elas.solvenonlinear()
            lam = {c: float(MEmodel_.get_fch_swept_pressure(chamber=c)) for c in vols}
            for c in vols:
                if abs(float(_getv[c]()) - float(vols[c])) > _vol_tol:
                    raise RuntimeError("volume ramp: constraint violated for %s" % c)
            return lam

        _keep = [None]

        def _snapshot():
            return MEmodel_.w_me.vector().get_local().copy()

        def _restore(vec):
            MEmodel_.w_me.vector().set_local(vec)
            MEmodel_.w_me.vector().apply("insert")

        _ramp_log = get_logger("fe", comm_me)
        with _ramp_log.stage("loading", mode="volctrl4", **{("edp_%s_mmhg" % c): round(v * 0.0075, 3)
                                                            for c, v in _edp_pa.items()}) as _st:
            def _on_step(v, p):
                if _st.due():
                    _st.progress(sum(min(p[c] / _edp_pa[c], 1.0) for c in p), 4.0,
                                 **{("v_%s_ml" % c): round(v[c], 3) for c in v},
                                 **{("p_%s_mmhg" % c): round(p[c] * 0.0075, 3) for c in p})
            try:
                _res = ramp_volumes_to_pressures(
                    v_start=_v0, p_targets=_edp_pa, solve_at=_solve_at, snapshot=_snapshot,
                    restore=_restore, volinc=float(SimDet.get("fch_volctrl_volinc_ml", 1.0)),
                    tol_p=float(SimDet.get("fch_volctrl_edp_tol_mmhg", 0.05)) / 0.0075,
                    volinc_floor=float(SimDet.get("fch_volctrl_volinc_floor_ml", 1.0e-4)),
                    max_solves=int(SimDet.get("fch_volctrl_max_solves", 800)),
                    on_step=_on_step,
                    warn=lambda m: printout("WARNING: " + m, comm_me, stage="loading", level="WARN"))
            except VolumeRampError as _exc:
                raise RuntimeError("four-cavity volume ramp FAILED to reach the per-chamber EDPs: %s"
                                   % _exc) from _exc
        # mirror the multipliers into the Constants (the pressure path's post-load state)
        MEmodel_.LVCavitypres.assign(_res["p"]["lv"]); MEmodel_.RVCavitypres.assign(_res["p"]["rv"])
        MEmodel_.LACavitypres.assign(_res["p"]["la"]); MEmodel_.RACavitypres.assign(_res["p"]["ra"])
        export.writePV(MEmodel_, 0)
        printout("four-cavity volume ramp reached the per-chamber EDPs", comm_me, stage="loading",
                 level="INFO", solves=_res["n_solves"], halvings=_res["n_halvings"],
                 **{("v_%s_ml" % c): round(v, 3) for c, v in _res["v"].items()},
                 **{("p_%s_mmhg" % c): round(v * 0.0075, 3) for c, v in _res["p"].items()})
    elif not SimDet.get("isrestart"):
        while 1:
            base_mmPa = EDP / 0.0075
            MEmodel_.LVCavitypres.assign(
                float(MEmodel_.LVCavitypres) + SimDet.get("P_LV", base_mmPa) / nloadstep
            )

            defaults = {
                "P_RV": base_mmPa,
                "P_LA": base_mmPa / 5.0,
                "P_RA": base_mmPa / 2.0,
            }
            MEmodel_.RVCavitypres.assign(
                float(MEmodel_.RVCavitypres)
                + SimDet.get("P_RV", defaults["P_RV"]) / nloadstep
            )
            MEmodel_.LACavitypres.assign(
                float(MEmodel_.LACavitypres)
                + SimDet.get("P_LA", defaults["P_LA"]) / nloadstep
            )
            MEmodel_.RACavitypres.assign(
                float(MEmodel_.RACavitypres)
                + SimDet.get("P_RA", defaults["P_RA"]) / nloadstep
            )

            if not SimDet.get("isrestart"):
                solver_elas.solvenonlinear()

            export.writePV(MEmodel_, 0)
            export.hdf.write(MEmodel_.get_displacement(), "ME/u_loading", it)
            it += 1

            printout(
                "cavity state after load step", comm_me, stage="loading",
                step=it, of=nloadstep,
                p_lv_mmhg=MEmodel_.get_lv_pressure() * 0.0075,
                v_lv_ml=MEmodel_.get_lv_volume(),
                p_rv_mmhg=MEmodel_.get_rv_pressure() * 0.0075,
                v_rv_ml=MEmodel_.get_rv_volume(),
                edp_target_mmhg=EDP,
            )

            if float(MEmodel_.LVCavitypres) * 0.0075 >= EDP:
                break

    # FCH-contained restart checkpoint. The shared exportfiles Data.h5 is the VIZ stream --
    # it is opened "w" (truncated) on every construction, including restart, BEFORE this
    # point, so it cannot hold a survivable checkpoint. The dedicated restart_state.h5 +
    # restart_state.csv (src/sim_protocols/fch_restart.py) carry the full mixed FE state and
    # the CONSISTENT CL/time state (same tstep) instead. Created for every run (so any run is
    # resumable); restored only on isrestart. The restart-defining manifest is validated by
    # the entry point (demo/fch_baseline.py) BEFORE the run, refusing incompatible resumes.
    _ckpt = FCHCheckpoint(
        os.path.join(outputfolder, folderName), comm_me, bool(SimDet.get("isrestart"))
    )
    _cl_restored = None
    if SimDet.get("isrestart"):
        _cl_restored = _ckpt.restore(MEmodel_.w_me)
        printout(
            "[restart] restored mixed FE state (w_me) + CL/time state at "
            f"tstep={_cl_restored.get('tstep')} (cycle {_cl_restored.get('cycle')})",
            comm_me,
        )

    MEmodel_.u_me_ED.assign(MEmodel_.get_displacement())
    printout("volume = " + str(MEmodel_.get_lv_volume()), comm_me)

    # Four-chamber strain emitter (output_strain.csv: per-chamber Ell/Ecc/Err in BOTH the
    # mesh-referenced and the ED-referenced frame + MAPSE/TAPSE/twist/atrial excursion) at the
    # write-step cadence; the ED reference is captured HERE, the instant the run defines ED.
    _strain = None
    if SimDet.get("fch_strain_emit", True) and SimDet.get("fch_fe", True):
        try:
            _strain = FchStrainEmitter(
                MEmodel_, output_dir,
                length_factor_mm=float(SimDet.get("fch_length_factor_mm", 10.0)),
                log=get_logger("fe", comm_me))
            _strain.capture_ed()
        except Exception as _exc:  # noqa: BLE001 -- never let a readout kill the solve
            printout("WARNING: four-chamber strain emitter DISABLED (construction failed: %r); "
                     "output_strain.csv will NOT be written" % (_exc,), comm_me,
                     stage="init", level="WARN")
            _strain = None
    _strain_step = int(SimDet.get("fch_strain_write_step", SimDet.get("writeStep", 40)))

    fStrain_uL = MEmodel_.get_fiber_strain_unloaded()
    # Cache UFL expression for repeated projection without reconstructing forms.

    # Closed-loop phase
    stop_iter = SimDet["closedloopparam"]["stop_iter"]

    isrestart = 0
    cnt = 0

    # Precompute common function spaces and reusable functions
    V_CG1 = df.FunctionSpace(MEmodel_.mesh_me, "CG", 1)
    V_DG0 = df.FunctionSpace(MEmodel_.mesh_me, "DG", 0)
    V_DG1 = df.FunctionSpace(MEmodel_.mesh_me, "DG", 1)

    potential_me = df.Function(V_CG1)
    # Preallocate reusable output Functions to avoid per-step allocations
    fstress_DG_fun = df.Function(V_DG0)
    fstress_DG_fun.rename("fstress", "fstress")
    eff_fun = df.Function(V_DG0)
    eff_fun.rename("Eff", "Eff")
    imp_fun = df.Function(V_DG1)
    imp_fun.rename("imp", "imp")
    imp2_fun = df.Function(V_DG1)
    imp2_fun.rename("imp2", "imp2")

    # Pre-create Probes once if probe points are provided
    probesfstress = probesEul_fiber = probesIMP = probesIMP2 = probesIMP3 = None
    probesE_circ_BiV = probesE_long_BiV = probesE_radi_BiV = None
    has_probes = bool(SimDet.get("probepts"))
    if has_probes:
        # Probes sample projected fields at fixed spatial points (useful for regressions/profiling).
        x_probe = np.array(SimDet["probepts"]).flatten()
        probesfstress = Probes(x_probe, V_DG1)
        probesEul_fiber = Probes(x_probe, V_DG1)
        probesIMP = Probes(x_probe, V_DG1)
        probesIMP2 = Probes(x_probe, V_DG1)
        probesIMP3 = Probes(x_probe, V_CG1)
        # Compatibility placeholders
        probesE_circ_BiV = Probes(x_probe, V_DG1)
        probesE_long_BiV = Probes(x_probe, V_DG1)
        probesE_radi_BiV = Probes(x_probe, V_DG1)

    cycle_signal_tags = ("LV", "RV", "LA", "RA")

    def _new_cycle_buf():
        buf = {"t": []}
        for tag in cycle_signal_tags:
            buf[f"P_{tag}"] = []
            buf[f"V_{tag}"] = []
        return buf

    def _append_cycle_sample(buf, sample):
        if buf is None or sample is None:
            return
        # Buffer values per-cycle; cycle boundaries are detected later via the time reset in `t`.
        buf["t"].append(float(sample.get("t", 0.0)))
        for tag in cycle_signal_tags:
            pkey = f"P_{tag}"
            vkey = f"V_{tag}"
            if pkey in buf and pkey in sample:
                buf[pkey].append(float(sample[pkey]))
            if vkey in buf and vkey in sample:
                buf[vkey].append(float(sample[vkey]))

    writecnt = 0

    P_LV = _report_p("lv", MEmodel_.get_lv_pressure())  # LVCavitypres (+ reference offset)
    V_LV = MEmodel_.get_lv_volume()  # GetVolumeComputation()

    if SimDet.get("isrestart"):
        # Restore the CL/time state from the CONSISTENT restart checkpoint (the row written at
        # the SAME tstep as the restored mixed FE state), NOT the last output_PV.csv row -- the
        # latter is up to writeStep steps ahead of the FE checkpoint, which would reintroduce a
        # discontinuity. (This is the fidelity gain over the legacy LV semantics.)
        r = _cl_restored or {}
        clp0 = SimDet["closedloopparam"]
        V_sa = r.get("V_sa", clp0["V_sa"])
        V_ad = r.get("V_ad", clp0["V_ad"])
        V_sv = r.get("V_sv", clp0["V_sv"])
        V_pa = r.get("V_pa", clp0["V_pa"])
        V_pv = r.get("V_pv", clp0["V_pv"])
        # Chamber P/V from the checkpoint; fall back to the (always-defined) FE getters --
        # the bare locals V_RV/V_LA/... are not yet assigned on the restart path.
        V_LV = r.get("V_LV", MEmodel_.get_lv_volume())
        V_RV = r.get("V_RV", MEmodel_.get_rv_volume())
        V_LA = r.get("V_LA", MEmodel_.get_la_volume())
        V_RA = r.get("V_RA", MEmodel_.get_ra_volume())
        P_LV = r.get("P_LV", MEmodel_.get_lv_pressure())
        P_RV = r.get("P_RV", MEmodel_.get_rv_pressure())
        P_LA = r.get("P_LA", MEmodel_.get_la_pressure())
        P_RA = r.get("P_RA", MEmodel_.get_ra_pressure())
        if "tstep" in r:
            state_obj.tstep = float(r["tstep"])
            state_obj.cycle = math.floor(state_obj.tstep / state_obj.BCL)
            state_obj.t = float(r.get("t", state_obj.tstep - state_obj.cycle * state_obj.BCL))
        # Trim the PV trace to the restart tstep so the appended continuation is one continuous
        # trace (rank 0 only; the checkpoint may lag the last-written PV row by < writeStep).
        if df.MPI.rank(comm_me) == 0:
            truncate_pv_to_tstep(
                os.path.join(outputfolder, folderName, "output_PV.csv"), state_obj.tstep
            )
        # Re-assert the restored cavity pressures onto the FE Constants (they live outside
        # w_me) so the warm-started coupling + any pressure read are consistent. The restored
        # P_* are REPORTED pressures; the Constants carry the FE load (offset removed).
        MEmodel_.LVCavitypres.assign(_fe_p("lv", P_LV))
        MEmodel_.RVCavitypres.assign(_fe_p("rv", P_RV))
        MEmodel_.LACavitypres.assign(_fe_p("la", P_LA))
        MEmodel_.RACavitypres.assign(_fe_p("ra", P_RA))
        if _volctrl:
            # The swept-volume Constants live outside w_me too: re-prescribe the checkpoint's
            # chamber volumes so the first resumed solve targets the state it was saved at.
            for _ch, _v in (("lv", V_LV), ("rv", V_RV), ("la", V_LA), ("ra", V_RA)):
                _swept_vols[_ch].assign(float(_v))
    else:
        V_sa = SimDet["closedloopparam"]["V_sa"]
        V_ad = SimDet["closedloopparam"]["V_ad"]
        V_sv = SimDet["closedloopparam"]["V_sv"]
        V_pa = SimDet["closedloopparam"]["V_pa"]
        V_pv = SimDet["closedloopparam"]["V_pv"]

    if not SimDet.get("isrestart"):
        # Fresh-run init of the remaining chamber P/V from the FE state. SKIPPED on restart:
        # those are already restored from the checkpoint above; re-reading the cavity-pressure
        # getters here would clobber the restored P_LA/P_RA/P_RV with stale Constant values.
        P_LA = _report_p("la", MEmodel_.get_la_pressure())  # initial P_LA/P_RA (+ offset)
        V_LA = MEmodel_.get_la_volume()
        P_RA = _report_p("ra", MEmodel_.get_ra_pressure())
        V_RA = MEmodel_.get_ra_volume()

        P_RV = _report_p("rv", MEmodel_.get_rv_pressure())
        V_RV = MEmodel_.get_rv_volume()

    CLmodel_ = CLmodel_biv(SimDet, V_LV, V_RV, V_LA, V_RA)

    def _advance_targets(params):
        """One circulation step (closed loop) or one trace lookup (open loop).

        Returns ``(V_LV, V_RV, extra)`` with ``extra`` in circBiV's fch_fe order
        ``[V_LA, V_RA, V_sa, V_ad, V_sv, V_pa, V_pv, Qmv, Qav, Qpvv, Qtv]`` so the caller is
        identical for both schemes. The open loop replays the trace at t + dt (the time the
        forward-Euler circulation step would have returned volumes for).
        """
        if not _openloop:
            v_lv_, v_rv_, *extra_ = CLmodel_.UpdateLVV(params)
            return v_lv_, v_rv_, extra_
        row = _trace.at(float(params["t"]) + float(params["delTat"]))
        extra_ = [row["V_LA"], row["V_RA"],
                  row.get("V_sa", params["V_sa"]), row.get("V_ad", params["V_ad"]),
                  row.get("V_sv", params["V_sv"]), row.get("V_pa", params["V_pa"]),
                  row.get("V_pv", params["V_pv"]),
                  row.get("Qmv", 0.0), row.get("Qav", 0.0), row.get("Qpvv", 0.0),
                  row.get("Qtv", 0.0)]
        return row["V_LV"], row["V_RV"], extra_

    # Event-time displacement snapshots (Figure 6 frames) -- opt-in, XDMF + HDF5 per event
    # per cycle from `fch_event_snapshot_first_cycle` on (see fch_snapshots.py).
    _snap = None
    if SimDet.get("fch_event_snapshots"):
        _snap = EventSnapshotWriter(
            os.path.join(output_dir, "snapshots"),
            load_event_times(SimDet["fch_event_snapshots"]), comm_me,
            bcl_ms=float(state_obj.BCL),
            first_cycle=int(SimDet.get("fch_event_snapshot_first_cycle", 1)),
            log=get_logger("fe", comm_me))
        printout("event snapshots ENABLED", comm_me, stage="init", level="INFO",
                 events=",".join("%s@%.1f" % (k, v) for k, v in _snap.events.items()),
                 first_cycle=_snap.first_cycle)

    write_active = bool(SimDet.get("write_active_csv"))
    enable_periodicity_checks = bool(SimDet.get("enable_periodicity_checks"))
    write_periodicity_log = enable_periodicity_checks and bool(
        SimDet.get("write_periodicity_log", False)
    )
    write_periodicity_metrics = enable_periodicity_checks and bool(
        SimDet.get("write_periodicity_metrics", True)
    )
    periodicity_log_path = (
        os.path.join(outputfolder, folderName, "periodicity_log.csv")
        if write_periodicity_log
        else None
    )
    periodicity_metrics_path = (
        os.path.join(outputfolder, folderName, "periodicity_metrics.json")
        if write_periodicity_metrics
        else None
    )

    def _log_periodicity_rows(rows):
        if not write_periodicity_log or df.MPI.rank(comm_me) != 0:
            return
        append_periodicity_rows(periodicity_log_path, rows)

    def _log_periodicity_metrics(scalar_metrics):
        if not write_periodicity_metrics or df.MPI.rank(comm_me) != 0:
            return
        write_periodicity_metrics_json(
            periodicity_metrics_path, case_id, scalar_metrics
        )

    csv_path = os.path.join(outputfolder, folderName, "output_PV.csv")
    csv_fh = None
    csv_writer = None
    if df.MPI.rank(comm_me) == 0:
        os.makedirs(os.path.dirname(csv_path), exist_ok=True)
        # On restart, append to maintain a single continuous PV trace.
        append_mode = SimDet.get("isrestart") and os.path.exists(csv_path)
        csv_fh = open(csv_path, "a" if append_mode else "w", newline="")
        csv_writer = csv.writer(csv_fh)
        if not append_mode:
            csv_writer.writerow(
                [
                    "tstep",
                    "t",
                    "V_LV",
                    "P_LV",
                    "V_RV",
                    "P_RV",
                    "V_LA",
                    "P_LA",
                    "V_RA",
                    "P_RA",
                    "V_sv",
                    "V_sa",
                    "V_ad",
                    "V_pv",
                    "V_pa",
                    "Qmv",
                    "Qav",
                    "Qpvv",
                    "Qtv",
                ]
            )

    # Initialize output_active.csv with header similar to output_PV.csv
    active_csv_path = None
    active_fh = None
    active_writer = None
    if write_active and df.MPI.rank(comm_me) == 0:
        # Separate CSV for chamber-wise active stress proxies (useful for calibration/debugging).
        active_csv_path = os.path.join(outputfolder, folderName, "output_active.csv")
        os.makedirs(os.path.dirname(active_csv_path), exist_ok=True)
        active_fh = open(active_csv_path, "w", newline="")
        active_writer = csv.writer(active_fh)
        active_writer.writerow(["tstep", "act_lv", "act_la", "act_rv", "act_ra"])

    probe_cfg = SimDet.get("fsolve_probe", {})
    probe_enabled = bool(probe_cfg.get("enabled", False))
    probe_label = probe_cfg.get("label", case_id)
    probe_output = probe_cfg.get("output")
    if not probe_output:
        probe_output = os.path.join(outputfolder, folderName, "fsolve_probe.csv")
    probe_output = os.path.normpath(probe_output)

    def _record_fsolve_probe(
        total_duration,
        fsolve_duration,
        root_duration,
        info_map,
        message,
        success_flag,
    ):
        if not probe_enabled:
            return
        total_duration = float(total_duration)
        max_total = comm_me.allreduce(total_duration, op=pyMPI.MAX)
        min_total = comm_me.allreduce(total_duration, op=pyMPI.MIN)
        if df.MPI.rank(comm_me) != 0:
            return
        os.makedirs(os.path.dirname(probe_output), exist_ok=True)
        header = [
            "timestamp",
            "label",
            "case_id",
            "mpi_size",
            "tstep",
            "t",
            "dt",
            "fsolve_time",
            "root_time",
            "total_time",
            "max_total_time",
            "min_total_time",
            "nfev",
            "message",
            "success",
        ]
        needs_header = not os.path.exists(probe_output)
        timestamp = time.strftime("%Y-%m-%dT%H:%M:%S")
        nfev = info_map.get("nfev", "") if isinstance(info_map, dict) else ""
        row = [
            timestamp,
            probe_label,
            case_id,
            pyMPI.COMM_WORLD.Get_size(),
            state_obj.tstep,
            state_obj.t,
            state_obj.dt.dt,
            fsolve_duration,
            root_duration,
            total_duration,
            max_total,
            min_total,
            nfev,
            str(message),
            bool(success_flag),
        ]
        with open(probe_output, "a", newline="") as probe_fh:
            writer = csv.writer(probe_fh)
            if needs_header:
                writer.writerow(header)
            writer.writerow(row)
        printout(
            f"Fsolve timing probe recorded ({total_duration:.6f}s) -> {probe_output}",
            comm_me,
        )

    cycle_buf_prev = None
    cycle_buf_curr = _new_cycle_buf()
    end_reason = None

    # --- Phase-based dt policy (ported from run_waorta; OFF by default) ----------
    # Set dt by cardiac phase, keyed on the (ventricular) active-twitch timing
    # (t0/t_trans/tau + activation onset) so the boundaries adapt per case:
    #   active-free (pre-activation t_rel<0, or post-decay filling): LARGE dt
    #   contraction+ejection (0 <= t_rel < t0): FINE dt -- the peak-systolic limit-
    #       point zone; a small dt here also BOOSTS the viscous tangent 2*eta/dt
    #       (stronger regularization) at ~no extra viscous stress.
    #   relaxation (t0 <= t_rel < t_trans + n_tau*tau): MODERATE dt
    # Falls back to the legacy single-switch (dt_diastasis) when no phase keys are
    # set, and to a uniform dt when neither is set -> byte-identical default.
    dt_base_cfg = float(SimDet["dt"])
    dt_diastasis = SimDet.get("dt_diastasis")
    dt_switch_ms = float(SimDet.get("dt_switch_ms", 0.55 * state_obj.BCL))
    _ap = SimDet.get("GiccioneParams", {}).get("Active params", {})
    _t0 = float(_ap.get("t0", 260.0))
    _t_trans = float(_ap.get("t_trans", 1.5 * _t0))
    _tau = float(_ap.get("tau", 35.0))
    _t_act = float(SimDet.get("homogeneous_activation_time",
                              _ap.get("homogeneous_activation_time", 0.0)))
    _n_tau = float(SimDet.get("dt_relax_n_tau", 4.0))
    _dt_systole = float(SimDet.get("dt_systole", dt_base_cfg))
    _dt_relax = float(SimDet.get("dt_relax", dt_base_cfg))
    _dt_filling = (float(SimDet["dt_filling"]) if SimDet.get("dt_filling")
                   else (float(dt_diastasis) if dt_diastasis else dt_base_cfg))
    _t_relax_end = _t_trans + _n_tau * _tau
    _phase_based = bool(SimDet.get("dt_phase_based", False)) or any(
        k in SimDet for k in ("dt_systole", "dt_relax", "dt_filling"))

    # Viscous (Kelvin-Voigt) regularization: when enabled, keep MEmodel_.dt_const in
    # sync with the dt actually used this (sub)step so the viscous tangent 2*eta/dt
    # tracks a fine systolic/backoff dt (the lever that traverses the ejection limit
    # point). When disabled the viscous term is zero, so dt_const is left untouched
    # -> byte-identical default.
    _viscous_on = bool(SimDet.get("_passive_viscous_")) or bool(
        SimDet.get("GiccioneParams", {}).get("Passive params", {}).get("eta", 0.0))
    # Strict coupling: by default a coupling solve that fails after backoff RAISES
    # (matches run_waorta and the repo's fail-loud principle). Set
    # coupling_allow_nonconverged=True to keep the legacy lenient "proceed with
    # previous pressures" path -- which now emits a loud WARNING when taken.
    _allow_nonconverged = bool(SimDet.get("coupling_allow_nonconverged", False))

    # Coupled-loop progress. The percent basis is elapsed simulated time over the whole run
    # (stop_iter is the LAST cycle index, so the run spans stop_iter+1 cycles), which advances
    # monotonically even when a backoff shortens dt.
    _fe_log = get_logger("fe", comm_me)
    _loop_total_ms = float(state_obj.BCL) * float(stop_iter + 1)
    _loop_t0 = time.time()
    _loop_progress = _fe_log.tracker("coupling")
    _fe_log.info("coupling", "BEGIN", cycles=stop_iter + 1, bcl_ms=state_obj.BCL,
                 dt_ms=state_obj.dt.dt, write_step=SimDet.get("writeStep"))

    while 1:
        if state_obj.cycle > stop_iter:
            end_reason = "stop_iter"
            break

        # Phase-adaptive dt: set the step dt by cardiac phase before the CL advance.
        if _phase_based:
            _trel = state_obj.t - _t_act
            if _trel < 0.0 or _trel >= _t_relax_end:
                state_obj.dt.dt = _dt_filling      # active-free filling/diastasis: large
            elif _trel < _t0:
                state_obj.dt.dt = _dt_systole      # contraction + ejection (limit point): fine
            else:
                state_obj.dt.dt = _dt_relax        # relaxation: moderate
        elif dt_diastasis:
            state_obj.dt.dt = (
                float(dt_diastasis) if state_obj.t >= dt_switch_ms else dt_base_cfg
            )

        # Pre-step CL state for the dt-backoff rollback. circBiV.UpdateLVV reads all
        # compartment volumes from `params` (it does not persist them) but MUTATES
        # the hysteretic valve flags in place, so a faithful rollback restores BOTH
        # the pre-step volumes (snapshotted here as locals) and the valve flags. This
        # fixes a latent double-advance: the legacy backoff re-ran UpdateLVV from the
        # already-advanced volumes with corrupted valve state.
        cl_pre = {
            "P_LV": P_LV, "V_LV": V_LV, "P_RV": P_RV, "V_RV": V_RV,
            "P_LA": P_LA, "V_LA": V_LA, "P_RA": P_RA, "V_RA": V_RA,
            "V_sa": V_sa, "V_ad": V_ad, "V_sv": V_sv, "V_pa": V_pa, "V_pv": V_pv,
        }
        valve_pre = CLmodel_.snapshot_valves()

        # Lumped-parameter circulation advances vascular state and outputs target chamber volumes/flows.
        params = {
            "P_LV": P_LV,
            "V_LV": V_LV,
            "t": state_obj.t,
            "delTat": state_obj.dt.dt,
        }

        params.update(
            {
                "P_RV": P_RV,
                "V_RV": V_RV,
                "P_LA": P_LA,
                "V_LA": V_LA,
                "P_RA": P_RA,
                "V_RA": V_RA,
                "V_sa": V_sa,
                "V_ad": V_ad,
                "V_sv": V_sv,
                "V_pa": V_pa,
                "V_pv": V_pv,
            }
        )

        V_LV, V_RV, extra = _advance_targets(params)
        V_LA, V_RA, V_sa, V_ad, V_sv, V_pa, V_pv, Qmv, Qav, Qpvv, Qtv = extra

        # Only rank 0 writes
        if df.MPI.rank(comm_me) == 0 and csv_writer is not None:
            row = [state_obj.tstep]
            row += [
                state_obj.t,
                V_LV,
                P_LV,
                V_RV,
                P_RV,
                V_LA,
                P_LA,
                V_RA,
                P_RA,
                V_sv,
                V_sa,
                V_ad,
                V_pv,
                V_pa,
                Qmv,
                Qav,
                Qpvv,
                Qtv,
            ]
            csv_writer.writerow(row)
            # Flush periodically to avoid data loss on long runs
            if (cnt % 10) == 0:
                try:
                    csv_fh.flush()
                except Exception:
                    pass

        cycle_sample = {
            "t": state_obj.t,
            "P_LV": P_LV,
            "V_LV": V_LV,
            "P_RV": P_RV,
            "V_RV": V_RV,
            "P_LA": P_LA,
            "V_LA": V_LA,
            "P_RA": P_RA,
            "V_RA": V_RA,
        }
        _append_cycle_sample(cycle_buf_curr, cycle_sample)

        # Solve for cavity pressures so the FE model matches the CL target volumes at this time step.

        def Rp_fch(plv, prv, pla, pra, lvvc, rvvc, lavc, ravc):
            # The root-find works in REPORTED pressures; the FE load is the offset-free part.
            MEmodel_.LVCavitypres.assign(_fe_p("lv", plv))
            MEmodel_.RVCavitypres.assign(_fe_p("rv", prv))
            MEmodel_.LACavitypres.assign(_fe_p("la", pla))
            MEmodel_.RACavitypres.assign(_fe_p("ra", pra))
            solver_elas.solvenonlinear()
            vlv = MEmodel_.get_lv_volume()
            vrv = MEmodel_.get_rv_volume()
            vla = MEmodel_.get_la_volume()
            vra = MEmodel_.get_ra_volume()
            return np.array([vlv - lvvc, vrv - rvvc, vla - lavc, vra - ravc])

        # Nonlinear solve uses SciPy root-finding around the FE volume residual.

        from scipy.optimize import (
            fsolve,
            root,
        )

        def Rp_fch_call(plrva):
            plv, prv, pla, pra = plrva
            return Rp_fch(plv, prv, pla, pra, V_LV, V_RV, V_LA, V_RA)

        x0 = [P_LV, P_RV, P_LA, P_RA]

        # Prepare backoff state and guardrails
        state_prev = df.Function(MEmodel_.w_me.function_space())
        state_prev.assign(MEmodel_.w_me)
        prev_delTat = state_obj.dt.dt
        orig_delTat = prev_delTat
        min_backoff_factor = float(SimDet.get("backoff_min_factor", 0.25))
        max_backoff_iters = 1 if single_solve_timing else int(SimDet.get("backoff_max_iters", 10))

        fsolve_duration = 0.0
        root_duration = 0.0
        msg = ""

        # Backoff strategy: rollback FE state and halve the CL dt used in UpdateLVV if root-finding fails.
        success_global = False
        solve_wall_start = time.perf_counter() if single_solve_timing else None
        solver_used = None
        for num_iter in range(max_backoff_iters):
            local_fail = False
            if _viscous_on:
                # Keep the viscous tangent 2*eta/dt consistent with the dt that
                # produced this step's target volumes. prev_delTat == state_obj.dt.dt
                # on the first attempt and halves on each backoff, so a fine systolic
                # (phase policy) or backoff dt boosts the regularization at the
                # ejection limit point while the viscous stress stays correct.
                try:
                    MEmodel_.dt_const.assign(float(prev_delTat))
                except Exception:
                    pass
            if _volctrl:
                # VOLUME-prescribed step: the four 0D targets go to the swept-cavity constraints,
                # ONE nonlinear solve, cavity pressures = the multipliers. No outer root-find.
                try:
                    for _ch, _vt in (("lv", V_LV), ("rv", V_RV), ("la", V_LA), ("ra", V_RA)):
                        _swept_vols[_ch].assign(float(_vt))
                    fsolve_start = time.perf_counter()
                    solver_elas.solvenonlinear()
                    fsolve_duration = time.perf_counter() - fsolve_start
                    _lam = {_ch: float(MEmodel_.get_fch_swept_pressure(chamber=_ch))
                            for _ch in ("lv", "rv", "la", "ra")}
                    _vfe = {"lv": MEmodel_.get_lv_volume(), "rv": MEmodel_.get_rv_volume(),
                            "la": MEmodel_.get_la_volume(), "ra": MEmodel_.get_ra_volume()}
                    _res = [_vfe["lv"] - V_LV, _vfe["rv"] - V_RV,
                            _vfe["la"] - V_LA, _vfe["ra"] - V_RA]
                    if not all(math.isfinite(v) for v in _lam.values()):
                        raise RuntimeError("volctrl4: non-finite multiplier pressure %r" % _lam)
                    _worst = max(abs(r) for r in _res)
                    if _worst > _vol_tol:
                        # Message text is a parsing contract (signature volctrl_constraint_violation).
                        raise RuntimeError(
                            "volctrl4: converged solve violates the volume constraint "
                            "(max |V_fe - V_target| = %.3e mL > %.1e mL; residuals lv/rv/la/ra = %s)"
                            % (_worst, _vol_tol, ["%.3e" % r for r in _res]))
                    # Mirror the multipliers into the (otherwise inert) pressure Constants so
                    # every getter, exportfiles.writePV and the restart re-assert report the
                    # true cavity pressure (the LVW volume-control convention).
                    MEmodel_.LVCavitypres.assign(_lam["lv"])
                    MEmodel_.RVCavitypres.assign(_lam["rv"])
                    MEmodel_.LACavitypres.assign(_lam["la"])
                    MEmodel_.RACavitypres.assign(_lam["ra"])
                    root1 = [_report_p("lv", _lam["lv"]), _report_p("rv", _lam["rv"]),
                             _report_p("la", _lam["la"]), _report_p("ra", _lam["ra"])]
                    info = {"nfev": 1, "fvec": _res}
                    msg = ("volctrl4: prescribed V=(%.4f, %.4f, %.4f, %.4f) mL, multipliers "
                           "P=(%.3f, %.3f, %.3f, %.3f) Pa"
                           % (V_LV, V_RV, V_LA, V_RA, _lam["lv"], _lam["rv"], _lam["la"], _lam["ra"]))
                    solver_used = "volctrl4"
                except Exception as exc:
                    local_fail = True
                    msg = repr(exc)
                    solver_used = "volctrl4"
            else:
                try:
                    # Adaptive tolerance: relax early cycles
                    base_xtol = float(SimDet.get("solver_xtol_base", 1e-4))
                    relax_cycles = int(SimDet.get("xtol_relax_cycles", 1))
                    relax_factor = float(SimDet.get("solver_xtol_factor_early", 10.0))
                    xtol_use = base_xtol * (
                        relax_factor if state_obj.cycle < relax_cycles else 1.0
                    )

                    # First attempt: fsolve (hybr)
                    fsolve_start = time.perf_counter() if (probe_enabled or single_solve_timing) else None
                    root1, info, ier, msg = fsolve(
                        Rp_fch_call,
                        x0,
                        xtol=xtol_use,
                        factor=0.01,
                        maxfev=int(SimDet.get("solver_maxfev", 400)),
                        full_output=True,
                    )
                    if (probe_enabled or single_solve_timing) and fsolve_start is not None:
                        fsolve_duration = time.perf_counter() - fsolve_start
                    solver_used = "fsolve"
                    if ier != 1:
                        # Second attempt: secant-type (Broyden) without changing dt
                        root_start = time.perf_counter() if (probe_enabled or single_solve_timing) else None
                        sol = root(
                            Rp_fch_call,
                            x0,
                            method="broyden1",
                            tol=xtol_use,
                            options={"maxiter": int(SimDet.get("solver_maxiter", 100))},
                        )
                        if not sol.success:
                            raise RuntimeError(
                                f"broyden1 did not converge: {sol.message}"
                            )
                        root1 = sol.x
                        info = {"nfev": sol.nfev, "fvec": sol.fun}
                        ier = 1
                        msg = sol.message
                        if (probe_enabled or single_solve_timing) and root_start is not None:
                            root_duration = time.perf_counter() - root_start
                        solver_used = "root"
                except Exception:
                    local_fail = True

            global_fail = comm.allreduce(bool(local_fail), op=pyMPI.LOR)
            if not global_fail:
                success_global = True
                break

            # Backoff: roll back BOTH the FE state (w_me) and the CL state (pre-step
            # volumes + valve hysteresis flags), halve the CL dt, and re-advance the
            # circulation FROM THE PRE-STEP STATE. The legacy path rolled back only
            # w_me and re-ran UpdateLVV from the already-advanced volumes with the
            # post-step valve flags -- a double-advance that drifted the circulation.
            MEmodel_.w_me.assign(state_prev)
            CLmodel_.restore_valves(valve_pre)
            prev_delTat = prev_delTat / 2.0
            if prev_delTat < orig_delTat * min_backoff_factor:
                # Message text kept verbatim: the campaign/autocal failure signature
                # `coupling_backoff_exhausted` matches "Backoff dt reached minimum factor".
                printout(
                    f"Backoff dt reached minimum factor ({min_backoff_factor}); coupling unconverged.",
                    comm_me, stage="coupling", level="ERROR",
                    cycle=state_obj.cycle, t_ms=state_obj.t, attempts=num_iter + 1,
                )
                break
            # Narrate each retry. This loop used to back off silently in the FCH driver (unlike
            # the LVW one), so a run that spent most of its wall clock retrying looked identical
            # to a healthy one until the floor was hit.
            _fe_log.warn(
                "coupling", "backoff: retrying the coupling solve with a halved dt",
                attempt="%d/%d" % (num_iter + 1, max_backoff_iters),
                cycle=state_obj.cycle, t_ms=state_obj.t,
                dt_ms=prev_delTat, solver=solver_used or "unknown",
                reason=(msg or "unknown failure"),
            )
            # Re-advance the CL from the pre-step volumes with the reduced delTat.
            params = {
                "P_LV": cl_pre["P_LV"],
                "V_LV": cl_pre["V_LV"],
                "P_RV": cl_pre["P_RV"],
                "V_RV": cl_pre["V_RV"],
                "P_LA": cl_pre["P_LA"],
                "V_LA": cl_pre["V_LA"],
                "P_RA": cl_pre["P_RA"],
                "V_RA": cl_pre["V_RA"],
                "V_sa": cl_pre["V_sa"],
                "V_ad": cl_pre["V_ad"],
                "V_sv": cl_pre["V_sv"],
                "V_pa": cl_pre["V_pa"],
                "V_pv": cl_pre["V_pv"],
                "t": state_obj.t,
                "delTat": prev_delTat,
            }
            V_LV, V_RV, extra = _advance_targets(params)
            V_LA, V_RA, V_sa, V_ad, V_sv, V_pa, V_pv, Qmv, Qav, Qpvv, Qtv = extra
            # Warm-start the next attempt from the pre-step pressures.
            x0 = [cl_pre["P_LV"], cl_pre["P_RV"], cl_pre["P_LA"], cl_pre["P_RA"]]

        if not success_global:
            # Coupling failed after exhausting backoff. DEFAULT: fail loudly -- roll
            # back the FE + CL state and RAISE (matches run_waorta and the repo's
            # non-negotiable error-handling principle; no silent degradation). The
            # legacy lenient "proceed with previous pressures" path is opt-in via
            # coupling_allow_nonconverged=True and emits a loud WARNING when taken.
            MEmodel_.w_me.assign(state_prev)
            CLmodel_.restore_valves(valve_pre)
            fail_msg = (
                f"FCH mechanics-circulation coupling failed to converge after "
                f"{max_backoff_iters} backoff attempts at t={state_obj.t:.4f} ms "
                f"(cycle {state_obj.cycle}); last solver message: {msg}"
            )
            if not _allow_nonconverged:
                raise RuntimeError(fail_msg)
            printout(
                "WARNING: " + fail_msg + " -- coupling_allow_nonconverged=True: "
                "proceeding with the PRE-STEP pressures (DEGRADED path).",
                comm_me,
            )
            # Degraded continuation: hold the pre-step pressures (state rolled back).
            root1 = [cl_pre["P_LV"], cl_pre["P_RV"], cl_pre["P_LA"], cl_pre["P_RA"]]

        if success_global and probe_enabled:
            total_duration = fsolve_duration + root_duration
            _record_fsolve_probe(
                total_duration,
                fsolve_duration,
                root_duration,
                info,
                msg,
                True,
            )
            end_reason = "fsolve_probe"
            break

        if single_solve_timing:
            if solver_used == "fsolve":
                solve_wall = fsolve_duration
            elif solver_used == "root":
                solve_wall = root_duration
            else:
                solve_wall = time.perf_counter() - solve_wall_start if solve_wall_start else 0.0
            printout(
                f"[single_solve_timing] {solver_used or 'solve'} wall time: {solve_wall:.6f} s",
                comm_me, stage="coupling", solver=solver_used or "solve", wall_s=solve_wall,
            )
            df.MPI.barrier(comm_me)
            import sys

            sys.exit(0)

        with open(os.path.join(output_dir, "output_nfev.txt"), "a") as nfev:
            if df.MPI.rank(comm_me) == 0:
                nfev.write(
                    f"t = {state_obj.t}, iter = {info['nfev']}, fun = {info['fvec']}, message = {msg} \n"
                )

        P_LV = root1[0]

        if success_global:
            P_RV = root1[1]
            P_LA = root1[2]
            P_RA = root1[3]

        if write_active:
            # Region-integrated active stress proxies (scaled by user coefficients) for monitoring.
            Fmat = MEmodel_.get_deformation_gradient()  # ensure activeforms uses current state
            Sactive_ventricular = MEmodel_.activeforms.PK2StressTensor()
            Sactive_atrial = MEmodel_.activeforms.PK2StressTensor_atr()

            lv_rid = SimDet.get("lv_rid", 10)
            rv_rid = SimDet.get("rv_rid", 9)
            la_rid = SimDet.get("la_rid", 11)
            ra_rid = SimDet.get("ra_rid", 8)

            # Region-integrated active-stress monitor, weighted by the SAME single
            # per-chamber factor the mechanics uses (Tmax_{ch}/Tmax_fallback) via the
            # shared resolver -- no lv_ac/rv_ac/la_ac/ra_ac scales, no bare base.
            def _ac_factor(ch):
                return fch_active_tmax_factor(SimDet, ch)

            act_lv = df.assemble(
                _ac_factor("lv") * df.inner(Fmat * Sactive_ventricular, Fmat) * MEmodel_.dx_me(lv_rid)
            )

            act_rv = df.assemble(
                _ac_factor("rv") * df.inner(Fmat * Sactive_ventricular, Fmat) * MEmodel_.dx_me(rv_rid)
            )

            act_la = df.assemble(
                _ac_factor("la") * df.inner(Fmat * Sactive_atrial, Fmat) * MEmodel_.dx_me(la_rid)
            )

            act_ra = df.assemble(
                _ac_factor("ra") * df.inner(Fmat * Sactive_atrial, Fmat) * MEmodel_.dx_me(ra_rid)
            )

            # Append a row to output_active.csv on rank 0, mirroring output_PV.csv structure
            if df.MPI.rank(comm_me) == 0 and active_writer is not None:
                active_writer.writerow([state_obj.tstep, act_lv, act_la, act_rv, act_ra])

        _t_prev, _cycle_prev = state_obj.t, state_obj.cycle
        state_obj.tstep = state_obj.tstep + state_obj.dt.dt
        state_obj.cycle = math.floor(state_obj.tstep / state_obj.BCL)
        state_obj.t = state_obj.tstep - state_obj.cycle * state_obj.BCL

        MEmodel_.t_a.vector()[:] = state_obj.t
        MEmodel_.cycle.vector()[:] = float(state_obj.cycle)
        MEmodel_.BCL = state_obj.BCL

        if _snap is not None and success_global:
            # The FE state now on the model is the solution at the ADVANCED time (the 0D step
            # returned the volumes for t + dt), so an event crossed by this advance is captured
            # here. Collective: every rank calls it with identical arguments.
            _snap.maybe_write(MEmodel_, t_prev=_t_prev, t_now=state_obj.t,
                              cycle=int(state_obj.cycle), cycle_prev=int(_cycle_prev),
                              tstep=state_obj.tstep)
        if _strain is not None and success_global and (cnt % _strain_step == 0):
            try:
                _strain.write_row(state_obj.tstep, state_obj.t, int(state_obj.cycle),
                                  pressure_offsets_pa=_ref_p)
            except Exception as _exc:  # noqa: BLE001 -- a readout must never kill the solve
                printout("WARNING: four-chamber strain row FAILED and the emitter is now OFF "
                         "for the rest of the run: %r" % (_exc,), comm_me,
                         stage="coupling", level="WARN")
                _strain = None

        isrestart = 0
        state_obj.dt.dt = delTat

        # At cycle start (t~0): reset EP state and check periodicity of the last two cycles.
        if enable_periodicity_checks and state_obj.t < state_obj.dt.dt:
            EPmodel_.reset()
            periodic_ok_local = 0
            if df.MPI.rank(comm_me) == 0:
                try:
                    eps_tol = SimDet.get("periodicity_eps", 1e-2)
                    mode = SimDet.get("periodicity_mode", "waveform").lower()
                    # Prefer in-memory cycle buffers; fall back to CSV parsing if buffers are incomplete.
                    have_two_cycles = (
                        cycle_buf_prev is not None and len(cycle_buf_curr["t"]) > 1
                    )
                    res = {}
                    if mode == "pvloop":
                        sigs = tuple(SimDet.get("periodicity_signals", ("LV",)))
                        if have_two_cycles:
                            res = evaluate_pvloop_periodicity_from_buffers(
                                cycle_buf_prev,
                                cycle_buf_curr,
                                eps=eps_tol,
                                signals=sigs,
                            )
                        else:
                            res = evaluate_pvloop_periodicity(
                                csv_path, eps=eps_tol, signals=sigs
                            )
                    else:
                        sigs = tuple(SimDet.get("periodicity_signals", ("LV", "RV")))
                        if have_two_cycles:
                            res = evaluate_full_waveform_periodicity_from_buffers(
                                cycle_buf_prev,
                                cycle_buf_curr,
                                eps=eps_tol,
                                signals=sigs,
                            )
                        else:
                            res = evaluate_full_waveform_periodicity(
                                csv_path, eps=eps_tol, signals=sigs
                            )

                    periodic_ok_local = 1 if res.get("ok", False) else 0
                    if write_periodicity_log:
                        rows = []
                        overall_ok = bool(res.get("ok", False))
                        for tag, info in res.get("signals", {}).items():
                            p_block = info.get("P", {})
                            v_block = info.get("V", {})
                            rows.append(
                                [
                                    state_obj.cycle,
                                    f"{state_obj.t:.6f}",
                                    mode,
                                    eps_tol,
                                    tag,
                                    p_block.get("rel_l2", ""),
                                    p_block.get("rel_linf", ""),
                                    v_block.get("rel_l2", ""),
                                    v_block.get("rel_linf", ""),
                                    bool(info.get("ok", False)),
                                    overall_ok,
                                ]
                            )
                        if rows:
                            _log_periodicity_rows(rows)
                except Exception:
                    periodic_ok_local = 0
            # Synchronize decision across ranks
            # Global decision (rank 0 computes metrics, then broadcasts pass/fail).
            periodic_ok = comm.allreduce(bool(periodic_ok_local), op=pyMPI.LOR)
            if periodic_ok:
                printout("Periodic solution reached; stopping.", comm_me)
                end_reason = "periodicity"
                break
            # Roll buffers so we compare last two cycles next time
            cycle_buf_prev = cycle_buf_curr
            cycle_buf_curr = _new_cycle_buf()

        # Throttled progress for the coupled time loop. This replaces a bare "Solving FHN" that
        # was emitted once per timestep and carried no time, cycle, pressure, volume or wall
        # clock -- thousands of lines from which a reader could not tell whether a multi-hour run
        # was advancing, stalled, or nearly done.
        if _loop_progress.due():
            _loop_progress.progress(
                state_obj.tstep, _loop_total_ms,
                cycle="%d/%d" % (state_obj.cycle, stop_iter),
                t_ms=state_obj.t,
                dt_ms=state_obj.dt.dt,
                p_lv_mmhg=P_LV * 0.0075,
                v_lv_ml=V_LV,
                p_rv_mmhg=P_RV * 0.0075,
                v_rv_ml=V_RV,
                coupling=_coupling_mode,
                wall_s=time.time() - _loop_t0,
            )

        if isrestart == 0:
            MEmodel_.UpdateVar()  # For damping
            EPmodel_.UpdateVar()

        # EP->ME coupling: interpolate transmembrane potential to the mechanics mesh for activation updates.
        potential_ref = EPmodel_.interpolate_potential_ep2me_phi(V_me=potential_me)
        potential_ref.rename("v_ref", "v_ref")
        # potential_me is already updated via V_me; avoid extra vector copy

        # --- Update activation timing ---
        MEmodel_.activeforms.update_activationTime(
            potential_n=potential_me, comm=comm_me
        )

        # Avoid expensive projections every step; compute only on write steps or when probes are active.
        compute_proj = (cnt % SimDet["writeStep"] == 0.0) or ("probepts" in SimDet)
        if compute_proj:
            # Access deformation gradient only when required by projections
            F_n = MEmodel_.get_deformation_gradient()
            df.project(
                MEmodel_.get_fiber_stress(),
                V_DG0,
                function=fstress_DG_fun,
                form_compiler_parameters={"representation": "uflacs"},
            )

            if probesfstress is not None:
                probesfstress(fstress_DG_fun)

            df.project(
                fStrain_uL,
                V_DG0,
                function=eff_fun,
                form_compiler_parameters={"representation": "uflacs"},
            )
            if probesEul_fiber is not None:
                probesEul_fiber(eff_fun)

        # Postprocess: project derived fields needed for probe/plot outputs.

        if compute_proj:
            # Compute IMP
            df.project(
                MEmodel_.get_imp(),
                V_DG1,
                function=imp_fun,
                form_compiler_parameters={"representation": "uflacs"},
            )

            df.project(
                MEmodel_.get_imp2(),
                V_DG1,
                function=imp2_fun,
                form_compiler_parameters={"representation": "uflacs"},
            )

            if probesIMP is not None:
                probesIMP(imp_fun)
            if probesIMP2 is not None:
                probesIMP2(imp2_fun)
            if probesIMP3 is not None:
                probesIMP3(MEmodel_.get_pressure_field())

                # broadcast from proc 0 to other processes
                rank = comm_me_.Get_rank()
                a = probesIMP3.array()  # probe will only send to rank =0
                if not rank == 0:
                    a = np.empty(len(x))

                comm_me_.Bcast(a, root=0)

        export.writePV(MEmodel_, state_obj.tstep)

        if cnt % SimDet["writeStep"] == 0.0:
            # HDF5 output is written in larger increments to reduce filesystem overhead on clusters.
            export.writetpt(MEmodel_, state_obj.tstep)
            export.hdf.write(MEmodel_.get_displacement(), "ME/u", writecnt)
            export.hdf.write(potential_ref, "ME/potential_ref", writecnt)

            # Heavy fields are computed when compute_proj is True, which includes write steps
            export.hdf.write(eff_fun, "ME/Eff", writecnt)
            export.hdf.write(fstress_DG_fun, "ME/fstress", writecnt)
            export.hdf.write(imp_fun, "ME/imp", writecnt)
            export.hdf.write(imp2_fun, "ME/imp2", writecnt)
            export.hdf.write(MEmodel_.get_pressure_field(), "ME/imp_constraint", writecnt)

            export.hdf.write(EPmodel_.getphivar(), "EP/phi", writecnt)
            export.hdf.write(EPmodel_.getrvar(), "EP/r", writecnt)
            export.hdf.write(potential_ref, "EP/potential_ref", writecnt)
            # INDEX the mixed state in the viz Data.h5 (analogous to the LV run_waorta fix):
            # the un-indexed write re-created "ME/w_me" every writeStep, which DOLFIN HDF5File
            # cannot overwrite -> the C error flooded the log AND only the FIRST step was kept.
            export.hdf.write(MEmodel_.w_me, "ME/w_me", writecnt)
            writecnt += 1
            # Dedicated restart checkpoint: full mixed FE state + CONSISTENT CL/time (same tstep
            # as this w_me) so a resume restores both without a write-step cadence mismatch.
            _ckpt.write(
                MEmodel_.w_me,
                {
                    "tstep": state_obj.tstep, "t": state_obj.t, "cycle": state_obj.cycle,
                    "V_LV": V_LV, "V_RV": V_RV, "V_LA": V_LA, "V_RA": V_RA,
                    "V_sa": V_sa, "V_ad": V_ad, "V_sv": V_sv, "V_pa": V_pa, "V_pv": V_pv,
                    "P_LV": P_LV, "P_RV": P_RV, "P_LA": P_LA, "P_RA": P_RA,
                },
            )
            # Batch and flush HDF5 writes to avoid handle churn
            try:
                export.hdf.flush()
            except Exception:
                pass

        if probesIMP is not None and compute_proj:
            fIMP = probesIMP.array()
            fIMP2 = probesIMP2.array()
            fIMP3 = probesIMP3.array()
            fStress = probesfstress.array()
            fStrain_vals = probesEul_fiber.array()
            E_circ_BiV = probesE_circ_BiV.array()
            E_long_BiV = probesE_long_BiV.array()
            E_radi_BiV = probesE_radi_BiV.array()

            export.writeIMP(MEmodel_, state_obj.tstep, fIMP)
            export.writeIMP2(MEmodel_, state_obj.tstep, fIMP2)
            export.writeIMP3(MEmodel_, state_obj.tstep, fIMP3)
            export.writefStress(MEmodel_, state_obj.tstep, fStress)
            export.writefStrain(MEmodel_, state_obj.tstep, fStrain_vals)
            export.writeCStrain(MEmodel_, state_obj.tstep, E_circ_BiV)
            export.writeLStrain(MEmodel_, state_obj.tstep, E_long_BiV)
            export.writeRStrain(MEmodel_, state_obj.tstep, E_radi_BiV)

        cnt += 1

        if state_obj.tstep % state_obj.BCL == 0:
            export.dump_restart_file(CLmodel_, V_LV, V_RV, V_LA, V_RA)

    # Terminal record for the coupling stage, mirroring the BEGIN emitted before the loop.
    _fe_log.info("coupling", "END", elapsed="%.1fs" % (time.time() - _loop_t0),
                 status="ok", reason=end_reason or "unknown",
                 cycles_done=state_obj.cycle, tstep_ms=state_obj.tstep)

    # After loop ends: write summary metrics once on rank 0 and synchronize
    try:
        if df.MPI.rank(comm_me) == 0:
            summary_path = write_cycle_summary_metrics(csv_path)
            if enable_periodicity_checks:
                try:
                    mode = SimDet.get("periodicity_mode", "waveform")
                    eps_tol = SimDet.get("periodicity_eps", "")
                    reason = end_reason if end_reason else "unknown"
                    if write_periodicity_log:
                        _log_periodicity_rows(
                            [
                                [
                                    state_obj.cycle,
                                    f"{state_obj.t:.6f}",
                                    "end",
                                    eps_tol,
                                    reason,
                                    "",
                                    "",
                                    "",
                                    "",
                                    "",
                                    "",
                                ]
                            ]
                        )
                except Exception:
                    pass
                try:
                    with open(summary_path, "r", newline="") as fh:
                        reader = csv.DictReader(fh)
                        rows = list(reader)
                    if rows and write_periodicity_metrics:
                        data = rows[-1]
                        metric_values = {}
                        tag_suffixes = {
                            "LV": ("EDV", "ESV", "SV", "EF", "P_EDP", "P_sys_peak"),
                            "RV": ("EDV", "ESV", "SV", "EF", "P_EDP", "P_sys_peak"),
                            "LA": (
                                "Vmax",
                                "Vmin",
                                "VpreA",
                                "P_mean",
                                "P_v_peak",
                                "P_a_amp",
                                "P_a_dur",
                                "t_v_peak",
                            ),
                        }
                        for tag, suffixes in tag_suffixes.items():
                            for suffix in suffixes:
                                key = f"{tag}_{suffix}"
                                value = data.get(key, "")
                                if value not in ("", None):
                                    metric_values[key] = value
                        if metric_values:
                            _log_periodicity_metrics(metric_values)
                except Exception:
                    pass
    except Exception as e:
        printout(f"Summary metrics write failed: {e}", comm_me)
    try:
        if hasattr(comm, "Barrier"):
            comm.Barrier()
    except Exception:
        pass

    # Close CSV handles cleanly on rank 0
    try:
        if _strain is not None:
            _strain.close()
        if df.MPI.rank(comm_me) == 0:
            if csv_fh is not None:
                csv_fh.flush()
                csv_fh.close()
            if active_fh is not None:
                active_fh.flush()
                active_fh.close()
    except Exception:
        pass

    # Close the restart checkpoint handle.
    try:
        _ckpt.close()
    except Exception:
        pass

    # --- Standalone invocation guard ---
    if __name__ == "__main__":
        printout("Testing...", comm_me)
        run_BiV_TimedGuccione(IODet=IODetails, SimDet=SimDetails)
