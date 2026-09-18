"""Run the LOCKED four-chamber surrogate (case-03) as a full coupled FE closed loop.

Seeds the canonical FCH baseline with the locked CASE03_SURROGATE_FINAL =
refine1/sample_0003 parameter set (configs/surrogate_fch_calibration/) and runs the
four-chamber driver (src/sim_protocols/run_light.run_BiV_ClosedLoop, fch_fe). The
FCH analogue of demo/lv_baseline.py.

Usage:
    source ../activate_fenics_mamba.sh         # legacy dolfin 2019.1.0 env
    mpirun -np 8 python3 demo/fch_baseline.py \
        --mesh-dir <UNLOADED case-03 ref dir> --output-root <dir> --stop-iter 3 \
        [--viscous-eta 500] [--dt-systole 0.5 --dt-relax 1.0 --dt-filling 2.0]

The mesh dir MUST hold the UNLOADED case-03 reference (fch_clregion_whole.hdf5 +
fch_clregion_refine.hdf5); the run loads it to per-chamber EDP.
"""

import argparse
import sys
from pathlib import Path

REPO_DIR = Path(__file__).resolve().parents[1]
MONO_ROOT = Path(__file__).resolve().parents[2]
for path in (REPO_DIR, MONO_ROOT):
    if str(path) not in sys.path:
        sys.path.append(str(path))

from heArt_legacy.orchestrate.fch_baseline_config import build_fch_baseline_config
from heArt_legacy.src.sim_protocols.run_light import run_BiV_ClosedLoop


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Locked FCH case-03 surrogate full-cycle run.")
    p.add_argument("--case_ID", default="fch_case03_surrogate")
    p.add_argument("--params", default=None,
                   help="selected_params.json (default: configs/surrogate_fch_calibration/).")
    p.add_argument("--material", default=None,
                   help="material_params.json (default: configs/surrogate_fch_calibration/).")
    p.add_argument("--mesh-dir", default=None,
                   help="Dir with the UNLOADED case-03 ref (default: base_config meshes/fch_remesh "
                        "-- which is the ED remesh, NOT unloaded; override for a correct run).")
    p.add_argument("--output-root", default=None)
    p.add_argument("--stop-iter", type=int, default=2, help="Number of cardiac cycles to run.")
    p.add_argument("--dt", type=float, default=None, help="Time step (ms); default = base_config.")
    p.add_argument("--nload-steps", type=int, default=None, help="Loading increments; default = base_config.")
    p.add_argument("--write-step", type=int, default=None,
                   help="Write outputs + a restart checkpoint every N coupled steps (default = base_config).")
    p.add_argument("--edp-mmhg", type=float, default=None,
                   help="LV loading end-diastolic pressure (mmHg). Default = locked LV operating "
                        "pressure (P_LV*0.0075 ~ 6.1); run_light's 12 mmHg default over-inflates. "
                        "MUST match the unload --edp-lv for a consistent round trip.")
    p.add_argument("--edp-rv-mmhg", type=float, default=4.0,
                   help="RV loading EDP (mmHg); MUST match the unload --edp-rv (default 4). The "
                        "RV loads in lockstep with the LV, so this sets P_RV = P_LV*edp_rv/edp_lv "
                        "to avoid over-inflating the near-indeterminate RV past its ED volume.")
    p.add_argument("--edp-la-mmhg", type=float, default=4.0,
                   help="LA loading EDP (mmHg); match the unload atrial reference (~4).")
    p.add_argument("--edp-ra-mmhg", type=float, default=4.0,
                   help="RA loading EDP (mmHg); match the unload atrial reference (~4).")
    p.add_argument("--sampled-rv-cparam", action="store_true", default=False,
                   help="Use the locked SAMPLED Cparam_rv (~380 Pa) instead of the operating soft "
                        "RV stiffness (~40 Pa). Default uses the operating value -- the sampled RV "
                        "is too stiff to de-load (the unload diverges). Other 7 locked dims kept.")
    p.add_argument("--tmax-lv", type=float, default=None,
                   help="Override the LV additive active contractility Tmax (Pa). Default = the "
                        "locked material_params FE-operating value (416379), which folds the full "
                        "cycle at the ejection transition. Use a MODERATE value (e.g. 150000) to "
                        "explore a contractility the full-cycle FE can converge. Does NOT affect "
                        "the passive unload (cache reused).")
    p.add_argument("--tmax-rv", type=float, default=None,
                   help="Override the RV additive active contractility Tmax (Pa). Default = locked "
                        "material_params (384059). Pair a moderate value with --tmax-lv.")
    p.add_argument("--tmax-la", type=float, default=None,
                   help="Override the LA active Tmax (Pa). Default = locked material_params (54662). "
                        "Pass all four in the calibrated active-twitch ratio when exploring.")
    p.add_argument("--tmax-ra", type=float, default=None,
                   help="Override the RA active Tmax (Pa). Default = locked material_params (295774).")
    p.add_argument("--active-l0", type=float, default=None,
                   help="Override the Burkhoff slack sarcomere length l0 (um) for every chamber "
                        "(canonical 1.5). Lower values keep active tension at shorter sarcomeres "
                        "(smaller ESV); a modelling choice below Guccione 1993's 1.58, WARNed.")
    p.add_argument("--active-strain", action="store_true", default=False,
                   help="Use the multiplicative active-strain split F=Fe*Fa (per-chamber gamma "
                        "REPLACES Tmax; removes the active-stress ellipticity ceiling). Pair with "
                        "--gamma-{lv,rv,la,ra}. Default OFF = additive active stress.")
    p.add_argument("--gamma-lv", type=float, default=None, help="Active-strain max fibre shortening, LV (~0.1-0.3).")
    p.add_argument("--gamma-rv", type=float, default=None, help="Active-strain gamma, RV.")
    p.add_argument("--gamma-la", type=float, default=None, help="Active-strain gamma, LA.")
    p.add_argument("--gamma-ra", type=float, default=None, help="Active-strain gamma, RA.")
    p.add_argument("--legacy-atrial-edp", action="store_true", default=False,
                   help="Load the atria to a positive EDP (legacy). Default: atria held at ~0 "
                        "(fch_atria_unstressed_ed) to match an `unloading fch --atria-unstressed` "
                        "reference (atria at atrial end-systole at the ventricular-ED frame).")
    # --- ported robustness knobs (run_light parity with lv_baseline) -------------
    p.add_argument("--viscous-eta", type=float, default=0.0,
                   help="Kelvin-Voigt viscous regularization eta (Pa*ms): adds ~2*eta/dt to the "
                        "tangent to traverse the late-systolic limit point. 0 disables. "
                        "Production-recommended starting value: 500.")
    p.add_argument("--dt-systole", type=float, default=None,
                   help="Fine dt (ms) for contraction+ejection (the limit-point zone; also boosts "
                        "the viscous tangent 2*eta/dt there). Setting any phase dt enables the policy.")
    p.add_argument("--dt-relax", type=float, default=None,
                   help="Moderate dt (ms) for isovolumic relaxation + early filling.")
    p.add_argument("--dt-filling", type=float, default=None,
                   help="Large dt (ms) for the active-free filling/diastasis plateau.")
    p.add_argument("--dt-relax-n-tau", type=float, default=4.0,
                   help="Relaxation ends (filling begins) at t_trans + n*tau.")
    p.add_argument("--backoff-min-factor", type=float, default=None,
                   help="Min dt-backoff factor before the coupling step fails (default base_config 0.25).")
    p.add_argument("--allow-nonconverged", action="store_true", default=False,
                   help="Keep the legacy lenient 'proceed with pre-step pressures' path on backoff "
                        "exhaustion (emits a loud WARNING). Default: fail loudly (raise).")
    p.add_argument("--single-solve-timing", action="store_true", default=False,
                   help="Run setup + loading + ONE coupled closed-loop solve, then exit (smoke/profiling).")
    p.add_argument("--newton-linesearch", action="store_true", default=False,
                   help="Globalized (backtracking line-search, damped) Newton for the FE solve "
                        "(SimDet['newton_linesearch']). Traverses the early-systolic active-stress "
                        "loss-of-ellipticity fold that stock full-step Newton overshoots; converges "
                        "to the true solution or raises (never masks).")
    # --- fch_coupled campaign: coupling scheme, reference convention, arms -----------------
    p.add_argument("--coupling", choices=("pressure", "volctrl4", "openloop"),
                   default="pressure",
                   help="FE-0D coupling scheme: 'pressure' = legacy 4-D cavity-pressure root-find "
                        "(default); 'volctrl4' = all four cavities under monolithic swept-volume "
                        "control, one solve per step, pressures = the multipliers (the HFpEF "
                        "LV+aorta scheme); 'openloop' = volctrl4 driven by --openloop-trace, "
                        "no feedback (PRELIMINARY one-way coupling).")
    p.add_argument("--no-preload", action="store_true", default=False,
                   help="The mesh IS the end-diastolic state: skip the inflation loading phase, "
                        "treat the ED geometry as loaded at the per-chamber EDPs via a uniform "
                        "reference-pressure offset, assert the FE cavity volumes against "
                        "--ed-manifest. Direct-ED campaign policy (ed_reference_no_preload).")
    p.add_argument("--ed-manifest", default=None,
                   help="ed_state_manifest.v1 JSON (measured per-chamber ED volumes) asserted "
                        "at u=0 under --no-preload.")
    p.add_argument("--openloop-trace", default=None,
                   help="0D output_PV.csv whose last complete cycle drives the chamber volumes "
                        "under --coupling openloop.")
    p.add_argument("--event-snapshots", default=None,
                   help="fch_events event-table JSON ({events:[{key,t_ms}]}) or {key: t_ms} "
                        "mapping: write the displacement (HDF5 + XDMF) when each event time is "
                        "crossed, every cycle from --event-snapshot-first-cycle on.")
    p.add_argument("--event-snapshot-first-cycle", type=int, default=1)
    p.add_argument("--transverse-fraction", type=float, default=0.0,
                   help="Cross-fibre active fraction kappa in [0,1] (0 = fibre-only, default).")
    p.add_argument("--transverse-structure", choices=("sheet", "normal", "inplane"),
                   default=None,
                   help="Transverse axis for kappa>0; MUST be measured (fch_events basis). "
                        "'inplane' needs --inplane-angle.")
    p.add_argument("--inplane-angle", default=None,
                   help="Measured per-cell in-plane angle field (fch_events basis --emit-axis).")
    p.add_argument("--spring-epi", default=None, help="ventricular epicardial Robin [k_n,k_t] e.g. 4000,200")
    p.add_argument("--spring-atrial", default=None, help="atrial-wall Robin [k_n,k_t]")
    p.add_argument("--spring-aorta", default=None, help="aorta-wall Robin [k_n,k_t] (default 1.5x epi)")
    p.add_argument("--aorta-cutface-robin", action="store_true", default=False,
                   help="Robin rim on the aortic cut plane instead of the hard u=0 pin")
    p.add_argument("--spring-aorta-cutface", default=None, help="cut-plane Robin [k_n,k_t] (default 3000,3000)")
    p.add_argument("--restart", action="store_true", default=False,
                   help="Resume from the dedicated checkpoint (restart_state.h5/.csv) in the run's "
                        "output dir. ONLY for a SAME-PARAMETER continuation (timeout / node failure / "
                        "extending an interrupted run): the restart manifest is validated and the "
                        "resume is REFUSED if any trajectory-defining input changed (material, EDP, "
                        "dt/solver knobs, viscous eta, backoff, mesh, unloaded ref, or code/config).")
    return p


def _pair_arg(text):
    if not text:
        return None
    parts = [float(x) for x in str(text).replace(";", ",").split(",") if x.strip()]
    if len(parts) != 2:
        raise SystemExit("expected 'k_n,k_t', got %r" % text)
    return parts


def _ed_volumes_from_manifest(path):
    """Per-chamber ED cavity volumes (mL) from an ed_state_manifest.v1 JSON, or None."""
    if not path:
        return None
    import json as _json
    with open(path) as fh:
        doc = _json.load(fh)
    edv = doc.get("edv_ml")
    if not isinstance(edv, dict):
        raise SystemExit("--ed-manifest %s carries no per-chamber 'edv_ml' mapping" % path)
    return {k: float(v) for k, v in edv.items()}


def main(argv=None) -> None:
    args = build_parser().parse_args(argv)
    IODet, SimDet = build_fch_baseline_config(
        case_id=args.case_ID,
        selected_path=args.params,
        material_path=args.material,
        mesh_dir=args.mesh_dir,
        output_root=args.output_root,
        stop_iter=args.stop_iter,
        dt_ms=args.dt,
        nload_steps=args.nload_steps,
        edp_mmhg=args.edp_mmhg,
        edp_rv_mmhg=args.edp_rv_mmhg,
        edp_la_mmhg=args.edp_la_mmhg,
        edp_ra_mmhg=args.edp_ra_mmhg,
        atria_unstressed_ed=not args.legacy_atrial_edp,
        rv_cparam_operating=not args.sampled_rv_cparam,
        tmax_lv=args.tmax_lv,
        tmax_rv=args.tmax_rv,
        tmax_la=args.tmax_la,
        tmax_ra=args.tmax_ra,
        active_l0_um=args.active_l0,
        active_strain=args.active_strain,
        gamma_lv=args.gamma_lv,
        gamma_rv=args.gamma_rv,
        gamma_la=args.gamma_la,
        gamma_ra=args.gamma_ra,
        viscous_eta=args.viscous_eta,
        dt_systole=args.dt_systole,
        dt_relax=args.dt_relax,
        dt_filling=args.dt_filling,
        dt_relax_n_tau=args.dt_relax_n_tau,
        backoff_min_factor=args.backoff_min_factor,
        coupling_allow_nonconverged=args.allow_nonconverged,
        coupling_mode={"pressure": "partitioned_pressure", "volctrl4": "volctrl4",
                       "openloop": "openloop_volume"}[args.coupling],
        no_preload=args.no_preload,
        ed_volumes_ml=_ed_volumes_from_manifest(args.ed_manifest),
        openloop_trace=args.openloop_trace,
        event_snapshots=args.event_snapshots,
        event_snapshot_first_cycle=args.event_snapshot_first_cycle,
        transverse_fraction=args.transverse_fraction,
        transverse_structure=args.transverse_structure,
        inplane_angle=args.inplane_angle,
        spring_epi=_pair_arg(args.spring_epi),
        spring_atrial=_pair_arg(args.spring_atrial),
        spring_aorta=_pair_arg(args.spring_aorta),
        aorta_cutface_robin=args.aorta_cutface_robin,
        spring_aorta_cutface=_pair_arg(args.spring_aorta_cutface),
    )
    if args.single_solve_timing:
        SimDet["single_solve_timing"] = True
    if args.write_step is not None:
        SimDet["writeStep"] = float(args.write_step)
    if args.newton_linesearch:
        SimDet["newton_linesearch"] = True

    # Startup provenance log (rank 0): surface the locked values actually seeded.
    try:
        import dolfin as df
        if df.MPI.rank(df.MPI.comm_world) == 0:
            prov = SimDet.get("_fch_surrogate_provenance", {})
            print(f"[fch_baseline] seeded from {prov.get('source')}")
            print(f"[fch_baseline] material: {prov.get('material')}")
            print(f"[fch_baseline] vascular keys transferred: "
                  f"{sorted((prov.get('vascular_transferred') or {}).keys())}")
            print(f"[fch_baseline] mesh dir: {IODet['directory_me']}  stop_iter={args.stop_iter}  "
                  f"viscous_eta={args.viscous_eta}")
    except Exception:
        pass

    # Checkpoint/restart: a restart-defining manifest is stamped on a fresh run and validated
    # on --restart (the resume is REFUSED if any trajectory-defining input changed). Guarded so
    # only a same-parameter continuation resumes; tuning changes must rerun fresh.
    import os as _os
    from heArt_legacy.src.sim_protocols.fch_restart import (
        write_restart_manifest, validate_restart_manifest,
    )
    out_dir = _os.path.join(IODet["outputfolder"], IODet.get("folderName", ""), IODet["caseID"])
    if args.restart:
        validate_restart_manifest(out_dir, IODet, SimDet)  # raises (refuses) on any mismatch
        SimDet["isrestart"] = True
    else:
        try:
            import dolfin as _df
            if _df.MPI.rank(_df.MPI.comm_world) == 0:
                write_restart_manifest(out_dir, IODet, SimDet)
        except Exception:
            pass

    run_BiV_ClosedLoop(IODet=IODet, SimDet=SimDet)


if __name__ == "__main__":
    main()
