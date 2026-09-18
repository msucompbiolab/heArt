"""Build legacy IODet/SimDet for the LOCKED four-chamber surrogate (case-03).

The FCH analogue of ``orchestrate/lv_baseline_config.py``: it starts from the
canonical FCH baseline (``orchestrate/base_config.py``) and seeds it with the
locked ``CASE03_SURROGATE_FINAL = refine1/sample_0003`` parameter set committed at
``configs/surrogate_fch_calibration/`` (see that dir's ``PROVENANCE.txt``):

  selected_params.json:fe_material -> the authoritative 8 per-chamber ABSOLUTE FE
                            material dims for an approved calibration run.  The committed
                            material_params.json remains a standalone compatibility fallback.
                            (Cparam_{lv,rv,la,ra}, Tmax_{lv,rv,la,ra}), applied via
                            ``canonical.apply_fch_material`` (explicit kwargs write
                            straight into the SimDet the forward run reads -- the
                            CALIBRATION_PARAMS_OVERRIDE env channel instead writes the
                            SAMPLING groups and would NOT reach this run).
  selected_params.json   -> the locked 0D circuit; only the VASCULAR subset + initial
                            compartment volumes are transferred (``_VASCULAR_KEYS``).
                            The chamber-law keys (Ees_*/A_*/B_*/Cpass_*/Pact_*/V0_*/
                            Tmax_*) are the 0D lumped-chamber proxy and are INERT under
                            ``fch_fe`` coupling; the initial CHAMBER volumes
                            (V_LV/V_RV/V_LA/V_RA) are FE-derived at load time; the valve
                            thresholds/sigmoid are a different (calibration) valve model
                            -- the forward run uses circBiV's native hysteresis-diode.

Units are Pa / mL / ms (== legacy convention), so the locked values pass through
unchanged. The forward problem is the four-chamber coupled closed loop
(``demo/case24.py`` driver ``run_light.run_BiV_ClosedLoop``, ``fch_fe``): 3-D FE
LV/RV/LA/RA cavity pressures from the volume root-find, vascular state from
``circBiV.CLmodel``.

Mesh: the run loads the reference up to per-chamber EDP, so it MUST be the UNLOADED
case-03 reference (ventricles at ED, atria at atrial end-systole). Pass it via
``mesh_dir``; the base_config default (``meshes/fch_remesh``) is the *ED* remesh and
will over-fill at EDP.
"""

import copy
import json
import os
from pathlib import Path
from typing import Any, Optional, Tuple

try:
    from heArt_legacy.orchestrate.base_config import base_config
    from heArt_legacy.calibration.params import canonical as _canon
except ImportError:  # when heArt_legacy itself (not its parent) is on sys.path
    from orchestrate.base_config import base_config
    from calibration.params import canonical as _canon

REPO_DIR = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG_DIR = REPO_DIR / "configs" / "surrogate_fch_calibration"
DEFAULT_SELECTED = DEFAULT_CONFIG_DIR / "selected_params.json"
DEFAULT_MATERIAL = DEFAULT_CONFIG_DIR / "material_params.json"

# Vascular circuit + initial compartment-volume keys transferred from the locked 0D
# circuit into the forward run. (See the module docstring for what is deliberately
# NOT transferred and why.)
_VASCULAR_KEYS = (
    "Csa", "Cad", "Csv", "Cpa", "Cpv",
    "Rsa", "Rad", "Rsv", "Rav", "Rmv", "Rpa", "Rpv", "Rpvv", "Rtv", "Rav_rg",
    "Vsa0", "Vsv0", "Vad0", "Vpa0", "Vpv0",
    "V_sa", "V_ad", "V_sv", "V_pa", "V_pv",
)

_MATERIAL_KEYS = (
    "Cparam_lv", "Cparam_rv", "Cparam_la", "Cparam_ra",
    "Tmax_lv", "Tmax_rv", "Tmax_la", "Tmax_ra",
)


def _load_json(path: os.PathLike) -> dict:
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)


def _material_overrides(material_payload: dict) -> dict:
    """material_params.json is the override-file format {"parameters": {"material.<key>":
    value}}. Strip the ``material.`` prefix -> {<key>: value} for apply_fch_material."""
    params = _canon.param_overrides_from_payload(material_payload)  # {id: value}
    out: dict = {}
    for pid, val in params.items():
        short = pid.split(".", 1)[1] if "." in pid else pid
        out[short] = float(val)
    missing = [k for k in _MATERIAL_KEYS if k not in out]
    if missing:
        raise ValueError(f"material_params is missing required dims: {missing}")
    return out


def _selected_material_overrides(selected_payload: dict) -> dict | None:
    """Return the final run's authoritative FE material, when present.

    A partial block is refused instead of being silently completed from the legacy
    committed material file; that would mix two calibration runs in one FE solve.
    """
    material = selected_payload.get("fe_material")
    if material is None:
        return None
    if not isinstance(material, dict):
        raise ValueError("selected_params.fe_material must be an object")
    missing = [k for k in _MATERIAL_KEYS if k not in material]
    if missing:
        raise ValueError(f"selected_params.fe_material is incomplete: {missing}")
    return {k: float(material[k]) for k in _MATERIAL_KEYS}


def load_unload_fitted_passive(mesh_dir) -> Optional[dict]:
    """The per-chamber Cparam the unloaded reference in ``mesh_dir`` was recovered with
    (``unload_fitted_passive.json``, written by ``unloading fch``); None when absent."""
    path = os.path.join(str(mesh_dir), "unload_fitted_passive.json")
    if not os.path.exists(path):
        return None
    with open(path) as fh:
        doc = json.load(fh)
    out = {k: float(v) for k, v in doc.items() if k.startswith("Cparam_") and isinstance(v, (int, float))}
    if isinstance(doc.get("exponents"), dict):
        out["exponents"] = {ch: {k: float(v) for k, v in rec.items()} for ch, rec in doc["exponents"].items()}
    return out


def _log_cfg(level: str, message: str, **kv) -> None:
    try:
        from heartlog import Logger
        getattr(Logger("orchestrate"), level)("config", message, **kv)
    except Exception:  # noqa: BLE001 - never let logging kill config construction
        print("%s: %s %s" % (level.upper(), message, kv))


def build_fch_baseline_config(
    case_id: str = "fch_case03_surrogate",
    selected_path: Optional[os.PathLike] = None,
    material_path: Optional[os.PathLike] = None,
    mesh_dir: Optional[os.PathLike] = None,
    output_root: Optional[str] = None,
    stop_iter: int = 2,
    dt_ms: Optional[float] = None,
    nload_steps: Optional[int] = None,
    edp_mmhg: Optional[float] = None,
    edp_rv_mmhg: float = 4.0,
    edp_la_mmhg: float = 4.0,
    edp_ra_mmhg: float = 4.0,
    atria_unstressed_ed: bool = True,
    rv_cparam_operating: bool = True,
    tmax_lv: Optional[float] = None,
    tmax_rv: Optional[float] = None,
    tmax_la: Optional[float] = None,
    tmax_ra: Optional[float] = None,
    active_l0_um: Optional[float] = None,
    active_strain: bool = False,
    gamma_lv: Optional[float] = None,
    gamma_rv: Optional[float] = None,
    gamma_la: Optional[float] = None,
    gamma_ra: Optional[float] = None,
    # --- ported solver-robustness knobs (run_light parity with lv_baseline) -------
    viscous_eta: float = 0.0,
    dt_systole: Optional[float] = None,
    dt_relax: Optional[float] = None,
    dt_filling: Optional[float] = None,
    dt_relax_n_tau: float = 4.0,
    backoff_min_factor: Optional[float] = None,
    coupling_allow_nonconverged: bool = False,
    # --- fch_coupled campaign: coupling scheme, reference convention, arms ---------------
    coupling_mode: str = "partitioned_pressure",
    no_preload: bool = False,
    ed_volumes_ml: Optional[dict] = None,
    openloop_trace: Optional[os.PathLike] = None,
    event_snapshots: Optional[os.PathLike] = None,
    event_snapshot_first_cycle: int = 1,
    transverse_fraction: float = 0.0,
    transverse_structure: Optional[str] = None,
    inplane_angle: Optional[os.PathLike] = None,
    spring_epi: Optional[tuple] = None,
    spring_atrial: Optional[tuple] = None,
    spring_aorta: Optional[tuple] = None,
    aorta_cutface_robin: bool = False,
    spring_aorta_cutface: Optional[tuple] = None,
) -> Tuple[dict, dict]:
    IODet, SimDet = base_config()
    selected = _load_json(selected_path or DEFAULT_SELECTED)
    selected_material = _selected_material_overrides(selected)
    material = _load_json(material_path or DEFAULT_MATERIAL) if selected_material is None else None

    # 1) FE material (writes SimDet[<key>] + re-applies the canonical Guccione/Burkhoff
    #    substrate; the explicit kwargs win). The ventricular FE-operating contractility's
    #    SOURCE OF TRUTH is selected_params.fe_material when present; otherwise the legacy
    #    material_params.json compatibility seed.  Both use FCH absolute-Pa Tmax, not the
    #    nested 0D/lower-level active-material representation. The legacy locked value
    #    (416379/384059 Pa) is the fallback default; it FOLDS the full cycle at ejection
    #    (active-stress
    #    ellipticity ceiling), so the campaign explores a MODERATE additive contractility via
    #    the explicit tmax_lv/tmax_rv overrides. PASSIVE Cparam stays from material_params
    #    (matches the committed unloaded reference; the unload-cache key keys on Cparam, NOT
    #    per-chamber Tmax). RV passive uses the OPERATING soft stiffness; atrial active Tmax
    #    stay from material_params (no FE block; the atria do not fold).
    mat = selected_material or _material_overrides(material)
    tmax_lv_eff = float(tmax_lv) if tmax_lv is not None else mat["Tmax_lv"]
    tmax_rv_eff = float(tmax_rv) if tmax_rv is not None else mat["Tmax_rv"]
    tmax_la_eff = float(tmax_la) if tmax_la is not None else mat["Tmax_la"]
    tmax_ra_eff = float(tmax_ra) if tmax_ra is not None else mat["Tmax_ra"]
    # RV passive-stiffness reconciliation (operator decision, 2026-06-15). The locked
    # Cparam_rv (sampled ~380 Pa, sampling convention) leaves the thin near-indeterminate
    # RV too stiff to de-load -- the unload diverges on the RV ramp. The four-chamber
    # OPERATING convention deliberately uses the soft Cparam_rv (canonical FCH_BASE_PASSIVE,
    # 40 Pa; the closed loop + unloaded reference are built for it). With
    # rv_cparam_operating=True (default) the forward run + its unload use the operating RV
    # stiffness. Set False to use the sampled Cparam_rv (expect the unload to struggle).
    cparam_rv = (mat["Cparam_rv"] if selected_material is not None else
                 (float(_canon.value(_canon.FCH_BASE_PASSIVE, "Cparam_rv"))
                  if rv_cparam_operating else mat["Cparam_rv"]))
    _canon.apply_fch_material(
        SimDet,
        cparam_lv=mat["Cparam_lv"], cparam_rv=cparam_rv,
        cparam_la=mat["Cparam_la"], cparam_ra=mat["Cparam_ra"],
        tmax_lv=tmax_lv_eff, tmax_rv=tmax_rv_eff,
        tmax_la=tmax_la_eff, tmax_ra=tmax_ra_eff,
    )
    # 1a) MATCHED PAIR: a PRELOADED run (the mesh is an UNLOADED reference) must use the passive
    #     stiffness that reference was recovered with -- `<mesh_dir>/unload_fitted_passive.json`,
    #     the sidecar `unloading fch` writes -- never the material set's own Cparam (MEASURED
    #     2026-09-10: forward v4b/v5b ran the operating 40 Pa RV on references recovered at
    #     287 Pa and folded at the known ~11 mmHg RV-wall limit within 75 ms). Same rule the
    #     isolated-event path applies (fch_events.solve._load_unload_passive).
    if not no_preload and mesh_dir:
        fitted = load_unload_fitted_passive(mesh_dir)
        if fitted:
            for ch in ("lv", "rv", "la", "ra"):
                key = "Cparam_%s" % ch
                if key in fitted:
                    SimDet[key] = float(fitted[key])
            expo = fitted.get("exponents") or {}
            for ch, rec in expo.items():
                for k in ("bff", "bfx", "bxx"):
                    SimDet["%s_%s" % (k, ch)] = float(rec[k])
            _log_cfg("info", "passive stiffness taken from the reference's own unload fit "
                             "(matched pair)", **{k: SimDet[k] for k in ("Cparam_lv", "Cparam_rv",
                                                                          "Cparam_la", "Cparam_ra")},
                     **{("b_scale_%s" % ch): rec.get("b_scale") for ch, rec in expo.items()})
        else:
            _log_cfg("warn", "WARNING: preloaded run on a mesh with no unload_fitted_passive.json "
                             "beside it -- the material set's Cparam is used, so the passive "
                             "stiffness and the reference are NOT a matched pair",
                     mesh_dir=str(mesh_dir))

    # 1a) Burkhoff length-tension slack length l0 (um), opt-in override of the canonical 1.5.
    #    The active tension vanishes as the sarcomere length approaches l0 (the length-tension
    #    "dead zone" lambda_f -> l0/lr); MEASURED on the c287 reference (2026-09-11): at 115 mL the
    #    LV develops 52 mmHg at l0 1.50 but ~99 at 1.28, so l0 sets the reachable ESV floor.
    #    Applied to EVERY chamber (one shared Active params block); WARNed loudly because it is a
    #    modelling choice below the Guccione 1993 value (1.58 um).
    if active_l0_um is not None:
        SimDet["GiccioneParams"]["Active params"]["l0"] = float(active_l0_um)
        _log_cfg("warn", "WARNING: Burkhoff slack sarcomere length l0 OVERRIDDEN (all chambers); the "
                         "length-tension dead zone moves to shorter sarcomeres", l0_um=float(active_l0_um))

    # 1b) ACTIVE-STRAIN contraction (opt-in). Replaces the additive active stress with the
    #    multiplicative split F = Fe*Fa (per-chamber gamma = max fibre shortening REPLACES Tmax);
    #    removes the active-stress loss-of-ellipticity ceiling. MEmodel reads SimDet["active_strain"]
    #    + per-chamber SimDet["active_strain_gamma_{ch}"] (falls back to the global default 0.15).
    #    Tmax_* become inert for the swept chamber's residual. Default OFF = additive (above).
    if active_strain:
        SimDet["active_strain"] = True
        for ch, g in (("lv", gamma_lv), ("rv", gamma_rv), ("la", gamma_la), ("ra", gamma_ra)):
            if g is not None:
                SimDet["active_strain_gamma_%s" % ch] = float(g)

    # 2) 0D vascular circuit overlay (fch_fe -> chamber-law keys inert).
    clp = SimDet["closedloopparam"]
    locked_clp = selected.get("closedloopparam", {})
    transferred = {}
    for key in _VASCULAR_KEYS:
        if key in locked_clp:
            clp[key] = float(locked_clp[key])
            transferred[key] = clp[key]
    clp["stop_iter"] = int(stop_iter)

    # 3) Mesh dir (UNLOADED case-03 reference; see module docstring).
    if mesh_dir is not None:
        md = str(Path(mesh_dir)) + os.sep
        IODet["directory_me"] = md
        IODet["directory_ep"] = md

    # 4) Identity + output root (honor OUTPUT_BASE/OUTPUT_ROOT like base_config).
    IODet["caseID"] = str(case_id)
    output_root = (
        output_root
        or os.environ.get("OUTPUT_BASE")
        or os.environ.get("OUTPUT_ROOT")
        or str(REPO_DIR / "outputs_fch_baseline")
    )
    IODet["outputfolder"] = output_root

    # 5) Cycle / loading overrides.
    if dt_ms is not None:
        SimDet["dt"] = float(dt_ms)
    if nload_steps is not None:
        SimDet["nLoadSteps"] = int(nload_steps)

    # PER-CHAMBER loading EDP -- MUST match the unload's per-chamber --edp-* so the
    # inflate->ED round trip closes. run_light loads the LV until LVCavitypres*0.0075 >=
    # SimDet["EDP"] (mmHg) and ramps RV/LA/RA in LOCKSTEP by their P_* increments, so each
    # non-LV chamber ends at EDP * P_ch/P_LV. With the base P_* that drove RV to
    # ~EDP*P_RV/P_LV ~ 9 mmHg -- far past the unload's edp-rv (4 mmHg) -- so the RV
    # OVER-INFLATED past its ED volume into a loading fold (DIVERGED_PC_FAILED), and the
    # atria (loaded to 0) did not match their 4 mmHg unload reference. Fix: set each P_ch
    # so chamber ch reaches edp_ch_mmhg exactly when the LV reaches edp_lv. run_light's
    # 12 mmHg default EDP would also over-inflate; the LV target defaults to the locked
    # operating pressure (P_LV*0.0075) unless edp_mmhg is given.
    edp_lv = float(edp_mmhg) if edp_mmhg is not None else \
        float(SimDet.get("P_LV", 819.831289364)) * 0.0075
    SimDet["EDP"] = edp_lv
    p_lv = float(SimDet.get("P_LV", 819.831289364))
    SimDet["P_RV"] = p_lv * (float(edp_rv_mmhg) / edp_lv)
    SimDet["P_LA"] = p_lv * (float(edp_la_mmhg) / edp_lv)
    SimDet["P_RA"] = p_lv * (float(edp_ra_mmhg) / edp_lv)
    # Unloaded-reference semantics flag (atrial V0 fit / unload convention). The atria are
    # loaded to edp_la/edp_ra above (= the unload's atrial reference pressure, ~4 mmHg),
    # NOT 0 -- a 0 atrial load leaves them off their unloaded reference and distorts the
    # AV-interface inflation.
    SimDet["fch_atria_unstressed_ed"] = bool(atria_unstressed_ed)

    # 6) Ported robustness knobs (default OFF -> byte-identical baseline; the demo/job
    #    set production values). See src/sim_protocols/CLAUDE.md.
    if viscous_eta and viscous_eta > 0.0:
        SimDet["_passive_viscous_"] = True
        SimDet["GiccioneParams"].setdefault("Passive params", {})["eta"] = float(viscous_eta)
    if dt_systole is not None:
        SimDet["dt_systole"] = float(dt_systole)
    if dt_relax is not None:
        SimDet["dt_relax"] = float(dt_relax)
    if dt_filling is not None:
        SimDet["dt_filling"] = float(dt_filling)
    SimDet["dt_relax_n_tau"] = float(dt_relax_n_tau)
    if backoff_min_factor is not None:
        SimDet["backoff_min_factor"] = float(backoff_min_factor)
    SimDet["coupling_allow_nonconverged"] = bool(coupling_allow_nonconverged)

    # 7) fch_coupled campaign: coupling scheme + reference convention + arms.
    #    coupling_mode: "partitioned_pressure" (legacy 4-D pressure root-find, byte-identical),
    #    "volctrl4" (all four cavities under monolithic swept-volume control, ONE solve per
    #    step) or "openloop_volume" (volctrl4 driven by a recorded 0D trace, no feedback).
    coupling_mode = str(coupling_mode or "partitioned_pressure")
    if coupling_mode not in ("partitioned_pressure", "volctrl4", "openloop_volume"):
        raise ValueError("coupling_mode must be partitioned_pressure | volctrl4 | openloop_volume, "
                         "got %r" % coupling_mode)
    SimDet["fch_coupling_mode"] = coupling_mode
    if coupling_mode in ("volctrl4", "openloop_volume"):
        SimDet["fch_swept_volctrl"] = "lv,rv,la,ra"
        if SimDet.get("auto_rigid_support"):
            raise ValueError("%s is not wired with auto_rigid_support" % coupling_mode)
    if coupling_mode == "openloop_volume":
        if not openloop_trace:
            raise ValueError("coupling_mode=openloop_volume requires openloop_trace (a 0D output_PV.csv)")
        tp = Path(openloop_trace).expanduser()
        if not tp.exists():
            raise ValueError("openloop_trace does not exist: %s" % tp)
        SimDet["fch_openloop_trace"] = str(tp.resolve())
    elif openloop_trace:
        raise ValueError("openloop_trace is only consumed by coupling_mode=openloop_volume")
    #    no_preload: the mesh IS the ED state; per-chamber EDP becomes the reference-pressure
    #    offset (reported pressure = FE load + EDP_ch) and the loading phase is skipped. The
    #    ED-state manifest's measured volumes (ed_volumes_ml) are asserted at u = 0.
    SimDet["fch_no_preload"] = bool(no_preload)
    if no_preload:
        SimDet["fch_reference_pressure_pa"] = {
            "lv": edp_lv / 0.0075, "rv": float(edp_rv_mmhg) / 0.0075,
            "la": float(edp_la_mmhg) / 0.0075, "ra": float(edp_ra_mmhg) / 0.0075,
        }
        if ed_volumes_ml:
            vols = {str(k).lower(): float(v) for k, v in dict(ed_volumes_ml).items()}
            missing = [c for c in ("lv", "rv", "la", "ra") if c not in vols]
            if missing:
                raise ValueError("ed_volumes_ml must carry lv/rv/la/ra; missing %s" % missing)
            SimDet["fch_ed_volumes_ml"] = vols
    elif ed_volumes_ml:
        raise ValueError("ed_volumes_ml is only consumed under no_preload")
    #    Event-time displacement snapshots (Figure 6 frames), XDMF + HDF5 per event per cycle.
    if event_snapshots:
        sp = Path(event_snapshots).expanduser()
        if not sp.exists():
            raise ValueError("event_snapshots table does not exist: %s" % sp)
        SimDet["fch_event_snapshots"] = str(sp.resolve())
        SimDet["fch_event_snapshot_first_cycle"] = int(event_snapshot_first_cycle)
    #    Cross-fibre active tension (kappa on a MEASURED axis); validation is shared with every
    #    FCH entry point. Lazy import: this module is also read by dolfin-free callers
    #    (autocal.gates, material_finalize) and the src package init pulls dolfin.
    if float(transverse_fraction or 0.0) > 0.0 or transverse_structure or inplane_angle:
        try:
            from heArt_legacy.src.mechanics.transverse_config import apply_transverse_overrides
        except ImportError:
            from src.mechanics.transverse_config import apply_transverse_overrides
        SimDet["_fch_transverse"] = apply_transverse_overrides(
            SimDet, transverse_fraction, transverse_structure,
            str(inplane_angle) if inplane_angle else None)

    # 8) Pericardial Robin support arms (fch_coupled). Per-region [k_n, k_t] pairs in the units
    #    the form uses (see spring_bc_forms._build_fch_form); the LV+aorta finding transferred
    #    here is that the TANGENTIAL epicardial stiffness is the longitudinal-strain lever
    #    (k_t ~ 200 with k_n ~ 4000-5000 was the accepted optimum). `aorta_cutface_robin` swaps
    #    the hard u=0 pin on the aortic cut plane for a Robin rim.
    def _pair(name, val):
        if val is None:
            return None
        v = [float(x) for x in val]
        if len(v) != 2 or min(v) < 0.0:
            raise ValueError("%s must be [k_n, k_t] >= 0, got %r" % (name, val))
        return v
    for key, val in (("springparam_epi", _pair("spring_epi", spring_epi)),
                     ("springparam_atrial", _pair("spring_atrial", spring_atrial)),
                     ("springparam_aorta", _pair("spring_aorta", spring_aorta)),
                     ("springparam_aorta_cutface", _pair("spring_aorta_cutface", spring_aorta_cutface))):
        if val is not None:
            SimDet[key] = val
    if aorta_cutface_robin:
        SimDet["aorta_cutface_robin"] = True
    SimDet["_fch_support"] = {k: SimDet.get(k) for k in ("springparam_epi", "springparam_atrial",
                                                          "springparam_aorta", "springparam_aorta_cutface",
                                                          "aorta_cutface_robin", "dashpotparam")}

    # Provenance: stash the applied locked values for a startup log + audit.
    SimDet["_fch_surrogate_provenance"] = {
        "source": ("selected_params.fe_material" if selected_material is not None
                   else "configs/surrogate_fch_calibration (refine1/sample_0003)"),
        "material": {k: SimDet[k] for k in _MATERIAL_KEYS},
        "ventricular_active_tmax": (
            f"Tmax_lv={SimDet['Tmax_lv']:.0f} Tmax_rv={SimDet['Tmax_rv']:.0f} Pa "
            + ("(EXPLICIT override -- moderate-Tmax campaign exploration)"
               if (tmax_lv is not None or tmax_rv is not None)
               else "(default = material_params FE-operating Tmax; folds the full cycle)")),
        "vascular_transferred": transferred,
    }
    return IODet, SimDet
