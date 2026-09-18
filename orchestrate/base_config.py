import copy
import os
from typing import Any, Optional

from dolfin import Constant

# Single authoritative parameter source (calibration/params/canonical.py). The FCH
# closed-loop OPERATING values -- per-chamber materials, the fallback Guccione/Burkhoff
# substrate, the mechanics support spring, the mesh scale, the cycle scalars, and the
# region/facet id contract -- live in the FCH_BASE_* groups there (mirrors how
# orchestrate/lv_baseline_config builds the LV SimDet from the LV_* groups). This module
# READS those values; it does not define material literals. The 0D circulation
# (Circparam), the initial cavity pressures, EP diffusion, solver tolerances, the
# fiber-dataset names and the mesh file/HDF5 paths remain configured here (they are not
# myocardial-material parameters).
try:
    from heArt_legacy.calibration.params import canonical as _canon
except ImportError:  # when heArt_legacy itself (not its parent) is on sys.path
    from calibration.params import canonical as _canon


def _warn_ho_scale(scale, params):
    """WARN (heartlog) that the Holzapfel-Ogden moduli were scaled away from the canonical set."""
    try:
        from heartlog import Logger
        Logger("orchestrate").warn("config", "WARNING: HO_A_SCALE applied -- Holzapfel-Ogden "
                                   "moduli scaled off the canonical set (a, a_f, a_s, a_fs)",
                                   scale=scale, a_f_pa=params["a_f"], a_pa=params["a"])
    except Exception:  # noqa: BLE001 - never let logging kill config construction
        print("WARNING: HO_A_SCALE=%g applied to the Holzapfel-Ogden moduli (a_f=%g Pa)"
              % (scale, params["a_f"]))


def _output_root() -> str:
    return os.environ.get("OUTPUT_BASE") or os.environ.get(
        "OUTPUT_ROOT", f"/mnt/gs21/scratch/{os.environ.get('USER', '')}/outputs_realfch"
    )


def _replace_constants(obj: Any) -> Any:
    """Recursively convert dolfin Constant objects to plain floats for copying."""
    if isinstance(obj, Constant):
        return float(obj)
    if isinstance(obj, dict):
        return {key: _replace_constants(value) for key, value in obj.items()}
    if isinstance(obj, list):
        return [_replace_constants(item) for item in obj]
    if isinstance(obj, tuple):
        return tuple(_replace_constants(item) for item in obj)
    return obj


def base_config(
    passive_viscous: Optional[bool] = None,
    collapse_passive_regions: Optional[bool] = None,
    legacy_fsolve: Optional[bool] = None,
):
    """
    Return baseline IODet and SimDet dictionaries for FCH BiV closed-loop runs.

    This mirrors the configuration used in demo/case25.py and is suitable for
    orchestration workflows. Callers should deepcopy or modify the returned
    objects as needed per-case.

    All myocardial-material / support / mesh-scale / cycle / region-id values come
    from the canonical FCH_BASE_* groups (calibration/params/canonical.py); they are
    not literals here. Toggling passive_viscous / collapse_passive_regions /
    legacy_fsolve sets the corresponding SimDet flag.
    """

    # Pull the canonical FCH closed-loop operating groups (single source).
    _passive = _canon.values(_canon.FCH_BASE_PASSIVE)   # Cparam(+per-chamber)/bff/bfx/bxx/eta/Kappa
    _active = _canon.values(_canon.FCH_BASE_ACTIVE)      # Tmax(+per-chamber)/timing
    _support = _canon.values(_canon.FCH_BASE_SUPPORT)    # springparam/dashpotparam
    _geom = _canon.values(_canon.FCH_BASE_GEOMETRY)      # mesh_scale_fch/mesh_subdir/mesh_basename
    _cycle = _canon.values(_canon.FCH_BASE_CYCLE)        # HeartBeatLength/dt/writeStep/nLoadSteps
    _rid = _canon.values(_canon.FCH_BASE_REGIONS)        # region/facet ids
    _unload = _canon.values(_canon.FCH_BASE_UNLOAD)      # atria_unstressed_ed + atrial V0 bands

    # Canonical FCH mesh dir = the big-RV Strocchi remesh under the repo
    # (canonical mesh_subdir); drivers may still override via --mesh-dir.
    _repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    _fch_mesh_dir = os.path.join(_repo_root, _geom.get("mesh_subdir", "meshes/fch_remesh"))

    IODet = {
        "casename": "fch_clregion",
        "directory_me": _fch_mesh_dir,
        "directory_ep": _fch_mesh_dir,
        "outputfolder": _output_root(),
        "folderName": "",
        "caseID": "case0",
        "matid_dataset": "matid1",
    }

    GuccioneParams = {
        "ParamsSpecified": True,
        "Passive model": {"Name": "Guccione"},
        "Passive params": {
            # LV-consistent passive substrate from canonical FCH_BASE_PASSIVE
            # (fallback Cparam=100, bff=29). Per-chamber absolute overrides are
            # applied via Cparam_{lv,rv,la,ra} below (direct values, no multipliers).
            "Cparam": _passive["Cparam"],
            "bff": _passive["bff"],
            "bfx": _passive["bfx"],
            "bxx": _passive["bxx"],
            "eta": _passive["eta"],
        },
        "Active model": {"Name": "Time-varying"},
        "Active params": {
            "tau": _active["tau"],               # ventricular relaxation time constant (ms)
            "tau_atr": _active["tau_atr"],        # atrial relaxation (faster than ventricle)
            "t_trans": _active["t_trans"],
            "t_trans_atr": _active["t_trans_atr"],
            "B": _active["B"],
            "t0": _active["t0"],
            "t0_atr": _active["t0_atr"],
            "tdelay_atr": _active["tdelay_atr"],
            "l0": _active["l0"],                  # length-tension slack length (um)
            "Tmax": _active["Tmax"],              # fallback contractility (per-chamber Tmax_* below)
            "Ca0": _active["Ca0"],
            "Ca0max": _active["Ca0max"],
            "lr": _active["lr"],
        },
        "HomogenousActivation": True,
        "deg": 4,
        "Kappa": _passive["Kappa"],
        "incompressible": True,
    }

    # OPT-IN Holzapfel-Ogden passive law (default stays Guccione). Selected via env
    # PASSIVE_MODEL=HolzapfelOgden (or a caller can overwrite GiccioneParams["Passive model"]
    # /["Passive params"] post-hoc). The stiff HO fiber term (a_f), DECOUPLED from the soft
    # bulk (a), is what lets active-strain develop physiological systolic tension — the single
    # soft Guccione Cparam starves it (src/mechanics/CLAUDE.md "ACTIVE-STRAIN"; canonical.HO_PASSIVE).
    # CAVEAT (Stage B): MEmodel's per-chamber passive scaling (_region_passive_scale) keys off the
    # Guccione "Cparam"; the multi-region FCH path needs an HO-aware branch before per-chamber HO.
    if os.environ.get("PASSIVE_MODEL", "Guccione") == "HolzapfelOgden":
        _ho = _canon.values(_canon.HO_PASSIVE)
        GuccioneParams["Passive model"] = {"Name": "HolzapfelOgden"}
        GuccioneParams["Passive params"] = {
            "a": _ho["a"], "b": _ho["b"],
            "a_f": _ho["a_f"], "b_f": _ho["b_f"],
            "a_s": _ho["a_s"], "b_s": _ho["b_s"],
            "a_fs": _ho["a_fs"], "b_fs": _ho["b_fs"],
            "eta": _passive["eta"],
        }
        GuccioneParams["Kappa"] = _ho["Kappa"]
        # OPT-IN stiffness scale HO_A_SCALE (multiplies every a* modulus, exponents untouched):
        # the canonical HO set is ~10x stiffer than the anchored Guccione references
        # (memory: guccione-to-ho-stiffness-mismatch), and the LV+aorta active-strain arms
        # reached ~80 mmHg only at a_f x0.2. Any value != 1 is a WARNed deviation from canon.
        _ho_scale = float(os.environ.get("HO_A_SCALE", "1.0"))
        if _ho_scale != 1.0:
            if not (0.05 <= _ho_scale <= 2.0):
                raise ValueError("HO_A_SCALE=%g outside the sanity band [0.05, 2.0]" % _ho_scale)
            for _k in ("a", "a_f", "a_s", "a_fs"):
                GuccioneParams["Passive params"][_k] *= _ho_scale
            _warn_ho_scale(_ho_scale, GuccioneParams["Passive params"])

    Circparam = {
        # Capacitances
        "Csa": 0.013,
        "Cad": 0.033,
        "Csv": 0.28,
        "Cpa": 0.01,
        "Cpv": 0.6,
        # Resistances
        "Rsa": 50000,
        "Rad": 55000,
        "Rav": 500.0,
        # Aortic-valve / proximal-aorta inertance (enables inertial valve dynamics).
        "Lav": 20000.0,
        # Aortic valve hysteresis thresholds (Pa) to avoid chatter when P_LV ~ P_sa.
        # ~133 Pa ≈ 1 mmHg.
        "dPav_open": 133.0,
        "dPav_close": 133.0,
        "Rsv": 100.0,
        "Rmv": 170.0,
        "Rpa": 8000.0,
        "Rpv": 1100.0,
        "Rpvv": 400,
        "Rtv": 800.0,
        # Unstressed volumes
        "Vsa0": 400,
        "Vsv0": 2550.0,
        "Vad0": 40,
        "Vpa0": 320,
        "Vpv0": 90,
        # Initial blood volumes and solver control
        "V_sv": 2818.50521502,
        "V_sa": 468.027406313,
        "V_ad": 160.544898276,
        "V_pa": 329.190706911,
        "V_pv": 720.775534117,
        "stop_iter": 2,
    }

    SimDet = {
        "diaplacementInfo_ref": False,
        "HeartBeatLength": _cycle["HeartBeatLength"],
        "dt": _cycle["dt"],
        "writeStep": _cycle["writeStep"],
        "GiccioneParams": GuccioneParams,
        "nLoadSteps": _cycle["nLoadSteps"],
        "DTI_EP": False,
        "DTI_ME": False,
        "_passive_viscous_": False,
        "d_iso": 1.5 * 0.01,
        "d_ani_factor": 4.0,
        "ploc": [[1.4, 1.4, -3.0, 2.0, 1]],
        "pacing_timing": [[4.0, 20.0]],
        "Isclosed": True,
        "closedloopparam": Circparam,
        "Ischemia": False,
        "Mechanics Discretization": "P1P1",
        "Technique Discretization": 1,
        "isLV": False,
        "facetboundaries_dataset": "facetboundaries2",
        # Optional mesh scale applied to FCH mechanics meshes (canonical
        # FCH_BASE_GEOMETRY.mesh_scale_fch). EF/ratio are scale-invariant (cavity vol
        # ~scale^3), so scale only sets the absolute EDV magnitude. 0.076 is for the
        # NATIVE (un-morphed) mesh (see fch-unload-oscillation-verdict / fch-tolerant-
        # targets memory): native RV/LV ~0.59, feasible under the tolerant targets;
        # 0.076 lifts the ED RV to ~103 mL and LV to ~177 mL.
        "mesh_scale_fch": _geom["mesh_scale_fch"],
        "aorta_wall": _rid["aorta_wall"],
        "LVendoid": _rid["LVendoid"],
        "RVendoid": _rid["RVendoid"],
        "LAendoid": _rid["LAendoid"],
        "RAendoid": _rid["RAendoid"],
        "epiid": _rid["epiid"],
        "atrialid": _rid["atrialid"],
        "apxid": _rid["apxid"],
        "aortaid": _rid["aortaid"],
        "abs_tol": 1e-8,
        "rel_tol": 1e-9,
        "isunloading": False,
        "isunloadingonly": False,
        "ispctrl": True,
        "isFCH": True,
        "springbc": 1,
        # WS3-validated PER-REGION spring stiffness [k_n, k_t] (canonical FCH_BASE_SUPPORT):
        # ventricular epicardium + atrial wall set independently (read by
        # spring_bc_forms._build_fch_form). `springparam` is the legacy single-surface
        # fallback for any surface without a per-region override.
        "springparam": list(_support["springparam"]),
        "springparam_epi": list(_support["springparam_epi"]),
        "springparam_atrial": list(_support["springparam_atrial"]),
        # LV-consistent dashpot [c_n, c_t] (canonical FCH_BASE_SUPPORT.dashpotparam);
        # negligible in the quasi-static passive sweep, set for cross-pipeline consistency.
        "dashpotparam": list(_support["dashpotparam"]),
        "active_region": list(_rid["active_region"]),
        # ACTIVE-STRAIN contraction (opt-in alternative to the additive active stress).
        # Default OFF = byte-identical to the legacy fibre-only active stress. When True,
        # MEmodel routes an incompressible fibre-shortening Fa = lf*f0(x)f0 + (1/sqrt(lf))*
        # (s0(x)s0 + n0(x)n0), lf = 1 - gamma*a(t), into the passive SEF via the growth-tensor
        # hook (F = Fe*Fa) and DISABLES the additive active stress. gamma is the max active
        # fibre shortening -- the active-strain contractility knob, REPLACING Tmax; per-chamber
        # active_strain_gamma_{lv,rv,la,ra} override the global default.
        "active_strain": False,
        "active_strain_gamma": 0.15,
        "Type": 0,
        "function_matid": False,
        "fch_fe": True,
        "valve_hyst": 0.1,
        # Fiber dataset names (non-DTI) for FCH meshes; override to match HDF5 field names.
        # Default = validated whole-heart fibers: rule-based biventricular (LV/RV/septum)
        # + RESILIENT-derived atria (LA/RA), smoothness/physiology-audited, orthonormal,
        # installed as eF_rb/eS_rb/eN_rb in fch_clregion.hdf5 (see
        # scripts/install_wholeheart_fch_fibers.py). The legacy eF_a/eS_a/eN_a atrial
        # fibers were degenerate (RA neighbour-cos ~0.70); override here to use them.
        "fiber_datasets": {"f0": "eF_rb", "s0": "eS_rb", "n0": "eN_rb"},
        # Canonical four-chamber MECHANICS mesh FILE. Decoupled from `casename`
        # (the HDF5 group, still "fch_clregion"): the FCH loader builds the file as
        # `directory_me/{mesh_basename}.hdf5` but reads every dataset from the
        # `casename` group. `fch_clregion_whole.hdf5` is the canonical whole-heart
        # file (currently byte-identical to fch_clregion.hdf5; the named target for
        # future whole-heart fiber/geometry updates), carrying eF_rb/eS_rb/eN_rb.
        # EP still loads `{casename}_refine.hdf5` (mesh_basename does not affect EP).
        "mesh_basename": "fch_clregion_whole",
        # Match lclee-style DG0 storage to reduce assembly cost.
        "fiber_storage_element": "DG0",
        # Evaluate fibers in a DG0 space instead of Quadrature to reduce assembly cost.
        "fiber_eval_element": "DG0",
        # Activation region ids (canonical FCH_BASE_REGIONS).
        "lv_rid": _rid["lv_rid"],
        "rv_rid": _rid["rv_rid"],
        # Atrial myocardial region ids. These MUST match the cavity (endo) ids:
        # LAendoid=2 bounds matid 11 and RAendoid=3 bounds matid 8, and the cavity
        # pressure/volume getters + circulation couple "LA" to LAendoid (matid 11).
        # So the LA wall is matid 11 and the RA wall is matid 8.
        "la_rid": _rid["la_rid"],
        "ra_rid": _rid["ra_rid"],
        "surface_quadrature_degree": 1,
        "enable_laplace_solver": False,
        "P_LV": 819.831289364,
        "P_RV": 929.279962238,
        # Atrial loading pressures (Pa). When fch_atria_unstressed_ed is set the atria are
        # held at the small near-unstressed pressure (atria_unstressed_pressure_mmhg, ~2 mmHg --
        # NOT 0, which loses ellipticity on the thin atrial wall) through the diastolic loading
        # ramp, so the ED reference atrial shape equals the unloaded shape recovered with
        # --atria-unstressed and the re-inflation round-trip closes; dynamic atrial pressures
        # during the closed loop still come from the 0D coupling (fch_fe). Default = legacy literals.
        "P_LA": (_unload["atria_unstressed_pressure_mmhg"] / 0.0075
                 if _unload["atria_unstressed_ed"] else 474.292782932),
        "P_RA": (_unload["atria_unstressed_pressure_mmhg"] / 0.0075
                 if _unload["atria_unstressed_ed"] else 950.226912156),
        # Zero-pressure-reference semantics switch (consumed by unloading/fch.py and the
        # calibration unload+passive-fit stage; surfaced here so every FCH consumer reads it).
        "fch_atria_unstressed_ed": bool(_unload["atria_unstressed_ed"]),
        "fch_atrial_v0_band_la": list(_unload["atrial_v0_band_la"]),
        "fch_atrial_v0_band_ra": list(_unload["atrial_v0_band_ra"]),
        # Direct per-chamber ABSOLUTE material (canonical FCH_BASE_ACTIVE.Tmax_{ch} /
        # FCH_BASE_PASSIVE.Cparam_{ch}) -- no hidden scales. The Burkhoff/Guccione forms
        # are linear in Tmax/Cparam, so each chamber's effective coefficient is the base
        # form times Tmax_{ch}/Tmax (resp. Cparam_{ch}/Cparam) -- i.e. the absolute
        # per-chamber value (the fallback cancels; see MEmodel3 _region_passive_cparam /
        # fch_active_tmax_factor).
        "Tmax_lv": _active["Tmax_lv"],
        "Tmax_rv": _active["Tmax_rv"],
        "Tmax_la": _active["Tmax_la"],
        "Tmax_ra": _active["Tmax_ra"],
        "Cparam_lv": _passive["Cparam_lv"],
        # RV operating stiffness 40 Pa (soft thin RV free wall, more compliant than LV
        # with the unloaded reference + scale 0.076; see fch-mesh-unloaded memory). The
        # twitch SAMPLING band (FCH_PASSIVE_CPARAM.rv=325) is a separate canonical record.
        "Cparam_rv": _passive["Cparam_rv"],
        "Cparam_la": _passive["Cparam_la"],
        "Cparam_ra": _passive["Cparam_ra"],
        # Per-chamber Guccione EXPONENT overrides (bff_rv/bfx_rv/bxx_rv) are SUPPORTED by
        # MEmodel (DG0 per-region fields) but deliberately LEFT GLOBAL for the RV: the
        # correct RV compliance lever is the linear Cparam_rv (above); softening the
        # exponents removes the transverse stiffness that keeps the thin RV free wall
        # stable and the solve diverges at low pressure. Exponents stay global
        # (bff=29/bfx=13.3/bxx=26.6).
    }

    if passive_viscous is not None:
        SimDet["_passive_viscous_"] = bool(passive_viscous)
    if collapse_passive_regions is not None:
        SimDet["collapse_passive_regions"] = bool(collapse_passive_regions)
    if legacy_fsolve is not None:
        SimDet["use_legacy_fsolve"] = bool(legacy_fsolve)

    safe_IODet = _replace_constants(IODet)
    safe_SimDet = _replace_constants(SimDet)
    return copy.deepcopy(safe_IODet), copy.deepcopy(safe_SimDet)
