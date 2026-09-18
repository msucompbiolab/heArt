"""Build legacy IODet/SimDet for the surrogate-calibrated lv_baseline BVP.

This reproduces, in legacy FEniCS (dolfin 2019.1.0), the LV mechanics problem the
main heArt repo calibrated on DOLFINx. The calibrated parameter set is imported
verbatim from the selected surrogate candidate:

    heArt/.outputs/surrogate_lv_calibration/bff29_tmax130_lit_20260521T011039Z/
        selected_params.json            (closed-loop + active contract)

mirrored here at configs/surrogate_lv_calibration/selected_params.json.

Model units are Pa / mL / ms (== legacy convention), so closed-loop values pass
through unchanged. The BVP is: Guccione passive + Burkhoff ("Time-varying")
active, pressure-controlled closed-loop (3D-FE LV + lumped LA + systemic via
src/sim_protocols/circ.py), MUMPS direct Newton — the same physics heArt used,
adapted only where legacy FEniCS requires it (see docstring notes below).

Mesh: meshes/geometry.hdf5 — the *exact* surrogate mesh (4804 vtx) in legacy
dolfin format (internal group "geometry"; fibers eF/eS/eN as Quadrature deg-4;
facetboundaries endo=2, epi=1, base=4; matid {0,1}).
"""

import copy
import json
import os
from pathlib import Path
from typing import Any, Optional, Tuple

# Single authoritative parameter source (calibration/params/canonical.py).
try:
    from heArt_legacy.calibration.params import canonical as _canon
except ImportError:  # when heArt_legacy itself (not its parent) is on sys.path
    from calibration.params import canonical as _canon

REPO_DIR = Path(__file__).resolve().parents[1]
DEFAULT_PARAMS = REPO_DIR / "configs" / "surrogate_lv_calibration" / "selected_params.json"
# Free-base (free-top BC) surrogate param set (active unchanged, circulation
# re-tuned for the free base). See configs/.../PROVENANCE_freetop.txt.
DEFAULT_PARAMS_FREETOP = REPO_DIR / "configs" / "surrogate_lv_calibration" / "selected_params_freetop.json"
DEFAULT_MESH_DIR = REPO_DIR / "meshes"

# --- Facet markers in meshes/geometry.hdf5 (resolved geometrically) ---------
LV_ENDO_ID = 2   # endocardium (cavity pressure surface / volume)
EPI_ID = 1       # epicardium
BASE_ID = 4      # base plane (z ~ 0)

# --- Material / loading: values come from the canonical source (not literals) -
_LV_PASSIVE = _canon.values(_canon.LV_PASSIVE)        # Cparam/bff/bfx/bxx/Kappa
KAPPA = _LV_PASSIVE.pop("Kappa")
PASSIVE_PARAMS = _LV_PASSIVE                          # {Cparam, bff, bfx, bxx}
EDP_MMHG = _canon.value(_canon.LOADING, "EDP_mmhg")
HEARTBEAT_MS = _canon.value(_canon.LOADING, "heartbeat_ms")


def _load_selected(params_path: Path) -> dict:
    with open(params_path, "r", encoding="utf-8") as fh:
        return json.load(fh)


def _active_params(selected: dict) -> dict:
    """LV Burkhoff active material from the canonical source (FE-isovolumic-feasible
    contractility/timing). `selected` (the surrogate output) is NOT the active-material
    source — the FE material is set once in calibration/params/canonical.py."""
    a = _canon.values(_canon.LV_ACTIVE)
    return {
        "Tmax": float(a["Tmax"]),
        "B": float(a["B"]),
        "t0": float(a["t0"]),
        "t_trans": float(a["t_trans"]),
        "tau": float(a["tau"]),
        "l0": float(a["l0"]),
        "lr": float(a["lr"]),
        "Ca0": float(a["Ca0"]),
        "Ca0max": float(a["Ca0max"]),
    }, float(_canon.value(_canon.ACTIVE_PROTOCOL, "activation_time_ms"))


def build_lv_baseline_config(
    case_id: str = "lv_baseline_surrogate",
    params_path: Optional[os.PathLike] = None,
    mesh_dir: Optional[os.PathLike] = None,
    output_root: Optional[str] = None,
    stop_iter: int = 2,
    dt_ms: float = 1.0,
    nload_steps: int = 64,
    softstart_beats: float = 1.0,
    free_top: bool = False,
    viscous_eta: float = 0.0,
    dt_diastasis: float = 0.0,
    dt_switch_ms: Optional[float] = None,
    dt_systole: Optional[float] = None,
    dt_relax: Optional[float] = None,
    dt_filling: Optional[float] = None,
    dt_relax_n_tau: float = 4.0,
    anchor_preload_to_surrogate_edv: bool = True,
    preload_target_volume_ml: Optional[float] = None,
    tmax_override: Optional[float] = None,
    passive_overrides: Optional[dict] = None,
) -> Tuple[dict, dict]:
    if params_path:
        params_path = Path(params_path)
    elif free_top:
        params_path = DEFAULT_PARAMS_FREETOP
    else:
        params_path = DEFAULT_PARAMS
    mesh_dir = str(Path(mesh_dir) if mesh_dir else DEFAULT_MESH_DIR) + "/"
    selected = _load_selected(params_path)

    # Closed-loop params pass through (model units already Pa/mL/ms == legacy).
    clp = copy.deepcopy(selected["closedloopparam"])
    clp["stop_iter"] = int(stop_iter)
    clp.setdefault("valve_sigmoid_center", 0.0)
    clp.setdefault("valve_sigmoid_slope", 0.1)

    # Surrogate->FE reconciliation: anchor the FE preload to the surrogate
    # END-DIASTOLIC VOLUME (= the surrogate operating V_LV) rather than to EDP.
    # The FE passive law is more compliant than the surrogate's reduced fit, so
    # pressure-ramping to EDP over-fills the cavity and lowers EF; loading to the
    # surrogate EDV reproduces the surrogate EDV/EF. Explicit value wins; else use
    # the surrogate operating V_LV when anchoring is enabled.
    if preload_target_volume_ml is None and anchor_preload_to_surrogate_edv:
        try:
            preload_target_volume_ml = float(clp["V_LV"])
        except (KeyError, TypeError, ValueError):
            preload_target_volume_ml = None

    active, homog_act_ms = _active_params(selected)
    # Contractility override (peak active tension Pa). Default None -> canonical
    # LV_ACTIVE Tmax. Used to sweep contractility (e.g. against the late-systolic
    # over-ejection limit point, where lower Tmax raises ESV out of the fold).
    if tmax_override is not None:
        active["Tmax"] = float(tmax_override)
    # Contractility soft-start: ramp Tmax 0->full over the first `softstart_beats`
    # beats to clear the high-EDV active-onset Jacobian cliff (the cold-start of
    # full Tmax at peak EDV otherwise stalls the pressure root-find). 0 disables.
    active["tmax_softstart_beats"] = float(softstart_beats)

    # LV passive material: PASSIVE_PARAMS (+ KAPPA) is the single authoritative
    # source for Cparam/bff/bfx/bxx/Kappa.
    passive_params = {k: float(v) for k, v in PASSIVE_PARAMS.items()}
    _kappa = KAPPA
    # Per-case passive material (e.g. an unloading run's Klotz-fitted Cparam and the Guccione
    # exponents it solved at). Canonical stays the DEFAULT: only keys explicitly supplied are
    # replaced, so omitting the override reproduces the previous behaviour exactly. Applied here,
    # after the canonical read and before GuccioneParams is assembled, so the viscous `eta` added
    # below is unaffected.
    if passive_overrides:
        for _key, _value in passive_overrides.items():
            if _value is None:
                continue
            if _key == "Kappa":
                _kappa = float(_value)
            elif _key in passive_params:
                passive_params[_key] = float(_value)
            else:
                raise ValueError(
                    "unknown passive override %r; expected one of %s or Kappa"
                    % (_key, sorted(passive_params))
                )
    # Viscous (Kelvin-Voigt) regularization: S_vis = 2*eta*(E-E_old)/dt adds
    # ~2*eta/dt to the tangent, removing the zero pivot at the late-systolic
    # limit point (the "short value" to avoid the singular tangent). 0 = off.
    if viscous_eta and viscous_eta > 0.0:
        passive_params["eta"] = float(viscous_eta)

    GuccioneParams = {
        "ParamsSpecified": True,
        "Passive model": {"Name": "Guccione"},
        "Passive params": passive_params,
        "Active model": {"Name": "Time-varying"},   # -> BurkhoffTimevarying3
        "Active params": active,
        "HomogenousActivation": True,               # matches heArt lv_baseline
        "homogeneous_activation_time": homog_act_ms,
        "deg": 4,
        "Kappa": _kappa,
        "incompressible": True,
    }

    output_root = output_root or os.environ.get("OUTPUT_BASE") or os.environ.get(
        "OUTPUT_ROOT", str(REPO_DIR / "outputs_lv_baseline")
    )

    IODet = {
        "casename": "geometry",          # internal HDF5 group name
        "directory_me": mesh_dir,
        "directory_ep": mesh_dir,
        "outputfolder": output_root,
        "folderName": "",
        "caseID": str(case_id),
        "matid_dataset": "matid",
    }

    SimDet = {
        "HeartBeatLength": HEARTBEAT_MS,
        "dt": float(dt_ms),
        "writeStep": 50.0,
        "GiccioneParams": GuccioneParams,
        "homogeneous_activation_time": homog_act_ms,
        "nLoadSteps": int(nload_steps),
        "EDP": EDP_MMHG,
        # Volume-anchored preload target (surrogate EDV); None falls back to EDP.
        "preload_target_volume_ml": preload_target_volume_ml,
        "DTI_EP": False,
        "DTI_ME": False,
        "d_iso": 1.5 * 0.005,
        "d_ani_factor": 4.0,
        "mesh_scale_waorta": 1.0,         # mesh already in cm/mL units
        "ploc": [[1.4, 1.4, -3.0, 2.0, 1]],
        "pacing_timing": [[4.0, 20.0]],
        "closedloopparam": clp,
        "Mechanics Discretization": "P1P1",
        "Technique Discretization": 1,
        # --- pure LV (no aorta) ---
        "isLV": True,
        "iswaorta": False,
        "isFCH": False,
        "isBiV": False,
        "aorta_pres": False,
        "ispctrl": True,
        # --- facet markers (meshes/geometry.hdf5) ---
        "matid_dataset": "matid",
        "facetboundaries_dataset": "facetboundaries",
        "LVendoid": LV_ENDO_ID,
        "RVendoid": 0,          # no RV in a pure-LV model (MEmodel.Problem reads it)
        "epiid": EPI_ID,
        "topid": BASE_ID,
        "active_region": [0, 1],
        # --- fibers stored as VectorElement(Quadrature, tet, 4) -> must match ---
        "fiber_storage_element": "Quadrature:4",
        # --- base/epi support: physiological "pericardial sliding" model. The
        # EPICARDIUM carries an absolute spring springparam=[k_n,k_t]=[8000,4000]
        # Pa/mm (springparam is the effective stiffness). The BASE is freed from the
        # main epi spring (spring_facetids_lv excludes topid) and instead gets a
        # TANGENTIAL-ONLY basal spring spring_basal=[k_n_base,k_t_base]=[0,2000]
        # Pa/mm at topid: this suppresses in-plane basal eversion/buckling while
        # leaving the NORMAL (long-axis) basal DOF free, preserving longitudinal
        # shortening / MAPSE (a normal basal spring would damp GLS -- avoid).
        # spring_basal is absolute. free_top=True instead disables the spring AND the
        # base Dirichlet (rigid-body-multiplier free base variant). The passive
        # PV / EDPVR is set by Cparam (PASSIVE_PARAMS) against this support.
        "springbc": 0 if free_top else 1,
        "free_top": bool(free_top),
        # Support stiffness values come from the canonical source. spring_facetids_lv
        # is the geometry's epi facet marker (mesh-specific, not a calibration param).
        "springparam": list(_canon.value(_canon.LV_SUPPORT, "springparam")),
        "dashpotparam": list(_canon.value(_canon.LV_SUPPORT, "dashpotparam")),
        "spring_facetids_lv": [EPI_ID],
        "spring_basal": list(_canon.value(_canon.LV_SUPPORT, "spring_basal")),
        # Unloaded-referenced spring so it stays active at the ED operating point;
        # otherwise the ED-referenced default relaxes at ED and the diastolic
        # operating P_LV collapses (~3.5 vs ~13.7 mmHg). Keeps loading +
        # operating-point stiffness consistent.
        "spring_unloaded_reference": bool(_canon.value(_canon.LV_SUPPORT, "spring_unloaded_reference")),
        # Looser tolerances for tractable full-cycle runtime (volumes still
        # accurate to ~0.1%); tighten for production accuracy.
        "abs_tol": 1e-6,
        "rel_tol": 1e-7,
        "solver_xtol_base": 1e-3,
        "solver_maxfev": 200,
        # Phase-adaptive dt: larger steps in diastasis (cycle-local t >=
        # dt_switch_ms, after the active twitch relaxes). 0 disables.
        "dt_diastasis": float(dt_diastasis),
        "dt_switch_ms": (float(dt_switch_ms) if dt_switch_ms is not None
                         else 0.55 * HEARTBEAT_MS),
        # Phase-BASED dt policy (run_waorta): keyed on the active-twitch timing
        # (t0/t_trans/tau + activation onset). Fine dt in contraction+ejection
        # (the limit-point zone -- also boosts the viscous tangent 2*eta/dt there),
        # moderate in relaxation, large in the active-free filling. Any of these
        # set -> phase policy active (supersedes the dt_diastasis single-switch);
        # all unset -> legacy behavior. Defaults: systole/relax = base dt,
        # filling = dt_diastasis.
        **({"dt_systole": float(dt_systole)} if dt_systole is not None else {}),
        **({"dt_relax": float(dt_relax)} if dt_relax is not None else {}),
        **({"dt_filling": float(dt_filling)} if dt_filling is not None
           else ({"dt_filling": float(dt_diastasis)} if dt_diastasis else {})),
        "dt_relax_n_tau": float(dt_relax_n_tau),
        # Viscous regularization toggle (eta is in Passive params above).
        "_passive_viscous_": bool(viscous_eta and viscous_eta > 0.0),
        # Allow the dt-backoff to drop further when traversing a limit point.
        "backoff_min_factor": 0.05 if free_top else 0.25,
        "Type": 0,
        # ejection-transient tolerances (heArt demo_args)
        "ejection_transient_abs_tol": 1e-6,
        "ejection_transient_rel_tol": 1e-7,
    }
    return IODet, SimDet
