"""Shared LV spring-support + deformation/strain kernel (legacy FEniCS).

Reusable, driver-agnostic machinery for the LV pericardial-sliding spring support
and the deformation/strain probes built on top of it:

- ``apply_spring_overrides`` / ``apply_material_baseline`` — mutate ``SimDet`` so the
  spring support and myocardial substrate are identical across unloading, calibration,
  and simulation (``springparam=[k_n,k_t]`` the only effective epicardial stiffness).
- ``build_model`` — assemble the lv_baseline LV ``MEmodel`` with those overrides
  (isolated-twitch flavour: contractility soft-start off), returning ``(ME, solver,
  SimDet, af, act)``.
- strain/twist probes on the reference geometry: ``_cylindrical_basis`` (analytic
  eCC/eRR/eLL, no vtk_py3), ``_surface_disp`` (facet displacement), ``_mean_natural_strain``
  (volume-averaged natural strain), ``twist_setup`` / ``twist_degrees`` (LV torsion),
  and ``active_metrics`` (ED-referenced active deformation row).

This is the single home for these helpers; the spring-calibration drivers and the
geometry-unloading / motion-metrics / strain consumers all import them from here.
"""

import json
import math
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
MONO = REPO.parent
for p in (REPO, MONO):
    if str(p) not in sys.path:
        sys.path.append(str(p))

import numpy as np
import dolfin as df

from heArt_legacy.orchestrate.lv_baseline_config import build_lv_baseline_config
from heArt_legacy.src.mechanics.MEmodel3 import MEmodel
from heArt_legacy.src.utils.oops_objects_MRC2 import State_Variables

MMHG_PER_PA = 0.0075
PA_PER_MMHG = 1.0 / MMHG_PER_PA


def _rank0():
    return df.MPI.rank(df.MPI.comm_world) == 0


def apply_spring_overrides(SimDet, k_pair, base_mode="free", top_fraction=0.25,
                           c_normal=0.0, c_tangential=0.0, disc=None,
                           basal_spring=None):
    """Mutate SimDet so springparam=[k_n,k_t] is the ONLY effective epicardial spring
    stiffness (scales neutralized) and the base follows `base_mode` (free|fraction|full).
    Shared by the passive and active calibration drivers AND the unloading driver, so
    the spring support is identical across unloading, calibration, and simulation.

    `basal_spring=[k_n_base, k_t_base]` adds a light, ABSOLUTE basal spring at `topid`
    (only in base-free modes). The canonical default is TANGENTIAL-ONLY (`[0, k_t]`,
    `calibration.params.canonical.LV_SUPPORT["spring_basal"]`, k_n range pinned `[0,0]`):
    the base normal ~ the long axis, so k_n=0 deliberately leaves the LONG AXIS FREE — the
    base can descend (physiological mitral-annular descent / MAPSE) rather than being pinned
    axially — while k_t resists in-plane base motion. The rigid longitudinal mode is carried
    by the EPICARDIAL spring's curved-surface normals (apex normal ~ axial). CAVEAT: if the
    epi spring is ALSO absent (k_n_epi=0), nothing constrains the long-axis rigid mode (an
    ill-posed Z null mode that only "converges" via load symmetry); for a well-posed
    spring-free reference use the unloader's `support.rigid_body_only` (rigid-body removal)
    instead. A normal basal component (k_n_base>0) is available if basal eversion ever stalls
    a solve, but it is NOT the default (it would fight the physiological base descent)."""
    if disc:
        SimDet["Mechanics Discretization"] = disc
    SimDet["springbc"] = 1
    SimDet["free_top"] = False
    SimDet["springparam"] = [float(k_pair[0]), float(k_pair[1])]
    SimDet["dashpotparam"] = [float(c_normal), float(c_tangential)]
    SimDet["spring_unloaded_reference"] = True
    SimDet["spring_atbase"] = False
    if basal_spring is not None:
        SimDet["spring_basal"] = [float(basal_spring[0]), float(basal_spring[1])]
    else:
        SimDet.pop("spring_basal", None)
    epiid, apxid, topid = SimDet.get("epiid"), SimDet.get("apxid"), SimDet.get("topid")
    if base_mode == "free":
        SimDet["spring_facetids_lv"] = [epiid, apxid]
        SimDet.pop("spring_top_fraction", None)
    elif base_mode == "fraction":
        SimDet["spring_facetids_lv"] = [epiid, apxid]
        SimDet["spring_top_fraction"] = float(top_fraction)
    else:  # full (legacy: epi + apex + base share the spring)
        SimDet["spring_facetids_lv"] = [epiid, apxid, topid]
        SimDet.pop("spring_top_fraction", None)
    return SimDet


def apply_material_baseline(SimDet, material_json=None, cparam=None, bff=None,
                            bfx=None, bxx=None, kappa=None, tmax=None):
    """Set a FIXED, literature-grounded myocardial substrate so the spring study
    tunes only external support, not tissue behavior. A `material_json` preset
    is applied first; explicit kwargs then override. Anything left None keeps the
    lv_baseline_config value."""
    pp = SimDet["GiccioneParams"]["Passive params"]
    doc = {}
    if material_json:
        doc = json.loads(Path(material_json).read_text())
    def pick(key, override):
        return override if override is not None else doc.get(key)
    for key, ov in (("Cparam", cparam), ("bff", bff), ("bfx", bfx), ("bxx", bxx)):
        val = pick(key, ov)
        if val is not None:
            pp[key] = float(val)
    kp = pick("Kappa", kappa)
    if kp is not None:
        SimDet["GiccioneParams"]["Kappa"] = float(kp)
    tm = pick("Tmax", tmax)
    if tm is not None:
        SimDet["GiccioneParams"]["Active params"]["Tmax"] = float(tm)
    return SimDet


def build_model(args, k_pair):
    """lv_baseline LV MEmodel with pure-[k_n,k_t] spring + active soft-start off.

    Returns (ME, solver, SimDet, af, act). `args` supplies mesh_dir, base_mode,
    top_fraction, disc, basal_spring, and the material overrides (material_json,
    cparam, bff, bfx, bxx, kappa, tmax)."""
    df.set_log_level(df.LogLevel.ERROR)
    df.parameters["form_compiler"]["representation"] = "uflacs"
    df.parameters["form_compiler"]["quadrature_degree"] = 4

    IODet, SimDet = build_lv_baseline_config(mesh_dir=args.mesh_dir)
    apply_spring_overrides(SimDet, k_pair, base_mode=args.base_mode,
                           top_fraction=args.top_fraction, disc=args.disc,
                           basal_spring=getattr(args, "basal_spring", None))
    apply_material_baseline(SimDet, material_json=args.material_json,
                            cparam=args.cparam, bff=args.bff, bfx=args.bfx,
                            bxx=args.bxx, kappa=args.kappa, tmax=args.tmax)
    # Isolated-twitch calibration: drop the closed-loop contractility soft-start
    # (matches fe_active_twitch_legacy / the DOLFINx active benchmark).
    SimDet["GiccioneParams"]["Active params"]["tmax_softstart_beats"] = 0.0

    state = State_Variables(df.MPI.comm_world, SimDet)
    state.dt.dt = SimDet["dt"]
    mesh_me = df.Mesh()
    mesh_me_params = {
        "directory": IODet["directory_me"], "casename": IODet["casename"],
        "fibre_quad_degree": 4, "outputfolder": IODet["outputfolder"],
        "foldername": "spring_active_calib/", "state_obj": state,
        "common_communicator": mesh_me.mpi_comm(), "MEmesh": mesh_me,
        "matid_dataset": IODet["matid_dataset"],
    }
    ME = MEmodel(mesh_me_params, SimDet)
    solver = ME.Solver()
    ME.LVCavityvol.assign(ME.get_lv_volume())
    af = ME.activeforms
    act = SimDet["GiccioneParams"]["Active params"]
    return ME, solver, SimDet, af, act


def _surface_disp(ME, u, N, facet_id, length_factor):
    """Area-averaged displacement on a facet set: (rms, normal_mean, tang_rms) in mm."""
    if facet_id is None:
        return float("nan"), float("nan"), float("nan")
    ds = ME.ds_me(int(facet_id), domain=ME.mesh_me, subdomain_data=ME.facetboundaries_me)
    area = df.assemble(df.Constant(1.0) * ds)
    if area <= 0.0:
        return float("nan"), float("nan"), float("nan")
    un = df.inner(u, N)
    u_t = u - un * N
    rms = (df.assemble(df.inner(u, u) * ds) / area) ** 0.5
    normal_mean = df.assemble(un * ds) / area
    tang_rms = (max(df.assemble(df.inner(u_t, u_t) * ds) / area, 0.0)) ** 0.5
    return rms * length_factor, normal_mean * length_factor, tang_rms * length_factor


def _stretch_along(ME, basis, F_ref=None):
    """UFL stretch lambda_e = sqrt(e . (F^T F) . e) along `basis`, F referred to `F_ref`."""
    F = ME.get_deformation_gradient()
    if F_ref is not None:
        F = F * df.inv(F_ref)
    C = F.T * F
    return df.sqrt(df.inner(C * basis, basis))


def push_forward(basis, F_ref):
    """The material direction `basis` CONVECTED to the `F_ref` configuration, renormalized.

    The strain readouts above evaluate along `basis` as given, i.e. along the direction as it sits
    in the REFERENCE (unloaded) geometry, while referring the deformation to `F_ref`. When `F_ref`
    is the ED state that mixes two configurations: the strain is ED-referenced but the direction it
    is measured along is not. For an ED-referenced systolic strain the material line that was
    circumferential at ED is ``F_ED . e`` (renormalized), not ``e``.

    The offset this removes is systematic and shared by all three components, so it largely cancels
    in a DIFFERENCE between two contractility levels -- which is why the Dang reproduction
    (10.1152/ajpheart.00961.2003) is unaffected and deliberately keeps the un-pushed convention for
    comparability. It does NOT cancel in an absolute per-zone strain compared against a measurement,
    which is what an inverse fit does, so that path pushes the basis forward.

    `F_ref=None` returns `basis` unchanged (the unloaded reference IS the measurement frame).
    """
    if F_ref is None:
        return basis
    v = F_ref * basis
    return v / df.sqrt(df.inner(v, v))


def _mean_natural_strain(ME, basis, vol, F_ref=None, dx=None):
    """Volume-averaged natural strain along `basis`.

    Mirrors activeforms_MRC2.CalculateFiberNaturalStrain: F = F_live * inv(F_ref);
    E = 0.5*(1 - 1/C_e), C_e = e . (F^T F) . e. With `F_ref=None` the reference is
    the unloaded mesh (F=F_live) — passive, stretch>0 / thinning<0. Pass the ED
    deformation gradient as `F_ref` for active (ED-referenced) strain — then active
    shortening<0 and wall thickening>0.

    `dx` restricts the average to a subdomain measure (e.g. ``ME.dx_me(region_id)``); `vol`
    must then be that subdomain's volume. Default `None` = the whole domain, unchanged.
    """
    dx = ME.dx_me if dx is None else dx
    C_e = _stretch_along(ME, basis, F_ref) ** 2
    E_e = 0.5 * (1.0 - 1.0 / C_e)
    return df.assemble(E_e * dx) / vol


def _mean_green_lagrange(ME, basis, vol, F_ref=None, dx=None):
    """Volume-averaged GREEN-LAGRANGE strain along `basis`: E = 0.5*(C_e - 1)."""
    dx = ME.dx_me if dx is None else dx
    C_e = _stretch_along(ME, basis, F_ref) ** 2
    return df.assemble(0.5 * (C_e - 1.0) * dx) / vol


def _mean_engineering_strain(ME, basis, vol, F_ref=None, dx=None):
    """Volume-averaged ENGINEERING strain along `basis`: lambda_e - 1.

    This is literally "the fractional change of wall thickness" when `basis` is the radial
    direction, i.e. the definition Dang et al. 2005 (10.1152/ajpheart.00961.2003) state for
    their radial strain RS. The three measures agree to <1e-4 inside their akinesis window
    (|RS| <= 0.01) but differ by several percent at their dyskinetic/hypokinetic magnitudes
    (-0.038, +0.079), so a comparison against that paper reports all three rather than
    assuming one.
    """
    dx = ME.dx_me if dx is None else dx
    return df.assemble((_stretch_along(ME, basis, F_ref) - 1.0) * dx) / vol


def cellwise_engineering_strain(ME, basis, F_ref=None):
    """Per-CELL engineering strain along `basis`, as a (n_cells,) array aligned with `matid`.

    The regional means above collapse a zone to one number; this keeps the element-level
    distribution, which is what a position-resolved comparison needs -- Dang et al. 2005 report
    a spread per infarct position (their Fig. 4), and a mean alone cannot be compared against a
    claim about VARIABILITY. DG0 is the natural space: one dof per cell, and its mass matrix is
    diagonal, so the projection is cheap.

    Returns ``(strain, cell_volume, matid)``, every array indexed by the LOCAL cell index, so a
    caller can form any volume-weighted per-region statistic it wants.
    """
    mesh = ME.mesh_me
    DG0 = df.FunctionSpace(mesh, "DG", 0)
    eps = df.project(_stretch_along(ME, basis, F_ref) - 1.0, DG0)
    dofmap = DG0.dofmap()
    n = mesh.num_cells()
    order = np.empty(n, dtype=np.int64)
    for c in range(n):
        order[c] = dofmap.cell_dofs(c)[0]
    strain = eps.vector().get_local()[order]
    cell_vol = np.array([df.Cell(mesh, c).volume() for c in range(n)], dtype=float)
    matid = ME.matid_me.array().astype(np.int64) if hasattr(ME, "matid_me") else None
    return strain, cell_vol, matid


def region_volume(ME, region_id):
    """Reference volume of one `matid` region (the denominator for a regional average)."""
    return df.assemble(df.Constant(1.0) * ME.dx_me(int(region_id)))


def regional_strain(ME, basis, region_ids, F_ref=None):
    """Per-`matid`-region volume-averaged strain along `basis`, in all three measures.

    Returns ``{region_id: {"natural":, "green_lagrange":, "engineering":, "volume":}}``. A
    region whose reference volume is ~0 (an id present in the config but absent from the mesh)
    yields None rather than a divide-by-zero — an absent region is a fact to report, not a NaN
    to propagate.
    """
    out = {}
    for rid in region_ids:
        rid = int(rid)
        dx_r = ME.dx_me(rid)
        vol_r = region_volume(ME, rid)
        if vol_r <= 1.0e-12:
            out[rid] = None
            continue
        out[rid] = {
            "volume": float(vol_r),
            "natural": float(_mean_natural_strain(ME, basis, vol_r, F_ref=F_ref, dx=dx_r)),
            "green_lagrange": float(_mean_green_lagrange(ME, basis, vol_r, F_ref=F_ref, dx=dx_r)),
            "engineering": float(_mean_engineering_strain(ME, basis, vol_r, F_ref=F_ref, dx=dx_r)),
        }
    return out


def _cylindrical_basis(ME):
    """Analytic circumferential/radial/longitudinal basis (eCC, eRR, eLL) on the
    reference (unloaded) geometry — dependency-free (no vtk_py3 / fiber LDRB).

    Long (apicobasal) axis = unit vector from the myocardial centroid toward the
    base-plane (topid) centroid; radial = position component perpendicular to that
    axis; circumferential = axis x radial. UFL expressions over the reference mesh,
    so they pair with the unloaded-reference natural strain above.
    """
    mesh = ME.mesh_me
    x = df.SpatialCoordinate(mesh)
    vol = df.assemble(df.Constant(1.0) * ME.dx_me)
    centroid = [df.assemble(x[i] * ME.dx_me) / vol for i in range(3)]
    topid = int(ME.SimDet.get("topid"))
    ds_top = ME.ds_me(topid, domain=mesh, subdomain_data=ME.facetboundaries_me)
    area = df.assemble(df.Constant(1.0) * ds_top)
    base = [df.assemble(x[i] * ds_top) / area for i in range(3)]
    axis = [base[i] - centroid[i] for i in range(3)]
    nrm = sum(a * a for a in axis) ** 0.5
    axis = [a / nrm for a in axis]
    k = df.Constant(tuple(axis))
    p0 = df.Constant(tuple(base))
    d = x - p0
    r_vec = d - df.dot(d, k) * k
    eRR = r_vec / df.sqrt(df.dot(r_vec, r_vec) + 1e-12)
    eLL = k
    c_vec = df.cross(k, eRR)
    eCC = c_vec / df.sqrt(df.dot(c_vec, c_vec) + 1e-12)
    return eCC, eRR, eLL


def twist_setup(ME):
    """Reference-geometry pieces for LV twist: long axis k (centroid->base), the
    circumferential direction eCC=k x r, the off-axis radius r, and the axial
    extent [s_min,s_max]. Used to integrate apical vs basal rotation."""
    mesh = ME.mesh_me
    x = df.SpatialCoordinate(mesh)
    vol = df.assemble(df.Constant(1.0) * ME.dx_me)
    centroid = [df.assemble(x[i] * ME.dx_me) / vol for i in range(3)]
    topid = int(ME.SimDet.get("topid"))
    ds_top = ME.ds_me(topid, domain=mesh, subdomain_data=ME.facetboundaries_me)
    area = df.assemble(df.Constant(1.0) * ds_top)
    base = [df.assemble(x[i] * ds_top) / area for i in range(3)]
    axis = [base[i] - centroid[i] for i in range(3)]
    nrm = sum(a * a for a in axis) ** 0.5
    axis = [a / nrm for a in axis]
    k = df.Constant(tuple(axis))
    p0 = df.Constant(tuple(base))
    d = x - p0
    r_vec = d - df.dot(d, k) * k
    r_expr = df.sqrt(df.dot(r_vec, r_vec) + 1e-12)
    c_vec = df.cross(k, r_vec)
    eCC = c_vec / df.sqrt(df.dot(c_vec, c_vec) + 1e-12)
    coords = mesh.coordinates()
    s = (coords - np.array(base)).dot(np.array(axis))
    return {"k": k, "p0": p0, "eCC": eCC, "r": r_expr,
            "s_min": float(s.min()), "s_max": float(s.max())}


def twist_degrees(ME, u_rel, st):
    """LV twist magnitude (deg) = |apical rotation - basal rotation| about the long
    axis. Local rotation angle ~ (u . eCC)/r, volume-averaged over the apical and
    basal axial thirds. Proves the support still permits physiological torsion."""
    x = df.SpatialCoordinate(ME.mesh_me)
    s = df.dot(x - st["p0"], st["k"])
    rng = (st["s_max"] - st["s_min"]) or 1.0
    apex_mask = df.conditional(df.lt(s, st["s_min"] + 0.33 * rng), 1.0, 0.0)
    base_mask = df.conditional(df.gt(s, st["s_max"] - 0.33 * rng), 1.0, 0.0)
    theta = df.dot(u_rel, st["eCC"]) / st["r"]
    dx = ME.dx_me

    def avg(mask):
        den = df.assemble(mask * dx)
        return df.assemble(theta * mask * dx) / den if den > 1e-12 else 0.0

    return abs(math.degrees(avg(apex_mask) - avg(base_mask)))


def active_metrics(ME, basis, vol, N, lf, F_ED, u_ED, k_pair, base_mode,
                   t_ms, act_level, P_ED, converged, twist_st):
    eCC, eRR, eLL = basis
    u_rel = ME.get_displacement() - u_ED          # displacement relative to ED
    # Decompose epicardial motion: NORMAL (radial) is the pericardial constraint
    # that matters; TANGENTIAL (longitudinal/circumferential sliding) is
    # physiologically expected (the heart slides+twists within the sac), so the
    # high-k_n / low-k_t "pericardial sliding" support should drive epi NORMAL down
    # while leaving epi tangential (and thus GLS/twist) large.
    epi_rms, epi_n, epi_t = _surface_disp(ME, u_rel, N, ME.SimDet.get("epiid"), lf)
    _, base_n, _ = _surface_disp(ME, u_rel, N, ME.SimDet.get("topid"), lf)
    apx_rms, _, _ = _surface_disp(ME, u_rel, N, ME.SimDet.get("apxid"), lf)
    P_total = ME.get_lv_pressure() * MMHG_PER_PA
    return {
        "k_normal": k_pair[0], "k_tangential": k_pair[1], "base_mode": base_mode,
        "phase": "active", "t_ms": t_ms, "activation_level": act_level,
        "pressure_total_mmhg": P_total, "dev_pressure_mmhg": P_total - P_ED,
        "volume_ml": ME.get_lv_volume(),
        "Ell_active": _mean_natural_strain(ME, eLL, vol, F_ref=F_ED),
        "Ecc_active": _mean_natural_strain(ME, eCC, vol, F_ref=F_ED),
        "Err_active": _mean_natural_strain(ME, eRR, vol, F_ref=F_ED),
        "base_disp_normal_mm": base_n, "base_descent_active_mm": abs(base_n),
        "apex_disp_mm": apx_rms, "epi_disp_active_mm": epi_rms,
        "epi_disp_normal_mm": abs(epi_n), "epi_disp_tang_mm": epi_t,
        "twist_deg": twist_degrees(ME, u_rel, twist_st),
        "converged": bool(converged),
    }
