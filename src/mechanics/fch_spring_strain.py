"""Four-chamber spring-support + deformation/strain kernel (legacy FEniCS).

The FCH analogue of ``lv_spring_strain.py``. It extends the LV pericardial-sliding
spring methodology to the full four-chamber (FCH) geometry: a per-region Robin
support (ventricular epicardium / atrial wall / aorta) plus FCH-aware kinematic
probes used to score that support against literature acceptance bands.

Two halves:

1. **Support overrides** — ``apply_fch_spring_overrides`` mutates ``SimDet`` to set
   the per-region spring keys consumed by ``spring_bc_forms._build_fch_form``
   (``springparam_epi`` / ``springparam_atrial`` / ``springparam_aorta``; absent =>
   the single ``springparam``, byte-unchanged). ``build_model_fch_spring`` assembles
   the four-chamber ``MEmodel`` (pressure-controlled, FCH-native) with those
   overrides, mirroring ``unloading/fch.py:build_model_fch``.

2. **FCH kinematic probes** — generalize the LV-centric (single-axis) probes to the
   four chambers. Each chamber's apicobasal axis is derived analytically from its
   ENDOCARDIAL surface (principal axis of the surface second-moment tensor — no
   vtk_py3/fiber LDRB), giving per-chamber natural-strain bases. On top:
     - per-ventricle GLS / GCS / GRS (``chamber_strains`` — region-averaged natural
       strain over the chamber matid, ED-referenced for active),
     - per-ventricle MAPSE / TAPSE (``annular_descent`` — longitudinal AV-plane
       descent proxy, orientation-invariant),
     - per-ventricle twist (``chamber_twist`` — apical-minus-basal rotation),
     - atrial reservoir excursion (``surface_excursion`` on the atrial endo),
     - whole-heart rigid-body drift (``rigid_body_drift`` — L2 projection of the
       motion field onto the 6 rigid modes; quantifies BOTH under-support drift
       and over-constraint).

Length-unit note: displacement metrics are multiplied by ``mm_per_mesh_unit`` to
report in mm; natural strains and twist are dimensionless / scale-invariant.

This module is the single home for the FCH probes; the ``fch_spring_tune`` drivers
import from here. It reuses ``lv_spring_strain`` helpers (``_surface_disp``,
``MMHG_PER_PA``, ``_rank0``) where they apply unchanged.
"""

import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
MONO = REPO.parent
for _p in (REPO, MONO):
    if str(_p) not in sys.path:
        sys.path.append(str(_p))

import math

import numpy as np
import dolfin as df

from heArt_legacy.orchestrate.base_config import base_config
from heArt_legacy.src.mechanics.MEmodel3 import MEmodel
from heArt_legacy.src.utils.oops_objects_MRC2 import State_Variables
from heArt_legacy.src.mechanics.lv_spring_strain import (
    MMHG_PER_PA, PA_PER_MMHG, _rank0, _surface_disp,
)

# Chamber wall material ids (matid1) and endocardial facet ids in the canonical
# FCH mesh (orchestrate/base_config + the fch_clregion mesh). LV/RV are the
# ventricles (MAPSE/TAPSE/GLS/twist); LA/RA are the atria (reservoir excursion).
FCH_MATID = {"LV": 10, "RV": 9, "LA": 11, "RA": 8}
FCH_ENDOID = {"LV": 1, "RV": 6, "LA": 2, "RA": 3}
VENTRICLES = ("LV", "RV")
ATRIA = ("LA", "RA")


# --------------------------------------------------------------------------- #
# Support overrides + model build
# --------------------------------------------------------------------------- #
def apply_fch_spring_overrides(SimDet, epi=None, atrial=None, aorta=None,
                               base=None, dashpot=None):
    """Set the per-region FCH spring stiffness on ``SimDet`` (in place).

    ``epi``/``atrial``/``aorta`` are ``[k_n, k_t]`` pairs for the ventricular
    epicardium / atrial wall / aorta surface respectively; each defaults to the
    single ``base`` (or the existing ``springparam``) when None, so a call with all
    Nones leaves the assembled form byte-identical to the default FCH support.
    ``spring_bc_forms._resolve_fch_region_kpairs`` performs the actual fallback.
    """
    SimDet["springbc"] = 1
    if base is not None:
        SimDet["springparam"] = [float(base[0]), float(base[1])]
    if dashpot is not None:
        SimDet["dashpotparam"] = [float(dashpot[0]), float(dashpot[1])]
    for name, val in (("epi", epi), ("atrial", atrial), ("aorta", aorta)):
        key = "springparam_%s" % name
        if val is not None:
            SimDet[key] = [float(val[0]), float(val[1])]
        else:
            SimDet.pop(key, None)
    # Clear any stale dict grouping so the explicit per-region keys are authoritative.
    SimDet.pop("springparam_by_region", None)
    return SimDet


def build_model_fch_spring(mesh_dir, output_dir, epi=None, atrial=None, aorta=None,
                           base=None, cparam=None, foldername="fch_spring_calib/"):
    """Assemble the four-chamber MEmodel (pressure control) with per-region springs.

    Mirrors ``unloading/fch.py:build_model_fch`` (same ``mesh_me_params`` shape,
    FCH-native ``ispctrl=True``), then layers the per-region spring overrides via
    ``apply_fch_spring_overrides``. Optional ``cparam={ch: value}`` sets per-chamber
    absolute passive stiffness ``Cparam_{ch}`` (else base_config defaults). Returns
    ``(ME, solver, IODet, SimDet)``. Cavity pressures start at zero (unloaded ref).
    """
    df.set_log_level(df.LogLevel.ERROR)
    df.parameters["form_compiler"]["representation"] = "uflacs"
    df.parameters["form_compiler"]["quadrature_degree"] = 4

    IODet, SimDet = base_config()
    IODet["directory_me"] = str(mesh_dir)
    IODet["directory_ep"] = str(mesh_dir)
    apply_fch_spring_overrides(SimDet, epi=epi, atrial=atrial, aorta=aorta, base=base)
    if cparam:
        for ch, val in cparam.items():
            if val is not None:
                SimDet["Cparam_%s" % ch.lower()] = float(val)

    state = State_Variables(df.MPI.comm_world, SimDet)
    state.dt.dt = SimDet["dt"]
    mesh_me = df.Mesh()
    out = str(Path(output_dir).expanduser().resolve()) + "/"
    mesh_me_params = {
        "directory": IODet["directory_me"],
        "casename": IODet["casename"],
        "fibre_quad_degree": 4,
        "outputfolder": out,
        "foldername": foldername,
        "state_obj": state,
        "common_communicator": mesh_me.mpi_comm(),
        "MEmesh": mesh_me,
        "matid_dataset": IODet["matid_dataset"],
    }
    ME = MEmodel(mesh_me_params, SimDet)
    solver = ME.Solver()
    ME.LVCavitypres.assign(0.0)
    ME.RVCavitypres.assign(0.0)
    ME.LACavitypres.assign(0.0)
    ME.RACavitypres.assign(0.0)
    return ME, solver, IODet, SimDet


def solve_nonlinear(solver):
    """Run one nonlinear solve; True on convergence, False on raised failure.

    Type-0 ``solvenonlinear`` returns None on success and RAISES on non-convergence
    (per src/utils/nsolver.py), so a clean return is success."""
    try:
        res = solver.solvenonlinear()
    except Exception:
        return False
    if isinstance(res, (tuple, list)) and len(res) >= 2:
        return bool(res[1])
    return True


def inflate_to_edp(ME, solver, edp_mmhg, nsub=8):
    """Quasi-statically inflate the four cavities 0 -> per-chamber EDP (pressure
    control), ramping all four pressures together over ``nsub`` sub-steps with a
    step-halving backoff on a failed solve. ``edp_mmhg`` = {LV,RV,LA,RA: mmHg}.

    Returns (converged: bool, frac_reached: float in [0,1]). Per the repo
    error-handling rule, a stalled ramp is surfaced (warning printed by caller via
    the returned fraction < 1)."""
    cav = {"LV": ME.LVCavitypres, "RV": ME.RVCavitypres,
           "LA": ME.LACavitypres, "RA": ME.RACavitypres}
    targets_pa = {ch: float(edp_mmhg.get(ch, 0.0)) * PA_PER_MMHG for ch in cav}

    def assign(frac):
        for ch, const in cav.items():
            const.assign(frac * targets_pa[ch])

    frac, step = 0.0, 1.0 / max(1, int(nsub))
    assign(0.0)
    while frac < 1.0 - 1e-9:
        nxt = min(1.0, frac + step)
        assign(nxt)
        if solve_nonlinear(solver):
            frac = nxt
        else:
            assign(frac)            # roll back to last good level
            step *= 0.5
            if step < 1e-3:
                return False, frac
    return True, 1.0


# --------------------------------------------------------------------------- #
# Reference-geometry per-chamber axes / bases (analytic, no vtk_py3)
# --------------------------------------------------------------------------- #
def _surface_geometry(ME, facet_id):
    """(area, centroid[3], principal_axis[3], sigma_long) for an endocardial facet
    set on the REFERENCE mesh. The principal axis = eigenvector of the largest
    eigenvalue of the area-weighted second-moment tensor M_ij = <(x_i-c_i)(x_j-c_j)>;
    for a ventricular/atrial cavity surface that is the apicobasal (long) axis.
    sigma_long = sqrt(<s^2>), s = (x-c).axis, used to define basal/apical thirds.
    """
    mesh = ME.mesh_me
    x = df.SpatialCoordinate(mesh)
    ds = ME.ds_me(int(facet_id), domain=mesh, subdomain_data=ME.facetboundaries_me)
    area = df.assemble(df.Constant(1.0) * ds)
    if area <= 0.0:
        raise ValueError("empty facet set %s (zero area)" % facet_id)
    c = np.array([df.assemble(x[i] * ds) / area for i in range(3)])
    cc = df.Constant(tuple(c))
    d = x - cc
    M = np.empty((3, 3))
    for i in range(3):
        for j in range(i, 3):
            M[i, j] = M[j, i] = df.assemble(d[i] * d[j] * ds) / area
    evals, evecs = np.linalg.eigh(M)
    axis = evecs[:, int(np.argmax(evals))]
    axis = axis / (np.linalg.norm(axis) + 1e-30)
    sigma = float(max(evals)) ** 0.5
    return area, c, axis, sigma


def chamber_geometry(ME):
    """Precompute, ONCE on the reference mesh, every per-chamber geometric quantity
    the probes need: endo-surface area/centroid/long-axis/sigma, the matid region
    volume, and the UFL natural-strain basis (eCC,eRR,eLL). Returned as a dict keyed
    by chamber name. Geometry-only, so it is computed before any solve and reused."""
    geom = {}
    mesh = ME.mesh_me
    x = df.SpatialCoordinate(mesh)
    for ch in ("LV", "RV", "LA", "RA"):
        area, c, axis, sigma = _surface_geometry(ME, FCH_ENDOID[ch])
        k = df.Constant(tuple(axis))
        p0 = df.Constant(tuple(c))
        d = x - p0
        r_vec = d - df.dot(d, k) * k
        eRR = r_vec / df.sqrt(df.dot(r_vec, r_vec) + 1e-12)
        eLL = k
        c_vec = df.cross(k, eRR)
        eCC = c_vec / df.sqrt(df.dot(c_vec, c_vec) + 1e-12)
        matid = FCH_MATID[ch]
        vol = df.assemble(df.Constant(1.0) * ME.dx_me(matid))
        geom[ch] = {
            "endoid": FCH_ENDOID[ch], "matid": matid,
            "area": area, "centroid": c, "axis": axis, "sigma": sigma,
            "k": k, "p0": p0, "eCC": eCC, "eRR": eRR, "eLL": eLL, "vol": vol,
        }
    return geom


def _region_natural_strain(ME, basis, matid, vol, F_ref=None):
    """Volume-averaged natural strain along ``basis`` over the chamber matid region.

    Same kinematics as ``lv_spring_strain._mean_natural_strain`` (E = 0.5(1-1/C_e),
    C_e = e.(F^T F).e) but restricted to ``dx_me(matid)``. ``F_ref=None`` => unloaded
    reference (filling strain); pass the ED deformation gradient for ED-referenced
    active strain (systolic GLS/GCS/GRS)."""
    F = ME.get_deformation_gradient()
    if F_ref is not None:
        F = F * df.inv(F_ref)
    C = F.T * F
    C_e = df.inner(C * basis, basis)
    E_e = 0.5 * (1.0 - 1.0 / C_e)
    return df.assemble(E_e * ME.dx_me(int(matid))) / vol


def chamber_strains(ME, g, F_ref=None):
    """(GLS, GCS, GRS) = region-averaged longitudinal/circumferential/radial natural
    strain for one chamber (geometry record ``g`` from chamber_geometry)."""
    return (
        _region_natural_strain(ME, g["eLL"], g["matid"], g["vol"], F_ref),
        _region_natural_strain(ME, g["eCC"], g["matid"], g["vol"], F_ref),
        _region_natural_strain(ME, g["eRR"], g["matid"], g["vol"], F_ref),
    )


def annular_descent(ME, u, g, length_factor, third=0.5):
    """AV-plane (MAPSE/TAPSE) longitudinal descent proxy for a ventricle, in mm.

    = | <u.k>_basal-band  -  <u.k>_apical-band | over the chamber ENDOCARDIAL surface,
    where the bands are the axial extremes (|s| > third*sigma, s=(x-p0).k). This is the
    net longitudinal shortening of the cavity (annulus moving toward the apex), the
    finite-element counterpart of mitral/tricuspid annular plane systolic excursion.

    Orientation-invariant: flipping the (sign-arbitrary) principal axis swaps the
    bands AND flips u.k, leaving the difference unchanged."""
    mesh = ME.mesh_me
    x = df.SpatialCoordinate(mesh)
    ds = ME.ds_me(int(g["endoid"]), domain=mesh, subdomain_data=ME.facetboundaries_me)
    s = df.dot(x - g["p0"], g["k"])
    thr = float(third) * float(g["sigma"])
    hi = df.conditional(df.gt(s, thr), 1.0, 0.0)
    lo = df.conditional(df.lt(s, -thr), 1.0, 0.0)
    un = df.dot(u, g["k"])
    a_hi = df.assemble(hi * ds)
    a_lo = df.assemble(lo * ds)
    if a_hi <= 0.0 or a_lo <= 0.0:
        return float("nan")
    mean_hi = df.assemble(un * hi * ds) / a_hi
    mean_lo = df.assemble(un * lo * ds) / a_lo
    return abs(mean_hi - mean_lo) * length_factor


def chamber_twist(ME, u, g):
    """Per-ventricle twist (deg) = |apical - basal rotation| about the chamber long
    axis, volume-averaged over the apical and basal axial thirds of the chamber matid
    region. Local rotation angle ~ (u.eCC)/r. Mirrors lv_spring_strain.twist_degrees,
    generalized to the chamber's own axis/centroid and restricted to its matid."""
    mesh = ME.mesh_me
    x = df.SpatialCoordinate(mesh)
    dxm = ME.dx_me(int(g["matid"]))
    s = df.dot(x - g["p0"], g["k"])
    thr = 0.43 * float(g["sigma"])
    apex = df.conditional(df.lt(s, -thr), 1.0, 0.0)
    base = df.conditional(df.gt(s, thr), 1.0, 0.0)
    d = x - g["p0"]
    r_vec = d - df.dot(d, g["k"]) * g["k"]
    r = df.sqrt(df.dot(r_vec, r_vec) + 1e-12)
    theta = df.dot(u, g["eCC"]) / r

    def avg(mask):
        den = df.assemble(mask * dxm)
        return df.assemble(theta * mask * dxm) / den if den > 1e-12 else 0.0

    return abs(math.degrees(avg(apex) - avg(base)))


def surface_excursion(ME, u, facet_id, length_factor):
    """Area-averaged RMS displacement on a facet set (mm). Reuses the LV kernel.
    Used for atrial reservoir excursion (bounded, not clamped) and epicardial
    sliding sanity."""
    rms, _, _ = _surface_disp(ME, u, df.FacetNormal(ME.mesh_me), facet_id, length_factor)
    return rms


def rigid_body_drift(ME, u, length_factor):
    """Decompose the whole-heart motion field into its rigid-body part (translation +
    rotation about the centroid) by L2 projection over the myocardial volume, and the
    residual (genuinely deformational) part.

    Returns dict:
      translation_mm  : |T|, T = <u>_vol             (mean translation)
      rotation_deg    : |omega|, J.omega = <r x u>    (mean rotation; J = inertia tensor)
      rigid_fraction  : ||u_rigid||^2 / ||u||^2 in [0,1]  (motion explained by rigid modes)

    Interpretation of the two pericardial-support failure modes:
      * UNDER-supported  -> large translation_mm / rotation_deg AND rigid_fraction -> 1
        (the heart drifts/spins as a body within the sac).
      * OVER-constrained -> twist/MAPSE/TAPSE/strains all collapse toward 0 (caught by
        the per-ventricle metrics), with tiny rigid drift here.
    """
    mesh = ME.mesh_me
    x = df.SpatialCoordinate(mesh)
    dxm = ME.dx_me
    V = df.assemble(df.Constant(1.0) * dxm)
    T = np.array([df.assemble(u[i] * dxm) / V for i in range(3)])
    c = np.array([df.assemble(x[i] * dxm) / V for i in range(3)])
    cc = df.Constant(tuple(c))
    r = x - cc
    # Inertia tensor J_ij = <|r|^2 dij - r_i r_j> * V  and moment L = <r x (u - T)> * V.
    r2 = df.dot(r, r)
    J = np.empty((3, 3))
    for i in range(3):
        for j in range(3):
            delta = 1.0 if i == j else 0.0
            J[i, j] = df.assemble((r2 * delta - r[i] * r[j]) * dxm)
    uT = u - df.Constant(tuple(T))
    rxu = df.cross(r, uT)
    L = np.array([df.assemble(rxu[i] * dxm) for i in range(3)])
    try:
        omega = np.linalg.solve(J, L)
    except np.linalg.LinAlgError:
        omega = np.zeros(3)
    # Residual (non-rigid) energy: ||u - (T + omega x r)||^2.
    Tc = df.Constant(tuple(T))
    wc = df.Constant(tuple(omega))
    u_rigid = Tc + df.cross(wc, r)
    e_total = df.assemble(df.dot(u, u) * dxm)
    e_resid = df.assemble(df.dot(u - u_rigid, u - u_rigid) * dxm)
    rigid_fraction = (1.0 - e_resid / e_total) if e_total > 1e-30 else 0.0
    return {
        "translation_mm": float(np.linalg.norm(T)) * length_factor,
        "rotation_deg": math.degrees(float(np.linalg.norm(omega))),
        "rigid_fraction": float(max(0.0, min(1.0, rigid_fraction))),
    }


def fch_passive_metrics(ME, geom, length_factor, F_ref=None):
    """Full four-chamber metric row at the current (solved) state.

    Per-ventricle (LV,RV): GLS/GCS/GRS, MAPSE/TAPSE (annular descent), twist, epi-
    sliding excursion. Per-atrium (LA,RA): endo reservoir excursion + longitudinal
    strain. Whole-heart: rigid-body drift. ``F_ref=None`` => filling (passive)
    strains; pass the ED deformation gradient for ED-referenced active strains."""
    u = ME.get_displacement()
    lf = length_factor
    row = {
        "P_LV_mmhg": ME.get_lv_pressure() * MMHG_PER_PA,
        "P_RV_mmhg": ME.get_rv_pressure() * MMHG_PER_PA,
        "P_LA_mmhg": ME.get_la_pressure() * MMHG_PER_PA,
        "P_RA_mmhg": ME.get_ra_pressure() * MMHG_PER_PA,
        "V_LV_ml": ME.get_lv_volume(), "V_RV_ml": ME.get_rv_volume(),
        "V_LA_ml": ME.get_la_volume(), "V_RA_ml": ME.get_ra_volume(),
    }
    for ch in VENTRICLES:
        g = geom[ch]
        gls, gcs, grs = chamber_strains(ME, g, F_ref=F_ref)
        row["%s_GLS" % ch] = gls
        row["%s_GCS" % ch] = gcs
        row["%s_GRS" % ch] = grs
        row["%s_AVPD_mm" % ch] = annular_descent(ME, u, g, lf)
        row["%s_twist_deg" % ch] = chamber_twist(ME, u, g)
        row["%s_epi_excursion_mm" % ch] = surface_excursion(ME, u, ME.SimDet.get("epiid"), lf)
    # MAPSE = LV AV-plane descent; TAPSE = RV AV-plane descent (clinical aliases).
    row["MAPSE_mm"] = row.get("LV_AVPD_mm")
    row["TAPSE_mm"] = row.get("RV_AVPD_mm")
    for ch in ATRIA:
        g = geom[ch]
        row["%s_excursion_mm" % ch] = surface_excursion(ME, u, g["endoid"], lf)
        gls, _, _ = chamber_strains(ME, g, F_ref=F_ref)
        row["%s_long_strain" % ch] = gls
    row.update(rigid_body_drift(ME, u, lf))
    return row
