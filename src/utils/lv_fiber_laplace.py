"""Rule-based LV fiber generation by Laplace solves — pure legacy DOLFIN, no vtk_py3.

This is a self-contained, vectorized port of the rule-based (Bayer/LDRB) LV fiber
recipe used by `external/vtk_py3/addLVfiber_LDRB.py` and by the dolfinx generator
`/srv/server_lvw/.../generate_rule_based_fibers.py`. It exists so the solver can
(re)generate LV fibers WITHOUT the fragile/slow vtk_py3 path (which needs the `vtk`
wheel, JIT-compiles C++ that wants VTK dev headers, and runs a per-DOF quaternion
bislerp loop). For an LV-only mesh the LDRB pipeline collapses to a single local
frame per quadrature point, so no quaternion bislerp is needed.

Algorithm (matching addLVfiber_LDRB conventions exactly):
  1. Transmural Laplace  phi:  phi=0 on LV endo, phi=1 on epi.        (t in [0,1])
  2. Apicobasal Laplace  psi:  psi=0 at apex,    psi=1 on base.
  3. Per quadrature point, from gA=grad(psi), gT=grad(phi):
        e1 = gA/|gA|                              (apicobasal/longitudinal)
        e2 = (gT - (gT.e1) e1)/|...|              (transmural, orthogonalized)
        e0 = e1 x e2                              (circumferential)
     (this Gram-Schmidt frame is what addLVfiber's axisf root-solve converges to.)
  4. Helix angle alpha(t) = alpha_endo*(1-t) + alpha_epi*t  (deg; +60/-60 default),
     rotate the frame by alpha in the e0-e1 plane:
        eF = cos(a) e0 + sin(a) e1   (fiber)
        eS = -sin(a) e0 + cos(a) e1  (sheet)
        eN = e2                      (sheet-normal, transmural)
     (column 0/1/2 == eF/eS/eN, matching addLVfiber_LDRB / create_EDFibers.)

The output Functions live in the *fiber storage space* passed in (the model's
Quadrature:4 vector space), so they are drop-in replacements for `MEmodel.f0_me`
etc. Projection onto Quadrature spaces requires the 'quadrature' form-compiler
representation (as addLVfiber does); this module sets/restores it locally.
"""

import numpy as np
import dolfin as df


def _normalize_rows(a, fallback=None):
    n = np.linalg.norm(a, axis=1)
    bad = n < 1e-12
    n[bad] = 1.0
    out = a / n[:, None]
    if fallback is not None:
        out[bad] = fallback
    return out, bad


def generate_lv_fibers_laplace(
    mesh,
    facetboundaries,
    *,
    lvendoid,
    epiid,
    baseid,
    fiber_space,
    deg=4,
    alpha_endo=60.0,
    alpha_epi=-60.0,
    apex_axis=2,
    apex_tol_frac=0.02,
    comm=None,
):
    """Generate (eF, eS, eN) LV fibers on `mesh` into `fiber_space`.

    Parameters
    ----------
    mesh, facetboundaries : the LV mechanics mesh and its facet MeshFunction.
    lvendoid, epiid, baseid : facet ids for LV endocardium, epicardium, base plane.
    fiber_space : a vector FunctionSpace (e.g. Quadrature:4) — the storage space
        the fibers must end up in. eF/eS/eN are returned as df.Function on it.
    deg : quadrature degree for the Laplace/projection (match fiber_space degree).
    alpha_endo, alpha_epi : subendo/subepi helix angles in degrees (+60/-60).
    apex_axis : long-axis coordinate index (default 2 = z; apex at the min end).
    apex_tol_frac : apex Dirichlet band as a fraction of the long-axis extent.

    Returns
    -------
    (eF, eS, eN) : df.Function in `fiber_space`, unit-normalized.
    """
    comm = comm or mesh.mpi_comm()

    fc = df.parameters["form_compiler"]
    prev_repr = fc["representation"]
    prev_qd = fc["quadrature_degree"]
    fc["representation"] = "quadrature"
    fc["quadrature_degree"] = deg
    try:
        V = df.FunctionSpace(mesh, df.FiniteElement("Lagrange", mesh.ufl_cell(), 2))

        # Apex band along the long axis (apex = min end), matching addLVfiber's
        # right_boundary(x): x[axis] < minz + tol.
        coords = mesh.coordinates()
        amin = df.MPI.min(comm, float(coords[:, apex_axis].min()))
        amax = df.MPI.max(comm, float(coords[:, apex_axis].max()))
        tol = apex_tol_frac * (amax - amin)

        class _Apex(df.SubDomain):
            def inside(self, x, on_boundary):
                return x[apex_axis] < amin + tol

        # --- transmural Laplace: endo=0, epi=1 ---
        u = df.TrialFunction(V)
        v = df.TestFunction(V)
        a = df.inner(df.nabla_grad(u), df.nabla_grad(v)) * df.dx
        L = df.Constant(0.0) * v * df.dx
        phi = df.Function(V)
        bc_tm = [
            df.DirichletBC(V, df.Constant(0.0), facetboundaries, int(lvendoid)),
            df.DirichletBC(V, df.Constant(1.0), facetboundaries, int(epiid)),
        ]
        df.solve(a == L, phi, bc_tm, solver_parameters={"linear_solver": "mumps"})

        # --- apicobasal Laplace: apex=0, base=1 ---
        psi = df.Function(V)
        bc_ab = [
            df.DirichletBC(V, df.Constant(0.0), _Apex()),
            df.DirichletBC(V, df.Constant(1.0), facetboundaries, int(baseid)),
        ]
        df.solve(a == L, psi, bc_ab, solver_parameters={"linear_solver": "mumps"})

        # --- gradients: smooth via CG2 vector projection (as addLVfiber does),
        # then sample at the fiber-space quadrature points. ---
        Vg = df.VectorFunctionSpace(mesh, "CG", 2)
        grad_tm_cg = df.project(df.grad(phi), Vg, solver_type="mumps")
        grad_ab_cg = df.project(df.grad(psi), Vg, solver_type="mumps")

        gT = df.project(grad_tm_cg, fiber_space, solver_type="mumps").vector().get_local().reshape(-1, 3)
        gA = df.project(grad_ab_cg, fiber_space, solver_type="mumps").vector().get_local().reshape(-1, 3)

        # transmural coordinate at the same quadrature points
        Sq = df.FunctionSpace(
            mesh,
            df.FiniteElement("Quadrature", mesh.ufl_cell(), degree=deg, quad_scheme="default"),
        )
        t = df.project(phi, Sq, solver_type="mumps").vector().get_local()
        t = np.clip(t, 0.0, 1.0)
    finally:
        fc["representation"] = prev_repr
        fc["quadrature_degree"] = prev_qd

    n_pts = gT.shape[0]
    if t.shape[0] != n_pts:
        raise RuntimeError(
            "lv_fiber_laplace: scalar/vector quadrature point counts differ "
            "(%d vs %d) — degree/space mismatch." % (t.shape[0], n_pts)
        )

    # --- vectorized local frame + helix rotation ---
    e1, _ = _normalize_rows(gA, fallback=np.array([0.0, 0.0, 1.0]))         # apicobasal
    gT_perp = gT - np.sum(gT * e1, axis=1)[:, None] * e1
    e2, bad = _normalize_rows(gT_perp, fallback=np.array([1.0, 0.0, 0.0]))   # transmural
    e0 = np.cross(e1, e2)                                                    # circumferential
    e0, _ = _normalize_rows(e0, fallback=np.array([0.0, 1.0, 0.0]))

    alpha = np.deg2rad(alpha_endo * (1.0 - t) + alpha_epi * t)
    c = np.cos(alpha)[:, None]
    s = np.sin(alpha)[:, None]
    eF = c * e0 + s * e1
    eS = -s * e0 + c * e1
    eN = e2

    f0 = df.Function(fiber_space)
    s0 = df.Function(fiber_space)
    n0 = df.Function(fiber_space)
    f0.vector().set_local(np.ascontiguousarray(eF).reshape(-1))
    s0.vector().set_local(np.ascontiguousarray(eS).reshape(-1))
    n0.vector().set_local(np.ascontiguousarray(eN).reshape(-1))
    for fn in (f0, s0, n0):
        fn.vector().apply("insert")

    n_bad = int(df.MPI.sum(comm, float(bad.sum())))
    if n_bad and df.MPI.rank(comm) == 0:
        print("[lv_fiber_laplace] %d degenerate quadrature points used fallback "
              "directions (likely apex/base singularities)." % n_bad, flush=True)

    return f0, s0, n0
