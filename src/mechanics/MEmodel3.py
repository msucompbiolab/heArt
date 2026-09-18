import dolfin as df
import ufl as ufl
from ufl import derivative, CellDiameter
import numpy as np
import os as os
from ..utils.nsolver import NSolver as NSolver
from ..utils.oops_objects_MRC2 import biventricle_mesh as biv_mechanics_mesh
from ..utils.oops_objects_MRC2 import lv_mesh as lv_mechanics_mesh
from ..utils.oops_objects_MRC2 import fch_mesh as fch_mechanics_mesh
from ..utils.oops_objects_MRC2 import update_mesh
from ..utils.oops_objects_MRC2 import printout
from ..utils.oops_objects_MRC2 import _marker_ids_present
from ..utils.edgetypebc import *
from .forms_MRC2 import Forms
from .activeforms_MRC2 import activeForms
from .spring_bc_forms import SpringBCContext, build_spring_form
from .waorta_support import (
    build_passive_form_waorta,
    build_stabilization_mu_waorta,
    dirichlet_bcs_waorta,
    laplace_waorta,
    lv_volume_waorta,
    maybe_add_aorta_pressure_term,
)
from ..utils.mesh_partitionMeshforEP_J import defCPP_Matprop, defCPP_Matprop_DIsch
from ..utils.mesh_scale_create_fiberFiles import create_EDFibers
from ..utils.lv_fiber_laplace import generate_lv_fibers_laplace


class UnloadingConvergenceError(RuntimeError):
    """Expected numerical failure of an unloading continuation.

    Drivers may recover from this exception with a documented alternative
    continuation.  RuntimeError from output, HDF5, or unrelated orchestration
    is intentionally *not* converted to this type and must fail the run.
    """


def fch_active_tmax_factor(SimDet, ch):
    """Per-chamber absolute-Tmax factor ``Tmax_<ch> / Tmax_fallback``.

    The Burkhoff active stress is linear in Tmax, so multiplying the base active form
    (built at the documented fallback Tmax = ``GiccioneParams["Active params"]["Tmax"]``)
    by this factor yields the chamber's ABSOLUTE contractility. This is the SINGLE source
    of the per-chamber active scaling, shared by the MEmodel active assembly (``Problem``)
    and the ``run_light`` active-stress monitor -- no context reads a bare base. ``ch`` is
    one of "lv"/"rv"/"la"/"ra" (``None`` or a missing ``Tmax_<ch>`` -> factor 1.0).
    """
    # Under SimDet["Tmax_region"] the "Tmax" entry has been REPLACED by a DG0 field, so
    # float() on it raises "Cannot convert spatially varying function to float" and the whole
    # FCH per-chamber assembly dies at build time. "Tmax_region_base" is the scalar that field
    # was built around and is written beside it precisely so the fallback stays recoverable;
    # without this the regional-contractility feature and the four-chamber per-chamber
    # contractility feature could not be used together at all.
    _act = SimDet["GiccioneParams"]["Active params"]
    _tmax = _act["Tmax"]
    tmax_fallback = float(_tmax if isinstance(_tmax, (int, float)) else _act["Tmax_region_base"])
    if ch is None:
        return 1.0
    v = SimDet.get("Tmax_%s" % ch)
    return (float(v) / tmax_fallback) if v is not None else 1.0


def aitken_omega_step(omega_prev, r_prev, r_cur, omega_min, omega_max, comm):
    """Irons-Tuck Aitken relaxation update, shared by every unloading() caller.

    w_{n+1} = -w_n * <r_n, dr> / <dr, dr>, dr = r_{n+1} - r_n, clamped to
    [omega_min, omega_max]. MPI-summed dot products (exact for -np 1 runs).
    Extracted from ``_unloading_fch`` (research_dossier_unload_continuation_
    numerics_lv.json, Irons & Tuck 1969) so the LV path can reuse the identical
    arithmetic; caller-side policy (fixed-omega stages, MFF-active holds,
    restart-on-stage-advance) is NOT part of this function -- it stays at each
    call site so existing behavior is provably unchanged by the extraction.
    """
    dr = r_cur - r_prev
    denom = df.MPI.sum(comm, float(np.dot(dr, dr)))
    if denom <= 1e-30:
        return omega_prev
    num = df.MPI.sum(comm, float(np.dot(r_prev, dr)))
    w = -omega_prev * num / denom
    return min(max(w, omega_min), omega_max)


def mff_stall_step(stall, exhausted, res, res_prev, patience):
    """Advance the MFF fit-exhaustion counter; return ``(stall, freeze)``.

    The MFF hook reports ``exhausted=True`` for an iteration whose clamped 1-DOF stiffness update
    was an EXACT no-op (the DOF sits on a physiological bound and the residual asks it to step
    further across -- see ``unloading.params.clamp_exhausted_bound``). One such iteration does not
    justify stopping: the reference is still moving, so the V0 residual could in principle come
    back across the bound and re-open the fit. Two conditions together do:

    * the no-op has persisted for ``patience`` CONSECUTIVE iterations, and
    * the Sellier mesh residual is strictly CONTRACTING across them (``res < res_prev``),

    i.e. the frozen-material iteration is converging to its own fixed point rather than wandering.
    A contracting reference cannot then move V0 by more than the geometric tail of that residual,
    which is orders of magnitude below the V0 gap that pinned the DOF in the first place. Any
    iteration that is not exhausted, or whose residual failed to contract, RESETS the counter --
    so a fit that is merely passing through a bound is never frozen.

    ``freeze`` is advisory in exactly the sense the existing ``mff_max_iter`` backstop is: it stops
    RESCALING the material. It does not terminate the unload -- termination still requires the
    unchanged ``mesh_converged`` criterion.
    """
    if not exhausted or res_prev is None or not (res < res_prev):
        return 0, False
    stall += 1
    return stall, stall >= patience


class MEmodel(object):
    def __init__(self, params, SimDet):
        self.parameters = self.default_parameters()
        self.parameters.update(params)
        self.SimDet = SimDet
        self.isLV = self.SimDet.get("isLV", False)
        giccione_params = SimDet["GiccioneParams"]
        self.deg_me = giccione_params["deg"]

        self.discretization = self.SimDet.get("Mechanics Discretization", "P2P1")
        self.discretization_technique = self.SimDet.get("Technique Discretization", 1)

        self.ispctrl = self.SimDet.get("ispctrl", False)
        self.iswaorta = self.SimDet.get("iswaorta", False)
        self.isFCH = self.SimDet.get("isFCH", False)
        self.isBiV = self.SimDet.get("isBiV", False)
        state_obj = self.parameters.get("state_obj")
        dt_init = (
            getattr(getattr(state_obj, "dt", None), "dt", 0.0) if state_obj is not None else 0.0
        )
        # Keep `dt` as a UFL Constant so forms can depend on it without rebuilding expression trees.
        self.dt_const = df.Constant(dt_init)

        mesh_builders = [
            ((self.isLV or self.iswaorta), lv_mechanics_mesh),
            (self.isFCH, fch_mechanics_mesh),
            (self.isBiV, biv_mechanics_mesh),
        ]
        # Select the mesh wrapper based on mutually exclusive model flags (LV/waorta > FCH > BiV).
        for condition, builder in mesh_builders:
            if condition:
                self.Mesh = builder(self.parameters, SimDet)
                break
        else:
            raise ValueError("Mechanics mesh type could not be resolved.")

        self._set_attrs(
            self.Mesh,
            {"f0_me": "f0", "s0_me": "s0", "n0_me": "n0"},
        )
        self._set_attrs(
            self.Mesh,
            {
                "eL0_me": "eL0_ao",
                "eC0_me": "eC0_ao",
                "eclgn0_me": "eclgn0_ao",
                "eclgn1_me": "eclgn1_ao",
            },
            optional=True,
        )

        attr_map = {
            "mesh_me": "mesh",
            "facetboundaries_me": "facetboundaries",
            "edgeboundaries_me": "edgeboundaries",
            "matid_me": "matid",
            "ds_me": "ds",
            "dx_me": "dx",
        }
        for target, source in attr_map.items():
            setattr(self, target, getattr(self.Mesh, source))

        # Detect a basally-TRIMMED (open-base) single-chamber LV mesh: the endocardial
        # surface alone is not closed (it is cut off at the mitral-annulus plane), so the
        # divergence-theorem cavity-volume integral needs the base-facet-centroid offset
        # (Forms.LVcavityvol_mvb) rather than the origin-anchored form (Forms.LVcavityvol),
        # which is only exact for a CLOSED endocardial surface (offset-invariant there).
        # Mesh-driven, not a manual flag: an untrimmed LV mesh carries no `topid` facets at
        # all, so this is false for it automatically (mirrors the matid graceful-default
        # pattern elsewhere in this loader).
        self.lv_open_base = False
        if self.isLV:
            topid = self.SimDet.get("topid")
            top_ids = [i for i in np.atleast_1d(topid).tolist() if i is not None] if topid is not None else []
            present = _marker_ids_present(self.facetboundaries_me, top_ids, self.mesh_me.mpi_comm())
            self.lv_open_base = bool(present)
            # ANNOUNCE the cavity-volume CONVENTION once, at build. Everything downstream (the
            # Klotz anchor, the prescribed ED/ES volumes, every reported volume_ml) is expressed
            # in whichever functional this flag selects, and the two differ by ~16% on a real
            # trimmed patient mesh (136.73 origin-anchored vs 117.24 mvb). A wrong value here is
            # SILENT: prescribed and reported volumes still agree with each other, because both
            # would use the same wrong form -- so consistency checks cannot catch it and only the
            # convention itself can be asserted. Logged, never assumed.
            printout(
                "cavity volume convention: lv_open_base=%s -> %s (topid=%s %s in facetboundaries)"
                % (self.lv_open_base,
                   "LVcavityvol_mvb (base-centroid offset; REQUIRED for a trimmed/open base)"
                   if self.lv_open_base else
                   "LVcavityvol (origin-anchored; valid ONLY for a closed endocardium)",
                   top_ids, "present" if present else "ABSENT"),
                self.mesh_me.mpi_comm(), stage="setup")

        LVendoid = self.SimDet["LVendoid"]
        ids = np.atleast_1d(LVendoid).tolist()
        # LV endocardium may be split across multiple facet ids; aggregate into a single measure.
        ds_measures = [
            self.ds_me(id_, domain=self.mesh_me, subdomain_data=self.facetboundaries_me)
            for id_ in ids
        ]
        if not ds_measures:
            raise ValueError("No LV endocardial ids provided for mechanics endocardium.")
        dsendo = sum(ds_measures[1:], ds_measures[0])
        self.LVendo_area_me = df.Constant(
            df.assemble(
                df.Constant(1.0) * dsendo,
                form_compiler_parameters={"representation": "uflacs"},
            )
        )

        # Scalars stored as Constants to avoid JIT of Expressions
        self.LVCavityvol = df.Constant(0.0)
        self.RVCavityvol = df.Constant(0.0)
        # FCH swept-cavity volume-control target (the prescribed cavity volume the
        # monolithic Lagrange multiplier enforces; used only when fch_swept_volctrl set).
        self.FCHsweptCavityvol = df.Constant(0.0)
        # One prescribed-volume Constant per swept cavity. With a single swept chamber this
        # dict simply aliases FCHsweptCavityvol, so existing callers keep working unchanged.
        _swept_chs = self.fch_swept_chambers()
        self.FCHsweptCavityvols = (
            {c: df.Constant(0.0) for c in _swept_chs} if len(_swept_chs) > 1
            else ({_swept_chs[0]: self.FCHsweptCavityvol} if _swept_chs else {}))
        self.LVCavitypres = df.Constant(0.0)
        self.RVCavitypres = df.Constant(0.0)
        self.p_a = df.Constant(0.0)
        self.p_v = df.Constant(1300.0)
        self.LACavitypres = df.Constant(0.0)
        self.RACavitypres = df.Constant(0.0)
        self.AortaCavitypres = df.Constant(0.0)

        self.isincomp = giccione_params["incompressible"]
        #  - - - - - - - - - - - -- - - - - - - - - - - - - - - -- - - - - - -

        if self.discretization == "P1P1":
            Velem = df.VectorElement(
                "CG", self.mesh_me.ufl_cell(), 1, quad_scheme="default"
            )
        else:
            Velem = df.VectorElement(
                "CG", self.mesh_me.ufl_cell(), 2, quad_scheme="default"
            )

        Qelem = df.FiniteElement("CG", self.mesh_me.ufl_cell(), 1, quad_scheme="default")
        Qelem._quad_scheme = "default"
        Relem = df.FiniteElement("Real", self.mesh_me.ufl_cell(), 0, quad_scheme="default")
        Relem._quad_scheme = "default"
        # Quadrature spaces are used for history-dependent constitutive quantities (stress/activation).
        Quadelem = df.FiniteElement(
            "Quadrature",
            self.mesh_me.ufl_cell(),
            degree=self.deg_me,
            quad_scheme="default",
        )
        Quadelem._quad_scheme = "default"
        #  - - - - - - - - - - - -- - - - - - - - - - - - - - - -- - - - - - -

        #  - - - - - - - - - - - -- - - - - - - - - - - - - - - -- - - - - - -
        Telem2 = df.TensorElement(
            "Quadrature",
            self.mesh_me.ufl_cell(),
            degree=self.deg_me,
            shape=2 * (3,),
            quad_scheme="default",
        )
        Telem2._quad_scheme = "default"
        for e in Telem2.sub_elements():
            e._quad_scheme = "default"
        Telem4 = df.TensorElement(
            "Quadrature",
            self.mesh_me.ufl_cell(),
            degree=self.deg_me,
            shape=4 * (3,),
            quad_scheme="default",
        )
        Telem4._quad_scheme = "default"
        for e in Telem4.sub_elements():
            e._quad_scheme = "default"
        #  - - - - - - - - - - - -- - - - - - - - - - - - - - - -- - - - - - -

        # Mixed Element for rigid body motion
        # Real-valued multipliers constrain translations/rotations when no explicit anchoring BC is used.
        # Default 5 = X,Y translation + 3 rotations (Z translation is normally
        # anchored by the basal spring/Dirichlet support). The free-top variant
        # has NO base support, so Z translation is an unconstrained null mode ->
        # use 6 multipliers for full rigid-body removal (physically neutral: zero
        # net force/moment; does not affect deformation/stress).
        #  - - - - - - - - - - - -- - - - - - - - - - - - - - - -- - - - - - -
        # auto_rigid_support derives the MINIMAL rigid-body removal from the spring set
        # (rigid_support.py); it always carries the full 6 Real multipliers (the inactive ones are
        # pinned to zero), so the unconstrained Z/tilt modes a partial spring leaves are covered.
        n_rigid = 6 if (bool(self.SimDet.get("free_top"))
                        or bool(self.SimDet.get("auto_rigid_support"))) else 5
        VRelem = df.MixedElement([Relem] * n_rigid)
        #  - - - - - - - - - - - -- - - - - - - - - - - - - - - -- - - - - - -

        #  - - - - - - - - - - - -- - - - - - - - - - - - - - - -- - - - - - -
        # Mixed unknown layout depends on incompressibility, p-control, chamber count, and spring BC mode.
        mixed_structure = self._mechanics_mixed_elements(Velem, Qelem, Relem, VRelem)
        self.W = df.FunctionSpace(self.mesh_me, df.MixedElement(mixed_structure))

        self.Quad = df.FunctionSpace(self.mesh_me, Quadelem)
        self.TF = df.FunctionSpace(self.mesh_me, Telem2)
        self.Q = df.FunctionSpace(self.mesh_me, "CG", 1)
        self.QDG = df.FunctionSpace(self.mesh_me, "DG", 0)
        self.V_CG1 = df.VectorFunctionSpace(self.mesh_me, "CG", 1)

        self.we_n = df.Function(self.W.sub(0).collapse())

        self.w_me = df.Function(self.W)
        self.w_me_n = df.Function(self.W)
        self.dw_me = df.TrialFunction(self.W)
        self.wtest_me = df.TestFunction(self.W)

        self.u_me_ED = df.Function(self.V_CG1)
        self.Ftotal, self.Jac, self.bcs = self.Problem()

    def default_parameters(self):
        return {"probeloc": [3.5, 0.0, -2.0]}

    def _set_attrs(self, source, mapping, optional=False):
        for target_attr, source_attr in mapping.items():
            if optional and not hasattr(source, source_attr):
                setattr(self, target_attr, None)
                continue
            setattr(self, target_attr, getattr(source, source_attr))

    def _init_expressions(self, specs, degree=2):
        # Legacy helper kept for compatibility; prefer Constants over Expressions.
        for attr_name, component, values in specs:
            setattr(self, attr_name, df.Constant(list(values.values())[0]))

    @staticmethod
    def _zero_vector_expr():
        return df.Constant((0.0, 0.0, 0.0))

    def _normalize_basis(self):
        def normalize_inplace(func):
            vec = func.vector()
            local = vec.get_local() if hasattr(vec, "get_local") else vec.array()
            value_size = func.function_space().ufl_element().value_size()
            if value_size <= 0:
                return
            data = local.reshape((-1, value_size))
            norms = np.linalg.norm(data, axis=1)
            norms[norms == 0.0] = 1.0
            data /= norms[:, None]
            if hasattr(vec, "set_local"):
                vec.set_local(data.reshape((-1,)))
            else:
                vec[:] = data.reshape((-1,))
            vec.apply("insert")

        for attr in ("f0_me", "s0_me", "n0_me"):
            vec = getattr(self, attr, None)
            if vec is not None:
                # Prefer in-place normalization when the basis is stored as a df.Function (avoids UFL rebinding).
                if hasattr(vec, "vector") and hasattr(vec, "function_space"):
                    normalize_inplace(vec)
                else:
                    setattr(self, attr, vec / df.sqrt(df.inner(vec, vec)))

    def fch_swept_chambers(self):
        """Chambers held by a monolithic swept-cavity constraint, as an ordered list.

        ``SimDet["fch_swept_volctrl"]`` accepts a single chamber name (the original form, and
        still the common case), a comma-separated string, or a list. One chamber keeps the
        original single ``"fch_swept"`` layout block, so every existing caller -- twitch_bench,
        fch_active_stability_sweep, the LVW path -- is byte-identical. TWO OR MORE cavities are
        what an end-systolic four-chamber state needs: it drives BOTH ventricles far below their
        unloaded volumes at once, and a partitioned pressure root-find cannot follow a cavity
        through the limit point of its equilibrium branch.
        """
        if not self.isFCH:
            return []
        raw = self.SimDet.get("fch_swept_volctrl")
        if not raw:
            return []
        if isinstance(raw, str):
            chs = [c.strip() for c in raw.split(",")]
        elif isinstance(raw, (list, tuple)):
            chs = [str(c).strip() for c in raw]
        else:
            # A truthy NON-name flag is the legacy spelling (and what the layout tests use):
            # it selects one swept block without naming a chamber, exactly as before.
            return []
        chs = [c for c in chs if c]
        bad = [c for c in chs if c not in ("lv", "rv", "la", "ra")]
        if bad:
            raise ValueError("fch_swept_volctrl names unknown chamber(s) %s; expected lv/rv/la/ra"
                             % bad)
        if len(set(chs)) != len(chs):
            raise ValueError("fch_swept_volctrl repeats a chamber: %s" % chs)
        return chs

    def _fch_swept_block_names(self):
        """Layout block name per swept chamber. One chamber -> the original "fch_swept"."""
        chs = self.fch_swept_chambers()
        if len(chs) > 1:
            return ["fch_swept_%s" % c for c in chs]
        # One named chamber, or the legacy truthy flag: the original single block.
        return ["fch_swept"] if (self.isFCH and self.SimDet.get("fch_swept_volctrl")) else []

    def mechanics_mixed_layout(self):
        """Ordered NAMES of the mixed-space sub-functions for the active configuration.

        Single source of truth for the mixed layout: `_mechanics_mixed_elements` builds
        the FunctionSpace from this list, and `_split_state` selects sub-functions from
        `w_me` BY INDEX using it. Any accessor that needs `u`/`p`/a multiplier must go
        through those two, never a fixed-arity `w_me.split()` unpack — the trailing
        blocks vary with springbc / auto_rigid_support / fch_swept_volctrl / chamber count.

        Names: "u" (displacement), "p" (pressure, incompressible only), "lv_pendo" /
        "rv_pendo" (cavity-volume Real multipliers), "fch_swept" (FCH swept-cavity Real
        multiplier), "rigid" (the rigid-body Real multiplier block, VRelem).
        """
        spring_bc = bool(self.SimDet.get("springbc"))
        # auto_rigid_support adds the rigid-body multipliers ALONGSIDE the spring (the spring still
        # anchors the modes it can; the multipliers cover the rest), so "rigid" is present even when
        # springbc is on. `need_rigid` is the single source of truth for "are c_me multipliers in
        # the space" — it MUST match the df.split branches in Problem().
        auto_rigid = bool(self.SimDet.get("auto_rigid_support"))
        need_rigid = (not spring_bc) or auto_rigid
        is_single_chamber = self.isLV or self.iswaorta
        is_multi_chamber = self.isBiV or self.isFCH

        if not (is_single_chamber or is_multi_chamber):
            raise ValueError("Unable to determine chamber type for mechanics function space.")

        def maybe_append(container, condition, name):
            if condition:
                container.append(name)

        if self.isincomp:
            # Incompressible: include pressure; non-pctrl uses Real scalars for volume constraints (LV/RV).
            if self.ispctrl:
                names = ["u", "p"]
                # FCH swept-cavity volume control: ONE extra Real multiplier (the swept
                # chamber's cavity pressure) enforcing a monolithic volume constraint, so
                # the isovolumic twitch holds volume through high-tension states the
                # partitioned pressure root-find can't reach. Opt-in; default path unchanged.
                fch_swept = self.isFCH and bool(self.SimDet.get("fch_swept_volctrl"))
                # LVW swept-cavity volume control: the SAME monolithic constraint for the
                # single waorta cavity (LVendoid + aortic_valvep + mitral_valvep). Opt-in via
                # SimDet["lvw_swept_volctrl"]; default path unchanged.
                lvw_swept = self.iswaorta and bool(self.SimDet.get("lvw_swept_volctrl"))
                if spring_bc:
                    # The LVW swept cavity occupies the SAME layout slot as the FCH one
                    # (one Real multiplier; the split branches below key on either flag).
                    if fch_swept:
                        names.extend(self._fch_swept_block_names())
                    elif lvw_swept:
                        names.append("fch_swept")
                    # auto_rigid_support: carry the 6 rigid-body Real multipliers ALONGSIDE
                    # the springs (springs anchor what they can; the multipliers cover the residual
                    # null modes -> well-posed for ANY spring coverage). This applies to both
                    # pressure-controlled LV and FCH models and must match the split branches.
                    if auto_rigid and (self.isLV or self.isFCH):
                        names.append("rigid")
                    return names
                names.append("rigid")
                return names

            names = ["u", "p", "lv_pendo"]
            maybe_append(names, is_multi_chamber, "rv_pendo")
            maybe_append(names, need_rigid, "rigid")
            return names

        # Compressible: no pressure unknown; keep only volume constraints / rigid-body multipliers as needed.
        names = ["u"]
        if self.ispctrl:
            maybe_append(names, need_rigid, "rigid")
            return names

        names.append("lv_pendo")
        maybe_append(names, is_multi_chamber, "rv_pendo")
        maybe_append(names, not spring_bc, "rigid")
        return names

    def _mechanics_mixed_elements(self, Velem, Qelem, Relem, VRelem):
        element_for = {
            "u": Velem,
            "p": Qelem,
            "lv_pendo": Relem,
            "rv_pendo": Relem,
            "fch_swept": Relem,
            "rigid": VRelem,
        }
        return [Relem if name.startswith("fch_swept") else element_for[name]
                for name in self.mechanics_mixed_layout()]

    def _split_state(self, w=None):
        """Split a mixed state function into a {name: sub-function} map.

        Selects by INDEX from `mechanics_mixed_layout()` instead of unpacking a
        fixed-arity tuple, so an accessor stays correct when the layout gains a
        trailing block (auto_rigid_support, fch_swept_volctrl, a second chamber).
        """
        w = self.w_me if w is None else w
        names = self.mechanics_mixed_layout()
        parts = w.split(deepcopy=True) if len(names) > 1 else (w,)
        if len(parts) != len(names):
            raise RuntimeError(
                "Mechanics mixed-space layout mismatch: expected {} sub-function(s) {} "
                "but w_me splits into {}.".format(len(names), names, len(parts))
            )
        return dict(zip(names, parts))

    def unloading_pres(self, params):
        defaults = {"EDP": 12, "preinc": 1}
        defaults.update(params)
        EDP = defaults["EDP"]
        preinc = defaults["preinc"]

        # Simple ramp in cavity pressure until EDP is reached (used to initialize unloaded iterations).
        while float(self.LVCavitypres) <= EDP:
            self.LVCavitypres.assign(float(self.LVCavitypres) + 0.1)
            self.Solver.solvenonlinear()

        return preinc

    def unloading(self, params):
        # Four-chamber meshes use a pressure-control continuation variant (the
        # FCH model is ispctrl=True with four cavity-pressure Constants and only
        # LV/RV volume multipliers wired). It preserves the same Sellier backward
        # update, continuation staging, residual/dres convergence, and adaptive
        # backoff as the LV path below, sharing _sellier_backward_update and
        # _write_unload_mesh. See _unloading_fch.
        if self.isFCH:
            return self._unloading_fch(params)
        if self.isLV and self.ispctrl:
            return self._unloading_lv_pressure(params)
        default_params = {
            "EDP": 12,
            "maxit": 20,
            "restol": 1e-3,
            "drestol": 1e-4,
            "EDPtol": 1e-1,
            "volinc": 1,
        }
        default_params.update(params)
        # EDP continuation: reach a high target EDP by walking up a schedule of
        # intermediate EDPs. Each stage warm-starts from the previous stage's
        # (smaller, better-conditioned) reference, so the near-vertical EDPVR wall
        # that blocks a one-shot high-EDP unload on a stiff material is avoided.
        # `edp_schedule` (list, ascending) overrides the single `EDP`; intermediate
        # stages converge coarsely (`stage_restol`/`stage_maxit`), the final stage
        # to the tight `restol`/`maxit`.
        edp_schedule = list(default_params.get("edp_schedule") or [default_params["EDP"]])
        maxit = default_params["maxit"]
        restol = default_params["restol"]
        drestol = default_params["drestol"]
        EDPtol = default_params["EDPtol"]
        stage_maxit = int(default_params.get("stage_maxit", maxit))
        stage_restol = float(default_params.get("stage_restol", max(restol, 0.1)))
        # MFF (Marx 2022) coupled material co-fit hook (LV path). When provided, `mff` is a
        # callable invoked once per FINAL-stage outer iteration with the simulated loading PV
        # curve [(P_mmHg, V_mL), ...]; it fits the Klotz model function, RESCALES the passive
        # material in SimDet (read fresh by Problem() on the next rebuild), and returns
        # {"converged": bool, ...}. None (default, and ALL FCH/legacy callers) => byte-identical
        # legacy behavior (no curve accumulation, no extra termination condition).
        mff = default_params.get("mff")
        mff_max_iter = int(default_params.get("mff_max_iter", maxit))
        mff_it = 0
        mff_done = False  # set once scalings converge (or the backstop fires): freeze material
        # Fit-exhaustion stall detector (see mff_stall_step). Counts CONSECUTIVE outer iterations
        # whose material update was an exact clamp no-op while the mesh residual kept contracting.
        mff_stall = 0
        mff_stall_patience = int(default_params.get("mff_stall_patience", 3))
        mff_res_prev = None
        LVendoid = self.SimDet["LVendoid"]
        matid_dataset = self.SimDet.get("matid_dataset", "matid1")
        volinc = default_params["volinc"]
        outputfolder = self.parameters["outputfolder"]
        folderName = self.parameters["foldername"].rstrip("/")
        outfolder = os.path.join(outputfolder, folderName, "deformation_unloadED")

        targetmesh = df.Mesh(self.Mesh.mesh)
        # Target is the current reference coordinates; unloading iteratively updates the mesh to match it.
        xtarget = df.project(df.SpatialCoordinate(self.mesh_me), self.V_CG1).vector()

        comm_me = self.mesh_me.mpi_comm()
        is_root = df.MPI.rank(comm_me) == 0

        if is_root and not os.path.isdir(outfolder):
            os.makedirs(outfolder)

        fdataPV = open(outfolder + "BiV_unloadPV.txt", "w", buffering=1) if is_root else None
        # Per-iteration mesh/displacement dump to Data_unload.h5 is debug-only and
        # large (~100 MB over a full schedule). Off by default to keep the unloading
        # footprint small (only the final UnloadMesh.hdf5 ~37 MB matters).
        write_debug = bool(default_params.get("write_debug_h5", False))
        hdf = df.HDF5File(comm_me, outfolder + "Data_unload.h5", "w") if write_debug else None

        it = 0
        res = 1e9
        dres = 0
        # Sellier relaxation (Aitken/Irons-Tuck, shared with _unloading_fch via
        # aitken_omega_step). `alpha_lv` defaults to 1.0 (undamped) -- byte-
        # identical to the legacy hardcoded alpha=1.0 until explicitly enabled.
        # See research_dossier_unload_continuation_numerics_lv.json.
        omega_init = float(default_params.get("alpha_lv", 1.0))
        omega_min = float(default_params.get("omega_min_lv", 0.05))
        omega_max = float(default_params.get("omega_max_lv", 1.0))
        # Aitken adaptation is opt-in (`unload.aitken_lv`, default off): at the
        # default omega_init=1.0 this keeps the LV path byte-identical to the
        # pre-Phase-2 hardcoded alpha=1.0 (the Aitken update itself would drift
        # omega away from 1.0 starting the 2nd outer iteration even with
        # omega_init=1.0, so gating is required for the "no behavior change by
        # default" guarantee, not just the omega_init value).
        aitken_enabled = bool(default_params.get("aitken_lv", False))
        omega = omega_init
        r_prev = None  # previous Sellier residual vector (for Aitken)

        # Stage state for EDP continuation.
        stage_i = 0
        EDP = edp_schedule[stage_i]
        is_final_stage = (stage_i == len(edp_schedule) - 1)
        cur_restol = restol if is_final_stage else stage_restol
        cur_maxit = maxit if is_final_stage else stage_maxit
        it_stage = 0
        printout(
            "=== unloading EDP schedule = %s mmHg ===" % edp_schedule, comm_me
        )
        printout(
            "--- stage %d: EDP=%g mmHg (restol=%g, maxit=%d) ---"
            % (stage_i, EDP, cur_restol, cur_maxit),
            comm_me,
        )

        def log_cycle_data(iteration, pressure, volume):
            if is_root:
                print(iteration, pressure, volume, file=fdataPV)

        while True:
            it_load = 0
            # Constraint-functional value, not the physical volume -- see
            # lv_constraint_volume(). No-op under pressure control and on a closed
            # endocardium; corrects the open-base volume-control case.
            self.LVCavityvol.assign(self.lv_constraint_volume())
            LVP = self.get_lv_pressure() * 0.0075
            LVV = self.get_lv_volume()
            printout(f"Iteration number = {it}", comm_me,
                     stage="loading", it=it, p_mmhg=LVP, v_ml=LVV)
            log_cycle_data(it, LVP, LVV)
            # MFF: accumulate THIS outer iteration's simulated loading PV curve (from the bare
            # reference at ~0 pressure up to EDP); the first point's volume is the current
            # unloaded reference V0_sim used to normalize the model-function fit.
            pv_iter = [(LVP, LVV)] if mff is not None else None

            if hdf is not None:
                hdf.write(self.mesh_me, "unloading" + str(it) + "/mesh")
                hdf.write(
                    self.get_displacement(), "unloading" + str(it) + "/u_loading", it_load
                )

            volinc_floor = default_params.get("volinc_floor", 1e-3)
            # Telemetry only (verbosity-gated, no behavior change): distinguishes a
            # genuinely-converging-but-slow inner loop from a stuck one for diagnosis.
            # See src/mechanics/CLAUDE.md's unloading() section / research_dossier_
            # unload_continuation_numerics_lv.json for why this matters.
            verbose_inner = bool(default_params.get("verbose_inner_loop", False))
            if verbose_inner:
                # Shows the carried-forward starting volinc for THIS outer iteration
                # (only reset at EDP-schedule stage boundaries, not every iteration --
                # see the "volinc is intentionally NOT reset here" note below) so a
                # live run can confirm the carry-forward fix is actually in effect.
                printout(
                    "  [outer-loop telemetry] it=%d stage=%d starting_volinc=%.6g"
                    % (it, stage_i, volinc),
                    comm_me,
                )
            consecutive_halvings = 0
            any_halving_this_outer = False
            while True:
                # Inner loop: adjust cavity volume to hit EDP with adaptive step size.
                vol_prev = float(self.LVCavityvol)
                w_prev = self.w_me.vector().copy()
                self.LVCavityvol.assign(vol_prev + volinc)
                try:
                    self.Solver().solvenonlinear()
                    solve_ok = True
                except RuntimeError:
                    solve_ok = False

                if not solve_ok:
                    # Newton/PETSc failure (stiff material at high inflation): roll back
                    # state and volume, halve the increment, and retry a smaller step.
                    self.w_me.vector()[:] = w_prev
                    self.w_me.vector().apply("insert")
                    self.LVCavityvol.assign(vol_prev)
                    volinc *= 0.5
                    consecutive_halvings += 1
                    any_halving_this_outer = True
                    printout("Solve failed; halving volinc -> " + str(volinc), comm_me)
                    if verbose_inner:
                        printout(
                            "  [inner-loop telemetry] branch=solve_failure "
                            "consecutive_halvings=%d volinc=%.6g V=%.6g EDP=%.4g"
                            % (consecutive_halvings, volinc, vol_prev, EDP),
                            comm_me,
                        )
                    if volinc < volinc_floor:
                        raise UnloadingConvergenceError(
                            "unloading: volume-control solve diverged and volinc "
                            "fell below floor %g at V=%g mL, P<EDP=%g mmHg. The "
                            "material may be too stiff to inflate to this EDP on "
                            "this geometry; try a lower EDP." % (volinc_floor, vol_prev, EDP)
                        )
                    continue

                LVP = self.get_lv_pressure() * 0.0075
                LVV = self.get_lv_volume()

                if LVP > EDP:
                    printout("Decrease load step", comm_me)
                    self.LVCavityvol.assign(float(self.LVCavityvol) - volinc)
                    self.w_me.vector()[:] = w_prev
                    self.w_me.vector().apply("insert")
                    volinc *= 0.5
                    consecutive_halvings += 1
                    any_halving_this_outer = True
                    if verbose_inner:
                        printout(
                            "  [inner-loop telemetry] branch=edp_overshoot "
                            "consecutive_halvings=%d volinc=%.6g V=%.6g LVP=%.4g EDP=%.4g"
                            % (consecutive_halvings, volinc, vol_prev, LVP, EDP),
                            comm_me,
                        )
                    if volinc < volinc_floor:
                        # BUG FIX: this branch previously halved volinc with NO floor
                        # check at all (unlike the solve_failure branch above), so a
                        # solve that keeps nominally SUCCEEDING but repeatedly
                        # overshoots EDP could halve volinc indefinitely with zero net
                        # progress and no exit condition -- the exact root cause of a
                        # session-observed 178-halving, 11-CPU-minute stall with no
                        # crash. Fail loud with the same diagnostic pattern as the
                        # solve-failure branch, rather than looping silently forever.
                        raise UnloadingConvergenceError(
                            "unloading: volume-control repeatedly overshot EDP and "
                            "volinc fell below floor %g at V=%g mL, LVP=%g > EDP=%g "
                            "mmHg after %d consecutive halvings. The solve is "
                            "succeeding but the step size cannot resolve the target "
                            "EDP on this geometry/material -- check the EDP schedule "
                            "step size (unload.edp_step) or the material/geometry "
                            "conditioning at this stage; this is NOT a case to retry "
                            "blindly." % (volinc_floor, vol_prev, LVP, EDP, consecutive_halvings)
                        )
                    continue

                consecutive_halvings = 0

                printout(f"Loading iteration number = {it_load}", comm_me,
                         stage="loading", it=it_load, p_mmhg=LVP, v_ml=LVV)
                log_cycle_data(it, LVP, LVV)
                if pv_iter is not None:
                    pv_iter.append((LVP, LVV))

                if hdf is not None:
                    hdf.write(
                        self.get_displacement(),
                        "unloading" + str(it) + "/u_loading",
                        it_load,
                    )

                if abs(LVP - EDP) < EDPtol:
                    # Get Residual
                    x = df.project(
                        df.SpatialCoordinate(self.mesh_me) + self.get_displacement(),
                        self.V_CG1,
                    ).vector()

                    # Residual measures mismatch between deformed coordinates and the current target reference.
                    res_new = (x - xtarget).norm("l2")
                    dres = abs(res_new - res)
                    res = res_new
                    printout(
                        "Residual = " + str(res) + " dResidual = " + str(dres), comm_me
                    )

                    # REVERTED 2026-07-17 (HPCC evidence, job 12638955_2 vs 12658151_0/_1,
                    # identical b_scale=0.65/spring[4000,2000]/EDV/EDP config): carrying volinc
                    # forward (only growing it back x2/outer-iteration when nothing halved) was
                    # meant to skip re-discovering a shrunk step every outer iteration, but each
                    # ramp's LAST few steps shrink volinc for END-of-ramp EDPtol precision near
                    # EDP -- carrying THAT into the START of the next ramp (which begins back
                    # near V0, far from EDP, where a full-size step is perfectly safe) needlessly
                    # slows the easy early part of every subsequent ramp. Measured net effect on
                    # a real completed unload: halving events did drop (18->11) as intended, but
                    # total accepted volume-control steps ROSE 1615->2316 (+43%) and wall-clock
                    # ROSE 2073s->2905s (+40%, np=1, answer unchanged) -- a real regression, not
                    # noise (see src/mechanics/CLAUDE.md). Reset unconditionally, as before.
                    volinc = default_params["volinc"]

                    break

                it_load += 1

            it_stage += 1
            mesh_converged = res < cur_restol or dres < drestol
            # MFF (Marx 2022) coupled material co-fit: ONLY on the final EDP stage, once per outer
            # iteration after the forward inflation. The hook fits the simulated PV curve to the
            # Klotz model function and RESCALES the passive material in SimDet; the Sellier mesh
            # move + Problem() rebuild below then pick up the new material. The final break waits
            # until the scalings have converged (mff_converged) AND the mesh residual has settled,
            # bounded by the mff_max_iter backstop. When `mff` is None (legacy / all FCH callers)
            # mff_converged stays True => byte-identical legacy termination.
            mff_converged = True
            # mff_active also gates the Aitken hold below (mirrors _unloading_fch): while the
            # MFF co-fit is still rescaling material, the residual changes MEANING between
            # iterations, so Aitken must not adapt omega from it (validated 2026-07: enabling
            # aitken_lv without this hold produced a transient residual spike -- 1.45->5.21 --
            # during the final-stage MFF rescale on a real patient mesh, costing extra
            # iterations rather than saving them).
            mff_active = (mff is not None and is_final_stage and not mff_done)
            if mff_active:
                mff_it += 1
                mff_result = mff(pv_iter, it)
                if bool(mff_result.get("converged")):
                    mff_done = True  # scalings ~ 1: freeze material, let the mesh settle
                elif mff_it >= mff_max_iter:
                    printout(
                        "MFF: reached mff_max_iter=%d without scaling convergence; "
                        "freezing material (anchor QC reflects the V0 deviation)" % mff_max_iter,
                        comm_me,
                    )
                    mff_done = True  # backstop: stop rescaling
                else:
                    # Fit EXHAUSTED: the 1-DOF stiffness is pinned on a physiological clamp bound
                    # with the step pointing further across it, so this iteration rescaled NOTHING
                    # and neither will the next while that holds. Continuing to report the fit
                    # "not converged" would only suppress the mesh-residual termination below and
                    # burn full multi-stage forward inflations re-solving an unchanged material
                    # (observed: SLURM 13741898 ran 7 such iterations AFTER the mesh had already
                    # met restol). Freeze the material exactly as the max_iter backstop does and
                    # let the UNCHANGED mesh_converged criterion decide when to stop.
                    mff_stall, _mff_freeze = mff_stall_step(
                        mff_stall, bool(mff_result.get("exhausted")),
                        res, mff_res_prev, mff_stall_patience,
                    )
                    mff_res_prev = res
                    if _mff_freeze:
                        printout(
                            "WARNING MFF: passive fit EXHAUSTED after %d consecutive no-op "
                            "iteration(s) -- the fitted stiffness is pinned at its '%s' "
                            "physiological clamp bound and the V0 residual asks it to step "
                            "further across, so the material can no longer change. FREEZING the "
                            "material now (as the mff_max_iter=%d backstop would) instead of "
                            "re-solving it up to %d more time(s); the unload still runs to the "
                            "SAME mesh residual criterion, and the anchor QC reflects the "
                            "remaining V0 deviation (expect clamp flags in the sidecar)."
                            % (mff_stall, str(mff_result.get("bound")), mff_max_iter,
                               max(0, mff_max_iter - mff_it)),
                            comm_me,
                        )
                        mff_done = True
                    else:
                        mff_converged = False  # keep co-iterating (Sellier move + rescale)
            stage_converged = (
                it_stage >= cur_maxit or (mesh_converged and mff_converged)
            )
            if stage_converged:
                if is_final_stage:
                    # Fully done: mesh_me holds the unloaded reference for the
                    # highest EDP in the schedule.
                    self.Reset()
                    break
                # Advance to the next (higher) EDP, warm-starting from the current
                # reference. Do NOT update the mesh here — the converged mesh_me is
                # the warm start for the next stage.
                stage_i += 1
                EDP = edp_schedule[stage_i]
                is_final_stage = (stage_i == len(edp_schedule) - 1)
                cur_restol = restol if is_final_stage else stage_restol
                cur_maxit = maxit if is_final_stage else stage_maxit
                res = 1e9
                dres = 0
                it_stage = 0
                it += 1
                self.Reset()
                # Restart the volume-control step size at a stage boundary (the EDP
                # target genuinely changes, so a fresh full-size step is the right
                # starting guess) -- unlike the per-outer-iteration case, where the
                # step size that just worked is carried forward instead of reset.
                volinc = default_params["volinc"]
                # Restart Aitken across a stage boundary: the residual's meaning
                # changes (new EDP target), mirrors _unloading_fch's stage-advance
                # reset.
                omega = omega_init
                r_prev = None
                printout(
                    "--- stage %d: EDP=%g mmHg (restol=%g, maxit=%d) ---"
                    % (stage_i, EDP, cur_restol, cur_maxit),
                    comm_me,
                )
                continue

            if aitken_enabled and mff_active:
                # Hold omega at its initial value while MFF is still rescaling material (the
                # residual's meaning changes across a material change -- mirrors
                # _unloading_fch's mff_active hold). Byte-identical when aitken_lv=0.
                omega = omega_init
            r_cur = self._sellier_backward_update(targetmesh, omega)
            if aitken_enabled and (not mff_active) and r_prev is not None:
                omega = aitken_omega_step(omega, r_prev, r_cur, omega_min, omega_max, comm_me)
            r_prev = None if (aitken_enabled and mff_active) else r_cur
            if verbose_inner:
                # Confirms whether the outer Sellier residual oscillates (Aitken
                # damping helps -- per research_dossier_unload_continuation_
                # numerics_lv.json) or decays monotonically-but-slowly (Aitken
                # won't help; look elsewhere, e.g. EDP schedule step size or
                # spring/material conditioning). omega_init=1.0 (default) makes
                # this path byte-identical to the pre-Phase-2 hardcoded alpha=1.0.
                printout(
                    "  [outer-loop telemetry] it=%d stage=%d omega=%.6g "
                    "sellier_residual_l2norm=%.6g"
                    % (it, stage_i, omega, float((r_cur ** 2).sum() ** 0.5)),
                    comm_me,
                )

            # Update LV endo surface area for imposing BC
            dsendo = self.ds_me(
                LVendoid,
                domain=self.mesh_me,
                subdomain_data=self.facetboundaries_me,
            )
            self.LVendo_area_me.assign(
                df.assemble(
                    df.Constant(1.0) * dsendo,
                    form_compiler_parameters={"representation": "uflacs"},
                )
            )

            # Update the fiber basis on the updated reference geometry.
            #   "preserve" (DEFAULT): keep the SOURCE mesh's fibers FIXED through the
            #       unloading iterations, so the unloaded reference retains the original
            #       eF/eS/eN field EXACTLY (same per-quadrature-point vectors, same
            #       cell/dof correspondence and orientation). The Sellier backward update
            #       only moves vertices; the topology and the Quadrature:4 dofmap are
            #       invariant, so the loaded f0/s0/n0_me carry through byte-for-byte into
            #       _write_unload_mesh. Mirrors the FCH path (fibers kept FIXED).
            #   "laplace" (opt-in): pure-DOLFIN rule-based LV fibers (no vtk_py3); REPLACES
            #       the source field with a generic +/-60 deg helix (legacy, NOT preserving) —
            #       makes the whole unloading loop independent of the vtk_py3 toolkit.
            #   "ldrb": legacy vtk_py3 LDRB regeneration via get_deformed_basis.
            fiber_update = default_params.get("fiber_update", "preserve")
            if fiber_update == "laplace" and self.isLV:
                f0n, s0n, n0n = generate_lv_fibers_laplace(
                    self.mesh_me,
                    self.facetboundaries_me,
                    lvendoid=self.SimDet["LVendoid"],
                    epiid=self.SimDet["epiid"],
                    baseid=self.SimDet["topid"],
                    fiber_space=self.f0_me.function_space(),
                    deg=self.deg_me,
                    alpha_endo=default_params.get("alpha_endo", 60.0),
                    alpha_epi=default_params.get("alpha_epi", -60.0),
                    comm=comm_me,
                )
                self.f0_me = f0n
                self.s0_me = s0n
                self.n0_me = n0n
            elif fiber_update == "ldrb":
                default_params.update({"meshName": "unloadfiber_" + str(it)})
                (
                    f0_me_Gauss,
                    s0_me_Gauss,
                    n0_me_Gauss,
                    _,
                    _,
                ) = self.get_deformed_basis(default_params)
                self.f0_me = f0_me_Gauss
                self.s0_me = s0_me_Gauss
                self.n0_me = n0_me_Gauss
            else:
                # "preserve": keep the loaded source fibers unchanged (no regeneration);
                # only re-normalize below to guard against round-off drift.
                printout("Fiber update: preserve (source eF/eS/eN kept fixed)", comm_me)
            self._normalize_basis()

            # Restate problem
            self.Ftotal, self.Jac, self.bcs = self.Problem()

            it += 1

        # Write mesh (LV: default dataset names facetboundaries / eF / eS / eN).
        self._write_unload_mesh(outfolder, matid_dataset=matid_dataset)

        if is_root:
            fdataPV.close()

        return it_load, volinc, LVV, self.Solver()

    def _sellier_backward_update(self, targetmesh, relax=1.0, reset_state=True):
        """Apply one Sellier backward-displacement step with iterate relaxation.

        Full Sellier step:  Y = X0 - u(X_k)  (the FULL backward displacement on
        the fixed original reference X0; u is computed on the current moved mesh
        mesh_me, whose CG1 nodal values rebind onto a CG1 space over targetmesh
        since they share topology). Iterate under-relaxation then blends toward Y:

            X_{k+1} = (1 - relax)*X_k + relax*Y = (1-relax)*X_k + relax*(X0 - u)

        The fixed point X* = X0 - u(X*) is independent of `relax` (so the residual
        ||(X*+u(X*)) - X0|| -> 0 for ANY relax in (0,1]); `relax` only damps the
        iteration. This is the CORRECT under-relaxation: scaling the displacement
        instead (X0 - relax*u) would shift the fixed point to X0 - relax*u and
        leave a (1-relax)*||u|| residual bias. relax=1.0 is the plain Sellier step
        (LV default, byte-identical to the previous behavior). Mutates mesh_me
        coordinates + facetboundaries_me and rebuilds the bounding-box tree.
        Shared by the LV and FCH unloaders.
        """
        dispCG1 = df.project(-1.0 * self.get_displacement(), self.V_CG1)
        if getattr(self, "_V_CG1_target", None) is None:
            self._V_CG1_target = df.VectorFunctionSpace(targetmesh, "CG", 1)
        disp_target = df.Function(self._V_CG1_target)
        disp_target.vector().set_local(dispCG1.vector().get_local())
        disp_target.vector().apply("insert")
        newmesh, newboundaries = update_mesh(
            targetmesh, disp_target, self.facetboundaries_me
        )

        # Update mesh: blend current reference X_k with the full backward target Y.
        # reset_state=True (LV) zeroes the solution after the move; FCH passes
        # False to KEEP the loaded state as a warm start for the next solve (the
        # mesh barely moves between iterations, so 1 Newton solve then suffices).
        if reset_state:
            self.Reset()
        Xk = self.mesh_me.coordinates().copy()
        Y = newmesh.coordinates()
        blended = (1.0 - relax) * Xk + relax * Y  # relax=1.0 -> exactly Y
        self.mesh_me.coordinates()[:, 0] = blended[:, 0]
        self.mesh_me.coordinates()[:, 1] = blended[:, 1]
        self.mesh_me.coordinates()[:, 2] = blended[:, 2]

        self.facetboundaries_me.set_values(newboundaries.array())
        self.mesh_me.bounding_box_tree().build(self.mesh_me)
        # Fixed-point residual r = Y - X_k = (X0 - u(X_k)) - X_k, returned flat so
        # the FCH unloader can drive Aitken dynamic relaxation (LV ignores it).
        return (Y - Xk).ravel()

    def _write_unload_mesh(
        self,
        outfolder,
        facet_name="facetboundaries",
        fiber_names=("eF", "eS", "eN"),
        matid_dataset="matid1",
        coord_scale=1.0,
    ):
        """Write the recovered unloaded reference under HDF5 group 'UnloadMesh'.

        Dataset names are parameterized so the LV path keeps its historical names
        (facetboundaries / eF / eS / eN) while the FCH path writes the names its
        loader reads (facetboundaries2 / eF_rb / eS_rb / eN_rb). Fibers are the
        model's current f0/s0/n0 (DG0 for FCH, safe to round-trip through HDF5).
        A byte-identical *_refine copy is emitted for the EP loader. Collective:
        call on all ranks.

        `coord_scale` is the factor the mesh loader RE-APPLIES on read
        (mesh_scale_fch for FCH, mesh_scale_waorta for LV). mesh_me is held in
        post-scale units, so coordinates are written DIVIDED by coord_scale (raw,
        pre-scale units) — re-loading then reproduces the unloaded geometry rather
        than double-scaling it. coord_scale==1.0 (LV default) is an exact no-op,
        keeping the LV output byte-identical. Fibers/markers/matid are
        scale-invariant.
        """
        comm_me = self.mesh_me.mpi_comm()
        path = outfolder + "UnloadMesh.hdf5"
        unscale = abs(coord_scale - 1.0) > 1e-12
        if unscale:
            self.mesh_me.scale(1.0 / coord_scale)
        f = df.HDF5File(comm_me, path, "w")
        f.write(self.mesh_me, "UnloadMesh")
        f.close()

        f = df.HDF5File(comm_me, path, "a")
        f.write(self.facetboundaries_me, "UnloadMesh" + "/" + facet_name)
        f.write(self.edgeboundaries_me, "UnloadMesh" + "/" + "edgeboundaries")
        f.write(self.f0_me, "UnloadMesh" + "/" + fiber_names[0])
        f.write(self.s0_me, "UnloadMesh" + "/" + fiber_names[1])
        f.write(self.n0_me, "UnloadMesh" + "/" + fiber_names[2])

        if hasattr(self.Mesh, "eC0"):
            f.write(self.Mesh.eC0, "UnloadMesh" + "/" + "eC")
        if hasattr(self.Mesh, "eL0"):
            f.write(self.Mesh.eL0, "UnloadMesh" + "/" + "eL")
        if hasattr(self.Mesh, "eR0"):
            f.write(self.Mesh.eR0, "UnloadMesh" + "/" + "eR")

        f.write(self.Mesh.matid, "UnloadMesh" + "/" + matid_dataset)
        f.close()
        if unscale:
            self.mesh_me.scale(coord_scale)  # restore post-scale (in-memory) units

        # RANK-0-ONLY (bug fix): a raw shell `cp` is NOT a collective HDF5 write like the
        # df.HDF5File calls above -- running it on EVERY rank under mpirun -np N launches N
        # concurrent, unsynchronized `cp` processes racing to write the SAME destination file,
        # which surfaced live as "cp: cannot create regular file ...: File exists" under an
        # 8-rank nested unload (calibration/stages/ed_es_stress.py's _unload_reference), leaving
        # a corrupted/incomplete UnloadMesh_refine.hdf5 and a downstream ED-EDP mismatch. This is
        # the exact rank-0-only file-write contract already documented in src/utils/CLAUDE.md
        # ("do not add write calls outside a MPI.rank == 0 guard") and already applied to this
        # same unload path's OTHER writes (ed_es_stress.py's _rank0_write) -- this call was the
        # one write in _write_unload_mesh that had never been guarded. shutil.copyfile (not
        # os.system+cp) avoids the shell-injection surface of building a command string, matching
        # the same-purpose copy already used in unloading/convert.py.
        if df.MPI.rank(comm_me) == 0:
            import shutil
            shutil.copyfile(path, outfolder + "UnloadMesh_refine.hdf5")
        df.MPI.barrier(comm_me)

    def _log_residual_by_region(self, x, xtarget, comm_me, label=""):
        """Localize the Sellier shape residual r=(X+u)-X0 by chamber region (matid).

        The global residual ||(X+u)-xtarget||_2 is a single scalar that cannot tell
        a distributed convergence floor from a chamber-localized artifact (e.g. an
        RV/septum see-saw). This projects per-cell |r| to DG0 and groups by matid
        (lv_rid/rv_rid/la_rid/ra_rid), printing RMS/mean/max per region so the
        oscillating norm can be attributed to LV vs RV vs atria. Used end-of-run
        and (with --debug-residual-by-region) per final-stage iteration. The RV
        region includes the RV side of the septum (matid does not tag the septum
        separately), which is exactly the coupling we want to watch. Diagnostics
        must never break the run (broad except)."""
        try:
            rfun = df.Function(self.V_CG1)
            rfun.vector().set_local((x - xtarget).get_local())
            rfun.vector().apply("insert")
            rmag = df.project(df.sqrt(df.dot(rfun, rfun)), self.QDG)
            dofmap = rmag.function_space().dofmap()
            vals = rmag.vector().get_local()
            mids = self.matid_me.array()
            ncells = self.mesh_me.num_cells()
            percell = np.array(
                [vals[dofmap.cell_dofs(c)[0]] for c in range(ncells)]
            )
            region_names = [
                (int(self.SimDet.get("lv_rid", 10)), "LV"),
                (int(self.SimDet.get("rv_rid", 9)), "RV"),
                (int(self.SimDet.get("la_rid", 11)), "LA"),
                (int(self.SimDet.get("ra_rid", 8)), "RA"),
            ]
            parts = []
            for rid, name in region_names:
                m = mids == rid
                # COLLECTIVES MUST BE UNCONDITIONAL: under MPI a region may have cells
                # on some ranks and not others, so every rank computes a (possibly 0)
                # local contribution and ALL ranks call each reduce -- guarding the
                # reduce behind `if np.any(m)` desynchronizes the allreduce and aborts.
                local_ss = float(np.sum(percell[m] ** 2)) if np.any(m) else 0.0
                local_n = float(np.sum(m))
                local_max = float(np.max(percell[m])) if np.any(m) else 0.0
                ss = df.MPI.sum(comm_me, local_ss)
                nc = df.MPI.sum(comm_me, local_n)
                mx = df.MPI.max(comm_me, local_max)
                if nc > 0:
                    parts.append("%s RMS=%.4g max=%.4g" % (name, (ss / nc) ** 0.5, mx))
            printout("  [res-by-region%s] %s" % (label, " | ".join(parts)), comm_me)
        except Exception as _diag_exc:  # diagnostics must never break the run
            printout("residual localization skipped: %s" % _diag_exc, comm_me)

    def _unloading_lv_pressure(self, params):
        """Sellier unloading for a single LV using prescribed pressure.

        This is the open-base-safe alternative to the historical volume-control
        path.  It uses the same follower pressure load as forward mechanics and
        measures volume only through the authoritative open-base-aware accessor;
        no open-surface volume functional enters the constraint equation.
        """
        default_params = {
            "EDP": 12.0,
            "edp_schedule": None,
            "pressure_inc_mmhg": 0.5,
            "pressure_inc_floor_mmhg": 0.01,
            "maxit": 20,
            "restol": 1e-3,
            "drestol": 1e-4,
            "stage_maxit": 6,
            "stage_restol": 0.1,
            "alpha_lv": 1.0,
        }
        default_params.update(params)
        edp_schedule = sorted({float(x) for x in
                               (default_params.get("edp_schedule") or [default_params["EDP"]])})
        if not edp_schedule:
            edp_schedule = [float(default_params["EDP"])]
        maxit = int(default_params["maxit"])
        restol = float(default_params["restol"])
        drestol = float(default_params["drestol"])
        stage_maxit = int(default_params["stage_maxit"])
        stage_restol = float(default_params["stage_restol"])
        pressure_inc = float(default_params["pressure_inc_mmhg"])
        pressure_floor = float(default_params["pressure_inc_floor_mmhg"])
        relax = float(default_params.get("alpha_lv", 1.0))
        mff = default_params.get("mff")
        mff_max_iter = int(default_params.get("mff_max_iter", maxit))
        mff_it = 0

        outputfolder = self.parameters["outputfolder"]
        folder_name = self.parameters["foldername"].rstrip("/")
        outfolder = os.path.join(outputfolder, folder_name, "deformation_unloadED")
        targetmesh = df.Mesh(self.Mesh.mesh)
        xtarget = df.project(df.SpatialCoordinate(self.mesh_me), self.V_CG1).vector()
        comm_me = self.mesh_me.mpi_comm()
        is_root = df.MPI.rank(comm_me) == 0
        if is_root and not os.path.isdir(outfolder):
            os.makedirs(outfolder)
        fpv = open(outfolder + "BiV_unloadPV.txt", "w", buffering=1) if is_root else None

        def sample(iteration):
            pv = (self.get_lv_pressure() * 0.0075, self.get_lv_volume())
            if is_root:
                print(iteration, pv[0], pv[1], file=fpv)
            return pv

        def ramp_to(target_mmhg, iteration):
            self.Reset()
            self.LVCavitypres.assign(0.0)
            curve = [sample(iteration)]
            current = 0.0
            step = pressure_inc
            naccepted = 0
            while current < target_mmhg - 1e-10:
                trial = min(current + step, target_mmhg)
                w_prev = self.w_me.vector().copy()
                self.LVCavitypres.assign(trial / 0.0075)
                solve_failed = 0
                try:
                    self.Solver().solvenonlinear()
                except RuntimeError:
                    solve_failed = 1
                solve_failed = int(df.MPI.sum(comm_me, solve_failed))
                if solve_failed:
                    self.w_me.vector()[:] = w_prev
                    self.w_me.vector().apply("insert")
                    self.LVCavitypres.assign(current / 0.0075)
                    step *= 0.5
                    printout("LV pressure unload: solve failed; halving pressure step -> %g mmHg"
                             % step, comm_me)
                    if step < pressure_floor:
                        raise UnloadingConvergenceError(
                            "LV pressure-control ramp diverged: pressure step fell below "
                            "%g mmHg at P=%g mmHg toward EDP=%g mmHg"
                            % (pressure_floor, current, target_mmhg)
                        )
                    continue
                current = trial
                naccepted += 1
                curve.append(sample(iteration))

            # Passive pressure inflation must have a finite, nondecreasing P-V
            # response.  A decrease larger than assembly noise is a physical-QC
            # failure, not a continuation success.
            for (p0, v0), (p1, v1) in zip(curve, curve[1:]):
                tol_ml = max(0.1, 1e-3 * abs(v0))
                if (not np.isfinite(v1)) or v1 < v0 - tol_ml:
                    raise UnloadingConvergenceError(
                        "LV pressure-control produced non-monotone passive P-V: "
                        "(%g mmHg,%g mL) -> (%g mmHg,%g mL)" % (p0, v0, p1, v1)
                    )
            return curve, naccepted

        stage_i = 0
        it = 0
        it_stage = 0
        res = 1e9
        dres = 0.0
        last_naccepted = 0
        printout("=== LV unloading: pressure-control continuation %s mmHg ==="
                 % edp_schedule, comm_me)
        while True:
            is_final = stage_i == len(edp_schedule) - 1
            cur_tol = restol if is_final else stage_restol
            cur_maxit = maxit if is_final else stage_maxit
            # Keep evaluating MFF until material *and* shape converge on the
            # same iterate.  Freezing material on its first converged update can
            # let subsequent Sellier settlement move V0 back outside tolerance.
            mff_active = mff is not None and is_final and mff_it < mff_max_iter
            curve, last_naccepted = ramp_to(edp_schedule[stage_i], it)

            x = df.project(df.SpatialCoordinate(self.mesh_me) + self.get_displacement(),
                           self.V_CG1).vector()
            res_new = (x - xtarget).norm("l2")
            dres = abs(res_new - res)
            res = res_new
            printout("LV pressure unload: it=%d stage=%d P=%g V=%g residual=%g dres=%g"
                     % (it, stage_i, curve[-1][0], curve[-1][1], res, dres), comm_me,
                     stage="unloading", it=it, stage_i=stage_i,
                     p_mmhg=curve[-1][0], v_ml=curve[-1][1], residual=res, dres=dres)
            it_stage += 1
            mesh_converged = res < cur_tol or dres < drestol

            mff_converged = mff is None or not is_final or mff_it >= mff_max_iter
            if mff_active:
                mff_it += 1
                result = mff(curve, it)
                mff_converged = bool(result.get("converged"))
                if not mff_converged and mff_it >= mff_max_iter:
                    printout(
                        "MFF: reached mff_max_iter=%d without simultaneous material/shape "
                        "convergence; freezing material so anchor QC can classify the result"
                        % mff_max_iter,
                        comm_me,
                    )
                    mff_converged = True

            stage_complete = it_stage >= cur_maxit or (mesh_converged and mff_converged)
            if stage_complete:
                if is_final:
                    if not mesh_converged:
                        raise UnloadingConvergenceError(
                            "LV pressure-control Sellier iteration reached maxit=%d "
                            "without convergence (res=%g, dres=%g)" % (maxit, res, dres)
                        )
                    self.Reset()
                    break
                stage_i += 1
                it_stage = 0
                res = 1e9
                dres = 0.0
                it += 1
                self.Reset()
                continue

            self._sellier_backward_update(targetmesh, relax)
            self._normalize_basis()
            self.Ftotal, self.Jac, self.bcs = self.Problem()
            it += 1

        self._write_unload_mesh(
            outfolder, matid_dataset=self.SimDet.get("matid_dataset", "matid1")
        )
        if is_root:
            fpv.close()
        return last_naccepted, pressure_inc, curve[-1][1], self.Solver()

    def _unloading_fch(self, params):
        """Pressure-control continuation unloading for the four-chamber mesh.

        Mirrors the LV `unloading` philosophy but, because FCH runs pressure-
        controlled (ispctrl=True, four cavity-pressure Constants), drives the
        four cavities toward per-chamber EDP *pressure* targets instead of
        incrementing one cavity volume. Per Sellier iteration the (current
        reference) mesh is re-loaded to the stage's scaled pressure targets via
        an adaptive sub-stepped ramp (halving on solve failure, like the LV inner
        loop and run_light's loading), the L2 shape residual ||(X+u) - xtarget||
        is measured, convergence/staging is tested, and the mesh is updated by
        the shared Sellier backward step. Fibers are kept FIXED (the validated
        eF_rb/eS_rb/eN_rb transfer unchanged with the deforming reference; no
        vtk_py3 LDRB / Laplace regen). Output datasets use the FCH-configured
        names so the result is a drop-in fch_clregion_whole.hdf5.
        """
        default_params = {
            "edp_lv": 8.0,
            "edp_rv": 4.0,
            "edp_la": 8.0,
            "edp_ra": 4.0,
            "edp_schedule": [0.5, 1.0],
            # Optional PER-CHAMBER staged continuation: a list of dicts of
            # per-chamber fractions, e.g. [{"LV":1,"LA":1,"RV":0,"RA":0},
            # {"LV":1,"LA":1,"RV":1,"RA":1}] to unload LV/LA first and bring the
            # RV/RA up against an already-settled septum (reduces the LV/RV/septum
            # coupling that limit-cycles a simultaneous unload). When set it
            # overrides the uniform edp_schedule. The final stage is forced to
            # all-ones (full EDP) so the recovered reference re-inflates to ED.
            "chamber_schedule": None,
            "nsub": 8,
            "maxit": 20,
            "restol": 1e-3,
            "drestol": 1e-4,
            "stage_maxit": 6,
            "stage_restol": 0.1,
            "substep_floor": 1e-3,
            # Sellier ITERATE relaxation X_{k+1}=(1-w)*X_k+w*(X0-u): the relaxation
            # factor w is adapted per iteration by Aitken (Irons-Tuck) dynamic
            # relaxation, since a CONSTANT w converges only linearly here (the
            # coupled LV/RV/septum mode gives a slow damped oscillation; w=1 even
            # limit-cycles). `alpha` is the INITIAL w (first two iterations) before
            # Aitken takes over; bounds clamp the adapted w for stability. Relaxing
            # the iterate (not the displacement) keeps the fixed point X*=X0-u(X*)
            # for any w, so the residual still -> 0. LV defaults to fixed w=1
            # (unload.alpha_lv, aitken_lv=0); the same Aitken machinery is
            # available opt-in via unload.aitken_lv (aitken_omega_step, shared).
            "alpha": 0.5,
            "omega_min": 0.05,
            "omega_max": 1.0,
            # final_fixed_omega: when >0, the FINAL all-chamber stage holds the
            # Sellier relaxation at this CONSTANT value and disables Aitken
            # re-adaptation. The four-chamber final stage carries a standing
            # LV/RV trans-septal pressure gradient that drives an anti-correlated
            # LV<->RV cavity-volume see-saw (a 2-cycle) on the shared septum; a
            # single GLOBAL Aitken omega (computed from the LV-dominated whole-mesh
            # residual) cannot damp that chamber-localized mode -- it oscillates
            # omega between omega_min and omega_max and RE-EXCITES the see-saw every
            # time it adapts back up. A small constant omega makes the relaxed
            # oscillatory eigenvalue contract and the 2-cycle decays. 0 = legacy
            # (Aitken on every stage); use a small value (~0.08) for soft/thin-RV
            # FCH meshes where the final stage limit-cycles.
            "final_fixed_omega": 0.0,
            # final_average: when True, the FINAL stage replaces each new reference
            # with the AVERAGE of the previous and the just-computed reference
            # (Krasnoselskij/Mann iterate averaging). For a residual 2-cycle X_a<->X_b
            # the fixed point IS the midpoint (X_a+X_b)/2, so averaging converges it
            # DETERMINISTICALLY where a small fixed omega only marginally damps the
            # chamber-localized septum see-saw. Combine with final_fixed_omega.
            "final_average": False,
            # debug_residual_by_region: when True, log the per-chamber (matid) Sellier
            # residual EVERY final-stage iteration (not just at the end) so an
            # oscillating global norm can be attributed to LV vs RV vs atria -- the
            # discriminator for the unload-oscillation diagnosis. Default off.
            "debug_residual_by_region": False,
        }
        default_params.update(params)

        mmHg_to_Pa = 1.0 / 0.0075
        targets = {
            "LV": float(default_params["edp_lv"]) * mmHg_to_Pa,
            "RV": float(default_params["edp_rv"]) * mmHg_to_Pa,
            "LA": float(default_params["edp_la"]) * mmHg_to_Pa,
            "RA": float(default_params["edp_ra"]) * mmHg_to_Pa,
        }
        # Build the continuation as a list of PER-CHAMBER fraction dicts (`stages`).
        # Uniform edp_schedule -> the same fraction on all four chambers each stage;
        # an explicit chamber_schedule -> per-chamber fractions (staged unloading).
        chambers = ("LV", "RV", "LA", "RA")
        cs = default_params.get("chamber_schedule")
        if cs:
            stages = [{c: float(d.get(c, 0.0)) for c in chambers} for d in cs]
        else:
            fr = sorted({float(x) for x in default_params["edp_schedule"]})
            if not fr:
                fr = [1.0]
            stages = [{c: f for c in chambers} for f in fr]
        # The recovered reference must re-inflate to the full per-chamber EDP, so the
        # last stage must be all-ones; append it if the schedule doesn't end there.
        if any(abs(stages[-1][c] - 1.0) > 1e-9 for c in chambers):
            stages.append({c: 1.0 for c in chambers})
        nsub = int(default_params["nsub"])
        maxit = int(default_params["maxit"])
        restol = float(default_params["restol"])
        drestol = float(default_params["drestol"])
        stage_maxit = int(default_params["stage_maxit"])
        stage_restol = float(default_params["stage_restol"])
        substep_floor = float(default_params["substep_floor"])

        # MFF (Marx 2022) coupled material co-fit hook -- the FCH pressure-control port of the LV
        # `unloading()` hook. When `mff` is provided it is invoked once per FINAL-stage outer
        # iteration with the swept chamber's simulated loading PV curve [(P_mmHg, V_mL), ...]; it
        # fits the Klotz model function and RESCALES that chamber's per-region passive material in
        # SimDet (Cparam_<ch>/bff_<ch>/bfx_<ch>/bxx_<ch>, read fresh by Problem() on the next
        # rebuild) and returns {"converged": bool}. `mff_chamber` selects which cavity's curve is
        # sampled (LV default; RV opt-in). None (every legacy/byte-identical FCH caller) => no curve
        # accumulation, no extra termination condition, Aitken unchanged.
        mff = default_params.get("mff")
        mff_max_iter = int(default_params.get("mff_max_iter", maxit))
        mff_chamber = str(default_params.get("mff_chamber", "LV")).upper()
        mff_it = 0
        mff_done = False  # set once scalings converge (or the backstop fires): freeze material

        matid_dataset = self.SimDet.get("matid_dataset", "matid1")
        facet_name = self.SimDet.get("facetboundaries_dataset", "facetboundaries")
        fd = self.SimDet.get("fiber_datasets") or {}
        fiber_names = (fd.get("f0", "eF"), fd.get("s0", "eS"), fd.get("n0", "eN"))
        # The FCH loader (fch_mesh) re-scales coords by FCH_MESH_SCALE env or
        # mesh_scale_fch on read; write raw (pre-scale) coords so reloading the
        # unloaded mesh reproduces this geometry instead of double-scaling.
        _ms_env = os.environ.get("FCH_MESH_SCALE")
        try:
            _ms_env = float(_ms_env) if _ms_env is not None else None
        except (TypeError, ValueError):
            _ms_env = None
        coord_scale = _ms_env if _ms_env is not None else float(self.SimDet.get("mesh_scale_fch", 1.0))

        outputfolder = self.parameters["outputfolder"]
        folderName = self.parameters["foldername"].rstrip("/")
        outfolder = os.path.join(outputfolder, folderName, "deformation_unloadED")

        targetmesh = df.Mesh(self.Mesh.mesh)
        xtarget = df.project(df.SpatialCoordinate(self.mesh_me), self.V_CG1).vector()

        comm_me = self.mesh_me.mpi_comm()
        is_root = df.MPI.rank(comm_me) == 0
        if is_root and not os.path.isdir(outfolder):
            os.makedirs(outfolder)
        fdataPV = open(outfolder + "FCH_unloadPV.txt", "w", buffering=1) if is_root else None

        def assign_pressures(p):
            self.LVCavitypres.assign(p["LV"])
            self.RVCavitypres.assign(p["RV"])
            self.LACavitypres.assign(p["LA"])
            self.RACavitypres.assign(p["RA"])

        def _mff_chamber_pv():
            """(P_mmHg, V_mL) of the MFF-swept chamber at the current state (for the model-function
            fit). Both getters are globally-reduced (imposed pressure Constant + assembled cavity
            integral) -> identical on every rank -> the rescale is bit-identical across ranks."""
            if mff_chamber == "RV":
                return self.get_rv_pressure() * 0.0075, self.get_rv_volume()
            return self.get_lv_pressure() * 0.0075, self.get_lv_volume()

        def ramp_to(stage_targets, pv_accum=None):
            # Sub-stepped pressure ramp from the zero-displacement reference up to
            # stage_targets, with adaptive backoff (halve the sub-step and roll back
            # on a Newton/PETSc failure). Re-solving from the reference each Sellier
            # iteration is robust (a warm-start from the previous loaded state on the
            # just-moved mesh was tried and reliably diverged here). When `pv_accum` is
            # a list (MFF final-stage only), append the swept chamber's (P,V) at each
            # successful sub-step (seeded with the zero-pressure reference point, whose
            # volume is the simulated unloaded V0 that normalizes the model-function fit).
            self.Reset()  # start the ramp from the zero-displacement reference
            assign_pressures({k: 0.0 for k in stage_targets})
            if pv_accum is not None:
                pv_accum.append(_mff_chamber_pv())
            sfrac = 0.0
            sub = 1.0 / nsub
            while sfrac < 1.0 - 1e-9:
                w_prev = self.w_me.vector().copy()
                s_try = min(sfrac + sub, 1.0)
                assign_pressures({k: s_try * v for k, v in stage_targets.items()})
                try:
                    self.Solver().solvenonlinear()
                    solve_ok = True
                except RuntimeError:
                    solve_ok = False
                if not solve_ok:
                    self.w_me.vector()[:] = w_prev
                    self.w_me.vector().apply("insert")
                    sub *= 0.5
                    printout(
                        "FCH unload: solve failed; halving substep -> " + str(sub),
                        comm_me,
                    )
                    if sub < substep_floor:
                        raise RuntimeError(
                            "FCH unloading: pressure ramp diverged; substep fell "
                            "below floor %g (targets[mmHg] lv=%g rv=%g la=%g ra=%g). "
                            "Try lower per-chamber EDP or more continuation stages."
                            % (
                                substep_floor,
                                default_params["edp_lv"],
                                default_params["edp_rv"],
                                default_params["edp_la"],
                                default_params["edp_ra"],
                            )
                        )
                    continue
                sfrac = s_try  # keep the (possibly reduced) sub-step on success
                if pv_accum is not None:
                    pv_accum.append(_mff_chamber_pv())

        def log_chambers(iteration):
            pv = (
                iteration,
                self.get_lv_pressure() * 0.0075,
                self.get_lv_volume(),
                self.get_rv_pressure() * 0.0075,
                self.get_rv_volume(),
                self.get_la_pressure() * 0.0075,
                self.get_la_volume(),
                self.get_ra_pressure() * 0.0075,
                self.get_ra_volume(),
            )
            printout(
                "it=%d P[mmHg] LV=%.3f RV=%.3f LA=%.3f RA=%.3f | V[mL] "
                "LV=%.2f RV=%.2f LA=%.2f RA=%.2f"
                % (pv[0], pv[1], pv[3], pv[5], pv[7], pv[2], pv[4], pv[6], pv[8]),
                comm_me,
            )
            if is_root:
                print(*pv, file=fdataPV)

        def stage_target_dict(i):
            return {c: stages[i][c] * targets[c] for c in chambers}

        omega_init = float(default_params["alpha"])
        omega_min = float(default_params["omega_min"])
        omega_max = float(default_params["omega_max"])
        final_fixed_omega = float(default_params["final_fixed_omega"])
        final_average = bool(default_params["final_average"])
        debug_residual_by_region = bool(default_params["debug_residual_by_region"])

        def aitken_omega(omega_prev, r_prev, r_cur):
            return aitken_omega_step(omega_prev, r_prev, r_cur, omega_min, omega_max, comm_me)

        it = 0
        res = 1e9
        dres = 0.0
        omega = omega_init
        r_prev = None  # previous Sellier residual vector (for Aitken)
        final_converged = False
        stage_i = 0
        is_final_stage = stage_i == len(stages) - 1
        cur_restol = restol if is_final_stage else stage_restol
        cur_maxit = maxit if is_final_stage else stage_maxit
        it_stage = 0
        printout("=== FCH unloading: pressure-control continuation ===", comm_me)
        printout(
            "targets[mmHg] LV=%g RV=%g LA=%g RA=%g ; %d-stage schedule = %s"
            % (
                default_params["edp_lv"],
                default_params["edp_rv"],
                default_params["edp_la"],
                default_params["edp_ra"],
                len(stages),
                stages,
            ),
            comm_me,
        )
        printout(
            "--- stage %d: fracs=%s (restol=%g, maxit=%d) ---"
            % (stage_i, stages[stage_i], cur_restol, cur_maxit),
            comm_me,
        )

        while True:
            # MFF co-fit: accumulate the swept chamber's simulated loading PV curve during this
            # final-stage ramp (None on non-final stages / when no hook => no overhead, no behavior
            # change). mff_active also gates the Aitken hold below.
            mff_active = (mff is not None and is_final_stage and not mff_done)
            pv_iter = [] if mff_active else None
            # Re-load the (current reference) mesh to this stage's pressure targets.
            ramp_to(stage_target_dict(stage_i), pv_accum=pv_iter)

            x = df.project(
                df.SpatialCoordinate(self.mesh_me) + self.get_displacement(),
                self.V_CG1,
            ).vector()
            res_new = (x - xtarget).norm("l2")
            dres = abs(res_new - res)
            res = res_new
            log_chambers(it)
            # Parity with the LV variant above, which reports it/stage alongside the residuals.
            # Without them an FCH unload log gave no way to tell WHICH outer iteration or
            # continuation stage a residual belonged to.
            printout("FCH pressure unload: it=%d stage=%d residual=%g dres=%g"
                     % (it, stage_i, res, dres), comm_me, stage="unloading",
                     it=it, stage_i=stage_i, residual=res, dres=dres)
            if debug_residual_by_region and is_final_stage:
                self._log_residual_by_region(x, xtarget, comm_me, label=":it%d" % it)

            it_stage += 1
            # MFF (Marx 2022) coupled material co-fit: ONLY on the final stage, once per outer
            # iteration after the ramp. The hook fits the simulated PV curve to the Klotz model
            # function and RESCALES the swept chamber's passive material in SimDet; the Sellier
            # move + Problem() rebuild below pick up the new material. The final break waits until
            # the scalings have converged (mff_converged) AND the mesh residual has settled, bounded
            # by the mff_max_iter backstop. mff None (every legacy caller) => mff_converged stays
            # True => byte-identical termination.
            mff_converged = True
            if mff_active:
                mff_it += 1
                mff_result = mff(pv_iter, it)
                if bool(mff_result.get("converged")):
                    mff_done = True  # scalings ~ 1: freeze material, let the mesh settle
                elif mff_it >= mff_max_iter:
                    printout(
                        "MFF: reached mff_max_iter=%d without scaling convergence; freezing "
                        "material (anchor QC reflects the V0 deviation)" % mff_max_iter,
                        comm_me,
                    )
                    mff_done = True  # backstop: stop rescaling
                else:
                    mff_converged = False  # keep co-iterating (Sellier move + rescale)
            mesh_converged = res < cur_restol or dres < drestol
            stage_converged = (
                it_stage >= cur_maxit or (mesh_converged and mff_converged)
            )
            if stage_converged:
                if is_final_stage:
                    # Did we converge on tolerance, or merely hit the iteration cap?
                    final_converged = (res < cur_restol) or (dres < drestol)
                    self.Reset()
                    break
                stage_i += 1
                is_final_stage = stage_i == len(stages) - 1
                cur_restol = restol if is_final_stage else stage_restol
                cur_maxit = maxit if is_final_stage else stage_maxit
                res = 1e9
                dres = 0.0
                it_stage = 0
                it += 1
                omega = omega_init  # restart Aitken: the residual changes meaning
                r_prev = None
                printout(
                    "--- stage %d: fracs=%s (restol=%g, maxit=%d) ---"
                    % (stage_i, stages[stage_i], cur_restol, cur_maxit),
                    comm_me,
                )
                continue

            # Sellier backward update (fibers kept FIXED, no regen). Apply the
            # current Aitken-relaxed step, then adapt omega from the returned
            # residual for the next iteration (one-step-lagged Aitken).
            # On the final all-chamber stage, an explicit final_fixed_omega>0
            # pins omega constant and SKIPS Aitken: the LV/RV/septum see-saw is a
            # chamber-localized 2-cycle that the global-residual Aitken cannot
            # damp (it re-excites it), so a fixed small omega is required to
            # contract the oscillatory mode (see default_params note above).
            fix_omega = is_final_stage and final_fixed_omega > 0.0
            if fix_omega:
                omega = final_fixed_omega
            elif mff_active:
                # While the MFF co-fit is still rescaling the material, the residual changes meaning
                # between iterations, so hold omega at its initial value and do NOT adapt/keep Aitken
                # across a material change -- Aitken restarts cleanly once the material freezes (the
                # LV path sidesteps this entirely with fixed w=1). Byte-identical when mff is None.
                omega = omega_init
            do_avg = is_final_stage and final_average
            X_pre = self.mesh_me.coordinates().copy() if do_avg else None
            r_cur = self._sellier_backward_update(targetmesh, omega)
            if do_avg:
                # Krasnoselskij/Mann averaging: X <- 0.5*(X_pre + X_new). Kills a
                # residual 2-cycle (its midpoint is the fixed point) that fixed-omega
                # only marginally damps. Topology/markers unchanged -> only rebuild
                # the bounding-box tree; facetboundaries_me stays valid.
                X_new = self.mesh_me.coordinates()
                avg = 0.5 * (X_pre + X_new)
                for c in range(3):
                    self.mesh_me.coordinates()[:, c] = avg[:, c]
                self.mesh_me.bounding_box_tree().build(self.mesh_me)
            if (not fix_omega) and (not mff_active) and r_prev is not None:
                omega = aitken_omega(omega, r_prev, r_cur)
            r_prev = None if mff_active else r_cur
            printout(
                "Aitken omega -> %g%s%s"
                % (omega, " (fixed, final stage)" if fix_omega else "",
                   " +avg" if do_avg else ""),
                comm_me,
            )
            self.Ftotal, self.Jac, self.bcs = self.Problem()
            it += 1

        if not final_converged:
            # Loud, explicit non-convergence signal -- never silently write a
            # maxit-capped reference as if it were converged.
            printout(
                "WARNING: FCH unloading did NOT meet tolerance (final residual=%g "
                ">= restol=%g, dres>=drestol=%g) after maxit=%d (last Aitken omega=%g). "
                "The recovered mesh is NOT a converged unloaded reference. Raise "
                "--maxit, lower --alpha (initial omega), or add continuation stages."
                % (res, restol, drestol, maxit, omega),
                comm_me,
            )

        # Diagnostic: localize the final re-inflation residual r=(X+u)-X0 by chamber
        # region (matid), to tell a distributed floor from a localized artifact.
        printout("--- residual localization (per-cell |X+u-X0|, scaled units) ---", comm_me)
        self._log_residual_by_region(x, xtarget, comm_me, label=":final")

        self._write_unload_mesh(
            outfolder,
            facet_name=facet_name,
            fiber_names=fiber_names,
            matid_dataset=matid_dataset,
            coord_scale=coord_scale,
        )
        if is_root:
            fdataPV.close()

        volumes = {
            "LV": self.get_lv_volume(),
            "RV": self.get_rv_volume(),
            "LA": self.get_la_volume(),
            "RA": self.get_ra_volume(),
        }
        return it, res, volumes, final_converged

    def set_BCs(self):
        bc_specs = self.SimDet.get("dirichlet_bcs")
        if bc_specs:
            # Optional: SimDet-driven Dirichlet BCs (useful for parameter sweeps without code changes).
            custom_bcs = self._configured_dirichlet_bcs(bc_specs)
            if custom_bcs:
                return custom_bcs
        return self._default_dirichlet_bcs()
        #  - - - - - - - - - - - -- - - - - - - - - - - - - - - -- - - - - - -

    def _configured_dirichlet_bcs(self, bc_specs):
        facetboundaries = self.facetboundaries_me
        bcs = []

        for spec in bc_specs:
            if not spec or not spec.get("enabled", True):
                continue

            subspace = self._resolve_dirichlet_subspace(spec.get("subspace", spec.get("component", 0)))
            value = self._build_dirichlet_value(spec)
            facet_ids = spec.get("facet_ids") or spec.get("boundary_ids") or spec.get("boundary_id")
            if facet_ids is None:
                raise ValueError("Dirichlet BC specification missing 'facet_ids'.")

            if not isinstance(facet_ids, (list, tuple, set)):
                facet_ids = [facet_ids]

            for boundary_id in facet_ids:
                bcs.append(df.DirichletBC(subspace, value, facetboundaries, int(boundary_id)))

        return bcs

    def _resolve_dirichlet_subspace(self, descriptor):
        if descriptor is None:
            descriptor = 0

        if isinstance(descriptor, int):
            return self.W.sub(descriptor)

        if isinstance(descriptor, str):
            descriptor = descriptor.lower()
            displacement_aliases = {"u", "displacement", "vector", "mechanics"}
            if descriptor in displacement_aliases:
                return self.W.sub(0)

        raise ValueError(f"Unsupported Dirichlet BC subspace descriptor '{descriptor}'.")

    def _build_dirichlet_value(self, spec):
        if "value_expr" in spec:
            return spec["value_expr"]

        value = spec.get("value")
        if value is None:
            return self._zero_vector_expr()

        if hasattr(value, "ufl_shape"):
            return value

        if isinstance(value, str):
            if value.lower() in ("zero", "zero_vector", "zero_vec"):
                return self._zero_vector_expr()
            raise ValueError(f"Unsupported Dirichlet BC value descriptor '{value}'.")

        if isinstance(value, (list, tuple)):
            return df.Constant(tuple(value))

        if isinstance(value, (int, float)):
            return df.Constant(value)

        return value

    def _default_dirichlet_bcs(self):
        facetboundaries = self.facetboundaries_me
        W = self.W
        zero_vec = self._zero_vector_expr()

        def dirichlet_zero(boundary_id):
            return df.DirichletBC(W.sub(0), zero_vec, facetboundaries, boundary_id)

        topid = self.SimDet.get("topid")
        bc_fix = None

        if self.iswaorta:
            # waorta ring clamping is handled as a dedicated helper to keep waorta conditionals localized.
            return dirichlet_bcs_waorta(W, facetboundaries, self.SimDet, zero_vec)

        elif self.isLV or self.isBiV:
            bctop = dirichlet_zero(topid)
            fix_surfaces = self.SimDet.get("fix_surf")
            bc_fix = None
            if isinstance(fix_surfaces, list):
                bc_fix = [dirichlet_zero(boundary_id) for boundary_id in fix_surfaces]
            elif fix_surfaces is not None:
                bc_fix = dirichlet_zero(fix_surfaces)

        elif self.isFCH:
            aorta_wall = self.SimDet["aorta_wall"]
            bc_aorta_wall = dirichlet_zero(aorta_wall)

        def extend_fix(bc_list):
            if bc_fix is None:
                return
            if isinstance(bc_fix, list):
                bc_list.extend(bc_fix)
            else:
                bc_list.append(bc_fix)

        springbc = bool(self.SimDet.get("springbc"))
        # free_top: traction-free basal/top support (no spring AND no base
        # Dirichlet); the LV is held only by the rigid-body multipliers (VRelem,
        # included whenever springbc is off). Used by the free-base lv_baseline
        # variant. Default off -> existing behavior unchanged.
        free_top = bool(self.SimDet.get("free_top"))

        def build_bcs():
            if self.isLV or self.isBiV:
                base = [] if (springbc or free_top) else [bctop]
                extend_fix(base)
                return base
            if self.isFCH:
                # aorta_cutface_robin: the truncated aorta cut-rim is held by a MODEST omni-
                # directional Robin spring (spring_bc_forms._build_fch_form) instead of this hard
                # u=0 clamp -> drop the Dirichlet so the two never both act (no over-constraint).
                if bool(self.SimDet.get("aorta_cutface_robin")):
                    return []
                return [bc_aorta_wall] if springbc else []
            return []

        return build_bcs()

    # --- Per-chamber passive single-coefficient resolver -------------------------
    # The passive Cparam series resolves each mesh region to ONE absolute Guccione
    # Cparam: a per-chamber override (``Cparam_{lv,rv,la,ra}``) or the single documented
    # fallback (``GiccioneParams["Passive params"]["Cparam"]``). Shared by the per-region
    # passive loop AND the collapse fast path in ``Problem`` so no context reads a bare
    # base (the SEF is linear in Cparam, so the absolute value enters as the scale
    # Cparam_region/Cparam_fallback on the single Wp_me built at the fallback).
    def _passive_cparam_fallback(self):
        """The single documented fallback Cparam (Pa)."""
        return float(self.SimDet["GiccioneParams"]["Passive params"]["Cparam"])

    def _fch_passive_cparam_keymap(self):
        """Region-id -> per-chamber absolute-Cparam SimDet key. Non-chamber supporting
        regions inherit the most relevant chamber's Cparam (aorta wall -> LV, passive
        bulk -> RV); configurable + backward-compatible (absent -> fallback ratio 1.0)."""
        lv_rid = self.SimDet.get("lv_rid", 10)
        rv_rid = self.SimDet.get("rv_rid", 9)
        la_rid = self.SimDet.get("la_rid", 11)
        ra_rid = self.SimDet.get("ra_rid", 8)
        keymap = {lv_rid: "Cparam_lv", rv_rid: "Cparam_rv",
                  la_rid: "Cparam_la", ra_rid: "Cparam_ra"}
        aw_rid = self.SimDet.get("aorta_wall")
        if aw_rid is not None:
            keymap.setdefault(int(aw_rid),
                              self.SimDet.get("aorta_wall_cparam_key", "Cparam_lv"))
        pb_rid = self.SimDet.get("passive_bulk_rid", 12)
        keymap.setdefault(int(pb_rid),
                          self.SimDet.get("passive_bulk_cparam_key", "Cparam_rv"))
        return keymap

    def _region_passive_cparam(self, rid, keymap=None):
        """ABSOLUTE Guccione Cparam (Pa) for a mesh region id: the explicit per-REGION map, then
        the per-chamber override, else the documented fallback. ``rid=None`` -> fallback."""
        fallback = self._passive_cparam_fallback()
        if rid is None:
            return fallback
        # Explicit per-region absolute Cparam, the passive sibling of SimDet["Tmax_region"].
        # Consulted BEFORE the chamber keymap because sub-chamber zone ids (a regional LV
        # partition: 0 = remote, 1 = infarct) match no chamber and would otherwise silently
        # resolve to the fallback -- i.e. a requested regional stiffness contrast would be
        # assembled as uniform. Absent (every FCH/LV run today) -> unchanged.
        region_cparam = self.SimDet.get("Cparam_region")
        if region_cparam:
            val = dict(region_cparam).get(int(rid), dict(region_cparam).get(str(int(rid))))
            if val is not None:
                return float(val)
        keymap = self._fch_passive_cparam_keymap() if keymap is None else keymap
        key = keymap.get(rid)
        if key is None or self.SimDet.get(key) is None:
            return fallback
        return float(self.SimDet[key])

    def _region_passive_scale(self, rid, keymap=None):
        """Cparam_region / Cparam_fallback -- the multiplier on the single Wp_me (built at
        the fallback) that realizes the region's ABSOLUTE Cparam (SEF linear in Cparam).

        HO-aware: the Holzapfel-Ogden SEF has no single `Cparam`, but it IS linear in a global
        multiplier on the a-coefficients (a,a_f,a_s,a_fs) -- W(s*a_*) = s*W(a_*) -- so a per-chamber
        HO stiffness SCALE multiplies the single Wp_me (built at the base HO params) exactly like the
        Guccione Cparam ratio. The scale is `ho_stiffness_scale_<ch>` (default 1.0 = base HO); the
        unload's Klotz MFF co-fit drives it (mirrors the Cparam fit). `rid=None`/absent -> 1.0."""
        if str(self.SimDet["GiccioneParams"]["Passive model"].get("Name")) == "HolzapfelOgden":
            if rid is None:
                return 1.0
            keymap = self._fch_passive_cparam_keymap() if keymap is None else keymap
            ckey = keymap.get(rid)  # e.g. "Cparam_lv"
            if ckey is None:
                # A sub-chamber ZONE id (calibration.regions: 0 = remote, 1 = infarct,
                # 2 = border) matches no chamber. Under Guccione those resolve through
                # SimDet["Cparam_region"] (see _region_passive_cparam); the HO branch has no
                # equivalent channel, so a requested regional stiffness contrast would assemble
                # UNIFORM with nothing said -- the exact un-warned degradation CLAUDE.md's
                # error-handling principle forbids. Refuse only when this rid was ACTUALLY
                # requested; an HO run with no Cparam_region is unaffected.
                _creg = self.SimDet.get("Cparam_region") or {}
                if _creg and (int(rid) in {int(k) for k in _creg}):
                    raise ValueError(
                        "SimDet['Cparam_region'] requests an absolute Cparam for region %d, but "
                        "the passive model is Holzapfel-Ogden, which resolves per-region "
                        "stiffness only through the per-CHAMBER ho_stiffness_scale_<ch> keys. "
                        "Region %d matches no chamber, so the request would be silently dropped "
                        "and the stiffness contrast assembled UNIFORM. Use Guccione for a "
                        "regional passive contrast, or add an HO per-region scale channel."
                        % (int(rid), int(rid)))
                return 1.0
            skey = "ho_stiffness_scale_%s" % ckey.split("_")[-1]  # -> ho_stiffness_scale_lv
            s = self.SimDet.get(skey)
            return float(s) if s is not None else 1.0
        return self._region_passive_cparam(rid, keymap) / self._passive_cparam_fallback()

    def _regional_tmax_values(self, field, values, base):
        """Fill a DG0 cell field from a {matid: value} map, `base` where a region is absent."""
        dm = field.function_space().dofmap()
        mid = self.matid_me.array()
        vec = field.vector().get_local()
        for c in df.cells(self.mesh_me):
            vec[dm.cell_dofs(c.index())[0]] = values.get(int(mid[c.index()]), base)
        field.vector().set_local(vec)
        field.vector().apply("insert")

    def _active_tmax_field(self):
        """Per-REGION active contractility Tmax as a DG0 field over ``matid`` -- the REGIONAL
        generalization of the single scalar/Constant Tmax the global ED/ES inverse fits.

        OPT-IN via ``SimDet["Tmax_region"] = {matid: Tmax_Pa}``. Absent (the default) returns
        None and the caller leaves the existing scalar/Constant Tmax untouched, so the global
        path is byte-identical -- a DG0 coefficient and a Constant generate different code, so
        the regional field is built ONLY when regions are actually requested. A region with no
        entry keeps the base Tmax.

        Mirrors ``_active_strain_gamma_field`` (same DG0 + dofmap + matid.array() construction).
        Like the Constant it replaces, a DG0 coefficient is refillable BETWEEN solves without
        forcing an FFC/dijitso recompile, so the `_EsContext` model-reuse property that makes
        the ED/ES search affordable survives regionalization -- see `assign_tmax_region`.
        """
        over = self.SimDet.get("Tmax_region")
        if not over:
            return None
        if self.SimDet.get("active_strain"):
            # Under the multiplicative split the additive active stress (F4) is DISABLED and
            # contractility is carried by gamma, NOT Tmax -- so a regional Tmax map is assembled
            # into a form that never reads it.
            #
            # Refuse only a map that carries a CONTRAST. That is the dangerous case: a regional
            # SEARCH would evaluate every candidate at an identical state, converge on a flat
            # residual surface, and report a "fit". A UNIFORM map is inert but harmless -- gamma
            # supplies the (uniform) contractility from SimDet, and the callers that pass a
            # uniform map through zone_material_maps are running a legitimate uniform-gamma
            # experiment. Refusing that broke the working dang2005 probe path.
            vals = {float(v) for v in dict(over).values()}
            if len(vals) > 1:
                raise ValueError(
                    "SimDet['Tmax_region'] carries a CONTRAST %s together with "
                    "SimDet['active_strain']: under active strain the additive active term is "
                    "disabled and Tmax is INERT, so that contrast cannot change the solution. "
                    "Use SimDet['gamma_region'] = {matid: gamma} (with assign_gamma_region() for "
                    "the live refill), which is the active-strain regional contractility knob."
                    % sorted(vals))
            printout(
                "NOTE regional Tmax map is UNIFORM (%g Pa) and active strain is on, so it is "
                "inert -- contractility comes from the active-strain gamma. Harmless here; a "
                "CONTRAST would be refused." % vals.pop(),
                self.mesh_me.mpi_comm())
            return None
        over = {int(k): float(v) for k, v in dict(over).items()}
        base = float(self.SimDet["GiccioneParams"]["Active params"]["Tmax"])
        # The matid array is RANK-LOCAL: on a partitioned mesh a rank simply may not own any
        # cell of a given region, so a rank-local uniqueness test reports regions "absent" that
        # are present globally -- a loud, false warning on every parallel run. Reduce first.
        _local = set(int(m) for m in np.unique(self.matid_me.array()))
        present = set().union(*self.mesh_me.mpi_comm().allgather(_local))
        missing = sorted(set(over) - present)
        if missing:
            # A region id that matches no cell contributes NOTHING to the active form: the
            # caller believes it set a contractility that is silently absent from the solve.
            printout("WARNING: Tmax_region names matid %s absent from this mesh (present: %s) "
                     "-- those entries are INERT" % (missing, sorted(present)),
                     self.mesh_me.mpi_comm(), level="WARN")
        # A region can carry a Tmax in this field and STILL contribute no active stress, because
        # the active form is assembled only over SimDet["active_region"] (see the F4 loop). The
        # LV baseline hardcodes active_region = [0, 1], so a third zone -- a border ring, or a
        # position band -- would be given a contractility here and then left entirely out of the
        # solve: it converges, and the dead band is announced nowhere. The callers widen
        # active_region from this map; this is the invariant that catches one that forgot.
        _act = self.SimDet.get("active_region")
        if _act is not None:
            _inert = sorted(set(over) - {int(a) for a in _act})
            if _inert:
                raise ValueError(
                    "Tmax_region assigns a contractility to matid %s, but SimDet['active_region'] "
                    "is %s, so those regions get NO active term and the requested contractility "
                    "is silently discarded. Widen active_region to cover every region named in "
                    "Tmax_region." % (_inert, sorted(int(a) for a in _act)))
        DG0 = df.FunctionSpace(self.mesh_me, "DG", 0)
        fld = df.Function(DG0, name="Tmax_region")
        self._regional_tmax_values(fld, over, base)
        printout("Regional Tmax field set (regions %s; base=%.6g Pa)" % (over, base),
                 self.mesh_me.mpi_comm())
        return fld

    def _inplane_angle_field(self, mesh_me):
        """Per-cell angle theta placing the CONSTRUCTED in-plane cross-fibre axis in the
        (s0, n0) plane: ``c0 = cos(theta) s0 + sin(theta) n0``.

        OPT-IN via ``SimDet["transverse_active_inplane_angle"]`` = path to a DOLFIN HDF5 holding
        a DG0 scalar Function at dataset ``theta_inplane``. Absent (the default) returns None,
        and ``transverse_active_structure='inplane'`` then refuses rather than guessing.

        WHY a measured angle rather than a stored dataset: the biaxial cross-fibre literature
        loads the axis in the WALL PLANE, and which stored dataset that is, is a property of the
        mesh's fibre generator. On the four-chamber case-03 mesh NEITHER holds -- the transmural
        Laplace direction gives |s0.eR| ~ 0.78 and |n0.eR| ~ 0.56 on the LV wall, i.e. the pair
        is rotated about f0 -- so naming either one would load a mixture of the in-plane and
        transmural axes while reporting the literature's cell.
        """
        path = self.SimDet.get("transverse_active_inplane_angle")
        if not path:
            return None
        DG0 = df.FunctionSpace(mesh_me, "DG", 0)
        theta = df.Function(DG0, name="theta_inplane")
        with df.HDF5File(mesh_me.mpi_comm(), str(path), "r") as h5:
            if not h5.has_dataset("theta_inplane"):
                raise ValueError(
                    "%s carries no 'theta_inplane' dataset; it is not an in-plane angle field "
                    "(generate it with `python -m fch_events basis --emit-axis`)" % path)
            h5.read(theta, "theta_inplane")
        # The angle field is measured on ONE mesh and consumed on the solve mesh: a file from
        # another mesh (or another refinement) would silently load a mis-indexed axis. The
        # DG0 dof count is the global cell count, so compare it against the solve mesh's.
        n_theta = int(theta.vector().size())
        n_cells = int(mesh_me.num_entities_global(mesh_me.topology().dim()))
        if n_theta != n_cells:
            raise ValueError(
                "in-plane angle field %s has %d cell values but the mechanics mesh has %d "
                "cells: it was measured on a different mesh (re-run `python -m fch_events "
                "basis --emit-axis` on THIS mesh)" % (path, n_theta, n_cells))
        printout("Constructed in-plane cross-fibre axis loaded (theta field from %s, %d cells)"
                 % (path, n_cells), mesh_me.mpi_comm())
        return theta

    def assign_tmax_region(self, values, base=None):
        """Refill the regional Tmax field IN PLACE -- no form rebuild, no recompile.

        The regional analogue of ``.assign()``ing the scalar Tmax Constant, so a regional search
        or sweep reuses ONE model/solver across candidates exactly as the global ED/ES search
        does. Raises if the model was not built with ``SimDet["Tmax_region"]`` -- silently doing
        nothing here would let a sweep report a whole grid solved at one contractility.
        """
        fld = getattr(self, "tmax_region_field", None)
        if fld is None:
            raise RuntimeError(
                "assign_tmax_region() called on a model built WITHOUT SimDet['Tmax_region'] -- "
                "the active form references a scalar Tmax, so per-region values cannot take "
                "effect. Rebuild the model with the region map to use regional contractility.")
        base = float(self.SimDet["GiccioneParams"]["Active params"].get(
            "Tmax_region_base", 0.0)) if base is None else float(base)
        self._regional_tmax_values(fld, {int(k): float(v) for k, v in dict(values).items()}, base)

    def assign_gamma_region(self, values, base=None):
        """Refill the regional active-strain gamma field IN PLACE -- no rebuild, no recompile.

        The active-strain twin of ``assign_tmax_region``. gamma reaches the residual only through
        the DG0 coefficient inside Fa, so refilling it between solves is exactly as free as
        refilling the regional Tmax field, and a regional search keeps its one-build/one-ED-state
        property. Raises if the model was built without ``SimDet["gamma_region"]`` -- silently
        doing nothing would let a whole sweep run at one contractility and look converged."""
        fld = getattr(self, "gamma_region_field", None)
        if fld is None:
            raise RuntimeError(
                "assign_gamma_region() called on a model built WITHOUT SimDet['gamma_region'] "
                "-- Fa references a scalar/per-chamber gamma, so per-region values cannot take "
                "effect. Rebuild the model with the region map to use regional contractility.")
        base = float(self.SimDet.get("active_strain_gamma", 0.15) or 0.15) if base is None \
            else float(base)
        vals = {int(k): float(v) for k, v in dict(values).items()}
        bad = {k: v for k, v in vals.items() if not (0.0 <= v < 1.0)}
        if bad:
            raise ValueError("gamma_region values must satisfy 0 <= gamma < 1; got %s" % bad)
        self._regional_tmax_values(fld, vals, base)

    def _active_strain_gamma_field(self):
        """Per-chamber ACTIVE-STRAIN contractility gamma (max active fibre shortening) -- the
        active-strain knob that REPLACES Tmax. Reads SimDet ``active_strain_gamma_{lv,rv,la,
        ra}``; a chamber without an explicit value falls back to the global
        ``SimDet["active_strain_gamma"]`` (default 0.15). If any per-chamber override is set,
        gamma becomes a DG0 field (chamber value on that chamber's matid region, the global
        value elsewhere) -- mirrors the per-chamber passive bff/bfx/bxx field construction.
        Returns a dolfin.Constant when no per-chamber override is present."""
        base = float(self.SimDet.get("active_strain_gamma", 0.15) or 0.15)
        # SUB-CHAMBER regional gamma (opt-in, SimDet["gamma_region"] = {matid: gamma}) -- the
        # active-strain analogue of Tmax_region, and the parameter a REGIONAL inverse fits under
        # the multiplicative split. Consulted BEFORE the per-chamber map for the same reason
        # `_region_passive_cparam` consults Cparam_region first: a sub-chamber zone id (a remote
        # / border / infarct marker) matches no chamber rid, so it would otherwise fall through
        # to the global base and assemble UNIFORM with nothing said.
        reg = self.SimDet.get("gamma_region")
        if reg:
            reg = {int(k): float(v) for k, v in dict(reg).items()}
            # GLOBAL presence: matid_me.array() is the LOCAL partition, so under MPI a rank that
            # owns no cell of a region would otherwise report it 'absent' (a false WARN; the
            # per-cell fill below is local and correct). Called on every rank at build -> collective OK.
            _loc = [int(m) for m in np.unique(self.matid_me.array())]
            present = set(m for part in self.mesh_me.mpi_comm().allgather(_loc) for m in part)
            missing = sorted(set(reg) - present)
            if missing:
                printout("WARNING: gamma_region names matid %s absent from this mesh (present: "
                         "%s) -- those entries are INERT" % (missing, sorted(present)),
                         self.mesh_me.mpi_comm(), level="WARN")
            bad = {k: v for k, v in reg.items() if not (0.0 <= v < 1.0)}
            if bad:
                # lf = 1 - gamma*a(t) must stay strictly positive for ls = ln = 1/sqrt(lf).
                raise ValueError(
                    "gamma_region values must satisfy 0 <= gamma < 1 (fibre shortening "
                    "fraction); got %s" % bad)
            DG0 = df.FunctionSpace(self.mesh_me, "DG", 0)
            fld = df.Function(DG0, name="gamma_region")
            self._regional_tmax_values(fld, reg, base)
            printout("Regional active-strain gamma field set (regions %s; base=%.4g)"
                     % (reg, base), self.mesh_me.mpi_comm())
            return fld
        ch_rid = {int(self.SimDet.get("lv_rid", 10)): "lv",
                  int(self.SimDet.get("rv_rid", 9)): "rv",
                  int(self.SimDet.get("la_rid", 11)): "la",
                  int(self.SimDet.get("ra_rid", 8)): "ra"}
        over = {rid: float(self.SimDet["active_strain_gamma_%s" % ch])
                for rid, ch in ch_rid.items()
                if self.SimDet.get("active_strain_gamma_%s" % ch) is not None}
        if not over:
            return df.Constant(base)
        DG0 = df.FunctionSpace(self.mesh_me, "DG", 0)
        fld = df.Function(DG0)
        dm = DG0.dofmap()
        mid = self.matid_me.array()
        vals = fld.vector().get_local()
        for c in df.cells(self.mesh_me):
            vals[dm.cell_dofs(c.index())[0]] = over.get(int(mid[c.index()]), base)
        fld.vector().set_local(vals)
        fld.vector().apply("insert")
        printout("Active-strain gamma field set (regions %s; base=%.3g)"
                 % (over, base), self.mesh_me.mpi_comm())
        return fld

    def _build_active_strain_Fa(self, activeforms, f0, s0, n0):
        """Build the ACTIVE-STRAIN active deformation gradient Fa (UFL).

        Multiplicative split F = Fe*Fa with an incompressible, transversely-isotropic active
        fibre shortening (Ambrosi & Pezzuto 2012; Rossi et al. 2014; Barbarotta 2018):

            Fa = lf*(f0 (x) f0) + ls*(s0 (x) s0) + ln*(n0 (x) n0),
            lf = 1 - gamma*a(t)        (fibre shortens),
            ls = ln = 1/sqrt(lf)       => det(Fa) = lf*ls*ln = 1 (volume-preserving).

        a(t) in [0,1] is the SAME normalized Burkhoff activation time course the additive
        active-stress path uses (``activeforms.activeforms.time_course()`` = w1 + w2, tracking
        the twitch clock t_a / t_init / t0), so a twitch's reported ``activation_level`` keeps
        its meaning. gamma is the per-chamber active-strain contractility field. Because the
        passive SEF is built on Fe = F*inv(Fa) (forms_MRC2.Fe), the contraction enters as a
        multiplicative pre-strain of the (polyconvex) passive law -- NOT a rank-1 active
        stress -- so it does not lose strong ellipticity. With gamma=0 in a region Fa reduces
        to f0(x)f0 + s0(x)s0 + n0(x)n0 = I (orthonormal frame), i.e. no active strain there."""
        a_t = activeforms.activeforms.time_course()
        # Clamp a(t) to [0,1] so lf = 1 - gamma*a stays strictly positive (sqrt well-defined).
        a_t = df.conditional(df.gt(a_t, 1.0), df.Constant(1.0),
                             df.conditional(df.lt(a_t, 0.0), df.Constant(0.0), a_t))
        gamma = self._active_strain_gamma_field()
        # Keep the handle: a DG0 gamma is what assign_gamma_region() refills, and even the
        # Constant is worth holding for gamma-continuation (both are coefficients, so changing
        # them forces no recompile). Previously built and dropped.
        self.active_strain_gamma = gamma
        self.gamma_region_field = gamma if self.SimDet.get("gamma_region") else None
        lam_f = 1.0 - gamma * a_t
        lam_t = 1.0 / df.sqrt(lam_f)
        Fa = (lam_f * df.outer(f0, f0)
              + lam_t * df.outer(s0, s0)
              + lam_t * df.outer(n0, n0))
        return Fa

    def Problem(self):
        # Assemble residual/Jacobian for the current model variant (incomp/comp, pctrl, springbc, chambers).
        GuccioneParams = self.SimDet["GiccioneParams"]
        aorta_params = GuccioneParams.get("Aorta params")

        comm_me = self.mesh_me.mpi_comm()

        # isLV or isBiV
        topid = self.SimDet.get("topid")

        # iswaorta or isFCH
        aortic_valvep = self.SimDet.get("aortic_valvep")
        mitral_valvep = self.SimDet.get("mitral_valvep")
        aortaid = self.SimDet.get("aortaid")
        apxid = self.SimDet.get("apxid")

        # isFCH
        septumid = self.SimDet.get("septumid")
        aorta_wall = self.SimDet.get("aorta_wall")
        pulm_wall = self.SimDet.get("pulm_wall")
        pulmonary_valvep = self.SimDet.get("pulmonary_valvep")
        tricuspid_valvep = self.SimDet.get("tricuspid_valvep")

        # iswaorta
        aorta_int_wall = self.SimDet.get("aorta_int_wall")
        aorta_ext_wall = self.SimDet.get("aorta_ext_wall")
        aorta_ring = self.SimDet.get("aorta_ring")

        LVendoid = self.SimDet["LVendoid"]
        RVendoid = self.SimDet["RVendoid"]

        LAendoid = self.SimDet.get("LAendoid")
        RAendoid = self.SimDet.get("RAendoid")

        epiid = self.SimDet["epiid"]
        atrialid = self.SimDet.get("atrialid")

        LVPid = self.SimDet.get("LVPid", LVendoid)
        RVPid = self.SimDet.get("RVPid", RVendoid)

        isincomp = GuccioneParams["incompressible"]
        deg_me = GuccioneParams["deg"]

        mesh_me = self.mesh_me
        facetboundaries_me = self.facetboundaries_me
        f0_me = self.f0_me
        s0_me = self.s0_me
        n0_me = self.n0_me

        eL0_me = self.eL0_me
        eC0_me = self.eC0_me

        eclgn0_me = self.eclgn0_me
        eclgn1_me = self.eclgn1_me

        N_me = df.FacetNormal(mesh_me)
        W_me = self.W
        Q_me = self.Q
        TF_me = self.TF

        w_me = self.w_me
        w_me_n = self.w_me_n
        dw_me = self.dw_me
        wtest_me = self.wtest_me

        bcs_elas = self.set_BCs()

        # FCH swept-cavity volume control (opt-in): the swept chamber's cavity pressure is a
        # Real Lagrange multiplier (swept_pendo) enforcing its volume constraint monolithically.
        # None on every other path so the params dict / pressure_terms stay backward-compatible.
        swept_pendo = None
        fch_swept_chamber = self.SimDet.get("fch_swept_volctrl") if self.isFCH else None
        # LVW swept-cavity volume control (opt-in): the same monolithic constraint applied to
        # the single waorta cavity. The isovolumic active twitch needs it because the
        # partitioned pressure root-find cannot hold volume through high-tension states
        # (documented ~10% ESPVR truncation). Reuses the FCH swept machinery verbatim; only
        # the constrained surface differs (a LIST of three facet ids here).
        lvw_swept = bool(self.iswaorta and self.SimDet.get("lvw_swept_volctrl"))
        if lvw_swept and bool(self.SimDet.get("auto_rigid_support")):
            raise NotImplementedError(
                "lvw_swept_volctrl together with auto_rigid_support is not wired: "
                "_mechanics_mixed_elements appends the rigid-body multipliers only for "
                "isLV/isFCH, so the mixed-space layout and the df.split branch below would "
                "disagree. Disable one of the two."
            )

        if isincomp:
            # Match Trial/Test splits to the mixed element layout constructed in __init__.
            if self.isLV or self.iswaorta:
                if self.ispctrl:
                    if (
                        "springbc" in list(self.SimDet.keys())
                        and self.SimDet["springbc"]
                    ):
                        if self.isLV and bool(self.SimDet.get("auto_rigid_support")):
                            du, dp, dc = df.TrialFunctions(W_me)
                            (u_me, p_me, c_me) = df.split(w_me)
                            (u_me_n, p_me_n, c_me_n) = df.split(w_me_n)
                            (v_me, q_me, cq) = df.TestFunctions(W_me)
                        elif lvw_swept:
                            # LVW swept-cavity volume control: ONE extra Real multiplier
                            # (swept_pendo) = the cavity pressure enforcing V = V0.
                            du, dp, dswept = df.TrialFunctions(W_me)
                            (u_me, p_me, swept_pendo) = df.split(w_me)
                            (u_me_n, p_me_n, swept_pendo_n) = df.split(w_me_n)
                            (v_me, q_me, swept_qendo) = df.TestFunctions(W_me)
                        else:
                            du, dp = df.TrialFunctions(W_me)
                            (u_me, p_me) = df.split(w_me)
                            (u_me_n, p_me_n) = df.split(w_me_n)
                            (v_me, q_me) = df.TestFunctions(W_me)
                        lv_pendo = []
                        rv_pendo = []
                        LVendo_comp = 2
                        RVendo_comp = 1000
                    else:
                        du, dp, dc = df.TrialFunctions(W_me)
                        (u_me, p_me, c_me) = df.split(w_me)
                        (u_me_n, p_me_n, c_me_n) = df.split(w_me_n)
                        (v_me, q_me, cq) = df.TestFunctions(W_me)
                        lv_pendo = []
                        rv_pendo = []
                        LVendo_comp = 2
                        RVendo_comp = 1000

                else:
                    if (
                        "springbc" in list(self.SimDet.keys())
                        and self.SimDet["springbc"]
                        and not bool(self.SimDet.get("auto_rigid_support"))
                    ):
                        du, dp, dlv_pendo = df.TrialFunctions(W_me)
                        (u_me, p_me, lv_pendo) = df.split(w_me)
                        (u_me_n, p_me_n, lv_pendo_n) = df.split(w_me_n)
                        (v_me, q_me, lv_qendo) = df.TestFunctions(W_me)
                        rv_pendo = []
                        LVendo_comp = 2
                        RVendo_comp = 1000

                    else:
                        # auto_rigid_support OR no spring -> the 6 rigid-body Real multipliers c_me
                        # are in the space (need_rigid in _mechanics_mixed_elements); take them here.
                        du, dp, dlv_pendo, dc = df.TrialFunctions(W_me)
                        (u_me, p_me, lv_pendo, c_me) = df.split(w_me)
                        (u_me_n, p_me_n, lv_pendo_n, c_me_n) = df.split(w_me_n)
                        (v_me, q_me, lv_qendo, cq) = df.TestFunctions(W_me)
                        rv_pendo = []
                        LVendo_comp = 2
                        RVendo_comp = 1000

            elif self.isBiV or self.isFCH:
                if self.ispctrl:
                    if (
                        "springbc" in list(self.SimDet.keys())
                        and self.SimDet["springbc"]
                    ):
                        # FCH auto_rigid_support carries the 6 rigid-body Real multipliers c_me
                        # ALONGSIDE the springs (need_rigid in _mechanics_mixed_elements appends
                        # VRelem here only for self.isFCH). Element order: u, p, [swept_pendo], [c_me].
                        _auto_rigid_fch = self.isFCH and bool(self.SimDet.get("auto_rigid_support"))
                        _swept_blocks = self._fch_swept_block_names()
                        if len(_swept_blocks) > 1:
                            # MULTI-cavity volume control: N Real multipliers. The fixed-arity
                            # unpackings below cannot express a variable block count, so select
                            # by LAYOUT INDEX -- the same discipline _split_state already uses,
                            # and the reason a trailing rigid-body block cannot be mistaken for
                            # a cavity pressure.
                            _lay = self.mechanics_mixed_layout()
                            _tr, _sp, _spn, _te = (df.TrialFunctions(W_me), df.split(w_me),
                                                   df.split(w_me_n), df.TestFunctions(W_me))
                            u_me, p_me = _sp[_lay.index("u")], _sp[_lay.index("p")]
                            u_me_n, p_me_n = _spn[_lay.index("u")], _spn[_lay.index("p")]
                            v_me, q_me = _te[_lay.index("u")], _te[_lay.index("p")]
                            du, dp = _tr[_lay.index("u")], _tr[_lay.index("p")]
                            swept_pendos = {c: _sp[_lay.index("fch_swept_%s" % c)]
                                            for c in self.fch_swept_chambers()}
                            swept_qendos = {c: _te[_lay.index("fch_swept_%s" % c)]
                                            for c in self.fch_swept_chambers()}
                            swept_pendo = None
                            if _auto_rigid_fch:
                                c_me, cq = _sp[_lay.index("rigid")], _te[_lay.index("rigid")]
                                c_me_n = _spn[_lay.index("rigid")]
                        elif fch_swept_chamber and _auto_rigid_fch:
                            du, dp, dswept, dc = df.TrialFunctions(W_me)
                            (u_me, p_me, swept_pendo, c_me) = df.split(w_me)
                            (u_me_n, p_me_n, swept_pendo_n, c_me_n) = df.split(w_me_n)
                            (v_me, q_me, swept_qendo, cq) = df.TestFunctions(W_me)
                        elif fch_swept_chamber:
                            # Swept-cavity volume control: one extra Real multiplier
                            # (swept_pendo) = the swept chamber's cavity pressure.
                            du, dp, dswept = df.TrialFunctions(W_me)
                            (u_me, p_me, swept_pendo) = df.split(w_me)
                            (u_me_n, p_me_n, swept_pendo_n) = df.split(w_me_n)
                            (v_me, q_me, swept_qendo) = df.TestFunctions(W_me)
                        elif _auto_rigid_fch:
                            du, dp, dc = df.TrialFunctions(W_me)
                            (u_me, p_me, c_me) = df.split(w_me)
                            (u_me_n, p_me_n, c_me_n) = df.split(w_me_n)
                            (v_me, q_me, cq) = df.TestFunctions(W_me)
                        else:
                            du, dp = df.TrialFunctions(W_me)
                            (u_me, p_me) = df.split(w_me)
                            (u_me_n, p_me_n) = df.split(w_me_n)
                            (v_me, q_me) = df.TestFunctions(W_me)
                        LVendo_comp = 2
                        RVendo_comp = 3
                        lv_pendo = []
                        rv_pendo = []

                    else:
                        du, dp, dc = df.TrialFunctions(W_me)
                        (u_me, p_me, c_me) = df.split(w_me)
                        (u_me_n, p_me_n, c_me_n) = df.split(w_me_n)
                        (v_me, q_me, cq) = df.TestFunctions(W_me)
                        LVendo_comp = 2
                        RVendo_comp = 3
                        lv_pendo = []
                        rv_pendo = []

                else:
                    if (
                        "springbc" in list(self.SimDet.keys())
                        and self.SimDet["springbc"]
                    ):
                        du, dp, dlv_pendo, drv_pendo = df.TrialFunctions(W_me)
                        (u_me, p_me, lv_pendo, rv_pendo) = df.split(w_me)
                        (u_me_n, p_me_n, lv_pendo_n, rv_pendo_n) = df.split(w_me_n)
                        (v_me, q_me, lv_qendo, rv_qendo) = df.TestFunctions(W_me)
                        LVendo_comp = 2
                        RVendo_comp = 3

                    else:
                        du, dp, dlv_pendo, drv_pendo, dc = df.TrialFunctions(W_me)
                        (u_me, p_me, lv_pendo, rv_pendo, c_me) = df.split(w_me)
                        (u_me_n, p_me_n, lv_pendo_n, rv_pendo_n, c_me_n) = df.split(w_me_n)
                        (v_me, q_me, lv_qendo, rv_qendo, cq) = df.TestFunctions(W_me)
                        LVendo_comp = 2
                        RVendo_comp = 3
        else:
            if self.isLV or self.iswaorta:
                if self.ispctrl:
                    if (
                        "springbc" in list(self.SimDet.keys())
                        and self.SimDet["springbc"]
                    ):
                        du = df.TrialFunctions(W_me)
                        (u_me) = df.split(w_me)
                        (v_me,) = df.TestFunctions(W_me)
                        p_me = df.Function(Q_me)
                        lv_pendo = []
                        rv_pendo = []
                        LVendo_comp = 1
                        RVendo_comp = 1000
                    else:
                        du, dc = df.TrialFunctions(W_me)
                        (u_me, c_me) = df.split(w_me)
                        (v_me, cq) = df.TestFunctions(W_me)
                        p_me = df.Function(Q_me)
                        lv_pendo = []
                        rv_pendo = []
                        LVendo_comp = 1
                        RVendo_comp = 1000
                else:
                    if (
                        "springbc" in list(self.SimDet.keys())
                        and self.SimDet["springbc"]
                    ):
                        du, dlv_pendo = df.TrialFunctions(W_me)
                        (u_me, lv_pendo) = df.split(w_me)
                        (v_me, lv_qendo) = df.TestFunctions(W_me)
                        p_me = df.Function(Q_me)
                        rv_pendo = []
                        LVendo_comp = 1
                        RVendo_comp = 1000
                    else:
                        du, dlv_pendo, dc = df.TrialFunctions(W_me)
                        (u_me, lv_pendo, c_me) = df.split(w_me)
                        (v_me, lv_qendo, cq) = df.TestFunctions(W_me)
                        p_me = df.Function(Q_me)
                        rv_pendo = []
                        LVendo_comp = 1
                        RVendo_comp = 1000
            elif self.isBiV or self.isFCH:
                if self.ispctrl:
                    if (
                        "springbc" in list(self.SimDet.keys())
                        and self.SimDet["springbc"]
                    ):
                        du = df.TrialFunctions(W_me)
                        (u_me) = df.split(w_me)
                        (u_me_n) = df.split(w_me_n)
                        (v_me,) = df.TestFunctions(W_me)
                        p_me = df.Function(Q_me)
                        lv_pendo = []
                        rv_pendo = []
                        LVendo_comp = 1
                        RVendo_comp = 2

                    else:
                        du, dlv_pendo, drv_pendo, dc = df.TrialFunctions(W_me)
                        (u_me, lv_pendo, rv_pendo, c_me) = df.split(w_me)
                        (v_me, lv_qendo, rv_qendo, cq) = df.TestFunctions(W_me)
                        p_me = df.Function(Q_me)
                        lv_pendo = []
                        rv_pendo = []
                        LVendo_comp = 1
                        RVendo_comp = 2
                else:
                    if (
                        "springbc" in list(self.SimDet.keys())
                        and self.SimDet["springbc"]
                    ):
                        du, dlv_pendo, drv_pendo = df.TrialFunctions(W_me)
                        (u_me, lv_pendo, rv_pendo) = df.split(w_me)
                        (v_me, lv_qendo, rv_qendo) = df.TestFunctions(W_me)
                        p_me = df.Function(Q_me)
                        LVendo_comp = 1
                        RVendo_comp = 2

                    else:
                        du, dlv_pendo, drv_pendo, dc = df.TrialFunctions(W_me)
                        (u_me, lv_pendo, rv_pendo, c_me) = df.split(w_me)
                        (v_me, lv_qendo, rv_qendo, cq) = df.TestFunctions(W_me)
                        p_me = df.Function(Q_me)
                        LVendo_comp = 1
                        RVendo_comp = 2

        self.t_a = df.Function(self.Quad)
        self.t_a.vector()[:] = 0.0

        self.cycle = df.Function(self.Quad)
        self.cycle.vector()[:] = 0.0
        self.BCL = 0.0

        ds_me = self.ds_me
        dx_me = self.dx_me
        LVendo_area_me = self.LVendo_area_me

        # ACTIVE-STRAIN (opt-in, SimDet["active_strain"], default False). The active
        # deformation Fa (built below) routes the activation Quadrature element a(t) into the
        # passive SEF (Fe = F*inv(Fa)) AND into every J/Fe-dependent volume term (e.g. the
        # P1P1 pressure-stabilization, since J = det(Fe)). Pin ALL volume integrals to the
        # activation element's quadrature degree + "default" scheme so the QuadratureElement
        # tabulates on matching points (the additive active-stress F4 is implicitly pinned the
        # same way). OFF -> dx_me unchanged (byte-identical additive active stress).
        self._active_strain = bool(self.SimDet.get("active_strain", False))
        if self._active_strain:
            dx_me = df.Measure(
                "dx", domain=mesh_me, subdomain_data=self.matid_me,
                metadata={"quadrature_degree": int(deg_me),
                          "quadrature_scheme": "default"},
            )

        # FCH swept-cavity volume control: the swept chamber's endo facet id + its reference
        # endo area (domain-bound ds_me, mirrors LVendo_area_me) -- the area normalizes the
        # constraint's (1/area)*pendo*V0 term. df.Constant so the form divides by a scalar.
        fch_swept_endoid = (
            {"lv": LVendoid, "rv": RVendoid, "la": LAendoid, "ra": RAendoid}.get(fch_swept_chamber)
            if fch_swept_chamber else None
        )
        if lvw_swept:
            # The waorta cavity is closed by THREE surfaces -- exactly the set
            # LVcavityvol_waorta integrates over, so the constraint and the reported volume
            # are the same functional.
            fch_swept_endoid = [LVendoid, self.SimDet["aortic_valvep"],
                                self.SimDet["mitral_valvep"]]
        # MULTI-cavity: one (endoid, area, multiplier, V0) tuple per swept chamber. Built from
        # the SAME id map and the same reference-area assembly as the single-chamber path, so a
        # two-cavity run differs from two one-cavity runs only in that both constraints are
        # solved simultaneously with the displacement.
        _endo_by_ch = {"lv": LVendoid, "rv": RVendoid, "la": LAendoid, "ra": RAendoid}
        fch_swept_multi = []
        if len(self._fch_swept_block_names()) > 1:
            for _c in self.fch_swept_chambers():
                _eid = _endo_by_ch[_c]
                _a = df.assemble(df.Constant(1.0) * ds_me(_eid),
                                 form_compiler_parameters={"representation": "uflacs"})
                fch_swept_multi.append({
                    "chamber": _c, "endoid": _eid,
                    "area": df.Constant(_a if _a > 1.0e-12 else 1.0),
                    "pendo": swept_pendos[_c],
                    "V0": self.FCHsweptCavityvols[_c],
                })

        if fch_swept_endoid is not None:
            _swept_measure = ds_me(fch_swept_endoid[0]) if isinstance(fch_swept_endoid, list) \
                else ds_me(fch_swept_endoid)
            if isinstance(fch_swept_endoid, list):
                for _idx in fch_swept_endoid[1:]:
                    _swept_measure = _swept_measure + ds_me(_idx)
            _swept_area = df.assemble(
                df.Constant(1.0) * _swept_measure,
                form_compiler_parameters={"representation": "uflacs"},
            )
            fch_swept_area_me = df.Constant(_swept_area if _swept_area > 1.0e-12 else 1.0)
        else:
            fch_swept_area_me = df.Constant(1.0)

        dim_me = mesh_me.geometry().dim()
        I_me = df.Identity(dim_me)
        F_old = I_me + df.grad(u_me_n)
        # Previous-step strain is used by passive viscosity terms (if enabled).
        Ea_old_expr = 0.5 * (F_old.T * F_old - I_me)

        # Active-tension forms are built BEFORE the passive `params` dict so the opt-in
        # ACTIVE-STRAIN path (below) can wire its active deformation Fa into the passive SEF
        # through the growth-tensor hook (Forms.Fe() => Fe = F*inv(Fa)). The activeforms
        # object itself is unchanged either way; in active-strain mode its additive stress
        # (F4) is disabled and only its activation time course a(t) is reused to build Fa.
        activeparams = {
            "mesh": mesh_me,
            "dx": dx_me,
            "deg": GuccioneParams["deg"],
            "facetboundaries": facetboundaries_me,
            "facet_normal": N_me,
            "displacement_variable": u_me,
            "pressure_variable": p_me,
            "fiber": f0_me,
            "sheet": s0_me,
            "sheet-normal": n0_me,
            "inplane-angle": self._inplane_angle_field(mesh_me),
            "t_a": self.t_a,
            "cycle": self.cycle,
            "BCL": self.BCL,
            "Threshold_Potential": 0.9,
            "growth_tensor": None,
        }

        if "Active model" in list(GuccioneParams.keys()):
            activeparams.update({"material model": GuccioneParams["Active model"]})

        if "Active params" in list(GuccioneParams.keys()):
            activeparams.update({"material params": GuccioneParams["Active params"]})

        if "HeartBeatLength" in list(self.SimDet.keys()):
            activeparams.update({"HeartBeatLength": self.SimDet["HeartBeatLength"]})

        if "HomogenousActivation" in list(GuccioneParams.keys()):
            activeparams.update(
                {"HomogenousActivation": GuccioneParams["HomogenousActivation"]}
            )
        else:
            activeparams.update({"HomogenousActivation": True})

        # Transverse (cross-fiber) active-tension fraction kappa is consumed by
        # activeForms._fiber_projection off THIS activeparams dict; it was previously set only
        # on the Forms `params` dict below, so kappa never reached the active-stress projection
        # (a silent no-op -- the transverse arm was byte-identical to additive). Resolve it
        # identically here and pass it through. Absent / 0.0 => byte-identical (fiber-only
        # outer(f0,f0)).
        activeparams["transverse_active_fraction"] = float(
            self.SimDet.get("transverse_active_fraction",
                            self.SimDet["GiccioneParams"]["Active params"].get(
                                "transverse_active_fraction", 0.0)) or 0.0)
        # Which transverse axes kappa loads. Default "sheet" is the Guccione-school
        # convention (Genet 2014 10.1152/japplphysiol.00255.2014: 40% of maximum in the
        # local SHEET direction); "both" reproduces the superseded pre-2026-08-11 form and
        # warns. Inert when kappa=0. See activeforms_MRC2._fiber_projection.
        activeparams["transverse_active_structure"] = str(
            self.SimDet.get("transverse_active_structure",
                            self.SimDet["GiccioneParams"]["Active params"].get(
                                "transverse_active_structure", "sheet")) or "sheet")

        # REGIONAL contractility (opt-in, SimDet["Tmax_region"]): swap the scalar Tmax the
        # active law reads for a DG0-over-matid field BEFORE activeForms captures the params
        # dict. `activeparams["material params"]` IS GuccioneParams["Active params"], and
        # BurkhoffTimevarying3 reads params["Tmax"] into UFL arithmetic at PK2StressTensor()
        # call time, so the regional field reaches the form through the EXISTING per-region
        # assembly loop below with no change to that loop. None (the default) => untouched.
        self.tmax_region_field = self._active_tmax_field()
        if self.tmax_region_field is not None:
            _act_params = self.SimDet["GiccioneParams"]["Active params"]
            # Keep the scalar the field was built around: assign_tmax_region() defaults to it
            # for any region the caller does not name.
            _act_params["Tmax_region_base"] = float(_act_params["Tmax"])
            _act_params["Tmax"] = self.tmax_region_field

        activeforms = activeForms(activeparams)
        self.activeforms = activeforms

        # ACTIVE-STRAIN (opt-in): build the active deformation gradient Fa and route it into
        # the passive SEF via growth_tensor below (self._active_strain set above). Default OFF
        # -> Fa_active None -> growth_tensor None -> byte-identical to the additive active stress.
        Fa_active = None
        if self._active_strain:
            Fa_active = self._build_active_strain_Fa(activeforms, f0_me, s0_me, n0_me)
            printout(
                "ACTIVE-STRAIN ON: F = Fe*Fa fibre shortening via growth_tensor; additive "
                "active stress (F4) DISABLED (contraction not double-counted).",
                self.mesh_me.mpi_comm(),
            )

        params = {
            "mesh": mesh_me,
            "facetboundaries": facetboundaries_me,
            "facet_normal": N_me,
            "mixedfunctionspace": W_me,
            "mixedfunction": w_me,
            "displacement_variable": u_me,
            "pressure_variable": p_me,
            "lv_volconst_variable": lv_pendo,
            "lv_constrained_vol": self.LVCavityvol,
            "rv_volconst_variable": rv_pendo,
            "rv_constrained_vol": self.RVCavityvol,
            # FCH swept-cavity volume control (only consulted when fch_swept_volctrl set):
            "fch_swept_volconst_variable": swept_pendo,
            "fch_swept_constrained_vol": self.FCHsweptCavityvol,
            "fch_swept_endoid": fch_swept_endoid,
            "fch_swept_area": fch_swept_area_me,
            "fch_swept_multi": fch_swept_multi,
            "LVendoid": LVendoid,
            "RVendoid": RVendoid,
            "LAendoid": LAendoid,
            "RAendoid": RAendoid,
            "epiid": epiid,
            "atrialid": atrialid,
            "topid": topid,
            "aortaid": aortaid,
            "LVPid": LVPid,
            "RVPid": RVPid,
            "aortic_valvep": aortic_valvep,
            "mitral_valvep": mitral_valvep,
            "pulmonary_valvep": pulmonary_valvep,
            "tricuspid_valvep": tricuspid_valvep,
            "septumid": septumid,
            "aorta_wall": aorta_wall,
            "pulm_wall": pulm_wall,
            "apxid": apxid,
            "aorta_int_wall": aorta_int_wall,
            "aorta_ext_wall": aorta_ext_wall,
            "aorta_ring": aorta_ring,
            "LVendo_comp": LVendo_comp,
            "RVendo_comp": RVendo_comp,
            "fiber": f0_me,
            "sheet": s0_me,
            "sheet-normal": n0_me,
            # Optional transverse (cross-fiber) active-tension fraction kappa: kappa>0 makes
            # the active stress non-rank-1 (f x f + kappa * s x s, the Guccione-school SHEET
            # convention -- see activeforms_MRC2._fiber_projection).
            # Default 0.0 = legacy fiber-only active stress (byte-identical).
            "transverse_active_fraction": float(
                self.SimDet.get("transverse_active_fraction",
                                self.SimDet["GiccioneParams"]["Active params"].get(
                                    "transverse_active_fraction", 0.0)) or 0.0),
            "transverse_active_structure": str(
                self.SimDet.get("transverse_active_structure",
                                self.SimDet["GiccioneParams"]["Active params"].get(
                                    "transverse_active_structure", "sheet")) or "sheet"),
            "fiberz-aorta": eL0_me,
            "fiberc-aorta": eC0_me,
            "fiberclgn0-aorta": eclgn0_me,
            "fiberclgn1-aorta": eclgn1_me,
            "growth_tensor": Fa_active,
            "material model": GuccioneParams["Passive model"],
            "material params": GuccioneParams["Passive params"],
            "aorta params": aorta_params,
            # Constitutive law of the LVW annular PLUG (waorta ``rubber_region``):
            # "guccione" (default, the historical Cparam_av/Cparam Guccione form) or
            # "neohookean" (isotropic Ibar1 neo-Hookean at Passive params["plug_mu"]).
            # Validated by GuccionePas.plug_law(); inert off the waorta path.
            "plug_law": str(self.SimDet.get("plug_law", "guccione") or "guccione"),
            # Law of the AORTIC-VALVE CAP alone (waorta region SimDet["aocap_region"],
            # default 3): None => the cap is the plug (byte-identical default path);
            # "guccione"/"neohookean" at Passive params["aocap_cparam"/"aocap_mu"].
            "aocap_law": self.SimDet.get("aocap_law"),
            # Law of the fibrous ANNULUS (waorta regions SimDet["annulus_regions"],
            # default [4, 5]): "guccione"/"neohookean" at Passive params["annulus_cparam"/
            # "annulus_mu"]. Absent => the annulus keeps the shared plug law.
            "annulus_law": self.SimDet.get("annulus_law"),
            # Volumetric-isochoric split of the AORTIC GROUND MATRIX (opt-in,
            # default OFF = byte-identical legacy form). See GuccionePas.
            # _aorta_isochoric_ground: with the coupled I1 form the aortic wall
            # is NOT stress-free at F = I, and the resulting rest-state
            # prestress deflates the LV cavity below its geometric reference.
            "aorta_isochoric_ground": bool(
                self.SimDet.get(
                    "aorta_isochoric_ground",
                    (aorta_params or {}).get("isochoric_ground", False),
                )
            ),
            "Ea_old": Ea_old_expr,
            "dt": self.dt_const,
            "_passive_viscous_": self.SimDet.get("_passive_viscous_", False),
            "incompressible": GuccioneParams["incompressible"],
            "LVendo_area": LVendo_area_me,
            "lv_constrained_pres": self.LVCavitypres,
            "rv_constrained_pres": self.RVCavitypres,
            "la_constrained_pres": self.LACavitypres,
            "ra_constrained_pres": self.RACavitypres,
            "aorta_constrained_pres": self.AortaCavitypres,
        }
        if params["aorta_isochoric_ground"]:
            printout(
                "AORTA ISOCHORIC GROUND MATRIX ON: aortic ground-matrix term built on "
                "Ibar1 = J^(-2/3)*I1 (volumetric-isochoric split), so the aortic wall is "
                "stress-free at F=I; fibre invariants stay the full I4.",
                self.mesh_me.mpi_comm(),
                stage="mesh",
            )
        # Per-chamber PASSIVE-EXPONENT overrides (bff/bfx/bxx). Unlike Cparam (a linear
        # prefactor handled by region-scaling below), bff/bfx/bxx live INSIDE the Guccione
        # exponential Q, so a per-chamber value cannot be a scale -- it must enter the SEF
        # as a spatially-varying field. If any `{bff,bfx,bxx}_{lv,rv,la,ra}` is set in
        # SimDet, replace the corresponding Constant with a DG0 field (chamber value on that
        # chamber's matid region, the global value elsewhere). Physiological use: a thin,
        # compliant RV needs a LOWER bff (it fills to a larger EDV at the same low filling
        # pressure than the stiff shared bff=29 allows). A no-op if no override is set.
        _pp = dict(GuccioneParams["Passive params"])
        _ch_rid = {int(self.SimDet.get("lv_rid", 10)): "lv",
                   int(self.SimDet.get("rv_rid", 9)): "rv",
                   int(self.SimDet.get("la_rid", 11)): "la",
                   int(self.SimDet.get("ra_rid", 8)): "ra"}
        _DG0b = None
        for _pname in ("bff", "bfx", "bxx"):
            _over = {rid: float(self.SimDet["%s_%s" % (_pname, ch)])
                     for rid, ch in _ch_rid.items()
                     if self.SimDet.get("%s_%s" % (_pname, ch)) is not None}
            if not _over:
                continue
            if _DG0b is None:
                _DG0b = df.FunctionSpace(self.mesh_me, "DG", 0)
            _base = float(_pp[_pname])
            _fld = df.Function(_DG0b)
            _dm = _DG0b.dofmap()
            _mid = self.matid_me.array()
            _vals = _fld.vector().get_local()
            for _c in df.cells(self.mesh_me):
                _vals[_dm.cell_dofs(_c.index())[0]] = _over.get(int(_mid[_c.index()]), _base)
            _fld.vector().set_local(_vals); _fld.vector().apply("insert")
            _pp[_pname] = _fld
            printout("Per-chamber passive %s set (regions %s; base=%.3g)"
                     % (_pname, _over, _base), self.mesh_me.mpi_comm())
        # PLUG stiffness scalar (LVW annular plug, waorta ``rubber_region``): Cparam_av
        # (Guccione plug) / plug_mu (neo-Hookean plug). Stored as a dolfin.Constant so the
        # form references ONE object a driver can .assign() (the loading-phase stiffness
        # continuation in run_waorta) with no recompile. Absent Cparam_av => the plug
        # falls back to Cparam exactly as before (byte-identical default path).
        self.plug_stiffness_constant = None
        _plug_law = str(self.SimDet.get("plug_law", "guccione") or "guccione").lower()
        _plug_key = "plug_mu" if _plug_law == "neohookean" else "Cparam_av"
        if self.iswaorta and _pp.get(_plug_key) is not None:
            if not isinstance(_pp[_plug_key], df.Constant):
                _pp[_plug_key] = df.Constant(float(_pp[_plug_key]))
            self.plug_stiffness_constant = _pp[_plug_key]
            printout("LVW plug law=%s %s=%.6g Pa on rubber_region=%s (myocardial Cparam=%.6g)"
                     % (_plug_law, _plug_key, float(self.plug_stiffness_constant),
                        list(self.SimDet.get("rubber_region", [])), float(_pp["Cparam"])),
                     self.mesh_me.mpi_comm(), stage="mesh")
        elif self.iswaorta and _plug_law == "neohookean":
            raise KeyError("plug_law='neohookean' requires Passive params['plug_mu'] "
                           "(note key PLUG_MU); refusing to build a plug with no stiffness")
        # AORTIC-VALVE CAP stiffness scalar (region ``aocap_region``, default 3, under
        # ``aocap_law``): its own Constant, exposed as ``aocap_stiffness_constant`` for the
        # loading-phase continuation. The mitral cap / basal ring never read it.
        self.aocap_stiffness_constant = None
        self.aocap_region = None
        _aocap_law = self.SimDet.get("aocap_law")
        _aocap_law = str(_aocap_law).lower() if _aocap_law not in (None, "") else None
        if _aocap_law is not None:
            if not self.iswaorta:
                raise ValueError("aocap_law is only defined on the waorta topology")
            if _aocap_law not in ("guccione", "neohookean"):
                raise ValueError("aocap_law=%r is not guccione|neohookean" % _aocap_law)
            _aocap_key = "aocap_mu" if _aocap_law == "neohookean" else "aocap_cparam"
            if _pp.get(_aocap_key) is None:
                raise KeyError("aocap_law=%r requires Passive params[%r] (note key %s); "
                               "refusing to build a cap with no stiffness"
                               % (_aocap_law, _aocap_key, _aocap_key.upper()))
            if not isinstance(_pp[_aocap_key], df.Constant):
                _pp[_aocap_key] = df.Constant(float(_pp[_aocap_key]))
            self.aocap_stiffness_constant = _pp[_aocap_key]
            self.aocap_region = int(self.SimDet.get("aocap_region", 3))
            _rubber_ids = [int(r) for r in self.SimDet.get("rubber_region", [])]
            if self.aocap_region not in _rubber_ids:
                raise ValueError("aocap_region=%d is not one of rubber_region=%s"
                                 % (self.aocap_region, _rubber_ids))
            _n_cap = int(df.MPI.sum(self.mesh_me.mpi_comm(),
                                    float((self.matid_me.array() == self.aocap_region).sum())))
            if _n_cap == 0:
                raise ValueError("aocap_region=%d has no cells on this mesh" % self.aocap_region)
            printout("AORTIC-VALVE CAP law=%s %s=%.6g Pa on region %d (%d cells); the other "
                     "rubber regions %s keep plug law=%s"
                     % (_aocap_law, _aocap_key, float(self.aocap_stiffness_constant),
                        self.aocap_region, _n_cap,
                        [r for r in _rubber_ids if r != self.aocap_region], _plug_law),
                     self.mesh_me.mpi_comm(), stage="mesh")
        # FIBROUS ANNULUS stiffness scalar (regions ``annulus_regions``, default [4, 5],
        # under ``annulus_law``): its own Constant, exposed as ``annulus_stiffness_constant``
        # for the loading-phase continuation. Independent of the cap (region 3) and of the
        # shared plug law; a region listed in both the cap and the annulus is refused.
        self.annulus_stiffness_constant = None
        self.annulus_regions = None
        _ann_law = self.SimDet.get("annulus_law")
        _ann_law = str(_ann_law).lower() if _ann_law not in (None, "") else None
        if _ann_law is not None:
            if not self.iswaorta:
                raise ValueError("annulus_law is only defined on the waorta topology")
            if _ann_law not in ("guccione", "neohookean"):
                raise ValueError("annulus_law=%r is not guccione|neohookean" % _ann_law)
            _ann_key = "annulus_mu" if _ann_law == "neohookean" else "annulus_cparam"
            if _pp.get(_ann_key) is None:
                raise KeyError("annulus_law=%r requires Passive params[%r] (note key %s); "
                               "refusing to build an annulus with no stiffness"
                               % (_ann_law, _ann_key, _ann_key.upper()))
            if not isinstance(_pp[_ann_key], df.Constant):
                _pp[_ann_key] = df.Constant(float(_pp[_ann_key]))
            self.annulus_stiffness_constant = _pp[_ann_key]
            self.annulus_regions = [int(r) for r in self.SimDet.get("annulus_regions", [4, 5])]
            _rubber_ids = [int(r) for r in self.SimDet.get("rubber_region", [])]
            for _r in self.annulus_regions:
                if _r not in _rubber_ids:
                    raise ValueError("annulus_regions=%s: %d is not one of rubber_region=%s"
                                     % (self.annulus_regions, _r, _rubber_ids))
                if self.aocap_region is not None and _r == self.aocap_region:
                    raise ValueError("region %d is claimed by BOTH aocap_region and "
                                     "annulus_regions" % _r)
                _n = int(df.MPI.sum(self.mesh_me.mpi_comm(),
                                    float((self.matid_me.array() == _r).sum())))
                if _n == 0:
                    raise ValueError("annulus region %d has no cells on this mesh" % _r)
            printout("FIBROUS ANNULUS law=%s %s=%.6g Pa on regions %s; the remaining rubber "
                     "regions %s keep plug law=%s"
                     % (_ann_law, _ann_key, float(self.annulus_stiffness_constant),
                        self.annulus_regions,
                        [r for r in _rubber_ids if r not in self.annulus_regions
                         and r != self.aocap_region], _plug_law),
                     self.mesh_me.mpi_comm(), stage="mesh")
        params["material params"] = _pp

        # Forms(...) consumes this dict to define kinematics, chamber constraints, and material models.
        form_option_keys = [
            "surface_quadrature_degree",
            "kappa_main_multiplier",
            "laplace_fch_bc",
            "laplace_waorta_bc",
            "laplace_ideal_bc",
            "enable_laplace_solver",
        ]
        for key in form_option_keys:
            if key in self.SimDet:
                params[key] = self.SimDet[key]

        uflforms = Forms(params)
        self.uflforms = uflforms

        Fmat = uflforms.Fmat()
        Emat = df.variable(uflforms.Emat())
        J = uflforms.J()

        # Deformed surface normal for the spring follower BC uses the TOTAL deformation
        # (Nanson det(Fmat)*Fmat^-T*N). With an active-strain growth tensor uflforms.J() =
        # det(Fe) would carry the cell-Quadrature activation onto facet integrals; det(Fmat)
        # is CG-displacement only and byte-identical when growth_tensor is None.
        n_me = df.det(Fmat) * df.inv(Fmat.T) * N_me

        Wp_me = uflforms.PassiveMatSEF(Ea=Emat)
        WpRub_me = None
        WpAorta_me = None
        WpAoCap_me = None
        WpAnnulus_me = None
        if self.iswaorta:
            WpRub_me = uflforms.PassiveRubSEF()
            WpAorta_me = uflforms.aorta_strain_energy()
            if self.aocap_stiffness_constant is not None:
                WpAoCap_me = uflforms.PassiveAoCapSEF()
            if self.annulus_stiffness_constant is not None:
                WpAnnulus_me = uflforms.PassiveAnnulusSEF()
        self._aocap_sef = WpAoCap_me
        self._annulus_sef = WpAnnulus_me
        # Per-region override maps for the two waorta builders: the cap and the annulus
        # each contribute their own regions; None when neither is set (byte-identical).
        _rub_sef_override = {}
        _rub_mu_override = {}
        if WpAoCap_me is not None:
            _rub_sef_override[self.aocap_region] = WpAoCap_me
            _rub_mu_override[self.aocap_region] = uflforms.shear_aocap()
        if WpAnnulus_me is not None:
            for _r in self.annulus_regions:
                _rub_sef_override[_r] = WpAnnulus_me
                _rub_mu_override[_r] = uflforms.shear_annulus()
        self._rubber_sef_override = _rub_sef_override or None
        self._rubber_mu_override = _rub_mu_override or None

        use_passive_viscosity = bool(self.SimDet.get("_passive_viscous_", False))
        viscous_stress = None
        if use_passive_viscosity:
            # Extract the non-hyperelastic part of the configured passive stress as a viscous correction.
            passive_stress = uflforms.passiveforms.PassiveStress(Ea=Emat)
            hyperelastic_stress = df.diff(Wp_me, Emat)
            viscous_stress = passive_stress - hyperelastic_stress

        def add_passive_region(current_form, measure, scale=1.0):
            base_form = df.derivative(scale * Wp_me, w_me, wtest_me) * measure
            if viscous_stress is not None:
                base_form += scale * df.inner(Fmat * viscous_stress, df.grad(v_me)) * measure
            if current_form is None:
                return base_form
            return current_form + base_form

        if not self.ispctrl:
            LV_Wvol = uflforms.LVV0constrainedE()
            if self.isBiV:
                RV_Wvol = uflforms.RVV0constrainedE()

        X_me = df.SpatialCoordinate(mesh_me)

        F1 = None
        if self.iswaorta:
            F1 = build_passive_form_waorta(
                add_passive_region,
                self.SimDet,
                dx_me,
                w_me,
                wtest_me,
                WpRub_me,
                WpAorta_me,
                rubber_region_sef=self._rubber_sef_override,
            )
        else:  # not iswaorta, not fch_fe
            collapse_passive = bool(self.SimDet.get("collapse_passive_regions", False))
            # Per-chamber passive stiffness resolves each region to ONE ABSOLUTE Guccione
            # Cparam via the single-coefficient resolver ``_region_passive_scale``: the
            # per-chamber override ``Cparam_{lv,rv,la,ra}`` or the documented fallback.
            # The SEF is linear in Cparam, so the region term is the single Wp_me (built
            # at the fallback) times Cparam_region/Cparam_fallback -- the region's absolute
            # stiffness. The collapse fast path uses the SAME resolver over the whole
            # domain (rid=None -> fallback, scale 1.0); no context reads a bare base.
            if collapse_passive:
                # Optional fast path: assemble passive response in one shot over the full
                # domain at the single documented fallback Cparam (rid=None -> scale 1.0).
                F1 = add_passive_region(None, dx_me, scale=self._region_passive_scale(None))
            else:
                keymap = self._fch_passive_cparam_keymap()
                region_cnt = 0
                # Integrate passive energy across all matid values present in the mesh.
                for regionid in np.arange(
                    min(self.matid_me.array()), max(self.matid_me.array()) + 1
                ):
                    scale = self._region_passive_scale(int(regionid), keymap)
                    if region_cnt == 0:
                        F1 = add_passive_region(None, dx_me(int(regionid)), scale=scale)
                    else:
                        F1 = add_passive_region(F1, dx_me(int(regionid)), scale=scale)
                    region_cnt += 1

        if self._active_strain:
            # Active strain delivers contraction through Fa (growth_tensor) acting on the
            # passive SEF (Fe = F*inv(Fa)); the additive active stress is SKIPPED so the
            # contraction is not double-counted. F4 = None -> Ftotal = F1 (passive-on-Fe).
            F4 = None
        elif "active_region" in list(self.SimDet.keys()):
            printout(
                f"Active region = {self.SimDet['active_region']}",
                self.mesh_me.mpi_comm(),
            )
            region_cnt = 0

            # Per-chamber ABSOLUTE active Tmax (``Tmax_{lv,rv,la,ra}`` in SimDet),
            # mirroring the dolfinx per-chamber convention. The Burkhoff active stress is
            # linear in Tmax, so each region resolves to ONE absolute contractility via the
            # shared single-coefficient factor ``fch_active_tmax_factor`` =
            # Tmax_chamber/Tmax_fallback applied to the base form. The SAME helper weights
            # the run_light active-stress monitor; no context reads a bare base. The atrial
            # time course (PK2StressTensor_atr) is preserved.
            _active_ids = [int(r) for r in self.SimDet["active_region"]]
            if self.SimDet.get("plug_active"):
                # CONTRACTILE PLUG (opt-in, LVW only): assemble the active stress over the
                # annular plug regions too. Deliberately NOT done by widening
                # SimDet["active_region"]: waorta_support.build_passive_form_waorta adds the
                # VENTRICULAR passive form over every active_region id (the plug would carry
                # its rubber form AND a second myocardial one), and strain_emit averages the
                # LV strain over active_region (the plug would dilute it). The plug's fibre
                # vectors are unit vectors of ARBITRARY orientation, so this is a bounding
                # experiment on the plug's role, not a physiological annulus.
                if not self.iswaorta or not self.SimDet.get("rubber_region"):
                    raise ValueError("plug_active requires the waorta topology with a "
                                     "non-empty rubber_region")
                _plug_ids = [int(r) for r in self.SimDet["rubber_region"]
                             if int(r) not in _active_ids]
                _active_ids = _active_ids + _plug_ids
                printout("CONTRACTILE PLUG ON: active stress also assembled over "
                         "rubber_region=%s at the full Tmax on the plug's own (arbitrarily "
                         "oriented) fibre vectors" % _plug_ids,
                         self.mesh_me.mpi_comm(), stage="mesh", level="WARN")
            for regionid in _active_ids:
                # Active stress tensor selection differs for atria vs ventricles.
                lv_rid = self.SimDet.get("lv_rid", 10)
                rv_rid = self.SimDet.get("rv_rid", 9)
                la_rid = self.SimDet.get("la_rid", 11)
                ra_rid = self.SimDet.get("ra_rid", 8)

                # Resolve the region to its absolute per-chamber contractility factor
                # (a region not in the chamber map, or without a Tmax_{ch}, -> 1.0).
                _region_ch = {lv_rid: "lv", rv_rid: "rv", la_rid: "la", ra_rid: "ra"}
                _ch = _region_ch.get(regionid)
                factor = fch_active_tmax_factor(self.SimDet, _ch) if _ch is not None else 1.0
                if regionid in (lv_rid, rv_rid):
                    Sactive = activeforms.PK2StressTensor()
                elif regionid in (la_rid, ra_rid):
                    Sactive = activeforms.PK2StressTensor_atr()
                else:
                    Sactive = activeforms.PK2StressTensor()

                if region_cnt == 0:
                    F4 = (
                        factor
                        * df.inner(Fmat * Sactive, df.grad(v_me))
                        * dx_me(int(regionid))
                    )
                    printout(
                        f"Assigning active stress to {regionid}",
                        self.mesh_me.mpi_comm(),
                    )
                else:
                    F4 += (
                        factor
                        * df.inner(Fmat * Sactive, df.grad(v_me))
                        * dx_me(int(regionid))
                    )
                    printout(
                        f"Assigning active stress to {regionid}",
                        self.mesh_me.mpi_comm(),
                    )

                region_cnt += 1
        else:
            Sactive = activeforms.PK2StressTensor()
            F4 = df.inner(Fmat * Sactive, df.grad(v_me)) * dx_me

        Ftotal = F1 if F4 is None else F1 + F4

        def pressure_terms():
            # Two coupling modes: volume constraints (Lagrange multipliers) vs prescribed cavity pressures.
            if not self.ispctrl:
                if self.isLV:
                    return derivative(LV_Wvol, w_me, wtest_me), None
                if self.isBiV:
                    return derivative(LV_Wvol + RV_Wvol, w_me, wtest_me), None
                return None, None

            if self.isLV or self.iswaorta:
                if lvw_swept:
                    # Volume control: the cavity load comes from the monolithic constraint
                    # (its cavity pressure IS the Lagrange multiplier), not from the
                    # prescribed-pressure Constant. The aorta term, if enabled, still applies.
                    Fp_me = uflforms.FCHsweptV0constrainedE()
                    Fp_me = maybe_add_aorta_pressure_term(Fp_me, uflforms, self.SimDet)
                    return None, derivative(Fp_me, w_me, wtest_me)
                Fp_me = uflforms.LVcavitypres()
                if self.iswaorta:
                    Fp_me = maybe_add_aorta_pressure_term(Fp_me, uflforms, self.SimDet)
                else:
                    if self.SimDet.get("aorta_pres"):
                        Fp_me += uflforms.Aortacavitypres()
                return None, derivative(Fp_me, w_me, wtest_me)

            if self.isBiV or self.isFCH:
                if fch_swept_chamber:
                    # Swept-cavity volume control: every SWEPT chamber's load comes from its own
                    # monolithic volume constraint (its cavity pressure = that constraint's
                    # multiplier); the remaining cavities stay prescribed-pressure Constants.
                    swept_set = set(self.fch_swept_chambers())
                    pres_fn = {"lv": uflforms.LVcavitypres, "rv": uflforms.RVcavitypres,
                               "la": uflforms.LAcavitypres, "ra": uflforms.RAcavitypres}
                    Fp = None
                    for ch, fn in pres_fn.items():
                        if ch in swept_set:
                            continue
                        if ch in ("la", "ra") and not self.SimDet.get("fch_fe"):
                            continue
                        term = derivative(fn(), w_me, wtest_me)
                        Fp = term if Fp is None else Fp + term
                    multi = uflforms.parameters.get("fch_swept_multi") or []
                    if multi:
                        for _e in multi:
                            swept_term = derivative(
                                uflforms.FCHsweptV0constrainedE(
                                    endoid=_e["endoid"], pendo=_e["pendo"],
                                    V0=_e["V0"], area=_e["area"]),
                                w_me, wtest_me)
                            Fp = swept_term if Fp is None else Fp + swept_term
                    else:
                        swept_term = derivative(uflforms.FCHsweptV0constrainedE(), w_me, wtest_me)
                        Fp = swept_term if Fp is None else Fp + swept_term
                    return None, Fp
                Fp = derivative(uflforms.LVcavitypres(), w_me, wtest_me) + derivative(
                    uflforms.RVcavitypres(), w_me, wtest_me
                )
                if self.SimDet.get("fch_fe"):
                    Fp += derivative(uflforms.LAcavitypres(), w_me, wtest_me)
                    Fp += derivative(uflforms.RAcavitypres(), w_me, wtest_me)
                return None, Fp

        F2, Fp = pressure_terms()
        if F2 is not None:
            Ftotal += F2
        if Fp is not None:
            Ftotal += Fp

        F3 = None
        if self.SimDet.get("springbc"):
            spring_option = self.SimDet.get("springbc_option")
            if not spring_option:
                if self.iswaorta:
                    spring_option = "waorta"
                elif self.isFCH:
                    spring_option = "fch"
                elif self.isLV:
                    spring_option = "lv"
                elif self.isBiV:
                    spring_option = "biv"

            if spring_option:
                # Spring BC is implemented variationally and may replace rigid-body multipliers/Dirichlet anchoring.
                bc_context = SpringBCContext(
                    displacement=u_me,
                    previous_displacement=u_me_n,
                    test_function=v_me,
                    measure=ds_me,
                    current_normal=n_me,
                    reference_normal=N_me,
                    identity=df.Identity(u_me.ufl_shape[0]),
                    ed_displacement=self.u_me_ED,
                    top_spring_getter=self.get_top_spring if hasattr(self, "get_top_spring") else None,
                )
                F3 = build_spring_form(spring_option, bc_context, self.SimDet)

        if F3 is not None:
            Ftotal += F3

        if bool(self.SimDet.get("auto_rigid_support")) and (self.isLV or self.isFCH):
            # AUTOMATIC minimal rigid-body removal (rigid_support.py): derive the rigid-mode
            # combinations the CONFIGURED spring set fails to constrain (the null space of the
            # 6x6 spring-stiffness Gram on the rigid modes) and gauge-fix EXACTLY those — one Real
            # multiplier per null eigenvector enforcing the zero-projection constraint
            # int(u . phi_k) = 0, the rest pinned to zero (so they add no constraint). Works
            # ALONGSIDE the spring form F3 (kept above): the spring holds the modes it can, the
            # multipliers cover the residual null/near-null modes -> well-posed for ANY spring set,
            # minimal (no over-constraint of modes the springs already hold), no manual BC tuning.
            # Grounded: research_dossier auto_rigid_body_support@mechanics (Farhat-Roux/Felippa-Park
            # rigid-mode null space; Boffi-Brezzi-Fortin saddle-point well-posedness).
            from . import rigid_support as _rs
            _surfs = (_rs.fch_spring_surfaces(self.SimDet) if self.isFCH
                      else _rs.lv_spring_surfaces(self.SimDet))
            _G, _modes, _xc = _rs.spring_rigid_gram(mesh_me, facetboundaries_me, _surfs,
                                                    reference_normal=N_me)
            _nm = _rs.rigid_null_modes(_G)
            printout("auto_rigid_support: spring rank=%d -> %d rigid constraint(s) "
                     "(null eig/max=%s, max_gap=%.1fx)"
                     % (_nm["rank"], _nm["n_null"],
                        np.array2string(_nm["norm_eigenvalues"][_nm["null_mask"]], precision=2),
                        _nm["max_gap"]), comm_me)
            Wrigid = None
            for _k in range(6):
                if _nm["null_mask"][_k]:
                    # phi_k = sum_j V[j,k] * rigid_mode_j ; constrain int(u . phi_k)=0
                    _phi = sum((df.Constant(float(_nm["eigenvectors"][_j, _k])) * _modes[_j]
                                for _j in range(6)), df.as_vector([0.0, 0.0, 0.0]))
                    _term = c_me[_k] * df.inner(u_me, _phi)
                else:
                    # inactive multiplier: pin c_k = 0 (decoupled, well-conditioned)
                    _term = 0.5 * c_me[_k] * c_me[_k]
                Wrigid = _term if Wrigid is None else Wrigid + _term
            F5 = derivative(Wrigid * dx_me, w_me, wtest_me)
            Ftotal += F5
        elif F3 is None and (self.isLV or self.isBiV):
            # Prevent rigid-body drift when the model is not otherwise anchored.
            Wrigid = (
                df.inner(df.as_vector([c_me[0], c_me[1], 0.0]), u_me)
                + df.inner(df.as_vector([0.0, 0.0, c_me[2]]), df.cross(X_me, u_me))
                + df.inner(df.as_vector([c_me[3], 0.0, 0.0]), df.cross(X_me, u_me))
                + df.inner(df.as_vector([0.0, c_me[4], 0.0]), df.cross(X_me, u_me))
            )
            if bool(self.SimDet.get("free_top")):
                # 6th multiplier: constrain Z translation (otherwise a free null
                # mode for the fully traction-free LV).
                Wrigid += df.inner(df.as_vector([0.0, 0.0, c_me[5]]), u_me)
            F5 = derivative(Wrigid, w_me, wtest_me) * dx_me
            Ftotal += F5
        else:
            F5 = None

        # Add stabilization
        if self.discretization == "P1P1":

            h_elem = CellDiameter(mesh_me)

        # Pressure stabilization (equal-order P1P1) is only needed for incompressible/pctrl modes.
        Fs = None
        if self.discretization == "P1P1":
            if self.iswaorta:
                F_mu = build_stabilization_mu_waorta(
                    self.QDG, uflforms, self.matid_me, self.SimDet,
                    region_mu=self._rubber_mu_override,
                )
            else:
                F_mu = df.Function(self.QDG)  # DG0
                self._p1p1_stab_mu = F_mu
                self._p1p1_stab_expr = uflforms.shear_main()
                self._set_p1p1_stab_mu()

            if self.discretization_technique == 1:
                # Technique 1: pressure gradient stabilization in the current configuration.
                Fs = (
                    -(
                        h_elem
                        * h_elem
                        * df.Constant(0.5)
                        / F_mu
                        * J
                        * df.inner(df.inv(Fmat.T) * df.grad(p_me), df.inv(Fmat.T) * df.grad(q_me))
                        * dx_me
                    )
                )
                Ftotal += Fs

        Jac = derivative(F1, w_me, dw_me)
        # Jacobian includes only the terms enabled for this configuration (springbc, pctrl, stabilization, ...).
        if F2 is not None:
            Jac += derivative(F2, w_me, dw_me)
        if Fp is not None:
            Jac += derivative(Fp, w_me, dw_me)

        if F4 is not None:
            Jac += derivative(F4, w_me, dw_me)

        # F3 (spring) and F5 (rigid-body constraints) were historically mutually exclusive
        # (springbc XOR multipliers). With auto_rigid_support BOTH are present (spring + the minimal
        # rigid constraints it leaves), so each Jacobian contribution is added independently.
        if F3 is not None:
            Jac += derivative(F3, w_me, dw_me)
        if F5 is not None:
            Jac += derivative(F5, w_me, dw_me)

        if self.discretization == "P1P1" and Fs is not None:
            Jacs = derivative(Fs, w_me, dw_me)
            Jac += Jacs

        # Initialize LV cavity volume. This slot feeds `LVV0constrainedE`, so it must hold the
        # value of THAT form's own (origin-anchored) functional -- not the open-base-aware
        # physical volume. Seeding the physical volume on a basally trimmed LV asks the solver
        # to deform the reference configuration until the two agree, a spurious perturbation the
        # size of `lv_cavity_constraint_offset()`. Identical on a closed endocardium.
        if not self.ispctrl:
            self.LVCavityvol.assign(uflforms.LVcavityvol())
            if self.isBiV:
                self.RVCavityvol.assign(uflforms.RVcavityvol())

            if self.isBiV:
                self.RVP_cav = uflforms.RVcavitypressure()
                self.RVV_cav = uflforms.RVcavityvol()
        return Ftotal, Jac, bcs_elas

    def _set_p1p1_stab_mu(self):
        """Project the equal-order P1P1 stabilization modulus into the DG0 coefficient
        `self._p1p1_stab_mu`.

        The stabilization is `-h^2/(2*mu) * J * (grad p, grad q)`, so a LARGER
        `SimDet["p1p1_stab_scale"]` means MORE stabilization -> the scale divides mu. It is
        applied to the coefficient VECTOR, never to the UFL graph, so at the default 1.0 the
        assembled form and its values are bit-for-bit the historical ones.

        WHY the knob exists, and what it settled: `shear_main()` is the material's ground-state
        shear modulus -- for Holzapfel-Ogden the isotropic `a` ALONE (59 Pa canonically), while
        the wall's stiffness is carried by the fiber family once fibers engage, so the coefficient
        is both unrepresentative and frozen at build time. That was the standing hypothesis for
        the isochoric-split first-increment failure. It is FALSIFIED by measurement (dossier
        12.18): scales 0.01, 0.1, 1, 3, 10, 30 -- three orders of magnitude, both directions --
        produce a bit-identical failure, and disabling MUMPS null-pivot regularization changes
        nothing either. The cause was the line-search damping floor; see nsolver. The knob is
        kept as the diagnostic that settled it, default 1.0 and byte-identical."""
        scale = float(self.SimDet.get("p1p1_stab_scale", 1.0))
        if scale <= 0.0:
            raise ValueError("SimDet['p1p1_stab_scale'] must be > 0 (got %r)" % (scale,))
        mu_main = df.project(self._p1p1_stab_expr, self.QDG)
        vals = mu_main.vector().get_local()
        if scale != 1.0:
            vals = vals / scale
        self._p1p1_stab_mu.vector().set_local(vals)
        self._p1p1_stab_mu.vector().apply("insert")
        if scale != 1.0:
            from heartlog import announce_once
            from ..utils.log_mpi import get_logger
            if announce_once("MEmodel3:p1p1_stab:%s" % (scale,)):
                get_logger("fe", self.mesh_me.mpi_comm()).warn(
                    "mechanics",
                    "P1P1 pressure stabilization is NON-DEFAULT -- the assembled stabilization "
                    "coefficient no longer equals the historical h^2/(2*mu_ground) form",
                    stab_scale=scale,
                    mu_min=float(vals.min()) if vals.size else float("nan"),
                    mu_max=float(vals.max()) if vals.size else float("nan"))

    def Solver(self):
        # Thin wrapper around NSolver to keep solver configuration centralized.
        solverparams = {
            "Jacobian": self.Jac,
            "F": self.Ftotal,
            "w": self.w_me,
            "boundary_conditions": self.bcs,
            "Type": 0,  # Default
            "mesh": self.mesh_me,
            # mode 1 keeps the primary path quiet; SimDet["newton_verbose"]=2 opts into the
            # per-Newton-iteration alpha/res/rel records (nsolver._newton_linesearch_type0),
            # which is the only way to tell an alpha-floor exhaustion from a max-iter stall.
            "mode": int(self.SimDet.get("newton_verbose", 1)),
        }
        if "max_iter" in list(self.SimDet.keys()):
            # Previously unreachable: NSolver read max_iter from params but Solver() never
            # forwarded it, so every solve ran at the 200-iteration default.
            solverparams.update({"max_iter": int(self.SimDet["max_iter"])})

        if "abs_tol" in list(self.SimDet.keys()):
            solverparams.update({"abs_tol": self.SimDet["abs_tol"]})
        if "rel_tol" in list(self.SimDet.keys()):
            solverparams.update({"rel_tol": self.SimDet["rel_tol"]})
        if "Type" in list(self.SimDet.keys()):
            solverparams.update({"Type": self.SimDet["Type"]})
        # Opt-in globalized (backtracking line-search) Newton for the Type-0 path
        # (e.g. the isovolumic active twitch, where the stock full-step Newton
        # overshoots into a non-finite residual). Off by default -> stock Newton.
        if self.SimDet.get("newton_linesearch", False):
            solverparams.update({"newton_linesearch": True})
            if "newton_alpha_min" in self.SimDet:
                solverparams.update({"newton_alpha_min": self.SimDet["newton_alpha_min"]})
        # Opt-in MUMPS null-pivot detection (ICNTL 24) for the Type-0 direct solve:
        # crosses the (near-)singular equal-order P1P1 saddle-point pivot at the
        # late-systolic over-ejection limit point (DIVERGED_PC_FAILED). Off by default.
        if self.SimDet.get("mumps_null_pivot", False):
            solverparams.update({"mumps_null_pivot": True})
            if "mumps_cntl_3" in self.SimDet:
                solverparams.update({"mumps_cntl_3": self.SimDet["mumps_cntl_3"]})
        solver_eals = NSolver(solverparams)
        return solver_eals

    def get_displacement(self):
        # Select the displacement block by INDEX from the active layout (which varies with
        # incompressibility, p-control, chamber count, springbc, auto_rigid_support and
        # fch_swept_volctrl) rather than unpacking a fixed-arity tuple.
        parts = self._split_state()
        if "rigid" in parts:
            self.c = parts["rigid"]
        u = parts["u"]

        u.rename("u_", "u_")

        return u

    def UpdateVar(self):
        self.w_me_n.assign(self.w_me)

    def Reset(self):
        self.w_me.assign(self.w_me_n)

    def get_pressure_field(self):
        # Pressure is only an unknown for incompressible formulations. Select it by INDEX
        # from the active layout — the trailing blocks (rigid-body multipliers under
        # auto_rigid_support, the FCH swept-cavity multiplier, cavity multipliers) shift the
        # arity, which a fixed 2/3/4-tuple unpack cannot survive.
        parts = self._split_state()
        if "rigid" in parts:
            self.c = parts["rigid"]
        if "p" not in parts:
            raise RuntimeError(
                "get_pressure_field: no pressure unknown in the mechanics mixed space "
                "(compressible formulation, layout {}).".format(self.mechanics_mixed_layout())
            )
        p = parts["p"]

        p.rename("p_", "p_")

        return p

    def get_flow_rate(self):
        Fmat = self.uflforms.Fmat()
        Cmat = Fmat.T * Fmat
        Emat = self.uflforms.Emat()
        J = self.uflforms.J()
        LVendoid = self.SimDet["LVendoid"]
        ds_me = self.ds_me
        dx_me = self.dx_me
        state_obj = self.parameters["state_obj"]

        permeability = self.SimDet.get("permeability")
        p_a = self.SimDet.get("p_a")
        p_v = self.SimDet.get("p_v")
        beta_a = self.SimDet.get("beta_a")
        beta_v = self.SimDet.get("beta_v")

        parts = self._split_state()
        parts_n = self._split_state(self.w_me_n)
        if "rigid" in parts:
            self.c = parts["rigid"]
            self.c_n = parts_n["rigid"]
        u, p = parts["u"], parts["p"]
        u_n, p_n = parts_n["u"], parts_n["p"]
        rv_pendo = []
        lv_pendo = []

        p.rename("p_", "p_")
        Sourceterm = beta_a * (self.p_a - p) * dx - beta_v * (p - self.p_v) * dx

        q = df.assemble((J - 1) * dx_me)
        dimension = u_n.ufl_domain().geometric_dimension()
        Ide = df.Identity(dimension)
        F_n = Ide + df.grad(u_n)
        J_n = df.det(F_n)

        q = df.assemble((J - 1) * dx_me)  # Use displacement to get flow rate

        q1 = df.assemble(
            df.sqrt(df.inner((-permeability * df.grad(p)), (-permeability * df.grad(p)))) * dx_me
        )

        q2 = df.assemble(beta_a * (self.p_a - p) * dx_me)
        return q2

    def get_flow_rate_prev(self):
        Fmat = self.uflforms.Fmat()
        Cmat = Fmat.T * Fmat
        Emat = self.uflforms.Emat()
        J = self.uflforms.J()
        LVendoid = self.SimDet["LVendoid"]
        ds_me = self.ds_me
        dx_me = self.dx_me
        state_obj = self.parameters["state_obj"]

        permeability = self.SimDet.get("permeability")
        p_a = self.SimDet.get("p_a")
        p_v = self.SimDet.get("p_v")
        beta_a = self.SimDet.get("beta_a")
        beta_v = self.SimDet.get("beta_v")

        parts = self._split_state()
        parts_n = self._split_state(self.w_me_n)
        if "rigid" in parts:
            self.c = parts["rigid"]
            self.c_n = parts_n["rigid"]
        u, p = parts["u"], parts["p"]
        u_n, p_n = parts_n["u"], parts_n["p"]
        rv_pendo = []
        lv_pendo = []

        p.rename("p_", "p_")

        dimension = u_n.ufl_domain().geometric_dimension()
        Ide = df.Identity(dimension)
        F_n = Ide + df.grad(u_n)
        J_n = df.det(F_n)
        q = df.assemble((J_n - 1) * dx_me)  # Use displacement to get flow rate

        q1 = df.assemble(
            df.sqrt(df.inner((-permeability * df.grad(p_n)), (-permeability * df.grad(p_n))))
            * dx_me
        )

        q2 = df.assemble(beta_v * (p - self.p_v) * dx_me)

        return q2

    def get_flow_rate_endo(self):
        V1 = self.W.sub(1).collapse()
        Fmat = self.uflforms.Fmat()
        Cmat = Fmat.T * Fmat
        Emat = self.uflforms.Emat()
        J = self.uflforms.J()
        LVendoid = self.SimDet["LVendoid"]
        ds_me = self.ds_me
        dx_me = self.dx_me
        permeability = self.SimDet.get("permeability")
        p_a = self.SimDet.get("p_a")
        p_v = self.SimDet.get("p_v")
        beta_a = self.SimDet.get("beta_a")
        beta_v = self.SimDet.get("beta_v")

        parts = self._split_state()
        if "rigid" in parts:
            self.c = parts["rigid"]
        u, p = parts["u"], parts["p"]
        rv_pendo = []
        lv_pendo = []

        p.rename("p_", "p_")
        Sourceterm = beta_a * (self.p_a - p) * dx - beta_v * (p - self.p_v) * dx
        q = df.assemble((J - 1) * dx_me)

        return q

    def get_mass(self):
        V1 = self.W.sub(1).collapse()

        permeability = self.SimDet.get("permeability")
        p_a = self.SimDet.get("p_a")
        p_v = self.SimDet.get("p_v")
        beta_a = self.SimDet.get("beta_a")
        beta_v = self.SimDet.get("beta_v")
        parts = self._split_state()
        if "rigid" in parts:
            self.c = parts["rigid"]
        u, p = parts["u"], parts["p"]
        rv_pendo = []
        lv_pendo = []

        p.rename("p_", "p_")
        Fmat = self.uflforms.Fmat()
        Cmat = Fmat.T * Fmat
        Emat = self.uflforms.Emat()
        J = self.uflforms.J()
        Sourceterm = beta_a * (self.p_a - p) * dx - beta_v * (p - self.p_v) * dx
        # q = df.assemble(Sourceterm)
        # Q = df.div(permeability*df.grad(p))
        # q = df.assemble(Q * dx)
        # q = df.assemble((J-1) * dx)
        q = J - 1
        m = df.project(q, V1)
        return m

    def get_tmax1(self):
        return self.Tmax1.value

    def get_tmax2(self):
        return self.Tmax2.value

    def get_tmax3(self):
        return self.Tmax3.value

    def get_deformation_gradient(self):
        return self.uflforms.Fmat()

    def get_fiber_strain(self, F_ref):
        return self.uflforms.fiberstrain(F_ref=F_ref)

    def get_fiber_strain_unloaded(self):
        F_Identity = df.Identity(self.get_displacement().ufl_domain().geometric_dimension())
        return self.uflforms.fiberstrain(F_ref=F_Identity)

    def get_imp(self):
        return self.uflforms.IMP()

    def get_imp2(self):
        return self.uflforms.IMP2()

    def get_fiber_stress(self):
        return self.uflforms.fiberstress() + self.activeforms.fiberstress()

    def get_fiber_cauchy_stress(self):
        """Total (passive+active) fiber-direction CAUCHY (true) stress, resolved on the
        DEFORMED fiber direction a = F*f0 (per unit deformed area). This is the
        physiological "fiber wall stress" the literature reports (e.g. Genet et al. 2014,
        normal human LV: ED ~1.9 kPa, ES ~15 kPa) -- unlike get_fiber_stress(), which
        returns the fiber *PK2* component and carries the incompressibility Lagrange
        pressure, so it is neither a true stress nor sign-comparable to those values.

        sigma = (1/J) F (S_passive+vol + S_active) F^T ; sigma_ff = (F f0).sigma.(F f0)/|F f0|^2.
        """
        F = self.uflforms.Fmat()
        J = self.uflforms.J()
        f0 = self.f0_me
        S_total = self.uflforms.PK2() + self.activeforms.PK2StressTensor()
        sigma = (1.0 / J) * F * S_total * F.T
        a = F * f0
        i, j = ufl.indices(2)
        return (a[i] * sigma[i, j] * a[j]) / df.inner(a, a)

    def get_fiber_cauchy_stress_dev(self):
        """Deviatoric fiber-direction Cauchy stress = the hydrostatic-removed companion of
        get_fiber_cauchy_stress() (sigma_dev = sigma - (1/3) tr(sigma) I), for the case the
        incompressibility pressure should be excluded from the reported wall stress."""
        F = self.uflforms.Fmat()
        J = self.uflforms.J()
        f0 = self.f0_me
        d = self.mesh_me.geometry().dim()
        S_total = self.uflforms.PK2() + self.activeforms.PK2StressTensor()
        sigma = (1.0 / J) * F * S_total * F.T
        sigma_dev = sigma - (1.0 / 3.0) * df.tr(sigma) * df.Identity(d)
        a = F * f0
        i, j = ufl.indices(2)
        return (a[i] * sigma_dev[i, j] * a[j]) / df.inner(a, a)

    def get_fiber_natural_strain(self, F_ED, basis_dir, AHA_segments):
        F_n = self.get_deformation_gradient()

        return self.activeforms.CalculateFiberNaturalStrain(
            F_=F_n, F_ref=F_ED, e_fiber=basis_dir, VolSeg=AHA_segments
        )

    def get_fiber_biot_strain(self, F_ED, basis_dir, AHA_segments):
        F_n = self.get_deformation_gradient()

        return self.activeforms.CalculateFiberBiotStrain(
            F_=F_n, F_ref=F_ED, e_fiber=basis_dir, VolSeg=AHA_segments
        )

    def get_fiber_green_strain(self, F_ED, basis_dir, AHA_segments):
        F_n = self.get_deformation_gradient()

        return self.activeforms.CalculateFiberGreenStrain(
            F_=F_n, F_ref=F_ED, e_fiber=basis_dir, VolSeg=AHA_segments
        )

    def get_laplace_solution(self):
        # Optional Laplace solve used for some rule-based fields (e.g., transmural coordinates).
        if not self.SimDet.get("enable_laplace_solver"):
            return None
        if self.isFCH:
            return
        elif self.iswaorta:
            return laplace_waorta(self.uflforms)
        elif self.isLV or self.isBiV:
            return self.uflforms.solveLaplaceEquation_ideal()

    def get_lv_pressure(self):
        if self.iswaorta and self.SimDet.get("lvw_swept_volctrl"):
            # Volume control: the cavity pressure is the Lagrange multiplier, NOT the
            # prescribed-pressure Constant (which is unused and stays at its last value).
            return self.get_fch_swept_pressure()
        if self.ispctrl:
            return float(self.LVCavitypres)
        else:
            return self.uflforms.LVcavitypressure()

    def get_fch_swept_pressure(self, chamber=None):
        """Cavity pressure (Pa) of the swept FCH chamber under volume control = the value of
        its Lagrange-multiplier DOF. MPI-safe read mirroring LVcavitypressure. Only meaningful
        when SimDet['fch_swept_volctrl'] is set.

        The block index comes from `mechanics_mixed_layout()`, NOT from "the last sub-space":
        with auto_rigid_support the rigid-body multipliers trail the swept multiplier, so a
        last-block read would silently return a rigid-body DOF as the cavity pressure."""
        layout = self.mechanics_mixed_layout()
        blocks = [n for n in layout if n.startswith("fch_swept")]
        if not blocks:
            raise RuntimeError(
                "get_fch_swept_pressure: no swept-cavity multiplier in the mechanics mixed "
                "space (layout {}); set SimDet['fch_swept_volctrl'].".format(layout)
            )
        if chamber is None:
            if len(blocks) > 1:
                raise ValueError(
                    "get_fch_swept_pressure() is ambiguous with %d swept cavities %s -- name "
                    "the chamber. Returning one of several multipliers as 'the' cavity pressure "
                    "is exactly the silent mix-up the layout indexing exists to prevent."
                    % (len(blocks), blocks))
            name = blocks[0]
        else:
            name = "fch_swept" if len(blocks) == 1 else "fch_swept_%s" % str(chamber).lower()
            if name not in layout:
                raise ValueError("chamber %r is not swept (layout %s)" % (chamber, layout))
        W = self.w_me.function_space()
        comm = W.mesh().mpi_comm()
        dofmap = W.sub(layout.index(name)).dofmap()
        try:
            val_dof = dofmap.cell_dofs(0)[0]
            val_local = self.w_me.vector()[val_dof]
        except IndexError:
            val_local = 0.0
        return df.MPI.sum(comm, float(val_local))

    def _lv_endo_cavity_volume(self):
        """Single-chamber LV endocardial cavity volume, open-base-aware. The origin-anchored
        divergence-theorem form (`Forms.LVcavityvol`) is exact only for a CLOSED endocardial
        surface (offset-invariant there); a basally-trimmed mesh's endocardium is open at the
        mitral-annulus cut, so the base-facet-centroid-offset form (`Forms.LVcavityvol_mvb`) is
        required instead — see `self.lv_open_base` (detected once, mesh-driven, in __init__)."""
        if self.lv_open_base:
            return self.uflforms.LVcavityvol_mvb()
        return self.uflforms.LVcavityvol()

    def lv_cavity_constraint_offset(self):
        """Signed offset (mL) between the cavity volume the VOLUME-CONTROL CONSTRAINT
        actually enforces and the physical volume `get_lv_volume()` reports, in the
        current deformed state: `Forms.LVcavityvol() - Forms.LVcavityvol_mvb()`.

        Zero for a closed endocardium (and for every non-single-chamber-LV topology),
        so callers stay on the historical single-solve path there.

        Why this exists rather than an open-base variant of `Forms.LVV0constrainedE`:
        that constraint's UFL term is `Wvol = pendo * (V0 - V(u))`, whose `d/d(pendo)`
        enforces the volume and whose `d/du` supplies the endocardial pressure load.
        With the ORIGIN-ANCHORED `V(u)`, `d/du` is exactly the uniform follower
        pressure `∮ v·n da` — the correct physics — but `d/d(pendo)` then constrains
        the origin-anchored integral, which is NOT the cavity volume on an open base.
        Substituting the centroid-offset form would fix the volume but break the load:
        `V_mvb = V_origin + (1/3) b·A` with `A = ∮_endo n da`, and on a trimmed mesh
        `A` is the (nonzero) mitral-orifice area vector that varies with `u`, so the
        extra `(1/3) b·δA` term is a SPURIOUS traction. Measured on the bundled trimmed
        LV mesh: `A = [0.034, -0.238, 20.480]`, `|A| = 20.48` — decisively nonzero, and
        the identity above reproduces the offset to 8 decimals.

        So the residual form is left alone (correct traction, unchanged Jacobian, no
        re-JIT) and the PRESCRIBED target is corrected by this offset instead. Because
        the offset depends on the state it induces a fixed point, which converges
        linearly and fast (~1e-6 mL in 5 iterations on the bundled mesh); see
        `scripts/fe_active_twitch_legacy.py::_solve_at_volume`."""
        if self.ispctrl or not (self.isLV and self.lv_open_base):
            return 0.0
        return float(self.uflforms.LVcavityvol()) - float(self.uflforms.LVcavityvol_mvb())

    def lv_constraint_volume(self):
        """The value the volume-control constraint functional takes in the CURRENT state —
        i.e. what `LVCavityvol` must hold for the present configuration to satisfy the
        constraint exactly. Equals `get_lv_volume()` on a closed endocardium.

        Use this (not `get_lv_volume()`) to initialize `LVCavityvol`: seeding the physical
        volume into the constraint slot on an open base asks the solver to deform the
        reference configuration until the ORIGIN-ANCHORED integral equals the physical
        volume, a spurious perturbation the size of `lv_cavity_constraint_offset()`."""
        return float(self.get_lv_volume()) + self.lv_cavity_constraint_offset()

    def get_lv_volume(self):
        if self.ispctrl:
            return self.LV_closedsurf()
        elif self.isLV:
            return self._lv_endo_cavity_volume()
        else:
            return self.uflforms.LVcavityvol()

    def LV_closedsurf(self):
        # Cavity volume formulas depend on topology (ideal LV vs waorta vs fch variants).
        if self.iswaorta:
            return lv_volume_waorta(self.uflforms)
        elif self.isFCH:
            return self.uflforms.LVcavityvol_fch()
        elif self.isLV:
            return self._lv_endo_cavity_volume()
        elif self.isBiV:
            return self.uflforms.LVcavityvol()

    def get_top_spring(self):
        return self.uflforms.topspringbc()

    def get_rv_pressure(self):
        if self.ispctrl:
            return float(self.RVCavitypres)
        else:
            return self.uflforms.RVcavitypressure()

    def get_rv_volume(self):
        if self.isBiV:
            if "springbc" in list(self.SimDet.keys()) and self.SimDet["springbc"]:
                return self.uflforms.RVcavityvol()
            else:
                return self.uflforms.RVcavityvol()
        elif self.isFCH:
            if "springbc" in list(self.SimDet.keys()) and self.SimDet["springbc"]:
                return self.uflforms.RVcavityvol_fch()
            else:
                return

    def get_la_pressure(self):
        if self.SimDet.get("fch_fe"):
            return float(self.LACavitypres)

    def get_la_volume(self):
        if self.SimDet.get("fch_fe"):
            return self.uflforms.LAcavityvol_fch()

    def get_ra_pressure(self):
        if self.SimDet.get("fch_fe"):
            return float(self.RACavitypres)

    def get_ra_volume(self):
        if self.SimDet.get("fch_fe"):
            return self.uflforms.RAcavityvol_fch()

    def get_active_stress(self):
        i, j = ufl.indices(2)
        Sactive = self.activeforms.PK2StressTensor()
        Sactive_ = df.project(self.f0_me[i] * Sactive[i, j] * self.f0_me[j], self.QDG)
        Sactive_.rename("Sact", "Sact")

        return Sactive_

    def get_activation_time(self):
        return self.activeforms.Get_t_a()

    def get_deformed_basis(self, params):
        default_params = {
            "LVangle": [0, 0],
            "SPangle": [0, 0],
            "RVangle": [0, 0],
            "meshName": "EDfile",
        }
        default_params.update(params)

        LVangle = default_params["LVangle"]
        SPangle = default_params["SPangle"]
        RVangle = default_params["RVangle"]
        meshName = default_params["meshName"]

        #  - - - - - - - - - - - -- - - - - - - - - - - - - - - -- - - - - - -
        mesh_me = self.mesh_me
        facetboundaries_me = self.facetboundaries_me
        deg_me = self.deg_me
        LVendoid = self.SimDet["LVendoid"]
        RVendoid = self.SimDet["RVendoid"]
        epiid = self.SimDet["epiid"]
        isLV = self.isLV

        meshDispFunc = df.VectorFunctionSpace(mesh_me, "CG", 1)
        VQuadelem_me = df.VectorElement(
            "Quadrature", mesh_me.ufl_cell(), degree=deg_me, quad_scheme="default"
        )
        VQuadelem_me._quad_scheme = "default"
        fiberFS = df.FunctionSpace(mesh_me, VQuadelem_me)

        # Generate fibers on the currently deformed mesh, then copy into the reference-mesh quadrature space.
        meshDisplacement = df.project(self.get_displacement(), meshDispFunc)
        deformedMesh, deformedBoundary = update_mesh(
            mesh=mesh_me, displacement=meshDisplacement, boundaries=facetboundaries_me
        )

        outputfolder = self.parameters["outputfolder"]
        folderName = self.parameters["foldername"].rstrip("/")

        EDmeshData = {
            "epiid": epiid,
            "rvid": RVendoid,
            "lvid": LVendoid,
            "LVangle": LVangle,  # [0, 0],
            "Septangle": [0, 0],
            "RVangle": [0, 0],
            "isepiflip": False,
            "isendoflip": False,
            "iscaling": False,
            "mesh": deformedMesh,
            "facets": deformedBoundary,
            "mFileName": os.path.join(outputfolder, folderName, "deformation_unloadED") + "/",
            "isLV": isLV,
            "meshName": meshName,
        }

        eCC_ED, eLL_ED, eRR_ED = create_EDFibers(EDmeshData)

        # Copy directional field from functionspace with mesh_me to deformedmesh
        eCC = df.Function(fiberFS)
        eRR = df.Function(fiberFS)
        eLL = df.Function(fiberFS)
        eCC.vector()[:] = eCC_ED.vector().get_local()[:]
        eRR.vector()[:] = eRR_ED.vector().get_local()[:]
        eLL.vector()[:] = eLL_ED.vector().get_local()[:]

        return eCC, eRR, eLL, deformedMesh, deformedBoundary
