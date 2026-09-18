import dolfin as df
from collections.abc import Iterable
from numbers import Integral
from ufl import cofac, indices, ln

from .aorta_defaults import fill_aorta_params

DEFAULT_KAPPA_MAIN_MULTIPLIER = 1e2
DEFAULT_LAPLACE_BCS = {
    # Laplace coordinate solves use simple base/apex Dirichlet values; these are model-specific defaults.
    "laplace_fch_bc": {"base": 0.25, "apex": 1.0},
    "laplace_waorta_bc": {"base": 0.25, "apex": 1.0},
    "laplace_ideal_bc": {"base": 0.5, "apex": 2.0},
}


class Forms(object):
    def __init__(self, params):
        self.parameters = self.default_parameters()
        self.parameters.update(params)
        # Normalize aorta model configuration so downstream callers can assume required keys exist.
        self.parameters["aorta params"] = fill_aorta_params(
            self.parameters.get("aorta params")
        )
        # Only materialize Laplace defaults if solver is requested
        self._laplace_enabled = bool(self.parameters.get("enable_laplace_solver"))
        if self._laplace_enabled:
            for key, default_val in DEFAULT_LAPLACE_BCS.items():
                self.parameters.setdefault(key, default_val.copy())
        self._surface_measure_cache = None
        self._surface_degree = None
        self._const_minus_third = df.Constant(-1.0 / 3.0)

        mat_model = self.parameters["material model"]["Name"]
        # Material models consume `F` directly; store it once to keep consistent use of growth-corrected kinematics.
        self.parameters.update({"F": self.Fe()})

        if mat_model == "Guccione":
            from .GuccionePas import GuccionePas as Passive
        elif mat_model == "HolzapfelOgden":
            from .holzapfelogden import HolzapfelOgden as Passive
        else:
            raise ValueError("Material model not implemented")

        self.passiveforms = Passive(self.parameters)
        self.matparams = self.passiveforms.get_material_params()

    def get_material_params(self):
        return self.parameters["material params"]

    def default_parameters(self):
        defaults = {
            "material model": {"Name": "Guccione"},
            "Kappa": 1.0e3,
            "incompressible": True,
            "surface_quadrature_degree": 4,
            "kappa_main_multiplier": DEFAULT_KAPPA_MAIN_MULTIPLIER,
            "aorta params": fill_aorta_params(),
            "enable_laplace_solver": False,
        }
        return defaults

    def _laplace_solver_enabled(self):
        return self._laplace_enabled

    def _surface_quadrature_degree(self):
        return self.parameters.get("surface_quadrature_degree", 4)

    def _surface_measure(self):
        current_degree = self._surface_quadrature_degree()
        if (
            self._surface_measure_cache is None
            or self._surface_degree != current_degree
        ):
            # Cache ds(...) because it is reused for repeated cavity volume/pressure integrals.
            self._surface_degree = current_degree
            self._surface_measure_cache = df.ds(
                domain=self.parameters["mesh"],
                subdomain_data=self.parameters["facetboundaries"],
                metadata={"quadrature_degree": current_degree},
            )
        return self._surface_measure_cache

    def _surface_sum(self, ids, measure=None):
        if ids is None:
            return None
        ds = measure if measure is not None else self._surface_measure()
        if isinstance(ids, Iterable) and not isinstance(ids, (str, bytes, Integral)):
            # Allow callers to specify multi-surface cavities via a list/tuple of facet markers.
            surface = None
            for piece in (ds(idx) for idx in ids if idx is not None):
                surface = piece if surface is None else surface + piece
            return surface
        return ds(ids)

    def _normalize(self, vector):
        return vector / df.sqrt(df.inner(vector, vector))

    def _normalized_push_forward(self, F, vector):
        return self._normalize(F * vector)

    def _assemble_volume(self, surface, offset=None):
        if surface is None:
            raise ValueError("Surface measure required for volume computation.")
        # Volume from surface integral using a divergence-theorem identity on the (deformed) cavity boundary.
        u = self.parameters["displacement_variable"]
        mesh = self.parameters["mesh"]
        X = df.SpatialCoordinate(mesh)
        F = self.Fmat()
        N = self.parameters["facet_normal"]
        displacement = X + u if offset is None else X + u - offset
        integrand = self._const_minus_third * df.inner(
            df.det(F) * df.dot(df.inv(F).T, N), displacement
        )
        return df.assemble(
            integrand * surface,
            form_compiler_parameters={"representation": "uflacs"},
        )

    def _pressure_integrand(self):
        # Follower cavity-pressure load: the deformed surface normal uses the TOTAL
        # deformation gradient (Nanson J*F^-T*N with F = Fmat, J = det(Fmat)), the same
        # kinematics as the cavity-volume forms (cofac(Fmat)). With an ACTIVE-STRAIN growth
        # tensor self.J() = det(Fe) would instead carry the cell-Quadrature activation onto
        # this facet integral (untabulatable); det(Fmat) is purely CG-displacement and is
        # byte-identical to self.J() when growth_tensor is None (Fe == Fmat).
        F = self.Fmat()
        J = df.det(F)
        u = self.parameters["displacement_variable"]
        N = self.parameters["facet_normal"]
        return df.inner(J * df.inv(F.T) * N, u)

    def _laplace_bc(self, param_name, default_base, default_apex):
        values = self.parameters.get(param_name, {})
        base = values.get("base", default_base)
        apex = values.get("apex", default_apex)
        return base, apex

    def PassiveMatSEF(self, Ea=None):
        return self.passiveforms.PassiveMatSEF(Ea=Ea) + self.Wvolumetric()

    def PassiveRubSEF(self):
        return self.passiveforms.PassiveRubSEF() + self.Wvolumetric_main()

    def PassiveAoCapSEF(self):
        """The aortic-valve cap (waorta region 3) under its OWN law (``aocap_law``),
        with the same bulk penalty the plug carries."""
        return self.passiveforms.PassiveAoCapSEF() + self.Wvolumetric_main()

    def PassiveAnnulusSEF(self):
        """The fibrous annulus (waorta regions ``annulus_regions``, default 4 + 5) under its
        OWN law (``annulus_law``), with the same bulk penalty the plug carries."""
        return self.passiveforms.PassiveAnnulusSEF() + self.Wvolumetric_main()

    def _shear_model(self, kind):
        if kind == "annulus":
            return self.passiveforms.shear_modulus_annulus()
        if kind == "main":
            return self.passiveforms.shear_modulus_main()
        if kind == "rubber":
            return self.passiveforms.shear_modulus_rubber()
        if kind == "aocap":
            return self.passiveforms.shear_modulus_aocap()
        if kind == "aorta":
            model_name = self.parameters.get("aorta params", {}).get("Name")
            if model_name == "NeoHookean":
                return self.parameters["aorta params"]["mu"]
            if model_name == "Delfino":
                return self.passiveforms.shear_modulus_delfino()
            if model_name == "HGO_twofiber":
                return self.passiveforms.shear_modulus_hgo_twofiber_fdc()
            if model_name == "HGO_fourfiber":
                # Alternative: shmod_HGO_fourfiber_minanalytic()
                return self.passiveforms.shear_modulus_hgo_fourfiber_max()
            return None
        raise ValueError(f"Unknown shear kind: {kind}")

    def shear_main(self):
        return self._shear_model("main")

    def shear_rubber(self):
        return self._shear_model("rubber")

    def shear_aocap(self):
        return self._shear_model("aocap")

    def shear_annulus(self):
        return self._shear_model("annulus")

    def shear_aorta(self):
        return self._shear_model("aorta")

    def aorta_strain_energy(self):
        model_name = self.parameters.get("aorta params", {}).get("Name")
        volumetric = self.Wvolumetric_K()
        model_map = {
            "NeoHookean": self.passiveforms.aorta_strain_energy_neohookean,
            "Delfino": self.passiveforms.aorta_strain_energy_delfino,
            "HGO_twofiber": self.passiveforms.aorta_strain_energy_hgo_twofiber,
            "HGO_fourfiber": self.passiveforms.aorta_strain_energy_hgo_fourfiber,
        }
        passive = model_map.get(model_name, self.passiveforms.aorta_strain_energy_neohookean)
        return passive() + volumetric

    def PK1(self):
        return self.passiveforms.PK1() + self.PK1volumetric()

    def PK2(self):
        return df.inv(self.Fmat()) * self.PK1()

    def sigma(self):
        return (1.0 / self.J()) * self.PK1() * self.Fmat().T

    def Fmat(self):
        u = self.parameters["displacement_variable"]
        d = u.ufl_domain().geometric_dimension()
        I = df.Identity(d)
        F = I + df.grad(u)
        return F

    def Fe(self):
        Fg = self.parameters["growth_tensor"]
        F = self.Fmat()
        return F if Fg is None else F * df.inv(Fg)

    def Emat(self):
        u = self.parameters["displacement_variable"]
        d = u.ufl_domain().geometric_dimension()
        I = df.Identity(d)
        F = self.Fe()
        i, j, k = indices(3)
        return 0.5 * (df.as_tensor(F[k, i] * F[k, j] - I[i, j], (i, j)))

    def J(self):
        F = self.Fe()
        return df.det(F)

    def Wvolumetric(self):
        isincomp = self.parameters["incompressible"]
        u = self.parameters["displacement_variable"]
        d = u.geometric_dimension()
        I = df.Identity(d)
        F = I + df.grad(u)
        F = df.variable(F)
        J = df.det(F)

        if isincomp:
            p = self.parameters["pressure_variable"]
            Wvolumetric = -1.0 * p * (J - 1.0)
        else:
            Kappa = self.parameters["Kappa"]
            Wvolumetric = Kappa / 2.0 * (J - 1.0) ** 2.0

        return Wvolumetric

    def Wvolumetric_K(self):
        u = self.parameters["displacement_variable"]
        d = u.geometric_dimension()
        I = df.Identity(d)
        F = df.variable(I + df.grad(u))
        J = df.det(F)
        Kappa = self.parameters["Kappa"]
        return Kappa / 2.0 * (ln(J)) ** 2.0

    def Wvolumetric_main(self):
        u = self.parameters["displacement_variable"]
        d = u.geometric_dimension()
        I = df.Identity(d)
        F = df.variable(I + df.grad(u))
        J = df.det(F)

        # Use a separate bulk penalty for "main" regions to avoid locking rubber/aorta submodels.
        bulk_scale = self.parameters.get("kappa_main_multiplier", 1e2)
        Kappa = self.parameters["Kappa"] * bulk_scale
        Wvolumetric = Kappa / 2.0 * (ln(J)) ** 2.0
        return Wvolumetric

    def PK1volumetric(self):
        isincomp = self.parameters["incompressible"]
        u = self.parameters["displacement_variable"]
        d = u.geometric_dimension()
        I = df.Identity(d)
        F = I + df.grad(u)
        F = df.variable(F)
        J = df.det(F)

        if isincomp:
            Wvolumetric = -self.parameters["pressure_variable"] * (J - 1.0)
        else:
            Kappa = self.parameters["Kappa"]
            Wvolumetric = Kappa / 2.0 * (J - 1.0) ** 2.0
        return df.diff(Wvolumetric, F)

    def LVcavityvol_mvb(self):  # cavity volume for lv with moving base
        u = self.parameters["displacement_variable"]
        N = self.parameters["facet_normal"]
        mesh = self.parameters["mesh"]
        X = df.SpatialCoordinate(mesh)
        F = self.Fmat()
        top_surface = self._surface_sum(self.parameters["topid"])
        area = df.assemble(1.0 * top_surface)
        vol_x = df.assemble((X[0] + u[0]) * top_surface) / area
        vol_y = df.assemble((X[1] + u[1]) * top_surface) / area
        vol_z = df.assemble((X[2] + u[2]) * top_surface) / area
        b = df.Constant((vol_x, vol_y, vol_z))

        surface = self._surface_sum(self.parameters["LVendoid"])
        return self._assemble_volume(surface, offset=b)

    def LVcavityvol_waorta(self):  # cavity volume for lv with aorta
        surface_ids = [
            self.parameters["LVendoid"],
            self.parameters["aortic_valvep"],
            self.parameters["mitral_valvep"],
        ]
        surface = self._surface_sum(surface_ids)
        return self._assemble_volume(surface)

    def _assemble_fch_cavity_volume(self, surface_ids):
        u = self.parameters["displacement_variable"]
        N = self.parameters["facet_normal"]
        mesh = self.parameters["mesh"]
        X = df.SpatialCoordinate(mesh)
        ds = self._surface_measure()
        F = self.Fmat()
        integrand = self._const_minus_third * df.inner(df.det(F) * df.dot(df.inv(F).T, N), X + u)
        surface = self._surface_sum(surface_ids, measure=ds)
        return df.assemble(integrand * surface, form_compiler_parameters={"representation": "uflacs"})

    def LVcavityvol_fch(self):  # cavity volume for lv in fch mesh
        surface_ids = [self.parameters["LVendoid"]]
        if self.parameters.get("aortic_valvep") and self.parameters.get("mitral_valvep"):
            surface_ids += [self.parameters["aortic_valvep"], self.parameters["mitral_valvep"]]
        return self._assemble_fch_cavity_volume(surface_ids)

    def RVcavityvol_fch(self):  # cavity volume for rv in fch mesh
        surface_ids = [self.parameters["RVendoid"]]
        if self.parameters.get("pulmonary_valvep") and self.parameters.get("tricuspid_valvep"):
            surface_ids += [self.parameters["pulmonary_valvep"], self.parameters["tricuspid_valvep"]]
        return self._assemble_fch_cavity_volume(surface_ids)

    def LAcavityvol_fch(self):  # cavity volume for la in fch mesh
        surface_ids = [self.parameters["LAendoid"]]
        return self._assemble_fch_cavity_volume(surface_ids)

    def RAcavityvol_fch(self):  # cavity volume for ra in fch mesh
        surface_ids = [self.parameters["RAendoid"]]
        return self._assemble_fch_cavity_volume(surface_ids)

    def topspringbc(self):  # for v_base computation
        N = self.parameters["facet_normal"]
        ds = self._surface_measure()
        pe = self.parameters["lv_constrained_pres"]
        F = self.Fmat()
        J = self.J()

        JFN = J * df.inv(F.T) * N
        JFN_norm = df.sqrt(df.dot(JFN, JFN))

        JFN_x = df.assemble(pe * JFN[0] * ds(self.parameters["LVendoid"]))
        JFN_y = df.assemble(pe * JFN[1] * ds(self.parameters["LVendoid"]))
        JFN_z = df.assemble(pe * JFN[2] * ds(self.parameters["LVendoid"]))

        JFN_all = df.as_vector((JFN_x, JFN_y, JFN_z))
        int_JFN = df.assemble(JFN_norm * ds(self.parameters["topid"]))

        return JFN_norm / int_JFN, JFN_all

    def LVcavityvol(self):
        surface = self._surface_sum(self.parameters["LVendoid"])
        return self._assemble_volume(surface)

    def RVcavityvol(self):
        surface = self._surface_sum(self.parameters["RVendoid"])
        return self._assemble_volume(surface)

    def LVcavitypressure(self):
        W = self.parameters["mixedfunctionspace"]
        w = self.parameters["mixedfunction"]

        comm = W.mesh().mpi_comm()
        dofmap = W.sub(self.parameters["LVendo_comp"]).dofmap()
        val_dof = dofmap.cell_dofs(0)[0]

        try:
            val_local = w.vector()[val_dof]
        except IndexError:
            val_local = 0.0

        pressure = df.MPI.sum(comm, val_local)

        return pressure

    def LVcavitypres(self):
        pe = self.parameters["lv_constrained_pres"]
        surface = self._surface_sum(self.parameters["LVPid"])
        return pe * self._pressure_integrand() * surface

    def LAcavitypres(self):
        pe = self.parameters["la_constrained_pres"]
        surface = self._surface_sum(self.parameters["LAendoid"])
        return pe * self._pressure_integrand() * surface

    def RAcavitypres(self):
        pe = self.parameters["ra_constrained_pres"]
        surface = self._surface_sum(self.parameters["RAendoid"])
        return pe * self._pressure_integrand() * surface

    def Aortacavitypres(self):
        pe = self.parameters["aorta_constrained_pres"]
        surface = self._surface_sum(self.parameters["aortaid"])
        return pe * self._pressure_integrand() * surface

    def RVcavitypres(self):
        pe = self.parameters["rv_constrained_pres"]
        if "RVPid" not in self.parameters:
            self.parameters["RVPid"] = self.parameters["RVendoid"]

        surface = self._surface_sum(self.parameters["RVPid"])
        return pe * self._pressure_integrand() * surface

    def RVcavitypressure(self):
        W = self.parameters["mixedfunctionspace"]
        w = self.parameters["mixedfunction"]

        comm = W.mesh().mpi_comm()
        dofmap = W.sub(self.parameters["RVendo_comp"]).dofmap()
        val_dof = dofmap.cell_dofs(0)[0]

        try:
            val_local = w.vector().get_local()[val_dof]
        except IndexError:
            val_local = 0.0

        pressure = df.MPI.sum(comm, val_local)

        return pressure

    def LVV0constrainedE(self):
        mesh = self.parameters["mesh"]
        u = self.parameters["displacement_variable"]
        dsendo = self._surface_sum(self.parameters["LVendoid"])
        area = self.parameters["LVendo_area"]
        pendo = self.parameters["lv_volconst_variable"]
        V0 = self.parameters["lv_constrained_vol"]

        X = df.SpatialCoordinate(mesh)
        x = u + X

        F = self.Fmat()
        N = self.parameters["facet_normal"]
        n = cofac(F) * N

        V_u = self._const_minus_third * df.inner(x, n)
        Wvol = (df.Constant(1.0) / area * pendo * V0 * dsendo) - (pendo * V_u * dsendo)
        return Wvol

    def RVV0constrainedE(self):
        mesh = self.parameters["mesh"]
        u = self.parameters["displacement_variable"]
        dsendo = self._surface_sum(self.parameters["RVendoid"])
        pendo = self.parameters["rv_volconst_variable"]
        V0 = self.parameters["rv_constrained_vol"]

        X = df.SpatialCoordinate(mesh)
        x = u + X

        F = self.Fmat()
        N = self.parameters["facet_normal"]
        n = cofac(F) * N

        area = df.assemble(
            df.Constant(1.0) * dsendo,
            form_compiler_parameters={"representation": "uflacs"},
        )
        V_u = self._const_minus_third * df.inner(x, n)
        Wvol = (df.Constant(1.0 / area) * pendo * V0 * dsendo) - (pendo * V_u * dsendo)

        return Wvol

    def FCHsweptV0constrainedE(self, endoid=None, pendo=None, V0=None, area=None):
        """Monolithic cavity-volume constraint for ONE swept FCH chamber: the cavity
        pressure is the Lagrange multiplier ``pendo`` (an unknown solved WITH the
        displacement) enforcing the swept cavity volume = ``V0``. Mirrors
        LVV0constrainedE/RVV0constrainedE but the chamber is selected at run time by
        ``fch_swept_endoid`` so the isovolumic twitch holds any one chamber's volume
        robustly through the high-active-tension states the partitioned pressure
        root-find could not reach (SOTA: monolithic 3D-0D volume constraint)."""
        mesh = self.parameters["mesh"]
        u = self.parameters["displacement_variable"]
        # Explicit arguments let a MULTI-chamber configuration build one constraint per swept
        # cavity from the same functional; omitted, the single-chamber parameters are used and
        # the form is identical to the one-cavity version.
        dsendo = self._surface_sum(self.parameters["fch_swept_endoid"] if endoid is None else endoid)
        pendo = self.parameters["fch_swept_volconst_variable"] if pendo is None else pendo
        V0 = self.parameters["fch_swept_constrained_vol"] if V0 is None else V0

        X = df.SpatialCoordinate(mesh)
        x = u + X
        F = self.Fmat()
        N = self.parameters["facet_normal"]
        n = cofac(F) * N

        area = (self.parameters["fch_swept_area"] if area is None else area)
        V_u = self._const_minus_third * df.inner(x, n)
        Wvol = (df.Constant(1.0) / area * pendo * V0 * dsendo) - (pendo * V_u * dsendo)
        return Wvol

    def fiberstress(self):
        F = self.Fmat()
        J = self.J()
        PK1 = self.PK1()

        Tca = (1.0 / J) * PK1 * F.T
        Sca = df.inv(F) * PK1

        f0 = self.parameters["fiber"]
        i, j = indices(2)
        return f0[i] * Sca[i, j] * f0[j]

    def fiberstrain(self, F_ref):
        u = self.parameters["displacement_variable"]
        d = u.ufl_domain().geometric_dimension()
        I = df.Identity(d)

        F = self.Fe() * df.inv(F_ref)
        f0 = self.parameters["fiber"]
        i, j, k = indices(3)
        Emat = 0.5 * (df.as_tensor(F[k, i] * F[k, j] - I[i, j], (i, j)))

        return f0[i] * Emat[i, j] * f0[j]

    def fiberwork(self, F_ref):
        F = self.Fe() * df.inv(F_ref)

        J = self.J()
        PK1 = self.PK1()

        Tca = (1.0 / J) * PK1 * F.T
        Sca = df.inv(F) * PK1

        f0 = self.parameters["fiber"]
        Emat = self.Emat()

        i, j = indices(2)
        f = f0[i] * Sca[i, j] * f0[j]
        s = f0[i] * Emat[i, j] * f0[j]

        return df.dot(f, s)

    def _solve_laplace(self, bc_ids, base_val, apex_val):
        mesh = self.parameters["mesh"]
        facets = self.parameters["facetboundaries"]
        V = df.FunctionSpace(mesh, df.FiniteElement("Lagrange", mesh.ufl_cell(), 1))

        bc_bas = [df.DirichletBC(V, df.Constant(base_val), facets, id_) for id_ in bc_ids]
        bc_apx = [df.DirichletBC(V, df.Constant(apex_val), facets, self.parameters["apxid"])]

        u = df.TrialFunction(V)
        v = df.TestFunction(V)
        a = df.inner(nabla_grad(u), nabla_grad(v)) * dx
        L = df.Constant(0) * v * dx

        u_sol = df.Function(V)
        df.solve(a == L, u_sol, bc_bas + bc_apx)
        return u_sol

    def solveLaplaceEquation_fch(self):
        if not self._laplace_solver_enabled():
            return None
        bas_ids = [self.parameters["aorta_wall"], self.parameters["pulm_wall"]]
        base_val, apex_val = self._laplace_bc("laplace_fch_bc", 0.25, 1.0)
        return self._solve_laplace(bas_ids, base_val, apex_val)

    def solveLaplaceEquation_waorta(self):
        if not self._laplace_solver_enabled():
            return None
        bas_ids = [
            self.parameters["aorta_int_wall"],
            self.parameters["aorta_ext_wall"],
            self.parameters["aorta_ring"],
        ]
        base_val, apex_val = self._laplace_bc("laplace_waorta_bc", 0.25, 1.0)
        return self._solve_laplace(bas_ids, base_val, apex_val)

    def solveLaplaceEquation_ideal(self):
        if not self._laplace_solver_enabled():
            return None
        bas_ids = [self.parameters["topid"]]
        base_val, apex_val = self._laplace_bc("laplace_ideal_bc", 0.5, 2.0)
        return self._solve_laplace(bas_ids, base_val, apex_val)

    def _IMP_components(self):
        F = self.Fe()
        PK1 = self.PK1()
        J = self.J()
        Tca = (1.0 / J) * PK1 * F.T

        s = self._normalized_push_forward(F, self.parameters["sheet"])
        n = self._normalized_push_forward(F, self.parameters["sheet-normal"])
        return Tca, s, n

    def _IMP_surface_pressure(self, surface_id):
        N = self.parameters["facet_normal"]
        F = self.Fe()
        PK1 = self.PK1()

        ds = self._surface_measure()
        n = self._normalized_push_forward(F, N)

        i, j = indices(2)
        return -n[i] * PK1[i, j] * N[j] * ds(surface_id)

    def _area_form(self, surface_id):
        u = self.parameters["displacement_variable"]

        d = u.geometric_dimension()
        I = df.Identity(d)
        F = I + df.grad(u)
        J = df.det(F)

        N = self.parameters["facet_normal"]
        ds = self._surface_measure()
        n = self._normalized_push_forward(F, N)

        i, j = indices(2)
        return (J * df.inv(F.T) * N)[i] * n[i] * ds(surface_id)

    def IMP(self):
        Tca, s, _ = self._IMP_components()

        i, j = indices(2)
        Ipressure = s[i] * Tca[i, j] * s[j]

        return Ipressure

    def IMP2(self):
        Tca, s, n = self._IMP_components()

        i, j = indices(2)
        Ipressure = 0.5 * (s[i] * Tca[i, j] * s[j] + n[i] * Tca[i, j] * n[j])

        return Ipressure

    def IMPendo(self):
        return self._IMP_surface_pressure(self.parameters["LVendoid"])

    def IMPepi(self):
        return self._IMP_surface_pressure(self.parameters["epiid"])

    def areaendo(self):
        return self._area_form(self.parameters["endoid"])

    def areaepi(self):
        return self._area_form(self.parameters["epiid"])
