import dolfin as df
import numpy as np
from ufl import indices, outer


class activeForms(object):
    def __init__(self, params):
        self.parameters = self.default_parameters()
        self.parameters.update(params)

        Matmodel = self.parameters["material model"]["Name"]
        assert (
            Matmodel == "Guccione" or Matmodel == "Time-varying"
        ), "Material model not implemented"

        if Matmodel == "Guccione":
            from .GuccioneAct import GuccioneAct as Active
        else:  # Time-varying
            from .BurkhoffTimevarying3 import BurkhoffTimevarying as Active

        deg = self.parameters["deg"]
        mesh = self.parameters["mesh"]

        # Store activation fields in a Quadrature space to avoid oscillations from nodal interpolation.
        Quadelem = df.FiniteElement(
            "Quadrature", mesh.ufl_cell(), degree=deg, quad_scheme="default"
        )
        Quadelem._quad_scheme = "default"
        self.Quadelem = Quadelem
        self.Quad = df.FunctionSpace(mesh, Quadelem)

        self.t_init = self.get_t_init()
        self.isActive = self.get_isActive()
        self._Vn_quad = df.Function(self.Quad)

        self.parameters.update(
            {"t_init": self.t_init, "isActive": self.isActive, "Fmat": self.Fmat()}
        )

        self.activeforms = Active(self.parameters)
        self.matparams = self.activeforms.get_material_params()

    def default_parameters(self):
        return {
            "material model": {"Name": "Time-varying"},
            "Threshold_Potential": df.Constant(0.9),
            "activation_threshold": 1e-1,
            "homogeneous_activation_time": df.Constant(10.0),
            "t_init_fill_value": df.Constant(9999.0),
        }

    def get_isActive(self):
        isActive = df.Function(self.Quad)
        isActive.vector()[:] = 0.0
        return isActive

    def Get_t_a(self):
        return self.parameters["t_a"].vector().get_local()

    def Get_cycle(self):
        return self.parameters["cycle"].vector().get_local()

    def update_activationTime(self, potential_n, comm):
        """
        if V[i] >= V_thres[i] and isActivation[i] == 0
            then t0[i] = t_a
        """
        # `comm` is kept for legacy call sites; activation updates are purely local on each rank's dof ownership.

        V_thres = self.parameters["Threshold_Potential"]
        current_ta_array = self.parameters["t_a"].vector().get_local()

        t_init_array = self.t_init.vector().get_local()
        isActive_array = self.isActive.vector().get_local()

        self._Vn_quad.assign(df.interpolate(potential_n, self.Quad))
        V_n_array = self._Vn_quad.vector().get_local()

        # One-way activation: once active (isActive≈1), t_init is frozen at the first threshold crossing.
        tol_isActive = float(self.parameters.get("activation_threshold", 1e-1))
        mask = (np.abs(isActive_array) <= tol_isActive) & (V_n_array >= V_thres)
        isActive_array[mask] = 1.0
        t_init_array[mask] = current_ta_array[mask]

        if self.parameters["HomogenousActivation"]:
            # Debug/simplification mode: enforce a uniform activation time everywhere.
            fill_val = float(self.parameters.get("homogeneous_activation_time", 10.0))
            t_init_array[:] = fill_val

        self.t_init.vector()[:] = t_init_array
        self.isActive.vector()[:] = isActive_array

    def get_t_init(self):
        Quad = self.Quad

        t_init = df.Function(Quad)
        t_init.vector()[:] = float(self.parameters.get("t_init_fill_value", 9999.0))
        return t_init

    def restart_t_init(self):
        self.t_init = self.get_t_init()
        self.isActive = self.get_isActive()

    def Fmat(self):
        u = self.parameters["displacement_variable"]
        return df.Identity(u.ufl_domain().geometric_dimension()) + df.grad(u)

    def Fe(self):
        Fg = self.parameters["growth_tensor"]
        F = self.Fmat()

        return F if Fg is None else F * df.inv(Fg)

    def _log(self):
        from ..utils.log_mpi import get_logger

        mesh = self.parameters.get("mesh")
        comm = mesh.mpi_comm() if mesh is not None else None
        return get_logger("fe", comm)

    def _fiber_axis_projection(self):
        """Rank-1 fiber structural tensor f0 x f0 — the pressure-generating axis alone.

        This is the projection the FIBER-stress diagnostics must use, independently of any
        transverse active tension: with kappa>0 the full active structural tensor is no
        longer rank-1, so projecting onto it would report kappa-contaminated "fiber" stress
        (inner(P,P)=1+kappa^2 rather than 1). See fiberstress().
        """
        f0 = self.parameters["fiber"]
        return outer(f0, f0)

    def _fiber_projection(self):
        """Active-stress structural tensor P, with S_act = T(t) * P.

        Default = rank-1 fiber-only ``outer(f0, f0)``; ``transverse_active_fraction`` kappa=0
        (the default everywhere in this repo) is byte-identical to the legacy form.

        OPTIONAL transverse active tension (kappa>0) makes the active stress non-rank-1, which
        is why it interacts with the additive-active-stress loss of ellipticity (Ambrosi &
        Pezzuto 2012, ``10.1007/s10659-011-9351-4``). NOTE it is not a remedy for the
        activation ceiling: diverting contractile effort off the fibre axis LOWERS cavity
        pressure (see src/mechanics/CLAUDE.md).

        DIRECTION CONVENTION (corrected 2026-08-11). The Guccione-school 40% figure this
        parameter is named for loads ONE transverse direction, the SHEET axis -- Genet et al.
        2014 (``10.1152/japplphysiol.00255.2014``, same constitutive + active law as this
        repo) states the active force is orthotropic "with maximal contractile force
        developing in the local myofiber direction and a contractile force 40% of the maximum
        developing in the local sheet direction". Sun et al. 2009 (``10.1115/1.3148464``)
        likewise says cross-fiber IN-PLANE, and the Guccione 1991/1993 coordinate system
        distinguishes in-plane from radial. Cross-fibre tension itself is well founded --
        Walker et al. 2005 (``10.1152/ajpheart.01226.2004``) improved strain agreement 27%
        (RMS 0.074 -> 0.054) by adding it -- so the term belongs; only the direction COUNT was
        wrong. This previously added kappa to BOTH s0 x s0 and n0 x n0, roughly doubling the
        transverse active stress and loading a radial axis the sources do not.

        ``transverse_active_structure`` selects the axes, defaulting to the literature value:
          "sheet"  (default) : P = f0 x f0 + kappa * s0 x s0
          "normal"           : P = f0 x f0 + kappa * n0 x n0
          "both"   (LEGACY)  : P = f0 x f0 + kappa * (s0 x s0 + n0 x n0) -- reproduces the
                               pre-2026-08-11 form for comparison ONLY; warns loudly.

        WHICH DATASET IS THE IN-PLANE AXIS IS A PROPERTY OF THE MESH, NOT OF THIS CODE.
        The biaxial measurements the fraction comes from (Lin & Yin 1998
        ``10.1115/1.2798021``; Walker 2005) load the cross-fibre direction IN THE WALL PLANE
        (thin tangential slices). On a Holzapfel-Ogden / LDRB triad (Holzapfel & Ogden 2009
        ``10.1098/rsta.2009.0091``: s0 lies "in the plane of the layer", and the layers run
        radially, LeGrice 1995 ``10.1152/ajpheart.1995.269.2.H571``; Bayer 2012
        ``10.1007/s10439-012-0593-5``: S is the transmural axis rotated by the sheet angle)
        ``s0`` is the TRANSMURAL axis and ``n0`` the in-plane cross-fibre axis. MEASURED on the
        swine-8159 LVW mesh: |eS.eR| = 1.000 on every myocardial cell, |eN.eR| = 0.000. On
        such a mesh "sheet" loads the radial axis the sources do not, and "normal" is the
        literature's in-plane cross-fibre stress. A caller must measure its triad against the
        wall normal (scripts/lvw_active_formulation_variants.py ``basis``) and select the
        structure accordingly; nothing here can tell which axis is which.
        """
        proj = self._fiber_axis_projection()
        kappa = float(self.parameters.get("transverse_active_fraction", 0.0) or 0.0)
        if kappa <= 0.0:
            return proj

        s0 = self.parameters.get("sheet")
        n0 = self.parameters.get("sheet-normal")
        if s0 is None:
            # Root cause is a caller that requested transverse tension without supplying the
            # sheet basis; degrading to fiber-only silently would look like kappa "not
            # working" (exactly the earlier silent no-op). Surface it.
            self._log().warn(
                "init",
                "transverse active tension requested but the sheet basis is unavailable; "
                "falling back to FIBER-ONLY rank-1 active stress",
                transverse_active_fraction=kappa, sheet="missing")
            return proj

        structure = str(self.parameters.get("transverse_active_structure", "sheet") or "sheet")
        if structure == "both":
            if n0 is None:
                self._log().warn(
                    "init",
                    "transverse_active_structure='both' requested but the sheet-normal basis "
                    "is unavailable; applying the SHEET-only structure instead",
                    transverse_active_fraction=kappa, sheet_normal="missing")
                return proj + df.Constant(kappa) * outer(s0, s0)
            self._log().warn(
                "init",
                "transverse_active_structure='both' loads kappa on BOTH the sheet and "
                "sheet-normal axes: this is the superseded pre-2026-08-11 form, roughly 2x "
                "the transverse active stress the Guccione-school literature specifies "
                "(Genet 2014 10.1152/japplphysiol.00255.2014 loads the SHEET axis only). "
                "Retained for comparison against archived runs; not a production setting",
                transverse_active_fraction=kappa,
                transverse_active_structure=structure)
            return proj + df.Constant(kappa) * (outer(s0, s0) + outer(n0, n0))
        if structure == "normal":
            if n0 is None:
                raise ValueError(
                    "transverse_active_structure='normal' requested but the sheet-normal "
                    "basis is unavailable; supply parameters['sheet-normal'] or select 'sheet'")
            self._log().info(
                "init",
                "transverse active tension loads the SHEET-NORMAL axis n0 (the in-plane "
                "cross-fibre direction on an LDRB / Holzapfel-Ogden triad whose s0 is "
                "transmural); the caller is responsible for having measured that",
                transverse_active_fraction=kappa, transverse_active_structure=structure)
            return proj + df.Constant(kappa) * outer(n0, n0)
        if structure == "inplane":
            # The CONSTRUCTED in-plane cross-fibre axis, for a mesh whose stored (s0, n0) pair
            # is an arbitrary rotation about f0 rather than the (transmural, in-plane) pair.
            # It is supplied as a per-cell ANGLE theta in the (s0, n0) plane, so
            #   c0 = cos(theta) s0 + sin(theta) n0
            # is orthonormal to f0 by construction and needs no vector field of its own.
            # MEASURE theta (fch_events.basis emit-axis); do NOT guess it. On a triad whose s0
            # IS transmural this reduces to theta = +-pi/2, i.e. exactly the 'normal' option.
            theta = self.parameters.get("inplane-angle")
            if n0 is None or theta is None:
                raise ValueError(
                    "transverse_active_structure='inplane' requires BOTH the sheet-normal "
                    "basis and the measured per-cell in-plane angle "
                    "(SimDet['transverse_active_inplane_angle']); refusing to substitute a "
                    "stored dataset, which is the assumption this option exists to avoid")
            c0 = df.cos(theta) * s0 + df.sin(theta) * n0
            self._log().info(
                "init",
                "transverse active tension loads the CONSTRUCTED in-plane cross-fibre axis "
                "c0 = cos(theta) s0 + sin(theta) n0, from the measured transmural direction",
                transverse_active_fraction=kappa, transverse_active_structure=structure)
            return proj + df.Constant(kappa) * outer(c0, c0)
        if structure != "sheet":
            raise ValueError(
                f"transverse_active_structure={structure!r} is not recognized; expected "
                f"'sheet', 'normal', 'inplane' (constructed, measured) or 'both' (superseded)")
        return proj + df.Constant(kappa) * outer(s0, s0)

    def GetPact(self):
        return self.activeforms.PK1Stress()

    def PK2StressTensor(self):
        # Active PK2 = T(t) * P, with P the active structural tensor (rank-1 outer(f,f) by
        # default; + kappa*outer(s0,s0) when transverse active tension is enabled).
        return self.activeforms.pk2_stress() * self._fiber_projection()

    def PK2StressTensor_atr(self):
        # Atrial active stress = the atrial time course, fiber-projected. Magnitude is
        # set ONLY by the per-chamber Tmax_{la,ra} (applied as Tmax_chamber/Tmax_base
        # in MEmodel3), exactly like the ventricles (direct per-chamber Tmax).
        return self.activeforms.pk2_stress_atrial() * self._fiber_projection()

    def getCt(self):
        return self.activeforms.getCt()

    def getw1_atr(self):
        return self.activeforms.getw1_atr()

    def getw2_atr(self):
        return self.activeforms.getw2_atr()

    def get_t_atr_since_act(self):
        return self.activeforms.get_t_atr_since_act()

    def fiberstress(self):
        # Project onto the FIBRE AXIS f0 x f0, not the full active structural tensor P.
        # With P rank-1 (kappa=0) the two are identical, so this is byte-identical for every
        # production run. With kappa>0 they are NOT: inner(P, T*P) = T*(1+kappa^2) would
        # report an inflated "fibre" stress that also counts the transverse component, while
        # inner(f0 x f0, T*P) = T is the actual fibre-direction active stress. This feeds
        # MEmodel3.get_fiber_cauchy_stress, the literature-comparable measure the ED/ES stage
        # reports against Genet 2014 -- so contaminating it would corrupt exactly the
        # comparison the transverse arm exists to make.
        return df.inner(self._fiber_axis_projection(), self.PK2StressTensor())

    def fiberstress_atr(self):
        return df.inner(self._fiber_axis_projection(), self.PK2StressTensor_atr())

    def CalculateFiberNaturalStrain(self, F_, F_ref, e_fiber, VolSeg):
        # Pull back to ED
        F = F_ * df.inv(F_ref)
        # Right Cauchy Green
        C = F.T * F

        C_fiber = df.inner(C * e_fiber, e_fiber)
        E_fiber = 0.5 * (1 - 1 / C_fiber)

        mesh = self.parameters["mesh"]
        dx = self.parameters["dx"]

        E_fiber_BiV = [
            df.assemble(
                E_fiber * dx(ii), form_compiler_parameters={"representation": "uflacs"}
            )
            for ii in VolSeg
        ]

        return E_fiber_BiV, E_fiber
