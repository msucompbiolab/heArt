import collections.abc
import numpy as np
import dolfin as df
from ufl import conditional, diff, exp, max_value, indices

from .aorta_defaults import fill_aorta_params


class GuccionePas(object):
    def __init__(self, params):
        self.parameters = self._deep_update(self.default_parameters(), params)
        # Fill missing aorta defaults since some callers only set a subset of the parameter dict.
        self.parameters["aorta params"] = fill_aorta_params(
            self.parameters.get("aorta params")
        )

    def _deep_update(self, base, updates):
        # Recursive merge: preserves nested dict defaults while allowing leaf overrides.
        for key, value in updates.items():
            if isinstance(value, collections.abc.Mapping):
                base[key] = self._deep_update(base.get(key, {}), value)
            else:
                base[key] = value
        return base

    def default_parameters(self):
        return {
            "material params": {
                "bff": df.Constant(29.9),
                "bfx": df.Constant(13.3),
                "bxx": df.Constant(26.6),
                "Cparam": df.Constant(100),
                "eta": df.Constant(0.2),
            },
            "porous params": {
                "Ks": df.Constant(5.0e4),
                "phi0": df.Constant(0.35),
                "c1": df.Constant(1.33),
                "c2": df.Constant(550.0),
                "c3": df.Constant(45.0),
            },
            "aorta params": fill_aorta_params(),
        }

    def get_material_params(self):
        return self.parameters["material params"]

    # ------------------------------------------------------------------ aorta
    def _aorta_isochoric_ground(self):
        """Is the aortic GROUND-MATRIX term built on the isochoric invariant?

        OFF (default) -> the historical coupled form W_gr = f(I1), which is NOT
        stress-free at F = I: dW/dF|_I = Cgr*I (HGO/NeoHookean) or (D1/2)*I
        (Delfino), a residual hydrostatic tension that the volumetric penalty
        Kappa/2*(ln J)^2 cannot balance (its own derivative vanishes at J = 1).
        ON  -> the volumetric-isochoric (decoupled) construction W_gr = f(Ibar1),
        Ibar1 = J^(-2/3)*I1, whose first derivative vanishes identically at
        F = I, so the reference configuration is genuinely load-free.
        """
        return bool(self.parameters.get("aorta_isochoric_ground", False))

    def _aorta_ground_invariant(self, F):
        """First invariant used by the aortic ground matrix: I1, or the
        isochoric Ibar1 = J^(-2/3) I1 when the split is enabled."""
        Ic = df.tr(F.T * F)
        if not self._aorta_isochoric_ground():
            return Ic
        J = df.det(F)
        return J ** (-2.0 / 3.0) * Ic

    def Emat(self):
        u = self.parameters["displacement_variable"]
        d = u.ufl_domain().geometric_dimension()
        I = df.Identity(d)
        # Use the deformation gradient supplied by the caller (already growth-corrected if needed).
        F = self.parameters["F"]
        i, j, k = indices(3)
        Emat = 0.5 * (df.as_tensor(F[k, i] * F[k, j] - I[i, j], (i, j)))

        return Emat

    def PassiveRubSEF_neo(self):
        # Neo-Hookean rubber-like SEF for aortic regions (legacy support).
        u = self.parameters["displacement_variable"]
        d = u.ufl_domain().geometric_dimension()
        I = df.Identity(d)
        F = df.variable(I + df.grad(u))
        Ic = df.tr(F.T * F)
        mu = self.parameters["aorta params"]["mu"]
        Wp = (mu / 2) * (Ic - 3)
        return Wp

    def shear_modulus_main(self):
        metrics = self._fiber_metrics()
        return self._shear_energy(self.parameters["material params"]["Cparam"], metrics)

    # ------------------------------------------------------------------ plug
    #: Constitutive laws the LVW annular PLUG (waorta ``rubber_region``: aortic plug 3,
    #: base slab 4, mitral plug 5) can carry. ``guccione`` is the historical path: the
    #: myocardial Guccione SEF at ``Cparam_av`` (falling back to ``Cparam``) on the plug's
    #: own -- arbitrarily oriented -- fibre vectors. ``neohookean`` is an ISOTROPIC
    #: Ibar1-based neo-Hookean at ``plug_mu``, stress-free at F = I.
    PLUG_LAWS = ("guccione", "neohookean")

    def plug_law(self):
        law = str(self.parameters.get("plug_law", "guccione") or "guccione").lower()
        if law not in self.PLUG_LAWS:
            raise ValueError(
                "plug_law=%r is not one of %s" % (law, list(self.PLUG_LAWS)))
        return law

    def plug_stiffness_parameter(self):
        """The ONE scalar that sets the plug's stiffness under the active plug law:
        ``material params["Cparam_av"]`` (Guccione; the documented ``Cparam`` fallback
        when absent) or ``material params["plug_mu"]`` (neo-Hookean; REQUIRED). Returned
        as stored (a dolfin.Constant when the note set it) so a driver can continue on it."""
        mp = self.parameters["material params"]
        if self.plug_law() == "neohookean":
            if mp.get("plug_mu") is None:
                raise KeyError(
                    "plug_law='neohookean' needs material params['plug_mu'] (Pa); "
                    "the note key is PLUG_MU")
            return mp["plug_mu"]
        return mp.get("Cparam_av", mp["Cparam"])

    # ------------------------------------------------------------ aortic-valve cap
    # The AORTIC-VALVE CAP (waorta cell region 3 -- the slab whose lumen face lies in the
    # pressurised `aorta_int_wall` facet set and whose other face is `aortic_valvep` on
    # the LV cavity) may carry its OWN law and stiffness, independent of the mitral cap
    # (5) and basal ring (4), which keep the shared plug law above byte-for-byte. Absent
    # (``aocap_law`` None) the cap is the plug: every existing note is unchanged.
    def aocap_law(self):
        """``None`` (the cap is the plug -- the historical path) or one of PLUG_LAWS."""
        raw = self.parameters.get("aocap_law")
        if raw is None or str(raw).strip() == "":
            return None
        law = str(raw).lower()
        if law not in self.PLUG_LAWS:
            raise ValueError(
                "aocap_law=%r is not one of %s" % (law, list(self.PLUG_LAWS)))
        return law

    def aocap_stiffness_parameter(self):
        """The ONE scalar that sets the aortic cap's stiffness under ``aocap_law``:
        ``material params["aocap_mu"]`` (neo-Hookean, REQUIRED; note AOCAP_MU) or
        ``material params["aocap_cparam"]`` (Guccione, REQUIRED; note AOCAP_CPARAM). No
        silent fallback to the plug or myocardial value: a cap law with no stiffness is
        refused so a note cannot half-configure the cap."""
        law = self.aocap_law()
        if law is None:
            raise KeyError("aocap_law is not set; the aortic cap carries the plug law")
        mp = self.parameters["material params"]
        key = "aocap_mu" if law == "neohookean" else "aocap_cparam"
        if mp.get(key) is None:
            raise KeyError(
                "aocap_law=%r needs material params[%r] (Pa); the note key is %s"
                % (law, key, key.upper()))
        return mp[key]

    def _neo_plug_sef(self, mu):
        F = self.parameters["F"]
        J = df.det(F)
        Ibar1 = J ** (-2.0 / 3.0) * df.tr(F.T * F)
        return (mu / 2.0) * (Ibar1 - 3.0)

    def _guccione_plug_sef(self, C):
        metrics = self._fiber_metrics()
        QQ = self._strain_invariant(metrics)
        return C / 2.0 * (exp(QQ) - 1.0)

    def PassiveAoCapSEF(self):
        """Aortic-valve cap SEF under ``aocap_law`` at :meth:`aocap_stiffness_parameter`
        -- the same two forms the plug can carry (:meth:`PassiveRubSEF`), with their own
        stiffness scalar. Raises when ``aocap_law`` is unset (callers must not build a cap
        form on the default path)."""
        law = self.aocap_law()
        if law is None:
            raise KeyError("aocap_law is not set; the aortic cap carries the plug law")
        k = self.aocap_stiffness_parameter()
        if law == "neohookean":
            return self._neo_plug_sef(k)
        return self._guccione_plug_sef(k)

    def shear_modulus_aocap(self):
        """P1P1 stabilization shear modulus of the aortic cap under ``aocap_law``."""
        law = self.aocap_law()
        if law is None:
            raise KeyError("aocap_law is not set; the aortic cap carries the plug law")
        k = self.aocap_stiffness_parameter()
        if law == "neohookean":
            return k
        return self._shear_energy(k, self._fiber_metrics())

    # ------------------------------------------------------------ fibrous annulus
    # The ANNULUS (waorta rubber regions 4 = basal interface ring and 5 = mitral cap by
    # default) may carry its OWN law and stiffness, independent of the aortic-valve cap (3)
    # and of the shared plug law.  Section 14.5 of the campaign doc: the ring that reacts
    # the closed-valve thrust carries the myocardial 100 Pa and yields; in vivo it is the
    # fibrous annulus.  Absent (``annulus_law`` None) nothing changes.
    def annulus_law(self):
        raw = self.parameters.get("annulus_law")
        if raw is None or str(raw).strip() == "":
            return None
        law = str(raw).lower()
        if law not in self.PLUG_LAWS:
            raise ValueError(
                "annulus_law=%r is not one of %s" % (law, list(self.PLUG_LAWS)))
        return law

    def annulus_stiffness_parameter(self):
        """``material params["annulus_mu"]`` (neo-Hookean; note ANNULUS_MU) or
        ``["annulus_cparam"]`` (Guccione; note ANNULUS_CPARAM), REQUIRED under the law."""
        law = self.annulus_law()
        if law is None:
            raise KeyError("annulus_law is not set; the annulus carries the plug law")
        mp = self.parameters["material params"]
        key = "annulus_mu" if law == "neohookean" else "annulus_cparam"
        if mp.get(key) is None:
            raise KeyError(
                "annulus_law=%r needs material params[%r] (Pa); the note key is %s"
                % (law, key, key.upper()))
        return mp[key]

    def PassiveAnnulusSEF(self):
        """Annulus SEF under ``annulus_law`` at :meth:`annulus_stiffness_parameter`."""
        law = self.annulus_law()
        if law is None:
            raise KeyError("annulus_law is not set; the annulus carries the plug law")
        k = self.annulus_stiffness_parameter()
        if law == "neohookean":
            return self._neo_plug_sef(k)
        return self._guccione_plug_sef(k)

    def shear_modulus_annulus(self):
        """P1P1 stabilization shear modulus of the annulus under ``annulus_law``."""
        law = self.annulus_law()
        if law is None:
            raise KeyError("annulus_law is not set; the annulus carries the plug law")
        k = self.annulus_stiffness_parameter()
        if law == "neohookean":
            return k
        return self._shear_energy(k, self._fiber_metrics())

    def PassiveRubSEF_neo_plug(self):
        """Isotropic neo-Hookean plug SEF ``mu/2 (Ibar1 - 3)``, ``Ibar1 = J^(-2/3) tr(F^T F)``.

        Built on the ISOCHORIC invariant so its first derivative vanishes at F = I; the
        volumetric response is left to ``Forms.Wvolumetric_main`` (the same split the
        aortic ground matrix uses under ``aorta_isochoric_ground``). The coupled
        ``mu/2 (I1 - 3)`` form of :meth:`PassiveRubSEF_neo` is NOT stress-free at F = I
        (dW/dF|_I = mu I) and is kept only for its legacy aortic callers."""
        return self._neo_plug_sef(self.plug_stiffness_parameter())

    def shear_modulus_rubber(self):
        if self.plug_law() == "neohookean":
            # The neo-Hookean shear modulus IS mu (used by the P1P1 stabilization).
            return self.plug_stiffness_parameter()
        C = self.parameters["material params"].get(
            "Cparam_av", self.parameters["material params"]["Cparam"]
        )
        metrics = self._fiber_metrics()
        return self._shear_energy(C, metrics)

    def shear_modulus_delfino(self):
        # Delfino shear modulus for aortic wall (non-fibered).
        u = self.parameters["displacement_variable"]
        d = u.ufl_domain().geometric_dimension()
        I = df.Identity(d)
        F = df.variable(I + df.grad(u))
        Ic = df.tr(F.T * F)
        mu = self.parameters["aorta params"]["mu"]
        D1 = self.parameters["aorta params"]["D1"]
        D2 = self.parameters["aorta params"]["D2"]
        Wpd = D1 * exp(D2 / 2.0 * (Ic - 3)) * (1 + D2 * (Ic - 3))

        return Wpd

    def _hgo_shear(self, iv_values, C1, C2, base):
        """Shared HGO shear modulus builder: only active for iv>1 to avoid compressive fiber stiffening."""
        terms = [
            conditional(
                iv > 1,
                C1[i] * exp(C2[i] * (iv - 1) ** 2) * (1 + 2 * C2[i] * (iv - 1) ** 2) * iv,
                0,
            )
            for i, iv in enumerate(iv_values)
        ]
        return base + sum(terms)

    def _twofiber_invariants(self, F_override=None):
        u = self.parameters["displacement_variable"]
        d = u.ufl_domain().geometric_dimension()
        I = df.Identity(d)
        F = F_override if F_override is not None else I + df.grad(u)
        F = df.variable(F)
        Cmat = F.T * F

        ez = self.parameters["fiberz-aorta"]
        ec = self.parameters["fiberc-aorta"]
        gamma = self.parameters["aorta params"].get("gamma")
        if gamma is None:
            raise ValueError("HGO_twofiber requires gamma (degrees) to define collagen directions.")
        if isinstance(gamma, (int, float)):
            gamma = df.Constant(gamma)
        gamma = gamma * (df.pi / 180.0)
        n0 = df.cos(gamma) * ez + df.sin(gamma) * ec
        n1 = df.cos(gamma) * ez - df.sin(gamma) * ec

        iv0 = df.dot(n0, Cmat * n0)
        iv1 = df.dot(n1, Cmat * n1)
        return iv0, iv1

    def shear_modulus_hgo_twofiber(self):
        Cgr = self.parameters["aorta params"]["Cgr"]

        C1 = self.parameters["aorta params"]["C1"]
        C2 = self.parameters["aorta params"]["C2"]

        iv0, iv1 = self._twofiber_invariants()

        return self._hgo_shear([iv0, iv1], C1, C2, Cgr)

    def shear_modulus_hgo_twofiber_fdc(self):
        # Second derivative via finite differences around the current deformation; small g_ reduces bias.
        return self._finite_difference_shear_modulus(self.aorta_strain_energy_hgo_twofiber)

    def shear_modulus_hgo_fourfiber_ii(self):
        return self._finite_difference_shear_modulus(self.aorta_strain_energy_hgo_fourfiber)

    def shear_modulus_hgo_fourfiber(self):
        Cgr = self.parameters["aorta params"]["Cgr_ff"]
        C1 = self.parameters["aorta params"]["C1_ff"]
        C2 = self.parameters["aorta params"]["C2_ff"]

        lam_z, lam_c, lam_cl0, lam_cl1 = self.lmbda()
        iv = [lam_cl0**2, lam_cl1**2, lam_c**2, lam_z**2]

        return self._hgo_shear(iv, C1, C2, Cgr)

    def shear_modulus_hgo_fourfiber_min(self):
        Cgr = self.parameters["aorta params"]["Cgr_ff"]

        C1 = self.parameters["aorta params"]["C1_ff"]
        C2 = self.parameters["aorta params"]["C2_ff"]

        ez = self.parameters["fiberz-aorta"]
        ec = self.parameters["fiberc-aorta"]
        n0 = self.parameters["fiberclgn0-aorta"]
        n1 = self.parameters["fiberclgn1-aorta"]

        lam_z, lam_c, lam_cl0, lam_cl1 = self.lmbda()
        iv = [lam_cl0**2, lam_cl1**2, lam_c**2, lam_z**2]
        dirs = {"0": n0, "1": n1, "c": ec, "z": ez}

        plane_moduli = self._shear_modulus_hgo_fourfiber_planes(iv, dirs, C1, C2, Cgr)

        mu_min = plane_moduli[0]
        for mu_plane in plane_moduli[1:]:
            mu_min = conditional(mu_plane < mu_min, mu_plane, mu_min)

        return mu_min

    def shear_modulus_hgo_fourfiber_max(self):
        Cgr = self.parameters["aorta params"]["Cgr_ff"]

        C1 = self.parameters["aorta params"]["C1_ff"]
        C2 = self.parameters["aorta params"]["C2_ff"]

        ez = self.parameters["fiberz-aorta"]
        ec = self.parameters["fiberc-aorta"]
        n0 = self.parameters["fiberclgn0-aorta"]
        n1 = self.parameters["fiberclgn1-aorta"]

        lam_z, lam_c, lam_cl0, lam_cl1 = self.lmbda()
        iv = [lam_cl0**2, lam_cl1**2, lam_c**2, lam_z**2]
        dirs = {"z": ez, "c": ec, "0": n0, "1": n1}

        plane_moduli = self._shear_modulus_hgo_fourfiber_planes(iv, dirs, C1, C2, Cgr)

        mu_max = plane_moduli[0]
        for mu_plane in plane_moduli[1:]:
            mu_max = conditional(mu_plane > mu_max, mu_plane, mu_max)

        return mu_max

    def _fiber_metrics_from_E(self, Ea):
        return self._fiber_metrics_from_tensor(Ea)

    def _fiber_metrics_from_tensor(self, Ea):
        f0 = self.parameters["fiber"]
        s0 = self.parameters["sheet"]
        n0 = self.parameters["sheet-normal"]

        Eff = df.inner(f0, Ea * f0)
        Ess = df.inner(s0, Ea * s0)
        Enn = df.inner(n0, Ea * n0)
        Efs = df.inner(f0, Ea * s0)
        Efn = df.inner(f0, Ea * n0)
        Ens = df.inner(n0, Ea * s0)
        Esf = df.inner(s0, Ea * f0)
        Enf = df.inner(n0, Ea * f0)
        Esn = df.inner(s0, Ea * n0)

        return {
            "Eff": Eff,
            "Ess": Ess,
            "Enn": Enn,
            "Efs": Efs,
            "Efn": Efn,
            "Ens": Ens,
            "Esf": Esf,
            "Enf": Enf,
            "Esn": Esn,
        }

    def _finite_difference_shear_modulus(self, energy_func):
        """Finite-difference estimate of d²W/dF² along fiber-cross direction."""
        fz0 = self.parameters["fiberz-aorta"]
        fc0 = self.parameters["fiberc-aorta"]
        u = self.parameters["displacement_variable"]
        d = fz0.ufl_domain().geometric_dimension()
        I = df.Identity(d)

        g_ = df.Constant(1e-6)
        F0 = I + df.grad(u)
        Fp = F0 + g_ * df.outer(fz0, fc0)
        Fm = F0 - g_ * df.outer(fz0, fc0)

        W0 = energy_func(F_override=F0)
        Wp = energy_func(F_override=Fp)
        Wm = energy_func(F_override=Fm)

        return (Wp - 2 * W0 + Wm) / (g_**2)

    def _shear_modulus_hgo_fourfiber_planes(self, iv, dirs, C1, C2, Cgr):
        """Build shear modulus over all fiber planes for HGO 4-fiber aorta."""
        planes = [
            ("z", "c"),
            ("z", "0"),
            ("z", "1"),
            ("c", "0"),
            ("c", "1"),
            ("0", "1"),
        ]
        fiber_dirs = [dirs["0"], dirs["1"], dirs["c"], dirs["z"]]

        plane_moduli = []
        for a_key, b_key in planes:
            a = dirs[a_key]
            b = dirs[b_key]
            mu_plane = Cgr
            for i, d_i in enumerate(fiber_dirs):
                iv_i = iv[i]
                weight2 = df.inner(d_i, a) * df.inner(d_i, b)
                mu_i = conditional(
                    iv_i > 1,
                    C1[i]
                    * exp(C2[i] * (iv_i - 1) ** 2)
                    * (1 + 2 * C2[i] * (iv_i - 1) ** 2)
                    * iv_i
                    * weight2**2,
                    0,
                )
                mu_plane += mu_i
            plane_moduli.append(mu_plane)

        return plane_moduli

    def PassiveMatSEF(self, Ea=None):
        metrics = self._fiber_metrics_from_E(Ea) if Ea is not None else self._fiber_metrics()
        QQ = self._strain_invariant(metrics)
        C = self.parameters["material params"]["Cparam"]
        return C / 2.0 * (exp(QQ) - 1.0)

    def PassiveStress(self, Ea=None):
        """
        Passive PK2 stress with optional isotropic Kelvin–Voigt viscosity.
        Falls back to purely elastic stress if viscous parameters are absent.
        """
        Ea = df.variable(self.Emat() if Ea is None else Ea)
        Wp = self.PassiveMatSEF(Ea=Ea)
        S_hyperelastic = diff(Wp, Ea)  # 2nd Piola

        if not self.parameters.get("_passive_viscous_", False):
            return S_hyperelastic

        dt = self.parameters.get("dt")
        eta = self.parameters.get("material params", {}).get("eta")
        Eold = self.parameters.get("Ea_old")
        if dt is None or eta is None or Eold is None:
            return S_hyperelastic

        Edot = (Ea - Eold) / dt
        S_vis = 2.0 * eta * Edot
        return S_hyperelastic + S_vis

    def PassiveRubSEF(self):
        """Plug (waorta ``rubber_region``) SEF under the configured ``plug_law``.

        ``guccione`` (default, byte-identical to the historical form): the Guccione SEF
        at ``Cparam_av`` -- ``Cparam`` when absent -- on the plug's own fibre vectors.
        ``neohookean``: :meth:`PassiveRubSEF_neo_plug`."""
        if self.plug_law() == "neohookean":
            return self.PassiveRubSEF_neo_plug()
        C = self.parameters["material params"].get(
            "Cparam_av", self.parameters["material params"]["Cparam"]
        )
        return self._guccione_plug_sef(C)

    def PK1(self):
        u = self.parameters["displacement_variable"]
        C = self.parameters["material params"]["Cparam"]
        F = df.variable(df.Identity(u.geometric_dimension()) + df.grad(u))
        metrics = self._fiber_metrics(F_override=F)
        QQ = self._strain_invariant(metrics)
        Wp = C / 2.0 * (exp(QQ) - 1.0)
        return df.diff(Wp, F)

    def poro_volumetricstress(self):
        Ks = self.parameters["porous params"]["Ks"]
        u = self.parameters["displacement_variable"]
        d = u.ufl_domain().geometric_dimension()
        I = df.Identity(d)
        F = variable(I + df.grad(u))
        J = df.det(F)
        psi_skel = (Ks / 2) * (J - 1) * ln(J)
        return diff(psi_skel, F)

    def poro_porositystress(self):
        Ks = self.parameters["porous params"]["Ks"]
        phi = self.parameters["pressure_variable"]
        phi0 = self.parameters["porous params"]["phi0"]
        u = self.parameters["displacement_variable"]
        d = u.ufl_domain().geometric_dimension()
        I = df.Identity(d)
        F = variable(I + df.grad(u))
        J = df.det(F)
        phi_s = variable(J - phi)
        # phi_s0 = 1.0 - phi0
        psi_s = Ks * (
            phi_s - (1 - phi0) - ln(phi_s / (1 - phi0))
        )  # with phi_s ~= phi_s0 constrain
        P1 = diff(psi_s, phi_s) * J * df.inv(F.T)
        return P1

    def poro_pressure(self):
        Ks = self.parameters["porous params"]["Ks"]
        phi0 = self.parameters["porous params"]["phi0"]
        c1 = self.parameters["porous params"]["c1"]
        c2 = self.parameters["porous params"]["c2"]
        c3 = self.parameters["porous params"]["c3"]
        u = self.parameters["displacement_variable"]
        phi = self.parameters["pressure_variable"]
        d = u.ufl_domain().geometric_dimension()
        I = df.Identity(d)
        F = variable(I + df.grad(u))
        J = df.det(F)
        psi = c1 * exp(c3 * phi) + c2 * ln(c3 * phi)
        phi_s = variable(J - phi)
        # phi_s0 = 1.0 - phi0
        psi_s = Ks * (
            phi_s - (1 - phi0) - ln(phi_s / (1 - phi))
        )  # with phi_s ~= phi_s0 constrain
        p1 = diff(psi, variable(phi))
        p2 = diff(psi_s, phi_s)
        return p1 - p2

    def aorta_strain_energy_neohookean(self):
        u = self.parameters["displacement_variable"]
        d = u.ufl_domain().geometric_dimension()
        I = df.Identity(d)
        F = I + df.grad(u)
        F = df.variable(F)
        Ic = self._aorta_ground_invariant(F)
        mu = 50.0
        Wp = (mu / 2) * (Ic - 3)
        return Wp

    def aorta_strain_energy_delfino(self):
        u = self.parameters["displacement_variable"]
        d = u.ufl_domain().geometric_dimension()
        I = df.Identity(d)
        F = I + df.grad(u)
        F = df.variable(F)
        Ic = self._aorta_ground_invariant(F)
        mu = self.parameters["aorta params"]["mu"]
        D1 = self.parameters["aorta params"]["D1"]
        D2 = self.parameters["aorta params"]["D2"]
        Wp = D1 / D2 * (exp(D2 / 2.0 * (Ic - 3)) - 1)
        return Wp

    def _aorta_strain_energy_hgo(self, Cgr, C1, C2, iv_values, scale, F_override=None):
        """Shared HGO strain-energy density for aorta: base matrix + fiber families in extension."""
        u = self.parameters["displacement_variable"]
        d = u.ufl_domain().geometric_dimension()
        I = df.Identity(d)
        F_base = F_override if F_override is not None else I + df.grad(u)
        F = df.variable(F_base)
        Ic = self._aorta_ground_invariant(F)

        Wp_Cgr = Cgr / 2.0 * (Ic - 3)

        # NOTE on the fibre invariants: they stay the FULL I4 = n0.C.n0, not the
        # isochoric Ibar4, on purpose. Both are stress-free at F = I here (the
        # (<I4-1>_+)^2 plus-bracket is C^1 with vanishing derivative at I4 = 1),
        # so the choice does not affect the reference state; and Nolan et al.
        # (2014, J Mech Behav Biomed Mater 39:48-60, doi:10.1016/j.jmbbm.2014.06.016)
        # show that applying the volumetric-isochoric split to the ANISOTROPIC
        # term of an HGO-type law produces a non-physical response under
        # near-incompressible/compressible volumetric coupling. The split is
        # therefore applied to the isotropic ground matrix only.
        # C2 = 0 is a REMOVABLE singularity, not an invalid parameter set. Jadidi et al.
        # 2020 (10.1016/j.actbio.2019.12.024) Table 3 reports C2 = 0.00 for the
        # circumferential and/or axial family of the two youngest age groups (15.3 y:
        # C2^3 = C2^4 = 0; 24.4 y: C2^3 = 0), and the paper offers no explanation. The
        # printed energy C1/(s*C2)*(exp(C2*x^2) - 1) is then 0/0 and evaluates to NaN,
        # but its limit is finite -- lim_{C2->0} = C1*x^2/s -- and the paper's OWN Cauchy
        # stress (Appendix A.1.4, t_fibre = C1 (IV-1) exp(C2 (IV-1)^2) lambda^2) carries
        # no 1/C2 at all and is perfectly finite there. A C2 = 0 family is therefore a
        # fibre stress LINEAR in (IV - 1), i.e. the same law with its exponential
        # stiffening switched off -- physically sensible for a young aorta whose collagen
        # is not yet recruited over the tested range, and consistent with the monotone
        # age trend of C2 in that table. The branch is taken on the PARAMETER, so it is
        # bit-identical for every C2 > 0 (every set this repository ships).
        fiber_terms = []
        for i, iv in enumerate(iv_values):
            x2 = max_value(iv - 1, 0) ** 2
            c1, c2 = C1[i], C2[i]
            if float(c2) == 0.0:
                fiber_terms.append(c1 / scale * x2)
            else:
                fiber_terms.append(c1 / (scale * c2) * (exp(c2 * x2) - 1))
        return Wp_Cgr + sum(fiber_terms)

    def aorta_strain_energy_hgo_twofiber(self, F_override=None):
        Cgr = self.parameters["aorta params"]["Cgr"]
        C1 = np.array(self.parameters["aorta params"]["C1"])
        C2 = np.array(self.parameters["aorta params"]["C2"])
        iv0, iv1 = self._twofiber_invariants(F_override=F_override)
        iv_values = [iv0, iv1]
        return self._aorta_strain_energy_hgo(Cgr, C1, C2, iv_values, scale=2.0, F_override=F_override)

    def aorta_strain_energy_hgo_fourfiber(self, F_override=None):
        Cgr = self.parameters["aorta params"]["Cgr_ff"]
        C1 = np.array(self.parameters["aorta params"]["C1_ff"])
        C2 = np.array(self.parameters["aorta params"]["C2_ff"])

        # F_override MUST reach the fibre stretches too. It previously reached only the
        # ground matrix (lmbda() rebuilt F = I + grad(u) unconditionally), so a caller
        # perturbing F -- `_finite_difference_shear_modulus`, or any external probe of
        # this energy at a prescribed deformation gradient -- silently froze all four
        # fibre terms at the CURRENT displacement. The two-fibre twin already threaded
        # it (`_twofiber_invariants(F_override=...)`). Production is unaffected:
        # `forms_MRC2.aorta_strain_energy()` and `shear_modulus_hgo_fourfiber_max()`
        # both call with F_override = None, for which this is byte-identical.
        lmbda_z, lmbda_c, lmbda_clgn0, lmbda_clgn1 = self.lmbda(F_override=F_override)
        iv_values = [lmbda_clgn0**2, lmbda_clgn1**2, lmbda_c**2, lmbda_z**2]
        return self._aorta_strain_energy_hgo(Cgr, C1, C2, iv_values, scale=4.0, F_override=F_override)

    def lmbda(self, F_override=None):
        u = self.parameters["displacement_variable"]
        d = u.ufl_domain().geometric_dimension()
        I = df.Identity(d)
        F = F_override if F_override is not None else I + df.grad(u)
        F = df.variable(F)
        fz0 = self.parameters["fiberz-aorta"]
        fc0 = self.parameters["fiberc-aorta"]

        f_clgn0 = self.parameters["fiberclgn0-aorta"]
        f_clgn1 = self.parameters["fiberclgn1-aorta"]

        Cmat = F.T * F
        lmbda_z = df.sqrt(df.dot(fz0, Cmat * fz0))
        lmbda_c = df.sqrt(df.dot(fc0, Cmat * fc0))

        lmbda_clgn0 = df.sqrt(df.dot(f_clgn0, Cmat * f_clgn0))
        lmbda_clgn1 = df.sqrt(df.dot(f_clgn1, Cmat * f_clgn1))

        return lmbda_z, lmbda_c, lmbda_clgn0, lmbda_clgn1
    def _fiber_metrics(self, F_override=None):
        u = self.parameters["displacement_variable"]
        dim = u.ufl_domain().geometric_dimension()
        I = df.Identity(dim)
        F = F_override if F_override is not None else self.parameters["F"]
        Ea = 0.5 * (F.T * F - I)
        return self._fiber_metrics_from_tensor(Ea)

    def _strain_invariant(self, metrics):
        params = self.parameters["material params"]
        bff = params["bff"]
        bfx = params["bfx"]
        bxx = params["bxx"]
        Eff = metrics["Eff"]
        Ess = metrics["Ess"]
        Enn = metrics["Enn"]
        Ens = metrics["Ens"]
        Esn = metrics["Esn"]
        Efs = metrics["Efs"]
        Esf = metrics["Esf"]
        Efn = metrics["Efn"]
        Enf = metrics["Enf"]
        return (
            bff * Eff**2.0
            + bxx * (Ess**2.0 + Enn**2.0 + Ens**2.0 + Esn**2.0)
            + bfx * (Efs**2.0 + Esf**2.0 + Efn**2.0 + Enf**2.0)
        )

    def _shear_energy(self, C_value, metrics=None):
        metrics = metrics or self._fiber_metrics()
        QQ = self._strain_invariant(metrics)
        bfx = self.parameters["material params"]["bfx"]
        return C_value * (bfx**2) / 2.0 * exp(QQ) * (1 + QQ)
