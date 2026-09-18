# Burkhoff et al. 1993: Time varying elastance type active contraction model with length dependency for peak tension but not relaxation
# This modification account for nonuniform spatial activation
# to is a function of space, FE model using gauss points

import dolfin as df
import numpy as np


class BurkhoffTimevarying(object):
    def __init__(self, params):
        self.parameters = self.default_parameters()
        self.parameters.update(params)
        self.mat_params = self.parameters["material params"]
        # Guardrail: keep the transition time consistent with the rise time to avoid non-physical plateaus.
        self._enforce_t_trans_limit()

        mesh = self.parameters["mesh"]

        self.t_init = self.parameters["t_init"]
        self.isActive = self.parameters["isActive"]

        isActive_write_elem = df.FunctionSpace(mesh, "CG", 1)
        self.isActive_write_elem = isActive_write_elem

        t_init_write_elem = df.FunctionSpace(mesh, "CG", 1)
        self.t_init_write_elem = t_init_write_elem

    def default_parameters(self):
        return {
            "material params": {
                "tau": 30,
                "t_trans": 320,
                "B": 4.75,
                "t0": 275,
                "deg": 4,
                "l0": 1.6,
                "Ca0": 4.35,
                "Ca0max": 4.35,
                "lr": 1.85,
                "atrial_fill": 10.0,
                "l_range_buffer": 0.002,
                "l_softplus_k": 150.0,
            }
        }

    def get_material_params(self):
        return self.mat_params

    def _mat_param(self, key, default=None):
        if default is None:
            return self.mat_params[key]
        return self.mat_params.get(key, default)

    def _enforce_t_trans_limit(self):
        t0 = self.mat_params.get("t0")
        if t0 is None:
            return
        t_trans = self.mat_params.get("t_trans", 1.5 * t0)
        # Clamp overly large `t_trans` since the waveform assumes decay begins shortly after peak.
        if t_trans > 2.0 * t0:
            self.mat_params["t_trans"] = 1.5 * t0

    def _t_since_activation(self):
        return self.parameters["t_a"] - self.t_init

    def calcium_equivalent(self):
        params = self.mat_params
        ls_l0 = self.fiber_stretch_excess()
        return params["Ca0max"] / df.sqrt(df.exp(params["B"] * ls_l0) - 1)

    def fiber_stretch_excess(self):
        lmbda = self.fiber_stretch()
        lr = self._mat_param("lr")
        l0 = self._mat_param("l0")
        ls = lmbda * lr
        buffer = self._mat_param("l_range_buffer", 0.002)
        # Smooth (C-inf) length-tension slack floor. The hard
        #   conditional(ls <= l0+buffer, buffer, ls-l0)  ==  max(ls-l0, buffer)
        # has a C0 kink at the slack length: it makes the active-stress / fibre-
        # stretch (gamma) field discontinuous and clamps a flat near-zero plateau
        # right at the threshold, producing an artificial sub-endocardial "dead-
        # zone" (diag_deadzone_map.py: a broad sub-endo band of marginally-short
        # fibres). Replace with a softplus smooth-max with the SAME asymptotes
        # (-> ls-l0 well above slack; -> buffer well below, keeping the sqrt
        # argument in calcium_equivalent strictly positive) but a rounded knee, so
        # marginally-short fibres decline smoothly instead of dropping off a cliff.
        # k sets the knee sharpness (k -> inf recovers the hard hinge). Stable
        # softplus form (no exp overflow): softplus(z)=max(z,0)+ln(1+exp(-|z|)).
        k = self._mat_param("l_softplus_k", 150.0)
        z = k * (ls - l0 - buffer)
        softplus = df.conditional(df.gt(z, 0.0), z, 0.0) + df.ln(1.0 + df.exp(-abs(z)))
        ls_l0 = buffer + softplus / k

        return ls_l0

    def fiber_stretch(self):
        F = self.parameters["Fmat"]
        f0 = self.parameters["fiber"]
        Cmat = F.T * F
        lmbda = df.sqrt(df.dot(f0, Cmat * f0))

        return lmbda

    def time_course(self):
        return self.w1() + self.w2()

    def w1(self):
        params = self.mat_params
        t0 = params["t0"]
        t_trans = params.get("t_trans", 1.5 * t0)
        t_since_activation = self._t_since_activation()

        # Upstroke: half-cosine from 0 to Tmax, gated by (t_since_activation in [0, t_trans]).
        active = df.conditional(df.gt(t_since_activation, df.Constant(0.0)), 1.0, 0.0)
        within_window = df.conditional(
            df.lt(t_since_activation, params.get("trans0", t_trans)), 1.0, 0.0
        )
        w1 = (
            active * within_window * 0.5 * (1 - df.cos(df.pi * t_since_activation / t0))
        )

        return w1

    def w2(self):
        params = self.mat_params
        t0 = params["t0"]
        t_trans = params.get("t_trans", 1.5 * t0)
        tr = params["tau"]
        t_since_activation = self._t_since_activation()

        # Decay: exponential tail starting at `t_trans` with continuity enforced via the A scaling term.
        decay_active = df.conditional(df.le(t_trans, t_since_activation), 1.0, 0.0)
        A = 1.0 if "trans0" in params else 0.5 * (1 - df.cos(df.pi * t_trans / t0))
        w2 = decay_active * A * df.exp(-1.0 * (t_since_activation - t_trans) / tr)

        return w2

    def _softstart_factor(self):
        """Optional first-beat(s) contractility ramp in [0,1].

        Ramps active tension linearly from 0 to full over `tmax_softstart_beats`
        beats of total elapsed time (cycle*BCL + t_a), then holds at 1. This
        avoids the high-EDV + high-Tmax active-onset Jacobian cliff (a sudden
        cold start of full contractility at peak EDV makes the pressure
        root-find ill-conditioned). Enabled via material param
        `tmax_softstart_beats` (number of beats; <=0 or absent disables -> 1.0,
        so default behavior for the other demos is unchanged).
        """
        beats = float(self.mat_params.get("tmax_softstart_beats", 0.0) or 0.0)
        if beats <= 0.0:
            return 1.0
        t_a = self.parameters["t_a"]
        cycle = self.parameters["cycle"]
        BCL = self.parameters["HeartBeatLength"]
        total = cycle * BCL + t_a  # total elapsed time since closed-loop start
        return df.conditional(df.lt(total, beats * BCL), total / (beats * BCL), 1.0)

    def pk2_stress(self):
        params = self.mat_params
        Ca0 = params["Ca0"]
        Tmax = params["Tmax"]

        Ct = self.time_course()
        ECa = self.calcium_equivalent()

        return self._softstart_factor() * (Tmax * Ct * Ca0**2.0) / (Ca0**2.0 + ECa**2.0)

    def pk2_stress_atrial(self):
        params = self.mat_params
        Ca0 = params["Ca0"]
        Tmax = params["Tmax"]

        Ct = self.w1_atr() + self.w2_atr()
        ECa = self.calcium_equivalent()

        return (Tmax * Ct * Ca0**2.0) / (Ca0**2.0 + ECa**2.0)

    def _atrial_activation_state(self):
        params = self.parameters["material params"]
        t0_atr = params.get("t0_atr", params["t0"])
        t_trans_atr = params.get("t_trans_atr", 1.5 * t0_atr)
        tr_atr = params.get("tau_atr", params["tau"])
        tdelay_atr = params.get("tdelay_atr", 0.0)

        t_since_activation_atr = self._atrial_activation_time(
            self.parameters["t_a"],
            self.parameters["cycle"],
            tdelay_atr,
            self.parameters["HeartBeatLength"],
        )

        active = df.conditional(
            df.gt(t_since_activation_atr, df.Constant(0.0)), 1.0, 0.0
        )
        preload_window = df.conditional(
            df.lt(t_since_activation_atr, params.get("trans0", t_trans_atr)), 1.0, 0.0
        )
        decay_window = df.conditional(
            df.le(t_trans_atr, t_since_activation_atr), 1.0, 0.0
        )
        return (
            t_since_activation_atr,
            t0_atr,
            t_trans_atr,
            tr_atr,
            active,
            preload_window,
            decay_window,
        )

    def w1_atr(self):
        (
            t_since_activation_atr,
            t0_atr,
            t_trans_atr,
            _,
            active,
            preload_window,
            _,
        ) = self._atrial_activation_state()

        w1 = (
            active
            * preload_window
            * 0.5
            * (1 - df.cos(df.pi * t_since_activation_atr / t0_atr))
        )

        return w1

    def w2_atr(self):
        (
            t_since_activation_atr,
            t0_atr,
            t_trans_atr,
            tr_atr,
            _,
            _,
            decay_window,
        ) = self._atrial_activation_state()

        A = (
            1.0
            if "trans0" in self.mat_params
            else 0.5 * (1 - df.cos(df.pi * t_trans_atr / t0_atr))
        )

        w2 = (
            decay_window
            * A
            * df.exp(-1.0 * (t_since_activation_atr - t_trans_atr) / tr_atr)
        )

        return w2

    def getCt(self):
        return self.w1_atr() + self.w2_atr()

    def getw1_atr(self):
        return self.w1_atr()

    def getw2_atr(self):
        return self.w2_atr()

    def get_t_atr_since_act(self):
        t_a = self.parameters["t_a"]  # current time
        cycle = self.parameters["cycle"]  # current cycle
        BCL = self.parameters["HeartBeatLength"]
        tdelay_atr = self.parameters["material params"].get("tdelay_atr", 0.0)
        return self._atrial_activation_time(t_a, cycle, tdelay_atr, BCL)

    def _atrial_activation_time(self, t_a, cycle, tdelay_atr, BCL):
        t_init_atr = tdelay_atr
        t_atr = t_a - t_init_atr
        xpp = df.conditional(
            df.gt(cycle, df.Constant(0.001)),
            1.0,
            df.conditional(df.gt(t_a - t_init_atr, 0), 1.0, 0.0),
        )
        return xpp * df.conditional(
            df.gt(t_atr, 0), t_a - t_init_atr, t_a + (BCL - t_init_atr)
        )
