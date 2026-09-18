import math


# Closed Loop
class CLmodel(object):
    def et(self, va=1, lr=1):  # ventricle=1 / left=1
        chamber = self._chamber_params[(va, lr)]
        Tmax = chamber["Tmax"]
        tau = chamber["tau"]
        t_chamber = self._time_since_activation(chamber["tdelay"])

        # Piecewise activation: half-sine upstroke then exponential relaxation.
        if t_chamber <= 1.5 * Tmax:
            return 0.5 * (math.sin((math.pi / Tmax) * t_chamber - math.pi / 2.0) + 1.0)
        return 0.5 * math.exp((-t_chamber + (1.5 * Tmax)) / tau)

    def default_parameters(self):  # default values for Q
        return {"Qsa": 0.0, "Qad": 0.0, "Qsv": 0.0, "Qav": 0.0, "Qmv": 0.0}

    def __init__(self, SimDet, V_LV=None, V_RV=None, V_LA=None, V_RA=None):
        self.parameters = self.default_parameters()
        self.SimDet = SimDet
        cl_params = self.SimDet["closedloopparam"]

        # Systemic circulation
        self.Csa = cl_params["Csa"]
        self.Cad = cl_params["Cad"]
        self.Csv = cl_params["Csv"]
        # biv
        self.Cpa = cl_params["Cpa"]
        self.Cpv = cl_params["Cpv"]

        self.Vsa0 = cl_params["Vsa0"]
        self.Vad0 = cl_params["Vad0"]
        self.Vsv0 = cl_params["Vsv0"]
        # biv
        self.Vpa0 = cl_params["Vpa0"]
        self.Vpv0 = cl_params["Vpv0"]

        self.Rsa = cl_params["Rsa"]
        self.Rad = cl_params["Rad"]
        self.Rsv = cl_params["Rsv"]
        self.Rav = cl_params["Rav"]
        self.Rmv = cl_params["Rmv"]

        # Regurgitation resistance
        self.Rav_rg = cl_params.get("Rav_rg", 1e9)

        self.V_sa = cl_params["V_sa"]
        self.V_ad = cl_params["V_ad"]
        self.V_sv = cl_params["V_sv"]

        self.Rpv = cl_params["Rpv"]
        self.Rtv = cl_params["Rtv"]
        self.Rpa = cl_params["Rpa"]
        self.Rpvv = cl_params["Rpvv"]

        self.V_pv = cl_params["V_pv"]
        self.V_pa = cl_params["V_pa"]

        self.LVADrpm = cl_params.get("Q_lvad_rpm", 0)
        self.QLVADFn = cl_params.get("Q_lvad_characteristic")
        self._has_lvad = self.QLVADFn is not None

        # for LV
        self.Ees_lv = cl_params.get("Ees_lv")
        self.V0_lv = cl_params.get("V0_lv")
        self.A_lv = cl_params.get("A_lv")
        self.B_lv = cl_params.get("B_lv")
        self.Tmax_lv = cl_params.get("Tmax_lv")
        self.tau_lv = cl_params.get("tau_lv")
        self.tdelay_lv = cl_params.get("tdelay_lv")

        # for RV
        self.Ees_rv = cl_params.get("Ees_rv")
        self.V0_rv = cl_params.get("V0_rv")
        self.A_rv = cl_params.get("A_rv")
        self.B_rv = cl_params.get("B_rv")
        self.Tmax_rv = cl_params.get("Tmax_rv")
        self.tau_rv = cl_params.get("tau_rv")
        self.tdelay_rv = cl_params.get("tdelay_rv")

        # for LA
        self.Ees_la = cl_params.get("Ees_la")
        self.A_la = cl_params.get("A_la")
        self.B_la = cl_params.get("B_la")
        self.V0_la = cl_params.get("V0_la")
        self.Tmax_la = cl_params.get("Tmax_la")
        self.tau_la = cl_params.get("tau_la")
        self.tdelay_la = cl_params.get("tdelay_la")

        # for RA
        self.Ees_ra = cl_params.get("Ees_ra")
        self.A_ra = cl_params.get("A_ra")
        self.B_ra = cl_params.get("B_ra")
        self.V0_ra = cl_params.get("V0_ra")
        self.Tmax_ra = cl_params.get("Tmax_ra")
        self.tau_ra = cl_params.get("tau_ra")
        self.tdelay_ra = cl_params.get("tdelay_ra")

        self._valve_hyst = SimDet.get("valve_hyst", 0.5)
        self._av_open = False
        self._mv_open = False
        self._pv_open = False  # pulmonary valve (PRV -> Ppa) mapped to Qpvv
        self._tv_open = False
        self._chamber_params = self._build_chamber_params(cl_params)

    def _build_chamber_params(self, cl_params):
        chamber_map = {
            (1, 1): ("lv", "V_LV"),
            (1, 0): ("rv", "V_RV"),
            (0, 1): ("la", "V_LA"),
            (0, 0): ("ra", "V_RA"),
        }
        params = {}
        for key, (prefix, volume_key) in chamber_map.items():
            getter = cl_params.get
            params[key] = {
                "Ees": getter(f"Ees_{prefix}"),
                "V0": getter(f"V0_{prefix}"),
                "A": getter(f"A_{prefix}"),
                "B": getter(f"B_{prefix}"),
                "Tmax": getter(f"Tmax_{prefix}"),
                "tau": getter(f"tau_{prefix}"),
                "tdelay": getter(f"tdelay_{prefix}"),
                "volume_key": volume_key,
            }
        return params

    def _time_since_activation(self, tdelay):
        """Return time relative to chamber activation within a heartbeat."""
        heartbeat = self.SimDet["HeartBeatLength"]
        t = self.parameters["t"]
        # Wrap across cycle end so activation remains continuous at t=0 reset.
        return t + tdelay if t < heartbeat - tdelay else t - heartbeat + tdelay

    def _update_valve_state(self, pressure_diff, is_open):
        """Apply hysteresis rules and return the updated valve state."""
        # Hysteresis reduces valve chattering near dP≈0 in explicit CL integration.
        if is_open and pressure_diff < -self._valve_hyst:
            return False
        if not is_open and pressure_diff > self._valve_hyst:
            return True
        return is_open

    @staticmethod
    def _forward_flow(pressure_diff, resistance, is_open):
        # Enforce ideal diode behavior (no reverse flow) while still honoring the hysteretic open-state.
        return pressure_diff / resistance if (pressure_diff > 0.0 and is_open) else 0.0

    @staticmethod
    def _regurg_flow(pressure_diff, forward_resistance, regurg_resistance, is_open):
        # Regurgitation always uses a large resistance; forward flow depends on the valve open state.
        if pressure_diff <= 0.0:
            return pressure_diff / regurg_resistance
        return pressure_diff / forward_resistance if is_open else 0.0

    def snapshot_valves(self):
        """Capture the hysteretic valve open/close flags. These are the ONLY state
        ``UpdateLVV`` mutates (compartment volumes are read from ``params`` and
        returned, not persisted). The driver snapshots this before a coupling step
        and restores it on a dt-backoff retry so the re-advance starts from the same
        valve state (mirrors the FE ``w_me`` rollback)."""
        return (self._av_open, self._mv_open, self._pv_open, self._tv_open)

    def restore_valves(self, state):
        """Restore the valve flags captured by :meth:`snapshot_valves`."""
        self._av_open, self._mv_open, self._pv_open, self._tv_open = state

    def UpdateLVV(self, params):
        self.parameters.update(params)
        V_LV = self.parameters["V_LV"]
        V_RV = self.parameters["V_RV"]
        V_LA = self.parameters["V_LA"]
        V_RA = self.parameters["V_RA"]

        V_sa = self.parameters["V_sa"]
        V_ad = self.parameters["V_ad"]
        V_sv = self.parameters["V_sv"]
        V_pa = self.parameters["V_pa"]
        V_pv = self.parameters["V_pv"]

        fch_fe = self.SimDet.get("fch_fe", True)
        fch_lumped = self.SimDet.get("fch_lumped", False)
        fch_biv = self.SimDet.get("fch_biv")
        if fch_biv is None:
            fch_biv = not fch_fe and not fch_lumped

        # Mode selection: FE atria pressures (fch_fe), fully lumped atria+ventricles (fch_lumped), or BiV-only.
        if fch_fe:
            PLA, PRA = self.parameters["P_LA"], self.parameters["P_RA"]
        elif fch_lumped:
            PLV, PRV, PLA, PRA = (
                self.GetPVALR(params, va=1, lr=1),
                self.GetPVALR(params, va=1, lr=0),
                self.GetPVALR(params, va=0, lr=1),
                self.GetPVALR(params, va=0, lr=0),
            )
        elif fch_biv:
            PLA, PRA = self.GetPVALR(params, va=0, lr=1), self.GetPVALR(
                params, va=0, lr=0
            )
        else:
            raise ValueError("Execution mode must be one of fch_fe, fch_lumped, or fch_biv.")

        Psa = (V_sa - self.Vsa0) / self.Csa
        Pad = (V_ad - self.Vad0) / self.Cad
        Psv = (V_sv - self.Vsv0) / self.Csv
        # biv
        Ppa = (V_pa - self.Vpa0) / self.Cpa
        Ppv = (V_pv - self.Vpv0) / self.Cpv

        if not fch_lumped:
            PLV = self.parameters["P_LV"]
            PRV = self.parameters["P_RV"]

        # LV valves
        dP_av = PLV - Psa
        self._av_open = self._update_valve_state(dP_av, self._av_open)
        Qav = self._regurg_flow(dP_av, self.Rav, self.Rav_rg, self._av_open)

        # Mitral valve (PLA -> PLV) with hysteresis
        dP_mv = PLA - PLV
        self._mv_open = self._update_valve_state(dP_mv, self._mv_open)
        Qmv = self._forward_flow(dP_mv, self.Rmv, self._mv_open)

        # RV valves
        dP_pv = PRV - Ppa
        self._pv_open = self._update_valve_state(dP_pv, self._pv_open)
        Qpvv = self._forward_flow(dP_pv, self.Rpvv, self._pv_open)

        dP_tv = PRA - PRV
        self._tv_open = self._update_valve_state(dP_tv, self._tv_open)
        Qtv = self._forward_flow(dP_tv, self.Rtv, self._tv_open)

        Qsa = (Psa - Pad) / self.Rsa
        Qad = (Pad - Psv) / self.Rad
        Qsv = (Psv - PRA) / self.Rsv
        # biv
        Qpa = (Ppa - Ppv) / self.Rpa
        Qpv = (Ppv - PLA) / self.Rpv

        if self._has_lvad:
            # LVAD characteristic is defined in (mmHg, rpm) with Flowrate in mL/min; convert to mL/ms.
            H = (Psa - PLV) * 0.0075  # Pump head in mmHg
            Qlvad = self.QLVADFn.Flowrate(H, self.LVADrpm) / 60  # mL/ms
        else:
            Qlvad = 0.0

        # Forward-Euler update of compartment volumes; `delTat` is the CL integration dt.
        dt = self.parameters["delTat"]
        V_LV += dt * (Qmv - Qav - Qlvad)
        V_sa += dt * (Qav - Qsa + Qlvad)
        V_ad += dt * (Qsa - Qad)
        V_sv += dt * (Qad - Qsv)
        V_RA += dt * (Qsv - Qtv)
        V_RV += dt * (Qtv - Qpvv)
        V_pa += dt * (Qpvv - Qpa)
        V_pv += dt * (Qpa - Qpv)
        V_LA += dt * (Qpv - Qmv)

        if fch_fe or fch_lumped:
            return (
                V_LV,
                V_RV,
                V_LA,
                V_RA,
                V_sa,
                V_ad,
                V_sv,
                V_pa,
                V_pv,
                Qmv,
                Qav,
                Qpvv,
                Qtv,
            )
        elif fch_biv:
            return V_LV, V_RV
        else:
            raise ValueError("Execution mode must be one of fch_fe, fch_lumped, or fch_biv.")

    def GetPVALR(self, params, va=1, lr=1):  # ventricle=1 / left=1
        self.parameters.update(params)
        chamber = self._chamber_params[(va, lr)]
        E_es = chamber["Ees"]
        V0 = chamber["V0"]
        A = chamber["A"]
        B = chamber["B"]
        volume_key = chamber["volume_key"]
        V = self.parameters[volume_key]
        delta_V = V - V0
        # Standard time-varying elastance: blends E_es*ΔV with an exponential diastolic curve.
        et_val = self.et(va=1 if va else 0, lr=1 if lr else 0)
        return et_val * E_es * delta_V + (1.0 - et_val) * A * (
            math.exp(B * delta_V) - 1.0
        )
