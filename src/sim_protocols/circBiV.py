import math


# Closed Loop
class CLmodel(object):
    def et(self, va=1, lr=1):  # ventricle=1 / left=1
        tdelay = (
            (self.tdelay_lv if lr else self.tdelay_rv)
            if va
            else (self.tdelay_la if lr else self.tdelay_ra)
        )

        tau = (
            (self.tau_lv if lr else self.tau_rv)
            if va
            else (self.tau_la if lr else self.tau_ra)
        )

        Tmax = (
            (self.Tmax_lv if lr else self.Tmax_rv)
            if va
            else (self.Tmax_la if lr else self.Tmax_ra)
        )

        # t_la needs a rename
        t_la = (
            self.parameters["t"] + tdelay
            if self.parameters["t"] < self.SimDet["HeartBeatLength"] - tdelay
            else self.parameters["t"] - self.SimDet["HeartBeatLength"] + tdelay
        )

        if t_la <= 1.5 * Tmax:
            out = 0.5 * (math.sin((math.pi / Tmax) * t_la - math.pi / 2.0) + 1.0)
        else:
            out = 0.5 * math.exp((-t_la + (1.5 * Tmax)) / tau)
        return out

    def default_parameters(self):  # default values for Q
        return {"Qsa": 0.0, "Qad": 0.0, "Qsv": 0.0, "Qav": 0.0, "Qmv": 0.0}

    def __init__(self, SimDet, V_LV=None, V_RV=None, V_LA=None, V_RA=None):
        self.parameters = self.default_parameters()
        # self.parameters.update(params)
        self.SimDet = SimDet

        # Systemic circulation
        self.Csa = SimDet["closedloopparam"]["Csa"]
        self.Cad = SimDet["closedloopparam"]["Cad"]
        self.Csv = SimDet["closedloopparam"]["Csv"]
        # biv
        self.Cpa = SimDet["closedloopparam"]["Cpa"]
        self.Cpv = SimDet["closedloopparam"]["Cpv"]

        self.Vsa0 = SimDet["closedloopparam"]["Vsa0"]
        self.Vad0 = SimDet["closedloopparam"]["Vad0"]
        self.Vsv0 = SimDet["closedloopparam"]["Vsv0"]
        # biv
        self.Vpa0 = SimDet["closedloopparam"]["Vpa0"]
        self.Vpv0 = SimDet["closedloopparam"]["Vpv0"]

        self.Rsa = SimDet["closedloopparam"]["Rsa"]
        self.Rad = SimDet["closedloopparam"]["Rad"]
        self.Rsv = SimDet["closedloopparam"]["Rsv"]
        self.Rav = SimDet["closedloopparam"]["Rav"]
        self.Rmv = SimDet["closedloopparam"]["Rmv"]

        # Regurgitation resistance
        self.Rav_rg = 1e9
        if "Rav_rg" in list(SimDet["closedloopparam"].keys()):
            self.Rav_rg = SimDet["closedloopparam"]["Rav_rg"]

        self.V_sa = SimDet["closedloopparam"]["V_sa"]
        self.V_ad = SimDet["closedloopparam"]["V_ad"]
        self.V_sv = SimDet["closedloopparam"]["V_sv"]

        self.Rpv = SimDet["closedloopparam"]["Rpv"]
        self.Rtv = SimDet["closedloopparam"]["Rtv"]
        self.Rpa = SimDet["closedloopparam"]["Rpa"]
        self.Rpvv = SimDet["closedloopparam"]["Rpvv"]

        self.V_pv = SimDet["closedloopparam"]["V_pv"]
        self.V_pa = SimDet["closedloopparam"]["V_pa"]

        # if "Q_sa" in list(SimDet["closedloopparam"].keys()):
        #    self.parameters["Qsa"] = SimDet["closedloopparam"]["Q_sa"]
        # if "Q_ad" in list(SimDet["closedloopparam"].keys()):
        #    self.parameters["Qad"] = SimDet["closedloopparam"]["Q_ad"]
        # if "Q_sv" in list(SimDet["closedloopparam"].keys()):
        #    self.parameters["Qsv"] = SimDet["closedloopparam"]["Q_sv"]
        # if "Q_av" in list(SimDet["closedloopparam"].keys()):
        #    self.parameters["Qav"] = SimDet["closedloopparam"]["Q_av"]
        # if "Q_mv" in list(SimDet["closedloopparam"].keys()):
        #    self.parameters["Qmv"] = SimDet["closedloopparam"]["Q_mv"]

        # Parameters for LVAD #############################################
        self.LVADrpm = 0
        if "Q_lvad_rpm" in list(SimDet["closedloopparam"].keys()):
            self.LVADrpm = self.SimDet["closedloopparam"]["Q_lvad_rpm"]
        if "Q_lvad_characteristic" in list(SimDet["closedloopparam"].keys()):
            self.QLVADFn = self.SimDet["closedloopparam"]["Q_lvad_characteristic"]

        self.Qlvad = 0

        # for LV
        self.Ees_lv = SimDet["closedloopparam"].get("Ees_lv")
        self.V0_lv = SimDet["closedloopparam"].get("V0_lv")
        self.A_lv = SimDet["closedloopparam"].get("A_lv")
        self.B_lv = SimDet["closedloopparam"].get("B_lv")
        self.Tmax_lv = SimDet["closedloopparam"].get("Tmax_lv")
        self.tau_lv = SimDet["closedloopparam"].get("tau_lv")
        self.tdelay_lv = SimDet["closedloopparam"].get("tdelay_lv")

        # for RV
        self.Ees_rv = SimDet["closedloopparam"].get("Ees_rv")
        self.V0_rv = SimDet["closedloopparam"].get("V0_rv")
        self.A_rv = SimDet["closedloopparam"].get("A_rv")
        self.B_rv = SimDet["closedloopparam"].get("B_rv")
        self.Tmax_rv = SimDet["closedloopparam"].get("Tmax_rv")
        self.tau_rv = SimDet["closedloopparam"].get("tau_rv")
        self.tdelay_rv = SimDet["closedloopparam"].get("tdelay_rv")

        # for LA
        self.Ees_la = SimDet["closedloopparam"]["Ees_la"]
        self.A_la = SimDet["closedloopparam"]["A_la"]
        self.B_la = SimDet["closedloopparam"]["B_la"]
        self.V0_la = SimDet["closedloopparam"]["V0_la"]
        self.Tmax_la = SimDet["closedloopparam"]["Tmax_la"]
        self.tau_la = SimDet["closedloopparam"]["tau_la"]
        self.tdelay_la = SimDet["closedloopparam"]["tdelay_la"]

        # for RA
        self.Ees_ra = SimDet["closedloopparam"]["Ees_ra"]
        self.A_ra = SimDet["closedloopparam"]["A_ra"]
        self.B_ra = SimDet["closedloopparam"]["B_ra"]
        self.V0_ra = SimDet["closedloopparam"]["V0_ra"]
        self.Tmax_ra = SimDet["closedloopparam"]["Tmax_ra"]
        self.tau_ra = SimDet["closedloopparam"]["tau_ra"]
        self.tdelay_ra = SimDet["closedloopparam"]["tdelay_ra"]

        # initialize V
        self.V_LV = V_LV or SimDet["closedloopparam"]["V_LV"]
        self.V_RV = V_RV or SimDet["closedloopparam"]["V_RV"]
        self.V_LA = V_LA or SimDet["closedloopparam"]["V_LA"]
        self.V_RA = V_RA or SimDet["closedloopparam"]["V_RA"]

    def UpdateLVV(self, params):
        self.parameters.update(params)

        if self.SimDet.get("fch_fe"):
            self.PLA, self.PRA = self.parameters["P_LA"], self.parameters["P_RA"]
        elif self.SimDet.get("fch_lumped"):
            self.PLV, self.PRV, self.PLA, self.PRA = (
                self.GetPVALR(params, va=1, lr=1),
                self.GetPVALR(params, va=1, lr=0),
                self.GetPVALR(params, va=0, lr=1),
                self.GetPVALR(params, va=0, lr=0),
            )
        else:
            self.PLA, self.PRA = self.GetPVALR(params, va=0, lr=1), self.GetPVALR(
                params, va=0, lr=0
            )

        self.Psa = 1.0 / self.Csa * (self.V_sa - self.Vsa0)
        self.Pad = 1.0 / self.Cad * (self.V_ad - self.Vad0)
        self.Psv = 1.0 / self.Csv * (self.V_sv - self.Vsv0)
        # biv
        self.Ppa = 1.0 / self.Cpa * (self.V_pa - self.Vpa0)
        self.Ppv = 1.0 / self.Cpv * (self.V_pv - self.Vpv0)

        if not self.SimDet.get("fch_lumped"):
            self.PLV = self.parameters["P_LV"]
            self.PRV = self.parameters["P_RV"]

        # update Q
        ## For LV
        if self.PLV <= self.Psa:  # Aortic valve
            # self.Qav = 0.0
            self.Qav = 1.0 / self.Rav_rg * (self.PLV - self.Psa)
        else:
            self.Qav = 1.0 / self.Rav * (self.PLV - self.Psa)

        if self.PLV >= self.PLA:  # Mitral valve
            self.Qmv = 0.0
        else:
            self.Qmv = 1.0 / self.Rmv * (self.PLA - self.PLV)

        ## For RV
        if self.PRV <= self.Ppa:  # Pulmonary valve
            self.Qpvv = 0.0
        else:
            self.Qpvv = 1.0 / self.Rpvv * (self.PRV - self.Ppa)
        if self.PRV >= self.PRA:  # Tricuspid valve
            self.Qtv = 0.0
        else:
            self.Qtv = 1.0 / self.Rtv * (self.PRA - self.PRV)

        self.Qsa = 1.0 / self.Rsa * (self.Psa - self.Pad)
        self.Qad = 1.0 / self.Rad * (self.Pad - self.Psv)
        self.Qsv = 1.0 / self.Rsv * (self.Psv - self.PRA)
        # biv
        self.Qpa = 1.0 / self.Rpa * (self.Ppa - self.Ppv)
        self.Qpv = 1.0 / self.Rpv * (self.Ppv - self.PLA)

        if "Q_lvad_characteristic" in list(self.SimDet["closedloopparam"].keys()):
            H = (self.Psa - self.PLV) * 0.0075  # Pump head in mmHg
            self.Qlvad = (
                self.QLVADFn.Flowrate(H, self.LVADrpm) / 60
            )  # Flow rate of LVAD in mL/ms

        self.V_LV = self.V_LV + self.parameters["delTat"] * (
            self.Qmv - self.Qav
            - self.Qlvad)
        self.V_sa = self.V_sa + self.parameters["delTat"] * (
            self.Qav - self.Qsa
            + self.Qlvad)
        self.V_ad = self.V_ad + self.parameters["delTat"] * (self.Qsa - self.Qad)
        self.V_sv = self.V_sv + self.parameters["delTat"] * (self.Qad - self.Qsv)
        self.V_RA = self.V_RA + self.parameters["delTat"] * (self.Qsv - self.Qtv)
        self.V_RV = self.V_RV + self.parameters["delTat"] * (self.Qtv - self.Qpvv)
        self.V_pa = self.V_pa + self.parameters["delTat"] * (self.Qpvv - self.Qpa)
        self.V_pv = self.V_pv + self.parameters["delTat"] * (self.Qpa - self.Qpv)
        self.V_LA = self.V_LA + self.parameters["delTat"] * (self.Qpv - self.Qmv)  

        if self.SimDet.get("fch_fe") or self.SimDet.get("fch_lumped"):
            return self.V_LV, self.V_RV, self.V_LA, self.V_RA
        else:
            return self.V_LV, self.V_RV

    def GetPVALR(self, params, va=1, lr=1):  # ventricle=1 / left=1
        self.parameters.update(params)

        # ventricle or atrium / left or right
        E_es = (
            (self.Ees_lv if lr else self.Ees_rv)
            if va
            else (self.Ees_la if lr else self.Ees_ra)
        )
        V0 = (
            (self.V0_lv if lr else self.V0_rv)
            if va
            else (self.V0_la if lr else self.V0_ra)
        )
        V = (self.V_LV if lr else self.V_RV) if va else (self.V_LA if lr else self.V_RA)

        A = (self.A_lv if lr else self.A_rv) if va else (self.A_la if lr else self.A_ra)
        B = (self.B_lv if lr else self.B_rv) if va else (self.B_la if lr else self.B_ra)

        # et_val = self.et(va=0 if not va else None, lr=0 if not lr else None)
        et_val = self.et(va=0 if not va else 1, lr=0 if not lr else 1)

        out = et_val * E_es * (V - V0) + (1.0 - et_val) * A * (
            math.exp(B * (V - V0)) - 1.0
        )

        return out
