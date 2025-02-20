import math


# Closed Loop
class CLmodel(object):
    def et(self, la):
        if self.parameters["t"] < self.SimDet["HeartBeatLength"] - self.tdelay_la:
            t_la = self.parameters["t"] + self.tdelay_la
        else:
            t_la = (
                self.parameters["t"] - self.SimDet["HeartBeatLength"] + self.tdelay_la
            )  #

        if la:
            if t_la <= 1.5 * self.Tmax_la:
                out = 0.5 * (
                    math.sin((math.pi / self.Tmax_la) * t_la - math.pi / 2.0) + 1.0
                )
            else:
                out = 0.5 * math.exp((-t_la + (1.5 * self.Tmax_la)) / self.tau_la)
        else:
            if t_la <= 1.5 * self.Tmax_ra:
                out = 0.5 * (
                    math.sin((math.pi / self.Tmax_ra) * t_la - math.pi / 2.0) + 1.0
                )
            else:
                out = 0.5 * math.exp((-t_la + (1.5 * self.Tmax_ra)) / self.tau_ra)
        return out

    def default_parameters(self):  # default values for Q
        return {"Qsa": 0.0, "Qad": 0.0, "Qsv": 0.0, "Qav": 0.0, "Qmv": 0.0}

    def __init__(self, SimDet, V_LV, V_RV):
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
        self.V_LA = SimDet["closedloopparam"]["V_LA"]

        self.Rpv = SimDet["closedloopparam"]["Rpv"]
        self.Rtv = SimDet["closedloopparam"]["Rtv"]
        self.Rpa = SimDet["closedloopparam"]["Rpa"]
        self.Rpvv = SimDet["closedloopparam"]["Rpvv"]

        self.V_pv = SimDet["closedloopparam"]["V_pv"]
        self.V_pa = SimDet["closedloopparam"]["V_pa"]
        self.V_RA = SimDet["closedloopparam"]["V_RA"]

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

        # initialize V_LV
        self.V_LV = V_LV
        self.V_RV = V_RV

    def UpdateLVV(self, params):
        self.parameters.update(params)

        self.PLA = self.GetPLoRA(params, 1)  # la = 1
        self.PRA = self.GetPLoRA(params, 0)  # ra = 0

        self.Psa = 1.0 / self.Csa * (self.V_sa - self.Vsa0)
        self.Pad = 1.0 / self.Cad * (self.V_ad - self.Vad0)
        self.Psv = 1.0 / self.Csv * (self.V_sv - self.Vsv0)
        # biv
        self.Ppa = 1.0 / self.Cpa * (self.V_pa - self.Vpa0)
        self.Ppv = 1.0 / self.Cpv * (self.V_pv - self.Vpv0)

        self.PLV = self.parameters["P_LV"]
        self.PRV = self.parameters["P_RV"]

        # update Q
        ## For LV
        if self.PLV <= self.Psa:
            # self.Qav = 0.0
            self.Qav = 1.0 / self.Rav_rg * (self.PLV - self.Psa)
        else:
            self.Qav = 1.0 / self.Rav * (self.PLV - self.Psa)

        if self.PLV >= self.PLA:
            self.Qmv = 0.0  # Mitral valve
        else:
            self.Qmv = 1.0 / self.Rmv * (self.PLA - self.PLV)

        ## For RV
        if self.PRV <= self.Ppa:
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
        )  # - self.Qlvad)
        self.V_sa = self.V_sa + self.parameters["delTat"] * (
            self.Qav - self.Qsa
        )  # + self.Qlvad)
        self.V_ad = self.V_ad + self.parameters["delTat"] * (self.Qsa - self.Qad)
        self.V_sv = self.V_sv + self.parameters["delTat"] * (self.Qad - self.Qsv)
        self.V_RA = self.V_RA + self.parameters["delTat"] * (self.Qsv - self.Qtv)
        self.V_RV = self.V_RV + self.parameters["delTat"] * (self.Qtv - self.Qpvv)
        self.V_pa = self.V_pa + self.parameters["delTat"] * (self.Qpvv - self.Qpa)
        self.V_pv = self.V_pv + self.parameters["delTat"] * (self.Qpa - self.Qpv)
        self.V_LA = self.V_LA + self.parameters["delTat"] * (
            self.Qpv - self.Qmv
        )  # LCL fixed bug

        return self.V_LV, self.V_RV

    def GetPLoRA(self, params, la):
        self.parameters.update(params)
        # For PLA
        if self.parameters["t"] < self.SimDet["HeartBeatLength"] - self.tdelay_la:
            t_la = self.parameters["t"] + self.tdelay_la
        else:
            t_la = (
                self.parameters["t"] - self.SimDet["HeartBeatLength"] + self.tdelay_la
            )  #
        if la:
            out = self.et(la) * self.Ees_la * (self.V_LA - self.V0_la) + (
                1.0 - self.et(la)
            ) * self.A_la * (math.exp(self.B_la * (self.V_LA - self.V0_la)) - 1.0)
        else:
            out = self.et(la) * self.Ees_ra * (self.V_RA - self.V0_ra) + (
                1.0 - self.et(la)
            ) * self.A_ra * (math.exp(self.B_ra * (self.V_RA - self.V0_ra)) - 1.0)

        return out
