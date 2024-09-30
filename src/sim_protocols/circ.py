import math


# Closed Loop
class CLmodel(object):
    def et(self):  # for P_LA
        if self.t_la <= 1.5 * self.Tmax_la:
            out = 0.5 * (
                math.sin((math.pi / self.Tmax_la) * self.t_la - math.pi / 2) + 1
            )
        else:
            out = 0.5 * math.exp((-self.t_la + (1.5 * self.Tmax_la)) / self.tau_la)
        return out

    def et_lv(self):  # for P_LV
        if self.t_lv <= 1.5 * self.Tmax_lv:
            out = 0.5 * (
                math.sin((math.pi / self.Tmax_lv) * self.t_lv - math.pi / 2) + 1
            )
        else:
            out = 0.5 * math.exp((-self.t_lv + (1.5 * self.Tmax_lv)) / self.tau_lv)
        return out

    def default_parameters(self):  # default values for Q
        return {"Qsa": 0.0, "Qad": 0.0, "Qsv": 0.0, "Qav": 0.0, "Qmv": 0.0}

    def __init__(self, SimDet, V_LV):
        self.parameters = self.default_parameters()
        # self.parameters.update(params)
        self.SimDet = SimDet

        # Systemic circulation
        self.Csa = SimDet["closedloopparam"]["Csa"]
        self.Cad = SimDet["closedloopparam"]["Cad"]
        self.Csv = SimDet["closedloopparam"]["Csv"]

        self.Vsa0 = SimDet["closedloopparam"]["Vsa0"]
        self.Vad0 = SimDet["closedloopparam"]["Vad0"]
        self.Vsv0 = SimDet["closedloopparam"]["Vsv0"]

        self.Rsa = SimDet["closedloopparam"]["Rsa"]
        self.Rad = SimDet["closedloopparam"]["Rad"]
        self.Rsv = SimDet["closedloopparam"]["Rsv"]
        self.Rav = SimDet["closedloopparam"]["Rav"]
        self.Rmv = SimDet["closedloopparam"]["Rmv"]

        self.V_sa = SimDet["closedloopparam"]["V_sa"]
        self.V_ad = SimDet["closedloopparam"]["V_ad"]
        self.V_sv = SimDet["closedloopparam"]["V_sv"]
        self.V_LA = SimDet["closedloopparam"]["V_LA"]

        if "Q_sa" in list(SimDet["closedloopparam"].keys()):
            self.parameters["Qsa"] = SimDet["closedloopparam"]["Q_sa"]
        if "Q_ad" in list(SimDet["closedloopparam"].keys()):
            self.parameters["Qad"] = SimDet["closedloopparam"]["Q_ad"]
        if "Q_sv" in list(SimDet["closedloopparam"].keys()):
            self.parameters["Qsv"] = SimDet["closedloopparam"]["Q_sv"]
        if "Q_av" in list(SimDet["closedloopparam"].keys()):
            self.parameters["Qav"] = SimDet["closedloopparam"]["Q_av"]
        if "Q_mv" in list(SimDet["closedloopparam"].keys()):
            self.parameters["Qmv"] = SimDet["closedloopparam"]["Q_mv"]

        # self.Psa = 1.0/self.Csa*(self.V_sa - self.Vsa0)
        # self.Pad = 1.0/self.Cad*(self.V_ad - self.Vad0)
        # self.Psv = 1.0/self.Csv*(self.V_sv - self.Vsv0)

        # for LA
        self.Ees_la = SimDet["closedloopparam"]["Ees_la"]
        self.A_la = SimDet["closedloopparam"]["A_la"]
        self.B_la = SimDet["closedloopparam"]["B_la"]
        self.V0_la = SimDet["closedloopparam"]["V0_la"]
        self.Tmax_la = SimDet["closedloopparam"]["Tmax_la"]
        self.tau_la = SimDet["closedloopparam"]["tau_la"]
        self.tdelay_la = SimDet["closedloopparam"]["tdelay_la"]

        # For LV
        self.Ees_lv = SimDet["closedloopparam"]["Ees_lv"]
        self.V0_lv = SimDet["closedloopparam"]["V0_lv"]
        self.A_lv = SimDet["closedloopparam"]["A_lv"]
        self.B_lv = SimDet["closedloopparam"]["B_lv"]
        self.Tmax_lv = SimDet["closedloopparam"]["Tmax_lv"]
        self.tau_lv = SimDet["closedloopparam"]["tau_lv"]
        self.tdelay_lv = SimDet["closedloopparam"]["tdelay_lv"]
        # perhaps we initialized V_LV here
        self.V_LV = V_LV

    def UpdateLVV(self, params):
        self.parameters.update(params)
        self.PLA = self.GetPLoRA(params)

        self.Psa = 1.0 / self.Csa * (self.V_sa - self.Vsa0)
        self.Pad = 1.0 / self.Cad * (self.V_ad - self.Vad0)
        self.Psv = 1.0 / self.Csv * (self.V_sv - self.Vsv0)

        self.PLV = self.parameters["P_LV"]
        # self.V_LV = self.parameters["V_LV"]

        # update Q
        if self.PLV <= self.Psa:
            self.Qav = 0.0
        else:
            self.Qav = 1.0 / self.Rav * (self.PLV - self.Psa)

        if self.PLV >= self.PLA:
            self.Qmv = 0.0
        else:
            self.Qmv = 1.0 / self.Rmv * (self.PLA - self.PLV)

        self.Qsa = 1.0 / self.Rsa * (self.Psa - self.Pad)
        self.Qad = 1.0 / self.Rad * (self.Pad - self.Psv)
        self.Qsv = 1.0 / self.Rsv * (self.Psv - self.PLA)

        self.V_LV = self.V_LV + self.parameters["delTat"] * (self.Qmv - self.Qav)
        self.V_sa = self.V_sa + self.parameters["delTat"] * (self.Qav - self.Qsa)
        self.V_ad = self.V_ad + self.parameters["delTat"] * (self.Qsa - self.Qad)
        self.V_sv = self.V_sv + self.parameters["delTat"] * (self.Qad - self.Qsv)
        self.V_LA = self.V_LA + self.parameters["delTat"] * (self.Qsv - self.Qmv)

        return self.V_LV

    def GetPLoRA(self, params):
        self.parameters.update(params)
        # For PLA
        if self.parameters["t"] < self.SimDet["HeartBeatLength"] - self.tdelay_la:
            self.t_la = self.parameters["t"] + self.tdelay_la
        else:
            self.t_la = (
                self.parameters["t"] - self.SimDet["HeartBeatLength"] + self.tdelay_la
            )

        self.PLA = self.et() * self.Ees_la * (self.V_LA - self.V0_la) + (
            1 - self.et()
        ) * self.A_la * (math.exp(self.B_la * (self.V_LA - self.V0_la)) - 1)
        return self.PLA

    def GetLVV(self):
        return self.V_LV

    def GetPLoRV(self, params):

        self.parameters.update(params)
        # For PLV
        if self.parameters["t"] < self.SimDet["HeartBeatLength"] - self.tdelay_lv:
            self.t_lv = self.parameters["t"] + self.tdelay_lv
        else:
            self.t_lv = (
                self.parameters["t"] - self.SimDet["HeartBeatLength"] + self.tdelay_lv
            )

        PLV = 0
        # PLV = e(t_LV,Tmax_LV,tau_LV, trans_LV)*Ees_LV.*(VLV - V0_LV) + (1 - e(t_LV,Tmax_LV,tau_LV,trans_LV))*A_LV
        # Ees_lv, V0_lv, A_lv
        # PLV = (
        #    self.et_lv() * self.Ees_lv * (self.V_LV - self.V0_lv)
        #    + (1 - self.et_lv()) * self.A_lv
        # )
        self.PLV = self.et_lv() * self.Ees_lv * (self.V_LV - self.V0_lv) + (
            1 - self.et_lv()
        ) * self.A_lv * (math.exp(self.B_lv * (self.V_LV - self.V0_lv)) - 1)
        return self.PLV
