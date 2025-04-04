import math

def softplus(x, y, alpha):

    if alpha*(x - y) < 20:
         return 1.0/alpha*math.log(1 + math.exp(alpha*(x - y)));
    else:
         return (x - y);



# Closed Loop
class CLmodel(object):
    def et(self, va=1):  # ventricle / atrium=default(1)

        tdelay = self.tdelay_la if va else self.tdelay_lv
        t_la = (
            self.parameters["t"] + tdelay
            if self.parameters["t"] < self.SimDet["HeartBeatLength"] - tdelay # LA: (if t < 640 t+160 else t - 640)
            else self.parameters["t"] - self.SimDet["HeartBeatLength"] + tdelay
        )
        tau = self.tau_la if va else self.tau_lv

        Tmax = self.Tmax_la if va else self.Tmax_lv

        if t_la <= 1.5 * Tmax:
        # if self.parameters["t"] <= 1.5 * Tmax:
            out = 0.5 * (math.sin((math.pi / Tmax) * t_la - math.pi / 2) + 1)
            # out = 0.5 * (math.sin((math.pi / Tmax) * self.parameters["t"] - math.pi / 2) + 1)
        else:
            out = 0.5 * math.exp((-t_la + (1.5 * Tmax)) / tau)
            # out = 0.5 * math.exp((-self.parameters["t"] + (1.5 * Tmax)) / tau)
        return out

    def default_parameters(self):  # default values for Q
        return {"Qsa": 0.0, "Qad": 0.0, "Qsv": 0.0, "Qav": 0.0, "Qmv": 0.0}

    def __init__(self, SimDet, V_LV=None):
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

        # for LV
        self.Ees_lv = SimDet["closedloopparam"].get("Ees_lv")
        self.V0_lv = SimDet["closedloopparam"].get("V0_lv")
        self.A_lv = SimDet["closedloopparam"].get("A_lv")
        self.B_lv = SimDet["closedloopparam"].get("B_lv")
        self.Tmax_lv = SimDet["closedloopparam"].get("Tmax_lv")
        self.tau_lv = SimDet["closedloopparam"].get("tau_lv")
        self.tdelay_lv = SimDet["closedloopparam"].get("tdelay_lv")

        # for LA
        self.Ees_la = SimDet["closedloopparam"]["Ees_la"]
        self.A_la = SimDet["closedloopparam"]["A_la"]
        self.B_la = SimDet["closedloopparam"]["B_la"]
        self.V0_la = SimDet["closedloopparam"]["V0_la"]
        self.Tmax_la = SimDet["closedloopparam"]["Tmax_la"]
        self.tau_la = SimDet["closedloopparam"]["tau_la"]
        self.tdelay_la = SimDet["closedloopparam"]["tdelay_la"]

        # initialize V_LV here
        self.V_LV = V_LV or SimDet["closedloopparam"]["V_LV"]


    def UpdateLVV(self, params):
        self.parameters.update(params)

        self.PLA = self.GetPVA(params, va=1)

        self.Psa = 1.0 / self.Csa * (self.V_sa - self.Vsa0)
        self.Pad = 1.0 / self.Cad * (self.V_ad - self.Vad0)
        self.Psv = 1.0 / self.Csv * (self.V_sv - self.Vsv0)

        if self.SimDet.get("lv_lumped"):
            self.PLV = self.GetPVA(params, va=0)
        else:
            self.PLV = self.parameters["P_LV"]

        # update Q
        if self.PLV <= self.Psa:  # Aortic valve
            self.Qav = 0.0
        else:
            self.Qav = 1.0 / self.Rav * (self.PLV - self.Psa)

        if self.PLV >= self.PLA:  # Mitral valve
            self.Qmv = 0.0
        else:
            self.Qmv = 1.0 / self.Rmv * (self.PLA - self.PLV)

        # Use softplus to update Q 
        if "issoftplus" in list(self.SimDet["closedloopparam"].keys()):
            if self.SimDet["closedloopparam"]["issoftplus"] :
                alpha_av = 5e-3
                self.Qav = 1.0/self. Rav*softplus(self.PLV, self.Psa, alpha_av)
                alpha_mv = 1e-3
                self.Qmv = 1.0/self. Rmv*softplus(self.PLA, self.PLV, alpha_mv)
        
        self.Qsa = 1.0 / self.Rsa * (self.Psa - self.Pad)
        self.Qad = 1.0 / self.Rad * (self.Pad - self.Psv)
        self.Qsv = 1.0 / self.Rsv * (self.Psv - self.PLA)

        self.V_LV = self.V_LV + self.parameters["delTat"] * (self.Qmv - self.Qav)
        self.V_sa = self.V_sa + self.parameters["delTat"] * (self.Qav - self.Qsa)
        self.V_ad = self.V_ad + self.parameters["delTat"] * (self.Qsa - self.Qad)
        self.V_sv = self.V_sv + self.parameters["delTat"] * (self.Qad - self.Qsv)
        self.V_LA = self.V_LA + self.parameters["delTat"] * (self.Qsv - self.Qmv)

        if self.SimDet.get("lv_lumped"):
            return self.V_LV, self.V_LA
        else:
            return self.V_LV

    def GetPVA(self, params, va=1):
        self.parameters.update(params)
        # ventricle or atrium=default(1)
        E_es = self.Ees_la if va else self.Ees_lv
        V0 = self.V0_la if va else self.V0_lv
        V = self.V_LA if va else self.V_LV

        A = self.A_la if va else self.A_lv
        B = self.B_la if va else self.B_lv

        et_val = self.et(va=1 if va else 0)

        out = et_val * E_es * (V - V0) + (1 - et_val) * A * (math.exp(B * (V - V0)) - 1)
        return out
