from dolfin import *
import sys
sys.path.append("/mnt/Research")
from fenicstools import *
import numpy as np

from ..utils.oops_objects_MRC2 import printout
from .circ import CLmodel
from .circBiV import CLmodel as CLmodel_biv

class coupleMEandCirc(object):

    def __init__(self, MEmodel, MEparams, SimDet, state_obj):
        self.MEmodel = MEmodel
        self.MEparams = MEparams
        self.CLmodel = CLmodel
        self.SimDet = SimDet
        self.state_obj = state_obj
    
        if "isLV" in list(SimDet.keys()):
            self.isLV = SimDet["isLV"]
        else:
            self.isLV = False  # Default
        if "iswaorta" in list(SimDet.keys()):
            self.iswaorta = SimDet["iswaorta"]
        else:
            self.iswaorta = False  # Default
        if "isFCH" in list(SimDet.keys()):
            self.isFCH = SimDet["isFCH"]
        else:
            self.isFCH = False  # Default
        if "isBiV" in list(SimDet.keys()):
            self.isBiV = SimDet["isBiV"]
        else:
            self.isBiV = False  # Default

        if self.isLV or self.iswaorta:
            if SimDet.get("lv_lumped"):
                self.CLmodel = CLmodel(SimDet)
            else:
                self.CLmodel = CLmodel(SimDet, MEmodel.GetLVV())
        elif self.isBiV or self.isFCH:
            if SimDet.get("fch_fe"):
                self.CLmodel = CLmodel_biv(SimDet, MEmodel.GetLVV(), MEmodel.GetRVV(), 
                        MEmodel.GetLAV(), MEmodel.GetRAV())
            elif SimDet.get("fch_lumped"):
                self.CLmodel = CLmodel_biv(SimDet)
            else:
                self.CLmodel = CLmodel_biv(SimDet, MEmodel.GetLVV(), MEmodel.GetRVV())

    def UpdateMEandCirc(self):

        MEmodel_ = self.MEmodel
        SimDet = self.SimDet
        CLmodel_ = self.CLmodel
        state_obj = self.state_obj
        isLV = self.isLV 
        isBiV = self.isBiV
        iswaorta = self.iswaorta
        isFCH = self.isFCH
        CLmodel_ = self.CLmodel

        P_LV = MEmodel_.GetLVP()  # LVCavitypres.pres
        V_LV = MEmodel_.GetLVV()  # GetVolumeComputation()

        comm_me = MEmodel_.mesh.mpi_comm()
        solver_ME = MEmodel_.Solver()

        if isBiV or isFCH:
            if SimDet.get("fch_fe"):
                P_LA = MEmodel_.GetLAP()  # set an initial value for P_LA and P_RA
                V_LA = MEmodel_.GetLAV()
                P_RA = MEmodel_.GetRAP()
                V_RA = MEmodel_.GetRAV()
    
                P_RV = MEmodel_.GetRVP()
                V_RV = MEmodel_.GetRVV()
            elif SimDet.get("fch_lumped"):
                P_LV = SimDet.get("EDP") / 0.0075
                P_RV = SimDet.get("EDP") / 0.0075
            else:
                P_RV = MEmodel_.GetRVP()  # LVCavitypres.pres
                V_RV = MEmodel_.GetRVV()  # GetVolumeComputation()

        if not SimDet.get("fch_lumped") and not SimDet.get("lv_lumped"):
            params = {
                "P_LV": P_LV,
                "V_LV": V_LV,
                "t": state_obj.t,
                "delTat": state_obj.dt.dt,
            }
        else:
            params = {
                "t": state_obj.t,
                "delTat": state_obj.dt.dt,
            }

        if isBiV or isFCH:
            if SimDet.get("fch_fe"):
                params.update(
                    {
                        "P_RV": P_RV,
                        "V_RV": V_RV,
                        "P_LA": P_LA,
                        "V_LA": V_LA,
                        "P_RA": P_RA,
                        "V_RA": V_RA,
                    }
                )
            elif SimDet.get("fch_lumped"):
                pass
            else:
                params.update(
                    {
                        "P_RV": P_RV,
                        "V_RV": V_RV,
                    }
                )

        if isLV or iswaorta:
            if SimDet.get("lv_lumped"):
                V_LV, V_LA = CLmodel_.UpdateLVV(params)
            else:
                V_LV = CLmodel_.UpdateLVV(params)

        elif isBiV or isFCH:
            V_LV, V_RV, *extra = CLmodel_.UpdateLVV(params)
            if SimDet.get("fch_fe") or SimDet.get("fch_lumped"):
                V_LA, V_RA = extra

        print_message = (  # todo: Generalize for all cases
            "t = "
            + str(state_obj.t)
            + " VLV = "
            + str(CLmodel_.V_LV)
            + " Psa = "
            + str(CLmodel_.Psa)
            + " PLV = "
            + str(CLmodel_.PLV)
            + " P_LA = "
            + str(CLmodel_.PLA)
            + " Qmv = "
            + str(CLmodel_.Qmv)
            + " Qav = "
            + str(CLmodel_.Qav)
        )
        if isBiV or isFCH:
            if SimDet.get("fch_fe"):
                print_message += " PLA = " + str(P_LA)
            else:
                print_message += " PLA = " + str(CLmodel_.PLA)
            print_message += "\n"
            print_message += " VRV = " + str(V_RV)
            print_message += " Ppa = " + str(CLmodel_.Ppa)
            print_message += " PRV = " + str(CLmodel_.PRV)
            if SimDet.get("fch_fe"):
                print_message += " PRA = " + str(P_RA)
            else:
                print_message += " PRA = " + str(CLmodel_.PRA)

        printout(print_message, comm_me)


        MEmodel_.LVCavitypres.pres = P_LV #LCL

        def Rp(plv, vlv):
            MEmodel_.LVCavitypres.pres = plv
            if SimDet.get("aorta_pres"):
                MEmodel_.AortaCavitypres.pres = CLmodel_.Psa  # * 5e-1
            solver_ME.solvenonlinear()
            v_t = MEmodel_.GetLVV()
            return v_t - vlv

        def Rp_biv(plv, prv, lvvc, rvvc):
            MEmodel_.LVCavitypres.pres = plv
            MEmodel_.RVCavitypres.pres = prv
            solver_ME.solvenonlinear()
            vlv = MEmodel_.GetLVV()
            vrv = MEmodel_.GetRVV()
            return [vlv - lvvc, vrv - rvvc]

        def Rp_fch(plv, prv, pla, pra, lvvc, rvvc, lavc, ravc):
            MEmodel_.LVCavitypres.pres = plv
            MEmodel_.RVCavitypres.pres = prv
            MEmodel_.LACavitypres.pres = pla
            MEmodel_.RACavitypres.pres = pra
            solver_ME.solvenonlinear()
            vlv = MEmodel_.GetLVV()
            vrv = MEmodel_.GetRVV()
            vla = MEmodel_.GetLAV()
            vra = MEmodel_.GetRAV()
            return np.array([vlv - lvvc, vrv - rvvc, vla - lavc, vra - ravc])

        # Create the Newton solver
        from scipy.optimize import (
            newton,
            fsolve,
            #root_scalar,
            bisect,
            root,
            minimize,
        )

        def Rp_call(plv):
            # scale = 1e6
            return Rp(plv, V_LV)  # / scale

        def Rp_biv_call(plrv):
            plv, prv = plrv
            return Rp_biv(plv, prv, V_LV, V_RV)

        def Rp_fch_call(plrva):
            plv, prv, pla, pra = plrva
            return Rp_fch(plv, prv, pla, pra, V_LV, V_RV, V_LA, V_RA)

        if isLV or iswaorta:
            x0 = [P_LV]
        elif isBiV or isFCH:
            if SimDet.get("fch_fe"):
                x0 = [P_LV, P_RV, P_LA, P_RA]
            else:
                x0 = [P_LV, P_RV]

        if isLV or iswaorta:
            if not SimDet.get("lv_lumped"):
                root1, info, ier, msg = fsolve(
                    Rp_call, x0, xtol=1e-4, factor=0.01, full_output=True
                )

        elif isBiV or isFCH:
            if SimDet.get("fch_fe"):
                root1, info, ier, msg = fsolve(
                    Rp_fch_call, x0, xtol=1e-4, factor=0.01, full_output=True
                )
            elif SimDet.get("fch_lumped"):
                info = None
                msg = None
                pass
            else:
                root1, info, ier, msg = fsolve(
                    Rp_biv_call, x0, xtol=1e-4, factor=0.01, full_output=True
                )

        if not SimDet.get("fch_lumped") and not SimDet.get("lv_lumped"):
            P_LV = root1[0]

        if isBiV or isFCH:
            if SimDet.get("fch_fe"):
                P_RV = root1[1]
                P_LA = root1[2]
                P_RA = root1[3]
            elif SimDet.get("fch_lumped"):
                pass
            else:
                P_RV = root1[1]


        return info, msg


