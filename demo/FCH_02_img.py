import sys, pdb
from dolfin import *

sys.path.append("/mnt/Research")

from heArt_py3.src.sim_protocols.run_BiV_ClosedLoop_pctrl import (
    run_BiV_ClosedLoop as run_BiV_ClosedLoop,
)

#from heArt_py3.src.postprocessing.postprocessdata2 import (
#    postprocessdata as postprocessdata,
#)
#from heArt_py3.src.postprocessing.postprocessdata2 import dumpvtk as dumpvtk
#from heArt_py3.src.postprocessing.postprocessdata2 import (
#    compute_strain as compute_strain,
#)
#from heArt_py3.src.postprocessing.postprocessdata2 import (
#    plothemodynamics as plothemodynamics,
#)
#from heArt_py3.src.postprocessing.postprocessdata2 import (
#    extractdisplacement as extractdisplacement,
#)
#from heArt_py3.src.postprocessing.postprocessdata2 import (
#    extractdisplacementloading as extractdisplacementloading,
#)

#  - - - - - - - - - - - -- - - - - - - - - - - - - - - -- - - - - - -
# ellipsoidal_baselinegeo
IODetails = {
    "casename": "fchmesh_scale_w_biv_valves_remarked",
    "directory_me": "../FCHMesh/vh/",
    "directory_ep": "../FCHMesh/vh/",
    "outputfolder": "./outputs_FCHMesh/",
    "folderName": "",
    "caseID": "FCHMesh_img",
    "isLV": False,
    "isFCH": True,
}

contRactility = 400e3

GuccioneParams = {
    "ParamsSpecified": True,
    "Passive model": {"Name": "Guccione"},
    "Passive params": {
        "Cparam": Constant(130.0),
        "bff": Constant(29.0),
        "bfx": Constant(13.3),
        "bxx": Constant(26.6),
        "mu_iso": Constant(5e2),
        "b_iso": Constant(10.0),
    },
    "Active model": {"Name": "Time-varying"},
    "Active params": {
        "tau": 25,
        "t_trans": 300,
        "B": 4.75,
        "t0": 275,
        "l0": 1.58,
        "Tmax": Constant(contRactility),
        "Ca0": 4.35,
        "Ca0max": 4.35,
        "lr": 1.85,
    },
    "HomogenousActivation": True,
    "deg": 4,
    "Kappa": 1e5,
    "incompressible": True,
}

# Circparam = {
#    "Ees_la": 10,  # End-systolic elastance (60) --> Pa/ml
#    "A_la": 2.67,  # Scaling factor for EDPVR --> ml
#    "B_la": 0.019,  # Exponent for EDPVR --> ml-1
#    "V0_la": 10,  # volume axis intercept --> ml
#    "Tmax_la": 120,  # time to end-systole --> ms
#    "tau_la": 25,  # time constant of relaxation --> ms
#    "tdelay_la": 160,  #
#    "Csa": 0.0032,  # Proximal aorta compliance --> ml Pa
#    "Cad": 0.0330,  # Distal aorta compliance --> ml Pa
#    "Csv": 0.28,  # Venous compliance -> ml Pa
#    "Vsa0": 360,  # Resting volume for proximal aorta --> ml
#    "Vsv0": 3370.0,  # Resting venous volume (pre 3370.0 (2950, 3100, 3370)) --> ml
#    "Vad0": 40,  # Resting volume for distal aorta --> ml
#    "Rav": 3000.0,  # (pre 500 (500)) (aortic valve resistance) --> Pa ms ml-1
#    "Rsv": 100.0,  # Venous resistance --> Pa ms ml-1
#    "Rsa": 18000,  # Proximal aorta resistance --> Pa ms ml-1
#    "Rad": 21200,  # Distal aorta resistance (info not available) --> Pa ms ml-1
#    "Rmv": 200.0,  # Mitral valve resistance --> Pa ms ml-1
#    "V_sv": 3700,
#    "V_LV": 112,
#    "V_sa": 740,
#    "V_ad": 100,
#    "V_LA": 12,
#    "V_LV": 112,
#    "stop_iter": 1,
# }


Circparam = {
    "Ees_la": 350,
    "A_la": 58.67,
    "B_la": 0.049,
    "V0_la": 20,
    "Tmax_la": 120,
    "tau_la": 75,
    "tdelay_la": 160,
    "Csa": 0.0052,
    "Cad": 0.0330,
    "Csv": 0.3,
    "Vsa0": 700,
    "Vsv0": 2200.0,
    "Vad0": 40,
    "Rav": 500.0,
    "Rsv": 100.0,
    "Rsa": 9000,
    "Rad": 53000,
    "Rmv": 1500.0,
    # Pulmonary
    "Ees_ra": 81.33,
    "A_ra": 466.6,
    "B_ra": 0.033,
    "V0_ra": 20,
    "Tmax_ra": 120,
    "tau_ra": 25,
    "tdelay_ra": 100,
    "Cpa": 0.0125,
    "Cpv": 0.9,
    "Vpa0": 360,
    "Vpv0": 400,
    "Rpv": 500.0,
    "Rtv": 400.0,
    "Rpa": 13000.0,
    "Rpvv": 400,
    # flow rate
    "Q_lvad": 0.0,
    "Q_sv": 0.0659387173129,
    "Q_av": 0.0,
    "Q_sa": 0.0113526347222,
    "Q_ad": 0.0797504135138,
    "Q_mv": 0.0,
    "Q_tv": 0.0,
    "Q_pa": 0.001213376269,
    "Q_pv": 0.0646105314407,
    "Q_pvv": 0.0,
    # volumes
    "V_sv" : 2588.061593085814,
    "V_LV" : 385.4392936741486,
    "V_sa" : 745.5442367107188,
    "V_ad" : 324.09275211537874,
    "V_LA" : 100.27407807159136,
    "V_pv" : 3250.1659341523064,
    "V_RV" : 388.6390992866472,
    "V_pa" : 401.04292933635935,
    "V_RA" : 53.330654485216826,
    "stop_iter": 1,
    # LVAD
    #'Q_lvad_rpm' : 28,
    #'Q_lvad_scale' : 0.0
}


SimDetails = {
    "diaplacementInfo_ref": False,
    "HeartBeatLength": 800.0,
    "dt": 1.0,
    "writeStep": 40.0,
    "GiccioneParams": GuccioneParams,
    "nLoadSteps": 25,
    "DTI_EP": False,
    "DTI_ME": False,
    "d_iso": 1.5 * 0.01,
    "d_ani_factor": 4.0,
    #    "probepts": [
    #        [3.54982, 4.85747, -1.56241],
    #        [3.54982, 4.85747, -1.56241],
    #        [3.54982, 4.85747, -1.56241],
    #        [3.54982, 4.85747, -1.56241],
    #        [4.10888, 5.28499, -1.56241],
    #        [4.77476, 5.69628, -1.56241],
    #        [10.1261, 9.83341, -1.56241],
    #        [10.3596, 10.0373, -1.56241],
    #        [10.5715, 10.2127, -1.56241],
    #    ],
    "ploc": [[1.4, 1.4, -3.0, 2.0, 1]],  # , [-1.4, -1.4, -3.0, 2.0, 2]],
    "pacing_timing": [[4.0, 20.0]],  # , [20.0, 20.0]],
    "Isclosed": True,
    "closedloopparam": Circparam,
    "Ischemia": False,
    "Mechanics Discretization": "P1P1",
    "Technique Discretization": 1,
    "isLV": False,
    "aorta_wall": 11,  # aorta ring
    "pulm_wall": 12,  # pulmonary ring
    "LVendoid": 18,
    "RVendoid": 15,
    "epiid": 19,
    # "apxid": 20,
    # "septumid": 13,
    "mitral_vplane": 16,
    "aortic_vplane": 17,
    "first_rv_valve": 13,
    "second_rv_valve": 14,
    "abs_tol": 1e-8,
    "rel_tol": 1e-9,
    "isunloading": False,
    "isunloadingonly": False,
    "ispctrl": True,
    "isFCH": True,
    "springbc": 1,
    # "springparam": [2.0e3, 2.0e3],  # Kepi_n / Kepi_t
    # "dashpotparam": [2.0e2, 2.0e2],  # Cepi_n / Cepi_t
    "active_region": [1, 2],  # only lv is activated
    "Type": 0,
}

# Run Simulation
run_BiV_ClosedLoop(IODet=IODetails, SimDet=SimDetails)
# Postprocessing
# postprocessdata(IODet=IODetails, SimDet=SimDetails)
# dumpvtk(IODet=IODetails, SimDet=SimDetails)
# compute_strain(IODet=IODetails, SimDet=SimDetails, LVid = 1)
# plothemodynamics(IODet=IODetails, SimDet=SimDetails, cycle=None)
# extractdisplacementloading(IODet=IODetails, SimDet=SimDetails)
# extractdisplacement(IODet=IODetails, SimDet=SimDetails)
#  - - - - - - - - - - - -- - - - - - - - - - - - - - - -- - - - - - -
