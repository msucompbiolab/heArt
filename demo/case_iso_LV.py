import sys, pdb
from dolfin import *

sys.path.append("/home/ziaeirad")
from heArt.src.sim_protocols.run_BiV_Isovolumic import (
    run_BiV_Isovolumic as run_BiV_Isovolumic,
)
#from heArt.src.postprocessing.postprocessdataBiV2 import (
#    postprocessdata as postprocessdata,
#)

#  - - - - - - - - - - - -- - - - - - - - - - - - - - - -- - - - - - -
IODetails = {
    "casename": "ellipsoidal",
    "directory_me": "../LVMesh/",
    "directory_ep": "../LVMesh/",
    "outputfolder": "./outputs/",
    "folderName": "",
    "caseID": "sixc_p2",
}

contRactility = 22000.0

GuccioneParams = {
    "ParamsSpecified": True,
    "Passive model": {"Name": "Guccione"},
    "Passive params": {
        "Cparam": Constant(100.0),
        "bff": Constant(29.0),
        "bfx": Constant(13.3),
        "bxx": Constant(26.6),
    },
    "Active model": {"Name": "Time-varying"},
    # "Active params": {"tau" : 25*250.0/800.0, "t_trans" : 300*250.0/800.0, "B" : 4.75,  " #lclee version
    "Active params": {
        "tau": 15.0,
        "t_trans": 100,
        "B": 4.75,
        "t0": 80,
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

Circparam = {
    "V_LV": 0.26,
    "V_RV": 0.26,
    "stop_iter": 0,
}

SimDetails = {
    "HeartBeatLength": 250.0,
    "dt": 1.0,
    "writeStep": 10.0,
    "GiccioneParams": GuccioneParams,
    "nLoadSteps": 10,
    "DTI_EP": False,
    "DTI_ME": False,
    "d_iso": 1.5 * 0.01,
    "d_ani_factor": 4.0,
    "ploc": [[-0.083, 5.6, -1.16, 2.0, 1.0]],
    "probepts": [
        [3.54982, 4.85747, -1.56241],
        [3.54982, 4.85747, -1.56241],
        [3.54982, 4.85747, -1.56241],
        [3.54982, 4.85747, -1.56241],
        [4.10888, 5.28499, -1.56241],
        [4.77476, 5.69628, -1.56241],
        [10.1261, 9.83341, -1.56241],
        [10.3596, 10.0373, -1.56241],
        [10.5715, 10.2127, -1.56241],
    ],  # LVepi mid endo Rv endo mid epi based on imp1   thick RV wall
    "probepts_imp": "../BiVMesh/biv_idealized.npz",
    "pacing_timing": [[4.0, 20.0]],
    "closedloopparam": Circparam,
    "Ischemia": False,
    "isLV": True,  # from BiV to LV --> False to True
    "topid": 4,
    "LVendoid": 2,
    "RVendoid": 3,
    "epiid": 1,
    "abs_tol": 1e-9,
    "rel_tol": 1e-9,
}


# Run Simulation
run_BiV_Isovolumic(IODet=IODetails, SimDet=SimDetails)

#  - - - - - - - - - - - -- - - - - - - - - - - - - - - -- - - - - - -
