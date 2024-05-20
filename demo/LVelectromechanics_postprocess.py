import sys, pdb
from dolfin import *

sys.path.append("/mnt/home/ziaeirad")
from heArt.src.sim_protocols.run_BiV_ClosedLoop import (
    run_BiV_ClosedLoop as run_BiV_ClosedLoop,
)
from heArt.src.postprocessing.postprocessdata2 import postprocessdata as postprocessdata

#  - - - - - - - - - - - -- - - - - - - - - - - - - - - -- - - - - - -
IODetails = {
    "casename": "ellipsoidal_baselinegeo",
    "directory_me": "../LVMesh/",
    "directory_ep": "../LVMesh/",
    "outputfolder": "./outputs_LVelectromechanics/",
    "folderName": "",
    "caseID": "LVelectromechanics",
    "isLV": True,
}

contRactility = 130e3

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
    "HomogenousActivation": False,
    "deg": 4,
    "Kappa": 1e5,
    "incompressible": True,
}

Circparam = {
    "Ees_la": 10,
    "A_la": 2.67,
    "B_la": 0.019,
    "V0_la": 10,
    "Tmax_la": 120,
    "tau_la": 25,
    "tdelay_la": 160,
    "Csa": 0.0032,
    "Cad": 0.033,
    "Csv": 0.28,
    "Vsa0": 360,
    "Vad0": 40,
    "Vsv0": 3370.0,
    "Rav": 500,
    "Rsv": 100.0,
    "Rsa": 18000,
    "Rad": 106000,
    "Rmv": 200.0,
    "V_sv": 3700,
    "V_sa": 740,
    "V_ad": 100,
    "V_LA": 12,
    "V_LV": 112,
    "stop_iter": 1,
}

SimDetails = {
    "diaplacementInfo_ref": False,
    "HeartBeatLength": 800.0,
    "dt": 1.0,
    "writeStep": 2.0,
    "GiccioneParams": GuccioneParams,
    "nLoadSteps": 15,
    "DTI_EP": False,
    "DTI_ME": False,
    "d_iso": 1.5 * 0.005,
    "d_ani_factor": 4.0,
    "ploc": [[1.4, 1.4, -3.0, 2.0, 1]],  # , [-1.4, -1.4, -3.0, 2.0, 2]],
    "pacing_timing": [[4.0, 20.0]],  # , [20.0, 20.0]],
    "Isclosed": True,
    "closedloopparam": Circparam,
    "Ischemia": False,
    "isLV": True,
    "topid": 4,
    "LVendoid": 2,
    "RVendoid": 0,
    "epiid": 1,
    "abs_tol": 1e-9,
    "rel_tol": 5e-7,
    "isunloading": False,
}

# Run Simulation
# run_BiV_ClosedLoop(IODet=IODetails, SimDet=SimDetails)
# Postprocessing
postprocessdata(
    IODet=IODetails, SimDet=SimDetails
)  # vahid: will cut-off this into a separate file
#  - - - - - - - - - - - -- - - - - - - - - - - - - - - -- - - - - - -
