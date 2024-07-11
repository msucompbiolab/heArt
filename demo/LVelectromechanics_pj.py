import sys, pdb, vtk
import dolfin
from dolfin import *
from matplotlib import pylab as plt
import numpy as np

# sys.path.append("/home/hagersan/Desktop/github")
# sys.path.append('/home/hagersan/github')
# sys.path.append('/home/hagersan/Desktop/MSU/PK+iLVEP')
sys.path.append("/home/hagersan/Desktop/MSU-src")
sys.path.append("/mnt/Research")
import vtk_py3

# from purkinjee_fhn import *
from heArt.src.sim_protocols.run_BiV_ClosedLoop_pj_pointact import (
    run_BiV_ClosedLoop as run_BiV_ClosedLoop,
)
from heArt.src.postprocessing.postprocessdata2 import postprocessdata as postprocessdata

#  - - - - - - - - - - - -- - - - - - - - - - - - - - - -- - - - - - -
IODetails = {
    "casename": "ellipsoidal_baselinegeo",
    "directory_me": "../LV_Purkinje_mesh/",
    "directory_ep": "../LV_Purkinje_mesh/",
    "casename_pj": "PJ",
    "directory_pj": "../LV_Purkinje_mesh/",
    "outputfolder": "./outputs_LVelectromechanics_pj",
    "folderName": "/",
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
    "stop_iter": 2,
}

SimDetails = {
    "diaplacementInfo_ref": False,
    "HeartBeatLength": 800.0,
    "dt": 2.0,
    "writeStep": 1.0,
    "GiccioneParams": GuccioneParams,
    "nLoadSteps": 15,
    "DTI_EP": False,
    "DTI_ME": False,
    "d_pj": 10.0,
    "d_iso": 1.5,
    "d_ani_factor": 4.0,
    "ploc_tol": 0.5,
    "Isclosed": True,
    "closedloopparam": Circparam,
    "Ischemia": False,
    "Mechanics Discretization": "P1P1",
    "isLV": True,
    "topid": 4,
    "LVendoid": 2,
    "RVendoid": 0,
    "epiid": 1,
    "abs_tol": 1e-9,
    "rel_tol": 5e-7,
    "isunloading": False,
    "tnode": "PJ.csv",
    "d_iso_pj": 4.0,
    "ploc": [[0.763396, -1.843, -0.554806]],
    "pacing_timing": [[0.0, 10.0]],

}

# Run Simulation
run_BiV_ClosedLoop(IODet=IODetails, SimDet=SimDetails)
# Postprocessing
# postprocessdata(IODet=IODetails, SimDet=SimDetails)
#  - - - - - - - - - - - -- - - - - - - - - - - - - - - -- - - - - - -
