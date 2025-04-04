import sys, pdb, vtk
import dolfin
from dolfin import *
#from matplotlib import pylab as plt
import numpy as np

sys.setrecursionlimit(5000)  # Increase to a higher value
sys.path.append("/mnt/Research/heArt")
sys.path.append("/mnt/Research")
#sys.path.append("/mnt/Research/heArt_py3_original")
sys.path.append("/mnt/Output")

import vtk_py3

from heArt_py3.src.sim_protocols.run_BiV_ClosedLoop_pctrl_test import (
    run_BiV_ClosedLoop as run_BiV_ClosedLoop,
)
#from heArt_py3.src.sim_protocols.run_BiV_ClosedLoop_pj_pointact_test_time import (
#    run_BiV_ClosedLoop as run_BiV_ClosedLoop,
#)
from heArt_py3.src.postprocessing.postprocessdata2 import postprocessdata as postprocessdata
from heArt_py3.src.postprocessing.postprocessdata2 import dumpvtk as dumpvtk
from heArt_py3.src.postprocessing.postprocessdata2 import compute_strain as compute_strain
from heArt_py3.src.postprocessing.postprocessdata2 import compute_activation as compute_activation
from heArt_py3.src.postprocessing.postprocessdata2 import plothemodynamics as plothemodynamics
from heArt_py3.src.postprocessing.postprocessdata2 import plotpressure as plotpressure

#  - - - - - - - - - - - -- - - - - - - - - - - - - - - -- - - - - - -
IODetails = {
    "casename_me": "ellipsoidal_baselinegeo_coarse",
    "casename_ep": "ellipsoidal_baselinegeo_medium2",
    #"casename": "ellipsoidal_baselinegeo",
    #"directory_me": "../../../heArt_py3_purkinje/LV_pj_lc/",
    "directory_ep": "../LVMesh/lc/",
    "directory_me": "../LVMesh/lc/",
    #"directory_ep": "../LVMesh/vh/",
    "directory_pj": "../PJmesh/",
    "casename_pj": "PJmarked",
    "outputfolder": "./Outputs/",
    "folderName": "/",
    "caseID": "LVelectromechanics-PJ-test-Tmax1800",
    #"caseID": "LVelectromechanics-test-lbbb",
    "isLV": True,
}

contRactility = 1200e3

GuccioneParams = {
    "ParamsSpecified": True,
    "Passive model": {"Name": "Guccione"},
    "Passive params": {
        "Cparam": Constant(130.0),
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
    "Rav": 2000,#500,
    "Rsv": 100.0,
    "Rsa": 18000,
    "Rad": 106000,
    "Rmv": 2000.0,#200.0,
    "V_sv": 3709.681538561804,
    "V_sa": 386.4525256055264,
    "V_ad": 309.22012729232915,
    "V_LA": 157.01981722400419,
    "V_LV": 101.62599131634467,
    "stop_iter": 3,
}
#Circparam = {    
#    "Ees_la": 120,
#    "A_la": 60.0,
#    "B_la": 0.03,
#    "V0_la": 10,
#    "Tmax_la": 150,
#    "tau_la": 30,
#    "tdelay_la": 225,
#    "Csa": 0.0035,
#    "Cad": 0.04,
#    "Csv": 0.5,
#    "Vsa0": 320,
#    "Vsv0": 3370.0,
#    "Vad0": 40,
#    "Rav": 5000.0,
#    "Rsv": 100.0,
#    "Rsa": 18000,
#    "Rad": 25000,
#    "Rmv": 250.0,
#    # volumes
#    "V_sv": 3600,
#    "V_LV": 105,
#    "V_sa": 1750,
#    "V_ad": 37,
#    "V_LA": 35,
#    "stop_iter": 1,
#
#}

SimDetails = {
    "diaplacementInfo_ref": False,
    "HeartBeatLength": 800.0,
    "dt": 0.5,
    "writeStep": 5.0,
    "GiccioneParams": GuccioneParams,
    "nLoadSteps": 15,
    "DTI_EP": False,
    "DTI_ME": False,
    "d_pj": 5e1,#5e-2,#5e1,
    "d_iso": 0.02,#*5e-2,
    "d_ani_factor": 0.4,#*5e-2,
    "ani_factor": 1000.0,
    "ploc_tol": 0.07,
    "Isclosed": True,
    "closedloopparam": Circparam,
    "Ischemia": False,
    "springbc": True,#0,
    "Mechanics Discretization": "P1P1",#"P1P1",
    "isPJ": True,
    "isLV": True,
    "topid": 4,
    "LVendoid": 2,
    "RVendoid": 0,
    "epiid": 1,
    "abs_tol": 1e-8,
    "rel_tol": 1e-9,
    "isunloading": False,
    "isunloadingonly": False,
    "ispctrl": True,
    "epiid_Kadj_coeff": [50, 10],
    # "springparam": [2.0e3, 2.0e2],  # Kepi_n / Kepi_t
    # "dashpotparam": [2.0e2, 2.0e1],  # Cepi_n / Cepi_t
    # "spring_atbase": 0,
    "permeability": 1.0e-9,
    "p_a": 0.0,
    "p_v": 1300.0,
    "beta_a": 3.5e-5,
    "beta_v": 3.0e-5,
    "pj_tnodes": "PJ.csv",
    # "d_iso_pj": 150.0,
    "ploc_mode": False,
    "ploc": [[-0.574335, -1.8842, -0.168375]],
    "current_intensity": 10,#50,
    "PJ_current_intensity": 15,#5,#15,
    "pacing_timing": [[0.0, 10.0]],
    "lbbb": False,#True,
    "lbbb_delay": 1e-9,
    "lbbb_location" : 8,#[7,8], #node location in pj network
}

# Run Simulation
run_BiV_ClosedLoop(IODet=IODetails, SimDet=SimDetails)

# Postprocessing
#postprocessdata(IODet=IODetails, SimDet=SimDetails)

# Extract only Displacement
#dumpvtk(IODet=IODetails, SimDet=SimDetails, ME_var=[["u", "CG", 1]], EP_var=[], PJ_var=[])
#dumpvtk(IODet=IODetails, SimDet=SimDetails, ME_var=[["fstress", "DG", 0], ["potential_ref", "DG", 0]], EP_var=[["phi", "CG", 1]], PJ_var=[["phi", "CG", 1]])
#compute_activation(IODet=IODetails, SimDet=SimDetails)
#compute_strain(IODet=IODetails, SimDet=SimDetails, LVid = 0)
# plothemodynamics(IODet=IODetails, SimDet=SimDetails)
# plotpressure(IODet=IODetails, SimDet=SimDetails)
# dumpvtk(IODet=IODetails, SimDet=SimDetails)
# compute_activation(IODet=IODetails, SimDet=SimDetails)



#  - - - - - - - - - - - -- - - - - - - - - - - - - - - -- - - - - - -
