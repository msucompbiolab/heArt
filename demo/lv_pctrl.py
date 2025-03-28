import sys, pdb
from dolfin import *

sys.path.append("/mnt/Research")
#sys.path.append("/mnt/Research/heArt_py3_original")
sys.path.append("/mnt/Output")
sys.path.append("/mnt/Research/heArt")

from heArt_py3.src.sim_protocols.run_BiV_ClosedLoop_pctrl_test import (
    run_BiV_ClosedLoop as run_BiV_ClosedLoop,
)
from heArt_py3.src.postprocessing.postprocessdata2 import (
    postprocessdata as postprocessdata,
)
from heArt_py3.src.postprocessing.postprocessdata2 import dumpvtk as dumpvtk
from heArt_py3.src.postprocessing.postprocessdata2 import (
    compute_strain as compute_strain,
)
from heArt_py3.src.postprocessing.postprocessdata2 import (
    plothemodynamics as plothemodynamics,
)
from heArt_py3.src.postprocessing.postprocessdata2 import (
    extractdisplacementloading as extractdisplacementloading,
)
from heArt_py3.src.postprocessing.postprocessdata2 import (
    extractdisplacement as extractdisplacement,
)


#  - - - - - - - - - - - -- - - - - - - - - - - - - - - -- - - - - - -
# ellipsoidal_baselinegeo
IODetails = {
    "casename_me": "ellipsoidal_baselinegeo",
    "casename_ep": "ellipsoidal_baselinegeo",
    "directory_me": "../LVMesh/vh/",
    "directory_ep": "../LVMesh/vh/",
    #"outputfolder": "/mnt/Output/outputs_LVelectromechanics/",
    "outputfolder": "./Outputs/",
    "folderName": "",
    "caseID": "LVelectromechanics_ncircp_P1P1",
    "isLV": True,
}

contRactility = 600e3

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
    "HomogenousActivation": True,
    "deg": 4,
    "Kappa": 1e6,
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
    "Rav": 3000,#5000.0,#2000,#500,
    "Rsv": 100.0,
    "Rsa": 18000,
    "Rad": 106000,
    "Rmv": 3000,#5000.0,#2000.0,#200.0,
    "V_sv": 3709.681538561804,
    "V_sa": 386.4525256055264,
    "V_ad": 309.22012729232915,
    "V_LA": 157.01981722400419,
    "V_LV": 101.62599131634467,
    "stop_iter": 1,
    "issoftplus": False,
}

SimDetails = {
   # "poro": False,
    "diaplacementInfo_ref": False,
    "HeartBeatLength": 800.0,
    "dt": 0.5,
    "writeStep": 5.0,
    "GiccioneParams": GuccioneParams,
    "nLoadSteps": 15,
    "DTI_EP": False,
    "DTI_ME": False,
    "d_iso": 1.5 * 0.005,
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
    "springbc": True,#0,
    "Mechanics Discretization": "P1P1",#"P2P1",
    # "Technique Discretization": 1,
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
}

# Run Simulation
run_BiV_ClosedLoop(IODet=IODetails, SimDet=SimDetails)
#  - - - - - - - - - - - -- - - - - - - - - - - - - - - -- - - - - - -
# Postprocessing
#dumpvtk(IODet=IODetails, SimDet=SimDetails, ME_var=[["u", "CG", 1]], EP_var=[], PJ_var=[])
#dumpvtk(IODet=IODetails, SimDet=SimDetails, ME_var=[["fstress", "DG", 0]], EP_var=[["phi", "CG", 1]], PJ_var=[["phi", "CG", 1]])
# postprocessdata(IODet=IODetails, SimDet=SimDetails)
# extractdisplacement(IODet=IODetails, SimDet=SimDetails)
# compute_strain(IODet=IODetails, SimDet=SimDetails, LVid = 0)
# plothemodynamics(IODet=IODetails, SimDet=SimDetails, cycle=5)
# extractdisplacementloading(IODet=IODetails, SimDet=SimDetails)
