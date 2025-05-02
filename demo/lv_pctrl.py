import sys, pdb
from dolfin import *

sys.path.append("/mnt/Research/heArt")
sys.path.append("/mnt/Research")
sys.path.append("/mnt/Research/heArt_py3_original")
sys.path.append("/mnt/Output")

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
    #"casename_me": "ellipsoidal_baselinegeo_coarse",
    "casename_me": "ellipsoidal_baselinegeo_medium3",
    #"casename_ep": "ellipsoidal_baselinegeo_coarse",
    "casename_ep": "ellipsoidal_baselinegeo_medium3",
    "directory_me": "/mnt/home/lclee/heArt/heArt_py3/LVMesh/lc/",
    "directory_ep": "/mnt/home/lclee/heArt/heArt_py3/LVMesh/lc/",
    "outputfolder": "/mnt/scratch/lclee/output_heArt_py3/outputs_BiVelectromechanics/",
    #"outputfolder": "./Outputs/",
    "folderName": "",
    #"caseID": "LVelectromechanics_ncircp_P1P1_test_K10_10_mesh3_cap_contract180",
    "caseID": "LVelectromechanics_ncircp_P1P1_test_K30_10_mesh3_cap_contract220",
    "isLV": True,
}

contRactility = 220e3#150e3#400e3

GuccioneParams = {
    "ParamsSpecified": True,
    "Passive model": {"Name": "Guccione"},
    "Passive params": {
        "Cparam": Constant(50.0),#Constant(130.0),
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
    "Csa": 0.0052,
    "Cad": 0.013,
    "Csv": 0.28,
    "Vsa0": 360,
    "Vad0": 40,
    "Vsv0": 3370.0,
    "Rav": 2000,
    "Rsv": 100.0,
    "Rsa": 58000,
    "Rad": 106000,
    "Rmv": 2000,
    "V_sa": 407.9870929796549, #4.09767e2,
    "V_ad": 139.88354730982294,#1.44290e2,
    "V_sv": 3800.6771443568937,#3.80285e3,
    "V_LA": 193.99555092431984,#1.94894e2,
    "V_LV": 98.40525741977021, #8.39793e1,
    "stop_iter": 5,
    "issoftplus": False,
}

SimDetails = {
   # "poro": False,
    "diaplacementInfo_ref": False,
    "HeartBeatLength": 800.0,
    "dt": 0.5, 
    "EDP": 12.342740722563716,#1.24326e1,
    "writeStep": 5,
    "GiccioneParams": GuccioneParams,
    "nLoadSteps": 50,
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
    "springbc": True,
    #"springref": ["/mnt/Research/heArt/heArt_py3/demo/Data_test.h5", "ME/u_loading"],
    "Mechanics Discretization": "P1P1",#"P2P1",
    "isLV": True,
    "spring_atbase": True,
    "topid": 4,
    "LVendoid": 2,
    "RVendoid": 0,
    "epiid": 1,
    "abs_tol": 1e-8,
    "rel_tol": 1e-9,
    "isunloading": False,
    "isunloadingonly": False,
    "ispctrl": True,
    "epiid_Kadj_coeff": [30,10], #[10,5],#[40, 2],
    "active_region": [0],
    "annulus_region": [1],
    "annulus_stiffness_factor": 100,
    "dashpotparam": [10.0e1,2.0e1],
    "permeability": 1.0e-9,
    "p_a": 0.0,
    "p_v": 1300.0,
    "beta_a": 3.5e-5,
    "beta_v": 3.0e-5,
}

# Run Simulation
#run_BiV_ClosedLoop(IODet=IODetails, SimDet=SimDetails)
#  - - - - - - - - - - - -- - - - - - - - - - - - - - - -- - - - - - -
# Postprocessing
#dumpvtk(IODet=IODetails, SimDet=SimDetails, ME_var=[["fstress", "DG", 0]], EP_var=[["phi", "CG", 1]], PJ_var=[["phi", "CG", 1]])
# postprocessdata(IODet=IODetails, SimDet=SimDetails)
compute_strain(IODet=IODetails, SimDet=SimDetails, LVid = 0)
extractdisplacement(IODet=IODetails, SimDet=SimDetails)
plothemodynamics(IODet=IODetails, SimDet=SimDetails)
#extractdisplacementloading(IODet=IODetails, SimDet=SimDetails)
