import sys, pdb
from dolfin import *

sys.path.append("/mnt/Research")
sys.path.append("/mnt/Output")

from heArt_py3.src.sim_protocols.run_BiV_ClosedLoop_lumped import (
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
    "casename": "ellipsoidal_baselinegeo",
    "directory_me": "../LVMesh/vh/",
    "directory_ep": "../LVMesh/vh/",
    "outputfolder": "/mnt/Output/outputs_LVelectromechanics/",
    "folderName": "",
    "caseID": "LVelectromechanics_lumped",
    "isLV": True,
}

contRactility = 800e3

GuccioneParams = {
    "ParamsSpecified": True,
    "Passive model": {"Name": "Guccione"},
    "Passive params": {
        "Cparam": Constant(200.0),
        "bff": Constant(29.0),
        "bfx": Constant(13.3),
        "bxx": Constant(26.6),
    },
    "Aorta params": {
        "Name": "Delfino",
        # Neo-Hookean
        "mu": Constant(47.55e3),
        # Delfino
        "D1": Constant(33.04e3),
        "D2": Constant(5.05),
        # HGO two-fiber
        "Cgr": Constant(49.91),
        "gamma": Constant(24.65),
        "C1": [0.22, 0.22],
        "C2": [16.17, 16.17],
        # HGo four-fiber
    },
    "Active model": {"Name": "Time-varying"},
    "Active params": {
        "tau": 25,
        "t_trans": 300,
        "B": 4.75,
        "t0": 900,
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
    "Ees_lv": 1600,
    "V0_lv": 30,
    "A_lv": 135,
    "B_lv": 0.027,
    "Tmax_lv": 280,
    "tau_lv": 25,
    "tdelay_lv": 325,
    "Ees_la": 120,
    "A_la": 60.0,
    "B_la": 0.03,
    "V0_la": 10,
    "Tmax_la": 150,
    "tau_la": 30,
    "tdelay_la": 225,
    "Csa": 0.0035,
    "Cad": 0.04,
    "Csv": 0.5,
    "Vsa0": 320,
    "Vsv0": 3370.0,
    "Vad0": 40,
    "Rav": 5000.0,
    "Rsv": 100.0,
    "Rsa": 18000,
    "Rad": 25000,
    "Rmv": 250.0,
    # volumes
    "V_sv": 3600,
    "V_LV": 105,
    "V_sa": 1750,
    "V_ad": 37,
    "V_LA": 35,
    "stop_iter": 4,
}

SimDetails = {
    "diaplacementInfo_ref": False,
    "HeartBeatLength": 700.0,
    "dt": 0.5,
    "writeStep": 40.0,
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
    "springbc": 1,
    "Mechanics Discretization": "P1P1",
    "isLV": True,
    "topid": 4,
    "LVendoid": 2,
    "RVendoid": 0,
    "epiid": 1,
    "apxid": 5,
    "abs_tol": 1e-8,
    "rel_tol": 1e-9,
    "isunloading": False,
    "isunloadingonly": False,
    "ispctrl": True,
    "islumped": True,
    "epiid_Kadj_coeff": [50, 10],
    "springparam": [2.0e3, 2.0e2],  # Kepi_n / Kepi_t
    "dashpotparam": [2.0e2, 2.0e1],  # Cepi_n / Cepi_t
    # "spring_atbase": 0,
}

# Run Simulation
# run_BiV_ClosedLoop(IODet=IODetails, SimDet=SimDetails)
#  - - - - - - - - - - - -- - - - - - - - - - - - - - - -- - - - - - -
# Postprocessing
# postprocessdata(IODet=IODetails, SimDet=SimDetails)
# extractdisplacement(IODet=IODetails, SimDet=SimDetails)
# compute_strain(IODet=IODetails, SimDet=SimDetails, LVid = 0)
plothemodynamics(IODet=IODetails, SimDet=SimDetails, cycle=5)
# extractdisplacementloading(IODet=IODetails, SimDet=SimDetails)
