import sys, pdb
from dolfin import *

sys.path.append("/mnt/Research")
sys.path.append("/mnt/Output")

from heArt_py3.src.sim_protocols.run_BiV_ClosedLoop_pctrl import (
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
    # "casename": "8159_baseline_ES_t11",
    "casename": "8159_twofiber",
    "directory_me": "../LV_waorta/vh/",
    "directory_ep": "../LV_waorta/vh/",
    "outputfolder": "/mnt/Output/outputs_LV_waorta/",
    "folderName": "",
    "caseID": "8159_baseline_Delfino",
    "isLV": False,
}

contRactility = 700e3

GuccioneParams = {
    "ParamsSpecified": True,
    "Passive model": {"Name": "Guccione"},
    "Passive params": {
        "Cparam": Constant(130.0),
        "bff": Constant(29.0),
        "bfx": Constant(13.3),
        "bxx": Constant(26.6),
        "mu_iso": Constant(5e2),
        "b_iso": Constant(25.0),
        "aorta_comp_red": Constant(5.0),
    },
    "Aorta params": {
        "Name": "Delfino",
        # Neo-Hookean
        "mu": Constant(63.80),
        # Delfino
        "D1": Constant(33.04e1),
        "D2": Constant(5.05),
        # HGO two-fiber # age: 71-78
        "Cgr": Constant(51.68),
        "gamma": Constant(29.24),
        # "gamma": Constant(45.0),
        "C1": [0.51, 0.51],
        "C2": [27.99, 27.99],
        # HGo four-fiber # age: 71-78
        "Cgr_ff": Constant(12.67e3),
        "gamma_ff": Constant(39.55),
        "C1_ff": [6.87e3, 6.87e3, 25.63e3, 13.68e3],
        "C2_ff": [14.86, 14.86, 1.19, 11.86],
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
    "Rad": 30000,  # prev: 25000
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
    "ploc": [[1.4, 1.4, -3.0, 2.0, 1]],  # , [-1.4, -1.4, -3.0, 2.0, 2]],
    "pacing_timing": [[4.0, 20.0]],  # , [20.0, 20.0]],
    "Isclosed": True,
    "closedloopparam": Circparam,
    "Ischemia": False,
    "Mechanics Discretization": "P1P1",
    "Technique Discretization": 1,
    "isLV": False,
    "aorta_ext_wall": 3,
    "aorta_int_wall": 2,
    "aorta_ring": 1,
    "LVendoid": 8,
    "RVendoid": 0,
    "epiid": 5,
    "apxid": 9,
    "mitral_vplane": 7,
    "aortic_vplane": 6,
    "abs_tol": 1e-8,
    "rel_tol": 1e-9,
    "isunloading": False,
    "isunloadingonly": False,
    "ispctrl": True,
    "iswaorta": True,
    "springbc": 1,
    "mv_aorta": 0,
    "springparam": [2.0e3, 2.0e3],  # paper's values Kepi_n = 2e3 / Kepi_t = 2e2
    "dashpotparam": [2.0e2, 2.0e1],  # paper's values Cepi_n = 2e2 / Cepi_t = 2e1
    "springaortaparam": [5.0e1, 5.0e1],  # Kepi_n / Kepi_t
    "dashpotaortaparam": [5.0e1, 5.0e0],  # Cepi_n / Cepi_t
    "active_region": [0],
    "rubber_region": [3, 4, 5],
    "aorta_region": [2],
    "Type": 0,
}

# Run Simulation
run_BiV_ClosedLoop(IODet=IODetails, SimDet=SimDetails)
# Postprocessing
# dumpvtk(IODet=IODetails, SimDet=SimDetails)
# compute_strain(IODet=IODetails, SimDet=SimDetails, LVid = 0)
# plothemodynamics(IODet=IODetails, SimDet=SimDetails, cycle=5)
# extractdisplacementloading(IODet=IODetails, SimDet=SimDetails)
# extractdisplacement(IODet=IODetails, SimDet=SimDetails)


#  - - - - - - - - - - - -- - - - - - - - - - - - - - - -- - - - - - -
