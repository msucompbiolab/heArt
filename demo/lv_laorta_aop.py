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
    "casename": "8159_baseline_newmarked",
    "directory_me": "../LV_waorta/vh/",
    "directory_ep": "../LV_waorta/vh/",
    "outputfolder": "/mnt/Output/lv_waorta/",
    "folderName": "",
    "caseID": "8159_cladj2",
    "isLV": False,
}

contRactility = 800e3

GuccioneParams = {
    "ParamsSpecified": True,
    "Passive model": {"Name": "Guccione"},
    "Passive params": {
        "Cparam": Constant(130.0),
        "bff": Constant(29.0),
        "bfx": Constant(13.3),
        "bxx": Constant(26.6),
        "mu_iso": Constant(5e2),
        "b_iso": Constant(26.0),
        "aorta_comp": Constant(10.0),
        "base_comp": Constant(0.5),
    },
    "Aorta params": {
        "Name": "other",
        # Neo-Hookean
        "mu": Constant(63.80),
        # Delfino
        "D1": Constant(33.04),
        "D2": Constant(5.05),
        # HGO_twofiber # age: 71-78
        "Cgr": Constant(41.69),
        "gamma": Constant(56.18),
        # "gamma": Constant(45.0),
        "C1": [1.20, 1.20],
        "C2": [2.56, 2.56],
        # HGO_fourfiber # age: 71-78
        "Cgr_ff": Constant(12.67),
        "gamma_ff": Constant(39.55),
        "C1_ff": [6.87, 6.87, 25.63, 13.88],
        "C2_ff": [14.86, 14.86, 1.19, 11.86],
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

Circparam = {
    "Ees_la": 30, #x 10 --> 120
    "A_la": 2.67, #x 2.67 --> 60.0
    "B_la": 0.019, #x 0.019 --> 0.03
    "V0_la": 8,
    "Tmax_la": 120,
    "tau_la": 35, #x 75 --> 25
    "tdelay_la": 160, #x 220 --> 160
    "Csa": 0.0045,
    "Cad": 0.033,
    "Csv": 0.28,
    "Vsa0": 550,
    "Vsv0": 2150.0,
    "Vad0": 40,
    "Rav": 1000.0,
    "Rsv": 100.0,
    "Rsa": 12000,
    "Rad": 31800,
    "Rmv": 400.0, #x too much oscilation w 200
    # volumes
    "V_sv": 2600,
    #"V_LV": 105,
    "V_sa": 740,
    "V_ad": 40,
    "V_LA": 35,
    "stop_iter": 2,
}


SimDetails = {
    "diaplacementInfo_ref": False,
    "HeartBeatLength": 550.0,
    "dt": 0.5,
    "writeStep": 35.0,
    "GiccioneParams": GuccioneParams,
    "nLoadSteps": 64,
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
    "aorta_ext_wall": 2,
    "aorta_int_wall": 3,
    "aorta_ring": 1,
    "LVendoid": 6,
    "RVendoid": 0,
    "epiid": 8,
    "apxid": 4,
    "mitral_valvep": 7,
    "aortic_valvep": 5,
    "aortaid": 3,
    "abs_tol": 1e-8,
    "rel_tol": 1e-9,
    "isunloading": False,
    "isunloadingonly": False,
    "ispctrl": True,
    "iswaorta": True,
    "springbc": 1,
    "mv_aorta": 0,
    "aorta_pres": 0,
    "EDP": 24,
    "springparam": [5.0e3, 5.0e3],  # paper's values Kepi_n = 2e3 / Kepi_t = 2e2
    "dashpotparam": [5.0e2, 5.0e1],  # paper's values Cepi_n = 2e2 / Cepi_t = 2e1
    # "springaortaparam": [5.0e1, 5.0e1],  # Kepi_n / Kepi_t
    # "dashpotaortaparam": [5.0e1, 5.0e0],  # Cepi_n / Cepi_t
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
# plothemodynamics(IODet=IODetails, SimDet=SimDetails, cycle=3)
# extractdisplacementloading(IODet=IODetails, SimDet=SimDetails)
# extractdisplacement(IODet=IODetails, SimDet=SimDetails)


#  - - - - - - - - - - - -- - - - - - - - - - - - - - - -- - - - - - -
