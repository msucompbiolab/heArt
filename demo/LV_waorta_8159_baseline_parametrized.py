import sys, pdb
from dolfin import *

sys.path.append("/mnt/Research")

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
    "casename": "8159_baseline_ES_t11_basethick",
    "directory_me": "../LV_waorta/vh/",
    "directory_ep": "../LV_waorta/vh/",
    "outputfolder": "./outputs_LV_waorta/",
    "folderName": "",
    # "caseID": "8159_baseline_ES_t11",
    "caseID": "8159_baseline_ES_t11_thick_parametrized",
    "isLV": False,
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

Circparam = {
    "Ees_la": 80,  # End-systolic elastance (60) --> Pa/ml
    "A_la": 59.0,  # Scaling factor for EDPVR --> ml
    "B_la": 0.05,  # Exponent for EDPVR --> ml-1
    "V0_la": 10,  # volume axis intercept --> ml
    "Tmax_la": 150,  # time to end-systole --> ms
    "tau_la": 25,  # time constant of relaxation --> ms
    "tdelay_la": 225,  #
    "Csa": 0.006,  # Proximal aorta compliance --> ml Pa
    "Cad": 0.02,  # Distal aorta compliance --> ml Pa
    "Csv": 0.8,  # Venous compliance -> ml Pa
    "Vsa0": 1800.0,  # Resting volume for proximal aorta --> ml
    "Vsv0": 1980.0,  # Resting venous volume (pre 3370.0 (2950, 3100, 3370)) --> ml
    "Vad0": 20.0,  # Resting volume for distal aorta --> ml
    "Rav": 5000.0,  # (pre 500 (500)) (aortic valve resistance) --> Pa ms ml-1
    "Rsv": 900.0,  # Venous resistance --> Pa ms ml-1
    "Rsa": 18000,  # Proximal aorta resistance --> Pa ms ml-1
    "Rad": 105000,  # Distal aorta resistance (10600, 12800, 21200, 31800) --> Pa ms ml-1
    "Rmv": 1900.0,  # Mitral valve resistance --> Pa ms ml-1
    "V_sv": 3326,
    "V_sa": 1857,
    "V_ad": 37,
    "V_LA": 35,
    "V_LV": 114,
    "stop_iter": 0,
}

SimDetails = {
    "diaplacementInfo_ref": False,
    "HeartBeatLength": 800.0,
    "dt": 1.0,
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
    # "aorta_ext_wall": 2,
    "aorta_ext_wall": 1,  # Base thick
    # "aorta_int_wall": 5,
    "aorta_int_wall": 7,  # Base thick
    # "aorta_ring": 9,
    "aorta_ring": 6,  # Base thick
    # "LVendoid": 8,
    "LVendoid": 5,  # Base thick
    "RVendoid": 0,
    "epiid": 3,
    "basid": 2,
    # "apxid": 100,
    # "aortic_vplane": 7,
    "aortic_vplane": 6,  # Base thick
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
    "active_region": [1],
    "Type": 0,
}

# Run Simulation
run_BiV_ClosedLoop(IODet=IODetails, SimDet=SimDetails)
# Postprocessing
# dumpvtk(IODet=IODetails, SimDet=SimDetails)
# compute_strain(IODet=IODetails, SimDet=SimDetails, LVid = 1)
# plothemodynamics(IODet=IODetails, SimDet=SimDetails, cycle=1)
# extractdisplacementloading(IODet=IODetails, SimDet=SimDetails)
# extractdisplacement(IODet=IODetails, SimDet=SimDetails)


#  - - - - - - - - - - - -- - - - - - - - - - - - - - - -- - - - - - -
