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
from heArt_py3.src.postprocessing.postprocessdata2 import compute_strain as compute_strain
from heArt_py3.src.postprocessing.postprocessdata2 import plothemodynamics as plothemodynamics
from heArt_py3.src.postprocessing.postprocessdata2 import extractdisplacement as extractdisplacement
from heArt_py3.src.postprocessing.postprocessdata2 import extractdisplacementloading as extractdisplacementloading
#  - - - - - - - - - - - -- - - - - - - - - - - - - - - -- - - - - - -
# ellipsoidal_baselinegeo
IODetails = {
    "casename": "LV_w_aorta5",
    "directory_me": "../LV_waorta/",
    "directory_ep": "../LV_waorta/",
    "outputfolder": "./outputs_LV_waorta/",
    "folderName": "",
    "caseID": "LV_waorta",
    "isLV": False,
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
    "Kappa": 1e5,
    "incompressible": True,
}

Circparam = {
    "Ees_la": 10,  # End-systolic elastance (60) --> Pa/ml ok
    "A_la": 2.67,  # Scaling factor for EDPVR --> ml ok
    "B_la": 0.019,  # Exponent for EDPVR --> ml-1 ok
    "V0_la": 10,  # volume axis intercept --> ml ok
    "Tmax_la": 120,  # time to end-systole --> ms ok
    "tau_la": 25,  # time constant of relaxation --> ms ok
    "tdelay_la": 160,  # XXX
    "Csa": 0.0032,  # Proximal aorta compliance --> ml Pa ok
    "Cad": 0.0330,  # Distal aorta compliance --> ml Pa ok
    "Csv": 0.28,  # Venous compliance -> ml Pa ok
    "Vsa0": 360,  # Resting volume for proximal aorta --> ml ok
    "Vsv0": 2950.0,  # Resting venous volume (also (2950, 3100, 3370)) --> ml ok
    "Vad0": 40,  # Resting volume for distal aorta --> ml ok
    "Rav": 1000.0,  # (also 500 -- increase for less oscillation) (aortic valve resistance) --> Pa ms ml-1 ok (susp. 500 --> 1000)
    "Rsv": 100.0,  # Venous resistance --> Pa ms ml-1 ok
    "Rsa": 18000,  # Proximal aorta resistance --> Pa ms ml-1 ok
    "Rad": 21200,  # Distal aorta resistance (also (10600, 12800, 21200, 31800)) --> Pa ms ml-1 ok
    "Rmv": 200.0,  # Mitral valve resistance --> Pa ms ml-1 ok
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
    "writeStep": 20.0,
    "GiccioneParams": GuccioneParams,
    "nLoadSteps": 15, # u_loading
    "DTI_EP": False,
    "DTI_ME": False,
    "d_iso": 1.5 * 0.005,
    "d_ani_factor": 4.0,
    "ploc": [[1.4, 1.4, -3.0, 2.0, 1]],  # , [-1.4, -1.4, -3.0, 2.0, 2]],
    "pacing_timing": [[4.0, 20.0]],  # , [20.0, 20.0]],
    "Isclosed": True,
    "closedloopparam": Circparam,
    "Mechanics Discretization": "P1P1",
    "Technique Discretization": 0,
    "Ischemia": False,
    "Mechanics Discretization": "P1P1",
    "isLV": False,
    "aorta_ext_wall": 6,
    "aorta_int_wall": 5,
    "aorta_ring": 4,
    "LVendoid": 1,
    "RVendoid": 0,
    "epiid": 3,
    "aortic_vplane": 2,
    "abs_tol": 1e-8,
    "rel_tol": 1e-9,
    "isunloading": False,
    "isunloadingonly": False,
    "ispctrl": True,
    "iswaorta": True,
    "springbc": 1,
    "springparam": [2.0e4, 5.0e2],  # Kepi_n / Kepi_t
    "dashpotparam": [5.0e2, 5.0e1],  # Cepi_n / Cepi_t
    # "springparam": [5.0e4, 5.0e3],
    # "dashpotparam": [5.0e3, 5.0e2],
    "active_region": [1],
    "Type": 0,
}

# Run Simulation
run_BiV_ClosedLoop(IODet=IODetails, SimDet=SimDetails)
# Postprocessing
# dumpvtk(IODet=IODetails, SimDet=SimDetails)
# compute_strain(IODet=IODetails, SimDet=SimDetails, LVid = 1)
# plothemodynamics(IODet=IODetails, SimDet=SimDetails, cycle=None)
# extractdisplacementloading(IODet=IODetails, SimDet=SimDetails)


#  - - - - - - - - - - - -- - - - - - - - - - - - - - - -- - - - - - -
