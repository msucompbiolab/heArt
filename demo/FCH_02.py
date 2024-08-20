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
    extractdisplacement as extractdisplacement,
)
from heArt_py3.src.postprocessing.postprocessdata2 import (
    extractdisplacementloading as extractdisplacementloading,
)

#  - - - - - - - - - - - -- - - - - - - - - - - - - - - -- - - - - - -
# ellipsoidal_baselinegeo
IODetails = {
    "casename": "fchmesh_scale_w_valves_remarked",
    "directory_me": "../FCHMesh/",
    "directory_ep": "../FCHMesh/",
    "outputfolder": "./outputs_FCHMesh/",
    "folderName": "",
    "caseID": "FCHMesh",
    "isLV": False,
    "isFCH": True,
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
    "Ees_la": 10,  # End-systolic elastance (60) --> Pa/ml
    "A_la": 2.67,  # Scaling factor for EDPVR --> ml
    "B_la": 0.019,  # Exponent for EDPVR --> ml-1
    "V0_la": 10,  # volume axis intercept --> ml
    "Tmax_la": 120,  # time to end-systole --> ms
    "tau_la": 25,  # time constant of relaxation --> ms
    "tdelay_la": 160,  #
    "Csa": 0.0032,  # Proximal aorta compliance --> ml Pa
    "Cad": 0.0330,  # Distal aorta compliance --> ml Pa
    "Csv": 0.28,  # Venous compliance -> ml Pa
    "Vsa0": 360,  # Resting volume for proximal aorta --> ml
    "Vsv0": 3370.0,  # Resting venous volume (pre 3370.0 (2950, 3100, 3370)) --> ml
    "Vad0": 40,  # Resting volume for distal aorta --> ml
    "Rav": 3000.0,  # (pre 500 (500)) (aortic valve resistance) --> Pa ms ml-1
    "Rsv": 100.0,  # Venous resistance --> Pa ms ml-1
    "Rsa": 18000,  # Proximal aorta resistance --> Pa ms ml-1
    "Rad": 21200,  # Distal aorta resistance (info not available) --> Pa ms ml-1
    "Rmv": 200.0,  # Mitral valve resistance --> Pa ms ml-1
    "V_sv": 3700,
    "V_LV": 112,
    "V_sa": 740,
    "V_ad": 100,
    "V_LA": 12,
    "V_LV": 112,
    "stop_iter": 1,
}

SimDetails = {
    "diaplacementInfo_ref": False,
    "HeartBeatLength": 800.0,
    "dt": 2.0,
    "writeStep": 20.0,
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
    "Mechanics Discretization": "P1P1",
    "Technique Discretization": 1,
    "isLV": False,
    "aorta_wall": 9,  # aorta external wall indeed
    "pulm_wall": 7,  # pulmonary ring indeed
    "LVendoid": 18,
    "RVendoid": 0,
    "epiid": 17,
    "apxid": 1000,
    "septumid": 13,
    "mitral_vplane": 19,
    "aortic_vplane": 20,
    "abs_tol": 1e-8,
    "rel_tol": 1e-9,
    "isunloading": False,
    "isunloadingonly": False,
    "ispctrl": True,
    "isFCH": True,
    "springbc": 1,
    "springparam": [2.0e3, 2.0e3],  # Kepi_n / Kepi_t
    "dashpotparam": [2.0e2, 2.0e2],  # Cepi_n / Cepi_t
    "active_region": [1],  # only lv is activated
    "Type": 0,
}

# Run Simulation
run_BiV_ClosedLoop(IODet=IODetails, SimDet=SimDetails)
# Postprocessing
# postprocessdata(IODet=IODetails, SimDet=SimDetails)
# dumpvtk(IODet=IODetails, SimDet=SimDetails)
# compute_strain(IODet=IODetails, SimDet=SimDetails, LVid = 1)
# plothemodynamics(IODet=IODetails, SimDet=SimDetails, cycle=None)
# extractdisplacementloading(IODet=IODetails, SimDet=SimDetails)
# extractdisplacement(IODet=IODetails, SimDet=SimDetails)
#  - - - - - - - - - - - -- - - - - - - - - - - - - - - -- - - - - - -
