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
    extractdisplacement as extractdisplacement,
)
from heArt_py3.src.postprocessing.postprocessdata2 import (
    extractdisplacementloading as extractdisplacementloading,
)

#  - - - - - - - - - - - -- - - - - - - - - - - - - - - -- - - - - - -
# ellipsoidal_baselinegeo
IODetails = {
    "casename": "fch_clregion",
    "directory_me": "../FCHMesh/vh/",
    "directory_ep": "../FCHMesh/vh/",
    "outputfolder": "/mnt/Output/outputs_minifch/",
    "folderName": "",
    "caseID": "minifch_tricuspid_restv",
    "isLV": False,
    "isFCH": True,
}

contRactility = 500e3

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
    "Ees_la": 20, # 60 la end-sys elastance
    "A_la": 2.67, # 2.67 scaling factor
    "B_la": 0.019, # 0.019 exponent
    "V0_la": 10, # 10 vol axis intercept
    "Tmax_la": 120, # 120 time to end-sys
    "tau_la": 25, # 25 relaxation
    "tdelay_la": 160, # it's not in the paper, figure out why
    "Csa": 0.0055, # 0.0032 -- proximal aorta compliance
    "Cad": 0.033, # 0.033 -- distal aorta compliance
    "Csv": 0.28, # 0.28 -- venous compliance
    "Vsa0": 550, # 360 resting volume for proximal aorta (#sug increase to reduce Psa)
    "Vsv0": 2150.0, #xpaper resting volume for venous (#sug reduced)
    "Vad0": 40, # 40 resting volume for distal aorta
    "Rav": 500.0, # 500 aortic valve resistance
    "Rsv": 100.0, # 100 venous resistance #sug ? increase (--> 200)
    "Rsa": 6000, # 18000 proximal aorta resistance (--> 8000 aiming at reduce Psa)
    "Rad": 31800, #xpaper distal aorta resistance
    "Rmv": 200.0, # mitral valve resistance 
    # Pulmonary
    "Ees_ra": 40.0,
    "A_ra": 2.67,
    "B_ra": 0.019,
    "V0_ra": 10.0,
    "Tmax_ra": 120,
    "tau_ra": 25,
    "tdelay_ra": 130,
    "Cpa": 0.01, #prev 0.0125
    "Cpv": 0.9,
    "Vpa0": 320,
    "Vpv0": 1000,
    "Rpv": 500.0,
    "Rtv": 400.0,
    "Rpa": 10000.0,
    "Rpvv": 400,
    # Volumes
    "V_sv": 2600.0,
    "V_LV": 240.0,
    "V_sa": 739.0,
    "V_ad": 283.0,
    "V_LA": 84.0,
    "V_pv": 3829.0,
    "V_RV": 270.0,
    "V_pa": 402.0,
    "V_RA": 49.0,
    # no iteration
    "stop_iter": 9,
    # LVAD
    #'Q_lvad_rpm' : 28,
    #'Q_lvad_scale' : 0.0
}


SimDetails = {
    "diaplacementInfo_ref": False,
    "HeartBeatLength": 800.0,
    "dt": 0.5,
    "writeStep": 40.0,
    "GiccioneParams": GuccioneParams,
    "nLoadSteps": 64,
    "DTI_EP": False,
    "DTI_ME": False,
    "d_iso": 1.5 * 0.01,
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
    "aorta_wall": 7,  # aorta truncated surface (ring plus)
    # "pulm_wall": 12,  # pulmonary ring (not marked in current vtp)
    "LVendoid": 1,
    "RVendoid": 6,
    "epiid": 9,
    "apxid": 11,
    "aortaid": 8, # aorta external wall
    # "septumid": -1,
    # "mitral_valvep": 16, (not marked in current vtp)
    # "aortic_valvep": 17, (not marked in current vtp)
    # "pulmonary_valvep": 13,
    # "tricuspid_valvep": 14,
    "abs_tol": 1e-8,
    "rel_tol": 1e-9,
    "isunloading": False,
    "isunloadingonly": False,
    "ispctrl": True,
    "isFCH": True,
    "springbc": 1,
    "springparam": [7.0e3, 7.0e3],  # Kepi_n / Kepi_t
    "dashpotparam": [7.0e2, 7.0e1],  # Cepi_n / Cepi_t
    "active_region": [1, 2],
    "Type": 0,
    "function_matid": False,
}

# Run Simulation
run_BiV_ClosedLoop(IODet=IODetails, SimDet=SimDetails)
# Postprocessing
# postprocessdata(IODet=IODetails, SimDet=SimDetails)
# dumpvtk(IODet=IODetails, SimDet=SimDetails)
# compute_strain(IODet=IODetails, SimDet=SimDetails, LVid = 1)
# plothemodynamics(IODet=IODetails, SimDet=SimDetails, cycle=10)
# extractdisplacementloading(IODet=IODetails, SimDet=SimDetails)
# extractdisplacement(IODet=IODetails, SimDet=SimDetails)
#  - - - - - - - - - - - -- - - - - - - - - - - - - - - -- - - - - - -
