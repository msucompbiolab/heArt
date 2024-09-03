import sys, pdb
from dolfin import *

sys.path.append("/mnt/Research")
from heArt_py3.src.sim_protocols.run_BiV_ClosedLoop import (
    run_BiV_ClosedLoop as run_BiV_ClosedLoop,
)

# from heArt_py3.src.postprocessing.postprocessdataBiV2 import (
#    postprocessdata as postprocessdata,
# )

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
IODetails = {
    "casename": "biv_idealized",
    "directory_me": "../BiVMesh/",
    "directory_ep": "../BiVMesh/",
    "outputfolder": "./outputs_BiVelectromechanics/",
    "folderName": "",
    "caseID": "BiVelectromechanics",
}

contRactility = 100e3

GuccioneParams = {
    "ParamsSpecified": True,
    "Passive model": {"Name": "Guccione"},
    "Passive params": {
        "Cparam": Constant(100.0),
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

# Systemic
Circparam = {
    "Ees_la": 350,
    "A_la": 58.67,
    "B_la": 0.049,
    "V0_la": 10,
    "Tmax_la": 120,
    "tau_la": 75,
    "tdelay_la": 160,
    "Csa": 0.0052,
    "Cad": 0.0330,
    "Csv": 0.3,
    "Vsa0": 700,
    "Vsv0": 2500.0,
    "Vad0": 40,
    "Rav": 500.0,
    "Rsv": 100.0,
    "Rsa": 18000,
    "Rad": 106000,
    "Rmv": 1500.0,
    # Pulmonary
    "Ees_ra": 81.33,
    "A_ra": 466.6,
    "B_ra": 0.033,
    "V0_ra": 20,
    "Tmax_ra": 120,
    "tau_ra": 25,
    "tdelay_ra": 100,
    "Cpa": 0.0125,
    "Cpv": 0.9,
    "Vpa0": 360,
    "Vpv0": 15,
    "Rpv": 500.0,
    "Rtv": 400.0,
    "Rpa": 10000.0,
    "Rpvv": 400,
    # flow rate
    "Q_lvad": 0.0,
    "Q_sv": 0.0659387173129,
    "Q_av": 0.0,
    "Q_sa": 0.0113526347222,
    "Q_ad": 0.0797504135138,
    "Q_mv": 0.0,
    "Q_tv": 0.0,
    "Q_pa": 0.001213376269,
    "Q_pv": 0.0646105314407,
    "Q_pvv": 0.0,
    # volumes
    "V_sv": 2620.36391023,
    "V_LV": 114.197219883,
    "V_sa": 747.095750279,
    "V_ad": 332.137059531,
    "V_LA": 14.6706180758,
    "V_pv": 1135.52295648,
    "V_RV": 107.18412269,
    "V_pa": 375.714158013,
    "V_RA": 26.1123080465,
    "stop_iter": 1,
    # LVAD
    #'Q_lvad_rpm' : 28,
    #'Q_lvad_scale' : 0.0
}


SimDetails = {
    "HeartBeatLength": 800.0,
    "dt": 1.0,
    "writeStep": 40.0,
    "GiccioneParams": GuccioneParams,
    "nLoadSteps": 10,
    "DTI_EP": False,
    "DTI_ME": False,
    "d_iso": 1.5 * 0.01,
    "d_ani_factor": 4.0,
    "ploc": [[-0.083, 5.6, -1.16, 2.0, 1.0]],
    "pacing_timing": [[4.0, 20.0]],
    "closedloopparam": Circparam,
    "Ischemia": False,
    "Mechanics Discretization": "P1P1",
    "isLV": False,
    "topid": 4,
    "LVendoid": 2,
    "RVendoid": 3,
    "epiid": 1,
    "abs_tol": 1e-9,
    "rel_tol": 1e-9,
}


# Run Simulation
run_BiV_ClosedLoop(IODet=IODetails, SimDet=SimDetails)
# Postprocessing
# postprocessdata(IODet=IODetails, SimDet=SimDetails)
# extractdisplacement(IODet=IODetails, SimDet=SimDetails)

#  - - - - - - - - - - - -- - - - - - - - - - - - - - - -- - - - - - -
