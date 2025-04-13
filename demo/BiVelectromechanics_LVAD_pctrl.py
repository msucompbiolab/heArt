import sys, pdb
from dolfin import *

sys.path.append("/mnt/Research")
sys.path.append("/mnt/Output")
sys.path.append("/mnt/Research/heArt")
sys.path.append("/mnt/Research/heArt_py3_original")
sys.path.append("/mnt/Research/heArt/heArt_py3/demo/FittingLVAD")

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
from LVADFn import LVAD as LVAD
HeartMate = LVAD("./FittingLVAD/HeartMate.npz")


#  - - - - - - - - - - - -- - - - - - - - - - - - - - - -- - - - - - -
IODetails = {
    "casename_me": "biv_idealized",
    "casename_ep": "biv_idealized",
    "directory_me": "../BiVMesh/",
    "directory_ep": "../BiVMesh/",
    "outputfolder": "/mnt/scratch/lclee/output_heArt_py3/outputs_BiVelectromechanics/",
    "folderName": "/",
    "caseID": "BiVelectromechanics_pctrl_p1p1_LVAD_7K",
}

contRactility = 200e3

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
    "HomogenousActivation": True,
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
    "V_sa":7.53966e+02, # 7.53937e+02, #7.52360e+02, #7.39492e+02,# 7.39872e+02, #7.42631e+02,
    "V_ad":3.48677e+02, # 3.48060e+02, #3.35851e+02, #2.85198e+02,# 2.87456e+02, #3.04518e+02,
    "V_sv":2.75086e+03, # 2.73329e+03, #2.70405e+03, #2.68173e+03,# 2.65740e+03, #2.64335e+03,
    "V_pa":3.72723e+02, # 3.72950e+02, #3.73726e+02, #3.75004e+02,# 3.75685e+02, #3.75742e+02,
    "V_pv":9.07160e+02, # 9.42127e+02, #1.00256e+03, #1.09984e+03,# 1.14305e+03, #1.14776e+03,
    "V_LA":1.36161e+01, # 1.37609e+01, #1.39903e+01, #1.44591e+01,# 1.46367e+01, #1.46561e+01,
    "V_LV":7.68626e+01, # 7.88765e+01, #8.28927e+01, #9.37059e+01,# 9.51769e+01, #9.53730e+01,
    "V_RV":8.83728e+01, # 8.53658e+01, #8.37537e+01, #7.62423e+01,# 8.32750e+01, #8.23440e+01,
    "V_RA":3.09132e+01, # 3.01507e+01, #2.88819e+01, #2.79130e+01,# 2.68550e+01, #2.62437e+01,
    "stop_iter": 10,
    # LVAD
    'Q_lvad_rpm' : 7000,
    'Q_lvad_scale' : 1.0,
    'Q_lvad_characteristic': HeartMate
}


SimDetails = {
    "HeartBeatLength": 800.0,
    "dt": 1.0,
    "EDP": 9.11432e+02*0.0075, #1.77139e+03*0.0075,#1.90088e+03*0.0075,#1.91646e+03*0.0075,
    "RVEDPfactor": 8.73190e+02/9.11432e+02, #1.03761e+03/1.77139e+03,#8.48190e+02/1.90088e+03,#7.70257e+02/1.91646e+03,
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
    "springbc": True,#0,
    "Mechanics Discretization": "P1P1",
    "isLV": False,
    "ispctrl": True,
    "isBiV": True,
    "epiid_Kadj_coeff": [10, 3], 
    "dashpotparam": [2.0e2,2.0e1],
    "RVtopid": 6,
    "LVtopid": 5,
    "topid": 4,
    "LVendoid": 2,
    "RVendoid": 3,
    "epiid": 1,
    "abs_tol": 1e-8,
    "rel_tol": 1e-9,
}


# Run Simulation
run_BiV_ClosedLoop(IODet=IODetails, SimDet=SimDetails)
# Postprocessing
# dumpvtk(IODet=IODetails, SimDet=SimDetails)
#compute_strain(IODet=IODetails, SimDet=SimDetails, LVid = [0,1], RVid = 2)
#plothemodynamics(IODet=IODetails, SimDet=SimDetails, cycle=10)
# extractdisplacement(IODet=IODetails, SimDet=SimDetails)
# extractdisplacementloading(IODet=IODetails, SimDet=SimDetails)
#  - - - - - - - - - - - -- - - - - - - - - - - - - - - -- - - - - - -
