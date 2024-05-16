import sys, pdb
from dolfin import *

sys.path.append("/home/ziaeirad")

# from heArt.src.sim_protocols.run_BiV_Isovolumic import (
#    run_BiV_Isovolumic as run_BiV_Isovolumic,
# )
from heArt.src.postprocessing.postprocessdataLV_Iso import (
    postprocessdata as postprocessdata,
)

#  - - - - - - - - - - - -- - - - - - - - - - - - - - - -- - - - - - -
IODetails = {  # "casename" : "biv_HF",
    "outputfolder": "./outputs_LVelectromechanics",
    "caseID": "LVelectromechanics_p1_basefix",
}

Circparam = {
    "stop_iter": 0,
}

SimDetails = {
    "HeartBeatLength": 800.0,
    "closedloopparam": Circparam,
}

# Postprocessing
postprocessdata(IODet=IODetails, SimDet=SimDetails)

#  - - - - - - - - - - -
