import sys, pdb
from dolfin import *

sys.path.append("/home/vahid")

from heArt.src.sim_protocols.run_BiV_Isovolumic import (
    run_BiV_Isovolumic as run_BiV_Isovolumic,
)
from heArt.src.postprocessing.postprocessdataBiV_Iso import (
    postprocessdata as postprocessdata,
)

#  - - - - - - - - - - - -- - - - - - - - - - - - - - - -- - - - - - -
IODetails = {  # "casename" : "biv_HF",
    "outputfolder": "./outputs",
    # "caseID" : '91690'}
    "caseID": "test",
}

Circparam = {
    "stop_iter": 2,
}

SimDetails = {
    "HeartBeatLength": 250.0,
    "closedloopparam": Circparam,
}

# Postprocessing
postprocessdata(IODet=IODetails, SimDet=SimDetails)

#  - - - - - - - - - - -
