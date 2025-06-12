import sys, shutil, math
import os as os
import numpy as np
from mpi4py import MPI as pyMPI
from dolfin import *
import dolfin as dolfin
from fenicstools import *

import vtk_py3
import vtk

from ..utils.oops_objects_MRC2 import State_Variables
from ..mechanics.MEmodel3 import MEmodel

def createMEmodel(IODet, SimDet):

    mesh_me = Mesh()
    comm_me = mesh_me.mpi_comm()

    if "isLV" in list(SimDet.keys()):
        isLV = SimDet["isLV"]
    else:
        isLV = False  # Default

    # Define state variables
    #  - - - - - - - - - - - -- - - - - - - - - - - - - - - -- - - - - - -
    state_obj = State_Variables(comm_me, SimDet)
    state_obj.dt.dt = SimDet["dt"]
    #  - - - - - - - - - - - -- - - - - - - 

    casename_me = IODet["casename_me"]
    directory_me = IODet["directory_me"]
    outputfolder = IODet["outputfolder"]
    folderName = IODet["folderName"] + IODet["caseID"] + "/"

    #  - - - - - - - - - - - -- - - - - - - - - - - - - - - -- - - - - - -
    # Mechanics Mesh
    mesh_me_params = {
        "directory": directory_me,
        "casename": casename_me,
        "fibre_quad_degree": 4,
        "outputfolder": outputfolder,
        "foldername": folderName,
        "state_obj": state_obj,
        "common_communicator": comm_me,
        "MEmesh": mesh_me,
        "isLV": isLV,
    }

    MEmodel_ = MEmodel(mesh_me_params, SimDet)


    return MEmodel_, state_obj, mesh_me_params



