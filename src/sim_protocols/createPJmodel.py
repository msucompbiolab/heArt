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
#from ..ep.EPmodel_basic_test import EPmodel
from ..ep.EPmodel_cpp import EPmodel


def createPJmodel(IODet, SimDet):

    outputfolder = IODet["outputfolder"]
    folderName = IODet["folderName"] + IODet["caseID"] + "/"
    directory_pj = IODet["directory_pj"]
    casename = IODet["casename_pj"]
    delTat = SimDet["dt"]

    # Read EP data from HDF5 Files
    mesh_pj = Mesh()
    comm_pj = mesh_pj.mpi_comm()

    meshfilename_pj = directory_pj + casename + ".hdf5"
    f = HDF5File(comm_pj, meshfilename_pj, "r")
    f.read(mesh_pj, casename, False)
    File(outputfolder + folderName + "mesh_pj.pvd") << mesh_pj

    AHAid_pj = MeshFunction(
        "size_t", mesh_pj, 1, mesh_pj.domains()
    )
    AHAid_pj.set_all(0)

    matid_pj = MeshFunction(
        "size_t", mesh_pj, 1, mesh_pj.domains()
    )
    matid_pj.set_all(0)
    f.read(matid_pj, casename + "/" + "matid")

    # Define state variables
    #  - - - - - - - - - - - -- - - - - - - - - - - - - - - -- - - - - - -
    state_obj = State_Variables(comm_pj, SimDet)
    state_obj.dt.dt = delTat
 
    pj_intensity = SimDet["PJ_current_intensity"]

    # Define EP model and solver
    EPparams_pj = {
        "EPmesh": mesh_pj,
        "deg": 4,
        "state_obj": state_obj,
        "d_iso": SimDet["d_pj"],
        "ploc": SimDet["ploc"],
        "AHAid": AHAid_pj,
        "matid": matid_pj,
        "Ischemia": SimDet["Ischemia"],
        "ploc_mode":SimDet["ploc_mode"],
        "lbbb":SimDet["lbbb"],
        "lbbb_delay": SimDet["lbbb_delay"], 
        "lbbb_location": SimDet["lbbb_location"], 
    }

    if("d_iso_pj" in list(SimDet.keys())):
        EPparams_pj.update({"d_iso_pj": SimDet["ploc"]})


    # Define EP model and solver
    EPmodel_ = EPmodel(EPparams_pj)

    return EPmodel_, state_obj, EPparams_pj


