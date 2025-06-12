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



def createEPmodel(IODet, SimDet):

    directory_ep = IODet["directory_ep"]
    outputfolder = IODet["outputfolder"]
    folderName = IODet["folderName"] + IODet["caseID"] + "/"
    delTat = SimDet["dt"]

    if "casename_ep" in IODet:
        casename = IODet["casename_ep"]
    else:
        casename = IODet["casename"]

    if "isFCH" in list(SimDet.keys()):
        isFCH = SimDet["isFCH"]
    else:
        isFCH = False
    if "iswaorta" in list(SimDet.keys()):
        iswaorta = SimDet["iswaorta"]
    else:
        iswaorta = False  # Default

    #  - - - - - - - - - - - -- - - - - - - - - - - - - - - -- - - - - - -
    # Read EP data from HDF5 Files
    mesh_ep = Mesh()
    comm_common = mesh_ep.mpi_comm()

    meshfilename_ep = directory_ep + casename + ".hdf5"
    # meshfilename_ep = directory_ep + casename + "_refine.hdf"
    f = HDF5File(comm_common, meshfilename_ep, "r")
    f.read(mesh_ep, casename, False)
    if isFCH:
        mesh_ep.scale(6.5e-2)
    elif iswaorta:
        mesh_ep.scale(1.1)
        # pass

    File(outputfolder + folderName + "mesh_ep.pvd") << mesh_ep

    facetboundaries_ep = MeshFunction("size_t", mesh_ep, 2)
    f.read(facetboundaries_ep, casename + "/" + "facetboundaries")

    matid_ep = MeshFunction("size_t", mesh_ep, mesh_ep.topology().dim())
    AHAid_ep = MeshFunction("size_t", mesh_ep, mesh_ep.topology().dim())

    if f.has_dataset(casename + "/" + "matid"):
        if SimDet.get("function_matid"):
            VQuadelem = FiniteElement(
                "DG", mesh_ep.ufl_cell(), degree=0, quad_scheme="default"
            )
            matid_FS = FunctionSpace(mesh_ep, VQuadelem)
            matid_func = dolfin.Function(matid_FS)
            for cell in cells(mesh_ep):
                matid_ep[cell.index()] = round(matid_func(cell.midpoint()))
        else:
            f.read(matid_ep, casename + "/" + "matid")
    else:
        matid_ep.set_all(0)

    if f.has_dataset(casename + "/" + "AHAid"):
        f.read(AHAid_ep, casename + "/" + "AHAid")
    else:
        AHAid_ep.set_all(0)

    deg_ep = 4

    Quadelem_ep = FiniteElement(
        "Quadrature", mesh_ep.ufl_cell(), degree=deg_ep, quad_scheme="default"
    )
    Quadelem_ep._quad_scheme = "default"
    Quad_ep = FunctionSpace(mesh_ep, Quadelem_ep)

    if "fiber_fspace" in list(SimDet.keys()) and "fiber_fspace_deg" in list(
        SimDet.keys()
    ):
        VQuadelem_ep = VectorElement(
            SimDet["fiber_fspace"],
            mesh_ep.ufl_cell(),
            degree=SimDet["fiber_fspace_deg"],
            quad_scheme="default",
        )
        VQuadelem_ep._quad_scheme = "default"
    else:
        VQuadelem_ep = VectorElement(
            "Quadrature", mesh_ep.ufl_cell(), degree=deg_ep, quad_scheme="default"
        )
        VQuadelem_ep._quad_scheme = "default"

    fiberFS_ep = FunctionSpace(mesh_ep, VQuadelem_ep)

    f0_ep = Function(fiberFS_ep)
    s0_ep = Function(fiberFS_ep)
    n0_ep = Function(fiberFS_ep)

    if SimDet["DTI_EP"] is True:
        f.read(f0_ep, casename + "/" + "eF_DTI")
        f.read(s0_ep, casename + "/" + "eS_DTI")
        f.read(n0_ep, casename + "/" + "eN_DTI")
    else:
        f.read(f0_ep, casename + "/" + "eF")
        f.read(s0_ep, casename + "/" + "eS")
        f.read(n0_ep, casename + "/" + "eN")

    f.close()

    comm_ep = mesh_ep.mpi_comm()

    # Define state variables
    #  - - - - - - - - - - - -- - - - - - - - - - - - - - - -- - - - - - -
    state_obj = State_Variables(comm_ep, SimDet)
    state_obj.dt.dt = delTat
    #  - - - - - - - - - - - -- - - - - - - - - - - - - - - -- - - - - - -

    if "d_iso" in list(SimDet.keys()):
        d_iso = SimDet["d_iso"]
    else:
        d_iso = 0.02

    if "d_ani_factor" in list(SimDet.keys()):
        d_ani_factor = SimDet["d_ani_factor"]
    else:
        d_ani_factor = 0.02

    if "ani_factor" in list(SimDet.keys()):
        ani_factor = SimDet["ani_factor"]
    else:
        ani_factor = 1000.0

    EPparams = {
        "EPmesh": mesh_ep,
        "deg": 4,
        "matid": matid_ep,
        "facetboundaries": facetboundaries_ep,
        "f0": f0_ep,
        "s0": s0_ep,
        "n0": n0_ep,
        "state_obj": state_obj,
        "d_iso": d_iso,
        "d_ani": d_ani_factor,
        "ani_factor": ani_factor,
        "ploc": SimDet["ploc"],
        "AHAid": AHAid_ep,
        "matid": matid_ep,
        "Ischemia": SimDet["Ischemia"],
    }

    if "ploc" in list(SimDet.keys()):
        EPparams.update({"ploc": SimDet["ploc"]})

    if "isPJ" in list(SimDet.keys()):
        if SimDet["isPJ"] and "pj_tnodes" in list(SimDet.keys()):
            EPparams.update({"ploc": SimDet["pj_tnodes"]})
        if SimDet["isPJ"] and "PJ_current_intensity" in list(SimDet.keys()):
            EPparams.update({"current_intensity": SimDet["PJ_current_intensity"]})


    if "Ischemia" in list(SimDet.keys()):
        EPparams.update({"Ischemia": SimDet["Ischemia"]})
    if "pacing_timing" in list(SimDet.keys()):
        EPparams.update({"pacing_timing": SimDet["pacing_timing"]})

    # Define EP model and solver
    EPmodel_ = EPmodel(EPparams)

    return EPmodel_, state_obj, EPparams

