import sys, shutil, pdb, math
import os as os
import numpy as np
from mpi4py import MPI as pyMPI
from dolfin import *
import dolfin
from fenicstools import *
import vtk_py3
import vtk

from ..utils.oops_objects_MRC2 import printout
from ..utils.oops_objects_MRC2 import biventricle_mesh as biv_mechanics_mesh
from ..utils.oops_objects_MRC2 import lv_mesh as lv_mechanics_mesh
from ..utils.oops_objects_MRC2 import State_Variables
from ..utils.oops_objects_MRC2 import update_mesh
from ..utils.oops_objects_MRC2 import exportfiles
from ..utils.mesh_scale_create_fiberFiles import create_EDFibers

from ..ep.EPmodel_basic import EPmodel
from ..mechanics.MEmodel3 import MEmodel
from .circ import CLmodel


def run_BiV_ClosedLoop(IODet, SimDet):
    deg = 4
    flags = ["-O3", "-ffast-math", "-march=native"]
    parameters["form_compiler"]["representation"] = "quadrature"
    parameters["form_compiler"]["quadrature_degree"] = deg

    casename = IODet["casename"]
    casename_pj = IODet["casename_pj"]
    # directory_me = IODet["directory_me"]
    directory_pj = IODet["directory_pj"]
    outputfolder = IODet["outputfolder"]
    folderName = IODet["folderName"]
    caseID = IODet["caseID"]
    delTat = SimDet["dt"]
    stop_iter = SimDet["closedloopparam"]["stop_iter"]
    ploc_tol = SimDet["ploc_tol"]

    # Create Purkinjee EP model
    # Define Purkinje terminal nodes
    if "tnode" in list(SimDet.keys()):
        if isinstance(SimDet["tnode"], str):
            pj_t_nodes = np.genfromtxt(
                os.path.join(directory_pj, SimDet["tnode"]), delimiter=","
            )
        elif isinstance(SimDet["tnode"], list):
            pj_t_nodes = np.array(SimDet["tnode"])

    SimDet.update({"tnode": list(pj_t_nodes)})

    EPmodel_pj, state_obj_pj = createPJmodel(IODet, SimDet)
    solver_FHN_pj = EPmodel_pj.Solver()

    # Create Tissue EP model
    EPmodel_ep, state_obj_ep = createEPmodel(IODet, SimDet)
    solver_FHN_ep = EPmodel_ep.Solver()

    # Create Tissue ME model
    MEmodel_ = createMEmodel(IODet, SimDet, state_obj_ep)
    solver_elas = MEmodel_.Solver()

    # Define export for exporting data
    comm_ep = EPmodel_ep.mesh.mpi_comm()
    comm_pj = EPmodel_pj.mesh.mpi_comm()
    comm_me = MEmodel_.Mesh.mesh.mpi_comm()
    export = exportfiles(comm_me, comm_ep, IODet, SimDet)

    export.hdf.write(MEmodel_.Mesh.mesh, "ME/mesh")
    export.hdf.write(EPmodel_ep.mesh, "EP/mesh")
    export.hdf.write(EPmodel_pj.mesh, "PJ/mesh")

    Loading(MEmodel_, export, SimDet)
    V_LV = MEmodel_.GetLVV()
    P_LV = MEmodel_.GetLVP()  # LVCavitypres.pres
    CLmodel_ = CLmodel(SimDet, V_LV)

    tstart_arr = [-10] * len(pj_t_nodes)
    probesPJ = Probes(pj_t_nodes.flatten(), EPmodel_pj.w_ep.function_space().sub(0))
    comms = EPmodel_pj.mesh.mpi_comm()

    # File_EP = File(outputfolder + folderName + caseID + "/" + "EP.pvd")
    # File_PJ = File(outputfolder + folderName + caseID + "/" + "PJ.pvd")
    # File_ME = File(outputfolder + folderName + caseID + "/" + "ME.pvd")

    cnt = 0
    writecnt = 0
    potential_me = Function(FunctionSpace(MEmodel_.mesh_me, "CG", 1))

    while 1:
        if state_obj_ep.cycle > stop_iter:
            break

        if state_obj_ep.tstep > stop_iter * SimDet["HeartBeatLength"]:
            break

        # Activate PK fiber
        if (
            state_obj_ep.t > SimDet["pacing_timing"][0][0]
            and state_obj_ep.t
            < SimDet["pacing_timing"][0][0] + SimDet["pacing_timing"][0][1]
        ):

            EPmodel_pj.fstim_array[0].iStim = 5.0

        else:

            EPmodel_pj.fstim_array[0].iStim = 0.0

        params = {
            "P_LV": P_LV,
            "V_LV": V_LV,
            "t": state_obj_ep.t,
            "delTat": state_obj_ep.dt.dt,
        }

        V_LV = CLmodel_.UpdateLVV(params)
        P_LV = MEmodel_.GetLVP()  # LVCavitypres.pres

        printout(
            "t = "
            + str(state_obj_ep.t)
            + "V_LV = "
            + str(V_LV)
            + " Psa = "
            + str(CLmodel_.Psa)
            + " PLA = "
            + str(CLmodel_.GetPLA(params))
            + " P_LV = "
            + str(P_LV),
            comm_me,
        )

        MEmodel_.LVCavityvol.vol = V_LV

        solver_elas.solvenonlinear()
        isrestart = 0
        state_obj_ep.dt.dt = delTat

        state_obj_ep.tstep = state_obj_ep.tstep + state_obj_ep.dt.dt
        state_obj_ep.cycle = math.floor(state_obj_ep.tstep / state_obj_ep.BCL)
        state_obj_ep.t = state_obj_ep.tstep - state_obj_ep.cycle * state_obj_ep.BCL

        MEmodel_.t_a.vector()[:] = state_obj_ep.t

        isrestart = 0
        state_obj_ep.dt.dt = delTat

        # Reset phi and r in EP at end of diastole
        if state_obj_ep.t < state_obj_ep.dt.dt:
            EPmodel_ep.reset()
            EPmodel_pj.reset()
            tstart_arr = [-10] * len(pj_t_nodes)

        printout("Solving FHN EP", comm_ep)
        solver_FHN_ep.solvenonlinear()
        printout("Solving FHN PJ", comm_pj)
        solver_FHN_pj.solvenonlinear()

        if isrestart == 0:
            print("Update Var", flush=True)
            MEmodel_.UpdateVar()  # For damping
            EPmodel_ep.UpdateVar()
            EPmodel_pj.UpdateVar()

        # Interpolate phi to mechanics mesh
        potential_ref = EPmodel_ep.interpolate_potential_ep2me_phi(
            V_me=Function(FunctionSpace(MEmodel_.mesh_me, "CG", 1))
        )
        potential_ref.rename("v_ref", "v_ref")

        potential_me.vector()[:] = potential_ref.vector().get_local()[:]

        #  - - - - - - - - - - - -- - - - - - - - - - - - - - - -- - - - - - -
        if MPI.rank(comms) == 0:
            print("UPdating isActiveField and tInitiationField")

        MEmodel_.activeforms.update_activationTime(
            potential_n=potential_me, comm=comm_me
        )

        probesPJ(EPmodel_pj.getphivar())
        Nevals = probesPJ.number_of_evaluations()
        probes_val = probesPJ.array()

        # broadcast from proc 0 to other processes
        rank = MPI.rank(comms)

        probes_val_bcast = probesPJ.array(
            N=Nevals - 1
        )  ## probe will only send to rank =0
        if not rank == 0:
            probes_val_bcast = np.empty(len(pj_t_nodes))
        comms.Bcast(probes_val_bcast, root=0)

        for p in range(0, len(pj_t_nodes)):

            phi_pj_val = probes_val_bcast[p]

            if phi_pj_val > 0.9 and tstart_arr[p] < 1.0:
                if tstart_arr[p] < 0:
                    tstart_arr[p] = 0
                    EPmodel_ep.fstim_array[p].iStim = 1.0
                    # print("Activate T node", p)
                else:
                    tstart_arr[p] += state_obj_ep.dt.dt
            else:
                EPmodel_ep.fstim_array[p].iStim = 0.0
                # print("Deactivate T node", p)

        if cnt % SimDet["writeStep"] == 0.0:
            export.writetpt(MEmodel_, state_obj_ep.tstep)
            export.hdf.write(MEmodel_.GetDisplacement(), "ME/u", writecnt)
            export.hdf.write(potential_ref, "ME/potential_ref", writecnt)
            export.hdf.write(EPmodel_ep.getphivar(), "EP/phi", writecnt)
            export.hdf.write(EPmodel_ep.getrvar(), "EP/r", writecnt)
            export.hdf.write(EPmodel_pj.getphivar(), "PJ/phi", writecnt)
            export.hdf.write(EPmodel_pj.getrvar(), "PJ/r", writecnt)
            writecnt += 1

        #            File_EP << EPmodel_ep.getphivar()
        #            File_PJ << EPmodel_pj.getphivar()
        #            File_ME << MEmodel_.GetDisplacement()

        cnt += 1


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
    )  # CellFunction("size_t", meshEP.mesh)
    AHAid_pj.set_all(0)

    matid_pj = MeshFunction(
        "size_t", mesh_pj, 1, mesh_pj.domains()
    )  # CellFunction("size_t", meshEP.mesh)
    matid_pj.set_all(0)

    # Define state variables
    #  - - - - - - - - - - - -- - - - - - - - - - - - - - - -- - - - - - -
    state_obj = State_Variables(comm_pj, SimDet)
    state_obj.dt.dt = delTat

    # Define EP model and solver
    EPparams_pj = {
        "EPmesh": mesh_pj,
        "deg": 4,
        "state_obj": state_obj,
        "d_iso": 10.0,
        "ploc": SimDet["ploc"],
        "AHAid": AHAid_pj,
        "matid": matid_pj,
        "Ischemia": SimDet["Ischemia"],
        # "pacing_timing": SimDet["pacing_timing"],
        "ploc": SimDet["ploc"],
    }

    if "d_iso_pj" in list(SimDet.keys()):
        EPparams_pj.update({"d_iso_pj": SimDet["ploc"]})

    # Define EP model and solver
    EPmodel_ = EPmodel(EPparams_pj)

    return EPmodel_, state_obj


def createEPmodel(IODet, SimDet):

    outputfolder = IODet["outputfolder"]
    folderName = IODet["folderName"] + IODet["caseID"] + "/"
    directory_ep = IODet["directory_ep"]
    casename = IODet["casename"]
    delTat = SimDet["dt"]

    # Read EP data from HDF5 Files
    mesh_ep = Mesh()
    comm_common = mesh_ep.mpi_comm()

    deg_ep = 4
    meshfilename_ep = directory_ep + casename + ".hdf5"
    f = HDF5File(comm_common, meshfilename_ep, "r")
    f.read(mesh_ep, casename, False)
    File(outputfolder + folderName + "mesh_ep.pvd") << mesh_ep

    facetboundaries_ep = MeshFunction("size_t", mesh_ep, 2)
    f.read(facetboundaries_ep, casename + "/" + "facetboundaries")

    # # Set Fiber
    # f0 = Expression(("1.0", "0.0", "0.0"), degree=1)
    # s0 = Expression(("0.0", "1.0", "0.0"), degree=1)
    # n0 = Expression(("0.0", "0.0", "1.0"), degree=1)
    Quadelem_ep = FiniteElement(
        "Quadrature", mesh_ep.ufl_cell(), degree=deg_ep, quad_scheme="default"
    )
    Quadelem_ep._quad_scheme = "default"
    Quad_ep = FunctionSpace(mesh_ep, Quadelem_ep)

    VQuadelem_ep = VectorElement(
        "Quadrature", mesh_ep.ufl_cell(), degree=deg_ep, quad_scheme="default"
    )
    VQuadelem_ep._quad_scheme = "default"

    fiberFS_ep = FunctionSpace(mesh_ep, VQuadelem_ep)

    f0 = Function(fiberFS_ep)
    s0 = Function(fiberFS_ep)
    n0 = Function(fiberFS_ep)

    f.read(f0, casename_ep + "/" + "eF")
    f.read(s0, casename_ep + "/" + "eS")
    f.read(n0, casename_ep + "/" + "eN")

    AHAid_ep = MeshFunction(
        "size_t", mesh_ep, 3, mesh_ep.domains()
    )  # CellFunction("size_t", meshEP.mesh)
    AHAid_ep.set_all(0)

    matid_ep = MeshFunction(
        "size_t", mesh_ep, 3, mesh_ep.domains()
    )  # CellFunction("size_t", meshEP.mesh)
    matid_ep.set_all(0)

    comm_ep = mesh_ep.mpi_comm()

    # Define state variables
    #  - - - - - - - - - - - -- - - - - - - - - - - - - - - -- - - - - - -
    state_obj = State_Variables(comm_ep, SimDet)
    state_obj.dt.dt = delTat

    # Define EP model and solver
    EPparams = {
        "EPmesh": mesh_ep,
        "deg": 4,
        "facetboundaries": facetboundaries_ep,
        "f0": f0,
        "s0": s0,
        "n0": n0,
        "state_obj": state_obj,
        "d_iso": SimDet["d_iso"],
        "d_ani_factor": SimDet["d_ani_factor"],
        "ploc": SimDet["tnode"],
        "AHAid": AHAid_ep,
        "matid": matid_ep,
        "Ischemia": SimDet["Ischemia"],
        # "pacing_timing": SimDet["pacing_timing"],
    }

    # Define EP model and solver
    EPmodel_ = EPmodel(EPparams)

    return EPmodel_, state_obj


def createMEmodel(IODet, SimDet, state_obj):
    #####################################################################
    #  - - - - - - - - - - - -- - - - - - - - - - - - - - - -- - - - - - -
    # Mechanics Mesh
    mesh_me = Mesh()
    comm_me = mesh_me.mpi_comm()
    mesh_me_params = {
        "directory": IODet["directory_me"],
        "casename": IODet["casename"],
        "fibre_quad_degree": 4,
        "outputfolder": IODet["outputfolder"],
        "foldername": IODet["folderName"],
        "state_obj": state_obj,
        "common_communicator": comm_me,
        "MEmesh": mesh_me,
    }

    MEmodel_ = MEmodel(mesh_me_params, SimDet)

    return MEmodel_


def Loading(MEmodel_, export, SimDet):

    if "isLV" in list(SimDet.keys()):
        isLV = SimDet["isLV"]
    else:
        isLV = True

    comm_me = MEmodel_.Mesh.mesh.mpi_comm()
    F_ED = Function(MEmodel_.TF)
    solver_elas = MEmodel_.Solver()

    # Get Unloaded volumes
    V_LV_unload = MEmodel_.GetLVV()
    V_RV_unload = MEmodel_.GetRVV()
    nloadstep = SimDet["nLoadSteps"]

    export.writePV(MEmodel_, 0)

    for lmbda_value in range(0, nloadstep):
        if "V_LV" in list(SimDet["closedloopparam"].keys()):
            V_LV_target = SimDet["closedloopparam"]["V_LV"]
            MEmodel_.LVCavityvol.vol += (V_LV_target - V_LV_unload) / nloadstep
        else:
            MEmodel_.LVCavityvol.vol += 2.0

        if "V_RV" in list(SimDet["closedloopparam"].keys()):
            V_RV_target = SimDet["closedloopparam"]["V_RV"]
            MEmodel_.RVCavityvol.vol += (V_RV_target - V_RV_unload) / nloadstep
        else:
            MEmodel_.RVCavityvol.vol += 2.0

        #  - - - - - - - - - - - -- - - - - - - - - - - - - - - -- - - - - - -
        try:
            solver_elas.solvenonlinear()
        except:
            export.hdf.close()
        #  - - - - - - - - - - - -- - - - - - - - - - - - - - - -- - - - - - -

        if isLV:
            printout(
                "Loading phase step = "
                + str(lmbda_value)
                + " LVV = "
                + str(MEmodel_.GetLVV())
                + " LVP = "
                + str(MEmodel_.GetLVP() * 0.0075),
                comm_me,
            )
            export.printout(
                "Loading phase step = "
                + str(lmbda_value)
                + " LVV = "
                + str(MEmodel_.GetLVV())
            )
        else:
            printout(
                "Loading phase step = "
                + str(lmbda_value)
                + " LVV = "
                + str(MEmodel_.GetLVV())
                + " LVP = "
                + str(MEmodel_.GetLVP() * 0.0075)
                + " RVV = "
                + str(MEmodel_.GetRVV())
                + " RVP = "
                + str(MEmodel_.GetRVP() * 0.0075),
                comm_me,
            )
            export.printout(
                "Loading phase step = "
                + str(lmbda_value)
                + " LVV = "
                + str(MEmodel_.GetLVV())
                + " RVV = "
                + str(MEmodel_.GetRVV())
            )

        export.writePV(MEmodel_, 0)
        export.hdf.write(MEmodel_.GetDisplacement(), "ME/u_loading", lmbda_value)

        F_ED.vector()[:] = (
            project(MEmodel_.GetFmat(), MEmodel_.TF, solver_type="mumps")
            .vector()
            .get_local()[:]
        )


#  - - - - - - - - - - - -- - - - - - - - - - - - - - - -- - - - - - -
if __name__ == "__main__":
    print("Testing...")
    run_BiV_TimedGuccione(IODet=IODetails, SimDet=SimDetails)

#  - - - - - - - - - - - -- - - - - - - - - - - - - - - -- - - - - - -
