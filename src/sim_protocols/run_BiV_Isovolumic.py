import sys, shutil, pdb, math
import os as os
import numpy as np
from mpi4py import MPI as pyMPI

import warnings
from ffc.quadrature.deprecation import QuadratureRepresentationDeprecationWarning

warnings.simplefilter("ignore", QuadratureRepresentationDeprecationWarning)


from dolfin import *

# from fenicstools import *

import vtk_py3
import vtk

from ..utils.oops_objects_MRC2 import printout
from ..utils.oops_objects_MRC2 import biventricle_mesh as biv_mechanics_mesh
from ..utils.oops_objects_MRC2 import lv_mesh as lv_mechanics_mesh

from ..utils.oops_objects_MRC2 import State_Variables
from ..utils.oops_objects_MRC2 import update_mesh
from ..utils.oops_objects_MRC2 import exportfiles

from ..utils.mesh_scale_create_fiberFiles import create_EDFibers

from ..ep.EPmodel import EPmodel
from ..mechanics.MEmodel_isov import MEmodel


def run_BiV_Isovolumic(IODet, SimDet):
    deg = 4
    flags = ["-O3", "-ffast-math", "-march=native"]
    parameters["form_compiler"]["representation"] = "quadrature"
    parameters["form_compiler"]["quadrature_degree"] = deg

    casename = IODet["casename"]
    directory_me = IODet["directory_me"]
    directory_ep = IODet["directory_ep"]
    outputfolder = IODet["outputfolder"]
    folderName = IODet["folderName"] + IODet["caseID"] + "/"
    isLV = SimDet["isLV"]

    delTat = SimDet["dt"]

    #  - - - - - - - - - - - -- - - - - - - - - - - - - - - -- - - - - - -
    # Read EP data from HDF5 Files
    mesh_ep = Mesh()
    comm_common = mesh_ep.mpi_comm()

    meshfilename_ep = directory_ep + casename + "_refine.hdf5"
    f = HDF5File(comm_common, meshfilename_ep, "r")
    f.read(mesh_ep, casename, False)

    File(outputfolder + folderName + "mesh_ep.pvd") << mesh_ep

    facetboundaries_ep = MeshFunction("size_t", mesh_ep, 2)
    f.read(facetboundaries_ep, casename + "/" + "facetboundaries")

    matid_ep = MeshFunction("size_t", mesh_ep, mesh_ep.topology().dim())
    AHAid_ep = MeshFunction("size_t", mesh_ep, mesh_ep.topology().dim())

    if f.has_dataset(casename + "/" + "matid"):
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

    EPparams = {
        "EPmesh": mesh_ep,
        "deg": 4,
        "matid": matid_ep,
        "facetboundaries": facetboundaries_ep,
        "f0": f0_ep,
        "s0": s0_ep,
        "n0": n0_ep,
        "state_obj": state_obj,
        "d_iso": SimDet["d_iso"],
        "d_ani_factor": SimDet["d_ani_factor"],
        "AHAid": AHAid_ep,
        "matid": matid_ep,
    }

    if "ploc" in list(SimDet.keys()):
        EPparams.update({"ploc": SimDet["ploc"]})
    if "Ischemia" in list(SimDet.keys()):
        EPparams.update({"Ischemia": SimDet["Ischemia"]})
    if "pacing_timing" in list(SimDet.keys()):
        EPparams.update({"pacing_timing": SimDet["pacing_timing"]})

    # Define EP model and solver
    EPmodel_ = EPmodel(EPparams)
    EpiBCid_ep = EPmodel_.MarkStimulus()

    solver_FHN = EPmodel_.Solver()
    #  - - - - - - - - - - - -- - - - - - - - - - - - - - - -- - - - - - -
    # Mechanics Mesh

    mesh_me = Mesh()
    mesh_me_params = {
        "directory": directory_me,
        "casename": casename,
        "fibre_quad_degree": 4,
        "outputfolder": outputfolder,
        "foldername": folderName,
        "state_obj": state_obj,
        "common_communicator": comm_common,
        "MEmesh": mesh_me,
        "isLV": isLV,
    }

    MEmodel_ = MEmodel(mesh_me_params, SimDet)
    print("Tmax 1 = ", MEmodel_.GetTmax1())
    MEmodel_ = MEmodel(mesh_me_params, SimDet)
    print("Tmax 2 = ", MEmodel_.GetTmax2())
    MEmodel_ = MEmodel(mesh_me_params, SimDet)
    print("Tmax 3 = ", MEmodel_.GetTmax3())

    ###############
    solver_elas = MEmodel_.Solver()
    comm_me = MEmodel_.mesh_me.mpi_comm()

    # Set up export class
    export = exportfiles(comm_me, comm_ep, IODet, SimDet)
    export.exportVTKobj("facetboundaries_ep.pvd", facetboundaries_ep)
    export.exportVTKobj("EpiBCid_ep.pvd", EpiBCid_ep)

    F_ED = Function(MEmodel_.TF)

    if "AHA_segments" in list(SimDet.keys()):
        AHA_segments = SimDet["AHA_segments"]
    else:
        AHA_segments = [0]

    # Get Unloaded volumes
    V_LV_unload = MEmodel_.GetLVV()
    V_RV_unload = MEmodel_.GetRVV()

    nloadstep = SimDet["nLoadSteps"]

    # Unloading LV to get new reference geometry
    isunloading = False
    if "isunloading" in list(SimDet.keys()):
        if SimDet["isunloading"] is True:
            isunloading = True

    if isunloading:
        printout("Start UnLoading", comm_me)

        if "unloadparam" in list(SimDet.keys()):
            unloadparam = SimDet["unloadparam"]
        else:
            unloadparam = {}

        nloadstep_, volinc_ = MEmodel_.unloading(unloadparam)

        printout("Finish UnLoading and Reloading", comm_me)

        export.writePV(MEmodel_, 0)
        export.hdf.write(MEmodel_.mesh_me, "ME/mesh")
        export.hdf.write(EPmodel_.mesh_ep, "EP/mesh")

        MEmodel_.LVCavityvol.vol = MEmodel_.GetLVV()

        for it in np.arange(0, nloadstep_):
            MEmodel_.LVCavityvol.vol += volinc_
            solver_elas.solvenonlinear()
            printout(
                "Pressure = "
                + str(MEmodel_.GetLVP() * 0.0075)
                + " Vol = "
                + str(MEmodel_.GetLVV()),
                comm_me,
            )

            export.writePV(MEmodel_, 0)
            export.hdf.write(MEmodel_.GetDisplacement(), "ME/u_loading", it)

            F_ED.vector()[:] = (
                project(MEmodel_.GetFmat(), MEmodel_.TF, solver_type="mumps")
                .vector()
                .get_local()[:]
            )

    # No unloading
    else:
        export.writePV(MEmodel_, 0)
        export.hdf.write(MEmodel_.mesh_me, "ME/mesh")
        export.hdf.write(EPmodel_.mesh_ep, "EP/mesh")

        for lmbda_value in range(0, nloadstep):
            if "V_LV" in list(SimDet["closedloopparam"].keys()):
                V_LV_target = SimDet["closedloopparam"]["V_LV"]
                MEmodel_.LVCavityvol.vol += (V_LV_target - V_LV_unload) / nloadstep
            else:
                MEmodel_.LVCavityvol.vol += 0.0

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
                    + str(MEmodel_.LVCavityvol.vol),
                    comm_me,
                )
                export.printout(
                    "Loading phase step = "
                    + str(lmbda_value)
                    + " LVV = "
                    + str(MEmodel_.LVCavityvol.vol)
                )
            else:
                printout(
                    "Loading phase step = "
                    + str(lmbda_value)
                    + " LVV = "
                    + str(MEmodel_.LVCavityvol.vol)
                    + " RVV = "
                    + str(MEmodel_.RVCavityvol.vol),
                    comm_me,
                )
                export.printout(
                    "Loading phase step = "
                    + str(lmbda_value)
                    + " LVV = "
                    + str(MEmodel_.LVCavityvol.vol)
                    + " RVV = "
                    + str(MEmodel_.LVCavityvol.vol)
                )

            export.writePV(MEmodel_, 0)
            export.hdf.write(MEmodel_.GetDisplacement(), "ME/u_loading", lmbda_value)

            F_ED.vector()[:] = (
                project(MEmodel_.GetFmat(), MEmodel_.TF, solver_type="mumps")
                .vector()
                .get_local()[:]
            )

    #  - - - - - - - - - - - -- - - - - - - - - - - - - - - -- - - - - - -
    # Declare communicator based on mpi4py
    # comm_me_ = comm_me.tompi4py()
    # eCC, eRR, eLL, deformedMesh, deformedBoundary = MEmodel_.GetDeformedBasis({})

    # fStrain = MEmodel_.GetFiberstrain(F_ED)
    # fStrain_uL = MEmodel_.GetFiberstrainUL()
    #  - - - - - - - - - - - -- - - - - - - - - - - - - - - -- - - - - - -

    # Closed-loop phase
    stop_iter = SimDet["closedloopparam"]["stop_iter"]

    isrestart = 0
    prev_cycle = 0
    cnt = 0

    potential_me = Function(FunctionSpace(MEmodel_.mesh_me, "CG", 1))
    writecnt = 0

    ###############
    Ees1 = 1.0
    Ees2 = 1.0
    Ees3 = 1.0
    #################
    while 1:
        if state_obj.cycle > stop_iter:
            break

        #################################################################
        P_LV = MEmodel_.GetLVP()
        V_LV = MEmodel_.GetLVV()
        if not isLV:
            P_RV = MEmodel_.GetRVP()
            V_RV = MEmodel_.GetRVV()

        state_obj.tstep = state_obj.tstep + state_obj.dt.dt
        state_obj.cycle = math.floor(state_obj.tstep / state_obj.BCL)
        state_obj.t = state_obj.tstep - state_obj.cycle * state_obj.BCL

        # Update deformation at F_ED
        if state_obj.cycle > prev_cycle:
            F_ED.vector()[:] = (
                project(MEmodel_.GetFmat(), MEmodel_.TF).vector().get_local()[:]
            )

        prev_cycle = state_obj.cycle

        MEmodel_.t_a.vector()[:] = state_obj.t

        PLV = P_LV

        if not isLV:
            PRV = P_RV

        printout(
            "Cycle number = "
            + str(state_obj.cycle)
            + " cell time = "
            + str(state_obj.t)
            + " tstep = "
            + str(state_obj.tstep)
            + " dt = "
            + str(state_obj.dt.dt),
            comm_me,
        )
        export.printout(
            "Cycle number = "
            + str(state_obj.cycle)
            + " cell time = "
            + str(state_obj.t)
            + " tstep = "
            + str(state_obj.tstep)
            + " dt = "
            + str(state_obj.dt.dt)
        )

        printout("P_LV = " + str(PLV * 0.0075), comm_me)
        export.printout("P_LV = " + str(PLV))

        V_LV_prev = V_LV
        P_LV_prev = P_LV

        if not isLV:
            V_RV_prev = V_RV
            P_RV_prev = P_RV

            #        imp = project(
            #            MEmodel_.GetIMP(),
            #            FunctionSpace(MEmodel_.mesh_me, "DG", 1),
            #            form_compiler_parameters={"representation": "uflacs"},
            #        )
            #        imp.rename("imp", "imp")
            #
            #        imp2 = project(
            #            MEmodel_.GetIMP2(),
            #            FunctionSpace(MEmodel_.mesh_me, "DG", 1),
            #            form_compiler_parameters={"representation": "uflacs"},
            #        )
            #        imp2.rename("imp2", "imp2")
            #
            #        if "probepts" in list(SimDet.keys()):
            #            x = np.array(SimDet["probepts"])
            #            probesIMP = Probes(x.flatten(), FunctionSpace(MEmodel_.mesh_me, "DG", 1))
            #            probesIMP(imp)
            #
            #            probesIMP2 = Probes(x.flatten(), FunctionSpace(MEmodel_.mesh_me, "DG", 1))
            #            probesIMP2(imp2)
            #
            #            probesIMP3 = Probes(x.flatten(), FunctionSpace(MEmodel_.mesh_me, "CG", 1))
            #            probesIMP3(MEmodel_.GetP())

            # broadcast from proc 0 to other processes
            rank = comm_me_.Get_rank()
            imp_array = probesIMP3.array()
            if not rank == 0:
                imp_array = np.empty(len(x))

            comm_me_.Bcast(imp_array, root=0)
        ###################

        if isLV:
            V_LV = V_LV  # + state_obj.dt.dt*(Qmv - Qav - Qlvad);
        else:
            V_LV = V_LV  # + state_obj.dt.dt*(Qmv - Qav); #- Qlvad);
            V_RV = V_RV  # + state_obj.dt.dt*(Qtv - Qpvv);

        MEmodel_.LVCavityvol.vol = V_LV
        if not isLV:
            Ees1 = 0.0
            Ees2 = 0.0
            Ees3 = 0.0
            MEmodel_.RVCavityvol.vol = V_RV
            MEmodel_.Tmax1.value = Ees1
            MEmodel_.Tmax2.value = Ees2
            MEmodel_.Tmax3.value = Ees3

        printout("V_LV = " + str(V_LV), comm_me)
        export.printout("V_LV = " + str(V_LV))

        if not isLV:
            printout("V_RV = " + str(V_RV), comm_me)
            export.printout("V_RV = " + str(V_RV))

        printout("Solving elasticity", comm_me)
        export.printout("Solving elasticity")
        try:
            solver_elas.solvenonlinear()
            isrestart = 0
            state_obj.dt.dt = delTat

        except RuntimeError:
            export.printout(
                "Restart time step ********************************************* "
            )
            V_LV = V_LV_prev
            P_LV = P_LV_prev
            if not isLV:
                V_RV = V_RV_prev
                P_RV = P_RV_prev

            state_obj.tstep = state_obj.tstep - state_obj.dt.dt
            state_obj.dt.dt += 1.0  # Increase time step
            MEmodel_.Reset()
            EPmodel_.Reset()
            isrestart = 1
            if state_obj.dt.dt < 1e-3:
                export.printout("Smallest time step reached")
                exit(1)
            continue

        if isrestart == 0:
            export.writePV(MEmodel_, state_obj.tstep)
            if not isLV:
                export.writeP(
                    MEmodel_, [0, PLV, 0, 0, 0, PRV, 0, 0, 0, 0], state_obj.tstep
                )
            else:
                export.writeP(MEmodel_, [0, PLV, 0, 0], state_obj.tstep)

        # Reset phi and r in EP at end of diastole
        if state_obj.t < state_obj.dt.dt:
            EPmodel_.reset()

        printout("Solving FHN", comm_me)
        solver_FHN.solvenonlinear()
        if isrestart == 0:
            MEmodel_.UpdateVar()  # For damping
            EPmodel_.UpdateVar()

        # Interpolate phi to mechanics mesh
        potential_ref = EPmodel_.interpolate_potential_ep2me_phi(
            V_me=Function(FunctionSpace(MEmodel_.mesh_me, "CG", 1))
        )
        potential_ref.rename("v_ref", "v_ref")

        potential_me.vector()[:] = potential_ref.vector().get_local()[:]

        #  - - - - - - - - - - - -- - - - - - - - - - - - - - - -- - - - - - -
        if MPI.rank(comm_ep) == 0:
            print("UPdating isActiveField and tInitiationField")
        MEmodel_.activeforms.update_activationTime(
            potential_n=potential_me, comm=comm_me
        )

        F_n = MEmodel_.GetFmat()
        #        fstress_DG = project(
        #            MEmodel_.Getfstress(),
        #            FunctionSpace(MEmodel_.mesh_me, "DG", 0),
        #            form_compiler_parameters={"representation": "uflacs"},
        #        )
        #        fstress_DG.rename("fstress", "fstress")
        #        if "probepts" in list(SimDet.keys()):
        #            probesfstress = Probes(
        #                x.flatten(), FunctionSpace(MEmodel_.mesh_me, "DG", 1)
        #            )
        #            probesfstress(fstress_DG)
        #
        #        Eul_fiber_BiV_DG = project(
        #            fStrain_uL,
        #            FunctionSpace(MEmodel_.mesh_me, "DG", 0),
        #            form_compiler_parameters={"representation": "uflacs"},
        #        )
        #        Eul_fiber_BiV_DG.rename("Eff", "Eff")
        #        if "probepts" in list(SimDet.keys()):
        #            probesEul_fiber = Probes(
        #                x.flatten(), FunctionSpace(MEmodel_.mesh_me, "DG", 1)
        #            )
        #            probesEul_fiber(Eul_fiber_BiV_DG)
        #
        #            probesE_circ_BiV = Probes(
        #                x.flatten(), FunctionSpace(MEmodel_.mesh_me, "DG", 1)
        #            )
        #            probesE_long_BiV = Probes(
        #                x.flatten(), FunctionSpace(MEmodel_.mesh_me, "DG", 1)
        #            )
        #            probesE_radi_BiV = Probes(
        #                x.flatten(), FunctionSpace(MEmodel_.mesh_me, "DG", 1)
        #            )

        # postprocess and write
        #  - - - - - - - - - - - -- - - - - - - - - - - - - - - -- - - - - - -

        ## ----------------- Compute Natural Strain -----------------------------------------------------------------------------
        #        E_circ_BiV, E_circ_BiV_ = MEmodel_.GetFiberNaturalStrain(
        #            F_ED, eCC, AHA_segments
        #        )
        #        E_long_BiV, E_long_BiV_ = MEmodel_.GetFiberNaturalStrain(
        #            F_ED, eLL, AHA_segments
        #        )
        #        E_radi_BiV, E_radi_BiV_ = MEmodel_.GetFiberNaturalStrain(
        #            F_ED, eRR, AHA_segments
        #        )
        ## --------------------------------------------------------------------------------------------------------------------

        #        E_circ_BiV_DG = project(
        #            E_circ_BiV_,
        #            FunctionSpace(MEmodel_.mesh_me, "DG", 0),
        #            form_compiler_parameters={"representation": "uflacs"},
        #        )
        #        E_circ_BiV_DG.rename("Ecc", "Ecc")
        #        if "probepts" in list(SimDet.keys()):
        #            probesE_circ_BiV(E_circ_BiV_DG)
        #
        #        E_long_BiV_DG = project(
        #            E_long_BiV_,
        #            FunctionSpace(MEmodel_.mesh_me, "DG", 0),
        #            form_compiler_parameters={"representation": "uflacs"},
        #        )
        #        E_long_BiV_DG.rename("Ell", "Ell")
        #        if "probepts" in list(SimDet.keys()):
        #            probesE_long_BiV(E_long_BiV_DG)
        #
        #        E_radi_BiV_DG = project(
        #            E_radi_BiV_,
        #            FunctionSpace(MEmodel_.mesh_me, "DG", 0),
        #            form_compiler_parameters={"representation": "uflacs"},
        #        )
        #        E_radi_BiV_DG.rename("Err", "Err")
        #        if "probepts" in list(SimDet.keys()):
        #            probesE_radi_BiV(E_radi_BiV_DG)

        if cnt % SimDet["writeStep"] == 0.0:
            export.writetpt(MEmodel_, state_obj.tstep)
            export.hdf.write(MEmodel_.GetDisplacement(), "ME/u", writecnt)
            export.hdf.write(potential_ref, "ME/potential_ref", writecnt)
            #            export.hdf.write(E_circ_BiV_DG, "ME/Ecc", writecnt)
            #            export.hdf.write(E_long_BiV_DG, "ME/Ell", writecnt)
            #            export.hdf.write(E_radi_BiV_DG, "ME/Err", writecnt)
            #            export.hdf.write(Eul_fiber_BiV_DG, "ME/Eff", writecnt)
            #            export.hdf.write(fstress_DG, "ME/fstress", writecnt)
            #            export.hdf.write(imp, "ME/imp", writecnt)
            #            export.hdf.write(imp2, "ME/imp2", writecnt)
            #            export.hdf.write(MEmodel_.GetP(), "ME/imp_constraint", writecnt)
            #
            #            export.hdf.write(EPmodel_.getphivar(), "EP/phi", writecnt)
            #            export.hdf.write(EPmodel_.getrvar(), "EP/r", writecnt)
            #            export.hdf.write(potential_ref, "EP/potential_ref", writecnt)

            writecnt += 1

        #        if "probepts" in list(SimDet.keys()):
        #            fIMP = probesIMP.array()
        #            fIMP2 = probesIMP2.array()
        #            fIMP3 = probesIMP3.array()
        #            fStress = probesfstress.array()
        #            fStrain_vals = probesEul_fiber.array()
        #            E_circ_BiV = probesE_circ_BiV.array()
        #            E_long_BiV = probesE_long_BiV.array()
        #            E_radi_BiV = probesE_radi_BiV.array()
        #
        #            export.writeIMP(MEmodel_, state_obj.tstep, fIMP)
        #            export.writeIMP2(MEmodel_, state_obj.tstep, fIMP2)
        #            export.writeIMP3(MEmodel_, state_obj.tstep, fIMP3)
        #            export.writefStress(MEmodel_, state_obj.tstep, fStress)
        #            export.writefStrain(MEmodel_, state_obj.tstep, fStrain_vals)
        #            export.writeCStrain(MEmodel_, state_obj.tstep, E_circ_BiV)
        #            export.writeLStrain(MEmodel_, state_obj.tstep, E_long_BiV)
        #            export.writeRStrain(MEmodel_, state_obj.tstep, E_radi_BiV)

        cnt += 1


#  - - - - - - - - - - - -- - - - - - - - - - - - - - - -- - - - - - -
if __name__ == "__main__":
    print("Testing...")
    run_BiV_TimedGuccione(IODet=IODetails, SimDet=SimDetails)

#  - - - - - - - - - - - -- - - - - - - - - - - - - - - -- - - - - - -
