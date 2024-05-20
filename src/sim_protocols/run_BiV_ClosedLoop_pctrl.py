import sys, shutil, math
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

from ..mechanics.MEmodel3 import MEmodel
from .circ import CLmodel

# from ..mechanics.volume_ca import MeshModifier

import json

from ..mechanics.JRp import *


def run_BiV_ClosedLoop(IODet, SimDet):
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

    #        if(isunloading):
    #
    #            printout("Start UnLoading", comm_me)
    #            #V_LV_target = MEmodel_.GetLVV()
    #
    #            if("unloadparam" in SimDet.keys()):
    #                unloadparam = SimDet["unloadparam"]
    #            else:
    #                unloadparam = {};
    #
    #            nloadstep_, volinc_, V_LV_target, solver_elas = MEmodel_.unloading(unloadparam)
    #            printout("Target EDV = " + str(V_LV_target), comm_me)
    #
    #            printout("Finish UnLoading and Reloading", comm_me)
    #
    #
    #            export.writePV(MEmodel_, 0);
    #            export.hdf.write(MEmodel_.mesh_me, "ME/mesh")
    #            export.hdf.write(EPmodel_.mesh_ep, "EP/mesh")
    #
    #            MEmodel_.LVCavityvol.vol = MEmodel_.GetLVV()
    #            V_LV_unload = MEmodel_.GetLVV()
    #
    #            for it in np.arange(0,nloadstep):
    #
    #                MEmodel_.LVCavityvol.vol += (V_LV_target - V_LV_unload)/nloadstep
    #                solver_elas.solvenonlinear()
    #                printout("Pressure = " +  str(MEmodel_.GetLVP()*0.0075) +  " Vol = " + str(MEmodel_.GetLVV()), comm_me)
    #
    #                export.writePV(MEmodel_, 0);
    #                export.hdf.write(MEmodel_.GetDisplacement(), "ME/u_loading", it)
    #
    #                F_ED.vector()[:] = project(MEmodel_.GetFmat(), MEmodel_.TF, solver_type='mumps').vector().array()[:]
    #
    #        # No unloading
    #        else:
    #            #export.writePV(MEmodel_, 0);
    #            export.hdf.write(MEmodel_.mesh_me, "ME/mesh")
    #            export.hdf.write(EPmodel_.mesh_ep, "EP/mesh")
    #    #        for lmbda_value in range(0, nloadstep):
    #            lmbda_value = 0
    #            while (1):
    #
    #                if("V_LV" in SimDet["closedloopparam"].keys()):
    #                    V_LV_target = SimDet["closedloopparam"]["V_LV"]
    #                    MEmodel_.LVCavityvol.vol += (V_LV_target - V_LV_unload)/nloadstep
    #                    MEmodel_.LVCavitypres.pres += 20.0
    #                else:
    #                    MEmodel_.LVCavityvol.vol += 2.0
    #
    #                if("V_RV" in SimDet["closedloopparam"].keys()):
    #                    V_RV_target = SimDet["closedloopparam"]["V_RV"]
    #                    MEmodel_.RVCavityvol.vol += (V_RV_target - V_RV_unload)/nloadstep
    #                else:
    #                    MEmodel_.RVCavityvol.vol += 2.0
    #
    #
    #                #  - - - - - - - - - - - -- - - - - - - - - - - - - - - -- - - - - - -
    #                try:
    #                    solver_elas.solvenonlinear()
    #                except:
    #                    export.hdf.close()
    #                #  - - - - - - - - - - - -- - - - - - - - - - - - - - - -- - - - - - -
    #                printout("ME LVV = " + str(MEmodel_.GetLVV()), comm_me)
    #
    #                if(isLV):
    #                    printout("Loading phase step = " + str(lmbda_value) + " LVV = " + str(MEmodel_.GetLVV()) \
    #    #                                                                      " LVP = " + str(MEmodel_.GetLVP()*0.0075)  \
    #                                                                          , comm_me)
    #                    export.printout("Loading phase step = " + str(lmbda_value) + " LVV = " + str(MEmodel_.GetLVV()))
    #                else:
    #                    printout("Loading phase step = " + str(lmbda_value) + " LVV = " + str(MEmodel_.GetLVV()) + \
    #    #                                                                      " LVP = " + str(MEmodel_.GetLVP()*0.0075) + \
    #                                                                          " RVV = " + str(MEmodel_.GetRVV()) + \
    #                                                                          " RVP = " + str(MEmodel_.GetRVP()*0.0075)  \
    #                                                                          , comm_me)
    #                    export.printout("Loading phase step = " + str(lmbda_value) + " LVV = " + str(MEmodel_.GetLVV())+ \
    #                                                                                 " RVV = " + str(MEmodel_.GetRVV()))
    #
    #
    #                #export.writePV(MEmodel_, 0);
    #                export.hdf.write(MEmodel_.GetDisplacement(), "ME/u_loading", lmbda_value)
    #
    #                F_ED.vector()[:] = project(MEmodel_.GetFmat(), MEmodel_.TF, solver_type='mumps').vector().array()[:]
    #                printout("LVP = " + str(MEmodel_.LVCavitypres.pres), comm_me)
    #
    #                if MEmodel_.GetLVV() > V_LV_target:
    #                    break

    if "isunloadingonly" in list(SimDet.keys()):
        if SimDet["isunloadingonly"] is True:
            export.hdf.close()
            exit()

    # Unloading LV to get new reference geometry
    MEmodel_.LVCavityvol.vol = MEmodel_.GetLVV()
    MEmodel_.LVCavitypres.pres = 0.0

    # export.writePV(MEmodel_, 0);
    export.hdf.write(MEmodel_.mesh_me, "ME/mesh")
    export.hdf.write(EPmodel_.mesh_ep, "EP/mesh")

    default_params = {
        "EDP": 12.0,
        "maxit": 20,
        "restol": 1e-3,
        "drestol": 1e-4,
        "EDPtol": 1e-1,
        "preinc": 1,
        "LVangle": [60, -60],
    }
    # default_params.update(params)

    EDP = default_params["EDP"]
    LVangle = default_params["LVangle"]
    maxit = default_params["maxit"]
    restol = default_params["restol"]
    drestol = default_params["drestol"]
    EDPtol = default_params["EDPtol"]
    preinc = default_params["preinc"]

    it = 0
    while 1:
        MEmodel_.LVCavitypres.pres += 100.0

        solver_elas.solvenonlinear()

        # export.writePV(MEmodel_, 0);
        # export.hdf.write(MEmodel_.GetDisplacement(), "ME/u_loading", it)
        # it += 1

        #        F_ED.vector()[:] = (
        #            project(MEmodel_.GetFmat(), MEmodel_.TF, solver_type="mumps")
        #            .vector()
        #            .get_local()[:]
        #        )

        if MEmodel_.LVCavitypres.pres * 0.0075 >= EDP:
            break

    if "isunloadingonly" in list(SimDet.keys()):
        if SimDet["isunloadingonly"] is True:
            export.hdf.close()
            exit()

    prev_disp = MEmodel_.GetDisplacement()
    File(outputfolder + folderName + "prev_disp.pvd") << prev_disp
    printout("volume = " + str(MEmodel_.GetVolumeComputation()), comm_me)

    #  - - - - - - - - - - - -- - - - - - - - - - - - - - - -- - - - - - -
    # Declare communicator based on mpi4py
    eCC, eRR, eLL, deformedMesh, deformedBoundary = MEmodel_.GetDeformedBasis({})

    # fStrain = MEmodel_.GetFiberstrain(F_ED)
    fStrain_uL = MEmodel_.GetFiberstrainUL()
    #  - - - - - - - - - - - -- - - - - - - - - - - - - - - -- - - - - - -

    # Closed-loop phase
    stop_iter = SimDet["closedloopparam"]["stop_iter"]

    # Systemic circulation

    # Pulmonary circulation
    if not isLV:
        Cpa = SimDet["closedloopparam"]["Cpa"]
        Cpv = SimDet["closedloopparam"]["Cpv"]
        Vpa0 = SimDet["closedloopparam"]["Vpa0"]
        Vpv0 = SimDet["closedloopparam"]["Vpv0"]
        Rpv = SimDet["closedloopparam"]["Rpv"]
        Rtv = SimDet["closedloopparam"]["Rtv"]
        Rpa = SimDet["closedloopparam"]["Rpa"]
        Rpvv = SimDet["closedloopparam"]["Rpvv"]
        V_pv = SimDet["closedloopparam"]["V_pv"]
        V_pa = SimDet["closedloopparam"]["V_pa"]
        V_RA = SimDet["closedloopparam"]["V_RA"]

    isrestart = 0
    prev_cycle = 0
    cnt = 0

    Qtv = 0
    Qpa = 0
    Qpv = 0
    Qpvv = 0
    Qlvad = 0
    Qlara = 0

    if "Q_tv" in list(SimDet["closedloopparam"].keys()):
        Qtv = SimDet["closedloopparam"]["Q_tv"]
    if "Q_pa" in list(SimDet["closedloopparam"].keys()):
        Qpa = SimDet["closedloopparam"]["Q_pa"]
    if "Q_pv" in list(SimDet["closedloopparam"].keys()):
        Qpv = SimDet["closedloopparam"]["Q_pv"]
    if "Q_pvv" in list(SimDet["closedloopparam"].keys()):
        Qpvv = SimDet["closedloopparam"]["Q_pvv"]
    if "Q_lvad" in list(SimDet["closedloopparam"].keys()):
        Qlvad = SimDet["closedloopparam"]["Q_lvad"]
    if "Q_lara" in list(SimDet["closedloopparam"].keys()):
        Qlara = SimDet["closedloopparam"]["Q_lara"]

    # Parameters for LVAD #############################################
    LVADrpm = 0
    LVADscale = 0
    if "Q_lvad_rpm" in list(SimDet["closedloopparam"].keys()):
        LVADrpm = SimDet["closedloopparam"]["Q_lvad_rpm"]
    if "Q_lvad_scale" in SimDet["closedloopparam"].keys():
        LVADscale = SimDet["closedloopparam"]["Q_lvad_scale"]
    if "Q_lvad_characteristic" in list(SimDet["closedloopparam"].keys()):
        QLVADFn = SimDet["closedloopparam"]["Q_lvad_characteristic"]

    Qlad = 0
    Qlcx = 0

    # Parameters for Shunt #############################################
    Shuntscale = 0.0
    Rsh = 1e9
    if "Shunt_scale" in list(SimDet["closedloopparam"].keys()):
        Shuntscale = SimDet["closedloopparam"]["Shunt_scale"]
    if "Rsh" in list(SimDet["closedloopparam"].keys()):
        Rsh = SimDet["closedloopparam"]["Rsh"]

    potential_me = Function(FunctionSpace(MEmodel_.mesh_me, "CG", 1))
    writecnt = 0

    P_LV = MEmodel_.LVCavitypres.pres
    V_LV = MEmodel_.GetVolumeComputation()

    CLmodel_ = CLmodel(SimDet, V_LV)

    dict_PV = []

    it_ = 0
    while 1:
        if state_obj.cycle > stop_iter:
            break

        # Time varying elastance function for LA and RA ##################
        # def et(t, Tmax, tau):
        # if (t <= 1.5*Tmax):
        # out = 0.5*(math.sin((math.pi/Tmax)*t - math.pi/2) + 1);
        # else:
        # out = 0.5*math.exp((-t + (1.5*Tmax))/tau);
        # return out

        params = {
            "P_LV": P_LV,
            "V_LV": V_LV,
            "t": state_obj.t,
            "delTat": state_obj.dt.dt,
        }

        V_LV = CLmodel_.UpdateLVV(params)

        printout(
            "t = "
            + str(state_obj.t)
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
        dict_PV.append((state_obj.t, V_LV, P_LV))

        # prev displacement
        prev_displacement = MEmodel_.GetDisplacement()
        File(outputfolder + folderName + "prev_disp.pvd") << prev_displacement

        # Newton's solver
        tol = 1e-5  # Tolerance for convergence
        max_iter = 200  # Maximum number of iteration

        def estpres(P_LV):  # initial guess
            return 1.005 * P_LV

        def Jf(P_LV):
            MEmodel_.LVCavitypres.pres = P_LV
            solver_elas.solvenonlinear()
            est_fe_v1 = MEmodel_.GetVolumeComputation()

            P_LV2 = estpres(P_LV)

            MEmodel_.LVCavitypres.pres = P_LV2
            solver_elas.solvenonlinear()
            est_fe_v2 = MEmodel_.GetVolumeComputation()

            return (est_fe_v2 - est_fe_v1) / (P_LV2 - P_LV)

        def Rp(P_LV, V_LV):  # V_LV is from circulatory model
            MEmodel_.LVCavitypres.pres = P_LV
            solver_elas.solvenonlinear()

            v_t = MEmodel_.GetVolumeComputation()

            return v_t - V_LV

        # Create the Newton solver
        for iter in range(max_iter):
            # Compute the residual and Jacobian
            J = Jf(P_LV)
            F = Rp(P_LV, V_LV)

            # Solve for the update
            du = -F / J

            # Update the solution
            P_LV += du

            # Check for convergence
            if abs(F) < tol and abs(du) < tol:
                break

        #        printout("P_LV = "
        #            + str(P_LV)
        #            + " V_LV circ = "
        #            + str(V_LV)
        #            + " Rp is: "
        #            + str(F)
        #            + " J is: "
        #            + str(J),
        #            comm_me,
        #        )

        # new_displacement = MEmodel_.GetDisplacement()
        # File(outputfolder + folderName + "new_disp.pvd") << new_displacement

        # a_n = prev_displacement.vector().get_local()
        # b_n = new_displacement.vector().get_local()

        # c_n = Function(prev_displacement.function_space())
        ## sub_ab = a_n #- b_n
        # c_n.vector().set_local(a_n)
        # as_backend_type(c_n.vector()).vec().ghostUpdate()
        # d_n = MEmodel_.RobinC(c_n)
        # printout("d_n if applicable = " + str(d_n.vector().get_local()), comm_me)

        if cnt % SimDet["writeStep"] == 0.0:
            export.hdf.write(MEmodel_.GetDisplacement(), "ME/u_loading", writecnt)
            # export.hdf.write(c_n, "ME/u_diff", writecnt)
            # writecnt += 1

        state_obj.tstep = state_obj.tstep + state_obj.dt.dt
        state_obj.cycle = math.floor(state_obj.tstep / state_obj.BCL)
        state_obj.t = state_obj.tstep - state_obj.cycle * state_obj.BCL

        MEmodel_.t_a.vector()[:] = state_obj.t

        isrestart = 0
        state_obj.dt.dt = delTat

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

        #        F_n = MEmodel_.GetFmat()
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
        Eul_fiber_BiV_DG = project(
            fStrain_uL,
            FunctionSpace(MEmodel_.mesh_me, "DG", 0),
            form_compiler_parameters={"representation": "uflacs"},
        )

        Eul_fiber_BiV_DG.rename("Eff", "Eff")
        if "probepts" in list(SimDet.keys()):
            # x = np.array(SimDet["probepts"])
            probesEul_fiber = Probes(
                x.flatten(), FunctionSpace(MEmodel_.mesh_me, "DG", 1)
            )
            probesEul_fiber(Eul_fiber_BiV_DG)

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
        E_circ_BiV, E_circ_BiV_ = MEmodel_.GetFiberNaturalStrain(
            F_ED, eCC, AHA_segments
        )
        E_long_BiV, E_long_BiV_ = MEmodel_.GetFiberNaturalStrain(
            F_ED, eLL, AHA_segments
        )
        #        E_radi_BiV, E_radi_BiV_ = MEmodel_.GetFiberNaturalStrain(
        #            F_ED, eRR, AHA_segments
        #        )
        #        ## --------------------------------------------------------------------------------------------------------------------
        #
        E_circ_BiV_DG = project(
            E_circ_BiV_,
            FunctionSpace(MEmodel_.mesh_me, "DG", 0),
            form_compiler_parameters={"representation": "uflacs"},
        )
        E_circ_BiV_DG.rename("Ecc", "Ecc")
        if "probepts" in list(SimDet.keys()):
            probesE_circ_BiV(E_circ_BiV_DG)

        E_long_BiV_DG = project(
            E_long_BiV_,
            FunctionSpace(MEmodel_.mesh_me, "DG", 0),
            form_compiler_parameters={"representation": "uflacs"},
        )
        E_long_BiV_DG.rename("Ell", "Ell")
        if "probepts" in list(SimDet.keys()):
            probesE_long_BiV(E_long_BiV_DG)

        #
        #        E_radi_BiV_DG = project(
        #            E_radi_BiV_,
        #            FunctionSpace(MEmodel_.mesh_me, "DG", 0),
        #            form_compiler_parameters={"representation": "uflacs"},
        #        )
        #        E_radi_BiV_DG.rename("Err", "Err")
        #        if "probepts" in list(SimDet.keys()):
        #            probesE_radi_BiV(E_radi_BiV_DG)
        #
        if cnt % SimDet["writeStep"] == 0.0:
            # export.writetpt(MEmodel_, state_obj.tstep)
            export.hdf.write(MEmodel_.GetDisplacement(), "ME/u", writecnt)
            # export.hdf.write(potential_ref, "ME/potential_ref", writecnt)
            export.hdf.write(E_circ_BiV_DG, "ME/Ecc", writecnt)
            export.hdf.write(E_long_BiV_DG, "ME/Ell", writecnt)
            # export.hdf.write(E_radi_BiV_DG, "ME/Err", writecnt)
            # export.hdf.write(Eul_fiber_BiV_DG, "ME/Eff", writecnt)
            # export.hdf.write(fstress_DG, "ME/fstress", writecnt)
            ## export.hdf.write(imp, "ME/imp", writecnt)
            ## export.hdf.write(imp2, "ME/imp2",  writecnt)
            # export.hdf.write(MEmodel_.GetP(), "ME/imp_constraint", writecnt)

            # export.hdf.write(EPmodel_.getphivar(), "EP/phi", writecnt)
            # export.hdf.write(EPmodel_.getrvar(), "EP/r", writecnt)
            # export.hdf.write(potential_ref, "EP/potential_ref", writecnt)

            writecnt += 1

        if "probepts" in list(SimDet.keys()):
            #            fIMP = probesIMP.array()
            #            fIMP2 = probesIMP2.array()
            #            fIMP3 = probesIMP3.array()
            #            fStress = probesfstress.array()
            fStrain_vals = probesEul_fiber.array()
            #            E_circ_BiV = probesE_circ_BiV.array()
            #            E_long_BiV = probesE_long_BiV.array()
            #            E_radi_BiV = probesE_radi_BiV.array()
            #
            #            export.writeIMP(MEmodel_, state_obj.tstep, fIMP)
            #            export.writeIMP2(MEmodel_, state_obj.tstep, fIMP2)
            #            export.writeIMP3(MEmodel_, state_obj.tstep, fIMP3)
            #            export.writefStress(MEmodel_, state_obj.tstep, fStress)
            export.writefStrain(MEmodel_, state_obj.tstep, fStrain_vals)
        #            export.writeCStrain(MEmodel_, state_obj.tstep, E_circ_BiV)
        #            export.writeLStrain(MEmodel_, state_obj.tstep, E_long_BiV)
        #            export.writeRStrain(MEmodel_, state_obj.tstep, E_radi_BiV)

        cnt += 1

    with open("dict_PV.json", "w") as json_f:
        json.dump(dict_PV, json_f)


#  - - - - - - - - - - - -- - - - - - - - - - - - - - - -- - - - - - -
if __name__ == "__main__":
    print("Testing...")
    run_BiV_TimedGuccione(IODet=IODetails, SimDet=SimDetails)

#  - - - - - - - - - - - -- - - - - - - - - - - - - - - -- - - - - - -
