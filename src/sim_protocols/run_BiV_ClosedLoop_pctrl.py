import sys, shutil, math
import os as os
import numpy as np
from mpi4py import MPI as pyMPI

import warnings
from ffc.quadrature.deprecation import QuadratureRepresentationDeprecationWarning

warnings.simplefilter("ignore", QuadratureRepresentationDeprecationWarning)


from dolfin import *
import dolfin as dolfin

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

# from ..mechanics.MEmodel_pctrl import MEmodel
from .circ import CLmodel
from .circBiV import CLmodel as CLmodel_biv

# from ..mechanics.volume_ca import MeshModifier

import json

from ..mechanics.JRp import *


def run_BiV_ClosedLoop(IODet, SimDet):
    if "fiber_fspace_deg" in SimDet:
        deg = SimDet["fiber_fspace_deg"]
    else:
        deg = 4
    flags = ["-O3", "-ffast-math", "-march=native"]
    parameters["form_compiler"]["representation"] = "uflacs"
    parameters["form_compiler"]["quadrature_degree"] = deg

    casename = IODet["casename"]
    directory_me = IODet["directory_me"]
    directory_ep = IODet["directory_ep"]
    outputfolder = IODet["outputfolder"]
    folderName = IODet["folderName"] + IODet["caseID"] + "/"
    if "isLV" in list(SimDet.keys()):
        isLV = SimDet["isLV"]
    else:
        isLV = False  # Default
    if "iswaorta" in list(SimDet.keys()):
        iswaorta = SimDet["iswaorta"]
    else:
        iswaorta = False  # Default
    if "isFCH" in list(SimDet.keys()):
        isFCH = SimDet["isFCH"]
    else:
        isFCH = False  # Default
    if "isBiV" in list(SimDet.keys()):
        isBiV = SimDet["isBiV"]
    else:
        isBiV = False  # Default

    delTat = SimDet["dt"]

    #  - - - - - - - - - - - -- - - - - - - - - - - - - - - -- - - - - - -
    # Read EP data from HDF5 Files
    mesh_ep = Mesh()
    comm_common = mesh_ep.mpi_comm()

    meshfilename_ep = directory_ep + casename + "_refine.hdf5"
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
    # export.exportVTKobj("f0.pvd", project(MEmodel_.Mesh.f0, VectorFunctionSpace(MEmodel_.Mesh.mesh, "DG", 0)))

    F_ED = Function(MEmodel_.TF)

    if "AHA_segments" in list(SimDet.keys()):
        AHA_segments = SimDet["AHA_segments"]
    else:
        AHA_segments = [0]

    # Get Unloaded volumes
    V_LV_unload = MEmodel_.GetLVV()
    V_RV_unload = MEmodel_.GetRVV()

    printout("V_LV_unload = " + str(V_LV_unload), comm_me)
    printout("V_RV_unload = " + str(V_RV_unload), comm_me)

    nloadstep = SimDet["nLoadSteps"]

    # Unloading LV to get new reference geometry
    MEmodel_.LVCavityvol.vol = MEmodel_.GetLVV()
    MEmodel_.LVCavitypres.pres = 0.0
    MEmodel_.RVCavitypres.pres = 0.0
    MEmodel_.AortaCavitypres.pres = 0.0

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
    if SimDet.get("EDP"):
        EDP = SimDet["EDP"]
    else:
        EDP = default_params["EDP"]
    LVangle = default_params["LVangle"]
    maxit = default_params["maxit"]
    restol = default_params["restol"]
    drestol = default_params["drestol"]
    EDPtol = default_params["EDPtol"]
    preinc = default_params["preinc"]

    it = 0
    tempfile = File(outputfolder + folderName + "displacement.pvd")
    while 1:
        printout("Loading", comm_me)
        if not SimDet.get("fch_lumped") and not SimDet.get("lv_lumped"):
            MEmodel_.LVCavitypres.pres += (EDP / 0.0075) / nloadstep
        if SimDet.get("aorta_pres"):
            MEmodel_.AortaCavitypres.pres += (EDP * 10.0 / 0.0075) / nloadstep

        if isBiV or isFCH:
            if SimDet.get("fch_fe"):
                MEmodel_.RVCavitypres.pres += (EDP / 0.0075) / nloadstep
                MEmodel_.LACavitypres.pres += (EDP / 5.0 / 0.0075) / nloadstep
                MEmodel_.RACavitypres.pres += (EDP / 2.0 / 0.0075) / nloadstep
            elif SimDet.get("fch_lumped"):
                pass
            else:
                MEmodel_.RVCavitypres.pres += (EDP / 0.0075) / nloadstep
        if not SimDet.get("fch_lumped") and not SimDet.get("lv_lumped"):
            solver_elas.solvenonlinear()

        if it % 1 == 0:
            tempfile << MEmodel_.GetDisplacement()

        export.writePV(MEmodel_, 0)
        export.hdf.write(MEmodel_.GetDisplacement(), "ME/u_loading", it)
        it += 1

        # F_ED.vector()[:] = (
        #    project(
        #        MEmodel_.GetFmat(),
        #        MEmodel_.TF,
        #        solver_type="mumps",
        #        form_compiler_parameters={"representation": "quadrature"},
        #    )
        #    .vector()
        #    .get_local()[:]
        # )

        printout(
            "LV Pressure = "
            + str(MEmodel_.GetLVP() * 0.0075)
            + " LV Vol = "
            + str(MEmodel_.GetLVV())  # GetVolumeComputation()),
            + "RV Pressure = "
            + str(MEmodel_.GetRVP() * 0.0075)
            + "RV Vol = "
            + str(MEmodel_.GetRVV()),  # GetVolumeComputation()),
            comm_me,
        )

        if SimDet.get("fch_lumped") or SimDet.get("lv_lumped"):
            break
        if MEmodel_.LVCavitypres.pres * 0.0075 >= EDP:
            break

    printout("volume = " + str(MEmodel_.GetLVV()), comm_me)
    # import pdb; pdb.set_trace()

    # return
    #  - - - - - - - - - - - -- - - - - - - - - - - - - - - -- - - - - - -
    # Declare communicator based on mpi4py
    # eCC, eRR, eLL, deformedMesh, deformedBoundary = MEmodel_.GetDeformedBasis({})

    # fStrain = MEmodel_.GetFiberstrain(F_ED)
    fStrain_uL = MEmodel_.GetFiberstrainUL()
    #  - - - - - - - - - - - -- - - - - - - - - - - - - - - -- - - - - - -

    # Closed-loop phase
    stop_iter = SimDet["closedloopparam"]["stop_iter"]

    # Systemic circulation

    # Pulmonary circulation
    heart_shape = [isLV, iswaorta, isFCH]
    if all(not x for x in heart_shape):
        # if not isLV and not iswaorta and not isFCH:
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

    P_LV = MEmodel_.GetLVP()  # LVCavitypres.pres
    V_LV = MEmodel_.GetLVV()  # GetVolumeComputation()

    if isBiV or isFCH:
        if SimDet.get("fch_fe"):
            P_LA = MEmodel_.GetLAP()  # set an initial value for P_LA and P_RA
            V_LA = MEmodel_.GetLAV()
            P_RA = MEmodel_.GetRAP()
            V_RA = MEmodel_.GetRAV()

            P_RV = MEmodel_.GetRVP()
            V_RV = MEmodel_.GetRVV()
        elif SimDet.get("fch_lumped"):
            P_LV = EDP / 0.0075
            P_RV = EDP / 0.0075
        else:
            P_RV = MEmodel_.GetRVP()  # LVCavitypres.pres
            V_RV = MEmodel_.GetRVV()  # GetVolumeComputation()

    if isLV or iswaorta:
        if SimDet.get("lv_lumped"):
            CLmodel_ = CLmodel(SimDet)
        else:
            CLmodel_ = CLmodel(SimDet, V_LV)
    elif isBiV or isFCH:
        if SimDet.get("fch_fe"):
            CLmodel_ = CLmodel_biv(SimDet, V_LV, V_RV, V_LA, V_RA)
        elif SimDet.get("fch_lumped"):
            CLmodel_ = CLmodel_biv(SimDet)
        else:
            CLmodel_ = CLmodel_biv(SimDet, V_LV, V_RV)

    it_ = 0

    while 1:
        if state_obj.cycle > stop_iter:
            break
        # if state_obj.t > 100:
        # break

        if not SimDet.get("fch_lumped") and not SimDet.get("lv_lumped"):
            params = {
                "P_LV": P_LV,
                "V_LV": V_LV,
                "t": state_obj.t,
                "delTat": state_obj.dt.dt,
            }
        else:
            params = {
                "t": state_obj.t,
                "delTat": state_obj.dt.dt,
            }

        if isBiV or isFCH:
            if SimDet.get("fch_fe"):
                params.update(
                    {
                        "P_RV": P_RV,
                        "V_RV": V_RV,
                        "P_LA": P_LA,
                        "V_LA": V_LA,
                        "P_RA": P_RA,
                        "V_RA": V_RA,
                    }
                )
            elif SimDet.get("fch_lumped"):
                pass
            else:
                params.update(
                    {
                        "P_RV": P_RV,
                        "V_RV": V_RV,
                    }
                )

        if isLV or iswaorta:
            if SimDet.get("lv_lumped"):
                V_LV, V_LA = CLmodel_.UpdateLVV(params)
            else:
                V_LV = CLmodel_.UpdateLVV(params)

        elif isBiV or isFCH:
            V_LV, V_RV, *extra = CLmodel_.UpdateLVV(params)
            if SimDet.get("fch_fe") or SimDet.get("fch_lumped"):
                V_LA, V_RA = extra

        print_message = (  # todo: Generalize for all cases
            "t = "
            + str(state_obj.t)
            + " VLV = "
            + str(CLmodel_.V_LV)
            + " Psa = "
            + str(CLmodel_.Psa)
            + " PLV = "
            + str(CLmodel_.PLV)
            + " P_LA = "
            + str(CLmodel_.PLA)
            + " Qmv = "
            + str(CLmodel_.Qmv)
            + " Qav = "
            + str(CLmodel_.Qav)
        )
        if isBiV or isFCH:
            if SimDet.get("fch_fe"):
                print_message += " PLA = " + str(P_LA)
            else:
                print_message += " PLA = " + str(CLmodel_.PLA)
            print_message += "\n"
            print_message += " VRV = " + str(V_RV)
            print_message += " Ppa = " + str(CLmodel_.Ppa)
            print_message += " PRV = " + str(CLmodel_.PRV)
            if SimDet.get("fch_fe"):
                print_message += " PRA = " + str(P_RA)
            else:
                print_message += " PRA = " + str(CLmodel_.PRA)

        printout(print_message, comm_me)

        with open(outputfolder + folderName + "output_PV.txt", "a") as f_PV:
            if MPI.rank(comm_me) == 0:
                if isLV or iswaorta:
                    if SimDet.get("lv_lumped"):
                        f_PV.write(
                            f"{state_obj.tstep}, {CLmodel_.V_LV}, {CLmodel_.PLV} \n"
                        )
                    else:
                        f_PV.write(f"{state_obj.tstep}, {V_LV}, {P_LV} \n")
                elif isBiV or isFCH:
                    if SimDet.get("fch_lumped"):
                        f_PV.write(
                            f"{state_obj.tstep}, {V_LV}, {CLmodel_.PLV}, {V_RV}, {CLmodel_.PRV}, {V_LA}, {CLmodel_.PLA}, {V_RA}, {CLmodel_.PRA} \n"
                        )
                    elif SimDet.get("fch_fe"):
                        pass
                    else:
                        f_PV.write(
                            f"{state_obj.t}, {V_LV}, {P_LV}, {V_RV}, {P_RV}, {CLmodel_.V_LA}, {CLmodel_.PLA}, {CLmodel_.V_RA}, {CLmodel_.PRA}, {CLmodel_.V_sv}, {CLmodel_.V_sa}, {CLmodel_.V_ad}, {CLmodel_.V_pv}, {CLmodel_.V_pa} \n"
                        )

        # Newton's solver
        # import pdb; pdb.set_trace()

        def Rp(plv, vlv):
            MEmodel_.LVCavitypres.pres = plv
            if SimDet.get("aorta_pres"):
                MEmodel_.AortaCavitypres.pres = CLmodel_.Psa  # * 5e-1
            solver_elas.solvenonlinear()
            v_t = MEmodel_.GetLVV()
            return v_t - vlv

        def Rp_biv(plv, prv, lvvc, rvvc):
            MEmodel_.LVCavitypres.pres = plv
            MEmodel_.RVCavitypres.pres = prv
            solver_elas.solvenonlinear()
            vlv = MEmodel_.GetLVV()
            vrv = MEmodel_.GetRVV()
            return [vlv - lvvc, vrv - rvvc]

        def Rp_fch(plv, prv, pla, pra, lvvc, rvvc, lavc, ravc):
            MEmodel_.LVCavitypres.pres = plv
            MEmodel_.RVCavitypres.pres = prv
            MEmodel_.LACavitypres.pres = pla
            MEmodel_.RACavitypres.pres = pra
            solver_elas.solvenonlinear()
            vlv = MEmodel_.GetLVV()
            vrv = MEmodel_.GetRVV()
            vla = MEmodel_.GetLAV()
            vra = MEmodel_.GetRAV()
            return np.array([vlv - lvvc, vrv - rvvc, vla - lavc, vra - ravc])

        # Create the Newton solver

        from scipy.optimize import (
            newton,
            fsolve,
            # root_scalar,
            bisect,
            root,
            minimize,
        )

        def Rp_call(plv):
            # scale = 1e6
            return Rp(plv, V_LV)  # / scale

        def Rp_biv_call(plrv):
            plv, prv = plrv
            return Rp_biv(plv, prv, V_LV, V_RV)

        def Rp_fch_call(plrva):
            plv, prv, pla, pra = plrva
            return Rp_fch(plv, prv, pla, pra, V_LV, V_RV, V_LA, V_RA)

        if isLV or iswaorta:
            x0 = [P_LV]
        elif isBiV or isFCH:
            if SimDet.get("fch_fe"):
                x0 = [P_LV, P_RV, P_LA, P_RA]
            else:
                x0 = [P_LV, P_RV]

        if isLV or iswaorta:
            if not SimDet.get("lv_lumped"):
                root1, info, ier, msg = fsolve(
                    Rp_call, x0, xtol=1e-4, factor=0.01, full_output=True
                )

        elif isBiV or isFCH:
            if SimDet.get("fch_fe"):
                root1, info, ier, msg = fsolve(
                    Rp_fch_call, x0, xtol=1e-4, factor=0.01, full_output=True
                )
            elif SimDet.get("fch_lumped"):
                pass
            else:
                root1, info, ier, msg = fsolve(
                    Rp_biv_call, x0, xtol=1e-4, factor=0.01, full_output=True
                )

        with open(outputfolder + folderName + "output_nfev.txt", "a") as nfev:
            if (
                MPI.rank(comm_me) == 0
                and not SimDet.get("fch_lumped")
                and not SimDet.get("lv_lumped")
            ):
                nfev.write(
                    f"t = {state_obj.t}, iter = {info['nfev']}, fun = {info['fvec']}, message = {msg}, t_a = {MEmodel_.GetCt()} \n"
                )

        if not SimDet.get("fch_lumped") and not SimDet.get("lv_lumped"):
            P_LV = root1[0]

        if isBiV or isFCH:
            if SimDet.get("fch_fe"):
                P_RV = root1[1]
                P_LA = root1[2]
                P_RA = root1[3]
            elif SimDet.get("fch_lumped"):
                pass
            else:
                P_RV = root1[1]

        if cnt % 1 == 0:
            tempfile << MEmodel_.GetDisplacement()  # LCL

        state_obj.tstep = state_obj.tstep + state_obj.dt.dt
        state_obj.cycle = math.floor(state_obj.tstep / state_obj.BCL)
        state_obj.t = state_obj.tstep - state_obj.cycle * state_obj.BCL

        MEmodel_.t_a.vector()[:] = state_obj.t

        isrestart = 0
        state_obj.dt.dt = delTat
        # if state_obj.t >= 400.0:
        #     state_obj.dt.dt = 2.0 * delTat

        # Reset phi and r in EP at end of diastole
        if state_obj.t < state_obj.dt.dt:
            EPmodel_.reset()

        printout("Solving FHN", comm_me)
        if not SimDet.get("lv_lumped"):
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
        fstress_DG = project(
            MEmodel_.Getfstress(),
            FunctionSpace(MEmodel_.mesh_me, "DG", 0),
            form_compiler_parameters={"representation": "uflacs"},
        )
        fstress_DG.rename("fstress", "fstress")

        if "probepts" in list(SimDet.keys()):
            probesfstress = Probes(
                x.flatten(), FunctionSpace(MEmodel_.mesh_me, "DG", 1)
            )
            probesfstress(fstress_DG)

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

            probesE_circ_BiV = Probes(
                x.flatten(), FunctionSpace(MEmodel_.mesh_me, "DG", 1)
            )
            probesE_long_BiV = Probes(
                x.flatten(), FunctionSpace(MEmodel_.mesh_me, "DG", 1)
            )
            probesE_radi_BiV = Probes(
                x.flatten(), FunctionSpace(MEmodel_.mesh_me, "DG", 1)
            )

        # postprocess and write
        #  - - - - - - - - - - - -- - - - - - - - - - - - - - - -- - - - - - -

        ## ----------------- Compute Natural Strain -----------------------------------------------------------------------------
        # E_circ_BiV, E_circ_BiV_ = MEmodel_.GetFiberNaturalStrain(
        #    F_ED, eCC, AHA_segments
        # )
        # E_long_BiV, E_long_BiV_ = MEmodel_.GetFiberNaturalStrain(
        #    F_ED, eLL, AHA_segments
        # )
        # E_radi_BiV, E_radi_BiV_ = MEmodel_.GetFiberNaturalStrain(
        #    F_ED, eRR, AHA_segments
        # )
        ## --------------------------------------------------------------------------------------------------------------------
        #
        # E_circ_BiV_DG = project(
        #    E_circ_BiV_,
        #    FunctionSpace(MEmodel_.mesh_me, "DG", 0),
        #    form_compiler_parameters={"representation": "uflacs"},
        # )
        # E_circ_BiV_DG.rename("Ecc", "Ecc")
        # if "probepts" in list(SimDet.keys()):
        #    probesE_circ_BiV(E_circ_BiV_DG)

        # E_long_BiV_DG = project(
        #    E_long_BiV_,
        #    FunctionSpace(MEmodel_.mesh_me, "DG", 0),
        #    form_compiler_parameters={"representation": "uflacs"},
        # )
        # E_long_BiV_DG.rename("Ell", "Ell")
        # if "probepts" in list(SimDet.keys()):
        #    probesE_long_BiV(E_long_BiV_DG)

        # E_radi_BiV_DG = project(
        #    E_radi_BiV_,
        #    FunctionSpace(MEmodel_.mesh_me, "DG", 0),
        #    form_compiler_parameters={"representation": "uflacs"},
        # )
        # E_radi_BiV_DG.rename("Err", "Err")
        # if "probepts" in list(SimDet.keys()):
        #    probesE_radi_BiV(E_radi_BiV_DG)

        # Compute IMP
        imp = project(
            MEmodel_.GetIMP(),
            FunctionSpace(MEmodel_.mesh_me, "DG", 1),
            form_compiler_parameters={"representation": "uflacs"},
        )
        imp.rename("imp", "imp")

        imp2 = project(
            MEmodel_.GetIMP2(),
            FunctionSpace(MEmodel_.mesh_me, "DG", 1),
            form_compiler_parameters={"representation": "uflacs"},
        )
        imp2.rename("imp2", "imp2")

        if "probepts" in list(SimDet.keys()):
            x = np.array(SimDet["probepts"])
            probesIMP = Probes(x.flatten(), FunctionSpace(MEmodel_.mesh_me, "DG", 1))
            probesIMP(imp)

            probesIMP2 = Probes(x.flatten(), FunctionSpace(MEmodel_.mesh_me, "DG", 1))
            probesIMP2(imp2)

            probesIMP3 = Probes(x.flatten(), FunctionSpace(MEmodel_.mesh_me, "CG", 1))
            probesIMP3(MEmodel_.GetP())

            # broadcast from proc 0 to other processes
            rank = comm_me_.Get_rank()
            a = probesIMP3.array()  ## probe will only send to rank =0
            if not rank == 0:
                a = np.empty(len(x))

            comm_me_.Bcast(a, root=0)

        export.writePV(MEmodel_, state_obj.tstep)

        if cnt % SimDet["writeStep"] == 0.0:
            export.writetpt(MEmodel_, state_obj.tstep)
            export.hdf.write(MEmodel_.GetDisplacement(), "ME/u", writecnt)
            export.hdf.write(potential_ref, "ME/potential_ref", writecnt)
            # export.hdf.write(E_circ_BiV_DG, "ME/Ecc", writecnt)
            # export.hdf.write(E_long_BiV_DG, "ME/Ell", writecnt)
            # export.hdf.write(E_radi_BiV_DG, "ME/Err", writecnt)
            export.hdf.write(Eul_fiber_BiV_DG, "ME/Eff", writecnt)
            export.hdf.write(fstress_DG, "ME/fstress", writecnt)
            export.hdf.write(imp, "ME/imp", writecnt)
            export.hdf.write(imp2, "ME/imp2", writecnt)
            export.hdf.write(MEmodel_.GetP(), "ME/imp_constraint", writecnt)

            export.hdf.write(EPmodel_.getphivar(), "EP/phi", writecnt)
            export.hdf.write(EPmodel_.getrvar(), "EP/r", writecnt)
            export.hdf.write(potential_ref, "EP/potential_ref", writecnt)

            writecnt += 1

        if "probepts" in list(SimDet.keys()):
            fIMP = probesIMP.array()
            fIMP2 = probesIMP2.array()
            fIMP3 = probesIMP3.array()
            fStress = probesfstress.array()
            fStrain_vals = probesEul_fiber.array()
            E_circ_BiV = probesE_circ_BiV.array()
            E_long_BiV = probesE_long_BiV.array()
            E_radi_BiV = probesE_radi_BiV.array()

            export.writeIMP(MEmodel_, state_obj.tstep, fIMP)
            export.writeIMP2(MEmodel_, state_obj.tstep, fIMP2)
            export.writeIMP3(MEmodel_, state_obj.tstep, fIMP3)
            export.writefStress(MEmodel_, state_obj.tstep, fStress)
            export.writefStrain(MEmodel_, state_obj.tstep, fStrain_vals)
            export.writeCStrain(MEmodel_, state_obj.tstep, E_circ_BiV)
            export.writeLStrain(MEmodel_, state_obj.tstep, E_long_BiV)
            export.writeRStrain(MEmodel_, state_obj.tstep, E_radi_BiV)

        cnt += 1


#  - - - - - - - - - - - -- - - - - - - - - - - - - - - -- - - - - - -
if __name__ == "__main__":
    print("Testing...")
    run_BiV_TimedGuccione(IODet=IODetails, SimDet=SimDetails)

#  - - - - - - - - - - - -- - - - - - - - - - - - - - - -- - - - - - -
