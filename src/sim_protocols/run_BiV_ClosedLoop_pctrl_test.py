import sys, shutil, math
import os as os
import numpy as np
from mpi4py import MPI as pyMPI

import warnings
from ffc.quadrature.deprecation import QuadratureRepresentationDeprecationWarning

warnings.simplefilter("ignore", QuadratureRepresentationDeprecationWarning)


from dolfin import *
import dolfin as dolfin
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
from ..utils.oops_objects_MRC2 import json_serialize

#from ..ep.EPmodel_basic_test import EPmodel
from ..ep.EPmodel_cpp import EPmodel

from ..mechanics.MEmodel3 import MEmodel

# from ..mechanics.MEmodel_pctrl import MEmodel
#from .circ import CLmodel
#from .circBiV import CLmodel as CLmodel_biv

# from ..mechanics.volume_ca import MeshModifier
from ..postprocessing.postprocessdatalib2 import *

import json

from ..mechanics.JRp import *

from .createEPmodel import createEPmodel
from .createPJmodel import createPJmodel
from .createMEmodel import createMEmodel
from .coupleEPandPJ import coupleEPandPJ
from .coupleMEandCirc import coupleMEandCirc


def run_BiV_ClosedLoop(IODet, SimDet):
    if "fiber_fspace_deg" in SimDet:
        deg = SimDet["fiber_fspace_deg"]
    else:
        deg = 4
    flags = ["-O3", "-ffast-math", "-march=native"]
    parameters["form_compiler"]["representation"] = "uflacs"
    parameters["form_compiler"]["quadrature_degree"] = deg

    outputfolder = IODet["outputfolder"]
    folderName = IODet["folderName"] + IODet["caseID"] + "/"

    if "HomogenousActivation" in list(SimDet["GiccioneParams"].keys()):
        ishomo = SimDet["GiccioneParams"]["HomogenousActivation"]
    else:
        ishomo = True

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
    if "isPJ" in list(SimDet.keys()):
        isPJ = SimDet["isPJ"]
    else:
        isPJ = False

    if "AHA_segments" in list(SimDet.keys()):
        AHA_segments = SimDet["AHA_segments"]
    else:
        AHA_segments = [0]

    delTat = SimDet["dt"]

    if not ishomo:
        intensity = SimDet["current_intensity"]

    if isPJ:
        pj_intensity = SimDet["PJ_current_intensity"]

    # Define Purkinje terminal nodes
    if isPJ:
        if "pj_tnodes" in list(SimDet.keys()):
            if isinstance(SimDet["pj_tnodes"], str):
                pj_t_nodes = np.genfromtxt(os.path.join(IODet["directory_pj"], SimDet["pj_tnodes"]), delimiter=',')
            elif isinstance(SimDet["pj_tnodes"], list):
                pj_t_nodes = np.array(SimDet["pj_tnodes"])

            SimDet.update({"pj_tnodes": list(pj_t_nodes)})

        # Create Purkinje PJ model
        EPmodel_pj, state_obj_pj, PJparams = createPJmodel(IODet, SimDet)
        solver_FHN_pj = EPmodel_pj.Solver()

    # Create Tissue EP model
    EPmodel_ep, state_obj, EPparams = createEPmodel(IODet, SimDet)
    solver_FHN_ep = EPmodel_ep.Solver()
    comm_common = EPmodel_ep.mesh.mpi_comm()
    comm_ep = EPmodel_ep.mesh.mpi_comm()

    # Create Tissue ME model
    MEmodel_, state_obj_me, MEparams = createMEmodel(IODet, SimDet)
    solver_ME = MEmodel_.Solver()
    comm_me = MEmodel_.mesh.mpi_comm()

    # Set up export class
    export = exportfiles(comm_me, comm_ep, IODet, SimDet)
    export.exportVTKobj("facetboundaries_me.pvd", MEmodel_.facetboundaries_me)


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
    export.hdf.write(MEmodel_.mesh, "ME/mesh")
    export.hdf.write(EPmodel_ep.mesh, "EP/mesh")
    if isPJ:
        export.hdf.write(EPmodel_pj.mesh, "PJ/mesh")

    ## Dump input file
    export.dump_input_file()

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
    #MEmodel_.isspringon = 1.0

    # Coupling EP and PJ model
    if isPJ:
        coupleEPandPJ_ = coupleEPandPJ(EPmodel_ep, EPmodel_pj, EPparams, PJparams, state_obj)

    # Coupling ME and Circulatory model
    coupleMEandCL_ = coupleMEandCirc(MEmodel_, MEparams, SimDet, state_obj)
    CLmodel_ = coupleMEandCL_.CLmodel


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
                if SimDet.get("RVEDPfactor"):
                    MEmodel_.RVCavitypres.pres += (SimDet["RVEDPfactor"] * EDP / 0.0075) / nloadstep
                else:
                    MEmodel_.RVCavitypres.pres += (0.75 * EDP / 0.0075) / nloadstep
        if not SimDet.get("fch_lumped") and not SimDet.get("lv_lumped"):
            solver_ME.solvenonlinear()

        export.writePV(MEmodel_, 0)
        export.hdf.write(MEmodel_.GetDisplacement(), "ME/u_loading", it)
        it += 1


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

    #printout("volume = " + str(MEmodel_.GetLVV()), comm_me)

    # Assign displacement at end of loading (LCL)
    if "springref" in SimDet.keys():
        readh5fieldvar(SimDet["springref"][0], SimDet["springref"][1], MEmodel_.u_me_ED)
    else:
        MEmodel_.u_me_ED.assign(MEmodel_.GetDisplacement())
        #print("nothing")
    MEmodel_.isspringon = 1.0

    printout("volume = " + str(MEmodel_.GetLVV()), comm_me)

    # Closed-loop phase
    stop_iter = SimDet["closedloopparam"]["stop_iter"]

    isrestart = 0
    cnt = 0

    potential_me = Function(FunctionSpace(MEmodel_.mesh, "DG", 0))
    writecnt = 0

    it_ = 0


    while 1:

        if state_obj.cycle > stop_iter:
            break

        # Update ME and Circulatory model
        info, msg = coupleMEandCL_.UpdateMEandCirc()

        with open(outputfolder + folderName + "output_PV.txt", "a") as f_PV:
            if MPI.rank(comm_me) == 0:
                if isLV or iswaorta:
                    if SimDet.get("lv_lumped"):
                        f_PV.write(
                            f"{state_obj.tstep}, {CLmodel_.V_LV}, {CLmodel_.PLV} \n"
                        )
                    else:
                        f_PV.write(f"{state_obj.tstep}, {CLmodel_.V_LV}, {CLmodel_.PLV} \n")
                elif isBiV or isFCH:
                    if SimDet.get("fch_lumped"):
                        f_PV.write(
                            f"{state_obj.tstep}, {CLmodel_.V_LV}, {CLmodel_.PLV}, {CLmodel_.V_RV}, {CLmodel_.PRV}, {CLmodel_.V_LA}, {CLmodel_.PLA}, {CLmodel_.V_RA}, {CLmodel_.PRA} \n"
                        )
                    elif SimDet.get("fch_fe"):
                        pass
                    else:
                        f_PV.write(
                            f"{state_obj.t}, {CLmodel_.V_LV}, {CLmodel_PLV}, {CLmodel_.V_RV}, {CLmodel_.PRV}, {CLmodel_.V_LA}, {CLmodel_.PLA}, {CLmodel_.V_RA}, {CLmodel_.PRA}, {CLmodel_.V_sv}, {CLmodel_.V_sa}, {CLmodel_.V_ad}, {CLmodel_.V_pv}, {CLmodel_.V_pa} \n"
                        )
        with open(outputfolder + folderName + "output_nfev.txt", "a") as nfev:
            if (
                MPI.rank(comm_me) == 0
                and not SimDet.get("fch_lumped")
                and not SimDet.get("lv_lumped")
            ):
                nfev.write(
                    f"t = {state_obj.t}, iter = {info['nfev']}, fun = {info['fvec']}, message = {msg} \n"
                )

        state_obj.tstep = state_obj.tstep + state_obj.dt.dt
        state_obj.cycle = math.floor(state_obj.tstep / state_obj.BCL)
        state_obj.t = state_obj.tstep - state_obj.cycle * state_obj.BCL

        # Compute local time since activation
        t_init = MEmodel_.activeforms.t_init.vector().get_local()
        local_cycle = np.floor((MEmodel_.t_a.vector().get_local() - t_init)/ state_obj.BCL)
        local_cycle[local_cycle < 0] = 0 # Return zero if cycle is less than zero
        MEmodel_.cycle.vector()[:] = local_cycle
        MEmodel_.t_a.vector()[:] = MEmodel_.t_a.vector().get_local() + state_obj.dt.dt
        MEmodel_.t_since_activation.vector()[:] =  MEmodel_.t_a.vector().get_local()  - t_init - local_cycle * state_obj.BCL
       
        print("Init vector", flush=True)
        print(MEmodel_.activeforms.t_init.vector().get_local(), flush=True)
        print("Cycle", flush=True)
        print(MEmodel_.cycle.vector().get_local(), flush=True)

        isrestart = 0
        state_obj.dt.dt = delTat

        if isPJ:
           # Reset phi and r in EP at end of diastole
           if state_obj.t < state_obj.dt.dt:
               coupleEPandPJ_.reset()
               EPmodel_ep.reset()
               if isPJ:
                   EPmodel_pj.reset()

           if not SimDet.get("lv_lumped") and not ishomo:
               printout("Solving FHN EP", comm_me)
               solver_FHN_ep.solvenonlinear()

               # Activate PJ fiber network
               if(state_obj.t > SimDet["pacing_timing"][0][0] and \
                  state_obj.t < SimDet["pacing_timing"][0][0] + SimDet["pacing_timing"][0][1] ):
                   if("fstim_val_array" in dir(EPmodel_pj)):
                       EPmodel_pj.fstim_val_array[0] = pj_intensity
                       printout("pacing", comm_me)#, EPmodel_pj.fstim_val_array)
                   else:
                       EPmodel_pj.fstim_array[0].iStim = pj_intensity
                       printout("pacing", comm_me)#, EPmodel_pj.fstim_array)
               else:
                   if("fstim_val_array" in dir(EPmodel_pj)):
                       EPmodel_pj.fstim_val_array[0] = 0
                   else:
                       EPmodel_pj.fstim_array[0].iStim = 0

                   printout("not pacing", comm_me)

               if("UpdateActivation" in dir(EPmodel_pj)):
                   EPmodel_pj.UpdateActivation()
               printout("Solving FHN PJ", comm_me)
               solver_FHN_pj.solvenonlinear()

        else:
            if state_obj.t < state_obj.dt.dt:
                EPmodel_ep.reset()
 
            # Activate EP network
            if(state_obj.t > SimDet["pacing_timing"][0][0] and \
               state_obj.t < SimDet["pacing_timing"][0][0] + SimDet["pacing_timing"][0][1] ):
                    if("fstim_val_array" in dir(EPmodel_ep)):
                        for  p in range(0,len(EPmodel_ep.fstim_val_array)):
                            EPmodel_ep.fstim_val_array[p] = intensity
                    else:
                        for  p in range(0,len(EPmodel_ep.fstim_array)):
                            EPmodel_ep.fstim_array[p].iStim = intensity
                    printout("pacing", comm_me)#, EPmodel_ep.fstim_array)
            else:
                for  p in range(0,len(EPmodel_ep.fstim_array)):
                    if("fstim_val_array" in dir(EPmodel_ep)):
                        for  p in range(0,len(EPmodel_ep.fstim_val_array)):
                            EPmodel_ep.fstim_val_array[p] = 0.0
                    else:
                        for  p in range(0,len(EPmodel_ep.fstim_array)):
                            EPmodel_ep.fstim_array[p].iStim = 0.0

            if not SimDet.get("lv_lumped") and not ishomo:
                if("UpdateActivation" in dir(EPmodel_ep)):
                    EPmodel_ep.UpdateActivation()

                printout("Solving FHN EP", comm_me)
                solver_FHN_ep.solvenonlinear()

        if isrestart == 0:
            MEmodel_.UpdateVar()  # For damping
            EPmodel_ep.UpdateVar()
           
            if isPJ:
               EPmodel_pj.UpdateVar()

        # Interpolate phi to mechanics mesh
        potential_ref = EPmodel_ep.interpolate_potential_ep2me_phi(
            V_me=Function(FunctionSpace(MEmodel_.mesh, "DG", 0))
        )
        potential_ref.rename("v_ref", "v_ref")

        potential_me.vector()[:] = potential_ref.vector().get_local()[:]

        #  - - - - - - - - - - - -- - - - - - - - - - - - - - - -- - - - - - -
        if MPI.rank(comm_ep) == 0:
            print("UPdating isActiveField and tInitiationField")

        MEmodel_.activeforms.update_activationTime(
            potential_n=potential_me, comm=comm_me
        )

        # Update PJ activation time:
        if isPJ:
            coupleEPandPJ_.UpdatePJandEP()

        F_n = MEmodel_.GetFmat()
        fstress_DG = project(
            MEmodel_.Getfstress(),
            FunctionSpace(MEmodel_.mesh, "DG", 0),
            form_compiler_parameters={"representation": "uflacs"},
        )
        fstress_DG.rename("fstress", "fstress")


        # postprocess and write
        #  - - - - - - - - - - - -- - - - - - - - - - - - - - - -- - - - - - -
        export.writePV(MEmodel_, state_obj.tstep, CLmodel = CLmodel_)

        if isLV:
            export.writeQ(MEmodel_, [CLmodel_.Qsa, CLmodel_.Qad, CLmodel_.Qsv, 
                                     CLmodel_.Qmv, CLmodel_.Qav], state_obj.tstep)
            export.writeP(MEmodel_, np.array([CLmodel_.Psa, CLmodel_.Pad, CLmodel_.Psv, 
                                              CLmodel_.PLA, CLmodel_.PLV])*0.0075, state_obj.tstep)
            export.writeV(MEmodel_, np.array([CLmodel_.V_sa, CLmodel_.V_ad, CLmodel_.V_sv, 
                                              CLmodel_.V_LA, CLmodel_.V_LV]), state_obj.tstep)


        if isBiV:
            export.writeQ(MEmodel_, np.array([CLmodel_.Qsa, CLmodel_.Qad, CLmodel_.Qsv, CLmodel_.Qmv, CLmodel_.Qav,
                                              CLmodel_.Qtv, CLmodel_.Qpvv, CLmodel_.Qpa, CLmodel_.Qpv
                                              ]), state_obj.tstep)

            export.writeP(MEmodel_, np.array([CLmodel_.Psa, CLmodel_.Pad, CLmodel_.Psv, CLmodel_.PLA, CLmodel_.PLV,
                                              CLmodel_.Ppa, CLmodel_.Ppv, CLmodel_.PRV,
                                              CLmodel_.PRA])*0.0075, state_obj.tstep)

            export.writeV(MEmodel_, np.array([CLmodel_.V_sa, CLmodel_.V_ad, CLmodel_.V_sv, CLmodel_.V_LA, CLmodel_.V_LV,
                                              CLmodel_.V_pa, CLmodel_.V_pv, CLmodel_.V_RV,
                                              CLmodel_.V_RA]), state_obj.tstep)


        if cnt % SimDet["writeStep"] == 0.0:
            export.writetpt(MEmodel_, state_obj.tstep)
            export.hdf.write(MEmodel_.GetDisplacement(), "ME/u", writecnt)
            export.hdf.write(potential_ref, "ME/potential_ref", writecnt)
            export.hdf.write(MEmodel_.GetSActive(), "ME/Sactive", writecnt)
            export.hdf.write(MEmodel_.Get_t_a(), "ME/t_a", writecnt)
            export.hdf.write(MEmodel_.Get_t_init(), "ME/t_init", writecnt)
            export.hdf.write(MEmodel_.Get_isActive(), "ME/isActive", writecnt)
            export.hdf.write(MEmodel_.Get_local_cycle(), "ME/cycle", writecnt)
            export.hdf.write(MEmodel_.Get_t_since_act(), "ME/t_since_act", writecnt)
            export.hdf.write(fstress_DG, "ME/fstress", writecnt)
            export.hdf.write(MEmodel_.GetP(), "ME/imp_constraint", writecnt)

            export.hdf.write(EPmodel_ep.getphivar(), "EP/phi", writecnt)
            export.hdf.write(EPmodel_ep.getrvar(), "EP/r", writecnt)
            export.hdf.write(potential_ref, "EP/potential_ref", writecnt)

            if isPJ:
                export.hdf.write(EPmodel_pj.getphivar(), "PJ/phi", writecnt)
                export.hdf.write(EPmodel_pj.getrvar(), "PJ/r", writecnt)

            writecnt += 1


        cnt += 1

        if(state_obj.tstep % state_obj.BCL == 0) :
            export.dump_restart_file(CLmodel_)


#  - - - - - - - - - - - -- - - - - - - - - - - - - - - -- - - - - - -
if __name__ == "__main__":
    print("Testing...")
    run_BiV_TimedGuccione(IODet=IODetails, SimDet=SimDetails)

#  - - - - - - - - - - - -- - - - - - - - - - - - - - - -- - - - - - -
