import dolfin as df
from .postprocessdatalib2 import *
from ..mechanics.forms_MRC2 import Forms
from ..utils.oops_objects_MRC2 import State_Variables
from ..utils.oops_objects_MRC2 import lv_mesh as lv_mechanics_mesh
from ..utils.oops_objects_MRC2 import biventricle_mesh as biv_mechanics_mesh
from ..mechanics.MEmodel3 import MEmodel
import matplotlib

matplotlib.use("Agg")
from matplotlib import pylab as plt
from mpi4py import MPI as pyMPI


def postprocessdata(IODet, SimDet, cycle=None):
    directory = IODet["outputfolder"] + "/"
    casename = IODet["caseID"]
    BCL = SimDet["HeartBeatLength"]
    if cycle is None:
        cycle = SimDet["closedloopparam"]["stop_iter"]

    for ncycle in range(cycle - 1, cycle):
        filename = directory + casename + "/" + "BiV_PV.txt"
        homo_tptt, homo_LVP, homo_LVV, homo_RVP, homo_RVV, homo_Qmv = extract_PV(
            filename, BCL, ncycle, SimDet
        )

        filename = directory + casename + "/" + "BiV_Q.txt"
        (
            homo_tptt,
            homo_Qao,
            homo_Qmv,
            homo_Qper,
            homo_Qla,
            homo_Qlad,
            homo_Qlcx,
            Q_lvad,
        ) = extract_Q(filename, BCL, ncycle)

        filename = directory + casename + "/" + "BiV_P.txt"
        homo_tptt, homo_Pven, homo_LVPP, homo_Part, homo_PLA = extract_P(
            filename, BCL, ncycle
        )

        filename = directory + casename + "/" + "BiV_IMP_InC.txt"
        homo_tpt_IMP, homo_IMP = extract_probe(filename, BCL, ncycle)

        filename = directory + casename + "/" + "BiV_fiberStrain.txt"
        homo_tpt_Eff, homo_Eff = extract_probe(filename, BCL, ncycle)

        filename = directory + casename + "/" + "BiV_fiberStress.txt"
        homo_tpt_Sff, homo_Sff = extract_probe(filename, BCL, ncycle)

        ESP, ESV = extractESP(homo_LVP, homo_LVV)

        EDP, EDV = extractEDP(homo_LVP, homo_LVV)

        SBP = max(homo_Part) * 0.0075
        DBP = min(homo_Part) * 0.0075

        print(filename)
        print(
            (
                "EF = ",
                (max(homo_LVV) - min(homo_LVV)) / max(homo_LVV),
                " EDV = ",
                max(homo_LVV),
                " ESV = ",
                min(homo_LVV),
                " EDP = ",
                EDP,
                " SBP = ",
                SBP,
                " DBP = ",
                DBP,
            )
        )

        print(("Peak LV pressure = ", max(homo_LVP)))

        homo_directory = directory + casename + "/"

        tpt_array = readtpt(homo_directory + "tpt.txt")
        ind = np.where((tpt_array > (ncycle) * BCL) * (tpt_array < (ncycle + 1) * BCL))
        tpt = tpt_array[ind]

        ## Get Point cloud for probing
        # ptcloud, radialpos, vtkradialpos = getpointclouds(homo_directory, clipoffset=5e-1, npts=10000)
        ptcloud, radialpos, vtkradialpos = getpointclouds(
            directory + casename + "/", clipoffset=1e-5, npts=10000
        )

        vtk_py.writeXMLPData(vtkradialpos, casename + ".vtp")

        # Get transmural variation of IMP
        index = find_nearest(
            tpt, homo_tptt[np.argmax(homo_LVPP)]
        )  # Find ID correspond to peak LV pressure
        imp = probeqty(homo_directory, "ME/imp_constraint", ptcloud, ind, index)
        imp = imp * 0.0075

        ## Get transmural variation of WD
        Sff = probetimeseries(homo_directory, "ME/fstress", ptcloud, ind, "DG", 0)
        Eff = probetimeseries(homo_directory, "ME/Eff", ptcloud, ind, "DG", 0)
        WD = np.array(
            [
                -1.0 * np.trapz(Sff[:, i] * 0.0075, Eff[:, i])
                for i in range(0, len(Sff[1, :]))
            ]
        )

        # Convert to vtp flie
        for i in range(0, len(Sff[:, 1])):
            pdata = vtk.vtkPolyData()
            pdata.DeepCopy(vtkradialpos)
            Sff_VTK_data = numpy_support.numpy_to_vtk(
                num_array=0.0075 * Sff[i, :].ravel(),
                deep=True,
                array_type=vtk.VTK_FLOAT,
            )
            Sff_VTK_data.SetName("fstress_")
            pdata.GetPointData().AddArray(Sff_VTK_data)
            Eff_VTK_data = numpy_support.numpy_to_vtk(
                num_array=Eff[i, :].ravel(), deep=True, array_type=vtk.VTK_FLOAT
            )
            Eff_VTK_data.SetName("Eff_")
            pdata.GetPointData().AddArray(Eff_VTK_data)
            WD_VTK_data = numpy_support.numpy_to_vtk(
                num_array=WD.ravel(), deep=True, array_type=vtk.VTK_FLOAT
            )
            WD_VTK_data.SetName("WD_")
            pdata.GetPointData().AddArray(WD_VTK_data)
            # vtk_py.writeXMLPData(pdata, casename+"fstress"+str(i)+".vtp")

        ## Get Ecc
        Ecc = probetimeseries(homo_directory, "ME/Ecc", ptcloud, ind, "DG", 0)
        peakEcc = np.max(np.abs(np.mean(Ecc, axis=1) * 100))
        print(("Peak Ecc = ", peakEcc))

        # Get Ell
        Ell = probetimeseries(homo_directory, "ME/Ell", ptcloud, ind, "DG", 0)
        peakEll = np.max(np.abs(np.mean(Ell, axis=1) * 100))
        print(("Peak Ell = ", peakEll))

        np.savez(
            directory + casename + "/" + casename + ".npz",
            homo_tptt=homo_tptt,
            homo_LVP=homo_LVP,
            homo_LVV=homo_LVV,
            homo_Qmv=homo_Qmv,
            homo_Qao=homo_Qao,
            homo_Qper=homo_Qper,
            homo_Qla=homo_Qla,
            homo_Qlad=homo_Qlad,
            homo_Pven=0.0075 * homo_Pven,
            homo_LVPP=0.0075 * homo_LVPP,
            homo_Part=0.0075 * homo_Part,
            homo_PLA=0.0075 * homo_PLA,
            homo_tpt_IMP=0.0075 * homo_tpt_IMP,
            homo_IMP=homo_IMP,
            homo_tpt_Eff=homo_tpt_Eff,
            homo_Eff=homo_Eff,
            homo_tpt_Sff=homo_tpt_Sff,
            homo_Sff=homo_Sff,
            ESP=ESP,
            ESV=ESV,
            EDP=EDP,
            EDV=EDV,
            SBP=SBP,
            DBP=DBP,  # Qtotal       = Qtotal,\
            imp=imp,
            radialpos=radialpos,
            Eff=Eff,
            Sff=Sff,
            WD=WD,
            Ecc=Ecc,
            Ell=Ell,
            BCL=BCL,
            tpt=tpt,
            ncycle=ncycle,
        )


def compute_activation(IODet, SimDet, cycle=None):

    mesh = df.Mesh()
    hdf = df.HDF5File(mesh.mpi_comm(), IODet["outputfolder"] + "/" + IODet["caseID"] + "/" + "Data.h5", "r",)
    hdf.read(mesh, "EP/mesh", False)

    phi_arr = extractvtk(IODet["outputfolder"] + "/" + IODet["caseID"], "EP/phi", "CG", 1, IODet["outputfolder"] + "/" + IODet["caseID"] + "/" + "EP_" + "phi", "phi", group="EP", iswrite=False,)

    V_thres = 0.9
    time_act = df.Function(df.FunctionSpace(mesh, "CG", 1))
    time_act_vec = -1 * np.ones(len(time_act.vector()[:]))


    dt = SimDet["dt"]
    t = dt    
    write_t = SimDet["writeStep"]

    for phi in phi_arr:
        phi_vec = phi.sub(0).vector().get_local()[::3]
        for idx, (time_act_vec_, phi_vec_) in enumerate(zip(time_act_vec, phi_vec)):
            if phi_vec_ > V_thres and time_act_vec_ == -1:
                time_act_vec[idx] = t

        t += dt * write_t

    time_act.vector()[:] = time_act_vec
    time_act.rename("Activation Time", "Activation Time")

    act_outdirectory = os.path.join( IODet["outputfolder"], IODet["caseID"], "activation")
    if not os.path.exists(act_outdirectory):
        os.mkdir(act_outdirectory)
    File_act = df.File(os.path.join(act_outdirectory, "act.pvd"))
    File_act << time_act

    # PJ activation
    mesh_pj = df.Mesh()
    hdf = df.HDF5File(mesh_pj.mpi_comm(),IODet["outputfolder"] + "/" + IODet["caseID"] + "/" + "Data.h5","r",)
    hdf.read(mesh_pj, "PJ/mesh", False)
    pj_phi_arr = extractvtk(IODet["outputfolder"] + "/" + IODet["caseID"], "PJ/phi", "CG", 1, IODet["outputfolder"] + "/" + IODet["caseID"] + "/" + "PJ_" + "phi", "phi", group="PJ", iswrite=False,)

    # pj_V_thres = 0.9
    pj_time_act = df.Function(df.FunctionSpace(mesh_pj, "CG", 1))
    pj_time_act_vec = -1 * np.ones(len(pj_time_act.vector()[:]))
    t = dt
    for pj_phi in pj_phi_arr:
        pj_phi_vec = pj_phi.sub(0).vector().get_local()[::3]
        for idx, (pj_time_act_vec_, pj_phi_vec_) in enumerate(zip(pj_time_act_vec, pj_phi_vec)):
            if pj_phi_vec_ > V_thres and pj_time_act_vec_ == -1:
                pj_time_act_vec[idx] = t

        t += dt * write_t

    pj_time_act.vector()[:] = pj_time_act_vec
    pj_time_act.rename("PJ Activation Time", "PJ Activation Time")

    File_act_pj = df.File(os.path.join(act_outdirectory, "pj_act.pvd"))
    File_act_pj << pj_time_act


def normalize_directionalbasis(eC0, eL0, eR0, mesh, deg):

    #eC0 = Mesh_obj.eC0
    #eL0 = Mesh_obj.eL0
    #eR0 = Mesh_obj.eR0

    eC0_normalized = eC0 / df.sqrt(df.inner(eC0, eC0))
    eL0_normalized = eL0 / df.sqrt(df.inner(eL0, eL0))
    eR0_normalized = eR0 / df.sqrt(df.inner(eR0, eR0))

    eC0_normalized = (
        df.project(
            eC0_normalized,
            df.VectorFunctionSpace(mesh, "DG", 0),
            form_compiler_parameters={
                "representation": "uflacs",
                "quadrature_degree": deg,
            },
        )
        .vector()
        .get_local()
    )
    isnan_eC0_normalized = np.argwhere(np.isnan(eC0_normalized)).flatten()

    eL0_normalized = (
        df.project(
            eL0_normalized,
            df.VectorFunctionSpace(mesh, "DG", 0),
            form_compiler_parameters={
                "representation": "uflacs",
                "quadrature_degree": deg,
            },
        )
        .vector()
        .get_local()
    )
    isnan_eL0_normalized = np.argwhere(np.isnan(eL0_normalized)).flatten()

    eR0_normalized = (
        df.project(
            eR0_normalized,
            df.VectorFunctionSpace(mesh, "DG", 0),
            form_compiler_parameters={
                "representation": "uflacs",
                "quadrature_degree": deg,
            },
        )
        .vector()
        .get_local()
    )
    isnan_eR0_normalized = np.argwhere(np.isnan(eR0_normalized)).flatten()

    mesh_coordinates = df.FunctionSpace(
        mesh, "DG", 0
    ).tabulate_dof_coordinates()

    np.set_printoptions(threshold=sys.maxsize)
    if isnan_eC0_normalized.size != 0:
        for p in isnan_eC0_normalized:
            list_of_nan_ids = [p // 3 * 3 + i for i in range(0, 3)]
            distances_to_nan_pt = [
                np.linalg.norm(mesh_coordinates[i] - mesh_coordinates[p // 3])
                for i in range(0, len(mesh_coordinates))
            ]
            distances_to_nan_pt[p // 3] = 1000
            closest_pt_id = np.argmin(distances_to_nan_pt, axis=0)
            eC0_normalized[p // 3 * 3 : p // 3 * 3 + 3] = eC0_normalized[
                3 * closest_pt_id : 3 * closest_pt_id + 3
            ]

    if isnan_eL0_normalized.size != 0:
        for p in isnan_eL0_normalized:
            list_of_nan_ids = [p // 3 * 3 + i for i in range(0, 3)]
            distances_to_nan_pt = [
                np.linalg.norm(mesh_coordinates[i] - mesh_coordinates[p // 3])
                for i in range(0, len(mesh_coordinates))
            ]
            distances_to_nan_pt[p // 3] = 1000
            closest_pt_id = np.argmin(distances_to_nan_pt, axis=0)
            eL0_normalized[p // 3 * 3 : p // 3 * 3 + 3] = eL0_normalized[
                3 * closest_pt_id : 3 * closest_pt_id + 3
            ]

    if isnan_eR0_normalized.size != 0:
        for p in isnan_eR0_normalized:
            list_of_nan_ids = [p // 3 * 3 + i for i in range(0, 3)]
            distances_to_nan_pt = [
                np.linalg.norm(mesh_coordinates[i] - mesh_coordinates[p // 3])
                for i in range(0, len(mesh_coordinates))
            ]
            distances_to_nan_pt[p // 3] = 1000
            closest_pt_id = np.argmin(distances_to_nan_pt, axis=0)
            eR0_normalized[p // 3 * 3 : p // 3 * 3 + 3] = eR0_normalized[
                3 * closest_pt_id : 3 * closest_pt_id + 3
            ]

    eC0_normalized_ = df.Function(df.VectorFunctionSpace(mesh, "DG", 0))
    eL0_normalized_ = df.Function(df.VectorFunctionSpace(mesh, "DG", 0))
    eR0_normalized_ = df.Function(df.VectorFunctionSpace(mesh, "DG", 0))

    eC0_normalized_.vector()[:] = eC0_normalized
    eL0_normalized_.vector()[:] = eL0_normalized
    eR0_normalized_.vector()[:] = eR0_normalized

    return eC0_normalized_, eL0_normalized_, eR0_normalized_


def compute_strain(IODet, SimDet, LVid=1, RVid=2, cycle=None):

    mesh = df.Mesh()
    hdf = df.HDF5File(
        mesh.mpi_comm(),
        IODet["outputfolder"] + "/" + IODet["caseID"] + "/" + "Data.h5",
        "r",
    )

    u_arr = extractdisplacement(IODet, SimDet, cycle=None)

    if "isLV" in list(SimDet.keys()):
        isLV = SimDet["isLV"]

        if(isLV):

            mesh_params = {
                "directory": IODet["directory_me"],
                "casename": IODet["casename_me"],
                "outputfolder": IODet["outputfolder"],
                "foldername": IODet["folderName"],
                "isLV": isLV,
            }

            Mesh_obj = lv_mechanics_mesh(mesh_params, SimDet)
            try:
                eC0 = Mesh_obj.eC0
                eL0 = Mesh_obj.eL0
                eR0 = Mesh_obj.eR0

            except AttributeError:

                fiber_angle_param = {
                    "mesh": Mesh_obj.mesh,
                    "facetboundaries": Mesh_obj.facetboundaries,
                    "LV_fiber_angle": [0.01, -0.01],
                    "LV_sheet_angle": [0.1, -0.1],
                    "minztol": Mesh_obj.mesh.hmax()/2.0, # Coarse mesh
                    "isrotatept": False,
                    "isreturn": True,
                    "outfilename": IODet["casename_me"],
                    "outdirectory": IODet["outputfolder"] + IODet["caseID"] + IODet["folderName"],
                    "baseid": SimDet["topid"],
                    "epiid": SimDet["epiid"],
                    "lvid": SimDet["LVendoid"],
                    "degree": SimDet["GiccioneParams"]["deg"]
                }

                eC0, eL0, eR0  = vtk_py3.addLVfiber_LDRB(fiber_angle_param)

    if "isBiV" in list(SimDet.keys()):

        isBiV = SimDet["isBiV"]

        if(isBiV):
            mesh_params = {
                "directory": IODet["directory_me"],
                "casename": IODet["casename_me"],
                "outputfolder": IODet["outputfolder"],
                "foldername": IODet["folderName"],
                "isBiV": isBiV,
            }

            Mesh_obj = biv_mechanics_mesh(mesh_params, SimDet)

            try:
                eC0 = Mesh_obj.eC0
                eL0 = Mesh_obj.eL0
                eR0 = Mesh_obj.eR0

            except AttributeError:

                # Set BiVFiber
                baseid = [SimDet["topid"]]
                if "RVtopid" in SimDet.keys():
                    baseid.append(SimDet["RVtopid"])
                if "LVtopid" in SimDet.keys():
                    baseid.append(SimDet["LVtopid"])

                fiber_angle_param = {"mesh": Mesh_obj.mesh,\
                	 "facetboundaries": Mesh_obj.facetboundaries,\
                	 "LV_fiber_angle": [0.01,-0.01], \
                	 "LV_sheet_angle": [0.1, -0.1], \
                	 "Septum_fiber_angle": [0.01, -0.01],\
                	 "Septum_sheet_angle": [0.1, -0.1],\
                	 "RV_fiber_angle": [0.01, -0.01],\
                	 "RV_sheet_angle": [0.1, -0.1],\
                	 "LV_matid": 0,\
                	 "Septum_matid": 1,\
                	 "RV_matid": 2,\
                	 "matid":  Mesh_obj.matid,\
                	 "isrotatept": False,\
                	 "isreturn": True,\
                         "outfilename": IODet["casename_me"],
                         "outdirectory": IODet["outputfolder"] + IODet["caseID"] + IODet["folderName"],
                         "baseid": baseid,
                	 "epiid": SimDet["epiid"],\
                	 "rvid": SimDet["RVendoid"],\
                	 "lvid": SimDet["LVendoid"],\
                	 "degree": 4}
            
                eC0, eL0, eR0 = vtk_py.SetBiVFiber_Quad_PyQ(fiber_angle_param)

    deg = SimDet["GiccioneParams"]["deg"]
    #eC0_normalized, eL0_normalized, eR0_normalized = normalize_directionalbasis(
    #    eC0, eL0, eR0, Mesh_obj.mesh, deg
    #)
    eC0_normalized = eC0 / df.sqrt(df.inner(eC0, eC0))
    eL0_normalized = eL0 / df.sqrt(df.inner(eL0, eL0))
    eR0_normalized = eR0 / df.sqrt(df.inner(eR0, eR0))


    if SimDet["Mechanics Discretization"] is "P1P1":
        var_deg = 1
    else:
        var_deg = 2

    udisp = df.Function(df.VectorFunctionSpace(Mesh_obj.mesh, "CG", var_deg))
    udisp.vector()[:] = u_arr[0].vector().get_local()[:]

    GuccioneParams = SimDet["GiccioneParams"]
    params = {
        "mesh": Mesh_obj.mesh,
        "displacement_variable": udisp,
        "material model": GuccioneParams["Passive model"],
        "material params": GuccioneParams["Passive params"],
        "incompressible": GuccioneParams["incompressible"],
        "growth_tensor": None,
    }

    uflforms = Forms(params)
    Fref = df.project(uflforms.Fmat(), df.TensorFunctionSpace(Mesh_obj.mesh, "DG", 0))

    Ecc_outdirectory = os.path.join(IODet["outputfolder"], IODet["caseID"], "Ecc")
    if not os.path.exists(Ecc_outdirectory):
        os.mkdir(Ecc_outdirectory)
    #df.File(os.path.join(Ecc_outdirectory, "Ecc_direction.pvd")) << eC0_normalized
    File_Ecc = df.File(os.path.join(Ecc_outdirectory, "Ecc.pvd"))
    Ecc_arr = []
    Ecc_arr_RV = []

    Ell_outdirectory = os.path.join(IODet["outputfolder"], IODet["caseID"], "Ell")
    if not os.path.exists(Ell_outdirectory):
        os.mkdir(Ell_outdirectory)
    #df.File(os.path.join(Ell_outdirectory, "Ell_direction.pvd")) << eL0_normalized
    File_Ell = df.File(os.path.join(Ell_outdirectory, "Ell.pvd"))
    Ell_arr = []
    Ell_arr_RV = []

    Err_outdirectory = os.path.join(IODet["outputfolder"], IODet["caseID"], "Err")
    if not os.path.exists(Err_outdirectory):
        os.mkdir(Err_outdirectory)
    #df.File(os.path.join(Err_outdirectory, "Err_direction.pvd")) << eR0_normalized
    File_Err = df.File(os.path.join(Err_outdirectory, "Err.pvd"))
    Err_arr = []
    Err_arr_RV = []

    for u_arr_ in u_arr:

        if isinstance(LVid, str):
            wall_vol = df.assemble(
                df.Constant(1.0) * Mesh_obj.dx(LVid),
                form_compiler_parameters={"representation": "uflacs"},
            )
        elif isinstance(LVid, list):
            wall_vol = 0
            for LVid_ in LVid:
                wall_vol += df.assemble(
                    df.Constant(1.0) * Mesh_obj.dx(LVid_),
                    form_compiler_parameters={"representation": "uflacs"},
                )

        wall_vol_RV = df.assemble(
            df.Constant(1.0) * Mesh_obj.dx(RVid),
            form_compiler_parameters={"representation": "uflacs"},
        )

        udisp.vector()[:] = u_arr_.vector().get_local()[:]

        Fmat = uflforms.Fmat()
        F = Fmat * df.inv(Fref)
        Cmat = F.T * F

        Ccc = df.inner(eC0_normalized, Cmat * eC0_normalized)
        Ecc = 0.5 * (1 - 1 / Ccc)

        if isinstance(LVid, str):
            global_Ecc = (
                df.assemble(
                    Ecc * Mesh_obj.dx(LVid),
                    form_compiler_parameters={"representation": "uflacs"},
                )
                / wall_vol
            )
        elif isinstance(LVid, list):
            global_Ecc = 0
            for LVid_ in LVid:
                global_Ecc += (
                    df.assemble(
                        Ecc * Mesh_obj.dx(LVid_),
                        form_compiler_parameters={"representation": "uflacs"},
                    )
                )
            global_Ecc = global_Ecc/wall_vol 

        Ecc_arr.append(global_Ecc)
        if isBiV:
            global_Ecc_RV = (
                df.assemble(
                    Ecc * Mesh_obj.dx(RVid),
                    form_compiler_parameters={"representation": "uflacs"},
                )
                / wall_vol
            )
            Ecc_arr_RV.append(global_Ecc_RV)

        Ecc_field = df.project(
            Ecc,
            df.FunctionSpace(Mesh_obj.mesh, "DG", 0),
            form_compiler_parameters={
                "representation": "uflacs",
                "quadrature_degree": deg,
            },
        )
        Ecc_field.rename("Ecc", "Ecc")
        File_Ecc << Ecc_field
        Ecc_arr.append(global_Ecc)

        Cll = df.inner(eL0_normalized, Cmat * eL0_normalized)
        Ell = 0.5 * (1 - 1 / Cll)
        if isinstance(LVid, str):
            global_Ell = (
                df.assemble(
                    Ell * Mesh_obj.dx(LVid),
                    form_compiler_parameters={"representation": "uflacs"},
                )
                / wall_vol
            )
        elif isinstance(LVid, list):
            global_Ell = 0
            for LVid_ in LVid:
                global_Ell += (
                    df.assemble(
                        Ell * Mesh_obj.dx(LVid_),
                        form_compiler_parameters={"representation": "uflacs"},
                    )
                )
            global_Ell = global_Ell/wall_vol 

        Ell_arr.append(global_Ell)
        if isBiV:
            global_Ell_RV = (
                df.assemble(
                    Ell * Mesh_obj.dx(RVid),
                    form_compiler_parameters={"representation": "uflacs"},
                )
                / wall_vol
            )
            Ell_arr_RV.append(global_Ell_RV)

        Ell_field = df.project(
            Ell,
            df.FunctionSpace(Mesh_obj.mesh, "DG", 0),
            form_compiler_parameters={
                "representation": "uflacs",
                "quadrature_degree": deg,
            },
        )
        Ell_field.rename("Ell", "Ell")
        File_Ell << Ell_field
        Ell_arr.append(global_Ell)

        Crr = df.inner(eR0_normalized, Cmat * eR0_normalized)
        Err = 0.5 * (1 - 1 / Crr)
        if isinstance(LVid, str):
            global_Err = (
                df.assemble(
                    Err * Mesh_obj.dx(LVid),
                    form_compiler_parameters={"representation": "uflacs"},
                )
                / wall_vol
            )
        elif isinstance(LVid, list):
            global_Err = 0
            for LVid_ in LVid:
                global_Err += (
                    df.assemble(
                        Err * Mesh_obj.dx(LVid_),
                        form_compiler_parameters={"representation": "uflacs"},
                    )
                )
            global_Err = global_Err/wall_vol 

        Err_arr.append(global_Err)
        if isBiV:
            global_Err_RV = (
                df.assemble(
                    Err * Mesh_obj.dx(RVid),
                    form_compiler_parameters={"representation": "uflacs"},
                )
                / wall_vol
            )
            Err_arr_RV.append(global_Err_RV)

        Err_field = df.project(
            Err,
            df.FunctionSpace(Mesh_obj.mesh, "DG", 0),
            form_compiler_parameters={
                "representation": "uflacs",
                "quadrature_degree": deg,
            },
        )
        Err_field.rename("Err", "Err")
        File_Err << Err_field
        Err_arr.append(global_Err)

    print("Maximum Ecc :", min(Ecc_arr))
    print("Maximum Ell :", min(Ell_arr))
    print("Maximum Err :", max(Err_arr))

    np.savez(os.path.join(Ecc_outdirectory, "Ecc.npz"), Ecc_arr)
    np.savez(os.path.join(Ell_outdirectory, "Ell.npz"), Ell_arr)
    np.savez(os.path.join(Err_outdirectory, "Err.npz"), Err_arr)

    plt.figure()
    plt.plot(np.arange(0, len(Ecc_arr)), Ecc_arr)
    plt.xlabel("Time point", fontsize=14)
    plt.ylabel("Strain", fontsize=14)
    plt.savefig(os.path.join(Ecc_outdirectory, "Ecc.png"))
    plt.clf()

    plt.figure()
    plt.plot(np.arange(0, len(Ell_arr)), Ell_arr)
    plt.xlabel("Time point", fontsize=14)
    plt.ylabel("Strain", fontsize=14)
    plt.savefig(os.path.join(Ell_outdirectory, "Ell.png"))
    plt.clf()

    plt.figure()
    plt.plot(np.arange(0, len(Err_arr)), Err_arr)
    plt.xlabel("Time point", fontsize=14)
    plt.ylabel("Strain", fontsize=14)
    plt.savefig(os.path.join(Err_outdirectory, "Err.png"))
    plt.clf()

    if isBiV:
        print("Maximum RV Ecc :", min(Ecc_arr_RV))
        print("Maximum RV Ell :", min(Ell_arr_RV))
        print("Maximum RV Err :", max(Err_arr_RV))

        np.savez(os.path.join(Ecc_outdirectory, "Ecc_RV.npz"), Ecc_arr_RV)
        np.savez(os.path.join(Ell_outdirectory, "Ell_RV.npz"), Ell_arr_RV)
        np.savez(os.path.join(Err_outdirectory, "Err_RV.npz"), Err_arr_RV)

        plt.figure()
        plt.plot(np.arange(0, len(Ecc_arr_RV)), Ecc_arr_RV)
        plt.xlabel("Time point", fontsize=14)
        plt.ylabel("Strain", fontsize=14)
        plt.savefig(os.path.join(Ecc_outdirectory, "Ecc_RV.png"))
        plt.clf()

        plt.figure()
        plt.plot(np.arange(0, len(Ell_arr_RV)), Ell_arr_RV)
        plt.xlabel("Time point", fontsize=14)
        plt.ylabel("Strain", fontsize=14)
        plt.savefig(os.path.join(Ell_outdirectory, "Ell_RV.png"))
        plt.clf()

        plt.figure()
        plt.plot(np.arange(0, len(Err_arr_RV)), Err_arr_RV)
        plt.xlabel("Time point", fontsize=14)
        plt.ylabel("Strain", fontsize=14)
        plt.savefig(os.path.join(Err_outdirectory, "Err_RV.png"))
        plt.clf()


def extractdisplacement(IODet, SimDet, cycle=None):

    mesh = df.Mesh()
    hdf = df.HDF5File(
        mesh.mpi_comm(),
        IODet["outputfolder"] + "/" + IODet["caseID"] + "/" + "Data.h5",
        "r",
    )

    # Dump displacement
    if SimDet["Mechanics Discretization"] is "P1P1":
        var_deg = 1
    else:
        var_deg = 2

    try:
        u_arr = extractvtk(
            IODet["outputfolder"] + "/" + IODet["caseID"],
            "ME/" + "u",
            "CG",
            var_deg,
            IODet["outputfolder"] + "/" + IODet["caseID"] + "/" + "ME_" + "u",
            "u",
        )
    except RuntimeError:
        print("No attribute for ", var, " found")

    return u_arr


def extractdisplacementloading(IODet, SimDet, cycle=None):

    mesh = df.Mesh()
    hdf = df.HDF5File(
        mesh.mpi_comm(),
        IODet["outputfolder"] + "/" + IODet["caseID"] + "/" + "Data.h5",
        "r",
    )

    # Dump displacement
    if SimDet["Mechanics Discretization"] is "P1P1":
        var_deg = 1
    else:
        var_deg = 2

    try:
        u_arr = extractvtk(
            IODet["outputfolder"] + "/" + IODet["caseID"],
            "ME/" + "u_loading",
            "CG",
            var_deg,
            IODet["outputfolder"] + "/" + IODet["caseID"] + "/" + "ME_" + "u_loading",
            "u",
        )
    except RuntimeError:
        print("No attribute for ", var, " found")

    return u_arr


def dumpvtk(IODet, SimDet, cycle=None, ME_var = [], EP_var = [], PJ_var = []):

    hdf = df.HDF5File(df.MPI.comm_world, IODet["outputfolder"] + "/" + IODet["caseID"] + "/" + "Data.h5", "r",)

    list_of_ME_var = ME_var
    list_of_EP_var = EP_var
    list_of_PJ_var = PJ_var

    #list_of_ME_var = [
    #    ["u", "CG", 1],
    #    ["potential_ref", "CG", 1],
    #    ["Ecc", "DG", 0],
    #    ["Ell", "DG", 0],
    #    ["Err", "DG", 0],
    #    ["Eff", "DG", 0],
    #    ["fstress", "DG", 0],
    #    ["imp", "DG", 1],
    #    ["imp2", "DG", 1],
    #    ["imp_constraint", "DG", 1],
    #]

    #list_of_EP_var = [["phi", "CG", 1], ["r", "DG", 0], ["potential_ref", "CG", 1]]
    #list_of_PJ_var = [["phi", "CG", 1], ["r", "DG", 0], ["potential_ref", "CG", 1]]

    if hdf.has_dataset("ME"):
        for ME_var in list_of_ME_var:

            var = ME_var[0]
            var_space = ME_var[1]
            var_deg = ME_var[2]

            print("Extracting ME", var, " ", var_space, " ", var_deg)

            try:
                var_arr = extractvtk(
                    IODet["outputfolder"] + "/" + IODet["caseID"],
                    "ME/" + var,
                    var_space,
                    var_deg,
                    IODet["outputfolder"] + "/" + IODet["caseID"] + "/" + "ME_" + var,
                    var,
                    group="ME",
                )
            except RuntimeError:
                print("No attribute for ", var, " found")

            if var == "u":
                u_arr = var_arr.copy()

    if hdf.has_dataset("EP"):
        for EP_var in list_of_EP_var:

            var = EP_var[0]
            var_space = EP_var[1]
            var_deg = EP_var[2]
            print("Extracting EP", var)

            try:
                var_arr = extractvtk(
                    IODet["outputfolder"] + "/" + IODet["caseID"],
                    "EP/" + var,
                    var_space,
                    var_deg,
                    IODet["outputfolder"] + "/" + IODet["caseID"] + "/" + "EP_" + var,
                    var,
                    group="EP",
                )

            except RuntimeError:
                print("No attribute for ", var, " found")

    if hdf.has_dataset("PJ"):
        for PJ_var in list_of_PJ_var:

            var = PJ_var[0]
            var_space = PJ_var[1]
            var_deg = PJ_var[2]
            print("Extracting PJ", var)

            try:
                var_arr = extractvtk(
                    IODet["outputfolder"] + "/" + IODet["caseID"],
                    "PJ/" + var,
                    var_space,
                    var_deg,
                    IODet["outputfolder"] + "/" + IODet["caseID"] + "/" + "PJ_" + var,
                    var,
                    group="PJ",
                )

            except RuntimeError:
                print("No attribute for ", var, " found")


def plothemodynamics(IODet, SimDet, cycle=None):

    directory = IODet["outputfolder"] + "/"
    casename = IODet["caseID"]
    BCL = SimDet["HeartBeatLength"]
    if cycle is None:
        cycle = SimDet["closedloopparam"]["stop_iter"] + 1

    plt.figure()
    for ncycle in range(cycle):
        filename = directory + casename + "/" + "BiV_PV.txt"
        homo_tptt, homo_LVP, homo_LVV, homo_RVP, homo_RVV, homo_Qmv = extract_PV(
            filename, BCL, ncycle, SimDet
        )
        plt.plot(homo_LVV, homo_LVP, label=f"LV Cycle = {ncycle}")
        if SimDet.get("isBiV") or SimDet.get("isFCH"):
            plt.plot(homo_RVV, homo_RVP, label=f"RV Cycle = {ncycle}")

    hemodynamics_outdirectory = os.path.join(
        IODet["outputfolder"], IODet["caseID"], "hemodynamics"
    )
    if not os.path.exists(hemodynamics_outdirectory):
        os.mkdir(hemodynamics_outdirectory)

    # plt.plot(homo_LVV, homo_LVP)
    plt.savefig(os.path.join(hemodynamics_outdirectory, "PV.png"))
    plt.clf()


def plotpressure(IODet, SimDet, cycle=None, compartment="All"):

    directory = IODet["outputfolder"] + "/"
    casename = IODet["caseID"]
    BCL = SimDet["HeartBeatLength"]
    if cycle is None:
        cycle = SimDet["closedloopparam"]["stop_iter"] + 1

    plt.figure()
    for ncycle in range(cycle):
        filename = directory + casename + "/" + "BiV_P.txt"

        tpt, Psv, PLV, Psa, PLA, Ppv, PRV, Ppa, PRA = extract_P(
            filename, BCL, ncycle, SimDet
        )

        if ncycle == cycle - 1:
            meanPsv = time_average(tpt, Psv)
            print(
                "Mean Psv = ",
                meanPsv * 0.0075,
                "mmHg; Peak Psv = ",
                max(Psv) * 0.0075,
                "mmHg",
            )
            meanPLV = time_average(tpt, PLV)
            print(
                "Mean PLV = ",
                meanPLV * 0.0075,
                "mmHg; Peak PLV = ",
                max(PLV) * 0.0075,
                "mmHg",
            )
            meanPsa = time_average(tpt, Psa)
            print(
                "Mean Psa = ",
                meanPsa * 0.0075,
                "mmHg; Peak Psa = ",
                max(Psa) * 0.0075,
                "mmHg",
            )
            meanPLA = time_average(tpt, PLA)
            print(
                "Mean PLA = ",
                meanPLA * 0.0075,
                "mmHg; Peak PLA = ",
                max(PLA) * 0.0075,
                "mmHg",
            )

        if compartment == "All" or "sv" in compartment:
            plt.plot(tpt, Psv * 0.0075, label=f"Psv Cycle = {ncycle}")
        if compartment == "All" or "lv" in compartment:
            plt.plot(tpt, PLV * 0.0075, label=f"PLV Cycle = {ncycle}")
        if compartment == "All" or "sa" in compartment:
            plt.plot(tpt, Psa * 0.0075, label=f"Psa Cycle = {ncycle}")
        if compartment == "All" or "la" in compartment:
            plt.plot(tpt, PLA * 0.0075, label=f"PLA Cycle = {ncycle}")

        if "isBiV" in list(SimDet.keys()):
            if SimDet["isBiV"]:
                if compartment == "All" or "pv" in compartment:
                    plt.plot(tpt, Ppv * 0.0075, label=f"Ppv Cycle = {ncycle}")
                if compartment == "All" or "rv" in compartment:
                    plt.plot(tpt, PRV * 0.0075, label=f"PRV Cycle = {ncycle}")
                if compartment == "All" or "pa" in compartment:
                    plt.plot(tpt, Ppa * 0.0075, label=f"Ppa Cycle = {ncycle}")
                if compartment == "All" or "ra" in compartment:
                    plt.plot(tpt, PRA * 0.0075, label=f"PRA Cycle = {ncycle}")

                if ncycle == cycle - 1:
                    meanPpv = time_average(tpt, Ppv)
                    print(
                        "Mean Ppv = ",
                        meanPpv * 0.0075,
                        "mmHg; Peak Ppv = ",
                        max(Ppv) * 0.0075,
                        "mmHg",
                    )
                    meanPRV = time_average(tpt, PRV)
                    print(
                        "Mean PRV = ",
                        meanPRV * 0.0075,
                        "mmHg; Peak PRV = ",
                        max(PRV) * 0.0075,
                        "mmHg",
                    )
                    meanPpa = time_average(tpt, Ppa)
                    print(
                        "Mean Ppa = ",
                        meanPpa * 0.0075,
                        "mmHg; Peak Ppa = ",
                        max(Ppa) * 0.0075,
                        "mmHg",
                    )
                    meanPRA = time_average(tpt, PRA)
                    print(
                        "Mean PRA = ",
                        meanPRA * 0.0075,
                        "mmHg; Peak PRA = ",
                        max(PRA) * 0.0075,
                        "mmHg",
                    )

    hemodynamics_outdirectory = os.path.join(
        IODet["outputfolder"], IODet["caseID"], "hemodynamics"
    )
    if not os.path.exists(hemodynamics_outdirectory):
        os.mkdir(hemodynamics_outdirectory)

    plt.legend()
    plt.ylabel("Pressure (mmHg)")
    plt.xlabel("Time (s)")
    plt.savefig(os.path.join(hemodynamics_outdirectory, "Pressure.png"))
    plt.clf()


def plotflow(IODet, SimDet, cycle=None, compartment="All"):

    directory = IODet["outputfolder"] + "/"
    casename = IODet["caseID"]
    BCL = SimDet["HeartBeatLength"]
    if cycle is None:
        cycle = SimDet["closedloopparam"]["stop_iter"] + 1

    plt.figure()
    for ncycle in range(cycle):
        filename = directory + casename + "/" + "BiV_Q.txt"

        tpt, Qav, Qmv, Qsa, Qsv, Qpvv, Qtv, Qpa, Qpv, Qlvad = extract_Q(
            filename, BCL, ncycle, SimDet
        )

        if ncycle == cycle - 1:
            meanQav = time_average(tpt, Qav)
            print(
                "Mean Qav = ",
                meanQav * 60,
                "L/min; Peak Qav = ",
                max(Qav) * 60,
                "L/min",
            )
            meanQmv = time_average(tpt, Qmv)
            print(
                "Mean Qmv = ",
                meanQmv * 60,
                "L/min; Peak Qmv = ",
                max(Qmv) * 60,
                "L/min",
            )
            meanQsa = time_average(tpt, Qsa)
            print(
                "Mean Qsa = ",
                meanQsa * 60,
                "L/min; Peak Qsa = ",
                max(Qsa) * 60,
                "L/min",
            )
            meanQsv = time_average(tpt, Qsv)
            print(
                "Mean Qsv = ",
                meanQsv * 60,
                "L/min; Peak Qsv = ",
                max(Qsv) * 60,
                "L/min",
            )
            meanQlvad = time_average(tpt, Qlvad)
            print(
                "Mean Qlvad = ",
                meanQlvad * 60,
                "L/min; Peak Qlvad = ",
                max(Qlvad) * 60,
                "L/min",
            )

        if compartment == "All" or "av" in compartment:
            plt.plot(tpt, Qav * 60, label=f"Qav Cycle = {ncycle}")
        if compartment == "All" or "mv" in compartment:
            plt.plot(tpt, Qmv * 60, label=f"Qmv Cycle = {ncycle}")
        if compartment == "All" or "sa" in compartment:
            plt.plot(tpt, Qsa * 60, label=f"Qsa Cycle = {ncycle}")
        if compartment == "All" or "sv" in compartment:
            plt.plot(tpt, Qsv * 60, label=f"Qsv Cycle = {ncycle}")
        if compartment == "All" or "lvad" in compartment:
            plt.plot(tpt, Qlvad * 60, label=f"Qlvad Cycle = {ncycle}")

        if "isBiV" in list(SimDet.keys()):
            if SimDet["isBiV"]:
                if compartment == "All" or "ppv" in compartment:
                    plt.plot(tpt, Qpvv * 60, label=f"Qpvv Cycle = {ncycle}")
                if compartment == "All" or "tv" in compartment:
                    plt.plot(tpt, Qtv * 60, label=f"Qtv Cycle = {ncycle}")
                if compartment == "All" or "pa" in compartment:
                    plt.plot(tpt, Qpa * 60, label=f"Qpa Cycle = {ncycle}")
                if compartment == "All" or "pv" in compartment:
                    plt.plot(tpt, Qpv * 60, label=f"Qpv Cycle = {ncycle}")

                if ncycle == cycle - 1:
                    meanQpvv = time_average(tpt, Qpvv)
                    print(
                        "Mean Qpvv = ",
                        meanQpvv * 60,
                        "L/min; Peak Qpvv = ",
                        max(Qpvv) * 60,
                        "L/min",
                    )
                    meanQtv = time_average(tpt, Qtv)
                    print(
                        "Mean Qtv = ",
                        meanQtv * 60,
                        "L/min; Peak Qtv = ",
                        max(Qtv) * 60,
                        "L/min",
                    )
                    meanQpa = time_average(tpt, Qpa)
                    print(
                        "Mean Qpa = ",
                        meanQpa * 60,
                        "L/min; Peak Qpa = ",
                        max(Qpa) * 60,
                        "L/min",
                    )
                    meanQpv = time_average(tpt, Qpv)
                    print(
                        "Mean Qpv = ",
                        meanQpv * 60,
                        "L/min; Peak Qpv = ",
                        max(Qpv) * 60,
                        "L/min",
                    )

    hemodynamics_outdirectory = os.path.join(
        IODet["outputfolder"], IODet["caseID"], "hemodynamics"
    )
    if not os.path.exists(hemodynamics_outdirectory):
        os.mkdir(hemodynamics_outdirectory)

    # plt.plot(homo_LVV, homo_LVP)
    plt.legend()
    plt.ylabel("Flow rate (L/min)")
    plt.xlabel("Time (s)")
    plt.savefig(os.path.join(hemodynamics_outdirectory, "Flow.png"))
    plt.clf()


def time_average(time_array, data_array):
    """Calculates the time-weighted average of a data array.

    Args:
        time_array (list or numpy array): Array of timestamps.
        data_array (list or numpy array): Array of data values.

    Returns:
        float: Time-weighted average.
    """

    if len(time_array) != len(data_array):
        raise ValueError("Time and data arrays must have the same length.")

    time_array = np.array(time_array)
    data_array = np.array(data_array)

    # Calculate time differences
    time_diffs = np.diff(time_array)

    # Calculate weighted average
    weighted_sum = np.sum(data_array[:-1] * time_diffs)
    total_time = time_array[-1] - time_array[0]

    return weighted_sum / total_time
