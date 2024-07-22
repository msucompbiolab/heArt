import dolfin as df
from .postprocessdatalib2 import *
from ..mechanics.forms_MRC2 import Forms
from ..utils.oops_objects_MRC2 import State_Variables
from ..utils.oops_objects_MRC2 import lv_mesh as lv_mechanics_mesh
from ..mechanics.MEmodel3 import MEmodel
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
        homo_tptt, homo_LVP, homo_LVV, homo_Qmv = extract_PV(filename, BCL, ncycle)

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
    hdf = df.HDF5File(
        mesh.mpi_comm(),
        IODet["outputfolder"] + "/" + IODet["caseID"] + "/" + "Data.h5",
        "r",
    )
    hdf.read(mesh, "EP/mesh", False)

    phi_arr = extractvtk(
        IODet["outputfolder"] + "/" + IODet["caseID"],
        "EP/phi",
        "CG",
        1,
        IODet["outputfolder"] + "/" + IODet["caseID"] + "/" + "EP_" + "phi",
        "phi",
        group="EP",
        iswrite=False,
    )

    V_thres = 0.9
    time_act = df.Function(df.FunctionSpace(mesh, "CG", 1))
    time_act_vec = -1 * np.ones(len(time_act.vector()[:]))

    t = 0
    dt = SimDet["dt"]
    for phi in phi_arr:
        phi_vec = phi.sub(0).vector().get_local()[::3]
        for idx, (time_act_vec_, phi_vec_) in enumerate(zip(time_act_vec, phi_vec)):
            if phi_vec_ > V_thres and time_act_vec_ == -1:
                time_act_vec[idx] = t

        t += dt

    time_act.vector()[:] = time_act_vec
    time_act.rename("Activation Time", "Activation Time")

    act_outdirectory = os.path.join(
        IODet["outputfolder"], IODet["caseID"], "activation"
    )
    if not os.path.exists(act_outdirectory):
        os.mkdir(act_outdirectory)
    File_act = df.File(os.path.join(act_outdirectory, "act.pvd"))
    File_act << time_act


def compute_strain(IODet, SimDet, LVid=1, cycle=None):

    mesh = df.Mesh()
    hdf = df.HDF5File(
        mesh.mpi_comm(),
        IODet["outputfolder"] + "/" + IODet["caseID"] + "/" + "Data.h5",
        "r",
    )

    u_arr = extractdisplacement(IODet, SimDet, cycle=None)

    mesh_params = {
        "directory": IODet["directory_me"],
        "casename": IODet["casename"],
        "outputfolder": IODet["outputfolder"],
        "foldername": IODet["folderName"],
        "isLV": IODet["isLV"],
    }

    Mesh_obj = lv_mechanics_mesh(mesh_params, SimDet)
    eC0 = Mesh_obj.eC0
    eL0 = Mesh_obj.eL0
    eR0 = Mesh_obj.eR0

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
    deg = SimDet["GiccioneParams"]["deg"]

    uflforms = Forms(params)
    Fref = df.project(uflforms.Fmat(), df.TensorFunctionSpace(Mesh_obj.mesh, "DG", 0))

    Ecc_outdirectory = os.path.join(IODet["outputfolder"], IODet["caseID"], "Ecc")
    if not os.path.exists(Ecc_outdirectory):
        os.mkdir(Ecc_outdirectory)
    File_Ecc = df.File(os.path.join(Ecc_outdirectory, "Ecc.pvd"))
    Ecc_arr = []

    Ell_outdirectory = os.path.join(IODet["outputfolder"], IODet["caseID"], "Ell")
    if not os.path.exists(Ell_outdirectory):
        os.mkdir(Ell_outdirectory)
    File_Ell = df.File(os.path.join(Ell_outdirectory, "Ell.pvd"))
    Ell_arr = []

    Err_outdirectory = os.path.join(IODet["outputfolder"], IODet["caseID"], "Err")
    if not os.path.exists(Err_outdirectory):
        os.mkdir(Err_outdirectory)
    File_Err = df.File(os.path.join(Err_outdirectory, "Err.pvd"))
    Err_arr = []

    for u_arr_ in u_arr:

        wall_vol = df.assemble(
            df.Constant(1.0) * Mesh_obj.dx(LVid),
            form_compiler_parameters={"representation": "uflacs"},
        )
        udisp.vector()[:] = u_arr_.vector().get_local()[:]

        Fmat = uflforms.Fmat()
        F = Fmat * df.inv(Fref)
        Cmat = F.T * F

        Ccc = df.inner(eC0_normalized, Cmat * eC0_normalized)
        Ecc = 0.5 * (1 - 1 / Ccc)
        global_Ecc = (
            df.assemble(
                Ecc * Mesh_obj.dx(LVid),
                form_compiler_parameters={"representation": "uflacs"},
            )
            / wall_vol
        )
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
        print("Ecc : ", global_Ecc)

        Cll = df.inner(eL0_normalized, Cmat * eL0_normalized)
        Ell = 0.5 * (1 - 1 / Cll)
        global_Ell = (
            df.assemble(
                Ell * Mesh_obj.dx(LVid),
                form_compiler_parameters={"representation": "uflacs"},
            )
            / wall_vol
        )
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
        print("Ell : ", global_Ell)

        Crr = df.inner(eR0_normalized, Cmat * eR0_normalized)
        Err = 0.5 * (1 - 1 / Crr)
        global_Err = (
            df.assemble(
                Err * Mesh_obj.dx(LVid),
                form_compiler_parameters={"representation": "uflacs"},
            )
            / wall_vol
        )
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
        print("Err : ", global_Err)

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


def dumpvtk(IODet, SimDet, cycle=None):

    hdf = df.HDF5File(
        df.MPI.comm_world,
        IODet["outputfolder"] + "/" + IODet["caseID"] + "/" + "Data.h5",
        "r",
    )

    list_of_ME_var = [
        ["u", "CG", 1],
        ["potential_ref", "CG", 1],
        ["Ecc", "DG", 0],
        ["Ell", "DG", 0],
        ["Err", "DG", 0],
        ["Eff", "DG", 0],
        ["fstress", "DG", 1],
        ["imp", "DG", 1],
        ["imp2", "DG", 1],
        ["imp_constraint", "DG", 1],
    ]

    list_of_EP_var = [["phi", "CG", 1], ["r", "DG", 0], ["potential_ref", "CG", 1]]

    list_of_PJ_var = [["phi", "CG", 1], ["r", "DG", 0], ["potential_ref", "CG", 1]]

    if hdf.has_dataset("ME"):
        for ME_var in list_of_ME_var:

            var = ME_var[0]
            var_space = ME_var[1]
            var_deg = ME_var[2]
            print("Extracting ME", var)

            # Dump displacement
            if SimDet["Mechanics Discretization"] is "P1P1" and var == "u":
                var_deg = 1
            else:
                var_deg = 2

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
        homo_tptt, homo_LVP, homo_LVV, homo_Qmv = extract_PV(filename, BCL, ncycle)
        plt.plot(homo_LVV, homo_LVP * 0.0075, label=f"Cycle = {ncycle}")

    hemodynamics_outdirectory = os.path.join(
        IODet["outputfolder"], IODet["caseID"], "hemodynamics"
    )
    if not os.path.exists(hemodynamics_outdirectory):
        os.mkdir(hemodynamics_outdirectory)

    # plt.plot(homo_LVV, homo_LVP)
    plt.savefig(os.path.join(hemodynamics_outdirectory, "PV.png"))
    plt.clf()
