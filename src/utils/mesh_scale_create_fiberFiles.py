import os
import sys, os

sys.path.append("/home/fenics/shared/")
# The deployment parent, so `import heArt_legacy...` resolves. Defaults to the lab's
# SHARED research folder rather than an individual's home; override HPCC_HEART_GIT.
sys.path.append(os.environ.get("HPCC_HEART_GIT", "/mnt/research/compbiolab"))

import dolfin as df

# from addfiber_matid import *
# Fiber generation is delegated to the vendored vtk_py3 toolkit (external/vtk_py3).
# Its runtime deps (vtk, sympy, ...) are NOT in the solver env by default, so the
# import is optional: the solver runs on pre-built HDF5 meshes and never calls
# these fiber-generation helpers. See external/CLAUDE.md.
from mpi4py import MPI as pyMPI

try:
    import vtk_py3
    from vtk_py3 import SetBiVFiber_Quad_PyQ, addLVfiber_LDRB
    # vtk_py3/__init__ imports the C++ convertQuadDataToVTK *after* the pure-Python
    # one, so the cpp version shadows it at package level. The cpp version JIT-builds
    # a module that #include <vtk-9.2/...> C++ headers, which the pip `vtk` wheel does
    # NOT ship (Python bindings + .so only). The fiber routines only call
    # convertQuadDataToVTK to dump fibers to VTK for visualization, so restore the
    # header-free pure-Python implementation as the package-level converter.
    try:
        import importlib
        # import_module returns the *module* from sys.modules (robust to the
        # package-level name being shadowed by the re-exported function).
        _py_convert_mod = importlib.import_module("vtk_py3.convertQuadDataToVTK")
        vtk_py3.convertQuadDataToVTK = _py_convert_mod.convertQuadDataToVTK
    except Exception:
        pass
except ImportError:  # preprocessing-only deps absent (e.g. vtk) -> defer failure to call sites
    vtk_py3 = None
    SetBiVFiber_Quad_PyQ = None
    addLVfiber_LDRB = None

import pdb


def create_EDFibers(meshData):
    # if("meshName" in meshData.keys()):
    #       meshname = meshData["meshName"]

    # mesh = df.Mesh()
    mesh = meshData["mesh"]
    facetboundaries = meshData["facets"]
    isLV = meshData["isLV"]

    comm1 = mesh.mpi_comm()

    if "mFileName" in list(meshData.keys()):
        outdir = meshData["mFileName"]
        if df.MPI.rank(comm1) == 0:
            if not os.path.exists(outdir):
                os.makedirs(outdir)
    else:
        outdir = []

    epiid = meshData["epiid"]
    rvid = meshData["rvid"]
    lvid = meshData["lvid"]

    isepiflip = meshData["isepiflip"]
    isendoflip = meshData["isendoflip"]
    isscaling = meshData["iscaling"]

    if isscaling:
        # Some pipelines store meshes in arbitrary units; optionally rescale and rebuild BVH for queries.
        x = mesh.coordinates()
        scaling_factor = 0.1
        x[:, :] *= scaling_factor
        # mesh.intersection_operator().clear() # up to version 1.2.0
        mesh.bounding_box_tree().build(mesh)  # development version

    #  - - - - - - - - - - - -- - - - - - - - - - - - - - - -- - - - - - -

    # Fiber generation is delegated to vtk_py3 LDRB/quad routines (expects epicardial/endo markers).
    isrotatept = False
    isreturn = True
    outfilename = meshData["meshName"]

    quad_deg = 4

    LVangle = meshData["LVangle"]  # [60, -60]
    Septangle = meshData["Septangle"]  # [60, -60]
    RVangle = meshData["RVangle"]  # [60, -60]

    comm2 = pyMPI.COMM_WORLD
    matid = df.MeshFunction("size_t", mesh, 3, mesh.domains())

    fiber_angle_param = {
        "mesh": mesh,
        "facetboundaries": facetboundaries,
        "LV_fiber_angle": LVangle,
        "LV_sheet_angle": [0.1, -0.1],
        "Septum_fiber_angle": Septangle,
        "Septum_sheet_angle": [0.1, -0.1],
        "RV_fiber_angle": RVangle,
        "RV_sheet_angle": [0.1, -0.1],
        "LV_matid": 0,
        "Septum_matid": 1,
        "RV_matid": 2,
        "matid": matid,  # meshData["matid"],\
        "isrotatept": isrotatept,
        "isreturn": isreturn,
        "outfilename": outfilename,
        "epiid": epiid,
        "rvid": rvid,
        "lvid": lvid,
        "degree": quad_deg,
    }

    if "mFileName" in list(meshData.keys()):
        fiber_angle_param.update({"outdirectory": outdir})

    if isLV:
        # vtk_py3/__init__ re-exports these as functions (not submodules) via
        # `from vtk_py3.<mod> import *`, so call them directly (cf. line ~250).
        ef, es, en = addLVfiber_LDRB(fiber_angle_param)
    else:
        ef, es, en = SetBiVFiber_Quad_PyQ(fiber_angle_param)

    X = SpatialCoordinate(mesh)
    N = df.FacetNormal(mesh)
    ds = dolfin.ds(
        subdomain_data=facetboundaries, metadata={"quadrature_degree": quad_deg}
    )
    lv_vol_form = -df.Constant(1.0 / 3.0) * df.inner(N, X) * ds(lvid)
    lv_vol = df.assemble(
        lv_vol_form, form_compiler_parameters={"representation": "uflacs"}
    )
    # Sanity-check: compute cavity volume from a surface integral (divergence theorem identity).
    if not isLV:
        rv_vol_form = -df.Constant(1.0 / 3.0) * df.inner(N, X) * ds(rvid)
        rv_vol = df.assemble(
            rv_vol_form, form_compiler_parameters={"representation": "uflacs"}
        )

    disp_V = df.VectorFunctionSpace(mesh, "CG", 1)
    disp = df.Function(disp_V)

    disp.vector()[:] = 0.0

    if comm2.Get_rank() == 0:
        print("LV cavity vol = ", lv_vol)
        if not isLV:
            print("RV cavity vol = ", rv_vol)

    #  - - - - - - - - - - - -- - - - - - - - - - - - - - - -- - - - - - -
    return ef, es, en


def create_hdf5File(
    meshData,
    outdir="./output/",
    indir="",
    fname="CRT27_AS_smooth_fine",
    casename="CRT27_AS_smooth_fine",
    isscaling=False,
):
    meshname = casename
    mesh = df.Mesh()
    f = df.HDF5File(mpi_comm_world(), indir + fname + ".hdf5", "a")
    f.read(mesh, meshname, False)

    comm1 = mesh.mpi_comm()

    if df.MPI.rank(comm1) == 0:
        # https://stackoverflow.com/questions/273192/how-can-i-create-a-directory-if-it-does-not-exist
        if not os.path.exists(outdir):
            os.makedirs(outdir)

    # os.system("rm "+ outdir + "*.pvd")
    # os.system("rm "+ outdir + "*.vtu")
    # os.system("rm "+ outdir + "*.vtp")
    # os.system("rm "+ outdir + "*.pvtp")
    # os.system("rm "+ outdir + "*.pvtu")
    # os.system("rm "+ outdir + "*.txt")
    # os.system("rm "+ outdir + "*.hdf5")

    facetboundaries = df.MeshFunction("size_t", mesh, 2)
    f.read(facetboundaries, meshname + "/" + "facetboundaries")

    matid = df.MeshFunction("size_t", mesh, 3)
    f.read(matid, meshname + "/" + "matid")

    partId = df.MeshFunction("size_t", mesh, 3)
    # PurKid = FacetFunction('size_t', mesh)
    AHAid = df.MeshFunction("size_t", mesh, 3)
    EpiBCid = FacetFunction("size_t", mesh)

    f.read(partId, meshname + "/" + "partId")
    # f.read(PurKid, meshname+"/"+"PurKid")
    f.read(AHAid, meshname + "/" + "AHAid")
    f.read(EpiBCid, meshname + "/" + "EpiBCid")

    f.close()

    epiid = meshData["epiid"]  # 1
    rvid = meshData["rvid"]  # 3#2
    lvid = meshData["lvid"]  # 2#3

    isepiflip = False
    isendoflip = False

    """
    # moved to create_mesh_quad
    # Translate mesh
    ztop =  max(mesh.coordinates()[:,2])
    ztrans = df.Constant((0.0, 0.0, -ztop))

    if(dolfin.dolfin_version() != '1.6.0'):
        ALE.move(mesh,ztrans)
    else:
        mesh.move(ztrans)
    """

    if isscaling:
        x = mesh.coordinates()
        scaling_factor = 0.1
        x[:, :] *= scaling_factor
        # mesh.intersection_operator().clear() # up to version 1.2.0
        mesh.bounding_box_tree().build(mesh)  # development version

    #  - - - - - - - - - - - -- - - - - - - - - - - - - - - -- - - - - - -

    # BiV fiber
    isrotatept = False
    isreturn = True
    outfilename = meshname

    # isscaling=True
    # isrotatept = False
    # isreturn = True
    quad_deg = 4

    LVangle = meshData["LVangle"]  # [60, -60]
    Septangle = meshData["Septangle"]  # [60, -60]
    RVangle = meshData["RVangle"]  # [60, -60]

    comm2 = pyMPI.COMM_WORLD

    fiber_angle_param = {
        "mesh": mesh,
        "facetboundaries": facetboundaries,
        "LV_fiber_angle": LVangle,
        "LV_sheet_angle": [0.1, -0.1],
        "Septum_fiber_angle": Septangle,
        "Septum_sheet_angle": [0.1, -0.1],
        "RV_fiber_angle": RVangle,
        "RV_sheet_angle": [0.1, -0.1],
        "LV_matid": 0,
        "Septum_matid": 1,
        "RV_matid": 2,
        "matid": matid,
        "isrotatept": isrotatept,
        "isreturn": isreturn,
        "outfilename": meshname,
        "outdirectory": outdir,
        "epiid": epiid,
        "rvid": rvid,
        "lvid": lvid,
        "degree": quad_deg,
    }

    ef, es, en = SetBiVFiber_Quad_PyQ(fiber_angle_param)

    X = SpatialCoordinate(mesh)
    N = df.FacetNormal(mesh)
    ds = dolfin.ds(
        subdomain_data=facetboundaries, metadata={"quadrature_degree": quad_deg}
    )
    lv_vol_form = -df.Constant(1.0 / 3.0) * df.inner(N, X) * ds(lvid)
    rv_vol_form = -df.Constant(1.0 / 3.0) * df.inner(N, X) * ds(rvid)
    lv_vol = df.assemble(
        lv_vol_form, form_compiler_parameters={"representation": "uflacs"}
    )
    rv_vol = df.assemble(
        rv_vol_form, form_compiler_parameters={"representation": "uflacs"}
    )

    disp_V = df.VectorFunctionSpace(mesh, "CG", 1)
    disp = df.Function(disp_V)

    disp.vector()[:] = 0.0

    if comm2.Get_rank() == 0:
        print("LV cavity vol = ", lv_vol)
        print("RV cavity vol = ", rv_vol)

    #  - - - - - - - - - - - -- - - - - - - - - - - - - - - -- - - - - - -

    # f = df.HDF5File(mesh.mpi_comm(), outdir + fname+".hdf5", 'w')
    # f.write(mesh, meshname)
    # f.close()

    f = df.HDF5File(mesh.mpi_comm(), outdir + fname + ".hdf5", "a")
    # f.write(facetboundaries, meshname+"/"+"facetboundaries")
    f.write(ef, meshname + "/" + "eF")
    f.write(es, meshname + "/" + "eS")
    f.write(en, meshname + "/" + "eN")
    # f.write(matid, meshname+"/"+"matid")

    # f.write(partId, meshname+"/"+"partId")
    # f.write(PurKid, meshname+"/"+"PurKid")
    # f.write(AHAid, meshname+"/"+"AHAid")
    # f.write(EpiBCid, meshname+"/"+"EpiBCid")

    f.write(disp, meshname + "/" + "disp")

    f.close()

    df.File(outdir + "facetboundaries" + ".pvd") << facetboundaries
    df.File(outdir + "matid" + ".pvd") << matid

    df.File(outdir + "partId" + ".pvd") << partId
    # df.File(outdir+"PurKid" +".pvd") << PurKid
    df.File(outdir + "AHAid" + ".pvd") << AHAid
    df.File(outdir + "EpiBCid" + ".pvd") << EpiBCid


if __name__ == "__main__":
    create_hdf5File(
        outdir="./outFrom_scaleCreateFiber/",
        indir="./outFrom_CreateQuadPy/",
        fname="89216_mwps",
        casename="89216_mwps",
        isscaling=True,
    )
