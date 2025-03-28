from dolfin import *
import ufl as ufl
import numpy as np
import os as os
from ..utils.nsolver import NSolver as NSolver
from ..utils.oops_objects_MRC2 import biventricle_mesh as biv_mechanics_mesh
from ..utils.oops_objects_MRC2 import lv_mesh as lv_mechanics_mesh
from ..utils.oops_objects_MRC2 import fch_mesh as fch_mechanics_mesh

# from ..utils.oops_objects_MRC2 import PV_Elas
from ..utils.oops_objects_MRC2 import update_mesh
from ..utils.oops_objects_MRC2 import printout
from ..utils.edgetypebc import *
from .forms_MRC2 import Forms
from .activeforms_MRC2 import activeForms
from ..utils.mesh_partitionMeshforEP_J import defCPP_Matprop, defCPP_Matprop_DIsch
from ..utils.mesh_scale_create_fiberFiles import create_EDFibers


class MEmodel(object):
    def __init__(self, params, SimDet):
        self.parameters = self.default_parameters()
        self.parameters.update(params)
        self.SimDet = SimDet
        self.isLV = self.SimDet.get("isLV", False)
        self.deg_me = SimDet["GiccioneParams"]["deg"]

        self.discretization = self.SimDet.get("Mechanics Discretization", "P2P1")
        self.discretization_technique = self.SimDet.get("Technique Discretization", 1)

        self.ispctrl = self.SimDet.get("ispctrl", False)
        self.islumped = self.SimDet.get("islumped", False)
        self.iswaorta = self.SimDet.get("iswaorta", False)
        self.isFCH = self.SimDet.get("isFCH", False)
        self.isBiV = self.SimDet.get("isBiV", False)

        if self.isLV:
            self.Mesh = lv_mechanics_mesh(self.parameters, SimDet)
        elif self.iswaorta:
            self.Mesh = lv_mechanics_mesh(self.parameters, SimDet)
        elif self.isFCH:
            self.Mesh = fch_mechanics_mesh(self.parameters, SimDet)
        elif self.isBiV:
            self.Mesh = biv_mechanics_mesh(self.parameters, SimDet)

        f0_me_Gauss = self.Mesh.f0
        s0_me_Gauss = self.Mesh.s0
        n0_me_Gauss = self.Mesh.n0

        # eL0_me_Gauss = self.Mesh.eL0_ao
        # eC0_me_Gauss = self.Mesh.eC0_ao
        eL0_me_Gauss = getattr(self.Mesh, "eL0_ao", None)
        eC0_me_Gauss = getattr(self.Mesh, "eC0_ao", None)

        # eclgn0_me_Gauss = self.Mesh.eclgn0_ao
        # eclgn1_me_Gauss = self.Mesh.eclgn1_ao

        eclgn0_me_Gauss = getattr(self.Mesh, "eclgn0_ao", None)
        eclgn1_me_Gauss = getattr(self.Mesh, "eclgn1_ao", None)

        self.mesh_me = self.Mesh.mesh
        self.facetboundaries_me = self.Mesh.facetboundaries
        self.edgeboundaries_me = self.Mesh.edgeboundaries
        self.matid_me = self.Mesh.matid

        self.ds_me = self.Mesh.ds
        self.dx_me = self.Mesh.dx

        LVendoid = self.SimDet["LVendoid"]

        if isinstance(self.SimDet["LVendoid"], list):
            cnt = 0
            for id_ in self.SimDet["LVendoid"]:
                if cnt == 0:
                    dsendo = self.ds_me(
                        id_, domain=self.mesh_me, subdomain_data=self.facetboundaries_me
                    )
                else:
                    dsendo += self.ds_me(
                        id_, domain=self.mesh_me, subdomain_data=self.facetboundaries_me
                    )
                cnt += 1
        else:
            dsendo = self.ds_me(
                LVendoid, domain=self.mesh_me, subdomain_data=self.facetboundaries_me
            )
        self.LVendo_area_me = Expression(("val"), val=0.0, degree=2)
        self.LVendo_area_me.val = assemble(
            Constant(1.0) * dsendo,
            form_compiler_parameters={"representation": "uflacs"},
        )

        self.f0_me = f0_me_Gauss
        self.s0_me = s0_me_Gauss
        self.n0_me = n0_me_Gauss

        self.eL0_me = eL0_me_Gauss
        self.eC0_me = eC0_me_Gauss

        self.eclgn0_me = eclgn0_me_Gauss
        self.eclgn1_me = eclgn1_me_Gauss

        self.LVCavityvol = Expression(("vol"), vol=0.0, degree=2)
        self.RVCavityvol = Expression(("vol"), vol=0.0, degree=2)

        self.LVCavitypres = Expression(("pres"), pres=0.0, degree=2)
        self.RVCavitypres = Expression(("pres"), pres=0.0, degree=2)
        self.LACavitypres = Expression(("pres"), pres=0.0, degree=2)
        self.RACavitypres = Expression(("pres"), pres=0.0, degree=2)
        self.AortaCavitypres = Expression(("pres"), pres=0.0, degree=2)

        self.lumped_pres = 0.0
        self.lumped_vol = 0.0

        self.isincomp = SimDet["GiccioneParams"]["incompressible"]
        #  - - - - - - - - - - - -- - - - - - - - - - - - - - - -- - - - - - -

        if self.discretization == "P1P1":
            Velem = VectorElement(
                "CG", self.mesh_me.ufl_cell(), 1, quad_scheme="default"
            )
        else:
            Velem = VectorElement(
                "CG", self.mesh_me.ufl_cell(), 2, quad_scheme="default"
            )

        Qelem = FiniteElement("CG", self.mesh_me.ufl_cell(), 1, quad_scheme="default")
        Qelem._quad_scheme = "default"
        Relem = FiniteElement("Real", self.mesh_me.ufl_cell(), 0, quad_scheme="default")
        Relem._quad_scheme = "default"
        Quadelem = FiniteElement(
            "Quadrature",
            self.mesh_me.ufl_cell(),
            degree=self.deg_me,
            quad_scheme="default",
        )
        Quadelem._quad_scheme = "default"
        #  - - - - - - - - - - - -- - - - - - - - - - - - - - - -- - - - - - -

        #  - - - - - - - - - - - -- - - - - - - - - - - - - - - -- - - - - - -
        Telem2 = TensorElement(
            "Quadrature",
            self.mesh_me.ufl_cell(),
            degree=self.deg_me,
            shape=2 * (3,),
            quad_scheme="default",
        )
        Telem2._quad_scheme = "default"
        for e in Telem2.sub_elements():
            e._quad_scheme = "default"
        Telem4 = TensorElement(
            "Quadrature",
            self.mesh_me.ufl_cell(),
            degree=self.deg_me,
            shape=4 * (3,),
            quad_scheme="default",
        )
        Telem4._quad_scheme = "default"
        for e in Telem4.sub_elements():
            e._quad_scheme = "default"
        #  - - - - - - - - - - - -- - - - - - - - - - - - - - - -- - - - - - -

        # Mixed Element for rigid body motion
        #  - - - - - - - - - - - -- - - - - - - - - - - - - - - -- - - - - - -
        VRelem = MixedElement([Relem, Relem, Relem, Relem, Relem])
        #  - - - - - - - - - - - -- - - - - - - - - - - - - - - -- - - - - - -

        #  - - - - - - - - - - - -- - - - - - - - - - - - - - - -- - - - - - -
        if self.isincomp:
            if self.isLV or self.iswaorta:
                if self.ispctrl:
                    if (
                        "springbc" in list(self.SimDet.keys())
                        and self.SimDet["springbc"]
                    ):
                        self.W = FunctionSpace(
                            self.mesh_me, MixedElement([Velem, Qelem])
                        )
                    else:
                        self.W = FunctionSpace(
                            self.mesh_me, MixedElement([Velem, Qelem, VRelem])
                        )
                else:
                    if (
                        "springbc" in list(self.SimDet.keys())
                        and self.SimDet["springbc"]
                    ):
                        self.W = FunctionSpace(
                            self.mesh_me, MixedElement([Velem, Qelem, Relem])
                        )
                    else:
                        self.W = FunctionSpace(
                            self.mesh_me, MixedElement([Velem, Qelem, Relem, VRelem])
                        )
            elif self.isBiV or self.isFCH:
                if self.ispctrl:
                    if (
                        "springbc" in list(self.SimDet.keys())
                        and self.SimDet["springbc"]
                    ):
                        self.W = FunctionSpace(
                            self.mesh_me, MixedElement([Velem, Qelem])
                        )
                    else:
                        self.W = FunctionSpace(
                            self.mesh_me, MixedElement([Velem, Qelem, VRelem])
                        )
                else:
                    if (
                        "springbc" in list(self.SimDet.keys())
                        and self.SimDet["springbc"]
                    ):
                        self.W = FunctionSpace(
                            self.mesh_me, MixedElement([Velem, Qelem, Relem, Relem])
                        )
                    else:
                        self.W = FunctionSpace(
                            self.mesh_me,
                            MixedElement([Velem, Qelem, Relem, Relem, VRelem]),
                        )
        else:
            if self.isLV or self.iswaorta:
                if self.ispctrl:
                    if (
                        "springbc" in list(self.SimDet.keys())
                        and self.SimDet["springbc"]
                    ):
                        self.W = FunctionSpace(self.mesh_me, MixedElement([Velem]))
                    else:
                        self.W = FunctionSpace(
                            self.mesh_me, MixedElement([Velem, VRelem])
                        )
                else:
                    if (
                        "springbc" in list(self.SimDet.keys())
                        and self.SimDet["springbc"]
                    ):
                        self.W = FunctionSpace(
                            self.mesh_me, MixedElement([Velem, Relem])
                        )
                    else:
                        self.W = FunctionSpace(
                            self.mesh_me, MixedElement([Velem, Relem, VRelem])
                        )
            elif self.isBiV or self.isFCH:
                if self.ispctrl:
                    if (
                        "springbc" in list(self.SimDet.keys())
                        and self.SimDet["springbc"]
                    ):
                        self.W = FunctionSpace(self.mesh_me, MixedElement([Velem]))
                    else:
                        self.W = FunctionSpace(
                            self.mesh_me, MixedElement([Velem, VRelem])
                        )
                else:
                    if (
                        "springbc" in list(self.SimDet.keys())
                        and self.SimDet["springbc"]
                    ):
                        self.W = FunctionSpace(
                            self.mesh_me, MixedElement([Velem, Relem, Relem])
                        )
                    else:
                        self.W = FunctionSpace(
                            self.mesh_me, MixedElement([Velem, Relem, Relem, VRelem])
                        )

        self.Quad = FunctionSpace(self.mesh_me, Quadelem)
        self.TF = FunctionSpace(self.mesh_me, Telem2)
        self.Q = FunctionSpace(self.mesh_me, "CG", 1)
        self.QDG = FunctionSpace(self.mesh_me, "DG", 0)
        self.V_CG1 = VectorFunctionSpace(self.mesh_me, "CG", 1)

        self.we_n = Function(self.W.sub(0).collapse())

        self.w_me = Function(self.W)
        self.w_me_n = Function(self.W)
        self.dw_me = TrialFunction(self.W)
        self.wtest_me = TestFunction(self.W)

        self.Ftotal, self.Jac, self.bcs = self.Problem()

    def default_parameters(self):
        return {"probeloc": [3.5, 0.0, -2.0]}

    def unloading_pres(self, params):
        default_params = {
            "EDP": 12,
            "maxit": 20,
            "restol": 1e-3,
            "drestol": 1e-4,
            "preinc": 1,
            "LVangle": [60, -60],
        }
        default_params.update(params)
        EDP = default_params["EDP"]
        preinc = default_params["preinc"]
        it = 0
        while 1:
            if self.LVCavityrpes.pres > EDP:
                break

            self.LVCavitypres.pres += 0.1
            self.Solver.solvenonlinear()

        return preinc

    def unloading(self, params):
        default_params = {
            "EDP": 12,
            "maxit": 20,
            "restol": 1e-3,
            "drestol": 1e-4,
            "EDPtol": 1e-1,
            "volinc": 1,
            "LVangle": [60, -60],
        }
        default_params.update(params)
        EDP = default_params["EDP"]
        LVangle = default_params["LVangle"]
        maxit = default_params["maxit"]
        restol = default_params["restol"]
        drestol = default_params["drestol"]
        EDPtol = default_params["EDPtol"]
        LVendoid = self.SimDet["LVendoid"]
        volinc = default_params["volinc"]
        outputfolder = self.parameters["outputfolder"]
        folderName = self.parameters["foldername"]
        outfolder = outputfolder + folderName + "deformation_unloadED/"

        targetmesh = Mesh(self.Mesh.mesh)
        xtarget = project(SpatialCoordinate(self.mesh_me), self.V_CG1).vector()

        comm_me = self.mesh_me.mpi_comm()

        if MPI.rank(comm_me) == 0:
            if not os.path.isdir(outfolder):
                os.makedirs(outfolder)

        if MPI.rank(comm_me) == 0:
            fdataPV = open(outfolder + "BiV_unloadPV.txt", "w", 0)
        hdf = HDF5File(comm_me, outfolder + "Data_unload.h5", "w")

        it = 0
        res = 1e9
        dres = 0
        alpha = 1.0

        while 1:
            it_load = 0
            self.LVCavityvol.vol = self.GetLVV()
            LVP = self.GetLVP() * 0.0075
            LVV = self.GetLVV()
            printout("Iteration number = " + str(it), comm_me)
            printout("Pressure = " + str(LVP) + " Vol = " + str(LVV), comm_me)

            if MPI.rank(comm_me) == 0:
                print(it, LVP, LVV, file=fdataPV)

            hdf.write(self.mesh_me, "unloading" + str(it) + "/mesh")
            hdf.write(
                self.GetDisplacement(), "unloading" + str(it) + "/u_loading", it_load
            )

            while 1:
                self.LVCavityvol.vol += volinc
                self.Solver().solvenonlinear()

                LVP = self.GetLVP() * 0.0075
                LVV = self.GetLVV()

                if LVP > EDP:
                    print("Decrease load step")
                    self.LVCavityvol.vol -= volinc
                    volinc = volinc / 2.0
                    continue

                printout("Loading iteration number = " + str(it_load), comm_me)
                printout("Pressure = " + str(LVP) + " Vol = " + str(LVV), comm_me)

                if MPI.rank(comm_me) == 0:
                    print(it, LVP, LVV, file=fdataPV)

                hdf.write(
                    self.GetDisplacement(),
                    "unloading" + str(it) + "/u_loading",
                    it_load,
                )

                if abs(LVP - EDP) < EDPtol:
                    # Get Residual
                    x = project(
                        SpatialCoordinate(self.mesh_me) + self.GetDisplacement(),
                        self.V_CG1,
                    ).vector()

                    res_new = norm(x - xtarget, "L2")
                    dres = abs(res_new - res)
                    res = res_new
                    printout(
                        "Residual = " + str(res) + " dResidual = " + str(dres), comm_me
                    )

                    # Reset volume increment
                    volinc = default_params["volinc"]

                    break

                it_load += 1

            if it > maxit or res < restol or dres < drestol:
                self.Reset()
                break

            else:
                dispCG1 = project(-alpha * self.GetDisplacement(), self.V_CG1)
                newmesh, newboundaries = update_mesh(
                    targetmesh, dispCG1, self.facetboundaries_me
                )

                # Update mesh
                self.Reset()
                self.mesh_me.coordinates()[:, 0] = newmesh.coordinates()[:, 0]
                self.mesh_me.coordinates()[:, 1] = newmesh.coordinates()[:, 1]
                self.mesh_me.coordinates()[:, 2] = newmesh.coordinates()[:, 2]

                self.facetboundaries_me.set_values(newboundaries.array())

                # Update LV endo surface area for imposing BC
                dsendo = self.ds_me(
                    LVendoid,
                    domain=self.mesh_me,
                    subdomain_data=self.facetboundaries_me,
                )
                self.LVendo_area_me.val = assemble(
                    Constant(1.0) * dsendo,
                    form_compiler_parameters={"representation": "uflacs"},
                )
                self.mesh_me.bounding_box_tree().build(self.mesh_me)

                # Update fiber
                default_params.update({"meshName": "unloadfiber_" + str(it)})
                (
                    f0_me_Gauss,
                    s0_me_Gauss,
                    n0_me_Gauss,
                    deformedMesh,
                    deformedBoundary,
                ) = self.GetDeformedBasis(default_params)
                self.f0_me = f0_me_Gauss
                self.s0_me = s0_me_Gauss
                self.n0_me = n0_me_Gauss

                self.f0_me = self.f0_me / sqrt(inner(self.f0_me, self.f0_me))
                self.s0_me = self.s0_me / sqrt(inner(self.s0_me, self.s0_me))
                self.n0_me = self.n0_me / sqrt(inner(self.n0_me, self.n0_me))

                # Restate problem
                self.Ftotal, self.Jac, self.bcs = self.Problem()

                it += 1

        # Write mesh
        f = HDF5File(comm_me, outfolder + "UnloadMesh.hdf5", "w")
        f.write(self.mesh_me, "UnloadMesh")
        f.close()

        f = dolfin.HDF5File(comm_me, outfolder + "UnloadMesh.hdf5", "a")
        f.write(self.facetboundaries_me, "UnloadMesh" + "/" + "facetboundaries")
        f.write(self.edgeboundaries_me, "UnloadMesh" + "/" + "edgeboundaries")
        f.write(f0_me_Gauss, "UnloadMesh" + "/" + "eF")
        f.write(s0_me_Gauss, "UnloadMesh" + "/" + "eS")
        f.write(n0_me_Gauss, "UnloadMesh" + "/" + "eN")

        if hasattr(self.Mesh, "eC0"):
            f.write(self.Mesh.eC0, "UnloadMesh" + "/" + "eC")
        if hasattr(self.Mesh, "eL0"):
            f.write(self.Mesh.eL0, "UnloadMesh" + "/" + "eL")
        if hasattr(self.Mesh, "eR0"):
            f.write(self.Mesh.eR0, "UnloadMesh" + "/" + "eR")

        f.write(self.Mesh.matid, "UnloadMesh" + "/" + "matid")

        # np.savez(outfolder+"Udata.npz", f0_me_Gauss=f0_me_Gauss.vector().get_local()[:], \
        #                               s0_me_Gauss=s0_me_Gauss.vector().get_local()[:], \
        #                               n0_me_Gauss=n0_me_Gauss.vector().get_local()[:], \
        #        )
        # File(outfolder+"facetboundaries.pvd") << self.facetboundaries_me
        # File(outfolder+"mesh.pvd") << self.mesh_me
        f.close()

        os.system(
            "cp "
            + outfolder
            + "UnloadMesh.hdf5"
            + " "
            + outfolder
            + "UnloadMesh_refine.hdf5"
        )

        return it_load, volinc, LVV, self.Solver()

    def set_BCs(self):
        #  - - - - - - - - - - - -- - - - - - - - - - - - - - - -- - - - - - -
        # Using bubble element
        #  - - - - - - - - - - - -- - - - - - - - - - - - - - - -- - - - - - -
        # baseconstraint = project(Expression(("0.0"), degree=2), W.sub(0).sub(2).collapse())
        # bctop = DirichletBC(W.sub(0).sub(2), baseconstraint, facetboundaries, topid)
        #  - - - - - - - - - - - -- - - - - - - - - - - - - - - -- - - - - - -

        facetboundaries = self.facetboundaries_me
        edgeboundaries = self.edgeboundaries_me

        # isLV or isBiV
        topid = self.SimDet.get("topid")

        W = self.W

        if self.iswaorta:
            aorta_ext_wall = self.SimDet["aorta_ext_wall"]
            bc_aorta_ext_wall = DirichletBC(
                W.sub(0),
                Expression(("0.0", "0.0", "0.0"), degree=2),
                facetboundaries,
                aorta_ext_wall,
            )

            aorta_int_wall = self.SimDet["aorta_int_wall"]
            bc_aorta_int_wall = DirichletBC(
                W.sub(0),
                Expression(("0.0", "0.0", "0.0"), degree=2),
                facetboundaries,
                aorta_int_wall,
            )

            aorta_ring = self.SimDet["aorta_ring"]
            bc_aorta_ring = DirichletBC(
                W.sub(0),
                Expression(("0.0", "0.0", "0.0"), degree=2),
                facetboundaries,
                aorta_ring,
            )

        elif self.isLV or self.isBiV:

            bctop = DirichletBC(
                W.sub(0).sub(2),
                Expression(("0.0"), degree=2),
                facetboundaries,
                topid,
            )

            if "fix_surf" in self.SimDet.keys():
                if isinstance(self.SimDet["fix_surf"], list):
                    cnt = 0
                    bc_fix = []
                    for id_ in self.SimDet["fix_surf"]:
                        bc_fix.append(
                            DirichletBC(
                                W.sub(0),
                                Expression(("0.0", "0.0", "0.0"), degree=2),
                                facetboundaries,
                                id_,
                            )
                        )
                else:
                    fix_surf = self.SimDet["fix_surf"]
                    bc_fix = DirichletBC(
                        W.sub(0),
                        Expression(("0.0", "0.0", "0.0"), degree=2),
                        facetboundaries,
                        fix_surf,
                    )
            else:
                bc_fix = None

        elif self.isFCH:
            if self.SimDet.get("pulm_wall"):
                bc_pulm_wall = DirichletBC(
                    W.sub(0),
                    Expression(("0.0", "0.0", "0.0"), degree=2),
                    facetboundaries,
                    pulm_wall,
                )
            else:
                bc_pulm_wall = None

            aorta_wall = self.SimDet["aorta_wall"]
            bc_aorta_wall = DirichletBC(
                W.sub(0),
                Expression(("0.0", "0.0", "0.0"), degree=2),
                facetboundaries,
                aorta_wall,
            )

        # endoring = pick_endoring_bc(method="cpp")(edgeboundaries, 1)

        if "springbc" in list(self.SimDet.keys()) and self.SimDet["springbc"]:
            if self.iswaorta:
                if "mv_aorta" in list(self.SimDet.keys()) and self.SimDet["mv_aorta"]:
                    bcs = []
                else:
                    bcs = [bc_aorta_ring]
            elif self.isLV or self.isBiV:
                bcs = []
                if bc_fix is not None:
                    if isinstance(self.SimDet["fix_surf"], list):
                        for bc_fix_ in bc_fix:
                            bcs.append(bc_fix_)
                    else:
                        bcs.append(bc_fix)

            elif self.isFCH:
                bcs = [bc_aorta_wall]  # , bc_pulm_wall]
                # bcs = []
        else:
            if self.iswaorta:
                bcs = [bc_aorta_ring]
            elif self.isLV or self.isBiV:
                bcs = [bctop]  # LCL
                # bcs = []
                if bc_fix is not None:
                    if isinstance(self.SimDet["fix_surf"], list):
                        for bc_fix_ in bc_fix:
                            bcs.append(bc_fix_)
                    else:
                        bcs.append(bc_fix)
            elif self.isFCH:
                bcs = []
        return bcs
        #  - - - - - - - - - - - -- - - - - - - - - - - - - - - -- - - - - - -

    def Problem(self):
        GuccioneParams = self.SimDet["GiccioneParams"]
        aorta_params = GuccioneParams.get("Aorta params")

        comm_me = self.mesh_me.mpi_comm()

        # isLV or isBiV
        topid = self.SimDet.get("topid")

        # iswaorta or isFCH
        aortic_valvep = self.SimDet.get("aortic_valvep")
        mitral_valvep = self.SimDet.get("mitral_valvep")
        aortaid = self.SimDet.get("aortaid")
        apxid = self.SimDet.get("apxid")

        # isFCH
        septumid = self.SimDet.get("septumid")
        aorta_wall = self.SimDet.get("aorta_wall")
        pulm_wall = self.SimDet.get("pulm_wall")
        pulmonary_valvep = self.SimDet.get("pulmonary_valvep")
        tricuspid_valvep = self.SimDet.get("tricuspid_valvep")

        # iswaorta
        aorta_int_wall = self.SimDet.get("aorta_int_wall")
        aorta_ext_wall = self.SimDet.get("aorta_ext_wall")
        aorta_ring = self.SimDet.get("aorta_ring")

        LVendoid = self.SimDet["LVendoid"]
        RVendoid = self.SimDet["RVendoid"]

        LAendoid = self.SimDet.get("LAendoid")
        RAendoid = self.SimDet.get("RAendoid")

        epiid = self.SimDet["epiid"]
        atrialid = self.SimDet["atrialid"]

        if not "LVPid" in list(self.SimDet.keys()):
            LVPid = self.SimDet["LVendoid"]
        else:
            LVPid = self.SimDet["LVPid"]

        if not "RVPid" in list(self.SimDet.keys()):
            RVPid = self.SimDet["RVendoid"]
        else:
            RVPid = self.SimDet["RVPid"]

        isincomp = GuccioneParams["incompressible"]
        deg_me = GuccioneParams["deg"]

        mesh_me = self.mesh_me
        facetboundaries_me = self.facetboundaries_me
        f0_me = self.f0_me
        s0_me = self.s0_me
        n0_me = self.n0_me

        eL0_me = self.eL0_me
        eC0_me = self.eC0_me

        eclgn0_me = self.eclgn0_me
        eclgn1_me = self.eclgn1_me

        N_me = FacetNormal(mesh_me)
        W_me = self.W
        Q_me = self.Q
        TF_me = self.TF

        w_me = self.w_me
        w_me_n = self.w_me_n
        dw_me = self.dw_me
        wtest_me = self.wtest_me

        bcs_elas = self.set_BCs()

        if isincomp:
            if self.isLV or self.iswaorta:
                if self.ispctrl:
                    if (
                        "springbc" in list(self.SimDet.keys())
                        and self.SimDet["springbc"]
                    ):
                        du, dp = TrialFunctions(W_me)
                        (u_me, p_me) = split(w_me)
                        (u_me_n, p_me_n) = split(w_me_n)
                        (v_me, q_me) = TestFunctions(W_me)
                        lv_pendo = []
                        rv_pendo = []
                        LVendo_comp = 2
                        RVendo_comp = 1000

                    else:
                        du, dp, dc = TrialFunctions(W_me)
                        (u_me, p_me, c_me) = split(w_me)
                        (u_me_n, p_me_n, c_me_n) = split(w_me_n)
                        (v_me, q_me, cq) = TestFunctions(W_me)
                        lv_pendo = []
                        rv_pendo = []
                        LVendo_comp = 2
                        RVendo_comp = 1000

                else:
                    if (
                        "springbc" in list(self.SimDet.keys())
                        and self.SimDet["springbc"]
                    ):
                        du, dp, dlv_pendo = TrialFunctions(W_me)
                        (u_me, p_me, lv_pendo) = split(w_me)
                        (u_me_n, p_me_n, lv_pendo_n) = split(w_me_n)
                        (v_me, q_me, lv_qendo) = TestFunctions(W_me)
                        rv_pendo = []
                        LVendo_comp = 2
                        RVendo_comp = 1000

                    else:
                        du, dp, dlv_pendo, dc = TrialFunctions(W_me)
                        (u_me, p_me, lv_pendo, c_me) = split(w_me)
                        (u_me_n, p_me_n, lv_pendo_n, c_me_n) = split(w_me_n)
                        (v_me, q_me, lv_qendo, cq) = TestFunctions(W_me)
                        rv_pendo = []
                        LVendo_comp = 2
                        RVendo_comp = 1000

            elif self.isBiV or self.isFCH:
                if self.ispctrl:
                    if (
                        "springbc" in list(self.SimDet.keys())
                        and self.SimDet["springbc"]
                    ):
                        du, dp = TrialFunctions(W_me)
                        (u_me, p_me) = split(w_me)
                        (u_me_n, p_me_n) = split(w_me_n)
                        (v_me, q_me) = TestFunctions(W_me)
                        LVendo_comp = 2
                        RVendo_comp = 3
                        lv_pendo = []
                        rv_pendo = []

                    else:
                        du, dp, dc = TrialFunctions(W_me)
                        (u_me, p_me, c_me) = split(w_me)
                        (u_me_n, p_me_n, c_me_n) = split(w_me_n)
                        (v_me, q_me, cq) = TestFunctions(W_me)
                        LVendo_comp = 2
                        RVendo_comp = 3
                        lv_pendo = []
                        rv_pendo = []

                else:
                    if (
                        "springbc" in list(self.SimDet.keys())
                        and self.SimDet["springbc"]
                    ):
                        du, dp, dlv_pendo, drv_pendo = TrialFunctions(W_me)
                        (u_me, p_me, lv_pendo, rv_pendo) = split(w_me)
                        (u_me_n, p_me_n, lv_pendo_n, rv_pendo_n) = split(w_me_n)
                        (v_me, q_me, lv_qendo, rv_qendo) = TestFunctions(W_me)
                        LVendo_comp = 2
                        RVendo_comp = 3

                    else:
                        du, dp, dlv_pendo, drv_pendo, dc = TrialFunctions(W_me)
                        (u_me, p_me, lv_pendo, rv_pendo, c_me) = split(w_me)
                        (u_me_n, p_me_n, lv_pendo_n, rv_pendo_n, c_me_n) = split(w_me_n)
                        (v_me, q_me, lv_qendo, rv_qendo, cq) = TestFunctions(W_me)
                        LVendo_comp = 2
                        RVendo_comp = 3
        else:
            if self.isLV or self.iswaorta:
                if self.ispctrl:
                    if (
                        "springbc" in list(self.SimDet.keys())
                        and self.SimDet["springbc"]
                    ):
                        du = TrialFunctions(W_me)
                        (u_me) = split(w_me)
                        (v_me) = TestFunctions(W_me)
                        p_me = Function(Q_me)
                        lv_pendo = []
                        rv_pendo = []
                        LVendo_comp = 1
                        RVendo_comp = 1000
                    else:
                        du, dc = TrialFunctions(W_me)
                        (u_me, c_me) = split(w_me)
                        (v_me, cq) = TestFunctions(W_me)
                        p_me = Function(Q_me)
                        lv_pendo = []
                        rv_pendo = []
                        LVendo_comp = 1
                        RVendo_comp = 1000
                else:
                    if (
                        "springbc" in list(self.SimDet.keys())
                        and self.SimDet["springbc"]
                    ):
                        du, dlv_pendo = TrialFunctions(W_me)
                        (u_me, lv_pendo) = split(w_me)
                        (v_me, lv_qendo) = TestFunctions(W_me)
                        p_me = Function(Q_me)
                        rv_pendo = []
                        LVendo_comp = 1
                        RVendo_comp = 1000
                    else:
                        du, dlv_pendo, dc = TrialFunctions(W_me)
                        (u_me, lv_pendo, c_me) = split(w_me)
                        (v_me, lv_qendo, cq) = TestFunctions(W_me)
                        p_me = Function(Q_me)
                        rv_pendo = []
                        LVendo_comp = 1
                        RVendo_comp = 1000
            elif self.isBiV or self.isFCH:
                if self.ispctrl:
                    if (
                        "springbc" in list(self.SimDet.keys())
                        and self.SimDet["springbc"]
                    ):
                        du = TrialFunctions(W_me)
                        (u_me) = split(w_me)
                        (u_me_n) = split(w_me_n)
                        (v_me) = TestFunctions(W_me)
                        p_me = Function(Q_me)
                        lv_pendo = []
                        rv_pendo = []
                        LVendo_comp = 1
                        RVendo_comp = 2

                    else:
                        du, dlv_pendo, drv_pendo, dc = TrialFunctions(W_me)
                        (u_me, lv_pendo, rv_pendo, c_me) = split(w_me)
                        (v_me, lv_qendo, rv_qendo, cq) = TestFunctions(W_me)
                        p_me = Function(Q_me)
                        lv_pendo = []
                        rv_pendo = []
                        LVendo_comp = 1
                        RVendo_comp = 2
                else:
                    if (
                        "springbc" in list(self.SimDet.keys())
                        and self.SimDet["springbc"]
                    ):
                        du, dlv_pendo, drv_pendo = TrialFunctions(W_me)
                        (u_me, lv_pendo, rv_pendo) = split(w_me)
                        (v_me, lv_qendo, rv_qendo) = TestFunctions(W_me)
                        p_me = Function(Q_me)
                        LVendo_comp = 1
                        RVendo_comp = 2

                    else:
                        du, dlv_pendo, drv_pendo, dc = TrialFunctions(W_me)
                        (u_me, lv_pendo, rv_pendo, c_me) = split(w_me)
                        (v_me, lv_qendo, rv_qendo, cq) = TestFunctions(W_me)
                        p_me = Function(Q_me)
                        LVendo_comp = 1
                        RVendo_comp = 2

        self.t_a = Function(self.Quad)
        self.t_a.vector()[:] = 0

        # ls0 = 1.85
        # Tact = GuccioneParams["Tmax"]

        ds_me = self.ds_me
        dx_me = self.dx_me
        LVendo_area_me = self.LVendo_area_me

        params = {
            "mesh": mesh_me,
            "facetboundaries": facetboundaries_me,
            "facet_normal": N_me,
            "mixedfunctionspace": W_me,
            "mixedfunction": w_me,
            "displacement_variable": u_me,
            "pressure_variable": p_me,
            "lv_volconst_variable": lv_pendo,
            "lv_constrained_vol": self.LVCavityvol,
            "rv_volconst_variable": rv_pendo,
            "rv_constrained_vol": self.RVCavityvol,
            "LVendoid": LVendoid,
            "RVendoid": RVendoid,
            "LAendoid": LAendoid,
            "RAendoid": RAendoid,
            "epiid": epiid,
            "atrialid": atrialid,
            "topid": topid,
            "aortaid": aortaid,
            "LVPid": LVPid,
            "RVPid": RVPid,
            "aortic_valvep": aortic_valvep,
            "mitral_valvep": mitral_valvep,
            "pulmonary_valvep": pulmonary_valvep,
            "tricuspid_valvep": tricuspid_valvep,
            "septumid": septumid,
            "aorta_wall": aorta_wall,
            "pulm_wall": pulm_wall,
            "apxid": apxid,
            "aorta_int_wall": aorta_int_wall,
            "aorta_ext_wall": aorta_ext_wall,
            "aorta_ring": aorta_ring,
            "LVendo_comp": LVendo_comp,
            "RVendo_comp": RVendo_comp,
            "fiber": f0_me,
            "sheet": s0_me,
            "sheet-normal": n0_me,
            "fiberz-aorta": eL0_me,
            "fiberc-aorta": eC0_me,
            "fiberclgn0-aorta": eclgn0_me,
            "fiberclgn1-aorta": eclgn1_me,
            "growth_tensor": None,
            "material model": GuccioneParams["Passive model"],
            "material params": GuccioneParams["Passive params"],
            "aorta params": aorta_params,
            "incompressible": GuccioneParams["incompressible"],
            "LVendo_area": LVendo_area_me,
            "lv_constrained_pres": self.LVCavitypres,
            "rv_constrained_pres": self.RVCavitypres,
            "la_constrained_pres": self.LACavitypres,
            "ra_constrained_pres": self.RACavitypres,
            "aorta_constrained_pres": self.AortaCavitypres,
        }

        uflforms = Forms(params)
        self.uflforms = uflforms

        activeparams = {
            "mesh": mesh_me,
            "dx": dx_me,
            "deg": GuccioneParams["deg"],
            "facetboundaries": facetboundaries_me,
            "facet_normal": N_me,
            "displacement_variable": u_me,
            "pressure_variable": p_me,
            "fiber": f0_me,
            "sheet": s0_me,
            "sheet-normal": n0_me,
            "t_a": self.t_a,
            "Threshold_Potential": 0.9,
            "growth_tensor": None,
        }

        if "Active model" in list(GuccioneParams.keys()):
            activeparams.update({"material model": GuccioneParams["Active model"]})

        if "Active params" in list(GuccioneParams.keys()):
            activeparams.update({"material params": GuccioneParams["Active params"]})

        if "HomogenousActivation" in list(GuccioneParams.keys()):
            activeparams.update(
                {"HomogenousActivation": GuccioneParams["HomogenousActivation"]}
            )
        else:
            activeparams.update({"HomogenousActivation": True})

        activeforms = activeForms(activeparams)
        self.activeforms = activeforms

        F_ED = Function(TF_me)

        Fmat = uflforms.Fmat()
        Cmat = Fmat.T * Fmat
        Emat = uflforms.Emat()
        J = uflforms.J()

        n_me = J * inv(Fmat.T) * N_me

        Wp_me = uflforms.PassiveMatSEF()
        WpRub_me = uflforms.PassiveRubSEF()
        WpAorta_me = uflforms.PassiveAortaSEF()

        if not self.ispctrl:
            LV_Wvol = uflforms.LVV0constrainedE()
            if self.isBiV:
                RV_Wvol = uflforms.RVV0constrainedE()

        # Sactive = activeforms.PK2StressTensor()

        # printout("Total active force = " + str(assemble(activeforms.PK1Stress()*dx_me)), comm_me)

        X_me = SpatialCoordinate(mesh_me)

        state_obj = self.parameters["state_obj"]

        def poro_Forms():
            if "poro" in list(self.SimDet.keys()):
                permeability = self.SimDet["permeability"]
                p_a = self.SimDet["p_a"]  # prefusion_pressure
                p_v = self.SimDet["p_v"]  # ???
                beta_a = self.SimDet["beta_a"]
                beta_v = self.SimDet["beta_v"]
            else:
                permeability = 0
                p_a = 0
                p_v = 0
                beta_a = 0
                beta_v = 0

            p_a = p_a * 135  # perfusion pressure # converted to mmHg
            p_v = p_v * 135  # Q: needed?
            source_ = (
                J * beta_a * (p_a - uflforms.poro_pressure()) * q_me * dx_me
                - J * beta_v * (uflforms.poro_pressure() - p_v) * q_me * dx_me
            )

            F_1 = (1 / state_obj.dt.dt) * (p_me - p_me_n) * q_me * dx_me + dot(
                permeability * grad(uflforms.poro_pressure()), grad(q_me)
            ) * dx_me
            F_2 = (
                self.LVCavitypres * inner(v_me, N_me) * ds_me(LVendoid)
                + inner(grad(v_me), uflforms.poro_PK_1()) * dx_me
            )

            return F_1, F_2

        poro_F1, poro_F2 = poro_Forms()
        # Assem_poro_F4 = assemble(poro_F4)
        # print(Assem_poro_F4.get_local())

        # if "active_region" in list(self.SimDet.keys()) and self.SimDet["active_region"]:
        #     nonLVid = set(self.matid_me.array()) - set(self.SimDet["active_region"])
        if self.iswaorta:

            if (
                "active_region" in list(self.SimDet.keys())
                and self.SimDet["active_region"]
            ):
                region_cnt = 0
                for regionid in self.SimDet["active_region"]:
                    if region_cnt == 0:
                        F1 = derivative(Wp_me, w_me, wtest_me) * (dx_me(int(regionid)))
                    else:
                        F1 += derivative(Wp_me, w_me, wtest_me) * (dx_me(int(regionid)))
                    region_cnt += 1
            if (
                "rubber_region" in list(self.SimDet.keys())
                and self.SimDet["rubber_region"]
            ):
                for regionid in self.SimDet["rubber_region"]:
                    F1 += derivative(Wp_me, w_me, wtest_me) * dx_me(int(regionid))
            if (
                "aorta_region" in list(self.SimDet.keys())
                and self.SimDet["aorta_region"]
            ):
                for regionid in self.SimDet["aorta_region"]:
                    F1 += derivative(WpAorta_me, w_me, wtest_me) * dx_me(int(regionid))
            # for nonLVid_ in list(nonLVid):
            #    F1 += derivative(Wp_me, w_me, wtest_me) * dx_me(int(nonLVid_))

        elif "poro" in list(self.SimDet.keys()):
            F1 = poro_F1
        elif self.SimDet.get("fch_fe"):  # temp coz we'll have atrial fibers son
            # F1 = derivative(WpRub_me, w_me, wtest_me) * dx_me
            region_cnt = 0
            for regionid in self.SimDet["active_region"]:
                factor = 0.0 if regionid in (3, 4) else 1.0
                if region_cnt == 0:
                    if factor == 1:
                        F1 = derivative(Wp_me, w_me, wtest_me) * dx_me
                    else:
                        F1 = derivative(WpRub_me, w_me, wtest_me) * dx_me
                else:
                    if factor == 1:
                        F1 += derivative(Wp_me, w_me, wtest_me) * dx_me
                    else:
                        F1 += derivative(WpRub_me, w_me, wtest_me) * dx_me
                region_cnt += 1

        else:  # not iswaorta, not poro, not fch_fe
            F1 = derivative(Wp_me, w_me, wtest_me) * dx_me

        if "active_region" in list(self.SimDet.keys()):
            print("Active region = ", self.SimDet["active_region"])
            region_cnt = 0
            for regionid in self.SimDet["active_region"]:
                factor = {3: 0.2, 4: 0.1}.get(regionid, 1.0)  # hdf5-specific
                if int(factor) == 1:
                    Sactive = activeforms.PK2StressTensor()
                else:
                    Sactive = activeforms.PK2StressTensor_atr() # *
                if region_cnt == 0:
                    # F4 = inner(Fmat * Sactive, grad(v_me)) * (dx_me(int(regionid)))
                    F4 = (
                        factor
                        * inner(Fmat * Sactive, grad(v_me))
                        * dx_me(int(regionid))
                    )
                    print("Assigning active stress to ", regionid)
                else:
                    # F4 += inner(Fmat * Sactive, grad(v_me)) * (dx_me(int(regionid)))
                    F4 += (
                        factor
                        * inner(Fmat * Sactive, grad(v_me))
                        * dx_me(int(regionid))
                    )
                    print("Assigning active stress to ", regionid)

                region_cnt += 1

        else:
            F4 = inner(Fmat * Sactive, grad(v_me)) * dx_me

        Ftotal = F1 + F4

        if not self.ispctrl:
            if self.isLV:
                F2 = derivative(LV_Wvol, w_me, wtest_me)
            elif self.isBiV:
                F2 = derivative(LV_Wvol + RV_Wvol, w_me, wtest_me)

            Ftotal += F2

        else:  # pctrl
            if self.isLV or self.iswaorta:
                if "poro" in list(self.SimDet.keys()):
                    Fp = poro_F2
                else:  # no poroelasticity
                    Fp_me = uflforms.LVcavitypres()
                    if self.SimDet.get("aorta_pres"):  # pressure in aorta
                        Fp_me += uflforms.Aortacavitypres()
                    Fp = derivative(Fp_me, w_me, wtest_me)
            elif self.isBiV or self.isFCH:
                Fp_lv_me = uflforms.LVcavitypres()
                Fp_rv_me = uflforms.RVcavitypres()
                Fp = derivative(Fp_lv_me, w_me, wtest_me) + derivative(
                    Fp_rv_me, w_me, wtest_me
                )
                if self.SimDet.get("fch_fe"):
                    Fp_la_me = uflforms.LAcavitypres()
                    Fp_ra_me = uflforms.RAcavitypres()
                    Fp += derivative(Fp_la_me, w_me, wtest_me) + derivative(
                        Fp_ra_me, w_me, wtest_me
                    )

            Ftotal += Fp

        if "springbc" in list(self.SimDet.keys()) and self.SimDet["springbc"]:
            if "springparam" in list(self.SimDet.keys()):
                k_spring = self.SimDet["springparam"]
                if "springfacets" in list(self.SimDet.keys()):
                    spr_facetids = self.SimDet["springfacets"]
            else:
                k_spring = [2.0e3, 2.0e3]  # default

            if "dashpotparam" in list(self.SimDet.keys()):
                c_damping = self.SimDet["dashpotparam"]
            else:
                c_damping = [2.0e2, 2.0e1]  # default

            if self.isLV:
                epiid_Kadj_coeff = self.SimDet.get("epiid_Kadj_coeff", [10.0, 10.0])
            else:
                epiid_Kadj_coeff = self.SimDet.get("epiid_Kadj_coeff", 1.0)
                atrialid_Kadj_coeff = self.SimDet.get("atrialid_Kadj_coeff", 1.0)


            if self.iswaorta:

                F3_epi = inner(
                    outer(n_me, n_me)
                    * (
                        epiid_Kadj_coeff * k_spring[0] * u_me
                        + c_damping[0] * (u_me - u_me_n)
                    ),
                    v_me,
                ) * (ds_me(epiid) + ds_me(apxid)) + inner(
                    (Identity(u_me.ufl_shape[0]) - outer(n_me, n_me))
                    * (
                        epiid_Kadj_coeff * k_spring[1] * u_me
                        + c_damping[1] * (u_me - u_me_n)
                    ),
                    v_me,
                ) * (
                    ds_me(epiid) + ds_me(apxid)
                )

                F3 = F3_epi

                if self.SimDet.get("mv_aorta"):

                    # kaorta_spring = self.SimDet["springaortaparam"]
                    # caorta_damping = self.SimDet["dashpotaortaparam"]
                    # aorta_ring = self.SimDet["aorta_ring"]

                    F3_aorta_ring = inner(
                        outer(n_me, n_me)
                        * (
                            epiid_Kadj_coeff * k_spring[0] * u_me
                            + c_damping[0] * (u_me - u_me_n)
                        ),
                        v_me,
                    ) * ds_me(aorta_ring) + inner(
                        (Identity(u_me.ufl_shape[0]) - outer(n_me, n_me))
                        * (
                            epiid_Kadj_coeff * k_spring[1] * u_me
                            + c_damping[1] * (u_me - u_me_n)
                        ),
                        v_me,
                    ) * (
                        ds_me(aorta_ring)
                    )

                    F3 += F3_aorta_ring

                if self.SimDet.get("spring_on_aorta"):
                    F3_aorta_wall = inner(
                        outer(n_me, n_me)
                        * (
                            epiid_Kadj_coeff * k_spring[0] * u_me
                            + c_damping[0] * (u_me - u_e_n)
                        ),
                        v_me,
                    ) * ds_me(aorta_ext_wall) + inner(
                        (Identity(u_me.ufl_shape[0]) - outer(n_me, n_me))
                        * (
                            epiid_Kadj_coeff * k_spring[1] * u_me
                            + c_damping[1] * (u_me - u_me_n)
                        ),
                        v_me,
                    ) * (
                        ds_me(aorta_ext_wall)
                    )

                    F3 += F3_aorta_wall

            elif self.isFCH:
                F3_epi = inner(
                    outer(n_me, n_me)
                    * (
                        k_spring[0] * epiid_Kadj_coeff * u_me
                        + c_damping[0] * (u_me - u_me_n)
                    ),
                    v_me,
                ) * (ds_me(epiid) + ds_me(apxid)) + inner(
                    (Identity(u_me.ufl_shape[0]) - outer(n_me, n_me))
                    * (
                        k_spring[1] * epiid_Kadj_coeff * u_me
                        + c_damping[1] * (u_me - u_me_n)
                    ),
                    v_me,
                ) * (
                    ds_me(epiid) + ds_me(apxid)
                )
                F3 = F3_epi

                F3_aorta = inner(
                    outer(N_me, N_me)
                    * (
                        k_spring[0] * epiid_Kadj_coeff * u_me
                        + c_damping[0] * (u_me - u_me_n)
                    ),
                    v_me,
                ) * ds_me(aortaid) + inner(
                    (Identity(u_me.ufl_shape[0]) - outer(n_me, n_me))
                    * (
                        k_spring[1] * epiid_Kadj_coeff * u_me
                        + c_damping[1] * (u_me - u_me_n)
                    ),
                    v_me,
                ) * ds_me(
                    aortaid
                )
                F3 += F3_aorta

                if self.SimDet.get("fch_fe"):
                    F3_atrial = inner(
                        outer(N_me, N_me)
                        * (
                            k_spring[0] * atrialid_Kadj_coeff * u_me
                            + c_damping[0] * (u_me - u_me_n)
                        ),
                        v_me,
                    ) * ds_me(atrialid) + inner(
                        (Identity(u_me.ufl_shape[0]) - outer(n_me, n_me))
                        * (
                            k_spring[1] * atrialid_Kadj_coeff * u_me
                            + c_damping[1] * (u_me - u_me_n)
                        ),
                        v_me,
                    ) * ds_me(
                        atrialid
                    )

                    F3 += F3_atrial

            elif self.isLV:

                # Laplace_u = self.GetLaplace()

                F3_epi = inner(
                    outer(N_me, N_me)
                    * (
                        k_spring[0] * epiid_Kadj_coeff[0] * self.Mesh.poissonF * u_me
                        + c_damping[0] * (u_me - u_me_n)
                    ),
                    v_me,
                ) * (ds_me(epiid)) + inner(
                    (Identity(u_me.ufl_shape[0]) - outer(N_me, N_me))
                    * (
                        k_spring[1] * epiid_Kadj_coeff[1] * u_me
                        + c_damping[1] * (u_me - u_me_n)
                    ),
                    v_me,
                ) * (
                    ds_me(epiid)
                )

                F3 = F3_epi

                # spring at base
                if (
                    "spring_atbase" in list(self.SimDet.keys())
                    and self.SimDet["spring_atbase"]
                ):
                    a_, b_ = self.GetTopSpring()
                    F3_base = a_ * inner(b_, v_me) * ds_me(topid)
                    F3 -= F3_base

            elif self.isBiV:

                F3_spr = 0
                for k_s, c_d, spr_f in zip(k_spring, c_damping, spr_facetids):
                    F3_spr += inner(
                        outer(N_me, N_me) * (k_s[0] * u_me + c_d[0] * (u_me - u_me_n)),
                        v_me,
                    ) * (ds_me(spr_f)) + inner(
                        (Identity(u_me.ufl_shape[0]) - outer(N_me, N_me))
                        * (k_s[1] * u_me + c_d[1] * (u_me - u_me_n)),
                        v_me,
                    ) * (
                        ds_me(spr_f)
                    )

                F3 = F3_spr

            Ftotal += F3  # LCL

        else:
            if self.isLV or self.isBiV:
                Wrigid = (
                    inner(as_vector([c_me[0], c_me[1], 0.0]), u_me)
                    + inner(as_vector([0.0, 0.0, c_me[2]]), cross(X_me, u_me))
                    + inner(as_vector([c_me[3], 0.0, 0.0]), cross(X_me, u_me))
                    + inner(as_vector([0.0, c_me[4], 0.0]), cross(X_me, u_me))
                )
                F5 = derivative(Wrigid, w_me, wtest_me) * dx_me
                # F5 = derivative(Wrigid, w_me, wtest_me)*ds_me(LVendoid)
                Ftotal += F5

        # Add stabilization
        if self.discretization == "P1P1":

            # lhs_u = p_me * J * inv(Fmat) * inv(Fmat.T)
            # res_u = inner(lhs_u, Fmat.T * grad(v_me)) * dx_me

            Kappa = Constant(1.0e5)
            # res_p = ((J - 1) - p_me / Kappa) * q_me * dx_me

            h_elem = CellDiameter(mesh_me)
            mu = Constant(5.0e4)

            if self.discretization_technique == 1:
                Fs = (
                    -(
                        h_elem
                        * h_elem
                        * Constant(0.5)
                        / mu
                        * J
                        * inner(inv(Fmat.T) * grad(p_me), inv(Fmat.T) * grad(q_me))
                        * dx_me
                    )
                    #  - p_me * q_me / Kappa * dx_me
                )

            # elif self.discretization_technique == 2:
            #    Fs = (
            #        (1.0 / (CellVolume(mesh_me)) ** (1.0 / 3.0))
            #        * (p_me - p_me / CellVolume(mesh_me))
            #        * (q_me - q_me / CellVolume(mesh_me))
            #        * dx_me
            #    )
            # elif self.discretization_technique == 0:
            #    p_bar = project(p_me, self.QDG)
            #    Fs = (
            #        (1.0 / (CellVolume(mesh_me)) ** (1.0 / 3.0))
            #        * (p_me - p_bar)
            #        * (q_me)
            #        * dx_me
            #    )
            else:
                pass

            if "poro" in list(self.SimDet.keys()):
                pass
            else:
                Ftotal += Fs

        Jac = derivative(F1, w_me, dw_me)
        if not self.ispctrl:
            Jac2 = derivative(F2, w_me, dw_me)
            Jac += Jac2
        else:
            Jacp = derivative(Fp, w_me, dw_me)
            Jac += Jacp

        Jac4 = derivative(F4, w_me, dw_me)
        Jac += Jac4

        if "springbc" in list(self.SimDet.keys()) and self.SimDet["springbc"]:
            Jac3 = derivative(F3, w_me, dw_me)
            Jac += Jac3
        elif self.isLV or self.isBiV:
            Jac5 = derivative(F5, w_me, dw_me)
            Jac += Jac5

        if self.discretization == "P1P1":
            if "poro" in list(self.SimDet.keys()):
                pass
            else:
                Jacs = derivative(Fs, w_me, dw_me)
                Jac += Jacs

        # Initialize LV cavity volume
        if not self.ispctrl:
            self.LVCavityvol.vol = uflforms.LVcavityvol()
            if self.isBiV:
                self.RVCavityvol.vol = uflforms.RVcavityvol()

            if self.isBiV:
                self.RVP_cav = uflforms.RVcavitypressure()
                self.RVV_cav = uflforms.RVcavityvol()
        return Ftotal, Jac, bcs_elas

    def Solver(self):
        solverparams = {
            "Jacobian": self.Jac,
            "F": self.Ftotal,
            "w": self.w_me,
            "boundary_conditions": self.bcs,
            "Type": 0,  # Default
            "mesh": self.mesh_me,
            "mode": 1,
        }

        if "abs_tol" in list(self.SimDet.keys()):
            solverparams.update({"abs_tol": self.SimDet["abs_tol"]})
        if "rel_tol" in list(self.SimDet.keys()):
            solverparams.update({"rel_tol": self.SimDet["rel_tol"]})
        if "Type" in list(self.SimDet.keys()):
            solverparams.update({"Type": self.SimDet["Type"]})
        solver_eals = NSolver(solverparams)
        return solver_eals

    def GetDisplacement(self):
        if self.isLV or self.iswaorta:
            if self.ispctrl:
                if "springbc" in list(self.SimDet.keys()) and self.SimDet["springbc"]:
                    u, p = self.w_me.split(deepcopy=True)
                    rv_pendo = []
                    lv_pendo = []
                else:
                    u, p, self.c = self.w_me.split(deepcopy=True)
                    rv_pendo = []
                    lv_pendo = []
            else:
                if "springbc" in list(self.SimDet.keys()) and self.SimDet["springbc"]:
                    u, p, lv_pendo = self.w_me.split(deepcopy=True)
                    rv_pendo = []
                else:
                    u, p, lv_pendo, self.c = self.w_me.split(deepcopy=True)
                    rv_pendo = []
        elif self.isBiV or self.isFCH:
            if self.ispctrl:
                if "springbc" in list(self.SimDet.keys()) and self.SimDet["springbc"]:
                    u, p = self.w_me.split(deepcopy=True)
                    lv_pendo = []
                    rv_pendo = []

                else:
                    u, p, self.c = self.w_me.split(deepcopy=True)
                    lv_pendo = []
                    rv_pendo = []
            else:
                if "springbc" in list(self.SimDet.keys()) and self.SimDet["springbc"]:
                    u, p, lv_pendo, rv_pendo = self.w_me.split(deepcopy=True)

                else:
                    u, p, lv_pendo, rv_pendo, self.c = self.w_me.split(deepcopy=True)

        u.rename("u_", "u_")

        return u

    def UpdateVar(self):
        self.w_me_n.assign(self.w_me)

    def Reset(self):
        self.w_me.assign(self.w_me_n)

    def GetP(self):
        if self.isLV or self.iswaorta:
            if self.ispctrl:
                if "springbc" in list(self.SimDet.keys()) and self.SimDet["springbc"]:
                    u, p = self.w_me.split(deepcopy=True)
                    lv_pendo = []
                    rv_pendo = []
                else:
                    u, p, self.c = self.w_me.split(deepcopy=True)
                    lv_pendo = []
                    rv_pendo = []
            else:
                if "springbc" in list(self.SimDet.keys()) and self.SimDet["springbc"]:
                    u, p, lv_pendo = self.w_me.split(deepcopy=True)
                    rv_pendo = []
                else:
                    u, p, lv_pendo, self.c = self.w_me.split(deepcopy=True)
                    rv_pendo = []
        elif self.isBiV or self.isFCH:
            if self.ispctrl:
                if "springbc" in list(self.SimDet.keys()) and self.SimDet["springbc"]:
                    # u, p, lv_pendo, rv_pendo = self.w_me.split(deepcopy=True)
                    u, p = self.w_me.split(deepcopy=True)
                    lv_pendo = []
                    rv_pendo = []
                else:
                    u, p, self.c = self.w_me.split(deepcopy=True)
                    lv_pendo = []
                    rv_pendo = []
            else:
                if "springbc" in list(self.SimDet.keys()) and self.SimDet["springbc"]:
                    u, p, lv_pendo, rv_pendo = self.w_me.split(deepcopy=True)
                else:
                    u, p, lv_pendo, rv_pendo, self.c = self.w_me.split(deepcopy=True)

        p.rename("p_", "p_")

        return p

    def GetTmax1(self):
        return self.Tmax1.value

    def GetTmax2(self):
        return self.Tmax2.value

    def GetTmax3(self):
        return self.Tmax3.value

    def GetFmat(self):
        return self.uflforms.Fmat()

    def GetFiberstrain(self, F_ref):
        return self.uflforms.fiberstrain(F_ref=F_ref)

    def GetFiberstrainUL(self):
        F_Identity = Identity(self.GetDisplacement().ufl_domain().geometric_dimension())
        return self.uflforms.fiberstrain(F_ref=F_Identity)

    def GetIMP(self):
        return self.uflforms.IMP()

    def GetIMP2(self):
        return self.uflforms.IMP2()

    def Getfstress(self):
        return self.uflforms.fiberstress() + self.activeforms.fiberstress()

    def GetFiberNaturalStrain(self, F_ED, basis_dir, AHA_segments):
        F_n = self.GetFmat()

        return self.activeforms.CalculateFiberNaturalStrain(
            F_=F_n, F_ref=F_ED, e_fiber=basis_dir, VolSeg=AHA_segments
        )

    def GetFiberBiotStrain(self, F_ED, basis_dir, AHA_segments):
        F_n = self.GetFmat()

        return self.activeforms.CalculateFiberBiotStrain(
            F_=F_n, F_ref=F_ED, e_fiber=basis_dir, VolSeg=AHA_segments
        )

    def GetFiberGreenStrain(self, F_ED, basis_dir, AHA_segments):
        F_n = self.GetFmat()

        return self.activeforms.CalculateFiberGreenStrain(
            F_=F_n, F_ref=F_ED, e_fiber=basis_dir, VolSeg=AHA_segments
        )

    def GetLaplace(self):
        if self.isFCH:
            # return self.uflforms.solveLaplaceEquation_fch()
            return
        elif self.iswaorta:
            return self.uflforms.solveLaplaceEquation_waorta()
        elif self.isLV or self.isBiV:
            return self.uflforms.solveLaplaceEquation_ideal()

    def GetLVP(self):
        if self.ispctrl:
            return self.LVCavitypres.pres
        else:
            return self.uflforms.LVcavitypressure()

    def GetLVV(self):
        if self.ispctrl:
            return self.LV_closedsurf()
        else:
            return self.uflforms.LVcavityvol()

    def LV_closedsurf(self):
        if self.iswaorta:
            return self.uflforms.LVcavityvol_waorta()
        elif self.isFCH:
            return self.uflforms.LVcavityvol_fch()
        elif self.isLV:
            if "springbc" in list(self.SimDet.keys()) and self.SimDet["springbc"]:
                return self.uflforms.LVcavityvol_mvb()
            else:
                return self.uflforms.LVcavityvol()
        elif self.isBiV:
            return self.uflforms.LVcavityvol()

    def GetTopSpring(self):
        return self.uflforms.topspringbc()

    def GetRVP(self):
        if self.ispctrl:
            return self.RVCavitypres.pres
        else:
            return self.uflforms.RVcavitypressure()

    def GetRVV(self):
        if self.isBiV:
            if "springbc" in list(self.SimDet.keys()) and self.SimDet["springbc"]:
                return self.uflforms.RVcavityvol()  # *
            else:
                return self.uflforms.RVcavityvol()
        elif self.isFCH:
            if "springbc" in list(self.SimDet.keys()) and self.SimDet["springbc"]:
                return self.uflforms.RVcavityvol_fch()
            else:
                return

    def GetLAP(self):
        if self.SimDet.get("fch_fe"):
            return self.LACavitypres.pres

    def GetLAV(self):
        if self.SimDet.get("fch_fe"):
            return self.uflforms.LAcavityvol_fch()

    def GetRAP(self):
        if self.SimDet.get("fch_fe"):
            return self.RACavitypres.pres

    def GetRAV(self):
        if self.SimDet.get("fch_fe"):
            return self.uflforms.RAcavityvol_fch()

    def GetSActive(self):
        i, j = ufl.indices(2)
        Sactive = self.activeforms.PK2StressTensor()
        Sactive_ = project(self.f0_me[i] * Sactive[i, j] * self.f0_me[j], self.QDG)
        Sactive_.rename("Sact", "Sact")

        return Sactive_

   # def GetSActive(self):
   #     Sactive = self.activeforms.PK2StressTensor()
   #     Sactive_ = project(self.f0_me[i] * Sactive[i, j] * self.f0_me[j], self.QDG)
   #     Sactive_.rename("Sact", "Sact")

   #     return Sactive_

    def GetCt(self):
        return self.activeforms.Get_t_a()

    def GetDeformedBasis(self, params):
        default_params = {
            "LVangle": [0, 0],
            "SPangle": [0, 0],
            "RVangle": [0, 0],
            "meshName": "EDfile",
        }
        default_params.update(params)

        LVangle = default_params["LVangle"]
        SPangle = default_params["SPangle"]
        RVangle = default_params["RVangle"]
        meshName = default_params["meshName"]

        #  - - - - - - - - - - - -- - - - - - - - - - - - - - - -- - - - - - -
        mesh_me = self.mesh_me
        facetboundaries_me = self.facetboundaries_me
        deg_me = self.deg_me
        LVendoid = self.SimDet["LVendoid"]
        RVendoid = self.SimDet["RVendoid"]
        epiid = self.SimDet["epiid"]
        isLV = self.isLV

        meshDispFunc = VectorFunctionSpace(mesh_me, "CG", 1)
        VQuadelem_me = VectorElement(
            "Quadrature", mesh_me.ufl_cell(), degree=deg_me, quad_scheme="default"
        )
        VQuadelem_me._quad_scheme = "default"
        fiberFS = FunctionSpace(mesh_me, VQuadelem_me)

        meshDisplacement = project(self.GetDisplacement(), meshDispFunc)
        deformedMesh, deformedBoundary = update_mesh(
            mesh=mesh_me, displacement=meshDisplacement, boundaries=facetboundaries_me
        )

        outputfolder = self.parameters["outputfolder"]
        folderName = self.parameters["foldername"]

        EDmeshData = {
            "epiid": epiid,
            "rvid": RVendoid,
            "lvid": LVendoid,
            "LVangle": LVangle,  # [0, 0],
            "Septangle": [0, 0],
            "RVangle": [0, 0],
            "isepiflip": False,
            "isendoflip": False,
            "iscaling": False,
            "mesh": deformedMesh,
            "facets": deformedBoundary,
            "mFileName": outputfolder + folderName + "/deformation_unloadED/",
            "isLV": isLV,
            "meshName": meshName,
        }

        eCC_ED, eLL_ED, eRR_ED = create_EDFibers(EDmeshData)

        # Copy directional field from functionspace with mesh_me to deformedmesh
        eCC = Function(fiberFS)
        eRR = Function(fiberFS)
        eLL = Function(fiberFS)
        eCC.vector()[:] = eCC_ED.vector().get_local()[:]
        eRR.vector()[:] = eRR_ED.vector().get_local()[:]
        eLL.vector()[:] = eLL_ED.vector().get_local()[:]

        return eCC, eRR, eLL, deformedMesh, deformedBoundary
