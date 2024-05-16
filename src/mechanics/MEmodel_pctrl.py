from dolfin import *
import numpy as np
import os as os
from ..utils.nsolver import NSolver as NSolver
from ..utils.oops_objects_MRC2 import biventricle_mesh as biv_mechanics_mesh
from ..utils.oops_objects_MRC2 import lv_mesh as lv_mechanics_mesh

# from ..utils.oops_objects_MRC2 import PV_Elas
from ..utils.oops_objects_MRC2 import update_mesh
from ..utils.oops_objects_MRC2 import printout
from ..utils.edgetypebc import *
from .forms_MRC2 import Forms
from .activeforms_MRC2 import activeForms
from ..utils.mesh_partitionMeshforEP_J import defCPP_Matprop, defCPP_Matprop_DIsch

# from ..utils.mesh_scale_create_fiberFiles import create_EDFibers


class MEmodel(object):
    def __init__(self, params, SimDet):
        self.parameters = self.default_parameters()
        self.parameters.update(params)
        self.SimDet = SimDet
        self.isLV = SimDet["isLV"]
        self.deg_me = SimDet["GiccioneParams"]["deg"]

        if self.isLV:
            self.Mesh = lv_mechanics_mesh(self.parameters, SimDet)
        else:
            self.Mesh = biv_mechanics_mesh(self.parameters, SimDet)

        f0_me_Gauss = self.Mesh.f0
        s0_me_Gauss = self.Mesh.s0
        n0_me_Gauss = self.Mesh.n0

        self.mesh_me = self.Mesh.mesh
        self.facetboundaries_me = self.Mesh.facetboundaries
        self.edgeboundaries_me = self.Mesh.edgeboundaries

        self.ds_me = self.Mesh.ds
        self.dx_me = self.Mesh.dx

        LVendoid = self.SimDet["LVendoid"]
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

        self.LVCavityvol = Expression(("vol"), vol=0.0, degree=2)
        self.RVCavityvol = Expression(("vol"), vol=0.0, degree=2)

        self.LVCavitypres = Expression(("pres"), pres=0.0, degree=2)  # amend
        self.isincomp = SimDet["GiccioneParams"]["incompressible"]

        #  - - - - - - - - - - - -- - - - - - - - - - - - - - - -- - - - - - -

        Velem = VectorElement("CG", self.mesh_me.ufl_cell(), 1, quad_scheme="default")
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
            if self.isLV:
                if "springbc" in list(self.SimDet.keys()) and self.SimDet["springbc"]:
                    self.W = FunctionSpace(self.mesh_me, MixedElement([Velem, Qelem]))
                else:
                    self.W = FunctionSpace(
                        self.mesh_me, MixedElement([Velem, Qelem, VRelem])
                    )
            else:
                if "springbc" in list(self.SimDet.keys()) and self.SimDet["springbc"]:
                    self.W = FunctionSpace(
                        self.mesh_me, MixedElement([Velem, Qelem, Relem, Relem])
                    )
                else:
                    self.W = FunctionSpace(
                        self.mesh_me, MixedElement([Velem, Qelem, Relem, Relem, VRelem])
                    )
        else:
            if self.isLV:
                if "springbc" in list(self.SimDet.keys()) and self.SimDet["springbc"]:
                    self.W = FunctionSpace(self.mesh_me, MixedElement([Velem]))
                else:
                    self.W = FunctionSpace(self.mesh_me, MixedElement([Velem, VRelem]))
            else:
                if "springbc" in list(self.SimDet.keys()) and self.SimDet["springbc"]:
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

        ############
        self.Tmax1 = Expression("value", value=0, degree=2)
        self.Tmax2 = Expression("value", value=0, degree=2)
        self.Tmax3 = Expression("value", value=0, degree=2)
        ############

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
            "volinc": 1,
            "LVangle": [60, -60],
        }
        default_params.update(params)

        EDP = default_params["EDP"]
        LVangle = default_params["LVangle"]
        maxit = default_params["maxit"]
        restol = default_params["restol"]
        drestol = default_params["drestol"]
        volinc = default_params["volinc"]
        LVendoid = self.SimDet["LVendoid"]
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

                if LVP > EDP:
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

        # np.savez(outfolder+"Udata.npz", f0_me_Gauss=f0_me_Gauss.vector().array()[:], \
        #                               s0_me_Gauss=s0_me_Gauss.vector().array()[:], \
        #                               n0_me_Gauss=n0_me_Gauss.vector().array()[:], \
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
        topid = self.SimDet["topid"]
        W = self.W

        bctop = DirichletBC(
            W.sub(0),
            Expression(("0.0", "0.0", "0.0"), degree=0),
            facetboundaries,
            topid,
        )

        # bc_5 = DirichletBC(
        #    W.sub(0),
        #    Expression(("0.0", "0.0", "0.0"), degree=0),
        #    facetboundaries,
        #    5,
        # )
        # bc_6 = DirichletBC(
        #    W.sub(0), Expression(("0.0", "0.0", "0.0"), degree=0), facetboundaries, 6
        # )
        # bc_10 = DirichletBC(
        #    W.sub(0), Expression(("0.0", "0.0", "0.0"), degree=0), facetboundaries, 10
        # )

        # endoring = pick_endoring_bc(method="cpp")(edgeboundaries, 1)
        # bcedge = DirichletBC(
        #    W.sub(0),
        #    Expression(("0.0", "0.0", "0.0"), degree=0),
        #    endoring,
        #    method="pointwise",
        # )

        if "springbc" in list(self.SimDet.keys()) and self.SimDet["springbc"]:
            bcs = []
        else:
            bcs = [bctop]

        return bcs
        #  - - - - - - - - - - - -- - - - - - - - - - - - - - - -- - - - - - -

    #    def Rogazonni(self):
    #        Kepi_n = 2e5; Kepi_t = 2e4
    #
    #        ## epicardial
    #        PN_epi = Kepi_n * inner(outer(N_me, N_me) * u_me, v_me) * ds_me(epiid) + \
    #                Kepi_t * ((Identity(u.ufl_shape[0]) - outer(N_me, N_me)) * u_me, v_me) * ds_me(epiid) #Q u_me.ufl_shape?
    #
    #        ## base
    #        v_base = J * inv(Fmat.T) * N_me * ds_me(LVendoid) / (norm(J * inv(Fmat.T) * N_me) * ds_me(topid)) #Q: || in integral? / norm of tensor
    #        PN_base = P_LV * norm(J * inv(Fmat.T) * N_me) * v_base

    def Problem(self):
        GuccioneParams = self.SimDet["GiccioneParams"]
        topid = self.SimDet["topid"]
        LVendoid = self.SimDet["LVendoid"]
        RVendoid = self.SimDet["RVendoid"]
        epiid = self.SimDet["epiid"]
        isincomp = GuccioneParams["incompressible"]
        deg_me = GuccioneParams["deg"]

        mesh_me = self.mesh_me
        facetboundaries_me = self.facetboundaries_me
        f0_me = self.f0_me
        s0_me = self.s0_me
        n0_me = self.n0_me

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
            if self.isLV:
                if "springbc" in list(self.SimDet.keys()) and self.SimDet["springbc"]:
                    du, dp = TrialFunctions(W_me)
                    (u_me, p_me) = split(w_me)
                    (u_me_n, p_me_n) = split(w_me_n)
                    (v_me, q_me) = TestFunctions(W_me)
                    rv_pendo = []
                    LVendo_comp = 2
                    RVendo_comp = 1000

                else:
                    du, dp, dc = TrialFunctions(W_me)
                    (u_me, p_me, c_me) = split(w_me)
                    (u_me_n, p_me_n, c_me_n) = split(w_me_n)
                    (v_me, q_me, cq) = TestFunctions(W_me)
                    rv_pendo = []
                    LVendo_comp = 2
                    RVendo_comp = 1000

            else:
                if "springbc" in list(self.SimDet.keys()) and self.SimDet["springbc"]:
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
            if self.isLV:
                if "springbc" in list(self.SimDet.keys()) and self.SimDet["springbc"]:
                    du = TrialFunctions(W_me)
                    (u_me) = split(w_me)
                    (v_me) = TestFunctions(W_me)
                    p_me = Function(Q_me)
                    rv_pendo = []
                    LVendo_comp = 1
                    RVendo_comp = 1000
                else:
                    du, dc = TrialFunctions(W_me)
                    (u_me, c_me) = split(w_me)
                    (v_me, cq) = TestFunctions(W_me)
                    p_me = Function(Q_me)
                    rv_pendo = []
                    LVendo_comp = 1
                    RVendo_comp = 1000
            else:
                if "springbc" in list(self.SimDet.keys()) and self.SimDet["springbc"]:
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
            "lv_volconst_variable": rv_pendo,
            "lv_constrained_vol": self.LVCavityvol,
            # "rv_volconst_variable": lv_pendo,
            "rv_constrained_vol": self.RVCavityvol,
            "LVendoid": LVendoid,
            "RVendoid": RVendoid,
            "epiid": epiid,
            "topid": topid,  # new
            "LVendo_comp": LVendo_comp,
            "RVendo_comp": RVendo_comp,
            "fiber": f0_me,
            "sheet": s0_me,
            "sheet-normal": n0_me,
            "growth_tensor": None,
            "material model": GuccioneParams["Passive model"],
            "material params": GuccioneParams["Passive params"],
            "incompressible": GuccioneParams["incompressible"],
            "LVendo_area": LVendo_area_me,
            "lv_constrained_pres": self.LVCavitypres,
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

        if self.SimDet["Ischemia"] == True:
            kParam = {}
            # mId_List = SimDet["mId_List"]
            kParam["kabNormal"] = self.SimDet["kabNormal"]
            kParam["kNormal"] = self.SimDet["kNormal"]
            kappA = defCPP_Matprop(mesh=mesh, mId=bivMesh.AHAid, k=kParam)
            activeparams["Tmax"] = kappA

        activeforms = activeForms(activeparams)
        self.activeforms = activeforms

        F_ED = Function(TF_me)

        Fmat = uflforms.Fmat()
        Cmat = Fmat.T * Fmat
        Emat = uflforms.Emat()
        J = uflforms.J()

        n_me = J * inv(Fmat.T) * N_me

        Wp_me = uflforms.PassiveMatSEF()
        # LV_Wvol = uflforms.LVV0constrainedE()
        # if(not self.isLV):
        #    RV_Wvol = uflforms.RVV0constrainedE()

        Sactive = activeforms.PK2StressTensor()
        # printout("Total active force = " + str(assemble(activeforms.PK1Stress()*dx_me)), comm_me)

        X_me = SpatialCoordinate(mesh_me)

        Kspring = Constant(50)  # Original
        # Kspring = Constant(50)
        # Kspring  = Constant(100)

        ## ventricular volume computation

        F1 = derivative(Wp_me, w_me, wtest_me) * dx_me
        #        if(self.isLV):
        #            F2 = derivative(LV_Wvol, w_me, wtest_me)
        #        else:
        #            F2 = derivative(LV_Wvol + RV_Wvol, w_me, wtest_me)

        if "active_region" in list(self.SimDet.keys()) and self.SimDet["active_region"]:
            print("Active region = ", self.SimDet["active_region"])
            region_cnt = 0
            for regionid in self.SimDet["active_region"]:
                if region_cnt == 0:
                    F4 = inner(Fmat * Sactive, grad(v_me)) * (dx_me(int(regionid)))
                    print("Assigning active stress to ", regionid)
                else:
                    F4 += inner(Fmat * Sactive, grad(v_me)) * (dx_me(int(regionid)))
                    print("Assigning active stress to ", regionid)

                region_cnt += 1

        else:
            F4 = inner(Fmat * Sactive, grad(v_me)) * dx_me

        comm_me = self.mesh_me.mpi_comm()
        Fp_me = uflforms.LVcavitypres()
        Fp = derivative(Fp_me, w_me, wtest_me)

        # Fp = self.LVCavitypres * inner(n_me, v_me) * ds_me(LVendoid)
        # Fp = derivative(self.LVCavitypres * inner(n_me, u_me) * ds_me(LVendoid), w_me, wtest_me)

        Ftotal = F1 + Fp + F4

        if "springbc" in list(self.SimDet.keys()) and self.SimDet["springbc"]:
            # F3 = -Kspring * inner(u_me, v_me) * ds_me(epiid)

            ## epicardial
            Kepi_n = 5e4  # was 2e5 -- 1
            Cepi_n = 5e3  # was 2e4
            Kepi_t = 5e3  # was 2e4
            Cepi_t = 5e2  # was 2e3

            # F3_epi = Kepi_n * inner(outer(N_me, N_me) * u_me, v_me) * ds_me(
            #    epiid
            # ) + Kepi_t * inner(
            #    (Identity(u_me.ufl_shape[0]) - outer(N_me, N_me)) * u_me, v_me
            # ) * ds_me(
            #    epiid
            # )

            F3_epi = inner(
                outer(N_me, N_me) * (Kepi_n * u_me + Cepi_n * (u_me - u_me_n)), v_me
            ) * ds_me(epiid) + inner(
                (Identity(u_me.ufl_shape[0]) - outer(N_me, N_me))
                * (Kepi_t * u_me + Cepi_t * (u_me - u_me_n)),
                v_me,
            ) * ds_me(
                epiid
            )

            a_, b_ = self.GetSpringBC()
            F3_base = a_ * inner(b_, v_me) * ds_me(topid)

            F3 = F3_epi - F3_base

            Ftotal += F3

        else:
            Wrigid = (
                inner(as_vector([c_me[0], c_me[1], 0.0]), u_me)
                + inner(as_vector([0.0, 0.0, c_me[2]]), cross(X_me, u_me))
                + inner(as_vector([c_me[3], 0.0, 0.0]), cross(X_me, u_me))
                + inner(as_vector([0.0, c_me[4], 0.0]), cross(X_me, u_me))
            )

            F5 = derivative(Wrigid, w_me, wtest_me) * dx_me
            # F5 = derivative(Wrigid, w_me, wtest_me)*ds_me(LVendoid)
            Ftotal += F5

        lhs_u = p_me * J * inv(Fmat) * inv(Fmat.T)
        res_u = inner(lhs_u, Fmat.T * grad(v_me)) * dx_me

        Kappa = Constant(1.0e5)
        res_p = ((J - 1) - p_me / Kappa) * q_me * dx_me

        h_elem = CellDiameter(mesh_me)
        mu = Constant(1.0e4)

        stab = (
            h_elem
            * h_elem
            * Constant(0.5)
            / mu
            * J
            * inner(inv(Fmat.T) * grad(p_me), inv(Fmat.T) * grad(q_me))
            * dx_me
        )

        Fs = -stab
        Ftotal += Fs

        # cell_volume = CellVolume(mesh_me)
        # Fs = (
        #     1.0
        #     / (CellVolume(mesh_me)) ** (1.0 / 3.0)
        #     * (p_me - p_me / CellVolume(mesh_me))
        #     * (q_me - q_me / CellVolume(mesh_me))
        #     * dx_me
        # )
        # Ftotal += Fs

        Jac1 = derivative(F1, w_me, dw_me)
        # Jac2 = derivative(F2, w_me, dw_me)
        Jac4 = derivative(F4, w_me, dw_me)
        # Jac6 = derivative(F6, w_me, dw_me)
        Jacp = derivative(Fp, w_me, dw_me)

        Jac = Jac1 + Jacp + Jac4

        if "springbc" in list(self.SimDet.keys()) and self.SimDet["springbc"]:
            Jac3 = derivative(F3, w_me, dw_me)
            Jac += Jac3
        else:
            Jac5 = derivative(F5, w_me, dw_me)
            Jac += Jac5

        Jacs = derivative(Fs, w_me, dw_me)
        Jac += Jacs

        # Initialize LV cavity volume
        # self.LVCavityvol.vol = uflforms.LVcavityvol()
        # if(not self.isLV):
        #       self.RVCavityvol.vol = uflforms.RVcavityvol()

        # if(not self.isLV):
        #       self.RVP_cav = uflforms.RVcavitypressure()
        #       self.RVV_cav = uflforms.RVcavityvol()

        return Ftotal, Jac, bcs_elas

    def Solver(self):
        solverparams = {
            "Jacobian": self.Jac,
            "F": self.Ftotal,
            "w": self.w_me,
            "boundary_conditions": self.bcs,
            "Type": 0,
            "mesh": self.mesh_me,
            "mode": 1,
        }

        if "abs_tol" in list(self.SimDet.keys()):
            solverparams.update({"abs_tol": self.SimDet["abs_tol"]})
        if "rel_tol" in list(self.SimDet.keys()):
            solverparams.update({"rel_tol": self.SimDet["rel_tol"]})

        solver_eals = NSolver(solverparams)

        return solver_eals

    def GetDisplacement(self):
        if self.isLV:
            if "springbc" in list(self.SimDet.keys()) and self.SimDet["springbc"]:
                u, p = self.w_me.split(deepcopy=True)
                rv_pendo = []
            else:
                u, p, self.c = self.w_me.split(deepcopy=True)
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
        if self.isLV:
            if "springbc" in list(self.SimDet.keys()) and self.SimDet["springbc"]:
                u, p = self.w_me.split(deepcopy=True)
                rv_pendo = []
            else:
                u, p, self.c = self.w_me.split(deepcopy=True)
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

    def GetLVP(self):
        return self.uflforms.LVcavitypressure()

    def GetLVV(self):
        return self.uflforms.LVcavityvol()

    def GetVolumeComputation(self):
        return self.uflforms.volume_computation()

    def GetSpringBC(self):
        return self.uflforms.springbc()

    def GetRVP(self):
        return self.uflforms.RVcavitypressure()

    def GetRVV(self):
        return self.uflforms.RVcavityvol()

    def GetSActive(self):
        Sactive = self.activeforms.PK2StressTensor()
        Sactive_ = project(self.f0_me[i] * Sactive[i, j] * self.f0_me[j], self.QDG)
        Sactive_.rename("Sact", "Sact")

        return Sactive_

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
        # VQuadelem_me = VectorElement("DG", mesh_me.ufl_cell(), degree=0)
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

        #        eCC_ED, eLL_ED, eRR_ED = create_EDFibers(EDmeshData)

        # Copy directional field from functionspace with mesh_me to deformedmesh
        eCC = Function(fiberFS)
        eRR = Function(fiberFS)
        eLL = Function(fiberFS)
        #        eCC.vector()[:] = eCC_ED.vector().array()[:]
        #        eRR.vector()[:] = eRR_ED.vector().array()[:]
        #        eLL.vector()[:] = eLL_ED.vector().array()[:]

        return eCC, eRR, eLL, deformedMesh, deformedBoundary


#    def RobinC(self, c_n):
#        return 0
