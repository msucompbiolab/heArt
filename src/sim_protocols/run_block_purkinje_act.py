from dolfin import *
import dolfin
import ufl
import math as math
import numpy as np
from ..mechanics.forms_MRC2 import Forms
from ..mechanics.activeforms_MRC2 import activeForms
from ..ep.EPmodel_basic import EPmodel
from ..utils.oops_objects_MRC2 import State_Variables
from ..utils.nsolver import NSolver as NSolver
from ..utils.oops_objects_MRC2 import printout
from fenicstools.Probe import *


def run_block_purkinje_act(IODet, SimDet):
    parameters["form_compiler"]["cpp_optimize"] = True
    parameters["form_compiler"]["representation"] = "uflacs"
    parameters["form_compiler"]["quadrature_degree"] = 4

    outputfolder = IODet["outputfolder"]
    folderName = IODet["folderName"] + IODet["caseID"] + "/"

    EPmodel_ep, state_obj_ep = createEPmodel(IODet, SimDet)
    solver_FHN_ep = EPmodel_ep.Solver()

    EPmodel_pj, state_obj_pj = createPJmodel(IODet, SimDet)
    solver_FHN_pj = EPmodel_pj.Solver()

    cnt = 0

    File_EP = File(outputfolder + folderName + "EP.pvd")
    File_PJ = File(outputfolder + folderName + "PJ.pvd")

    pj_t_nodes = np.array(SimDet["tnode"])
    tstart_arr = [-10] * len(pj_t_nodes)
    probes = Probes(pj_t_nodes.flatten(), EPmodel_pj.w_ep.function_space().sub(0))
    comms = EPmodel_pj.mesh.mpi_comm()

    while 1:

        # Activate PK
        if state_obj_pj.tstep > 10 and state_obj_pj.tstep < 20:
            EPmodel_pj.fstim_array[0].iStim = 1.0
        else:
            EPmodel_pj.fstim_array[0].iStim = 0.0

        if state_obj_pj.tstep > 400:
            break

        state_obj_pj.tstep = state_obj_pj.tstep + state_obj_pj.dt.dt
        cnt = cnt + 1

        printout("Solving PJ", EPmodel_pj.mesh.mpi_comm())
        solver_FHN_pj.solvenonlinear()
        EPmodel_pj.UpdateVar()

        printout("Solving EP", EPmodel_pj.mesh.mpi_comm())
        solver_FHN_ep.solvenonlinear()
        EPmodel_ep.UpdateVar()

        probes(EPmodel_pj.getphivar())
        probes_val = probes.array()

        # broadcast from proc 0 to other processes
        rank = MPI.rank(comms)  # .Get_rank()

        probes_val_bcast = probes_val  ## probe will only send to rank =0
        if not rank == 0:
            if cnt == 1:
                probes_val_bcast = np.empty(len(pj_t_nodes))
            else:
                probes_val_bcast = np.empty((len(pj_t_nodes), cnt))

        comms.Bcast(probes_val_bcast, root=0)
        # print(rank, probes_val_bcast, flush=True)

        try:
            probesize = np.shape(probes.array())[1]
        except IndexError:
            probesize = 0

        # current_ta.vector()[:] = t
        # update_activationTime(phi_pj, current_ta, t_init, isActive, pj_mesh.mpi_comm())

        for p in range(0, len(pj_t_nodes)):

            try:
                phi_pj_val = probes_val_bcast[p][probesize - 1]
            except IndexError:
                phi_pj_val = probes_val_bcast[p]

            # if(phi_pj(pj_t_nodes[p][0],pj_t_nodes[p][1],pj_t_nodes[p][2]) > 0.9 and tstart_arr[p] < 10.0):
            if phi_pj_val > 0.9 and tstart_arr[p] < 1.0:
                if tstart_arr[p] < 0:
                    tstart_arr[p] = 0
                    EPmodel_ep.fstim_array[p].iStim = 1.0
                    # print("Activate T node", p)
                else:
                    tstart_arr[p] += state_obj_pj.dt.dt
            else:
                EPmodel_ep.fstim_array[p].iStim = 0.0
                # print("Deactivate T node", p)

        if cnt % SimDet["writeStep"] == 0.0:
            File_EP << EPmodel_ep.getphivar()
            File_PJ << EPmodel_pj.getphivar()


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
        "d_iso": SimDet["d_iso_pj"],
        "ploc": SimDet["ploc"],
        "AHAid": AHAid_pj,
        "matid": matid_pj,
        "Ischemia": SimDet["Ischemia"],
        "pacing_timing": SimDet["pacing_timing"],
        "ploc": SimDet["ploc"],
    }

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

    # Set Fiber
    f0 = Expression(("1.0", "0.0", "0.0"), degree=1)
    s0 = Expression(("0.0", "1.0", "0.0"), degree=1)
    n0 = Expression(("0.0", "0.0", "1.0"), degree=1)

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
        "pacing_timing": SimDet["pacing_timing"],
    }

    # Define EP model and solver
    EPmodel_ = EPmodel(EPparams)

    return EPmodel_, state_obj


#
#    EPmodel_ = EPmodel(EPparams)
#    EpiBCid_ep = EPmodel_.MarkStimulus()
#    solver_FHN = EPmodel_.Solver()
#    File(outputfolder + folderName + "EP_stimulus.pvd") << EpiBCid_ep
#
#    MEparams = {
#        "mesh": meshME.mesh,
#        "facetboundaries": meshME.subdomains,
#        "facetnormal": meshME.N,
#        "f0": f0,
#        "s0": s0,
#        "n0": n0,
#    }
#
#    MEmodel_ = mechanics(IODet, SimDet_, MEparams)
#    solver_ME = MEmodel_.Solver()
#
#    pload_arr = [0.0]
#
#    F_EP = File(outputfolder + folderName + "EP.pvd")
#    F_Act = File(outputfolder + folderName + "Act.pvd")
#    F_ActF = File(outputfolder + folderName + "ActF.pvd")
#    F_Disp = File(outputfolder + folderName + "Disp.pvd")
#    ntpt = SimDet["ntpt"]
#    dt = SimDet["dt"]
#
#    cnt = 0
#    potential_me = Function(FunctionSpace(MEmodel_.mesh, "CG", 1))
#    for tpt in np.arange(0, ntpt):
#        state_obj.tstep = state_obj.tstep + state_obj.dt.dt
#        state_obj.cycle = math.floor(state_obj.tstep / state_obj.BCL)
#        state_obj.t = state_obj.tstep - state_obj.cycle * state_obj.BCL
#
#        MEmodel_.t_a.vector()[:] = state_obj.t
#        print(
#            (
#                "Active contraction =",
#                MEmodel_.t_a.vector().get_local()[0],
#                " State obj t = ",
#                state_obj.t,
#            )
#        )
#
#        # Reset phi and r in EP at end of diastole
#        if state_obj.t < state_obj.dt.dt:
#            EPmodel_.reset()
#
#        # Solve EP
#        print("Solve EP")
#        solver_FHN.solvenonlinear()
#        EPmodel_.UpdateVar()
#
#        # Solve Mechanics
#        print("Solve ME")
#        solver_ME.solvenonlinear()
#        MEmodel_.UpdateVar()
#
#        potential_ref = EPmodel_.interpolate_potential_ep2me_phi(
#            V_me=Function(FunctionSpace(MEmodel_.mesh, "CG", 1))
#        )
#        potential_ref.rename("v_ref", "v_ref")
#        potential_me.vector()[:] = potential_ref.vector().get_local()[:]
#
#        MEmodel_.activeforms.update_activationTime(
#            potential_n=potential_me, comm=meshME.mesh.mpi_comm()
#        )
#
#        if cnt % SimDet["writeStep"] == 0.0:
#            F_EP << EPmodel_.getphivar()
#            F_Act << potential_me
#            F_Disp << MEmodel_.GetDisplacement()
#            F_ActF << MEmodel_.GetSActive()
#
#        cnt += 1
#
#    return
#
#
# class createmesh(object):
#    def __init__(self, IODet, SimDet, nelem):
#        self.IODet = IODet
#        self.SimDet = SimDet
#
#        length = SimDet["length"]
#        width = SimDet["width"]
#        outputfolder = IODet["outputfolder"]
#        folderName = IODet["folderName"] + IODet["caseID"] + "/"
#
#        # Create mesh
#        mesh = BoxMesh(
#            Point(0, 0, 0),
#            Point(length, width, width),
#            nelem * int(length / width),
#            nelem,
#            nelem,
#        )
#        print(("Number of Elements in mesh = " + str(mesh.num_cells())))
#        self.mesh = mesh
#
#        self.N = FacetNormal(mesh)
#
#        # Mark Facet
#        class Right(SubDomain):
#            def inside(self, x, on_boundary):
#                return x[0] > length * (1.0 - DOLFIN_EPS) and on_boundary
#
#        class Left(SubDomain):
#            def inside(self, x, on_boundary):
#                return x[0] < DOLFIN_EPS and on_boundary
#
#        class Front(SubDomain):
#            def inside(self, x, on_boundary):
#                return x[1] > width * (1.0 - DOLFIN_EPS) and on_boundary
#
#        class Back(SubDomain):
#            def inside(self, x, on_boundary):
#                return x[1] < DOLFIN_EPS and on_boundary
#
#        class Top(SubDomain):
#            def inside(self, x, on_boundary):
#                return x[2] > width * (1.0 - DOLFIN_EPS) and on_boundary
#
#        class Bot(SubDomain):
#            def inside(self, x, on_boundary):
#                return x[2] < DOLFIN_EPS and on_boundary
#
#        # Create mesh functions over the cell facets
#        sub_domains = MeshFunction(
#            "size_t", mesh, mesh.topology().dim() - 1, mesh.domains()
#        )
#
#        # Mark all facets as sub domain 1
#        sub_domains.set_all(0)
#
#        # Mark left facets
#        left = Left()
#        left.mark(sub_domains, 1)
#
#        right = Right()
#        right.mark(sub_domains, 2)
#
#        front = Front()
#        front.mark(sub_domains, 3)
#
#        back = Back()
#        back.mark(sub_domains, 4)
#
#        top = Top()
#        top.mark(sub_domains, 5)
#
#        bot = Bot()
#        bot.mark(sub_domains, 6)
#
#        self.subdomains = sub_domains
#        self.comm = mesh.mpi_comm()
#
#
# class mechanics(object):
#    def __init__(self, IODet, SimDet, MEparams):
#        passive_mat_input = {}
#        active_mat_input = {}
#        if "Passive model" in list(SimDet.keys()):
#            matmodel = SimDet["Passive model"]
#            passive_mat_input.update({"material model": matmodel})
#        if "Passive params" in list(SimDet.keys()):
#            matparams = SimDet["Passive params"]
#            passive_mat_input.update({"material params": matparams})
#        if "Active model" in list(SimDet.keys()):
#            matmodel = SimDet["Active model"]
#            active_mat_input.update({"material model": matmodel})
#        if "Active params" in list(SimDet.keys()):
#            matparams = SimDet["Active params"]
#            active_mat_input.update({"material params": matparams})
#
#        mesh = MEparams["mesh"]
#        sub_domains = MEparams["facetboundaries"]
#        f0 = MEparams["f0"]
#        s0 = MEparams["s0"]
#        n0 = MEparams["n0"]
#        N = MEparams["facetnormal"]
#        # t_a = MEparams["t_a"]
#
#        Quadelem = FiniteElement(
#            "Quadrature", mesh.ufl_cell(), degree=4, quad_scheme="default"
#        )
#        Quadelem._quad_scheme = "default"
#        t_a = Function(FunctionSpace(mesh, Quadelem))
#
#        # Define Integration domain
#        dx = dolfin.dx(mesh, metadata={"integration_order": 2})
#        ds = dolfin.ds(mesh, subdomain_data=sub_domains)
#
#        # Define function space
#        Velem = VectorElement("CG", mesh.ufl_cell(), 2, quad_scheme="default")
#        Qelem = FiniteElement("CG", mesh.ufl_cell(), 1, quad_scheme="default")
#        DGelem = FiniteElement("DG", mesh.ufl_cell(), 1, quad_scheme="default")
#
#        W = FunctionSpace(mesh, MixedElement([Velem, Qelem]))
#        V = FunctionSpace(mesh, Velem)
#        QDG = FunctionSpace(mesh, DGelem)
#
#        # Define function
#        w = Function(W)
#        w_n = Function(W)
#        wtest = TestFunction(W)
#        dw = TrialFunction(W)
#        du, dp = TrialFunctions(W)
#        (u, p) = split(w)
#        (v, q) = split(wtest)
#
#        params = {
#            "mesh": mesh,
#            "facetboundaries": sub_domains,
#            "facet_normal": N,
#            "mixedfunctionspace": W,
#            "mixedfunction": w,
#            "displacement_variable": u,
#            "pressure_variable": p,
#            "fiber": f0,
#            "sheet": s0,
#            "sheet-normal": n0,
#            "growth_tensor": None,
#            "incompressible": True,
#        }
#
#        params.update(passive_mat_input)
#        uflforms = Forms(params)
#
#        SEF = uflforms.PassiveMatSEF()
#        PK1pas = uflforms.PK1()
#        Fe = uflforms.Fmat()
#        n = uflforms.J() * inv(Fe.T) * N
#
#        # _a = Expression(("ta"), ta = 0.0, degree=0)
#        activeparams = {
#            "mesh": mesh,
#            "dx": dx,
#            "deg": 4,
#            "facetboundaries": sub_domains,
#            "facet_normal": N,
#            "displacement_variable": u,
#            "pressure_variable": p,
#            "fiber": f0,
#            "sheet": s0,
#            "sheet-normal": n0,
#            "t_a": t_a,
#            "mesh": mesh,
#            "Threshold_Potential": 0.9,
#            "growth_tensor": None,
#            "HomogenousActivation": SimDet["HomogenousActivation"],
#        }
#        activeparams.update(active_mat_input)
#
#        activeforms = activeForms(activeparams)
#        potential = Function(FunctionSpace(mesh, "CG", 1))
#        activeforms.update_activationTime(potential_n=potential, comm=mesh.mpi_comm())
#
#        # Define Passive SEF
#        SEF = uflforms.PassiveMatSEF()
#        PK1pas = uflforms.PK1()
#        Fe = uflforms.Fmat()
#        n = uflforms.J() * inv(Fe.T) * N
#
#        # Define Active Stress
#        Sactive = activeforms.PK2StressTensor()
#        PK1act = Fe * Sactive
#
#        # Boundary conditions
#        uright = Expression("val", val=0, degree=0)
#        bcleft = DirichletBC(W.sub(0).sub(0), Constant((0.0)), sub_domains, 1)
#        bcright = DirichletBC(W.sub(0).sub(0), uright, sub_domains, 2)
#        bcback = DirichletBC(W.sub(0).sub(1), Constant((0.0)), sub_domains, 4)
#        bcbot = DirichletBC(W.sub(0).sub(2), Constant((0.0)), sub_domains, 6)
#        bcs = [bcleft, bcback, bcbot]
#        pload = Expression("val", val=0.0, degree=0)
#
#        # Define Weak Form
#        Fact = inner(PK1act, grad(v)) * dx
#        Fpas = derivative(SEF, w, wtest) * dx
#        Fload = -inner(pload * n, v) * ds(2)
#        Ftotal = Fpas + Fact + Fload
#        Jac = derivative(Ftotal, w, dw)
#
#        # Solve variational problem
#        # Optimization options for the form compiler
#        ffc_options = {"optimize": True}
#        solver_options = {
#            "newton_solver": {
#                "maximum_iterations": 100,
#                "absolute_tolerance": 1e-8,
#                "relative_tolerance": 1e-7,
#            }
#        }
#
#        self.w = w
#        self.w_n = w_n
#        self.Ftotal = Ftotal
#        self.Jac = Jac
#        self.bcs = bcs
#        self.mesh = mesh
#        self.t_a = t_a
#        self.activeforms = activeforms
#        self.f0 = f0
#        self.QDG = QDG
#
#    def Solver(self):
#        solverparams_ME = {
#            "Jacobian": self.Jac,
#            "F": self.Ftotal,
#            "w": self.w,
#            "boundary_conditions": self.bcs,
#            "Type": 0,
#            "mesh": self.mesh,
#            "mode": 1,
#        }
#        solver_ME = NSolver(solverparams_ME)
#
#        return solver_ME
#
#    def UpdateVar(self):
#        self.w_n.assign(self.w)
#
#    def GetDisplacement(self):
#        u, p = self.w.split(deepcopy=True)
#        u.rename("u_", "u_")
#
#        return u
#
#    def GetSActive(self):
#        Sactive = self.activeforms.PK2StressTensor()
#        i, j = ufl.indices(2)
#        Sactive_ = project(self.f0[i] * Sactive[i, j] * self.f0[j], self.QDG)
#        Sactive_.rename("Sact", "Sact")
#
#        return Sactive_
