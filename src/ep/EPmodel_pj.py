# Electrophysiology model using Aliev Panilfov model
from dolfin import *
import dolfin
import numpy as np
from ..utils.nsolver import NSolver as NSolver
from fenicstools import *
import sys, shutil, pdb, math

# set parameters for FHN
alpha = Constant(0.01)
g = Constant(0.002)
b = Constant(0.15)
c = Constant(8)
mu1 = Constant(0.2)
mu2 = Constant(0.3)
Dmat_pj = Constant(10.0)


class EPmodel(object):
    def __init__(self, params):
        self.parameters = self.default_parameters()
        self.parameters.update(params)
        self.mesh_ep = self.parameters["EPmesh"]
        self.mesh_pj = self.parameters["PJmesh"]
        self.boundaries = self.parameters["boundaries"]

        # Define FHN problem for purkinjee fiber
        self.P1_pj = FiniteElement(
            "CG", self.mesh_pj.ufl_cell(), 1, quad_scheme="default"
        )
        self.P1_pj._quad_scheme = "default"
        self.P2_pj = FiniteElement(
            "DG", self.mesh_pj.ufl_cell(), 0, quad_scheme="default"
        )
        self.P2_pj._quad_scheme = "default"
        self.P1_ep = FiniteElement(
            "CG", self.mesh_ep.ufl_cell(), 1, quad_scheme="default"
        )
        self.P1_ep._quad_scheme = "default"
        self.P2_ep = FiniteElement(
            "DG", self.mesh_ep.ufl_cell(), 0, quad_scheme="default"
        )
        self.P2_ep._quad_scheme = "default"

        self.W_pj = FunctionSpace(self.mesh_pj, MixedElement([self.P1_pj, self.P2_pj]))
        self.w_pj = Function(self.W_pj)
        self.dw_pj = TrialFunction(self.W_pj)
        self.wtest_pj = TestFunction(self.W_pj)
        self.w_n_pj = Function(self.W_pj)
        self.W_ep = FunctionSpace(self.mesh_ep, MixedElement([self.P1_ep, self.P2_ep]))
        self.w_ep = Function(self.W_ep)
        self.dw_ep = TrialFunction(self.W_ep)
        self.wtest_ep = TestFunction(self.W_ep)
        self.w_n_ep = Function(self.W_ep)

        self.F_FHN_pj, self.J_FHN_pj, self.fstim_array_pj = self.Problem_pj()
        self.F_FHN, self.J_FHN, self.fstim_array = self.Problem_ep()

    def default_parameters(self):
        return {
            "ploc": [[0.0, 0.0, 0.0, 100.0, 1]],
            "pacing_timing": [[1, 20.0]],
        }

    def calculateDmat(self, f0, mesh, mId):
        d_iso = self.parameters["d_iso"]  # 0.01 #0.02
        d_ani = d_iso * self.parameters["d_ani_factor"]  # 0.08 #0.1 #0.2
        d_Ani = d_ani
        dParam = {}
        if self.parameters["Ischemia"]:
            dParam["kNormal"] = d_ani
            dParam["kIschemia"] = d_iso
            d_Ani = defCPP_Matprop_DIsch(mesh=mesh, mId=mId, k=dParam)
        D_iso = Constant(
            ((d_iso, "0.0", "0.0"), ("0.0", d_iso, "0.0"), ("0.0", "0.0", d_iso))
        )
        Dij = d_Ani * f0[i] * f0[j] + D_iso[i, j]
        D_tensor = as_tensor(Dij, (i, j))
        return D_tensor

    def MarkStimulus(self):
        term_nodes_coord = self.parameters["term_nodes_coord"]
        ploc_tol = self.parameters["ploc_tol"]
        act_cell_ep = np.array(self.mesh_ep.cells(), dtype=np.int32)
        act_node_ep = self.mesh_ep.coordinates()
        act_cell_ep_isActive = np.zeros(self.mesh_ep.num_cells())
        tree = self.mesh_ep.bounding_box_tree()
        ttcnt = len(term_nodes_coord) - 1
        cnt = 0

        # Define a condition function for volume pacing instead of facet
        def condition_fct(ploc_coord, vertex, ploc_tol):
            dist = distance_fct(ploc_coord, vertex)
            return dist < ploc_tol

        def distance_fct(term_nodes_coord, vertex):
            # Calculate the Euclidean distance
            diff = term_nodes_coord - vertex
            dist = np.sqrt(np.sum(diff**2))
            return dist

        # Map terminal nodes in the PJ mesh to the LV endocardium element
        for term_nodes_coord_ in term_nodes_coord:
            for act_node_ep_ in act_node_ep:
                # pdb.set_trace()  # Execution will pause here
                if condition_fct(term_nodes_coord_, act_node_ep_, ploc_tol) and cnt > 0:
                    # pdb.set_trace()  # Execution will pause here
                    cell_index = tree.compute_collisions(Point(act_node_ep_))
                    for cell_act_ in cell_index:
                        # if act_cell_ep_isActive[cell_act_] == 0:
                        act_cell_ep_isActive[
                            cell_act_
                        ] = cnt  # mask for active and non active elements in domain
            cnt = cnt + 1
            if cnt >= ttcnt:
                break

        EpiBCid_ep = CellFunction("size_t", self.mesh_ep)
        EpiBCid_ep.set_all(0)
        EpiBCid_ep.array()[:] = act_cell_ep_isActive.astype(int)
        File("facetfn.pvd") << EpiBCid_ep
        pdb.set_trace()  # Execution will pause here
        return EpiBCid_ep

    def Problem_pj(self):
        state_obj = self.parameters["state_obj"]
        f0_ep = self.parameters["f0"]
        AHAid_ep = self.parameters["AHAid"]
        matid_ep = self.parameters["matid"]
        facetboundaries_ep = self.parameters["facetboundaries"]
        mesh_pj = self.mesh_pj
        boundaries = self.boundaries

        W_pj = self.W_pj
        w_n_pj = self.w_n_pj
        w_pj = self.w_pj
        dw_pj = self.dw_pj
        wtest_pj = self.wtest_pj

        phi0_pj = interpolate(Expression("0.0", degree=0), W_pj.sub(0).collapse())
        r0_pj = interpolate(Expression("0.0", degree=0), W_pj.sub(1).collapse())

        assign(w_n_pj, [phi0_pj, r0_pj])

        phi_n_pj, r_n_pj = split(w_n_pj)
        phi_pj, r_pj = split(w_pj)
        pj_d2v = vertex_to_dof_map(FunctionSpace(mesh_pj, self.P1_pj))
        bcs_pj = []
        phi_test_pj, r_test_pj = split(wtest_pj)

        #####################################################################
        #################### Function Definition ############################
        #####################################################################
        def eps_FHN(phi, r):
            return g + (mu1 * r) / (mu2 + phi)

        def f_phi(phi, r):
            return -c * phi * (phi - alpha) * (phi - 1) - r * phi

        def f_r(phi, r):
            return eps_FHN(phi, r) * (-r - c * phi * (phi - b - 1.0))

        # set parameters for transport
        fhn_timeNormalizer = 12.9
        k = state_obj.dt.dt / fhn_timeNormalizer
        f_phi_pj = f_phi(phi_pj, r_pj)
        f_r_pj = f_r(phi_pj, r_pj)
        self.f_1_pj = Expression("iStim", iStim=0.001, degree=1)

        dx_pj = dolfin.dx(mesh_pj)
        ds_pj = dolfin.ds(mesh_pj, subdomain_data=boundaries)

        self.F_FHN_pj = (
            ((phi_pj - phi_n_pj) / k) * phi_test_pj * dx_pj
            + dot(Dmat_pj * grad(phi_pj), grad(phi_test_pj)) * dx_pj
            - f_phi_pj * phi_test_pj * dx_pj
            + ((r_pj - r_n_pj) / k) * r_test_pj * dx_pj
            - f_r_pj * r_test_pj * dx_pj
            - self.f_1_pj * phi_test_pj * ds_pj(1)
        )

        self.J_FHN_pj = derivative(self.F_FHN_pj, w_pj, dw_pj)

        return (
            self.F_FHN_pj,
            self.J_FHN_pj,
            self.f_1_pj,
        )  # self.f_1, self.f_2

    def Problem_ep(self):
        state_obj = self.parameters["state_obj"]
        f0_ep = self.parameters["f0"]
        AHAid_ep = self.parameters["AHAid"]
        matid_ep = self.parameters["matid"]
        facetboundaries_ep = self.parameters["facetboundaries"]
        EpiBCid_ep = self.MarkStimulus()
        mesh_ep = self.mesh_ep

        W_ep = self.W_ep
        w_n_ep = self.w_n_ep
        w_ep = self.w_ep
        dw_ep = self.dw_ep
        wtest_ep = self.wtest_ep

        phi0 = interpolate(Expression("0.0", degree=0), W_ep.sub(0).collapse())
        r0 = interpolate(Expression("0.0", degree=0), W_ep.sub(1).collapse())

        assign(w_n_ep, [phi0, r0])

        phi_n_ep, r_n_ep = split(w_n_ep)
        phi_ep, r_ep = split(w_ep)

        bcs_ep = []

        phi_test_ep, r_test_ep = split(wtest_ep)

        #####################################################################
        #################### Function Definition ############################
        #####################################################################
        def eps_FHN(phi, r):
            return g + (mu1 * r) / (mu2 + phi)

        def f_phi(phi, r):
            return -c * phi * (phi - alpha) * (phi - 1) - r * phi

        def f_r(phi, r):
            return eps_FHN(phi, r) * (-r - c * phi * (phi - b - 1.0))

        # set parameters for transport
        fhn_timeNormalizer = 12.9
        k = state_obj.dt.dt / fhn_timeNormalizer
        f_phi_ep = f_phi(phi_ep, r_ep)
        f_r_ep = f_r(phi_ep, r_ep)

        D_tensor = self.calculateDmat(f0_ep, mesh=mesh_ep, mId=AHAid_ep)
        Dmat = D_tensor

        self.max_pace_label = int(max(np.array(EpiBCid_ep)))
        pdb.set_trace()  # Execution will pause here
        # assert len(self.parameters["pacing_timing"]) == self.max_pace_label,\
        #     "Number of pacing timing not equal to number of ploc labels"

        self.fstim_array_ep = []
        for p in np.arange(0, self.max_pace_label):
            self.fstim_array_ep.append(Expression("iStim", iStim=0.001, degree=1))

        dx_ep = dolfin.dx(mesh_ep)
        dx_ep_epi = dolfin.dx(mesh_ep, subdomain_data=EpiBCid_ep)

        self.F_FHN_ep = (
            ((phi_ep - phi_n_ep) / k) * phi_test_ep * dx_ep
            + dot(Dmat * grad(phi_ep), grad(phi_test_ep)) * dx_ep
            - f_phi_ep * phi_test_ep * dx_ep
            + ((r_ep - r_n_ep) / k) * r_test_ep * dx_ep
            - f_r_ep * r_test_ep * dx_ep
        )
        label = 1
        # Loop through each mapped facet and assign a stimulant for it
        for fstim in self.fstim_array_ep:
            self.F_FHN_ep -= fstim * phi_test_ep * dx_ep_epi(label)
            label += 1

        self.J_FHN_ep = derivative(self.F_FHN_ep, w_ep, dw_ep)

        return (
            self.F_FHN_ep,
            self.J_FHN_ep,
            self.fstim_array_ep,
        )  # self.f_1, self.f_2

    def Solver_pj(self):
        solverparams_FHN_pj = {
            "Jacobian": self.J_FHN_pj,
            "F": self.F_FHN_pj,
            "w": self.w_pj,
            "boundary_conditions": [],
            "Type": 0,
            "mesh": self.mesh_pj,
            "mode": 1,
        }

        solver_FHN_pj = NSolver(solverparams_FHN_pj)

        return solver_FHN_pj

    def Solver_ep(self):
        solverparams_FHN_ep = {
            "Jacobian": self.J_FHN_ep,
            "F": self.F_FHN_ep,
            "w": self.w_ep,
            "boundary_conditions": [],
            "Type": 0,
            "mesh": self.mesh_ep,
            "mode": 1,
        }
        solver_FHN_ep = NSolver(solverparams_FHN_ep)

        return solver_FHN_ep

    # Function to update activation time in the mesh based on the action potential
    def UpdateVar(self):
        # Update EP variable
        self.w_n_ep.assign(self.w_ep)
        # Update Stimulus variable
        state_obj = self.parameters["state_obj"]

        pace_time_array = self.parameters["pacing_timing"]

        for fstim, pace_time in zip(self.fstim_array, pace_time_array):
            fstim.iStim = 0.0
            time = pace_time[0]
            duration = pace_time[1]
            if state_obj.t >= time and (state_obj.t <= time + duration):
                fstim.iStim = 0.3

    def Reset(self):
        self.w_ep.assign(self.w_n_ep)

    def getphivar(self):
        phi_, r_ = self.w_n_ep.split(deepcopy=True)
        phi_.rename("phi_", "phi_")
        return phi_

    def getrvar(self):
        phi_, r_ = self.w_n_ep.split(deepcopy=True)
        r_.rename("r_", "r_")
        return r_

    def interpolate_potential_ep2me_phi(self, V_me):
        lp = LagrangeInterpolator()
        lp.interpolate(V_me, self.getphivar())
        return V_me

    def reset(self):
        phi0 = interpolate(Expression("0.0", degree=0), self.W_ep.sub(0).collapse())
        r0 = interpolate(Expression("0.0", degree=0), self.W_ep.sub(1).collapse())
        assign(self.w_n_ep, [phi0, r0])
        assign(self.w_ep, [phi0, r0])
        return
