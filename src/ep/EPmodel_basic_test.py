# Electrophysiology model using Aliev Panilfov model
from dolfin import *
import numpy as np
from ..utils.nsolver import NSolver as NSolver

# from fenicstools import *
from ufl import indices
import dolfin as dolfin
from mpi4py import MPI as pyMPI
import pdb

class EPmodel(object):
    def __init__(self, params):
        self.parameters = self.default_parameters()
        self.parameters.update(params)
        self.mesh = self.parameters["EPmesh"]

        if "ploc" in list(self.parameters.keys()):
            self.max_pace_label = len(self.parameters["ploc"])
            self.ploc = self.parameters["ploc"]
        else:
            self.max_pace_label = 0
            self.ploc = None

        P1_ep = FiniteElement("CG", self.mesh.ufl_cell(), 1, quad_scheme="default")
        P1_ep._quad_scheme = "default"
        P2_ep = FiniteElement("DG", self.mesh.ufl_cell(), 0, quad_scheme="default")
        P2_ep._quad_scheme = "default"

        self.W_ep = FunctionSpace(self.mesh, MixedElement([P1_ep, P2_ep]))
        self.w_ep = Function(self.W_ep)
        self.dw_ep = TrialFunction(self.W_ep)
        self.wtest_ep = TestFunction(self.W_ep)
        self.w_n_ep = Function(self.W_ep)

        self.F_FHN, self.J_FHN, self.fstim_array = self.Problem()

    def default_parameters(self):
        return {"ploc": [[0.0, 0.0, 0.0, 100.0, 1]], "pacing_timing": [[1, 20.0]],}

    def calculateDmat(self, f0, s0, n0, mesh, mId):
        d_iso = self.parameters["d_iso"]  # 0.01 #0.02
        d_ani = self.parameters["d_ani"]  # 0.08 #0.1 #0.2
        ani_factor = self.parameters["ani_factor"]

        d_Ani = d_ani

        dParam = {}

        D_iso = Constant(
            ((d_iso, "0.0", "0.0"), ("0.0", d_iso, "0.0"), ("0.0", "0.0", d_iso))
        )

        i, j = indices(2)

        Dij = (d_Ani)*f0[i]*f0[j] + (d_Ani*1/ani_factor)*s0[i]*s0[j] + (d_Ani*1/ani_factor)*n0[i]*n0[j]+ D_iso[i,j]
        D_tensor = as_tensor(Dij, (i, j))
        return D_tensor

    # # Define a condition function for volume pacing instead of facet
    # def condition_fct(ploc_coord, vertex, ploc_tol):
    #     dist=distance_fct(ploc_coord,vertex)
    #     return dist < ploc_tol

    # def distance_fct(term_nodes_coord, vertex):
    #     # Calculate the Euclidean distance
    #     diff = term_nodes_coord - vertex
    #     dist = np.sqrt(np.sum(diff**2))
    #     return dist

    def MarkStimulus(self):
        mesh = self.mesh
        ploc = self.parameters["ploc"]
        EpiBCid_ep = MeshFunction("size_t", mesh, mesh.topology().dim() - 1)
        EpiBCid_ep.set_all(0)

        for ploc_ in ploc:
            class Omega_0(SubDomain):
                def inside(self, x, on_boundary):
                    return (
                        (x[0] - ploc_[0]) ** 2
                        + (x[1] - ploc_[1]) ** 2
                        + (x[2] - ploc_[2]) ** 2
                    ) <= ploc_[3] ** 2

            subdomain_0 = Omega_0()
            subdomain_0.mark(EpiBCid_ep, int(ploc_[4]))

        # Find maximum number of pace label
        self.max_pace_label = int(max(np.array(ploc)[:, 4]))
        return EpiBCid_ep

    def Problem(self):
        state_obj = self.parameters["state_obj"]

        if "f0" in list(self.parameters.keys()):
            f0_ep = self.parameters["f0"]

        if "s0" in list(self.parameters.keys()):
            s0_ep = self.parameters["s0"]

        if "n0" in list(self.parameters.keys()):
            n0_ep = self.parameters["n0"]

        AHAid_ep = self.parameters["AHAid"]
        matid_ep = self.parameters["matid"]
        if "facetboundaries" in list(self.parameters.keys()):
            facetboundaries_ep = self.parameters["facetboundaries"]
        else:
            facetboundaries_ep = None

        mesh = self.mesh

        W_ep = self.W_ep
        w_n_ep = self.w_n_ep
        w_ep = self.w_ep
        dw_ep = self.dw_ep
        wtest_ep = self.wtest_ep
        #state_obj = self.state_obj

        phi0 = interpolate(Expression("0.0", degree=0), W_ep.sub(0).collapse())
        r0 = interpolate(Expression("0.0", degree=0), W_ep.sub(1).collapse())

        assign(w_n_ep, [phi0, r0])

        phi_n, r_n = split(w_n_ep)
        phi, r = split(w_ep)

        bcs_ep = []

        phi_test, r_test = split(wtest_ep)

        alpha = Constant(0.01)
        g = Constant(0.002)
        b = Constant(0.15)
        c = Constant(8)
        mu1 = Constant(0.2)
        mu2 = Constant(0.3)

        def eps_FHN(phi, r):
            return g + (mu1 * r) / (mu2 + phi)

        def f_phi(phi, r):
            return -c * phi * (phi - alpha) * (phi - 1) - r * phi

        def f_r(phi, r):
            return eps_FHN(phi, r) * (-r - c * phi * (phi - b - 1.0))

        fhn_timeNormalizer = 12.9
        k = state_obj.dt / fhn_timeNormalizer
        f_phi = f_phi(phi, r)
        f_r = f_r(phi, r)

        # if('interval' in mesh.ufl_cell()._cellname):
        #     if self.parameters["lbbb"]:
        #         # Create a MeshFunction to mark the target cell
        #         cell_markers = MeshFunction("size_t", mesh, mesh.topology().dim())
        #         cell_markers.set_all(0)  # Default: all cells are unmarked
        #         # Define a piecewise scalar multiplier for Dmat_pj
        #         Dmat_scaling = Function(FunctionSpace(mesh, "DG", 0))

        #         # Mark the cell at `lbbb_location`
        #         for cell in cells(mesh):
        #             if cell.index() == self.parameters["lbbb_location"]:
        #                 cell_markers[cell] = 1  # Mark this cell with a special tag

        #         # Dmat = Constant(self.parameters["d_iso"] * self.parameters["lbbb_delay"])
        #         Dmat_scaling.vector()[:] = np.where(cell_markers.array() == 1, self.parameters["lbbb_delay"], 1.0)
        #         Dmat = Constant(self.parameters["d_iso"]) * Dmat_scaling
        #     else:
        #         Dmat = Constant(self.parameters["d_iso"])
        # else:
        #     D_tensor = self.calculateDmat(f0_ep,s0_ep, n0_ep, mesh=mesh, mId=AHAid_ep)
        #     Dmat = D_tensor

        comm = mesh.mpi_comm()  # MPI Communicator
        if 'interval' in mesh.ufl_cell()._cellname:
            if "lbbb" in list(self.parameters.keys()):
                if self.parameters["lbbb"]:
                    # Parallel MeshFunction for marking cells
                    #cell_markers = MeshFunction("size_t", mesh, mesh.topology().dim(), 0)

                    # Create a piecewise scaling Function
                    Dmat_scaling = Function(FunctionSpace(mesh, "DG", 0))
                    Dmat_values = Dmat_scaling.vector()

                    ## Get local cell indices for parallel execution
                    #local_cells = list(cells(mesh))  # Local portion of mesh
                    #local_indices = np.array([cell.index() for cell in local_cells])

                    ## Parallel: Mark the specific cell on each processor
                    #marked_cells = (local_indices == self.parameters["lbbb_location"]).astype(np.uint8)
                    #for cell, mark in zip(local_cells, marked_cells):
                    #    cell_markers[cell] = mark  # Only mark relevant local cells

                    ## Ensure all processors have updated `cell_markers`
                    #mesh.mpi_comm().barrier()

                    # Parallel: Assign `Dmat_scaling` values using `assemble()`
                    #Dmat_values[:] = np.where(cell_markers.array() == 1, self.parameters["lbbb_delay"], 1.0)
                    Dmat_values[:] = np.where(matid_ep.array() == 1, self.parameters["lbbb_delay"], 1.0)

                    # Set the final `Dmat` variable
                    Dmat = Constant(self.parameters["d_iso"]) * Dmat_scaling
                    #File("mesh_PJ_marked.pvd") << Dmat_scaling
                    #stop
                    #stop

                else:
                    Dmat = Constant(self.parameters["d_iso"])
            else:
                Dmat = Constant(self.parameters["d_iso"])
        else:
            D_tensor = self.calculateDmat(f0_ep, s0_ep, n0_ep, mesh=mesh, mId=AHAid_ep)
            Dmat = D_tensor

        self.fstim_array = []
        self.delta_array = []
        hmin = mesh.hmin()
        hmax = mesh.hmax()
        havg = 0.02*(hmin + hmax)

        cpp_code0 = """
            
            #include <pybind11/pybind11.h>
            #include <pybind11/eigen.h>
            #include <dolfin/function/Expression.h>
            #include <iostream>
            
            class test_exp : public dolfin::Expression {
              public:
                
                Eigen::VectorXd x0;
                double eps;
                
                test_exp() : dolfin::Expression() { }
        
                void eval(Eigen::Ref<Eigen::VectorXd> values, Eigen::Ref<const Eigen::VectorXd> x) const {
                    double norm_squared = 0.0;
                    for (int i = 0; i < x0.size(); ++i)
                    {
                        norm_squared += (x[i] - x0[i]) * (x[i] - x0[i]);
                    }
                    //values[0] = eps /  std::sqrt(norm_squared + eps * eps);
                    values[0] = eps / M_PI / (norm_squared + eps * eps);
                    //std::cout << values[0] << " " <<  x[0] << " " <<  x[1] << " " << x[2] << std::endl;
                    //std::cout << values[0] << " " <<  x[0] << " " <<  x[1] << " " << x[2] << std::endl;
                }
        
            };
            
            PYBIND11_MODULE(SIGNATURE, m) {
                pybind11::class_<test_exp, std::shared_ptr<test_exp>, dolfin::Expression>
                (m, "test_exp")
                .def(pybind11::init<>())
                .def_readwrite("x0", &test_exp::x0)
                .def_readwrite("eps", &test_exp::eps)
                ;
            }
        """

        for p in np.arange(0, self.max_pace_label):
            fstim = Expression("iStim", iStim=0.000, degree=1)

            self.fstim_array.append(Expression("iStim", iStim=0.000, degree=1))
            x0 = self.ploc[p]
            delta = dolfin.CompiledExpression(dolfin.compile_cpp_code(cpp_code0).test_exp(), degree=1)
            delta.eps = havg
            delta.x0 = np.array(x0, dtype=float)
            self.delta_array.append(delta)
            #self.delta_array.append(Delta(eps=havg, x0=np.array(x0), degree=1))
            #delta = Delta(eps=havg, x0=np.array(x0), degree=1)
            #self.F_FHN -=  fstim * delta * phi_test * dx_ep


        dx_ep = dolfin.dx(mesh)

        self.F_FHN = (
            ((phi - phi_n) / k) * phi_test * dx_ep
            + dot(Dmat * grad(phi), grad(phi_test)) * dx_ep
            - f_phi * phi_test * dx_ep
            + ((r - r_n) / k) * r_test * dx_ep
            - f_r * r_test * dx_ep
        )

#        self.F_FHN = (
#           ((phi - phi_n)) * phi_test * dx_ep
#           + k * dot(Dmat * grad(phi), grad(phi_test)) * dx_ep
#           - k * f_phi * phi_test * dx_ep
#           + ((r - r_n)) * r_test * dx_ep
#           - k * f_r * r_test * dx_ep
#        )


        for fstim, delta in zip(self.fstim_array, self.delta_array):
            self.F_FHN -=  fstim * delta * phi_test * dx_ep
            #self.F_FHN -=  k * fstim * delta * phi_test * dx_ep
        # for fstim in self.fstim_array:            
        #     self.F_FHN -=  fstim  * phi_test * dx_ep

        self.J_FHN = derivative(self.F_FHN, w_ep, dw_ep)
        return (self.F_FHN, self.J_FHN, self.fstim_array,)

    def Solver(self):
        solverparams_FHN = {
            "Jacobian": self.J_FHN,
            "F": self.F_FHN,
            "w": self.w_ep,
            "boundary_conditions": [],
            "Type": 0,
            "mesh": self.mesh,
            "mode": 1,
        }

        solver_FHN = NSolver(solverparams_FHN)
        return solver_FHN

    def UpdateVar(self):
        # Update EP variable
        self.w_n_ep.assign(self.w_ep)

        # Update Stimulus variable
        #self.state_obj = self.parameters["state_obj"]
        pace_time_array = self.parameters["pacing_timing"]

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
        LagrangeInterpolator.interpolate(V_me, self.getphivar())
        return V_me

    def reset(self):
        phi0 = interpolate(Expression("0.0", degree=0), self.W_ep.sub(0).collapse())
        r0 = interpolate(Expression("0.0", degree=0), self.W_ep.sub(1).collapse())
        assign(self.w_n_ep, [phi0, r0])
        assign(self.w_ep, [phi0, r0])
        return

class Delta(UserExpression):
    def __init__(self, eps, x0, **kwargs):
        super().__init__(**kwargs)
        self.eps = eps
        self.x0 = x0

    def eval(self, values, x):
        eps = self.eps
        x0 = self.x0
        values[0] = eps/pi/(np.linalg.norm(x-x0)**2 + eps**2)

    def value_shape(self):
        return ()



