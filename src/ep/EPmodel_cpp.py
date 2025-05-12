# Electrophysiology model using Aliev Panilfov model
from dolfin import *
import numpy as np
import os
import inspect

# from fenicstools import *
from ufl import indices
import dolfin as dolfin
from mpi4py import MPI as pyMPI
import pdb
from petsc4py import PETSc

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

        if "dt" in list(self.parameters.keys()):
            dt = self.parameters["dt"]
        else:
            dt = Constant(1.0) 

        if "fhn_timeNormalizer" in list(self.parameters.keys()):
            fhn_timeNormalizer = self.parameters["fhn_timeNormalizer"]
        else:
            fhn_timeNormalizer = 12.9 
 
        self.k = dt/fhn_timeNormalizer
        self.EPparams = {
            "g": 0.002,
            "mu1": 0.2,
            "mu2": 0.3,
            "c": 8,
            "alpha": 0.01,
            "b": 0.15,
            "k": self.k,
        }

        if "d_iso" in list(self.parameters.keys()):
            self.d_iso = self.parameters["d_iso"]
        else:
            self.d_iso = 1.0

        if "d_ani" in list(self.parameters.keys()):
            self.d_ani = self.parameters["d_ani"]
        else:
            self.d_ani = 0.0

        if "ani_factor" in list(self.parameters.keys()):
            self.ani_factor = self.parameters["ani_factor"]
        else:
            self.ani_factor = 1.0

        P1_ep = FiniteElement("CG", self.mesh.ufl_cell(), 1, quad_scheme="default")
        P1_ep._quad_scheme = "default"

        self.W_ep = FunctionSpace(self.mesh, P1_ep)
        self.w_ep = Function(self.W_ep)
        self.r_ep = Function(self.W_ep)
        self.f_phi_ = Function(self.W_ep)
        self.dw_ep = TrialFunction(self.W_ep)
        self.wtest_ep = TestFunction(self.W_ep)
        self.w_n_ep = Function(self.W_ep)
        self.t1_ep = PETScVector()
        self.t1_ep = self.w_n_ep.vector().copy()
        self.I_source = Function(self.W_ep)

        self.I_source_vec = PETScVector()
        self.I_source_vec = self.w_n_ep.vector().copy()


        self.Mass_P, self.MD_matrix, self.fstim_val_array, self.FHNmodel = self.Problem()
        #self.Solver = self.SolverEP(self)

    def Solver(self):
        return self.SolverEP(self)

    def default_parameters(self):
        return {"ploc": [[0.0, 0.0, 0.0, 100.0, 1]], "pacing_timing": [[1, 20.0]],}

    def calculateDmat(self, f0, s0, n0, mesh):
        d_iso = self.d_iso  # 0.01 #0.02
        d_ani = self.d_ani  # 0.08 #0.1 #0.2
        ani_factor = self.ani_factor

        d_Ani = d_ani

        dParam = {}

        D_iso = Constant(
            ((d_iso, "0.0", "0.0"), ("0.0", d_iso, "0.0"), ("0.0", "0.0", d_iso))
        )

        i, j = indices(2)

        Dij = (d_Ani)*f0[i]*f0[j] + (d_Ani*1/ani_factor)*s0[i]*s0[j] + (d_Ani*1/ani_factor)*n0[i]*n0[j]+ D_iso[i,j]
        D_tensor = as_tensor(Dij, (i, j))
        return D_tensor

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

        if "AHAid_ep" in list(self.parameters.keys()):
            AHAid_ep = self.parameters["AHAid"]

        if "matid" in list(self.parameters.keys()):
            matid_ep = self.parameters["matid"]

        if "facetboundaries" in list(self.parameters.keys()):
            self.facetboundaries_ep = self.parameters["facetboundaries"]
        else:
            self.facetboundaries_ep = None

        mesh = self.mesh

        W_ep = self.W_ep
        w_n_ep = self.w_n_ep
        w_ep = self.w_ep
        dw_ep = self.dw_ep
        wtest_ep = self.wtest_ep

        # Create a lumped mass matrix (M) --------------------------------------------
        mass_form = dw_ep*wtest_ep*dx
        mass_action_form = action(mass_form, Constant(1))
        M_lumped = assemble(mass_form)
        M_lumped.zero()
        M_lumped.set_diagonal(assemble(mass_action_form))
        M_lumped_P = as_backend_type(M_lumped).mat()
        M_lumped_P = PETScMatrix(M_lumped_P)
        self.Mass_P = M_lumped_P.copy()
        pv = PETSc.Viewer()
        #pv(self.Mass_P.mat())
        #-----------------------------------------------------------------------------

        # Create Conductivity matrix (A) ---------------------------------------------
        comm = mesh.mpi_comm()  # MPI Communicator
        if 'interval' in mesh.ufl_cell()._cellname:
            if "lbbb" in list(self.parameters.keys()):
                if self.parameters["lbbb"]:

                    # Create a piecewise scaling Function
                    Dmat_scaling = Function(FunctionSpace(mesh, "DG", 0))
                    Dmat_values = Dmat_scaling.vector()

                    Dmat_values[:] = np.where(matid_ep.array() == 1, self.parameters["lbbb_delay"], 1.0)

                    # Set the final `Dmat` variable
                    Dmat = Constant(self.d_iso) * Dmat_scaling

                else:
                    Dmat = Constant(self.d_iso)
            else:
                Dmat = Constant(self.d_iso)
        else:
            D_tensor = self.calculateDmat(f0_ep, s0_ep, n0_ep, mesh=mesh)
            Dmat = D_tensor

        A = PETScMatrix()
        Fconduct = inner(Dmat*grad(dw_ep),grad(wtest_ep))*dx
        assemble(Fconduct, tensor=A)

        # Create source term (I) -----------------------------------------------------
        self.fstim_array = []
        self.fstim_val_array = []
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
            #fstim = Expression("iStim", iStim=0.000, degree=1)
            #self.fstim_array.append(Expression("iStim", iStim=0.000, degree=1))
            self.fstim_val_array.append(0)
            x0 = self.ploc[p]
            delta = dolfin.CompiledExpression(dolfin.compile_cpp_code(cpp_code0).test_exp(), degree=1)
            delta.eps = havg
            delta.x0 = np.array(x0, dtype=float)
            self.delta_array.append(delta)

        dx_ep = dolfin.dx(mesh)

        #self.I_source_form = 0
        #for fstim, delta in zip(self.fstim_array, self.delta_array):
        for delta in self.delta_array:
            fstim_vector = self.I_source_vec.copy()
            fstim_form = Constant(1.0) * delta * wtest_ep * dx_ep
            assemble(fstim_form, tensor=fstim_vector)
            self.fstim_array.append(fstim_vector)
            #self.I_source_form +=  fstim * delta * wtest_ep * dx_ep

        #assemble(self.I_source_form, tensor=self.I_source.vector())
        #-----------------------------------------------------------------------------

        # Create fphi (that will be updated in cpp) ----------------------------------
        F_phi = Constant(0.0)*wtest_ep*dx
        assemble(F_phi, tensor=self.f_phi_.vector())
        #-----------------------------------------------------------------------------

        # Define FHN model from cpp to update fphi and r ----------------------------
        directory = os.path.dirname(inspect.getfile(self.__class__))
        cpp_file = open(os.path.join(directory,"fhn.cpp"),"r")
        code = cpp_file.read()
        cpp_file.close()
        ext_module = compile_cpp_code(code)
        self.FHNmodel = ext_module.FHNmodel(self.EPparams)
        #-----------------------------------------------------------------------------

        # Create M + kA matrix (for implicit time integration) -----------------------
        self.MD_matrix = self.Mass_P.copy()
        self.MD_matrix.mat().axpy(float(self.k), A.mat())
        #-----------------------------------------------------------------------------
         
        return (self.Mass_P, self.MD_matrix, self.fstim_val_array, self.FHNmodel)

    class SolverEP:
        def __init__(self, EPmodel):

            self.EPmodel = EPmodel
            self.solver = PETScLUSolver()  # Use direct LU solver if preferred
            self.r_ep = EPmodel.r_ep
            self.w_ep = EPmodel.w_ep
            self.w_n_ep = EPmodel.w_n_ep
            self.f_phi_ = EPmodel.f_phi_
            self.FHNmodel = EPmodel.FHNmodel
            self.Mass_P = EPmodel.Mass_P
            self.t1_ep = EPmodel.t1_ep
            self.k = EPmodel.k
            self.I_source_vec = EPmodel.I_source_vec
            self.MD_matrix = EPmodel.MD_matrix

        def solvenonlinear(self):
    
            r_vec = self.r_ep.vector()
            w_vec = self.w_ep.vector()
            f_phi_vec = self.f_phi_.vector()
    
            # Update fphi using cpp module
            self.FHNmodel.Update_fphi(f_phi_vec, w_vec, r_vec)
            self.f_phi_.vector().set_local(f_phi_vec)
        
            # Solve for u for (M + kA)*u = M*(un + k*fphi + I)
            self.w_n_ep.vector().axpy(float(self.k), self.f_phi_.vector())
            self.Mass_P.mult(self.w_n_ep.vector(), self.t1_ep)
            print("Solving t1: ", max(self.I_source_vec.get_local()))
            self.t1_ep.axpy(float(self.k), self.I_source_vec)
            self.solver.solve(self.MD_matrix, self.w_ep.vector(), self.t1_ep)
            if np.isnan(np.min(self.w_ep.vector().get_local())):
                print("Nan encountered")
                exit(1)

    def UpdateActivation(self):
        self.I_source_vec[:] = 0
        #print(self.fstim_val_array[0],self.fstim_array[0])
        for p in np.arange(0, self.max_pace_label):
            self.I_source_vec.axpy(self.fstim_val_array[p], self.fstim_array[p])
        #print("Updating Activation:", max(self.I_source_vec.get_local()))

    def UpdateVar(self):
        # Update EP variable
        self.w_n_ep.assign(self.w_ep)

        # Update Stimulus variable
        pace_time_array = self.parameters["pacing_timing"]

        # Update internal variable r using cpp module
        r_vec = self.r_ep.vector()
        w_vec = self.w_ep.vector()
        f_phi_vec = self.f_phi_.vector()

        self.FHNmodel.Update_r(w_vec, r_vec, self.EPparams)
        self.r_ep.vector().set_local(r_vec)
        self.t1_ep[:] = 0

    def Reset(self):
        self.w_ep.assign(self.w_n_ep)

    def getphivar(self):
        phi_ = self.w_ep
        phi_.rename("phi_", "phi_")
       
        return phi_

    def getrvar(self):
        r_ = self.r_ep
        r_.rename("r_", "r_")

        return r_

    def interpolate_potential_ep2me_phi(self, V_me):
        LagrangeInterpolator.interpolate(V_me, self.getphivar())
        return V_me

    def reset(self):
        self.w_n_ep.vector().zero()
        self.w_ep.vector().zero()
        self.r_ep.vector().zero()

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



