#
# .. _demo_hyperelasticity:
#
# Hyperelasticity

import matplotlib.pyplot as plt
from dolfin import *
import os

# Optimization options for the form compiler
parameters["form_compiler"]["cpp_optimize"] = True
parameters["form_compiler"]["representation"] = "uflacs"
parameters["form_compiler"]["quadrature_degree"] = 2


# Create mesh and define function space
# mesh = UnitCubeMesh(12, 8, 8)
hdf5_file = "fchmesh_scale_w_valves_remarked.hdf5"
case_name = "fchmesh_scale_w_valves_remarked"

base_path = os.getcwd()
mesh_path = os.path.join(base_path, "FCHMesh")


mesh = Mesh()
h = HDF5File(MPI.comm_world, mesh_path + "/" + hdf5_file, "r")
h.read(mesh, case_name, False)

set_log_level(20)

V = VectorElement("CG", mesh.ufl_cell(), 1, quad_scheme="default")
P = FiniteElement("CG", mesh.ufl_cell(), 1, quad_scheme="default")
VP = FunctionSpace(mesh, MixedElement([V, P]))


# Mark boundary subdomians
left = CompiledSubDomain("near(x[0], side) && on_boundary", side=0.0)
right = CompiledSubDomain("near(x[0], side) && on_boundary", side=1.0)


# The Dirichlet boundary values are defined using compiled expressions::
# Define Dirichlet boundary (x = 0 or x = 1)
c = Expression(("0.0", "0.0", "0.0"), degree=2)

facet_tag = MeshFunction("size_t", mesh, mesh.topology().dim() - 1, 0)
h.read(facet_tag, case_name + "/" + "facetboundaries")

# bc_1 = DirichletBC(VP.sub(0), c, facet_tag, 16)
# bc_2 = DirichletBC(VP.sub(0), c, facet_tag, 12)

bc_3 = DirichletBC(VP.sub(0), c, facet_tag, 9)
bc_4 = DirichletBC(VP.sub(0), c, facet_tag, 7)
bcs = [bc_3, bc_4]

# Define functions
dup = TrialFunction(VP)
(du, dp) = split(dup)
vq = TestFunction(VP)
(v, q) = split(vq)
up = Function(VP)
(u, p) = split(up)


# Kinematics
d = len(u)
I = Identity(d)  # Identity tensor
F = I + grad(u)  # Deformation gradient
C = F.T * F  # Right Cauchy-Green tensor

# Invariants of deformation tensors
Ic = tr(C)
J = det(F)


# Elasticity parameters
E, nu = 10.0, 0.3
mu, lmbda = Constant(E / (2 * (1 + nu))), Constant(E * nu / ((1 + nu) * (1 - 2 * nu)))

# Stored strain energy density (compressible neo-Hookean model)
psi = (mu / 2) * (Ic - 3) - p * (J - 1)

# Total potential energy
Pi = psi * dx

ds_ = Measure("ds", domain=mesh, subdomain_data=facet_tag)
P_LV = Constant(7.0e-2)
N = FacetNormal(mesh)
pres = P_LV * inner(J * inv(F) * N, u) * ds_(18)
Fp = derivative(pres, up, vq)

# stabilization
# Fs = (
#    1.0
#    / (CellVolume(mesh)) ** (1.0 / 3.0)
#    * (p - p / CellVolume(mesh))
#    * (q - q / CellVolume(mesh))
#    * dx
# )

# to dos:
# a) perhaps, with spring-dashpot, it becomes more stable
# b) a different stabilization term may work

h_elem = CellDiameter(mesh)
Fs = -(
    h_elem
    * h_elem
    * Constant(0.5)
    / mu
    * J
    * inner(inv(F.T) * grad(p), inv(F.T) * grad(q))
    * dx
)

k_spring = [2.0e0, 2.0e0]
F3 = inner(
    outer(N, N) * (k_spring[0] * u),
    v,
) * ds(17)
+inner(
    (Identity(u.ufl_shape[0]) - outer(N, N)) * (k_spring[1] * u),
    v,
) * ds(17)


# Compute first variation of Pi (directional derivative about u in the direction of v)
# F = derivative(Pi, u, v)
F1 = derivative(psi, up, vq) * dx
FF = F1 + Fp + Fs + F3


# Compute Jacobian of F
# J = derivative(F, u, du)
J1 = derivative(F1, up, dup)
Jp = derivative(Fp, up, dup)
Js = derivative(Fs, up, dup)
J3 = derivative(F3, up, dup)
JJ = J1 + Jp + Js + J3


# Solve variational problem
# solve(F == 0, u, bcs, J=J)

problem = NonlinearVariationalProblem(
    FF, up, bcs, JJ, form_compiler_parameters={"representation": "uflacs"}
)

solver = NonlinearVariationalSolver(problem)

# Optional: Set solver parameters
solver.parameters["nonlinear_solver"] = "newton"
solver.parameters["newton_solver"]["linear_solver"] = "mumps"
solver.parameters["newton_solver"]["report"] = True
solver.parameters["newton_solver"]["absolute_tolerance"] = 1e-9
solver.parameters["newton_solver"]["relative_tolerance"] = 1e-9
solver.parameters["newton_solver"]["maximum_iterations"] = 100

solver.solve()

(u_, p_) = up.split()

result_folder = os.path.join(os.getcwd(), "results_remarked_p1p1")

# Save solution in VTK format
u_.rename("u_", "u_")
f_u = File(os.path.join(result_folder, "displacement_quad.pvd"))
f_u << u_

p_.rename("p_", "p_")
f_p = File(os.path.join(result_folder, "pressure_quad.pvd"))
f_p << p_
