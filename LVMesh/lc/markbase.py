import sys
import os
import numpy as np
sys.path.append("/mnt/Research")
import vtk as vtk
import vtk_py3 as vtk_py3
import dolfin as dolfin
from mpi4py import MPI as pyMPI

def SetVaryingSpring(df_mesh, df_facet, param):

    minztol = param["minztol"] if ("minztol" in param) else 0.5

    VFS = dolfin.FunctionSpace(df_mesh, dolfin.FiniteElement("Lagrange", df_mesh.ufl_cell(), 1))
    # Vvec = FunctionSpace(df_mesh, VectorElement("DG", df_mesh.ufl_cell(), 0))

    comm = dolfin.MPI.comm_world
    minz = comm.allreduce(np.amin(df_mesh.coordinates()[:,2]), op=pyMPI.MIN)

    def apex_boundary(x):
            return x[2] < minz+minztol
 
    
    # bas_ids = [self.parameters["aorta_int_wall"], self.parameters["aorta_ext_wall"], self.parameters["aorta_ring"],]
    # bc_bas = [DirichletBC(VFS, Constant(0.0), df_facet, id_) for id_ in bas_ids]
    bc_bas = dolfin.DirichletBC(VFS, dolfin.Constant(4.0), df_facet, 4)
    # bc_apx = [DirichletBC(VFS, Constant(1.0), df_facet, self.parameters["apxid"])]
    bc_apx = dolfin.DirichletBC(VFS, dolfin.Constant(1.0), apex_boundary)
    bcs = [bc_bas, bc_apx]

    # Define variational problem
    u = dolfin.TrialFunction(VFS)
    v =dolfin.TestFunction(VFS)
    f =dolfin.Constant(0)
    a =dolfin.inner(dolfin.nabla_grad(u), dolfin.nabla_grad(v)) * dolfin.dx
    L = f * v * dolfin.dx
    
    # Compute solution
    u_VFS = dolfin.Function(VFS)
    dolfin.solve(a == L, u_VFS, bcs=bcs, solver_parameters={"linear_solver": "mumps"})
    # grad_u = project(grad(u), Vvec)  # , solver_type="petsc")

    return u_VFS



def MarkBaseVolume(df_mesh, dist2base):

    # Create a MeshFunction for cells (dimension = 2 in 2D)
    cell_markers = dolfin.MeshFunction("size_t", df_mesh, df_mesh.topology().dim())
    cell_markers.set_all(0)  # Set all cells to 0

    # Define a subdomain for x > 0.5
    class Top(dolfin.SubDomain):
        def inside(self, x, on_boundary):
            return x[2] > -dist2base 

    # Instantiate and mark the subdomain
    top = Top()
    top.mark(cell_markers, 1)  # 

    return cell_markers

outdir = "./"
#meshname = "ellipsoidal_baselinegeo_medium2"
#meshname = "ellipsoidal_baselinegeo_coarse"
#meshname = "ellipsoidal_baselinegeo"
meshname = "ellipsoidal_baselinegeo_fine1"
comm = pyMPI.COMM_WORLD

fenics_mesh_ref = dolfin.Mesh()
comm_common = fenics_mesh_ref.mpi_comm()
f = dolfin.HDF5File(comm_common, os.path.join(outdir, meshname + ".hdf5"), "r")
f.read(fenics_mesh_ref, meshname, False)

degree = 4
VQuad = dolfin.FunctionSpace(fenics_mesh_ref, dolfin.VectorElement("Quadrature", fenics_mesh_ref.ufl_cell(), degree=degree, quad_scheme="default"))
CGSpace = dolfin.FunctionSpace(fenics_mesh_ref, "CG", 1)

ef = dolfin.Function(VQuad)
es = dolfin.Function(VQuad)
en = dolfin.Function(VQuad)
u_VFS = dolfin.Function(CGSpace)

fenics_facet_ref = dolfin.MeshFunction("size_t", fenics_mesh_ref, 2)
fenics_edge_ref = dolfin.MeshFunction("size_t", fenics_mesh_ref, 1)
f.read(fenics_facet_ref, meshname + "/" + "facetboundaries")
f.read(fenics_edge_ref, meshname + "/" + "edgeboundaries")
f.read(ef, meshname + "/" + "eF")
f.read(es, meshname + "/" + "eS")
f.read(en, meshname + "/" + "eN")
try:
    f.read(u_VFS, meshname + "/"+ "varyingspring")
except RuntimeError:
    u_VFS = SetVaryingSpring(fenics_mesh_ref, fenics_facet_ref, {})
f.close()

dist2base = 0.7 #Medium and fine
#dist2base = 1.5 #Coarse
#dist2base = 1.0 #Default
matid = MarkBaseVolume(fenics_mesh_ref, dist2base)
dolfin.File("matid.pvd") << matid
dolfin.File("varyingstiffness.pvd") << u_VFS


# Remove file
os.remove(os.path.join(outdir, meshname + ".hdf5"))

# Write file
f = dolfin.HDF5File(comm, os.path.join(outdir, meshname + ".hdf5"), "w")
f.write(fenics_mesh_ref, meshname)

f = dolfin.HDF5File(comm, os.path.join(outdir, meshname + ".hdf5"), "a")
f.write(fenics_facet_ref, meshname + "/" + "facetboundaries")
f.write(fenics_edge_ref, meshname + "/" + "edgeboundaries")
f.write(matid, meshname + "/" + "matid")
f.write(ef, meshname + "/" + "eF")
f.write(es, meshname + "/" + "eS")
f.write(en, meshname + "/" + "eN")
f.write(u_VFS, meshname + "/"+ "varyingspring")
f.close()




