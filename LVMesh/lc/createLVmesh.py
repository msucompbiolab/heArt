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

outdir = "./"
#meshname = "ellipsoidal_baselinegeo_medium3"
meshname = "ellipsoidal_baselinegeo_fine1"
comm = pyMPI.COMM_WORLD

'''
ugrid = vtk_py3.readUGrid("ellipsoidal_mesh_fine1.vtk")
#ugrid = vtk_py3.readUGrid("ellipsoidal_mesh_medium2.vtk")
#ugrid = vtk_py3.readUGrid("ellipsoidal_mesh_medium3.vtk")
#ugrid = vtk_py3.readUGrid("ellipsoidal_mesh_coarse.vtk")
fenics_mesh_ref, fenics_facet_ref, fenics_edge_ref = vtk_py3.extractFeNiCsBiVFacet(ugrid, geometry="LV", tol=1e-2)

f = dolfin.HDF5File(comm, os.path.join(outdir, meshname + "mesh.hdf5"), "w")
f.write(fenics_mesh_ref, meshname)
f.close()

f = dolfin.HDF5File(comm, os.path.join(outdir, meshname + "mesh.hdf5"), "a")
f.write(fenics_facet_ref, meshname + "/" + "facetboundaries")
f.write(fenics_edge_ref, meshname + "/" + "edgeboundaries")
f.close()
stop
'''

os.system("rm *.pvd")
os.system("rm *.vtp")
os.system("rm *.pvtp")
os.system("rm *.vtu")
os.system("rm *.pvtu")


fenics_mesh_ref = dolfin.Mesh()
comm_common = fenics_mesh_ref.mpi_comm()
f = dolfin.HDF5File(comm_common, os.path.join(outdir, meshname + "mesh.hdf5"), "r")
f.read(fenics_mesh_ref, meshname, False)

fenics_facet_ref = dolfin.MeshFunction("size_t", fenics_mesh_ref, 2)
fenics_edge_ref = dolfin.MeshFunction("size_t", fenics_mesh_ref, 1)
f.read(fenics_facet_ref, meshname + "/" + "facetboundaries")
f.read(fenics_edge_ref, meshname + "/" + "edgeboundaries")
f.close()


dolfin.File("test_mesh.pvd") << fenics_mesh_ref
dolfin.File("test_facet.pvd") << fenics_facet_ref
dolfin.File("test_edge.pvd") << fenics_edge_ref

# Set Fiber Orientation
fiber_angle_param = {
    "mesh": fenics_mesh_ref,
    "facetboundaries": fenics_facet_ref,
    "LV_fiber_angle": [60, -60],
    "LV_sheet_angle": [0.1, -0.1],
    "minztol": 0.5,#0.8, # Coarse mesh
    "isrotatept": False,
    "isreturn": True,
    "outfilename": meshname,
    "outdirectory": outdir+"/",
    "baseid": 4,
    "epiid": 1,
    "lvid": 2,
    "degree": 4,
}

u_VFS = SetVaryingSpring(fenics_mesh_ref, fenics_facet_ref, fiber_angle_param)
dolfin.File("varyingstiffness.pvd") << u_VFS

ef, es, en = vtk_py3.addLVfiber_LDRB(fiber_angle_param)


# Write to hdf5 file
f = dolfin.HDF5File(comm, os.path.join(outdir, meshname + ".hdf5"), "w")
f.write(fenics_mesh_ref, meshname)
f.close()

f = dolfin.HDF5File(comm, os.path.join(outdir, meshname + ".hdf5"), "a")
f.write(fenics_facet_ref, meshname + "/" + "facetboundaries")
f.write(fenics_edge_ref, meshname + "/" + "edgeboundaries")
f.write(ef, meshname + "/" + "eF")
f.write(es, meshname + "/" + "eS")
f.write(en, meshname + "/" + "eN")
f.write(u_VFS, meshname + "/"+ "varyingspring")
f.close()

  


