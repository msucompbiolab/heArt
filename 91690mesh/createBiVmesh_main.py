import sys
sys.path.append("/mnt/Research")
import vtk_py as vtk_py
import dolfin as dolfin
import numpy as np
from mpi4py import MPI as pyMPI

# Parameters
epicutfilename = "91690_EP2_2.stl"
LVendocutfilename = "91690_LV2_2.stl"
RVendocutfilename = "91690_RV2_2.stl"
meshsize = 0.6
casename = "BiV"
epiid = 1
lvid = 2
rvid = 3
isLV = False
quad_deg=4
RVangle=[60,-60]
LVangle=[60,-60]
Septangle=[60,-60]
quad_deg = 4
isrotatept=False
isreturn=True
meshname = "91690"
outdir = "./"


# Create VTK mesh
ugrid = vtk_py.create_BiVmesh(epicutfilename, LVendocutfilename, RVendocutfilename, casename, meshsize = meshsize)

fenics_mesh_ref, fenics_facet_ref, fenics_edge_ref = vtk_py.extractFeNiCsBiVFacet(ugrid,tol=1e-1)

# Assigning material IDs
pdata_endLV = vtk_py.readSTL(LVendocutfilename, verbose=False)
pdata_endRV = vtk_py.readSTL(RVendocutfilename, verbose=False)
pdata_epi = vtk_py.readSTL(epicutfilename, verbose=False)
matid = vtk_py.addRegionsToBiV(ugrid, pdata_endLV, pdata_endRV, pdata_epi)
cnt = 0
for cell in dolfin.cells(fenics_mesh_ref):
	 idx = int(ugrid.GetCellData().GetArray("region_id").GetTuple(cnt)[0])
	 fenics_mesh_ref.domains().set_marker((cell.index(), idx), 3)
	 cnt += 1
matid = dolfin.MeshFunction('size_t', fenics_mesh_ref, 3, fenics_mesh_ref.domains())

# Translate mesh so that z = 0
comm = fenics_mesh_ref.mpi_comm().tompi4py()
ztop = comm.allreduce(np.amax(fenics_mesh_ref.coordinates()[:,2]), op=pyMPI.MAX)
ztrans = dolfin.Expression(("0.0", "0.0", str(-ztop)), degree = 1)

if(dolfin.dolfin_version() != '1.6.0'):
	dolfin.ALE.move(fenics_mesh_ref,ztrans)
else:
	mesh.move(ztrans)
	
dolfin.File("BiVFacet.pvd") << fenics_facet_ref
dolfin.File("matid.pvd") << matid

# Set Fiber Orientation
fiber_angle_param = {"mesh": fenics_mesh_ref,\
	 "facetboundaries": fenics_facet_ref,\
	 "LV_fiber_angle": LVangle, \
	 "LV_sheet_angle": [0.1, -0.1], \
	 "Septum_fiber_angle": Septangle,\
	 "Septum_sheet_angle": [0.1, -0.1],\
	 "RV_fiber_angle": RVangle,\
	 "RV_sheet_angle": [0.1, -0.1],\
	 "LV_matid": 0,\
	 "Septum_matid": 1,\
	 "RV_matid": 2,\
	 "matid": matid,\
	 "isrotatept": isrotatept,\
	 "isreturn": isreturn,\
	 "outfilename": meshname,\
	 "outdirectory":outdir,\
	 "epiid": epiid,\
	 "rvid": rvid,\
	 "lvid": lvid,\
	 "degree": quad_deg}
	
ef, es, en = vtk_py.SetBiVFiber_Quad_PyQ(fiber_angle_param)

# Write to hdf5 file
f = dolfin.HDF5File(fenics_mesh_ref.mpi_comm(), outdir + meshname+".hdf5", 'w')
f.write(fenics_mesh_ref, meshname)
f.close()
	
f = dolfin.HDF5File(fenics_mesh_ref.mpi_comm(), outdir + meshname+".hdf5", 'a') 
f.write(fenics_facet_ref, meshname+"/"+"facetboundaries") 
f.write(fenics_edge_ref, meshname+"/"+"edgeboundaries") 
f.write(matid, meshname+"/"+"matid") 
f.write(ef, meshname+"/"+"eF") 
f.write(es, meshname+"/"+"eS") 
f.write(en, meshname+"/"+"eN")
f.close()

