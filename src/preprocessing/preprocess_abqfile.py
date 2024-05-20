import dolfin
from dolfin import *
import vtk
import sys

sys.path.append("/home/hagersan/github")
import vtk_py
from vtk_py import *
import pdb

# from numpy import concatenate as cat


def thresholdpdata(pdata, id_):
    threshold = vtk.vtkThreshold()
    threshold.SetInputData(pdata)
    threshold.ThresholdBetween(id_ - 0.1, id_ + 0.1)
    threshold.SetInputArrayToProcess(
        0, 0, 0, vtk.vtkDataObject.FIELD_ASSOCIATION_CELLS, "facet id"
    )
    threshold.Update()

    geometry_filter = vtk.vtkGeometryFilter()
    geometry_filter.SetInputData(threshold.GetOutput())
    geometry_filter.Update()

    return geometry_filter.GetOutput()


# Read Mesh
filename = "mesh_vol2.inp"

ugrid = vtk_py.readAbaqusMesh(filename, "tet")

# Read in facets
filename = "mesh_surf2.inp"
# pdb.set_trace()  # Execution will pause here
pdata = vtk_py.readAbaqusFacet(filename, "quad")

# Build point locator about the cell center
pdata_cellcenter = getCellCenters(pdata)
pdata_pointlocator = vtk.vtkPointLocator()
pdata_pointlocator.SetDataSet(pdata_cellcenter)
pdata_pointlocator.BuildLocator()

pdata_endo = thresholdpdata(pdata, 1)
pdata_epi = thresholdpdata(pdata, 2)
# vtk_py.writeXMLPData(pdata_endo, "pdata_endo.vtp")
# vtk_py.writeXMLPData(pdata_epi, "pdata_epi.vtp")

# Write mesh and surface
vtk_py.writeUGrid(ugrid, "vol_mesh1.vtk")
vtk_py.writePData(pdata, "vol_surfaces1.vtk")

pdb.set_trace()  # Execution will pause here

# Add fiber direction by prescribing base to apex point
pt = np.array([16.9547, 4.3538, 72.4845])
nvec = np.array([-0.644247, -0.203088, -0.73736])
points_AB = vtk.vtkPoints()
points_AB.InsertNextPoint(pt)
points_AB.InsertNextPoint(pt + nvec)

addLocalProlateSpheroidalDirections(
    ugrid,
    pdata_endo,
    pdata_epi,
    type_of_support="cell",
    endoflip=True,
    points_AB=points_AB,
)
addLocalFiberOrientation(ugrid, 60, -60)


# Convert to dolfin mesh
dolfin_mesh = vtk_py.convertUGridToXMLMesh(ugrid)

# Mark facet of dolfin mesh
dolfin_facet = MeshFunction("size_t", dolfin_mesh, dolfin_mesh.topology().dim() - 1, 0)
DomainBoundary().mark(dolfin_facet, 1)  # Mark all the exterior facets as 1
mark_facets = np.zeros(len(dolfin_facet.array()))

for facet in dolfin.SubsetIterator(dolfin_facet, 1):
    for cell in cells(facet):
        cx = 0
        cy = 0
        cz = 0
        for vertex in vertices(facet):
            cx += vertex.point().array()[0] / 3.0
            cy += vertex.point().array()[1] / 3.0
            cz += vertex.point().array()[2] / 3.0
        pdata_cellid = pdata_pointlocator.FindClosestPoint(cx, cy, cz)
        facet_id = (
            pdata_cellcenter.GetPointData().GetArray("facet id").GetValue(pdata_cellid)
        )
        # print(pdata_cellid, facet_id)
        mark_facets[facet.index()] = int(facet_id)

dolfin_facet.array()[:] = mark_facets

pdb.set_trace()  # Execution will pause here
# Boundary edges of mesh
# Create an EdgeFunction to store boundary edge markers

mesh_xml, facet_xml, edge_boundaries = vtk_py.extractFeNiCsBiVFacet(
    ugrid, geometry="LV", tol=1e-2
)

pdb.set_trace()  # Execution will pause here

# Add fiber as DG0 space
VQuadelem = dolfin.VectorElement("DG", dolfin_mesh.ufl_cell(), degree=0)
fiberFS = dolfin.FunctionSpace(dolfin_mesh, VQuadelem)
fiber_fs = dolfin.Function(fiberFS)
sheet_fs = dolfin.Function(fiberFS)
normal_fs = dolfin.Function(fiberFS)

fvec_array = numpy_support.vtk_to_numpy(
    ugrid.GetCellData().GetArray("fiber vectors")
).flatten()
svec_array = numpy_support.vtk_to_numpy(
    ugrid.GetCellData().GetArray("sheet vectors")
).flatten()
nvec_array = numpy_support.vtk_to_numpy(
    ugrid.GetCellData().GetArray("sheet normal vectors")
).flatten()

fiber_fs.vector()[:] = fvec_array
sheet_fs.vector()[:] = svec_array
normal_fs.vector()[:] = nvec_array


File("vol_surfaces1.pvd") << dolfin_facet
File("vol_mesh1.pvd") << dolfin_mesh
File("fiber.pvd") << fiber_fs
File("edgeboundaries.pvd") << edge_boundaries

# Write to HDF5
hdf5_file = "vol_mesh1.hdf5"
case_name = "vol_mesh1"
f = HDF5File(mpi_comm_world(), hdf5_file, "w")
f.write(dolfin_mesh, case_name)
f.close()

f = HDF5File(mpi_comm_world(), hdf5_file, "a")
f.write(dolfin_facet, case_name + "/" + "facetboundaries")
f.write(edge_boundaries, case_name + "/" + "edgeboundaries")
f.write(fiber_fs, case_name + "/" + "eF")
f.write(sheet_fs, case_name + "/" + "eS")
f.write(normal_fs, case_name + "/" + "eN")
f.close()
