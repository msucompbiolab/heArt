import sys
sys.path.append("/home/ziaeirad")
import os
import vtk_py as vtk_py
import dolfin as dolfin
isepiflip = False
isendoflip = True


casename = "ellipsoidal"
LVangle = 60

meshfilename = casename + ".vtk"

cmd = "gmsh"+" -3 ellipsoidal.geo -o " + meshfilename
os.system(cmd)

ugrid = vtk_py.readUGrid(meshfilename)

xmlgrid = vtk_py.convertUGridToXMLMesh(ugrid)

xmlgrid, xmlfacet, xmledges = vtk_py.extractFeNiCsBiVFacet(ugrid, geometry="LV")

VQuadelem = dolfin.VectorElement("Quadrature",
                              xmlgrid.ufl_cell(),
                              degree=4,
                              quad_scheme="default")
VQuadelem._quad_scheme = 'default'

fiberFS = dolfin.FunctionSpace(xmlgrid, VQuadelem)

ef, es, en, eC, eL, eR = vtk_py.addLVfiber(xmlgrid, fiberFS, casename, LVangle, -LVangle, [] , isepiflip, isendoflip)

f = dolfin.HDF5File(xmlgrid.mpi_comm(), casename+".hdf5", 'w')
f.write(xmlgrid, casename)
f.close()

f = dolfin.HDF5File(xmlgrid.mpi_comm(), casename+".hdf5", 'a')
f.write(xmlfacet, casename+"/"+"facetboundaries")
f.write(xmledges, casename+"/"+"edgeboundaries")
f.write(ef, casename+"/"+"eF")
f.write(es, casename+"/"+"eS")
f.write(en, casename+"/"+"eN")
f.write(eC, casename+"/"+"eC")
f.write(eL, casename+"/"+"eL")
f.write(eR, casename+"/"+"eR")
f.close()

dolfin.File(casename+"facetboundaries.pvd")  << xmlfacet
dolfin.File(casename+"Edges.pvd")  << xmledges
