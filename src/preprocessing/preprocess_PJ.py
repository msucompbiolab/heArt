import vtk_py3
import vtk
from dolfin import *
import dolfin
from dolfin import *
import numpy as np
import sys, shutil, pdb, math


def preprocess_PJ(PJ_meshfilename, PJ_meshoutfilename):

    # Convert line mesh in vtu to fenics mesh
    ugrid = vtk_py3.readXMLUGrid(PJ_meshfilename)
    ncells = ugrid.GetNumberOfCells()
    npts = ugrid.GetNumberOfPoints()
    
    mesh_pj = Mesh()
    editor = MeshEditor()
    editor.open(mesh_pj, "interval", 1, 3)  # top. and geom. dimension are both 2
    editor.init_vertices(npts)  # number of vertices
    editor.init_cells(ncells)  # number of cells
    
    # searching for terminal nodes of the purkinje network that interacts with the myocardium
    term_nodes = []
    term_nodes_coord = []
    #is_term_nodes_active = []
    
    for p in range(0, npts):
        pt = ugrid.GetPoints().GetPoint(p)
        editor.add_vertex(p, Point(pt[0], pt[1], pt[2]))
        celllist = vtk.vtkIdList()
        ugrid.GetPointCells(p, celllist)
        if celllist.GetNumberOfIds() == 1:
            term_nodes.append(p)
            term_nodes_coord.append([pt[0], pt[1], pt[2]])
            #is_term_nodes_active.append(0)
    
    cnt = 0
    for p in range(0, ncells):
        pts = vtk.vtkIdList()
        ugrid.GetCellPoints(p, pts)
        if pts.GetNumberOfIds() == 2:
            editor.add_cell(cnt, np.array([pts.GetId(0), pts.GetId(1)]))
            cnt = cnt + 1
    
    editor.close()


    # Write to HDF5
    hdf5_file = PJ_meshoutfilename + ".hdf5"#"vol_mesh1.hdf5"
    case_name = PJ_meshoutfilename
    f = HDF5File(MPI.comm_world, hdf5_file, "w")
    f.write(mesh_pj, case_name)
    f.close()

    np.savetxt(PJ_meshoutfilename+".csv", term_nodes_coord, delimiter=",")

    return 


