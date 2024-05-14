########################################################################

import vtk
import numpy as np
from vtk.util import numpy_support

########################################################################

def readAbaqusFacet(mesh_filename,
                   elem_types="all",
                   verbose=True):

    if (verbose): print "*** readAbaqusMesh ***"

    points = vtk.vtkPoints()

    markid = 0
    cell_array = vtk.vtkCellArray()
    cell_data = []

    mesh_file = open(mesh_filename, "r")

    context = ""
    for line in mesh_file:
        if (line[-1:] == "\n"): line = line[:-1]
        #if (verbose): print "line =", line

        if line.startswith("**"): continue

        if (context == "reading nodes"):
            if line.startswith("*"):
                context = ""
            else:
                splitted_line = line.split(",")
                points.InsertNextPoint([float(coord) for coord in splitted_line[1:4]])

        if (context == "reading elems"):
            if line.startswith("*"):
                context = ""
            else:
                splitted_line = line.split(",")
                assert (len(splitted_line) == 1+cell_nb_points), "Wrong number of elements in line. Aborting."
                for num_point in range(cell_nb_points): cell.GetPointIds().SetId(num_point, int(splitted_line[1+num_point])-1)
                cell_array.InsertNextCell(cell)
                cell_data.append(markid)

        if line.upper().startswith("*NODE"):
            context = "reading nodes"
        if line.upper().startswith("*ELEMENT"):
            if ("TYPE=S3R" in line.upper()) and (("quad" in elem_types) or ("all" in elem_types)):
                context = "reading elems"
                cell_vtk_type = vtk.VTK_TRIANGLE
                cell_nb_points = 3
                cell = vtk.vtkTriangle()
                markid += 1
            else:
                print "Warning: element type not taken into account."

    mesh_file.close()

    if (verbose): print "Creating PData..."

    pdata = vtk.vtkPolyData()
    pdata.SetPoints(points)
    pdata.SetPolys(cell_array)
    vtk_cell_data = numpy_support.numpy_to_vtk(num_array = np.array(cell_data), deep=True, array_type=vtk.VTK_INT)
    vtk_cell_data.SetName("facet id") 
    pdata.GetCellData().AddArray(vtk_cell_data)

    if (verbose): print "nb_cells = " + str(pdata.GetNumberOfCells())



    return pdata


