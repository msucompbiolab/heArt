import dolfin as df
import sys

sys.path.append("/home/ziaeirad/")
sys.path.append("/mnt/Research/")
import vtk
import vtk_py3 as vtk_py
import glob
import numpy as np
import csv
import math
import os
from vtk.util import numpy_support


def find_nearest(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return idx


def extract_PV(filename, BCL, ncycle, SimDet):
    reader = csv.reader(open(filename), delimiter=" ")
    tpt_array = []
    LVP_array = []
    LVV_array = []
    RVP_array = []
    RVV_array = []
    Qmv_array = []
    for row in reader:
        tpt_array.append(float(row[0]))
        LVP_array.append(float(row[1]))
        LVV_array.append(float(row[2]))
        if SimDet.get("isBiV") or SimDet.get("isFCH"):
            RVP_array.append(float(row[3]))
            RVV_array.append(float(row[4]))

        try:
            Qmv_array.append(float(row[6]))
        except IndexError:
            Qmv_array.append(0)

    tpt_array = np.array(tpt_array)
    LVP_array = np.array(LVP_array)
    LVV_array = np.array(LVV_array)
    RVP_array = np.array(RVP_array)
    RVV_array = np.array(RVV_array)
    Qmv_array = np.array(Qmv_array)

    ind = np.where(
        np.logical_and(tpt_array >= ncycle * BCL, tpt_array <= (ncycle + 1) * BCL)
    )

    if SimDet.get("isBiV") or SimDet.get("isFCH"):
        return (
            tpt_array[ind],
            LVP_array[ind],
            LVV_array[ind],
            RVP_array[ind],
            RVV_array[ind],
            Qmv_array[ind],
        )
    else:
        return tpt_array[ind], LVP_array[ind], LVV_array[ind], [], [], Qmv_array[ind]


def extract_Q(filename, BCL, ncycle, SimDet):

    isBiV = False
    isBiV = SimDet["isBiV"]

    reader = csv.reader(open(filename), delimiter=" ")

    tpt_array = []
    Qav_array = []
    Qmv_array = []
    Qsa_array = []
    Qsv_array = []
    Qpvv_array = []
    Qtv_array = []
    Qpa_array = []
    Qpv_array = []
    Qlvad_array = []

    for row in reader:
        tpt_array.append(float(row[0]))
        Qav_array.append(float(row[1]))
        Qmv_array.append(float(row[2]))
        Qsa_array.append(float(row[3]))
        Qsv_array.append(float(row[4]))

        if isBiV:
            Qpvv_array.append(float(row[5]))
            Qtv_array.append(float(row[6]))
            Qpa_array.append(float(row[7]))
            Qpv_array.append(float(row[8]))
            Qlvad_array.append(float(row[9]))
        else:
            Qlvad_array.append(float(row[5]))

    tpt_array = np.array(tpt_array)
    Qav_array = np.array(Qav_array)
    Qmv_array = np.array(Qmv_array)
    Qsa_array = np.array(Qsa_array)
    Qsv_array = np.array(Qsv_array)
    Qpvv_array = np.array(Qpvv_array)
    Qtv_array = np.array(Qtv_array)
    Qpa_array = np.array(Qpa_array)
    Qpv_array = np.array(Qpv_array)
    Qlvad_array = np.array(Qlvad_array)

    ind = np.where(
        np.logical_and(tpt_array >= ncycle * BCL, tpt_array <= (ncycle + 1) * BCL)
    )

    if isBiV:
        return (
            tpt_array[ind],
            Qav_array[ind],
            Qmv_array[ind],
            Qsa_array[ind],
            Qsv_array[ind],
            Qpvv_array[ind],
            Qtv_array[ind],
            Qpa_array[ind],
            Qpv_array[ind],
            Qlvad_array[ind],
        )
    else:
        return (
            tpt_array[ind],
            Qav_array[ind],
            Qmv_array[ind],
            Qsa_array[ind],
            Qsv_array[ind],
            None,
            None,
            None,
            None,
            Qlvad_array[ind],
        )


def extract_P(filename, BCL, ncycle, SimDet):

    isBiV = False
    isBiV = SimDet["isBiV"]

    reader = csv.reader(open(filename), delimiter=" ")
    tpt_array = []
    Psv_array = []
    PLV_array = []
    Psa_array = []
    PLA_array = []
    Ppv_array = []
    PRV_array = []
    Ppa_array = []
    PRA_array = []
    for row in reader:
        tpt_array.append(float(row[0]))
        Psv_array.append(float(row[1]))
        PLV_array.append(float(row[2]))
        Psa_array.append(float(row[3]))
        PLA_array.append(float(row[4]))

        if isBiV:
            Ppv_array.append(float(row[5]))
            PRV_array.append(float(row[6]))
            Ppa_array.append(float(row[7]))
            PRA_array.append(float(row[8]))

    tpt_array = np.array(tpt_array)
    Psv_array = np.array(Psv_array)
    PLV_array = np.array(PLV_array)
    Psa_array = np.array(Psa_array)
    PLA_array = np.array(PLA_array)
    Ppv_array = np.array(Ppv_array)
    PRV_array = np.array(PRV_array)
    Ppa_array = np.array(Ppa_array)
    PRA_array = np.array(PRA_array)

    ind = np.where(
        np.logical_and(tpt_array >= ncycle * BCL, tpt_array <= (ncycle + 1) * BCL)
    )

    if isBiV:
        return (
            tpt_array[ind],
            Psv_array[ind],
            PLV_array[ind],
            Psa_array[ind],
            PLA_array[ind],
            Ppv_array[ind],
            PRV_array[ind],
            Ppa_array[ind],
            PRA_array[ind],
        )
    else:
        return (
            tpt_array[ind],
            Psv_array[ind],
            PLV_array[ind],
            Psa_array[ind],
            PLA_array[ind],
            None,
            None,
            None,
            None,
        )


def extract_probe(filename, BCL, ncycle):
    reader = csv.reader(open(filename), delimiter=" ")
    tpt_array = []
    array_1 = []
    array_2 = []
    array_3 = []
    array_4 = []
    for row in reader:
        tpt_array.append(float(row[0]))
        array_1.append(float(row[1]))
        array_2.append(float(row[2]))
        array_3.append(float(row[3]))
        array_4.append(float(row[4]))

    tpt_array = np.array(tpt_array)
    array_1 = np.array(array_1)
    array_2 = np.array(array_2)
    array_3 = np.array(array_3)
    array_4 = np.array(array_4)

    ind = np.where(
        np.logical_and(tpt_array >= ncycle * BCL, tpt_array <= (ncycle + 1) * BCL)
    )

    return tpt_array[ind], [array_1[ind], array_2[ind], array_3[ind], array_4[ind]]


def extractESP(LVP, LVV):
    # Get LVV associated with isovolumic relaxation
    ind = np.where(np.logical_and(LVV >= min(LVV) - 0.05, LVV < min(LVV) + 0.05))

    # Get LVP associated with isovolumic relaxation
    isoLVP = LVP[ind]

    # Get ind associated with ES
    ESind = ind[0][isoLVP.argmax(axis=0)]

    return LVP[ESind], LVV[ESind]


def extractEDP(LVP, LVV):
    # Get LVV associated with isovolumic contraction
    ind = np.where(np.logical_and(LVV >= max(LVV) - 0.05, LVV < max(LVV) + 0.05))

    # Get LVP associated with isovolumic contraction
    isoLVP = LVP[ind]

    # Get ind associated with ED
    EDind = ind[0][isoLVP.argmin(axis=0)]

    return LVP[EDind], LVV[EDind]


def findESPVR(ESP, ESV):
    pfit = np.polyfit(ESV, ESP, 1)
    ESPVR = np.poly1d(pfit)

    return pfit, ESPVR


def readcsv(filename, ncolstart, ncolend, skip=1, delimiter=","):
    reader = csv.reader(open(filename), delimiter=delimiter)
    array = []
    nrow = 0
    for row in reader:
        if nrow >= skip:
            try:
                array.append([(float(row[p])) for p in range(ncolstart, ncolend + 1)])
            except ValueError:
                break
        nrow += 1

    return np.array(array)


def getradialposition(ugrid, endo, epi):
    points = ugrid.GetPoints()

    endo_ptlocator = vtk.vtkPointLocator()
    endo_ptlocator.SetDataSet(endo)
    endo_ptlocator.BuildLocator()

    epi_ptlocator = vtk.vtkPointLocator()
    epi_ptlocator.SetDataSet(epi)
    epi_ptlocator.BuildLocator()

    radialpos = vtk.vtkFloatArray()
    radialpos.SetName("radial position")
    radialpos.SetNumberOfComponents(1)

    endoids = []
    epiids = []

    for ptid in range(0, ugrid.GetNumberOfPoints()):
        point = np.array([points.GetPoint(ptid)[k] for k in range(0, 3)])
        closestendopt = np.array(
            endo.GetPoints().GetPoint(endo_ptlocator.FindClosestPoint(point))
        )
        closestepipt = np.array(
            epi.GetPoints().GetPoint(epi_ptlocator.FindClosestPoint(point))
        )
        wallthickness = vtk.vtkMath.Norm(closestepipt - closestendopt)
        dist = math.sqrt(vtk.vtkMath.Distance2BetweenPoints(point, closestendopt))
        radialpos.InsertNextValue(dist / wallthickness)

        if dist < 0.5:
            endoids.append(ptid)
        else:
            epiids.append(ptid)

    ugrid.GetPointData().AddArray(radialpos)

    return ugrid, endoids, epiids


def getpointclouds(directory, clipoffset=1e-5, npts=10000):
    mesh = df.Mesh()
    hdf = df.HDF5File(mesh.mpi_comm(), directory + "/" + "Data.h5", "r")
    hdf.read(mesh, "ME/mesh", False)
    ugrid = vtk_py.convertXMLMeshToUGrid(mesh)
    hdf.close()

    # Merge the subdomain into 1 unstructuredgrid
    merge = vtk.vtkExtractUnstructuredGrid()
    merge.SetInputData(ugrid)
    merge.MergingOn()
    merge.Update()
    ugrid = merge.GetOutput()

    # Generate point cloud
    cx = 0.5 * (ugrid.GetBounds()[0] + ugrid.GetBounds()[1])
    cy = 0.5 * (ugrid.GetBounds()[2] + ugrid.GetBounds()[3])
    cz = 0.5 * (ugrid.GetBounds()[4] + ugrid.GetBounds()[5])
    bds = max(
        [
            abs(ugrid.GetBounds()[0] - ugrid.GetBounds()[1]),
            abs(ugrid.GetBounds()[2] - ugrid.GetBounds()[3]),
            abs(ugrid.GetBounds()[4] - ugrid.GetBounds()[5]),
        ]
    )

    ptsource = vtk.vtkPointSource()
    ptsource.SetCenter([cx, cy, cz])
    ptsource.SetRadius(bds / 2.0 * 1.2)
    ptsource.SetNumberOfPoints(npts)
    ptsource.Update()

    selectEnclosed = vtk.vtkSelectEnclosedPoints()
    selectEnclosed.SetInputData(ptsource.GetOutput())
    selectEnclosed.SetSurfaceData(vtk_py.convertUGridtoPdata(ugrid))
    selectEnclosed.SetTolerance(1e-9)
    selectEnclosed.Update()

    thresh = vtk.vtkFloatArray()
    thresh.SetNumberOfComponents(1)
    thresh.InsertNextValue(0.5)
    thresh.InsertNextValue(2.0)
    thresh.SetName("SelectedPoints")

    selectionNode = vtk.vtkSelectionNode()
    selectionNode.SetFieldType(1)  # POINT
    selectionNode.SetContentType(7)  # INDICES
    selectionNode.SetSelectionList(thresh)  # INDICES
    selection = vtk.vtkSelection()
    selection.AddNode(selectionNode)

    extractSelection = vtk.vtkExtractSelection()
    extractSelection.SetInputData(0, selectEnclosed.GetOutput())
    extractSelection.SetInputData(1, selection)
    extractSelection.Update()

    points = extractSelection.GetOutput().GetPoints()

    # Get radial position
    probepointpdata = vtk.vtkPolyData()
    probepointpdata.SetPoints(points)

    pdata = vtk_py.convertUGridtoPdata(ugrid)
    ztop = ugrid.GetBounds()[5]
    clippedpdata = vtk_py.clipheart(pdata, [0, 0, ztop - clipoffset], [0, 0, 1], 1)
    epi, endo = vtk_py.splitDomainBetweenEndoAndEpi(clippedpdata)

    ugrid, endoids, epiids = getradialposition(ugrid, endo, epi)

    radpos = probe(ugrid, probepointpdata)
    radpos_array = [
        radpos.GetPointData().GetArray("radial position").GetValue(p)
        for p in range(0, radpos.GetNumberOfPoints())
    ]

    vtk_py.writeXMLPData(radpos, "radialposition.vtp")

    return points, radpos_array, radpos


def probe(ugrid, pdata):
    probeFilter = vtk.vtkProbeFilter()
    probeFilter.SetSourceData(ugrid)
    if vtk.vtkVersion.GetVTKMajorVersion <= 5:
        probeFilter.SetInput(pdata)
    else:
        probeFilter.SetInputData(pdata)

    probeFilter.Update()

    return probeFilter.GetOutput()


def probeqty(directory, fieldvariable, points, ind, index):
    mesh = df.Mesh()
    hdf = df.HDF5File(mesh.mpi_comm(), directory + "/" + "Data.h5", "r")
    hdf.read(mesh, "ME/mesh", False)
    ugrid = vtk_py.convertXMLMeshToUGrid(mesh)
    attr = hdf.attributes(fieldvariable)
    nsteps = attr["count"]

    var_space = df.FunctionSpace(mesh, "CG", 1)
    var = df.Function(var_space)

    # print nsteps
    # print ind[0][index]
    dataset = fieldvariable + "/vector_%d" % ind[0][index]
    hdf.read(var, dataset)
    var.rename("var", "var")

    var_vtk = numpy_support.numpy_to_vtk(
        num_array=var.vector().array()[df.vertex_to_dof_map(var_space)].ravel(),
        deep=True,
        array_type=vtk.VTK_FLOAT,
    )
    var_vtk.SetName("var")
    ugrid.GetPointData().AddArray(var_vtk)

    probepointpdata = vtk.vtkPolyData()
    probepointpdata.SetPoints(points)
    npts = points.GetNumberOfPoints()

    probvar = probe(ugrid, probepointpdata).GetPointData().GetArray("var")

    point_fieldvararray = [probvar.GetValue(p) for p in range(0, npts)]

    hdf.close()

    return np.array(point_fieldvararray)


# def probetimeseries(directory, filebasename, fieldvariable, points, isparallel, ind):
def probetimeseries(directory, fieldvariable, points, ind, elemtype, deg):
    assert (elemtype == "CG" and deg == 1) or (
        elemtype == "DG" and deg == 0
    ), "element type not supported"

    mesh = df.Mesh()
    hdf = df.HDF5File(mesh.mpi_comm(), directory + "/" + "Data.h5", "r")
    hdf.read(mesh, "ME/mesh", False)
    ugrid = vtk_py.convertXMLMeshToUGrid(mesh)

    attr = hdf.attributes(fieldvariable)
    nsteps = attr["count"]

    var_space = df.FunctionSpace(mesh, elemtype, deg)
    var = df.Function(var_space)

    probepointpdata = vtk.vtkPolyData()
    probepointpdata.SetPoints(points)
    npts = points.GetNumberOfPoints()

    point_fieldvararray = []
    cnt = 1
    for p in ind[0]:
        dataset = fieldvariable + "/vector_%d" % p
        hdf.read(var, dataset)
        var.rename("var", "var")
        var_vtk = numpy_support.numpy_to_vtk(
            num_array=var.vector().array()[:].ravel(),
            deep=True,
            array_type=vtk.VTK_FLOAT,
        )
        var_vtk.SetName("var")
        if elemtype == "CG":
            ugrid.GetPointData().AddArray(var_vtk)
        elif elemtype == "DG":
            ugrid.GetCellData().AddArray(var_vtk)

        probvar = probe(ugrid, probepointpdata).GetPointData().GetArray("var")

        point_fieldvararray.append([probvar.GetValue(p) for p in range(0, npts)])

        cnt += 1

    hdf.close()

    return np.array(point_fieldvararray)


def readtpt(filename):
    reader = csv.reader(open(filename), delimiter=" ")
    tpt_array = []
    for row in reader:
        tpt_array.append(int(float(row[0])))

    return np.array(tpt_array)


def extractvtk(
    directory,
    fieldvariable,
    elemtype,
    deg,
    outdirectory,
    name,
    ind=None,
    group="ME",
    iswrite=True,
):

    mesh = df.Mesh()
    hdf = df.HDF5File(mesh.mpi_comm(), directory + "/" + "Data.h5", "r")
    hdf.read(mesh, group + "/mesh", False)
    ugrid = vtk_py.convertXMLMeshToUGrid(mesh)
    attr = hdf.attributes(fieldvariable)
    nsteps = attr["count"]
    var_array = []

    if ind is None:
        ind = np.arange(0, nsteps)

    var_space = df.VectorFunctionSpace(mesh, elemtype, deg)
    var = df.Function(var_space)

    if iswrite:
        if not os.path.exists(outdirectory):
            os.mkdir(outdirectory)

    point_fieldvararray = []
    cnt = 1
    fstream = df.File(outdirectory + "/" + name + ".pvd")
    for p in ind:
        dataset = fieldvariable + "/vector_%d" % p
        hdf.read(var, dataset)
        var.rename("var", "var")
        var_array.append(var.copy(deepcopy=True))
        if iswrite:
            fstream << var

        cnt += 1

    hdf.close()

    return var_array
