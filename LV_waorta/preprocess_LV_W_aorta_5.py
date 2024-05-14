from dolfin import *
import vtk
import sys
sys.path.append('../../')
from vtk_py import *
from vtk.util import numpy_support
import warnings

warnings.filterwarnings('error')


def appendpdata(pdata1, pdata2):

    apdata = vtk.vtkAppendPolyData()
    apdata.AddInputData(pdata1)
    apdata.AddInputData(pdata2)
    apdata.Update()

    return apdata.GetOutput()

def preparedolfinmesh():

    # Read Mesh
    filename = "LV_W_aorta2.vtk"
    endofilename = "LV_W_aorta_endo.vtp"
    epifilename = "LV_W_aorta_epi.vtp"
    LV_regionfilename = "LV_region.vtu"
    aorta_regionfilename = "aorta_region.vtu"
    AV_regionfilename = "AV_region.vtu"
    
    ugrid = vtk_py.readUGrid(filename)
    endo_pdata = vtk_py.readXMLPData(endofilename)
    epi_pdata = vtk_py.readXMLPData(epifilename)
    
    LV_region = vtk_py.readXMLUGrid(LV_regionfilename)
    LV_region_mesh = vtk_py.convertUGridToXMLMesh(LV_region)
    LV_region_bbtree = BoundingBoxTree()
    LV_region_bbtree.build(LV_region_mesh)
    
    aorta_region = vtk_py.readXMLUGrid(aorta_regionfilename)
    aorta_region_mesh = vtk_py.convertUGridToXMLMesh(aorta_region)
    aorta_region_bbtree = BoundingBoxTree()
    aorta_region_bbtree.build(aorta_region_mesh)

    AV_region = vtk_py.readXMLUGrid(AV_regionfilename)
    AV_region_mesh = vtk_py.convertUGridToXMLMesh(AV_region)
    AV_region_bbtree = BoundingBoxTree()
    AV_region_bbtree.build(AV_region_mesh)
    
    endo_cellids = numpy_support.vtk_to_numpy(endo_pdata.GetCellData().GetArray("GroupIds"))
    epi_cellids = numpy_support.vtk_to_numpy(epi_pdata.GetCellData().GetArray("GroupIds"))
    
    endo_cellids = endo_cellids+1
    epi_cellids += max(endo_cellids)+1
    
    endo_cellids_vtk = numpy_support.numpy_to_vtk(num_array = endo_cellids, deep=True, array_type=vtk.VTK_INT)
    endo_cellids_vtk.SetName("GroupIds")
    epi_cellids_vtk = numpy_support.numpy_to_vtk(num_array = epi_cellids, deep=True, array_type=vtk.VTK_INT)
    epi_cellids_vtk.SetName("GroupIds")
    
    endo_pdata.GetCellData().RemoveArray("GroupIds")
    endo_pdata.GetCellData().AddArray(endo_cellids_vtk)
    epi_pdata.GetCellData().RemoveArray("GroupIds")
    epi_pdata.GetCellData().AddArray(epi_cellids_vtk)
    
    pdata = appendpdata(endo_pdata, epi_pdata)
    
    # Build point locator about the cell center
    pdata_cellcenter = getCellCenters(pdata)
    pdata_pointlocator = vtk.vtkPointLocator()
    pdata_pointlocator.SetDataSet(pdata_cellcenter)
    pdata_pointlocator.BuildLocator()
    
    # Convert to dolfin mesh
    dolfin_mesh = vtk_py.convertUGridToXMLMesh(ugrid)
    
    # Mark region in mesh
    dolfin_region = MeshFunction('size_t', dolfin_mesh, dolfin_mesh.topology().dim(), 0)
    mark_region = np.zeros(len(dolfin_region.array()))
    cell_centers = vtk.vtkPoints()
    for cell in cells(dolfin_mesh):
        cell_centers.InsertNextPoint(cell.midpoint().array()[:])
        iscollided_LV = LV_region_bbtree.compute_entity_collisions(Point(cell.midpoint().array()[:]))
        iscollided_aorta = aorta_region_bbtree.compute_entity_collisions(Point(cell.midpoint().array()[:]))
        iscollided_AV = AV_region_bbtree.compute_entity_collisions(Point(cell.midpoint().array()[:]))
        if(len(iscollided_LV) == 1):
            mark_region[cell.index()] = 1.0
    
        if(len(iscollided_aorta) == 1):
            mark_region[cell.index()] = 2.0

        if(len(iscollided_AV) == 1):
            mark_region[cell.index()] = 3.0
    
    dolfin_region.array()[:] = mark_region
    
    # Mark facet of dolfin mesh
    dolfin_facet = MeshFunction('size_t', dolfin_mesh, dolfin_mesh.topology().dim()-1, 0)
    DomainBoundary().mark(dolfin_facet, 1) # Mark all the exterior facets as 1
    mark_facets = np.zeros(len(dolfin_facet.array()))
    
    for facet in dolfin.SubsetIterator(dolfin_facet, 1):
        for cell in cells(facet):
            cx = 0; cy = 0; cz = 0
            for vertex in vertices(facet):
                cx += vertex.point().array()[0]/3.0
                cy += vertex.point().array()[1]/3.0
                cz += vertex.point().array()[2]/3.0
            pdata_cellid = pdata_pointlocator.FindClosestPoint(cx, cy, cz)
            facet_id = pdata_cellcenter.GetPointData().GetArray("GroupIds").GetValue(pdata_cellid)
            mark_facets[facet.index()] = int(facet_id)
    
    dolfin_facet.array()[:] = mark_facets

    return dolfin_mesh, dolfin_facet, dolfin_region


def solveLaplaceEquation(df_mesh, df_facet, bc0_ids, bc1_ids):

    V = FunctionSpace(df_mesh, FiniteElement("Lagrange", df_mesh.ufl_cell(), 1))
    Vvec = FunctionSpace(df_mesh, VectorElement("DG", df_mesh.ufl_cell(), 0))

    bc_endo =[DirichletBC(V, Constant(0.0), df_facet, id_) for id_ in bc0_ids] 
    bc_epi =[DirichletBC(V, Constant(1.0), df_facet, id_) for id_ in bc1_ids] 

    # Define variational problem
    u = TrialFunction(V)
    v = TestFunction(V)
    f = Constant(0)
    a = inner(nabla_grad(u), nabla_grad(v))*dx
    L = f*v*dx
    
    # Compute solution
    u = Function(V)
    solve(a == L, u, bc_endo + bc_epi)#, solver_parameters={"linear_solver": "petsc"})

    #grad_u = project(grad(u)/sqrt(dot(grad(u),grad(u))), Vvec)#, solver_type="petsc")
    grad_u = project(grad(u), Vvec)#, solver_type="petsc")

    return u, grad_u

def add_apex_and_base(df_mesh, df_facet, apx_reg, bas_reg):

    # Mark Apex region
    new_facet_array = df_facet.array()[:]
    apx_id = max(df_facet.array()[:]) + 1
    bas_id = max(df_facet.array()[:]) + 2

    new_df_facet = MeshFunction('size_t', df_mesh, df_mesh.topology().dim()-1, 0)
    for facet in dolfin.SubsetIterator(df_facet, 4):
        for cell in cells(facet):
            cx = 0; cy = 0; cz = 0
            for vertex in vertices(facet):
                cx += vertex.point().array()[0]/3.0
                cy += vertex.point().array()[1]/3.0
                cz += vertex.point().array()[2]/3.0

            if((cx - apx_reg[0])**2 + (cy - apx_reg[1])**2 + (cz - apx_reg[2])**2 < apx_reg[3]**2):
                new_facet_array[facet.index()] = apx_id

    for facet in dolfin.SubsetIterator(df_facet, 5):
        for cell in cells(facet):
            cx = 0; cy = 0; cz = 0
            for vertex in vertices(facet):
                cx += vertex.point().array()[0]/3.0
                cy += vertex.point().array()[1]/3.0
                cz += vertex.point().array()[2]/3.0

            if((cx - bas_reg[0])**2 + (cy - bas_reg[1])**2 + (cz - bas_reg[2])**2 < bas_reg[3]**2):
                new_facet_array[facet.index()] = bas_id
 
    for facet in dolfin.SubsetIterator(df_facet, 7):
        for cell in cells(facet):
            cx = 0; cy = 0; cz = 0
            for vertex in vertices(facet):
                cx += vertex.point().array()[0]/3.0
                cy += vertex.point().array()[1]/3.0
                cz += vertex.point().array()[2]/3.0

            if((cx - bas_reg[0])**2 + (cy - bas_reg[1])**2 + (cz - bas_reg[2])**2 < bas_reg[3]**2):
                new_facet_array[facet.index()] = bas_id
     
    new_df_facet.array()[:] = new_facet_array
    File("new_df_facet.pvd") << new_df_facet

    return new_df_facet


def addfiber(df_mesh, df_facet, df_region, apex_region, base_region, endo_angle, epi_angle):

    endo_facet_ids = [1,2,5]
    epi_facet_ids = [4,7,8]

    u1, eR = solveLaplaceEquation(df_mesh, df_facet, endo_facet_ids, epi_facet_ids)
    new_df_facet = add_apex_and_base(df_mesh, df_facet, apx_reg, bas_reg)
    u2, eL = solveLaplaceEquation(df_mesh, new_df_facet, [9], [10])
    eC = project(cross(eL, eR), FunctionSpace(df_mesh, VectorElement("DG", df_mesh.ufl_cell(), 0)))

    u1_cell = project(u1, FunctionSpace(df_mesh, FiniteElement("DG", df_mesh.ufl_cell(), 0)))

    eF = Function(FunctionSpace(df_mesh, VectorElement("DG", df_mesh.ufl_cell(), 0)))
    eS = Function(FunctionSpace(df_mesh, VectorElement("DG", df_mesh.ufl_cell(), 0)))
    eN = Function(FunctionSpace(df_mesh, VectorElement("DG", df_mesh.ufl_cell(), 0)))
    trans_ang = Function(FunctionSpace(df_mesh, FiniteElement("DG", df_mesh.ufl_cell(), 0)))

    File("Transmural_dist.pvd") << u1_cell

    eF_vec_array = np.zeros(len(eC.vector().array()[:]))
    eN_vec_array = np.zeros(len(eC.vector().array()[:]))
    eS_vec_array = np.zeros(len(eC.vector().array()[:]))
    eC_vec_array = np.zeros(len(eC.vector().array()[:]))
    eL_vec_array = np.zeros(len(eC.vector().array()[:]))
    eR_vec_array = np.zeros(len(eC.vector().array()[:]))
    trans_ang_vec_array =np.zeros(len(u1_cell.vector().array()[:]))

    # Loop through all elements to assign eF, eS, eN
    for cell in cells(df_mesh):

        eC_vec = np.array(eC.vector().array()[3*cell.index():3*cell.index()+3])
        eR_vec = np.array(eR.vector().array()[3*cell.index():3*cell.index()+3])
        eL_vec = np.array(eL.vector().array()[3*cell.index():3*cell.index()+3])

        # Normalize vector
        with np.errstate(divide='raise'):
            try:
                eC_vec = eC_vec/np.linalg.norm(eC_vec)
                eC_vec_array[3*cell.index():3*cell.index()+3] = eC_vec
            except RuntimeWarning:
                eC_vec_array[3*cell.index():3*cell.index()+3] = [1.0, 0.0, 0.0]

            try:
                eR_vec = eR_vec/np.linalg.norm(eR_vec)
                eR_vec_array[3*cell.index():3*cell.index()+3] = eR_vec
            except RuntimeWarning:
                eR_vec_array[3*cell.index():3*cell.index()+3] = [0.0, 1.0, 0.0]

            try:
                eL_vec = eL_vec/np.linalg.norm(eL_vec)
                eL_vec_array[3*cell.index():3*cell.index()+3] = eL_vec
            except RuntimeWarning:
                eL_vec_array[3*cell.index():3*cell.index()+3] = [0.0, 0.0, 1.0]


        trans_dist = u1_cell.vector().array()[cell.index()]

        trans_ang_val = endo_angle*(1.0 - trans_dist) + epi_angle*trans_dist
        trans_ang_vec_array[cell.index()] = trans_ang_val

        # Calculate eF based on rotation of eC about eR
        eF_vec = eC_vec*np.cos(trans_ang_val/180.0*np.pi) + \
                 np.cross(eR_vec, eC_vec)*np.sin(trans_ang_val/180.0*np.pi) + \
                 eR_vec*(np.dot(eR_vec, eC_vec))*(1.0 - np.cos(trans_ang_val/180.0*np.pi))

        eN_vec = np.cross(eR_vec, eF_vec)

        with np.errstate(divide='raise'):
            try:
                eF_vec_array[3*cell.index():3*cell.index()+3] = eF_vec/np.linalg.norm(eF_vec)
            except RuntimeWarning:
                eF_vec_array[3*cell.index():3*cell.index()+3] = [1.0, 0.0, 0.0]

            try:
                eN_vec_array[3*cell.index():3*cell.index()+3] = eN_vec/np.linalg.norm(eN_vec)
            except RuntimeWarning:
                eN_vec_array[3*cell.index():3*cell.index()+3] = [0.0, 1.0, 0.0]

            try:
                eS_vec_array[3*cell.index():3*cell.index()+3] = eR_vec/np.linalg.norm(eR_vec)
            except RuntimeWarning:
                eS_vec_array[3*cell.index():3*cell.index()+3] = [0.0, 0.0, 1.0]


    print("Is Nan in eF:", np.isnan(eF_vec_array).any(), np.argwhere(np.isnan(eF_vec_array)))
    print("Is Nan in eN:", np.isnan(eN_vec_array).any(), np.argwhere(np.isnan(eN_vec_array)))
    print("Is Nan in eS:", np.isnan(eS_vec_array).any(), np.argwhere(np.isnan(eS_vec_array)))

    eF.vector()[:] = eF_vec_array
    eN.vector()[:] = eN_vec_array
    eS.vector()[:] = eS_vec_array
    trans_ang.vector()[:] = trans_ang_vec_array

    File("u1.pvd") << u1
    File("u2.pvd") << u2
    File("eR.pvd") << eR
    File("eL.pvd") << eL
    File("eC.pvd") << eC
    File("eF.pvd") << eF
    File("eN.pvd") << eN
    File("eS.pvd") << eS
    File("trans_angle.pvd") << trans_ang

    return eR, eL, eC, eF, eN, eS


    

dolfin_mesh, dolfin_facet, dolfin_region = preparedolfinmesh()
File("dolfin_facet.pvd") << dolfin_facet
File("dolfin_region.pvd") << dolfin_region

apx_reg = [83.9506, 45.4243, -254.97, 12.8582]
bas_reg = [-30.655, 103.872, -136.44, 31.4460]
eR, eL, eC, eF, eN, eS = addfiber(dolfin_mesh, dolfin_facet, dolfin_region, apx_reg, bas_reg, 60, -60)


# Write to HDF5
hdf5_file = 'LV_W_aorta.hdf5'
case_name = 'LV_W_aorta'
f = HDF5File(mpi_comm_world(), hdf5_file, 'w')
f.write(dolfin_mesh, case_name)
f.close()

f = HDF5File(mpi_comm_world(), hdf5_file, 'a')
f.write(dolfin_facet, case_name+'/'+'facetboundaries')
f.write(dolfin_region, case_name+'/'+'materialregion')
f.write(eR, case_name+'/'+'eR')
f.write(eL, case_name+'/'+'eL')
f.write(eC, case_name+'/'+'eC')
f.write(eF, case_name+'/'+'eF')
f.write(eN, case_name+'/'+'eN')
f.write(eS, case_name+'/'+'eS')
f.close()




