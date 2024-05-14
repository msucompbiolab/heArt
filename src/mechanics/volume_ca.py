from dolfin import *
import numpy as np

import sys

sys.path.append("/home/ziaeirad")
import os
import vtk_py3 as vtk_py

# import meshio as meshio
from ..utils.oops_objects_MRC2 import printout
from ..utils.oops_objects_MRC2 import printout

# import pyvista as pv
from mpi4py import MPI as pyMPI

# import tabulate


class MeshModifier(object):
    def __init__(self, mesh, mesh_t, edge_boundaries, facet_boundaries, u_, W_):
        self.mesh = mesh
        self.edge_boundaries = edge_boundaries
        self.facet_function = facet_boundaries
        self.comm_me = mesh.mpi_comm()

        self.sub_id = 8
        self.sel_label = 2

        self.lst_v2d = self.dofv_map(u_, W_)
        #        self.mesh_n, self.facet_function_n = self.modify_mesh(u_, W_)
        self.mesh_t = mesh_t
        self.mesh_n = mesh_t  # temp added

    def extract_mesht_connectivity(self, u_, W_):
        dof_map = u_.function_space().dofmap()
        l2g_dof_map = dof_map.tabulate_local_to_global_dofs()
        l2g_dof_ = l2g_dof_map.copy()
        unowned_dof = dof_map.local_to_global_unowned()

        for elem_ in unowned_dof:
            value_ = int(3 * elem_)
            if value_ in l2g_dof_:
                idx_ = np.where(l2g_dof_ == value_)
                l2g_dof_ = np.delete(l2g_dof_, [idx_[0], idx_[0] + 1, idx_[0] + 2])

        dim = self.mesh.geometry().dim()
        u_coord = u_.function_space().tabulate_dof_coordinates().reshape(-1, dim)

        V_n = VectorFunctionSpace(self.mesh, "CG", 1)  # change
        a_n = Function(V_n)
        a_n.rename("a_n", "a_n")

        #        b_n = u_coord[::3]
        #        b_n = b_n.ravel()
        #        a_n.vector().set_local(b_n)

        #        c_n = u_.vector().get_local()
        #        a_n.vector().set_local(c_n)
        #        as_backend_type(a_n.vector()).vec().ghostUpdate()

        #        pvd_real = File("real_.pvd")
        #        pvd_real << a_n

        W_n = VectorFunctionSpace(self.mesh_t, "CG", 2)
        w_n = Function(W_n)
        w_n.rename("w_n", "w_n")

        #        d_n = u_.vector().get_local()
        #        w_n.vector().set_local(d_n)
        #        as_backend_type(a_n.vector()).vec().ghostUpdate()

        w_coord = W_n.tabulate_dof_coordinates().reshape(-1, dim)
        #        samec = np.sum(np.all(u_coord[:, np.newaxis] == w_coord, axis=2))

        dof_map_w = W_n.dofmap()
        l2g_dof_map_w = dof_map_w.tabulate_local_to_global_dofs()
        l2g_dof_w = l2g_dof_map_w.copy()
        unowned_dof_w = dof_map_w.local_to_global_unowned()

        for elem_ in unowned_dof_w:
            value_ = int(3 * elem_)
            if value_ in l2g_dof_w:
                idx_ = np.where(l2g_dof_w == value_)
                l2g_dof_w = np.delete(l2g_dof_w, [idx_[0], idx_[0] + 1, idx_[0] + 2])

        return u_coord, w_coord, l2g_dof_, l2g_dof_w, w_n

    def u_map_cap(self, w_n, dof_shared, disp_vec, cap_dofs, uc_):
        b_n = w_n.vector().get_local()
        w_n.rename("w_n", "w_n")

        #       local_dofs = w_n.function_space().dofmap().dofs()

        a_n = dof_shared[:]

        b_n = disp_vec[a_n]

        #        u_cap = b_n[cap_dofs]

        b_n[cap_dofs] = uc_

        w_n.vector().set_local(b_n)
        as_backend_type(w_n.vector()).vec().ghostUpdate()

        #        pvd_mesht_wn = File("mesht_u_project.pvd")
        #        pvd_mesht_wn << w_n
        return w_n

    def meshn_Getu(self, u_):
        W_n = VectorFunctionSpace(self.mesh_t, "CG", 2)
        w_n = Function(W_n)
        w_n.rename("w_n", "w_n")

        #        c_n = u_.vector().get_local()
        d_n = w_n.vector().get_local()

        #        w_n.vector().set_local(c_n)
        #        as_backend_type(a_n.vector()).vec().ghostUpdate()

        dim = self.mesh_t.geometry().dim()
        u_coord = u_.function_space().tabulate_dof_coordinates().reshape(-1, dim)

        b_n = u_coord[::3]
        b_n = b_n.ravel()

        faz = len(d_n) - len(b_n)
        if faz > 0:
            b_n = np.concatenate((b_n, np.zeros(faz)))
        else:
            b_n = b_n[: -np.abs(faz)]

        print(("len of b = ", len(b_n), " len of d = ", len(d_n)))

        #        w_n.vector()[:] = 1
        w_n.vector().set_local(b_n)
        as_backend_type(w_n.vector()).vec().ghostUpdate()

        pvd_mesh_un = File("mesh_t_u.pvd")
        pvd_mesh_un << w_n

    def dofv_map(self, u_, W_):
        #        dim = W_.sub(0).dim()
        n = self.mesh.geometry().dim()

        u_coord = (
            u_.function_space().tabulate_dof_coordinates().reshape(-1, n)
        )  # dim --> -1
        v_coord = self.mesh.coordinates().reshape((-1, n))

        lst_d2v = []
        for i, u in enumerate(u_coord):
            for j, v in enumerate(v_coord):
                if np.array_equal(u, v):
                    lst_d2v.append((i, j))

        b = []
        for i in range(self.mesh.num_vertices()):
            for idx, tuple in enumerate(lst_d2v):
                if tuple[1] == i:
                    b.append((i, lst_d2v[idx][0]))

        lst_ = []  # change
        for i in range(len(b) / 3):
            lst_.append(b[3 * i][1] / 3)

        return lst_

    def real_map_mixed(self, u_, W_):
        #        dim = W_.sub(0).dim()
        #        n = self.mesh.geometry().dim()
        #
        #        u_coord = u_.function_space().tabulate_dof_coordinates().reshape(dim, n)
        #        v_coord = self.mesh.coordinates().reshape((-1, n))
        #
        #        lst_d2v = []
        #        for i, u in enumerate(u_coord):
        #            for j, v in enumerate(v_coord):
        #                if np.array_equal(u, v):
        #                    lst_d2v.append((i, j))
        #
        #        b = []
        #        for i in range(self.mesh.num_vertices()):
        #            for idx, tuple in enumerate(lst_d2v):
        #                if tuple[1] == i:
        #                    b.append((i, lst_d2v[idx][0]))
        #
        #        lst_v2d = []
        #        for i in range(len(b)/3):
        #            lst_v2d.append(b[3 * i][1] / 3)

        A_n = VectorFunctionSpace(self.mesh_n, "CG", 1, dim=3)
        a_n = Function(A_n)

        v2d_A_n = vertex_to_dof_map(A_n)
        v2d_A_n = v2d_A_n.reshape((-1, self.mesh_n.geometry().dim()))

        d2v_A_n = dof_to_vertex_map(A_n)
        d2v_A_n = d2v_A_n[range(0, len(d2v_A_n), 3)] / 3

        map_dofs_u = []

        for i in range(len(self.lst_v2d)):  # change
            map_dofs_u.append((v2d_A_n[i][0] / 3, self.lst_v2d[i]))  # change

        h = u_.compute_vertex_values(self.mesh)

        for i in range(self.mesh.num_vertices()):
            b_i = np.array(
                [
                    int(3 * map_dofs_u[i][0]),
                    int(3 * map_dofs_u[i][0] + 1),
                    int(3 * map_dofs_u[i][0] + 2),
                ]
            )
            c_i = np.array(
                [i, self.mesh.num_vertices() + i, 2 * self.mesh.num_vertices() + i]
            )

            a_n.vector()[b_i] = h[c_i]

        uc_x, uc_y, uc_z = self.u_centroid(u_, W_)

        for i in range(self.mesh.num_vertices(), self.mesh_n.num_vertices()):
            a_n.vector()[v2d_A_n[i][:]] = np.array([uc_x, uc_y, uc_z])

        return a_n

    #    def alt_extract_vol(self):
    #        d = 3
    #        I = Identity(d)
    #        X = SpatialCoordinate(self.mesh)
    #        N = FacetNormal(self.mesh)
    #        F = I + grad(a_n) #*

    def real_extract_vol(self, a_n, facet_t):
        d = a_n.ufl_domain().geometric_dimension()
        I = Identity(d)

        #        ds_x = Measure("ds", domain = self.mesh_n, subdomain_data = self.facet_function_n, subdomain_id = self.sub_id)
        ds_x = Measure(
            "ds", domain=self.mesh_t, subdomain_data=facet_t, subdomain_id=self.sub_id
        )

        #        vol = assemble(1.0 * ds_x)

        X = SpatialCoordinate(self.mesh_n)
        N = FacetNormal(self.mesh_n)

        F = I + grad(a_n)

        vol_form = (
            -Constant(1.0 / 3.0) * inner(det(F) * dot(inv(F).T, N), X + a_n) * ds_x
        )

        return assemble(vol_form, form_compiler_parameters={"representation": "uflacs"})

    #    def found_edges(self):
    #        id_sel = 1
    #        subset_iter = SubsetIterator(self.edge_boundaries, id_sel)
    #        for f in subset_iter:
    #            for v in vertices(f):
    #                x, y, z = v.point().x(), v.point().y(), v.point().z()
    #
    #        id_sel_ = 1
    #        iter_ = SubsetIterator(self.facet_function, id_sel_)
    #
    #        facet_function_ = FacetFunction("size_t", self.mesh)
    #        for g in iter_:
    #            for facet in facets(g):
    #                vertex_ = facet.entities(0)
    #                Z_coord = sum(self.mesh.coordinates()[vertex][2] for vertex in vertex_) / 3.0
    #                if abs(Z_coord) > 8.0:
    #                    facet_function_[facet] = 3
    #
    #        return facet_function_

    def u_centroid(self, u_, W_):
        #        lst_v2d = self.dofv_map(u_, W_)

        subset_iter = SubsetIterator(self.edge_boundaries, self.sel_label)
        #        num_edge_vertices = 0; uc_x = 0.0; uc_y = 0.0; uc_z = 0.0

        h_ = u_.compute_vertex_values(self.mesh)

        #        vertex_coor = []
        num_vertices = 0
        vertex_u_values = []
        for f in subset_iter:
            for v in vertices(f):
                v_idx = v.index()
                vertex_u_values.append(
                    (
                        h_[v_idx],
                        h_[self.mesh.num_vertices() + v_idx],
                        h_[2 * self.mesh.num_vertices() + v_idx],
                    )
                )
                #                vertex_coor.append((v.point().x(), v.point().y(), v.point().z()))
                num_vertices += 1

        uc_x = 0
        uc_y = 0
        uc_z = 0
        for u_x, u_y, u_z in vertex_u_values:
            uc_x += u_x
            uc_y += u_y
            uc_z += u_z

        sum_vertex_values = np.array([uc_x, uc_y, uc_z])

        #        for f in subset_iter:
        #            for v in vertices(f):
        #                v_idx = v.index()
        #
        #                dof_ = self.lst_v2d[v_idx]
        #
        #                x, y, z = v.point().x(), v.point().y(), v.point().z()
        #
        #                uc_x += x + u_.vector()[3 * dof_]
        #                uc_y += y + u_.vector()[3 * dof_ + 1]
        #                uc_z += z + u_.vector()[3 * dof_ + 2]
        #                num_edge_vertices += 1

        #        if num_edge_vertices > 0:
        #            uc_x /= num_edge_vertices
        #            uc_y /= num_edge_vertices
        #            uc_z /= num_edge_vertices
        #        else:
        #            raise ValueError("No Edge Found")

        #        return uc_x, uc_y, uc_z
        return sum_vertex_values, num_vertices

    def modify_mesh(self, u_, W_):
        subset_iter = SubsetIterator(self.edge_boundaries, self.sel_label)

        new_vertices = []
        new_triangles = []

        #        centroid_x, centroid_y, centroid_z = self.u_centroid(u_, W_) #vahid comment alternative is just to set zeros
        centroid_x, centroid_y, centroid_z = 0.0, 0.0, 0.0

        new_vertices.append([centroid_x, centroid_y, centroid_z])
        offset = 1.0
        new_vertices.append([centroid_x, centroid_y, centroid_z + offset])

        prev_num_vertices = self.mesh.num_vertices()
        prev_num_cells = self.mesh.num_cells()

        edges = [
            f.entities(0) for f in SubsetIterator(self.edge_boundaries, self.sel_label)
        ]

        tst_new_triangles = []
        for edge in edges:
            tst_triangle_vertices = [prev_num_vertices]
            for vertex_index in edge:
                tst_triangle_vertices.append(vertex_index)
            tst_new_triangles.append(tst_triangle_vertices)

        for edge in edges:
            triangle_vertices = [prev_num_vertices]
            for vertex_index in edge:
                triangle_vertices.append(vertex_index)
            triangle_vertices.append(prev_num_vertices + 1)
            new_triangles.append(triangle_vertices)

        n_cells = len(new_triangles) + prev_num_cells
        n_vertex = len(new_vertices) + prev_num_vertices

        mesh_n = Mesh(self.comm_me)
        editor_n = MeshEditor()

        editor_n.open(mesh_n, 3, 3)
        editor_n.init_vertices(n_vertex)

        for vertex_index in range(prev_num_vertices):
            vertex = self.mesh.coordinates()[vertex_index]
            editor_n.add_vertex(vertex_index, vertex)
        editor_n.add_vertex(
            prev_num_vertices, np.array([centroid_x, centroid_y, 0.0], dtype=np.double)
        )

        editor_n.add_vertex(
            prev_num_vertices + 1,
            np.array([centroid_x, centroid_y, offset], dtype=np.double),
        )

        editor_n.init_cells(n_cells)
        for i in range(prev_num_cells):
            cell = Cell(self.mesh, i)
            vertex_indices = [v.index() for v in vertices(cell)]
            editor_n.add_cell(i, np.array(vertex_indices, dtype=np.uintp))

        for i, triangle in enumerate(new_triangles):
            editor_n.add_cell(prev_num_cells + i, np.array(triangle, dtype=np.uintp))

        editor_n.close()

        facet_function_n = FacetFunction("size_t", mesh_n)
        for i, triangle_vertices in enumerate(tst_new_triangles):
            tetrahedron = Cell(mesh_n, prev_num_cells + i)
            for facet in facets(tetrahedron):
                facet_vertices = [vertex.index() for vertex in vertices(facet)]
                if set(facet_vertices) == set(triangle_vertices):
                    facet_function_n[facet] = self.sub_id

        #       return mesh_n, facet_function_n
        return mesh_n, facet_function_n
