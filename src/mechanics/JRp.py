import numpy as np


def volume_ucentroid(vertex_u_values, num_vertices, comm_, rank_):
    #
    #####u centroid
    #    vertex_u_values, num_vertices = MeshModifier_.u_centroid(MEmodel_.GetDisplacement(), MEmodel_.W)
    uc_allvec = comm_.gather(vertex_u_values, root=0)
    num_vertices_allvec = comm_.gather(num_vertices, root=0)

    if rank_ == 0:
        uc_x = 0.0
        uc_y = 0.0
        uc_z = 0.0
        for ux, uy, uz in uc_allvec:
            uc_x += ux
            uc_y += uy
            uc_z += uz
        uc_ = np.array([uc_x, uc_y, uc_z])

        sum_num_vertices = sum(num_vertices_allvec)
        uc_ /= sum_num_vertices

    else:
        uc_ = None
        sum_num_vertices = None

    uc_ = comm_.bcast(uc_, root=0)
    sum_num_vertices = comm_.bcast(sum_num_vertices, root=0)

    return uc_, sum_num_vertices


#### project u to mesh with cap
def volume_map_cap(u_, comm_, rank_):
    u_local = u_.vector().get_local()
    u_global = comm_.gather(u_local, root=0)
    u_global = comm_.bcast(u_global, root=0)

    lst_u_local = []
    if rank_ == 0:
        for i in range(len(u_global)):
            lst_u_local.append(u_global[i])
        u_allvec = np.hstack(lst_u_local)
    else:
        u_allvec = None
    u_allvec = comm_.bcast(u_allvec, root=0)

    #    w_n = MeshModifier_.u_map_cap(mesht_u, local_map_dof, u_allvec, lst_cap_dofs, uc_)
    #    real_vol = MeshModifier_.real_extract_vol(w_n, facet_t)
    #
    return u_allvec
