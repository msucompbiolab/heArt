# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
import sys, ast
import meshio
import numpy as np
from fractal_tree import generate_fractal_tree, FractalTreeParameters, Mesh
from fractal_tree.branch import Nodes
import pdb
import matplotlib as mpl
import matplotlib.pyplot as plt
import pdb, math



msh = meshio.read(sys.argv[1])
mesh = Mesh(verts=msh.points, connectivity=msh.cells[0].data)

param_base = FractalTreeParameters(
    filename=sys.argv[3],
    initial_direction=np.array(ast.literal_eval(sys.argv[4])),
    init_length=float(sys.argv[5]), # trunc length
    init_node=ast.literal_eval(sys.argv[2]),
    N_it=int(sys.argv[6]),
    mode=int(sys.argv[7]),
    fascicles_angles=[float(sys.argv[8]),-float(sys.argv[8])], #ast.literal_eval(sys.argv[8]),
    # branch_angle=float(sys.argv[9]),
    length=float(sys.argv[10]),
    l_segment=float(sys.argv[11]),
    fascicles_length=[float(sys.argv[12]),float(sys.argv[12])],
    # fasciles_length_base =float(sys.argv[12]), #branch length
    )
branches_base, nodes_base, lines_base, end_nodes_base, branches_to_grow_base, last_branch_base = generate_fractal_tree(mesh, param_base)

param_mid1 = FractalTreeParameters(
    filename="endo_pj_mid1",
    mode=2,
    N_it=15,
    length=0.5,
    l_segment=0.1,
    fascicles_length = [1.5, 1.5],
    branches=branches_base,
    nodes=nodes_base,
    lines=lines_base,
    end_nodes=end_nodes_base,
    last_branch=last_branch_base,
    branches_to_grow=branches_to_grow_base,
    repulsitivity=1.0,
    )
branches_m1, nodes_m1, lines_m1, end_nodes_m1, branches_to_grow_m1, last_branch_m1 = generate_fractal_tree(mesh, param_mid1)

param_mid2 = FractalTreeParameters(
    filename="endo_pj_mid2",
    mode=2,
    N_it=8,
    length=0.8,
    l_segment=0.4,
    fascicles_length = [0.5, 0.5],    
    branches=branches_m1,
    nodes=nodes_m1,
    lines=lines_m1,
    end_nodes=end_nodes_m1,
    last_branch=last_branch_m1,
    branches_to_grow=branches_to_grow_m1,
    repulsitivity=1.0,
    )
branches_m2, nodes_m2, lines_m2, end_nodes_m2, branches_to_grow_m2, last_branch_m2= generate_fractal_tree(mesh, param_mid2)




class generate_AHA:
    """ Class that generates AHA regions
    Args:
    Attributes:
    """
    def compute_aha_segments(mesh, his_point, aha_type = 'points'):
            his = his_point
            his_circ = np.rad2deg(np.arctan2(his[1], his[0]))

            rot_angle = -180 - (his_circ)
            xyz = rotate_z(mesh.points, rot_angle)

            mesh.point_data['rot_xyz'] = xyz

            apex_node = xyz[np.argmin(xyz[:,2])]
            mv_node = [0, 0, 0]

            # longitudinal vector
            long_vec = apex_node - mv_node
            long_length = np.linalg.norm(long_vec)
            long_vec = long_vec/long_length

            long = (xyz[:,2]/long_length) + 1.0

            if aha_type == 'points':
                xyz_aha = xyz
            elif aha_type == 'elems':
                ien = mesh.cells[0].data
                xyz_aha = np.mean(xyz[ien], axis=1) # centroid of element
                long = np.mean(long[ien], axis=1)

            mesh.cell_data['cell_centroid'] = [xyz_aha]

            # Long axis division
            base_marker = (long>=2/3)*(long<=1)
            mid_marker = (long>=1/3)*(long<2/3)
            apex_marker = (long>0)*(long<1/3)
            apex_apex_marker = (long<=0.05)

            # Compute circ for easiness, general centroid calculation
            base_centroid = np.mean(xyz_aha[base_marker], axis=0)
            mid_centroid = np.mean(xyz_aha[mid_marker], axis=0)
            apex_centroid = np.mean(xyz_aha[apex_marker], axis=0)

            base_xyz = xyz_aha - base_centroid
            mid_xyz = xyz_aha - mid_centroid
            apex_xyz = xyz_aha - apex_centroid

            base_circ = np.rad2deg(np.arctan2(base_xyz[:,1], base_xyz[:,0]))
            mid_circ = np.rad2deg(np.arctan2(mid_xyz[:,1], mid_xyz[:,0]))
            apex_circ = np.rad2deg(np.arctan2(apex_xyz[:,1], apex_xyz[:,0]))

            # Define AHA
            aha_region = np.zeros(len(xyz_aha))
            aha_region[(base_marker)*(base_circ>60)*(base_circ<=120)] = 1
            aha_region[(base_marker)*(base_circ>120)*(base_circ<=180)] = 2
            aha_region[(base_marker)*(base_circ>=-180)*(base_circ<=-120)] = 3
            aha_region[(base_marker)*(base_circ>-120)*(base_circ<=-60)] = 4
            aha_region[(base_marker)*(base_circ>-60)*(base_circ<=0)] = 5
            aha_region[(base_marker)*(base_circ>=0)*(base_circ<=60)] = 6

            aha_region[(mid_marker)*(mid_circ>60)*(mid_circ<=120)] = 7
            aha_region[(mid_marker)*(mid_circ>120)*(mid_circ<=180)] = 8
            aha_region[(mid_marker)*(mid_circ>=-180)*(mid_circ<=-120)] = 9
            aha_region[(mid_marker)*(mid_circ>-120)*(mid_circ<=-60)] = 10
            aha_region[(mid_marker)*(mid_circ>-60)*(mid_circ<=0)] = 11
            aha_region[(mid_marker)*(mid_circ>=0)*(mid_circ<=60)] = 12

            aha_region[(apex_marker)*(apex_circ>45)*(apex_circ<=135)] = 13
            aha_region[(apex_marker)*((apex_circ>135)+(apex_circ<=-135))] = 14
            aha_region[(apex_marker)*(apex_circ>-135)*(apex_circ<=-45)] = 15
            aha_region[(apex_marker)*(apex_circ>-45)*(apex_circ<=45)] = 16

            aha_region[apex_apex_marker] = 17

            if aha_type == 'points':
                mesh.point_data['aha'] = aha_region
            elif aha_type == 'elems':
                mesh.cell_data['aha'] = [aha_region]

            
def rotate_z(mesh, angle_degrees):
    """Rotates a mesh around the Z-axis by the specified angle in degrees."""
    angle_radians = np.radians(angle_degrees)
    rotation_matrix = np.array([
        [np.cos(angle_radians), -np.sin(angle_radians), 0],
        [np.sin(angle_radians), np.cos(angle_radians), 0],
        [0, 0, 1]
    ])
    return np.dot(mesh, rotation_matrix.T)


def visualize_angle(angle_degrees):
    angle_radians = np.radians(angle_degrees)

    fig, ax = plt.subplots()
    ax.set_aspect('equal')

    # Draw a line for the x-axis
    ax.plot([0, 1], [0, 0], color='black')

    # Draw a line for the angle
    x = np.cos(angle_radians)
    y = np.sin(angle_radians)
    ax.plot([0, x], [0, y], color='blue')

    # Add an arc to visualize the angle
    arc = plt.Circle((0, 0), 0.2, color='red', fill=False)
    ax.add_artist(arc)
    ax.text(0.15, 0.05, f"{angle_degrees}°", fontsize=12)

    plt.title("Angle Visualization")
    plt.xlim(-1.2, 1.2)
    plt.ylim(-1.2, 1.2)
    plt.grid(True)
    plt.show()


def rotate_point_cloud(point_cloud, angle_degrees):
    """
    Rotates a point cloud around the Z-axis.

    Parameters:
        point_cloud (numpy.ndarray): A 3D point cloud represented as a Nx3 array.
        angle_degrees (float): The rotation angle in degrees.

    Returns:
        numpy.ndarray: The rotated point cloud.
    """

    # Convert angle to radians
    angle_radians = np.radians(angle_degrees)

    # Create a rotation matrix around the Z-axis
    rotation_matrix = np.array([
        [np.cos(angle_radians), -np.sin(angle_radians), 0],
        [np.sin(angle_radians), np.cos(angle_radians), 0],
        [0, 0, 1]
    ])

    # Apply the rotation matrix to each point in the point cloud
    rotated_point_cloud = np.dot(point_cloud, rotation_matrix.T)

    return rotated_point_cloud


class bullseye():
    def bullseye_plot(ax, data, seg_bold=None, cmap="viridis", norm=None):
        """
        Bullseye representation for the left ventricle.

        Parameters
        ----------
        ax : Axes
        data : list[float]
            The intensity values for each of the 17 segments.
        seg_bold : list[int], optional
            A list with the segments to highlight.
        cmap : colormap, default: "viridis"
            Colormap for the data.
        norm : Normalize or None, optional
            Normalizer for the data.

        Notes
        -----
        This function creates the 17 segment model for the left ventricle according
        to the American Heart Association (AHA) [1]_

        References
        ----------
        .. [1] M. D. Cerqueira, N. J. Weissman, V. Dilsizian, A. K. Jacobs,
            S. Kaul, W. K. Laskey, D. J. Pennell, J. A. Rumberger, T. Ryan,
            and M. S. Verani, "Standardized myocardial segmentation and
            nomenclature for tomographic imaging of the heart",
            Circulation, vol. 105, no. 4, pp. 539-542, 2002.
        """

        data = np.ravel(data)
        if seg_bold is None:
            seg_bold = []
        if norm is None:
            norm = mpl.colors.Normalize(vmin=data.min(), vmax=data.max())

        r = np.linspace(0.2, 1, 4)

        ax.set(ylim=[0, 1], xticklabels=[], yticklabels=[])
        ax.grid(False)  # Remove grid

        # Fill segments 1-6, 7-12, 13-16.
        for start, stop, r_in, r_out in [
                (0, 6, r[2], r[3]),
                (6, 12, r[1], r[2]),
                (12, 16, r[0], r[1]),
                (16, 17, 0, r[0]),
        ]:
            n = stop - start
            dtheta = 2*np.pi / n
            ax.bar(np.arange(n) * dtheta + np.pi/2, r_out - r_in, dtheta, r_in,
                color=cmap(norm(data[start:stop])))

        # Now, draw the segment borders.  In order for the outer bold borders not
        # to be covered by inner segments, the borders are all drawn separately
        # after the segments have all been filled.  We also disable clipping, which
        # would otherwise affect the outermost segment edges.
        # Draw edges of segments 1-6, 7-12, 13-16.
        for start, stop, r_in, r_out in [
                (0, 6, r[2], r[3]),
                (6, 12, r[1], r[2]),
                (12, 16, r[0], r[1]),
        ]:
            n = stop - start
            dtheta = 2*np.pi / n
            ax.bar(np.arange(n) * dtheta + np.pi/2, r_out - r_in, dtheta, r_in,
                clip_on=False, color="none", edgecolor="k", linewidth=[
                    4 if i + 1 in seg_bold else 2 for i in range(start, stop)])
        # Draw edge of segment 17 -- here; the edge needs to be drawn differently,
        # using plot().
        ax.plot(np.linspace(0, 2*np.pi), np.linspace(r[0], r[0]), "k",
                linewidth=(4 if 17 in seg_bold else 2))


# Main script

# AHA segments in geometry
msh = meshio.read(sys.argv[1])
his_point = ast.literal_eval(sys.argv[2])
his_circ = np.rad2deg(np.arctan2(his_point[1], his_point[0]))
rot_angle = -180 - (his_circ)

# load fractal tree output file
f1 = open('endo_pj_mid2_endnodes.txt', 'r')
content1 = f1.read().splitlines()
end_nodes = np.array([line.split() for line in content1 if line], dtype=int)  # Avoid empty rows
f1.close()

f2 = open('endo_pj_mid2_xyz.txt', 'r')
content2 = f2.read().splitlines()
pj_nodes_coord = np.array([line.split() for line in content2 if line], dtype=float)
f2.close()
rot_pj = rotate_point_cloud(pj_nodes_coord, rot_angle)

f3 = open('endo_pj_mid2_lines.txt', 'r')
content3 = f3.read().splitlines()
pj_connectivity = np.array([line.split() for line in content3 if line], dtype=int)
f3.close()

# check for terminal nodes, end_nodes includes end nodes of the first generation 
flat_data = pj_connectivity.flatten() # Flatten the 2D array into a 1D array
unique, counts = np.unique(flat_data, return_counts=True) # Count occurrences of each unique value
values_once = unique[counts == 1] # Filter values that appear only once
pj_end_nodes = values_once[1:] # Exclude the first entry
pj_end_nodes_coord = rot_pj[pj_end_nodes]

# generate_AHA segments in LV surface
generate_AHA.compute_aha_segments(msh,his_point,aha_type='elems')

# Distance to closest elem
elem_centroids = msh.cell_data['cell_centroid'][0] # Extract the array from the list
elem_aha = msh.cell_data['aha'][0] # Extract the array from the list

nodes_aha = np.zeros((17,1))

for item in range(len(pj_end_nodes)):
    distances = np.linalg.norm(elem_centroids- pj_end_nodes_coord[item], axis=1) # Compute the Euclidean distances from the target point to each point
    closest_index = np.argmin(distances) # Find the index of the closest point
    closest_point = elem_centroids[closest_index] # Get the closest point
    closest_distance = float(distances[closest_index])
    nodes_aha[int(elem_aha[closest_index])-1] += 1

# Make a figure and Axes with dimensions as desired.
fig = plt.figure(figsize=(10, 5), layout="constrained")
fig.get_layout_engine().set(wspace=.1, w_pad=.2)
axs = fig.subplots(1, 1, subplot_kw=dict(projection='polar'))
fig.canvas.manager.set_window_title('Left Ventricle Bulls Eyes (AHA)')

# Set the colormap and norm to correspond to the data for which
# the colorbar will be used.
cmap = mpl.cm.viridis
norm = mpl.colors.Normalize(vmin=min(nodes_aha), vmax=max(nodes_aha))
# Create an empty ScalarMappable to set the colorbar's colormap and norm.
# The following gives a basic continuous colorbar with ticks and labels.
fig.colorbar(mpl.cm.ScalarMappable(cmap=cmap, norm=norm),
             cax=axs.inset_axes([0, -.15, 1, .1]),
             orientation='horizontal', label='Number of PMJ in AHA')

# Create the 17 segment model
bullseye.bullseye_plot(axs, nodes_aha, cmap=cmap, norm=norm)
axs.set_title('Bulls Eye Plot for number of PMJs in AHA region')

fig.tight_layout()
fig.savefig('LV_AHA.png', transparent=True)
