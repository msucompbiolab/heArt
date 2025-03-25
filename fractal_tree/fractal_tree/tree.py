"""
This module contains the function that creates the fractal tree.
"""
from __future__ import annotations
from dataclasses import dataclass
import sys
from typing import Optional, NamedTuple, Union
import numpy as np
import logging
import tqdm
from pathlib import Path

from .branch import Nodes, Branch
from .mesh import Mesh
from .viz import write_line_VTU
import pdb

logger = logging.getLogger(__name__)


def grow_fascicles(
    branches: dict[int, Branch],
    parameters: FractalTreeParameters,
    mesh: Mesh,
    nodes: Nodes,
    lines: list[tuple[int, int]],
    last_branch: int,
):
    brother_nodes = []
    if parameters.mode==1:
        brother_nodes += branches[0].nodes
        for i_branch in range(len(parameters.fascicles_angles)):
            last_branch += 1
            angle = parameters.fascicles_angles[i_branch]
            branches[last_branch] = Branch(
                mesh,
                branches[0].nodes[-1],
                branches[0].dir,
                branches[0].tri,
                parameters.fascicles_length[i_branch],
                angle,
                0.0,
                nodes,
                brother_nodes,
                int(parameters.fascicles_length[i_branch] / parameters.l_segment),
            )

            brother_nodes += branches[last_branch].nodes

            for i_n in range(len(branches[last_branch].nodes) - 1):
                lines.append(
                    (
                        branches[last_branch].nodes[i_n],
                        branches[last_branch].nodes[i_n + 1],
                    )
                )
        branches_to_grow = list(range(1, len(parameters.fascicles_angles) + 1))
    return branches_to_grow, last_branch


def run_generation(
    branches_to_grow: list[int],
    parameters: FractalTreeParameters,
    branches: dict[int, Branch],
    last_branch: int,
    mesh: Mesh,
    nodes: Nodes,
    lines: list[tuple[int, int]],
):

    # np.random.seed(42)  # Fixed seed for controlled randomness

    if parameters.mode == 1: #determinant
        choices = np.array([1 if i % 2 == 0 else -1 for i in range(len(branches_to_grow))])
        lengths = np.full(2 * len(branches_to_grow), parameters.std_length)
    else: #statistically
        choices = 2 * np.random.randint(2, size=len(branches_to_grow)) - 1
        lengths = np.random.normal(0, parameters.std_length, size=2 * len(branches_to_grow))

    k = 0
    k1 = 0

    # if parameters.mode !=1:
    #     np.random.shuffle(branches_to_grow)
    new_branches_to_grow = []
    # for branch_index in branches_to_grow:
    for branch_index in tqdm.tqdm(branches_to_grow, desc="Generating branches", miniters=1, maxinterval=1):
        branch = branches[branch_index]
        angle = -parameters.branch_angle * choices[k]
        k += 1
        for j in range(2):
            brother_nodes = branch.nodes.copy()
            if j > 0:
                brother_nodes += branches[last_branch].nodes

            # Add new branch
            last_branch += 1
            logger.debug(last_branch)
            total_length = parameters.length + lengths[k1]
            k1 += 1
            if total_length < parameters.min_length:
                total_length = parameters.min_length

            # pdb.set_trace()
            branches[last_branch] = Branch(
                mesh=mesh,
                initial_node=branch.nodes[-1],
                initial_direction=branch.dir,
                initial_triangle=branch.tri,
                length=total_length,
                angle=angle,
                repulsitivity=parameters.repulsitivity,
                nodes=nodes,
                brother_nodes=brother_nodes,
                num_segments=int(parameters.length / parameters.l_segment),
                )

            # Add nodes to lines
            for n1, n2 in zip(
                branches[last_branch].nodes[:-1], branches[last_branch].nodes[1:]
            ):
                lines.append((n1, n2))

            # Add to the new array
            if branches[last_branch].growing:
                new_branches_to_grow.append(last_branch)

            branch.child[j] = last_branch
            angle = -angle

    branches_to_grow = new_branches_to_grow
    return branches, nodes, lines, branches_to_grow, lines, last_branch


def save_tree(
    filename: Union[Path, str],
    nodes: np.ndarray,
    end_nodes: list[int],
    lines: list[tuple[int, int]],
    save_paraview: bool = True):
    logger.info("Finished growing, writing paraview file")
    write_line_VTU(nodes, lines, Path(filename).with_suffix(".vtu"))
    name = Path(filename).name
    np.savetxt(Path(filename).with_name(name + "_lines").with_suffix(".txt"), lines, fmt="%d")
    np.savetxt(Path(filename).with_name(name + "_xyz").with_suffix(".txt"), nodes)
    np.savetxt(Path(filename).with_name(name + "_endnodes").with_suffix(".txt"),end_nodes, fmt="%d")


@dataclass
class FractalTreeParameters:
    """Class to specify the parameters of the fractal tree.

    Attributes:
        filename (str):
            name of the output files.
        init_node (numpy array):
            the first node of the tree.
        second_node (numpy array):
            this point is only used to calculate the
            initial direction of the tree and is not
            included in the tree. Please avoid selecting
            nodes that are connected to the init_node by a
            single edge in the mesh, because it causes numerical issues.
            If no node is provided, a random node will be selected
        init_length (float):
            length of the first branch.
        N_it (int):
            number of generations of branches.
        length (float):
            average length of the branches in the tree.
        branch_angle (float):
            angle with respect to the direction of
            the previous branch and the new branch.
        repulsitivity (float):
            repulsitivity parameter.
        l_segment (float):
            length of the segments that compose one branch
            (approximately, because the length of the branch is random).
            It can be interpreted as the element length in a finite element mesh.
        Fascicles (bool):
            include one or more straight branches with different lengths and
            angles from the initial branch. It is motivated by the fascicles
            of the left ventricle.
        fascicles_angles (list):
            angles with respect to the initial branches of the fascicles.
            Include one per fascicle to include.
        fascicles_length (list):
            length  of the fascicles. Include one per fascicle to include.
            The size must match the size of fascicles_angles.
        save (bool):
            save text files containing the nodes, the connectivity and end
            nodes of the tree.
        save_paraview (bool):
            save a .vtu paraview file. The tvtk module must be installed.

    """
    filename: str = "results"
    second_node: Optional[np.ndarray] = None
    initial_direction: Optional[np.ndarray] = None
    init_length: float = 0.1
    N_it: int = 10  # Number of iterations (generations of branches)
    length: float = 0.1  # Median length of the branches
    branch_angle: float = 0.15
    repulsitivity: float = 0.1
    l_segment: float = 0.01  # Length of the segments (approximately, because
    # the length of the branch is random)
    init_node: Optional[np.ndarray] = None
    mode: int = 1
    generate_fascicles: bool = True
    fascicles_angles: tuple[float, float] = (-1.5, 0.2)  # rad
    fascicles_length: tuple[float, float] = (0.5, 0.5)
    fasciles_length_base: float = 0.4
    branches: Optional[np.ndarray] = None
    nodes: Optional[np.ndarray] = None
    lines: Optional[np.ndarray] = None
    end_nodes: Optional[np.ndarray] = None
    node_init: Optional[int] = None
    last_branch: Optional[int] = None
    branches_to_grow:Optional[np.ndarray] = None
    b1_direction: Optional[np.ndarray] = None
    b2_direction: Optional[np.ndarray] = None
    save: bool = True
    save_paraview: bool = True

    @property
    def std_length(self) -> float:
        """Standard deviation of the length.
        Set to zero to avoid random lengths."""
        # return np.sqrt(0.2) * self.length
        return 0

    @property
    def min_length(self) -> float:
        """Minimum length of the branches.
        To avoid randomly generated negative lengths."""
        return self.length / 10.0

    def as_dict(self):
        return {k: v for k, v in self.__dict__.items()}


class FractalTreeResult(NamedTuple):
    branches: dict[int, Branch]
    nodes: np.ndarray
    lines: list[tuple[int, int]]
    end_nodes: list[int]
    branches_to_grow: list[int]
    last_branch: list[int]

def node_direction(src: np.ndarray, target: Optional[np.ndarray] = None) -> np.ndarray:
    """Return the direction from src to target.

    Parameters
    ----------
    src : np.ndarray
        Source node
    target : Optional[np.ndarray]
        Target node

    Returns
    -------
    np.ndarray
        The unit vector from src to node 2
    """
    return (np.array(target) - np.array(src)) / np.linalg.norm(np.array(target) - np.array(src))


def generate_fractal_tree(
    mesh: Mesh, parameters: Optional[FractalTreeParameters] = None
) -> FractalTreeResult:
    """This function creates the fractal tree.

    Args:
        parameters (Optional[FractalTreeParameters]):
            This object contains all the parameters that
            define the tree. See the parameters module documentation for details:

    Returns:
        FractalTreeResult: branches containes a dictionary that contains
        all the branches objects, and nodes is the object that
        contains all the nodes of the tree.
    """

    if parameters is None:
        parameters = FractalTreeParameters()

    if parameters.mode == 1:
        if parameters.initial_direction is None:
            # Get the second node to define the initial direction
            second_node = parameters.second_node
            if second_node is None:
                # If no node is specified, lets just pick a random node
                second_node = mesh.verts[np.random.choice(mesh.valid_nodes), :]
            # Define the initial direction
            initial_direction = node_direction(src=parameters.init_node, target=second_node)
        else:
            initial_direction = parameters.initial_direction

        # Initialize the nodes object, contains the nodes and all the distance functions
        if parameters.mode == 1:
            nodes = Nodes(parameters.init_node,parameters.mode)

            # Project the first node to the mesh.
            point = mesh.project_new_point(nodes.nodes[0])
            if point.triangle_index >= 0:
                initial_triangle = point.triangle_index
            else:
                logger.error("initial point not in mesh")
                sys.exit(0)

            # Initialize the dictionary that stores the branches objects
            branches = {}
            last_branch = 0

        # Compute the first branch trunc downwards to second_node
            branches[last_branch] = Branch(
                mesh=mesh,
                initial_node=0,
                initial_direction=initial_direction,
                initial_triangle=initial_triangle,
                length=parameters.init_length,
                angle=0.0,
                repulsitivity=0.0,
                nodes=nodes,
                brother_nodes=[0],
                num_segments=int(parameters.init_length / parameters.l_segment),
            )

            branches_to_grow = []
            branches_to_grow.append(last_branch)

            lines = [
                (n1, n2)
                for n1, n2 in zip(
                    branches[last_branch].nodes[:-1], branches[last_branch].nodes[1:]
                )
            ]

            # To grow fascicles
            if parameters.generate_fascicles:
                branches_to_grow, last_branch = grow_fascicles(
                    branches, parameters, mesh, nodes, lines, last_branch
                    )
    else:
        branches = parameters.branches
        nodes = parameters.nodes
        lines=parameters.lines
        branches_to_grow=parameters.branches_to_grow
        last_branch=parameters.last_branch

    if parameters.mode == 1:
        print("Trunk generation done")
        # for _ in tqdm.tqdm(range(parameters.N_it)):
        #     pdb.set_trace()
        #     branches, nodes, lines, branches_to_grow, lines, last_branch = run_generation(
        #         branches_to_grow, parameters, branches, last_branch, mesh, nodes, lines
        #     )        
    else:
        for _ in tqdm.tqdm(range(parameters.N_it)):
            branches, nodes, lines, branches_to_grow, lines, last_branch = run_generation(
                branches_to_grow, parameters, branches, last_branch, mesh, nodes, lines
            )
        print("Branch generation done")

    xyz = np.array(nodes.nodes)

    if parameters.save:
        save_tree(
            filename=parameters.filename,
            nodes=xyz,
            lines=lines,
            end_nodes=nodes.end_nodes,
            save_paraview=parameters.save_paraview,
        )

    return FractalTreeResult(
        branches=branches, nodes=nodes, lines=lines, end_nodes=nodes.end_nodes, branches_to_grow=branches_to_grow, last_branch=last_branch)
