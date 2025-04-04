# Fractal Tree Generation
This code is to create a fractal tree over a surface discretized by triangles and includes further modifications from the initial code. It was developed to create a representation of the Purkinje network in the ventricles of the human heart. (Costobal et al. 2015)

## Modification
- Initial trunk is deterministic while further branches are based on stochastic modification

## Install
You can install the library with pip
```
python3 -m pip install fractal-tree
```
Note that you also need a way to load the mesh from e.g gmsh or another meshing tool. For this we recommend to use [`meshio`](https://github.com/nschloe/meshio) as it support the most common formats.

## After Install
Swap local fractal-tree folder with here provided fractal-tree folder


## Example
The following illustrates the base parameters of the fractal tree, assuming that you have surface mesh called `endo_refine.vtu` in your working direction. 
```
generate_base_file="script_fractal.py" # runfile
surface="endo_refine.vtu" # endocardial surface
init_node="[-0.574335,-1.8842,-0.168375]" # assumed to be His bundle
name="endo_pj_base"
init_dir="[-0.0,0.0,-1.0]" # intitial direction
init_length="1.5" # trunk length
N_init="0" # number of generations
mode="1" # identifies first or further generations
branch_angle="1.8" # angle of bifurcation branches with respect to initial trunk
length="1.8" # initial branch length
l_segment="0.18" # segment length of branch
fascicles_length=0.6 #1.0 # length of the fascies after trunk downward
fascicles_angles=0.7 # angle for the first branch related to the initial trunk

cd examples/pj_pig/refine/

python3 $generate_base_file $surface $init_node $name $init_dir $init_length $N_init $mode $fascicles_angles $branch_angle $length $l_segment $fascicles_length
```

For tree generation type
```
./run.sh
```
in command line. 

In script_fractal.py: 

```
branches_base, nodes_base, lines_base, end_nodes_base, branches_to_grow_base, last_branch_base = generate_fractal_tree(mesh, param_base)
```
generates the initial trunk and the bifurcation of the Purkinje network. 
The next level of the Purkinje fibers adds on to the generated base network:
```
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
    repulsitivity=0.3,
    )
```
Here the variable 
- mode: stochasticity (2) or deterministic (1)
- N_int: number of generations
- repulsitivity: defines how strongly branches are repelled by each other [0.0 - 1.0]

## Results
This specific code has tree levels of tree generation:
- base (output: endo_pj_base)
- mid (output: endo_pj_mid1)
- final (output: endo_pj_mid2)

The final outcome for the Purkinje fibers are stored in 
- endo_pj_mid2.vtu: geometry of Purkinje fiber network
- endo_pj_mid2_endnodes.txt: end nodes of Purkinje fiber network
- endo_pj_mid2_xyz.txt: coordinates of the end nodes
- endo_pj_mid2_lines.txt: connectivity of Purkinje fiber network
- LV_AHA.png: Bulls Eye Plot for number of Purkinje-Myocard junctions (PMJs) in AHA region



## fractal-tree Original
Note that this is a rewrite of the original code found at https://github.com/fsahli/fractal-tree

The details of the algorithm are presented in this [article](http://www.sciencedirect.com/science/article/pii/S0021929015007332). If you are going to use this code, please cite:

> Generating Purkinje networks in the human heart. F. Sahli Costabal, D. Hurtado and E. Kuhl. Journal of Biomechanics, doi:10.1016/j.jbiomech.2015.12.025

- Source code: https://github.com/finsberg/fractal-tree
- Documentation: https://github.com/finsberg/fractal-tree