# Fractal Tree Generation
This code is to create a fractal tree over a surface discretized by triangles and includes further modifications from the initial code. It was developed to create a representation of the Purkinje network in the ventricles of the human heart. (Costobal et al. 2015)

## Modification
- Initial trunc is deterministic while further branches are based on stochastic modification

## Install
You can install the library with pip
```
python3 -m pip install fractal-tree
```
Note that you also need a way to load the mesh from e.g gmsh or another meshing tool. For this we recommend to use [`meshio`](https://github.com/nschloe/meshio) as it support the most common formats.

## After Install
Swap local fractal-tree folder with here provided fractal-tree folder

## fractal-tree Original
Note that this is a rewrite of the original code found at https://github.com/fsahli/fractal-tree

The details of the algorithm are presented in this [article](http://www.sciencedirect.com/science/article/pii/S0021929015007332). If you are going to use this code, please cite:

> Generating Purkinje networks in the human heart. F. Sahli Costabal, D. Hurtado and E. Kuhl. Journal of Biomechanics, doi:10.1016/j.jbiomech.2015.12.025

- Source code: https://github.com/finsberg/fractal-tree
- Documentation: https://github.com/finsberg/fractal-tree

## Example

The following illustrates a minimal example, assuming that you have surface mesh called `endo_refine.vtu` in your working direction. 

```
#!/bin/sh

#parameters_base
generate_base_file="script_fractal.py" # runfile
surface="endo_refine.vtu" # endocardial surface
init_node="[-0.574335,-1.8842,-0.168375]" # assumed to be His bundle
name="endo_pj_base"
init_dir="[-0.0,0.0,-1.0]" # intitial direction
init_length="1.5" # trunc length
N_init="0" # number of branch generation
mode="1" # identifies first or further generations
branch_angle="1.8" # important
length="1.8" # initial branch length
l_segment="0.18" # segment length of branch
fascicles_length=0.6 #1.0 # length of the fascies after trunc downward
fascicles_angles=0.7 # angle for the first branch related to the initial trunc

cd examples/pj_pig/refine/

python3 $generate_base_file $surface $init_node $name $init_dir $init_length $N_init $mode $fascicles_angles $branch_angle $length $l_segment $fascicles_length
```