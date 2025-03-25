#!/bin/sh

# parameters_base
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