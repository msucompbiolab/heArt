#!/bin/sh

cd examples/idealized_biv/

# parameters_base
# generate_base_file="script_fractal_lv.py" # runfile
# surface="LV_endo.vtu" # endocardial surface
# # init_node="[0.01071,1.97936,-1.10766]"  
# # init_node="[-0.44504,1.94986,0]" # assumed to be His bundle
# init_node="[3e-05,1.13,0.0]" # assumed to be His bundle
# name="endoLV_pj_base"
# init_dir="[-0.0,0.0,-0.1]" # intitial direction
# init_length="1.5" # trunc length
# N_init="0" # number of branch generation
# mode="1" # identifies first or further generations
# branch_angle="1.8" # important
# length="1.8" # initial branch length
# l_segment="0.18" # segment length of branch
# fascicles_length=0.6 #1.0 # length of the fascies after trunc downward
# fascicles_angles=0.7 # angle for the first branch related to the initial trunc

# python3 $generate_base_file $surface $init_node $name $init_dir $init_length $N_init $mode $fascicles_angles $branch_angle $length $l_segment $fascicles_length

generate_base_file="script_fractal_rv.py" # runfile
surface="RV_endo.vtu" # endocardial surface
init_node="[0,3.13,0]" # assumed to be His bundle
name="endoRV_pj_base"
init_dir="[-0.0,0.0,-1.0]" # intitial direction
init_length="1.8" # trunc length
N_init="0" # number of branch generation
mode="1" # identifies first or further generations
branch_angle="2.1" # important
length="2.1" # initial branch length
l_segment="0.21" # segment length of branch
fascicles_length=0.6 #1.0 # length of the fascies after trunc downward
fascicles_angles=0.7 # angle for the first branch related to the initial trunc

python3 $generate_base_file $surface $init_node $name $init_dir $init_length $N_init $mode $fascicles_angles $branch_angle $length $l_segment $fascicles_length
