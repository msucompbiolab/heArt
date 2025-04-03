# Preprocessing

## Generate Biventricular Geometry
An idealized biventricular geometry and mesh can be created using the python script ``create_idealizedBiVmesh.py``. The script takes 3 input (to be modified in the script itself):

- `gmshcmd`: path to the gmsh (ver > 4.7.1)
- `meshname`: name of the gmsh file without the *.geo extension. Geometry and mesh size of the biventricular unit will be set in this file (See here)
- `directory`: directory of the gmsh file

The python script will  
1. Call gmsh to create a mesh based on `<directory>+<meshname>.geo`
2. Set fiber/sheet/sheet-normal directions
3. Set surface labels for the LV endo, RV endo, epi and base.
4. Set material IDs for the LVFW, septum and RVFW
5. Generate 2 HDF5 files `<meshname>.hdf5` and `<meshname>+refine.hdf5` that can be used for simulation

# Biventricular gmsh file
The biventricular geometry is formed by boolean operation of 2 half ellipsoid. A schematic with the geometrical parameters is shown below.

![Biventricular geometry](../../figures/Schematic.png)

The parameters can be adjusted in the gmsh file.
