# heArt
Simulator of electromechanics in the heart based on [FEniCS](https://fenicsproject.org/) library.

<!-- TOC -->

- [heArt](#heart)
  - [Installation and Running the code](#installation-and-running-the-code)
  - [Organization of the code](#organization-of-the-code)
  - [Simulation protocols](#simulation-protocols)

<!-- /TOC -->

### Installation and Running the code
A singularity "build" [file](./singularity/Singularity_fenics2017_msucompbiomechlab) is provided that will install all the libraries required to run the code.

1. Install singularity by following the instruction in [here](https://sylabs.io/guides/3.5/admin-guide/installation.html)

2. Build a singularity container using the "build" [file](./singularity/Singularity_fenics2017_msucompbiomechlab) with
```
sudo singularity build <container_name>.img Singularity_fenics2017_msucompbiomechlab
```

3. Once the container is built, you can launch the Singularity container by
```
 singularity run <container_name>.img
```

4. The code can be run within the singularity container. For example, for the code [isotonic_test](./cases/demo/isotonic_test.py)  
```
python isotonic_test.py
```
or in parallel
```
mpirun.mpich -np <# processors> python isotonic_test.py
```

### Organization of the code
The code is organized as follows:
- [mechanics module](./src/mechanics)
- [electrophysiology module](./src/ep)
- [simulation protocols](./src/sim_protocols/README.md)
- [utilities](./src/utils)
- [benchmark analytical solution](./src/bmark_analytical)
- [postprocessing](./src/postprocessing)

Demo python scripts are also provided to simulate
- [uniaxial passive stretching](./demo/uniaxial_passive_stretch.py)
- [isometric tension test](./demo/isometric_test.py)
- [isotonic tension test](./demo/isotonic_test.py)
- [closed loop simulation of LV](./demo/LVelectromechanics.py)
- [closed loop simulation of BiV](./demo/BiVelectromechanics.py)

### Simulation protocols
 Cardiac Simulator
