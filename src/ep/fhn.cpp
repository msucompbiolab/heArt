#include <pybind11/pybind11.h>
#include <petscvec.h>
#include <dolfin/la/PETScVector.h>
#include <pybind11/stl.h>
#include <dolfin/parameter/Parameters.h>
#include <iostream>
#include <map>
#include <string>

namespace py = pybind11;
using namespace dolfin;

class FHNmodel {

private:

  // Default parameter values
  double c = 8;
  double alpha = 0.01;
  double g = 0.002;
  double b = 0.15;
  double mu1 = 0.2;
  double mu2 = 0.3;
  double fhn_timeNormalizer = 12.9;
  double dt = 1.0;
  double k = dt/fhn_timeNormalizer ;

public:    

  // Define the FHNmodel object with a dictionary of the parameters
  FHNmodel(const py::dict& py_dict) {
      Parameters solver_params = convert_dict_to_parameters(py_dict);
      for (const auto& p : solver_params) {
          if(p.first == "c"){
              c = double(solver_params[p.first]);
              //std::cout << p.first << " = " << c << std::endl;
          };
          if(p.first == "alpha"){
              alpha = double(solver_params[p.first]);
              //std::cout << p.first << " = " << alpha << std::endl;
          };
          if(p.first == "g"){
              g = double(solver_params[p.first]);
              //std::cout << p.first << " = " << g << std::endl;
          };
          if(p.first == "b"){
              b = double(solver_params[p.first]);
              //std::cout << p.first << " = " << b << std::endl;
          };
          if(p.first == "mu1"){
              mu1 = double(solver_params[p.first]);
              //std::cout << p.first << " = " << mu1 << std::endl;
          };
          if(p.first == "mu2"){
              mu2 = double(solver_params[p.first]);
              //std::cout << p.first << " = " << mu2 << std::endl;
          };
          if(p.first == "k"){
              k = double(solver_params[p.first]);
              //std::cout << p.first << " = " << k << std::endl;
          };
      };
  };

  // Convert Python dictionary to Dolfin Parameters
  Parameters convert_dict_to_parameters(const py::dict& py_dict) {
      Parameters params;
      
      for (auto item : py_dict) {
          std::string key = item.first.cast<std::string>();
          double value = item.second.cast<double>();
          params.add(key, value);
      };
      
      return params;
  };

  // Update fphi locally to be -c*u * (u - alpha) * (u - 1.0) - r * u[i];
  void Update_fphi(std::shared_ptr<dolfin::PETScVector> fphi, std::shared_ptr<dolfin::PETScVector> phi, std::shared_ptr<dolfin::PETScVector> r)
  {
 
    Vec fphi_ = fphi->vec();
    Vec phi_ = phi->vec();
    Vec r_ = r->vec();
    assert(phi_);
    assert(r_);
    assert(fphi_);
  
    PetscInt local_size;
    PetscScalar *array_fphi;
    PetscScalar *array_phi;
    PetscScalar *array_r;
  
    VecGetLocalSize(fphi_, &local_size);
    VecGetArray(fphi_, &array_fphi);
    VecGetArray(phi_, &array_phi);
    VecGetArray(r_, &array_r);
    
    for (PetscInt i = 0; i < local_size; i++) {
        array_fphi[i] = -c * array_phi[i] * (array_phi[i] - alpha) * (array_phi[i] - 1.0) - array_r[i] * array_phi[i];
    };
  
    VecRestoreArray(fphi_, &array_fphi);
  
  }; 

  // Update r using explicit time integration i.e.
  // r =  rn + k*fr where fr = eps(r,u) * (-r - c * u *(u - b - 1.0));
  void Update_r(std::shared_ptr<dolfin::PETScVector> phi, std::shared_ptr<dolfin::PETScVector> r, const py::dict& py_dict)
  {
  
    double eps;
    double fr;

    Vec phi_ = phi->vec();
    Vec r_ = r->vec();
    assert(phi_);
    assert(r_);
  
    PetscInt local_size;
    PetscScalar *array_phi;
    PetscScalar *array_r;
  
    VecGetLocalSize(phi_, &local_size);
    VecGetArray(phi_, &array_phi);
    VecGetArray(r_, &array_r);
  
    for (PetscInt i = 0; i < local_size; i++) {
        eps = g + mu1 * array_r[i] / (mu2 + array_phi[i]);
        fr = eps * (-array_r[i] - c * array_phi[i] *(array_phi[i] - b - 1.0));
        array_r[i] = array_r[i] + k*fr;
        //std::cout << "i = " << i << " " << array_r[i]  <<  std::endl;
    };
  
    VecRestoreArray(r_, &array_r);

  };

  void Zero_r(std::shared_ptr<dolfin::PETScVector> r)
  {
    Vec r_ = r->vec();
    assert(r_);
  
    PetscInt local_size;
    PetscScalar *array_r;
  
    VecGetLocalSize(r_, &local_size);
  
    VecGetArray(r_, &array_r);
  
    for (PetscInt i = 0; i < local_size; i++) {
        array_r[i] = 0.0;
    };
  
    VecRestoreArray(r_, &array_r);

  };

};

// Bind cpp object to python
PYBIND11_MODULE(SIGNATURE, m) {
    py::class_<FHNmodel>(m, "FHNmodel")
        .def(py::init<py::dict>())  // Constructor with dictionary
        .def("Update_fphi", &FHNmodel::Update_fphi)
        .def("Update_r", &FHNmodel::Update_r)
        .def("Zero_r", &FHNmodel::Zero_r);
};





