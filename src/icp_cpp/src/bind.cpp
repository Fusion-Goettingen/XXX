#include "typedefs.hpp"
#include "util.hpp"
#include "VoxelHashMap.hpp"

#include "align.hpp"

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/eigen.h>

template<typename T>
void declare_VoxelHashMap(pybind11::module &m, const std::string &typestr) {
    using Class = VoxelHashMap<T>;
    std::string pyclass_name = std::string("VoxelHashMap") + typestr;

    pybind11::class_<Class>(m, pyclass_name.c_str(), pybind11::buffer_protocol(), pybind11::dynamic_attr())
        .def(pybind11::init(&Class::Create))
        .def("add", &Class::add)
        .def("remove_outside", &Class::remove_outside)
        .def("get_all", &Class::get_all)
        .def("get_world_size", &Class::get_world_size);

    std::string name = std::string("calculate_correspondences_") + typestr;
    m.def(name.c_str(), &calculate_correspondences<T,T>,"Aligns two point clouds");
}


PYBIND11_MODULE(icp_cpp, m){
    m.doc() = "this is an example docstring for the module";
    m.def("ICP",&ICP,"Aligns two point clouds");

    declare_VoxelHashMap<Eigen::Vector3d>(m,"3d");
}
