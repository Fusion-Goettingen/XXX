#define FMT_HEADER_ONLY
#include <fmt/format.h>

#include <eigen3/Eigen/Dense>
#include <sophus/se3.hpp>

namespace Eigen {
    using Matrix_3x4d = Eigen::Matrix<double, 3, 4>; 
    using Vector9d = Eigen::Matrix<double, 9, 1>;
    using Vector6d = Eigen::Matrix<double, 6, 1>;
    using Vector3d = Eigen::Matrix<double, 3, 1>;
    using Vector2d = Eigen::Matrix<double, 2, 1>;
    using Vector12d  = Eigen::Matrix<double, 12, 1>;
    using Matrix6d = Eigen::Matrix<double,6,6>;
    using Matrix3_6d  = Eigen::Matrix<double, 3, 6>;
    using Matrix12_6d  = Eigen::Matrix<double, 12, 6>;
    using Matrix9_6d  = Eigen::Matrix<double, 9, 6>;
    using Matrix9_3d   = Eigen::Matrix<double, 9, 3>;
    using Matrix3_2d   = Eigen::Matrix<double, 3, 2>;
}


using Voxel = Eigen::Vector3i;
using Stixel = Eigen::Vector2i;

using std::tuple;
using std::vector;
using std::pair;


using Correspondence = tuple<Eigen::Vector3d,Eigen::Vector3d,double>;
using JTJ_JTr = pair<Eigen::Matrix6d,Eigen::Vector6d>;
using J_r = pair<Eigen::Matrix3_6d,Eigen::Vector3d>;