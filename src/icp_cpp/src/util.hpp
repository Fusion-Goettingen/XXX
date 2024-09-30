#pragma once
#include "typedefs.hpp"

#include <tbb/parallel_for_each.h>
#include <tbb/global_control.h>

/*
void inline transform_points(std::vector<Eigen::Vector3d>& points, const Sophus::SE3d T) {
    for(size_t i=0; i<points.size(); i++) {
        points[i] = T * points[i];
    }
};*/

void inline transform_points(std::vector<Eigen::Vector3d>& points, const Sophus::SE3d T) {
    tbb::parallel_for_each(points.begin(),points.end(),
    [&](Eigen::Vector3d &point) -> void {
        point = T * point;
    });
};

/*
void inline calculate_weights(std::vector<double>& square_dists, double tau) {
    double tau_square = tau * tau;
    for(size_t i=0; i<square_dists.size(); i++) {
        double _temp = tau + square_dists[i];
        _temp = _temp * _temp;
        square_dists[i] = tau_square / _temp;
    }
};*/

void inline calculate_weights(vector<Correspondence> &correspondences, double tau) {
    double tau_square = tau * tau;

    tbb::parallel_for_each(correspondences.begin(),correspondences.end(),
        [&](Correspondence &correspondence) -> void {
            double e = std::get<2>(correspondence);
            double _temp = tau + e;
            _temp = _temp * _temp;
            std::get<2>(correspondence) = tau_square / _temp;
        }
    );

    /*
    for(size_t i=0; i<correspondences.size(); i++) {
        double e = std::get<2>(correspondences[i]);
        double _temp = tau + e;
        _temp = _temp * _temp;
        std::get<2>(correspondences[i]) = tau_square / _temp;
    }*/

};

template<typename T>
Voxel inline calc_voxel(const T& e, double voxel_size) {
    return (e.head(3)  / voxel_size).template cast<int>();
};