#pragma once
#define FMT_HEADER_ONLY
#include "fmt/format.h"

#include <Eigen/Dense>
#include <sophus/se3.hpp>

#include <tsl/robin_map.h>

#include "typedefs.hpp"

#include <iostream>
#include <Eigen/Dense>
#include <pybind11/pybind11.h>
#include <pybind11/eigen.h>

template<typename T>
struct VoxelHashMap {
    // For hash calculation
    struct VoxelHash {
        size_t operator()(const Voxel &voxel) const {
            const uint32_t *vec = reinterpret_cast<const uint32_t *>(voxel.data());
            return ((1 << 20) - 1) & (vec[0] * 73856093 ^ vec[1] * 19349669 ^ vec[2] * 83492791);
        }
    };

    std::vector<bool> add(std::vector<T> xs) {
        // Mask is a bool-mask of which elements have been inserted
        std::vector<bool> mask;
        mask.reserve(xs.size());
        //Iterate through all elements
        for(size_t i=0; i<xs.size(); i++) {
            const T &e = xs[i];
            Voxel voxel = calc_voxel(e,_voxel_size);
            auto search = _map.find(voxel);
            if(search == _map.end()) {
                // Create Voxel if not already present
                std::vector<T> voxel_list;
                voxel_list.push_back(e);
                _map.insert({voxel, voxel_list});
                mask.push_back(true);
            } else {
                // Insert element in Voxel if voxel already present
                std::vector<T> &voxel_list = search.value();
                if(voxel_list.size() < _max_points_per_voxel) {
                    voxel_list.push_back(e);
                    mask.push_back(true);
                } else {
                    mask.push_back(false);
                }
            }
        }
        return mask;
    };

    void remove_outside(const Eigen::Vector3d &center) {
        const double squared_radius = _world_size * _world_size;
        for (const auto &[voxel, voxel_list] : _map) {
            const auto &pt = voxel_list.front().head(3);
            if ((pt - center).squaredNorm() > squared_radius) {
                _map.erase(voxel);
            }
        }
    };

    std::vector<T> get_all() {
        std::vector<T> all_points;
        for (const auto &[voxel, voxel_list] : _map) {
            all_points.insert(all_points.end(), voxel_list.begin(), voxel_list.end());
        }
        return all_points;
    };

    VoxelHashMap(double voxel_size, size_t max_points_per_voxel, double world_size) {
        _voxel_size = voxel_size;
        _max_points_per_voxel = max_points_per_voxel;
        _world_size = world_size;
    };
    // Factory constructor for pybind
    static VoxelHashMap<T> Create(double voxel_size, size_t max_points_per_voxel, double world_size) {
        return VoxelHashMap<T>(voxel_size,max_points_per_voxel, world_size);
    };

    tsl::robin_map<Voxel, std::vector<T>, VoxelHash> _map;
    const Voxel _voxel_neighborhood[27] = {  Voxel(-1,-1,-1),Voxel(-1,-1,0),Voxel(-1,-1,1),Voxel(-1,0,-1),Voxel(-1,0,0),Voxel(-1,0,1),Voxel(-1,1,-1),Voxel(-1,1,0),Voxel(-1,1,1),
                                            Voxel(0,-1,-1),Voxel(0,-1,0),Voxel(0,-1,1),Voxel(0,0,-1),Voxel(0,0,0),Voxel(0,0,1),Voxel(0,1,-1),Voxel(0,1,0),Voxel(0,1,1),
                                            Voxel(1,-1,-1),Voxel(1,-1,0),Voxel(1,-1,1),Voxel(1,0,-1),Voxel(1,0,0),Voxel(1,0,1),Voxel(1,1,-1),Voxel(1,1,0),Voxel(1,1,1)};
    double _voxel_size;
    size_t _max_points_per_voxel;
    double _world_size;

    double get_world_size() {
        return _world_size;
    }
};


