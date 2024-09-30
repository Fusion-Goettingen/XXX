#pragma once
#include "typedefs.hpp"
#include "VoxelHashMap.hpp"
#include "util.hpp"

#include <tbb/parallel_reduce.h>


template <typename T, typename K>
vector<Correspondence> calculate_correspondences(const vector<T> &source, VoxelHashMap<K>& map, double max_correspondence_distance) {
    const double squared_max_correspondence_distance = max_correspondence_distance * max_correspondence_distance;

    vector<Correspondence> correspodences;
    correspodences.reserve(source.size());

    auto concat_correspondences = [](vector<Correspondence> a, const vector<Correspondence> &b) {
        a.insert(a.end(), b.cbegin(), b.cend());
        return a;
    };



    vector<Correspondence> correspondences = tbb::parallel_reduce(
        tbb::blocked_range<size_t>(static_cast<size_t>(0),source.size()),
        vector<Correspondence>(),
        [&](tbb::blocked_range<size_t> partial_range, vector<Correspondence> partial_correspondences) {
            for (size_t i=partial_range.begin(); i<partial_range.end(); ++i) {
                const T& e = source[i];
                const Voxel voxel = calc_voxel(e,map._voxel_size);
                double min_squared_distance = squared_max_correspondence_distance;
                K min_element;
                for(const Voxel &voxel_neighbor : map._voxel_neighborhood) {
                    const Voxel search_voxel = voxel - voxel_neighbor;
                    const auto search = map._map.find(search_voxel);
                    if (search != map._map.end()) {
                        const std::vector<K> &search_voxel_elements = search.value();
                        for(const K &element : search_voxel_elements) {
                            const double squared_distance = (e - element.head(e.rows())).squaredNorm();
                            if(squared_distance < min_squared_distance) {
                                min_squared_distance = squared_distance;
                                min_element = element;
                            }
                        }
                    }
                }
                if(min_squared_distance < squared_max_correspondence_distance) {
                    partial_correspondences.push_back(std::make_tuple(e,min_element,min_squared_distance));
                }
            }
            return partial_correspondences;
        },
        concat_correspondences
    );

    return correspondences;
};