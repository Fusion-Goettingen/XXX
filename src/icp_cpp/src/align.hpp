#pragma once
#define FMT_HEADER_ONLY
#include <fmt/format.h>

#include "typedefs.hpp"
#include "VoxelHashMap.hpp"
#include "util.hpp"
#include "correspondences.hpp"

#include <tbb/parallel_reduce.h>
#include <tbb/blocked_range.h>
#include <tbb/info.h>
#include <tbb/parallel_for.h>
#include <tbb/task_arena.h>

#include <climits>
#include <numeric>

JTJ_JTr align_point2point(vector<Correspondence> &correspondences) {
    auto compute_J_r = [&](const Correspondence &correspondence) -> J_r {
        Eigen::Matrix3_6d J;
        J.block<3,3>(0,0) = Eigen::Matrix3d::Identity();
        J.block<3,3>(0,3) = -1.0 * Sophus::SO3d::hat(std::get<0>(correspondence));

        Eigen::Vector3d r = std::get<0>(correspondence) - std::get<1>(correspondence);
        return J_r(J,r);
    };

    auto sum_JTJ_JTr = [](JTJ_JTr a, const JTJ_JTr &b) {
        a.first += b.first;
        a.second += b.second;
        return a;
    };

    const auto &res = tbb::parallel_reduce(
        tbb::blocked_range<vector<Correspondence>::const_iterator>{correspondences.cbegin(),correspondences.cend()},
        JTJ_JTr(Eigen::Matrix6d::Zero(), Eigen::Vector6d::Zero()),
        [&](const tbb::blocked_range<vector<Correspondence>::const_iterator> &r, JTJ_JTr sum) -> JTJ_JTr {
            return std::transform_reduce(
                r.begin(), r.end(), sum, sum_JTJ_JTr, [&](const auto &correspondence) {
                    const auto &[J, r] = compute_J_r(correspondence);
                    const double w = std::get<2>(correspondence);
                    return JTJ_JTr( J.transpose() * w * J,        // JTJ
                                    J.transpose() * w * r);  // JTr
                });
        },
        sum_JTJ_JTr);

    return res;
};


Eigen::Matrix_3x4d ICP(std::vector<Eigen::Vector3d> &frame,
                        VoxelHashMap<Eigen::Vector3d>& map,
                        double tau,
                        double max_corrs_dist,
                        size_t max_ICP_iters,
                        double min_transf_magn
                        ) {
    Sophus::SE3d T = Sophus::SE3d::exp(Eigen::Vector6d::Zero());

    size_t i = 0;
    double transformation_magnitude = std::numeric_limits<double>::max();

    while(i < max_ICP_iters && transformation_magnitude > min_transf_magn) {
        vector<Correspondence> correspondences = calculate_correspondences(frame,map,max_corrs_dist);
        calculate_weights(correspondences,tau);
        auto [JTJ, JTr] = align_point2point(correspondences);
        Eigen::Vector6d T_i_log = JTJ.ldlt().solve(-JTr);
        Sophus::SE3d T_i = Sophus::SE3d::exp(T_i_log);
        T = T_i * T;
        transform_points(frame,T_i);
        i++;
        transformation_magnitude = T_i_log.norm();
    }

    return T.matrix3x4();
};

