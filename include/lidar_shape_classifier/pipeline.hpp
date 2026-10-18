#pragma once

#include "lidar_shape_classifier/types.hpp"

#include <string>
#include <vector>

namespace lsc {

// Voxel grid. A non-positive leaf size keeps every finite point.
std::vector<Eigen::Vector3f> voxel_downsample(const std::vector<Eigen::Vector3f>& points,
                                              float leaf_size);

// Statistical outlier removal. Fewer points than mean_k, or a non-positive
// multiplier, keeps every finite point.
std::vector<Eigen::Vector3f> remove_statistical_outliers(const std::vector<Eigen::Vector3f>& points,
                                                         int mean_k, float stddev_mul);

struct GroundRemoval {
  std::vector<Eigen::Vector3f> non_ground;
  bool removed = false;
};

// RANSAC plane whose normal is close to +Z. Points on that plane are dropped.
GroundRemoval remove_ground_plane(const std::vector<Eigen::Vector3f>& points,
                                  const PipelineConfig& config);

std::vector<std::vector<Eigen::Vector3f>> euclidean_clusters(
    const std::vector<Eigen::Vector3f>& points, const PipelineConfig& config);

// Geometric label for one cluster: RANSAC plane and sphere, PCA cylinder
// radius, and an oriented-box face score.
ClassifiedCluster classify_cluster(const std::vector<Eigen::Vector3f>& points,
                                   const PipelineConfig& config);

PipelineResult run_pipeline(const std::vector<Eigen::Vector3f>& points,
                            const PipelineConfig& config = {});

std::vector<Eigen::Vector3f> load_cloud(const std::string& path);
void save_cloud(const std::string& path, const std::vector<Eigen::Vector3f>& points);

}  // namespace lsc
