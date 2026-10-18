#include "lidar_shape_classifier/pipeline.hpp"

#include "points.hpp"

#include <utility>
#include <vector>

namespace lsc {

PipelineResult run_pipeline(const std::vector<Eigen::Vector3f>& points, const PipelineConfig& config) {
  PipelineResult result;
  result.input_points = points.size();

  const std::vector<Eigen::Vector3f> voxel = voxel_downsample(points, config.voxel_leaf_size);
  result.voxel_points = voxel.size();

  const std::vector<Eigen::Vector3f> inliers =
      remove_statistical_outliers(voxel, config.outlier_mean_k, config.outlier_stddev_mul);
  result.inlier_points = inliers.size();

  GroundRemoval ground = remove_ground_plane(inliers, config);
  result.ground_removed = ground.removed;
  result.non_ground = std::move(ground.non_ground);
  result.non_ground_points = result.non_ground.size();

  const std::vector<std::vector<Eigen::Vector3f>> clusters =
      euclidean_clusters(result.non_ground, config);
  result.clusters.reserve(clusters.size());
  for (const std::vector<Eigen::Vector3f>& cluster : clusters) {
    result.clusters.push_back(classify_cluster(cluster, config));
  }
  return result;
}

}  // namespace lsc
