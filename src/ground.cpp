#include "lidar_shape_classifier/pipeline.hpp"

#include "cloud_convert.hpp"

#include <pcl/segmentation/sac_segmentation.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <vector>

namespace lsc {
namespace {

constexpr float kPi = 3.14159265358979323846f;

}  // namespace

GroundRemoval remove_ground_plane(const std::vector<Eigen::Vector3f>& points,
                                  const PipelineConfig& config) {
  GroundRemoval result;
  const std::vector<Eigen::Vector3f> finite = finite_points(points);
  result.non_ground = finite;
  if (finite.size() < 20 || config.ground_distance_threshold <= 0.f) {
    return result;
  }

  const pcl::PointCloud<pcl::PointXYZ>::Ptr cloud = to_pcl(finite);
  pcl::SACSegmentation<pcl::PointXYZ> seg(false);
  seg.setOptimizeCoefficients(true);
  seg.setModelType(pcl::SACMODEL_PERPENDICULAR_PLANE);
  seg.setMethodType(pcl::SAC_RANSAC);
  seg.setAxis(Eigen::Vector3f::UnitZ());
  const float eps = std::max(config.ground_eps_angle_deg, 1.f) * kPi / 180.f;
  seg.setEpsAngle(eps);
  seg.setDistanceThreshold(config.ground_distance_threshold);
  seg.setMaxIterations(std::max(config.ground_max_iterations, 50));
  seg.setProbability(0.99);
  seg.setInputCloud(cloud);

  pcl::PointIndices inliers;
  pcl::ModelCoefficients coefficients;
  seg.segment(inliers, coefficients);

  const std::size_t support = inliers.indices.size();
  const std::size_t minimum = std::max<std::size_t>(30, finite.size() / 10);
  if (support < minimum || coefficients.values.size() < 4) {
    return result;
  }

  const float nx = coefficients.values[0];
  const float ny = coefficients.values[1];
  const float nz = coefficients.values[2];
  const float norm = std::sqrt(nx * nx + ny * ny + nz * nz);
  if (norm < 1e-6f || std::abs(nz) / norm < std::cos(eps)) {
    return result;
  }

  std::vector<char> drop(finite.size(), 0);
  for (const int index : inliers.indices) {
    if (index >= 0 && static_cast<std::size_t>(index) < drop.size()) {
      drop[static_cast<std::size_t>(index)] = 1;
    }
  }
  result.non_ground.clear();
  result.non_ground.reserve(finite.size() - support);
  for (std::size_t i = 0; i < finite.size(); ++i) {
    if (drop[i] == 0) {
      result.non_ground.push_back(finite[i]);
    }
  }
  result.removed = true;
  return result;
}

}  // namespace lsc
