#pragma once

#include <Eigen/Core>

#include <cstddef>
#include <stdexcept>
#include <string_view>
#include <vector>

namespace lsc {

enum class ShapeClass {
  Box,
  Cylinder,
  Sphere,
  Plane,
  Unknown,
};

inline std::string_view to_string(ShapeClass shape) {
  switch (shape) {
    case ShapeClass::Box:
      return "box";
    case ShapeClass::Cylinder:
      return "cylinder";
    case ShapeClass::Sphere:
      return "sphere";
    case ShapeClass::Plane:
      return "plane";
    case ShapeClass::Unknown:
      return "unknown";
  }
  throw std::logic_error("unhandled ShapeClass");
}

struct PipelineConfig {
  float voxel_leaf_size = 0.03f;
  int outlier_mean_k = 20;
  float outlier_stddev_mul = 3.0f;
  float ground_distance_threshold = 0.08f;
  int ground_max_iterations = 200;
  float ground_eps_angle_deg = 15.0f;
  float cluster_tolerance = 0.10f;
  int cluster_min_size = 40;
  int cluster_max_size = 100000;
  float fit_distance_threshold = 0.04f;
  int fit_max_iterations = 200;
  int min_classify_points = 30;
};

struct ClassifiedCluster {
  ShapeClass shape = ShapeClass::Unknown;
  Eigen::Vector3f centroid = Eigen::Vector3f::Zero();
  Eigen::Vector3f obb_center = Eigen::Vector3f::Zero();
  Eigen::Vector3f axis_x = Eigen::Vector3f::UnitX();
  Eigen::Vector3f axis_y = Eigen::Vector3f::UnitY();
  Eigen::Vector3f axis_z = Eigen::Vector3f::UnitZ();
  Eigen::Vector3f obb_extents = Eigen::Vector3f::Zero();
  Eigen::Vector3f axis = Eigen::Vector3f::UnitZ();
  float confidence = 0.f;
  float radius = 0.f;
  float height = 0.f;
  std::size_t point_count = 0;
};

struct PipelineResult {
  std::size_t input_points = 0;
  std::size_t voxel_points = 0;
  std::size_t inlier_points = 0;
  std::size_t non_ground_points = 0;
  bool ground_removed = false;
  std::vector<ClassifiedCluster> clusters;
  std::vector<Eigen::Vector3f> non_ground;
};

}  // namespace lsc
