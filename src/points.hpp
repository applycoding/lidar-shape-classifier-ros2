#pragma once

#include <Eigen/Core>

#include <cmath>
#include <vector>

namespace lsc {

inline std::vector<Eigen::Vector3f> finite_points(const std::vector<Eigen::Vector3f>& points) {
  std::vector<Eigen::Vector3f> out;
  out.reserve(points.size());
  for (const Eigen::Vector3f& point : points) {
    if (std::isfinite(point.x()) && std::isfinite(point.y()) && std::isfinite(point.z())) {
      out.push_back(point);
    }
  }
  return out;
}

}  // namespace lsc
