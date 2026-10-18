#pragma once

#include "points.hpp"

#include <pcl/point_cloud.h>
#include <pcl/point_types.h>

#include <cstdint>
#include <memory>
#include <vector>

namespace lsc {

inline pcl::PointCloud<pcl::PointXYZ>::Ptr to_pcl(const std::vector<Eigen::Vector3f>& points) {
  auto cloud = std::make_shared<pcl::PointCloud<pcl::PointXYZ>>();
  cloud->reserve(points.size());
  for (const Eigen::Vector3f& point : points) {
    cloud->push_back(pcl::PointXYZ(point.x(), point.y(), point.z()));
  }
  cloud->width = static_cast<std::uint32_t>(cloud->size());
  cloud->height = 1;
  cloud->is_dense = true;
  return cloud;
}

inline std::vector<Eigen::Vector3f> from_pcl(const pcl::PointCloud<pcl::PointXYZ>& cloud) {
  std::vector<Eigen::Vector3f> points;
  points.reserve(cloud.size());
  for (const pcl::PointXYZ& point : cloud.points) {
    points.emplace_back(point.x, point.y, point.z);
  }
  return points;
}

}  // namespace lsc
