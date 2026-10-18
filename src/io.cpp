#include "lidar_shape_classifier/pipeline.hpp"

#include "cloud_convert.hpp"

#include <pcl/io/pcd_io.h>
#include <pcl/io/ply_io.h>

#include <algorithm>
#include <cctype>
#include <stdexcept>
#include <string>

namespace lsc {
namespace {

std::string extension_of(const std::string& path) {
  const std::size_t dot = path.find_last_of('.');
  if (dot == std::string::npos || dot + 1 >= path.size()) {
    return {};
  }
  std::string ext = path.substr(dot + 1);
  std::transform(ext.begin(), ext.end(), ext.begin(),
                 [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
  return ext;
}

}  // namespace

std::vector<Eigen::Vector3f> load_cloud(const std::string& path) {
  const std::string ext = extension_of(path);
  pcl::PointCloud<pcl::PointXYZ> cloud;
  int status = -1;
  if (ext == "pcd") {
    status = pcl::io::loadPCDFile(path, cloud);
  } else if (ext == "ply") {
    status = pcl::io::loadPLYFile(path, cloud);
  } else {
    throw std::runtime_error("unsupported point cloud extension (use .pcd or .ply): " + path);
  }
  if (status < 0) {
    throw std::runtime_error("failed to load point cloud: " + path);
  }
  return finite_points(from_pcl(cloud));
}

void save_cloud(const std::string& path, const std::vector<Eigen::Vector3f>& points) {
  const std::string ext = extension_of(path);
  const pcl::PointCloud<pcl::PointXYZ>::Ptr cloud = to_pcl(finite_points(points));
  int status = -1;
  if (ext == "pcd") {
    status = pcl::io::savePCDFileASCII(path, *cloud);
  } else if (ext == "ply") {
    status = pcl::io::savePLYFileASCII(path, *cloud);
  } else {
    throw std::runtime_error("unsupported point cloud extension (use .pcd or .ply): " + path);
  }
  if (status < 0) {
    throw std::runtime_error("failed to write point cloud: " + path);
  }
}

}  // namespace lsc
